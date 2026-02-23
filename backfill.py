"""
Historical backfill script — populates the database with predictions and trades
from Jan 1, 2026 to today by running the ML model against historical Nifty data.

Uses shared/feature_compute.py — the single source of truth for feature computation.
Features are loaded from merged_daily.parquet (same as backtest and live system).

IMPORTANT: For IDENTICAL results to the backtest, use run_backfill_local.py instead.
That script calls run_backtest() from nifty_options_model/src/backtest.py directly.
This server-side script uses a simplified trade engine (no rolling, no scale-in,
no special modes like VIX harvest / event crush / OI wall). It's suitable for
quick re-population but may diverge from the backtest on some trades.

Usage: Called via POST /api/backfill endpoint on Render.
       For production, prefer: cd backend && python3 run_backfill_local.py
"""

import logging
import os
import math
import numpy as np
import pandas as pd
from datetime import date, datetime, timedelta

import sys
BACKEND_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BACKEND_ROOT)

from shared.feature_compute import (
    compute_features_for_date,
    load_dhan_options_features,
    BASE_FEATURE_COLS,
)
from shared.config_profiles import get_config_profile
from shared.model_signals import (
    model_signal, model_signal_graduated, model_signal_graduated_gentle,
)
from shared.option_pricing_engine import (
    OptionPricer, price_bull_put_spread, price_bear_call_spread,
    compute_position_value,
)
from shared.utils_backtest import find_next_weekly_expiry, round_to_nearest

from core.timezone import today_ist
from db.models import Prediction, Trade, DailyPnl, DailyFeature
from core.model_runner import ModelRunner

logger = logging.getLogger(__name__)

BACKFILL_START = date(2026, 1, 1)

# Paths to the ground truth data (shipped in backend/data/)
MERGED_DAILY_PATH = os.path.join(BACKEND_ROOT, "data", "merged_daily.parquet")
DHAN_RAW_PATH = os.path.join(BACKEND_ROOT, "data", "dhan_raw_options.parquet")
DHAN_SKEW_PATH = os.path.join(BACKEND_ROOT, "data", "daily_iv_skew_params.parquet")

# The 35 training feature names (month/is_volatile_month removed)
TRAINING_FEATURES = BASE_FEATURE_COLS + [
    "deep_otm_oi_ratio", "deep_otm_oi_ratio_change_5d",
    "put_oi_buildup_ratio", "put_volume_surge_ratio",
    "iv_skew_steepness", "iv_skew_change_5d", "atm_iv",
    "atm_iv_percentile_252d", "atm_put_intraday_range",
]

# Version mapping: display version → config profile
VERSION_MAP = {
    "v5.4.2": "v5.4",
    "v5.4.3": "v5.4.3",
    "v5.4.4": "v5.4.4",
}


def _get_signal(model_runner, features, feature_names, cfg):
    """Get signal using the shared signal mapping functions."""
    prediction_value = model_runner.predict(features)
    mapping = cfg.get('SIGNAL_MAPPING', 'sharp')

    if mapping == 'graduated':
        # Graduated needs model object + features_row as pd.Series
        feat_series = pd.Series(
            [features.get(c, 0.0) for c in feature_names],
            index=feature_names,
        )
        signal_type, pred_dd, size_mult = model_signal_graduated(
            model_runner.model, feat_series, feature_names, cfg
        )
    elif mapping == 'graduated_gentle':
        feat_series = pd.Series(
            [features.get(c, 0.0) for c in feature_names],
            index=feature_names,
        )
        signal_type, pred_dd, size_mult = model_signal_graduated_gentle(
            model_runner.model, feat_series, feature_names, cfg
        )
    else:
        feat_series = pd.Series(
            [features.get(c, 0.0) for c in feature_names],
            index=feature_names,
        )
        signal_type, pred_dd = model_signal(
            model_runner.model, feat_series, feature_names, cfg
        )
        size_mult = 1.0
        if signal_type == 'bull_half':
            size_mult = 0.5

    # Determine trade type
    trade_type = None
    if signal_type in ('bull_full', 'bull_half'):
        trade_type = 'bull_put'
    elif signal_type == 'iron_condor':
        trade_type = 'iron_condor'

    return {
        'signal': signal_type,
        'predicted_drawdown': pred_dd,
        'size_mult': size_mult,
        'trade_type': trade_type,
    }


# ── main backfill ────────────────────────────────────────────────────

async def run_backfill(db_session) -> dict:
    """
    Run full historical backfill from Jan 2026 to today.
    Clears existing data first, then re-populates.

    NOTE: For IDENTICAL results to the backtest, use run_backfill_local.py instead.
    """
    from sqlalchemy import text
    from config import DOWNSIDE_MODEL_PATH, SCALER_PATH, FEATURE_NAMES_PATH

    logger.info("=== Starting historical backfill (server-side, simplified engine) ===")
    logger.warning("NOTE: For exact backtest-identical results, use run_backfill_local.py")

    # Clear only backtest-period data, preserve live/forward-test trades
    FORWARD_TEST_START = "2026-02-13"
    logger.info(f"Clearing backtest data (before {FORWARD_TEST_START}), preserving live trades...")
    await db_session.execute(text(f"DELETE FROM daily_pnl WHERE date < '{FORWARD_TEST_START}'"))
    await db_session.execute(text(f"DELETE FROM trades WHERE entry_date < '{FORWARD_TEST_START}'"))
    await db_session.execute(text(f"DELETE FROM predictions WHERE date < '{FORWARD_TEST_START}'"))
    await db_session.execute(text(f"DELETE FROM daily_features WHERE date < '{FORWARD_TEST_START}'"))
    await db_session.commit()
    logger.info("Backtest data cleared (live trades preserved)")

    # Load ground truth data
    if not os.path.exists(MERGED_DAILY_PATH):
        return {"error": f"merged_daily.parquet not found at {MERGED_DAILY_PATH}"}

    merged_daily = pd.read_parquet(MERGED_DAILY_PATH)
    logger.info(f"Loaded merged_daily: {len(merged_daily)} rows, "
                f"last date: {merged_daily.index[-1].date()}")

    # Load Dhan features
    dhan_features = None
    if os.path.exists(DHAN_RAW_PATH) and os.path.exists(DHAN_SKEW_PATH):
        dhan_features = load_dhan_options_features(DHAN_RAW_PATH, DHAN_SKEW_PATH)
        if dhan_features is not None:
            logger.info(f"Loaded Dhan features: {len(dhan_features)} days")

    # Pre-compute features
    logger.info("Computing features for backfill dates...")
    backfill_dates = [d for d in merged_daily.index if d.date() >= BACKFILL_START]
    feature_cache = {}
    for d in backfill_dates:
        features = compute_features_for_date(
            merged_daily, d, dhan_features,
            data_start="2014-01-01", data_end="2026-12-31",
        )
        if features is not None:
            feature_cache[d] = features
    logger.info(f"Computed features for {len(feature_cache)} trading days")

    # Load model
    model_runner = ModelRunner(DOWNSIDE_MODEL_PATH, SCALER_PATH, FEATURE_NAMES_PATH)
    if model_runner.model is None:
        return {"error": "Model not loaded"}

    backfill_dates_sorted = sorted(feature_cache.keys())
    logger.info(f"Backfilling {len(backfill_dates_sorted)} trading days from {BACKFILL_START}")

    # Per-version state
    version_state = {}
    for display_ver in VERSION_MAP:
        version_state[display_ver] = {
            "capital": 2_500_000,
            "cumulative_pnl": 0.0,
            "open_trades": [],
        }

    stats = {"days_processed": 0, "predictions_created": 0,
             "trades_opened": 0, "trades_closed": 0}

    for ts in backfill_dates_sorted:
        trade_date = ts.date() if hasattr(ts, 'date') else ts
        features = feature_cache[ts]
        spot = float(merged_daily.loc[ts, 'nifty_close'])
        vix = features["india_vix"]

        if pd.isna(spot) or spot <= 0:
            continue

        # Store daily features once
        daily_feature = DailyFeature(
            date=trade_date, features=features,
            vix=vix, vix_20d_avg=vix,
            nifty_20d_return=features.get("nifty_return_20d"),
            nifty_50d_return=features.get("nifty_distance_50dma"),
            iv_skew=features.get("iv_skew_steepness"),
            fii_net=0, dii_net=0,
            put_call_ratio=features.get("pcr_proxy"),
            rsi_14=features.get("rsi_14"), adx_14=0,
        )
        db_session.add(daily_feature)

        # Process each version
        for display_ver, profile_name in VERSION_MAP.items():
            cfg = get_config_profile(profile_name)
            state = version_state[display_ver]

            # Get signal
            signal = _get_signal(model_runner, features, TRAINING_FEATURES, cfg)
            pred_dd = signal['predicted_drawdown']

            # Confidence
            thresholds = [-0.020, -0.047, -0.076]
            confidence = min(abs(pred_dd - t) for t in thresholds) if thresholds else 0.5

            # Store prediction
            pred = Prediction(
                date=trade_date,
                timestamp=datetime.combine(trade_date, datetime.min.time().replace(hour=9, minute=20)),
                predicted_drawdown=pred_dd,
                signal_type=signal["signal"],
                version=display_ver,
                nifty_spot=spot, vix=vix,
                confidence_score=confidence,
                graduated_mult=signal["size_mult"],
                features=features,
            )
            db_session.add(pred)
            stats["predictions_created"] += 1

            # Check exits
            trades_to_close = []
            for ot in state["open_trades"]:
                dte = (ot["expiry"] - trade_date).days
                T = max(dte / 365.0, 1 / 365.0)
                sigma = vix / 100.0 if vix > 0 else 0.15

                # Use simple B-S pricing for server-side
                from shared.option_pricing_engine import OptionPricer as _OP
                from scipy.stats import norm as _norm

                # Compute spread value inline
                if ot["trade_type"] == "bull_put":
                    def _bs_put(S, K):
                        base_iv = max(sigma, 0.05)
                        m = (S - K) / S
                        adj = base_iv * max(1.0 + m * 3.0, 1.0)
                        d1 = (np.log(S / K) + (0.07 + 0.5 * adj ** 2) * T) / (adj * np.sqrt(T))
                        d2 = d1 - adj * np.sqrt(T)
                        return max(K * np.exp(-0.07 * T) * _norm.cdf(-d2) - S * _norm.cdf(-d1), 0) * 1.25
                    current_value = _bs_put(spot, ot["sell_strike"]) - _bs_put(spot, ot["buy_strike"])
                elif ot["trade_type"] == "iron_condor" and ot.get("ic_call_sell"):
                    def _bs_put(S, K):
                        base_iv = max(sigma, 0.05)
                        m = (S - K) / S
                        adj = base_iv * max(1.0 + m * 3.0, 1.0)
                        d1 = (np.log(S / K) + (0.07 + 0.5 * adj ** 2) * T) / (adj * np.sqrt(T))
                        d2 = d1 - adj * np.sqrt(T)
                        return max(K * np.exp(-0.07 * T) * _norm.cdf(-d2) - S * _norm.cdf(-d1), 0) * 1.25
                    def _bs_call(S, K):
                        base_iv = max(sigma, 0.05)
                        m = (K - S) / S
                        adj = base_iv * (1.0 + m * 1.5)
                        adj = max(adj, base_iv * 0.8)
                        d1 = (np.log(S / K) + (0.07 + 0.5 * adj ** 2) * T) / (adj * np.sqrt(T))
                        d2 = d1 - adj * np.sqrt(T)
                        return max(S * _norm.cdf(d1) - K * np.exp(-0.07 * T) * _norm.cdf(d2), 0) * 1.25
                    put_val = _bs_put(spot, ot["sell_strike"]) - _bs_put(spot, ot["buy_strike"])
                    call_val = _bs_call(spot, ot["ic_call_sell"]) - _bs_call(spot, ot["ic_call_buy"])
                    current_value = put_val + call_val
                else:
                    current_value = 0

                pnl_per_unit = ot["credit"] - current_value
                pnl_pct = pnl_per_unit / ot["credit"] if ot["credit"] > 0 else 0

                exit_reason = None

                if dte <= cfg.get("MIN_DTE_EXIT", 1):
                    exit_reason = "expiry"

                if not exit_reason:
                    days_held = (trade_date - ot["entry_date"]).days
                    if days_held <= 3:
                        pt = cfg.get("PROFIT_TARGET_EARLY", 0.50)
                    elif days_held <= 7:
                        pt = cfg.get("PROFIT_TARGET_MID", 0.65)
                    else:
                        pt = cfg.get("PROFIT_TARGET_LATE", 0.80)
                    if pnl_pct >= pt:
                        exit_reason = "profit_target"

                if not exit_reason:
                    ta = cfg.get("TRAILING_STOP_ACTIVATE", 0.40)
                    tl = cfg.get("TRAILING_STOP_LEVEL", 0.10)
                    if pnl_pct >= ta:
                        peak = ot.get("peak_pnl_pct", 0.0)
                        if peak > 0 and pnl_pct < peak - tl:
                            exit_reason = "trailing_stop"
                        ot["peak_pnl_pct"] = max(peak, pnl_pct)

                if not exit_reason:
                    sl_mult = cfg.get("IC_STOP_LOSS_MULTIPLIER", 3.0) if ot["trade_type"] == "iron_condor" else cfg.get("STOP_LOSS_MULTIPLIER", 3.0)
                    cd = cfg.get("IC_STOP_LOSS_CONFIRM_DAYS", 1) if ot["trade_type"] == "iron_condor" else cfg.get("STOP_LOSS_CONFIRM_DAYS", 1)
                    if current_value >= ot["credit"] * sl_mult + ot["credit"]:
                        ot["stop_loss_breach_days"] = ot.get("stop_loss_breach_days", 0) + 1
                        if ot["stop_loss_breach_days"] >= cd:
                            exit_reason = "stop_loss"
                    else:
                        ot["stop_loss_breach_days"] = 0

                if exit_reason:
                    realized_pnl = pnl_per_unit * ot["num_lots"] * 25
                    rpnl_pct = realized_pnl / ot["capital_deployed"] * 100 if ot["capital_deployed"] > 0 else 0

                    from sqlalchemy import update
                    await db_session.execute(
                        update(Trade).where(Trade.trade_id == ot["trade_id"]).values(
                            status="closed", exit_date=trade_date,
                            exit_time=datetime.combine(trade_date, datetime.min.time().replace(hour=15, minute=30)),
                            exit_spot=spot, exit_reason=exit_reason,
                            realized_pnl=realized_pnl, current_pnl=realized_pnl,
                            current_pnl_pct=rpnl_pct,
                        )
                    )
                    state["cumulative_pnl"] += realized_pnl
                    state["capital"] = 2_500_000 + state["cumulative_pnl"]
                    trades_to_close.append(ot["trade_id"])
                    stats["trades_closed"] += 1

            state["open_trades"] = [t for t in state["open_trades"] if t["trade_id"] not in trades_to_close]

            # Open new trade
            if signal["signal"] != "no_trade" and signal["trade_type"]:
                tt = signal["trade_type"]
                sm = signal["size_mult"]

                open_count = sum(1 for t in state["open_trades"] if t["trade_type"] == tt)
                max_c = cfg.get("IC_MAX_CONCURRENT", 2) if tt == "iron_condor" else cfg.get("MAX_CONCURRENT_POSITIONS", 3)
                last_entry = max((t["entry_date"] for t in state["open_trades"]), default=None)
                gap_ok = last_entry is None or (trade_date - last_entry).days >= cfg.get("MIN_ENTRY_GAP_DAYS", 2)

                if open_count < max_c and gap_ok:
                    sell_k = round_to_nearest(spot * (1 - cfg.get('BULL_OTM_SELL', 0.03)))
                    buy_k = round_to_nearest(spot * (1 - cfg.get('BULL_OTM_BUY', 0.055)))
                    ic_cs = round_to_nearest(spot * (1 + cfg.get('IC_CALL_OTM_SELL', 0.04))) if tt == "iron_condor" else None
                    ic_cb = round_to_nearest(spot * (1 + cfg.get('IC_CALL_OTM_BUY', 0.065))) if tt == "iron_condor" else None

                    expiry = find_next_weekly_expiry(trade_date)
                    expiry_date = expiry.date() if hasattr(expiry, 'date') else expiry
                    T = max((expiry_date - trade_date).days / 365.0, 1 / 365.0)
                    sigma_entry = vix / 100.0 if vix > 0 else 0.15

                    # Simple B-S pricing
                    from scipy.stats import norm as _norm2
                    def _bs_put2(S, K):
                        base_iv = max(sigma_entry, 0.05)
                        m = (S - K) / S
                        adj = base_iv * max(1.0 + m * 3.0, 1.0)
                        d1 = (np.log(S / K) + (0.07 + 0.5 * adj ** 2) * T) / (adj * np.sqrt(T))
                        d2 = d1 - adj * np.sqrt(T)
                        return max(K * np.exp(-0.07 * T) * _norm2.cdf(-d2) - S * _norm2.cdf(-d1), 0) * 1.25

                    if tt == "bull_put":
                        credit = max(_bs_put2(spot, sell_k) - _bs_put2(spot, buy_k), 0) * 0.98
                    else:
                        def _bs_call2(S, K):
                            base_iv = max(sigma_entry, 0.05)
                            m = (K - S) / S
                            adj = base_iv * (1.0 + m * 1.5)
                            adj = max(adj, base_iv * 0.8)
                            d1 = (np.log(S / K) + (0.07 + 0.5 * adj ** 2) * T) / (adj * np.sqrt(T))
                            d2 = d1 - adj * np.sqrt(T)
                            return max(S * _norm2.cdf(d1) - K * np.exp(-0.07 * T) * _norm2.cdf(d2), 0) * 1.25
                        put_cr = max(_bs_put2(spot, sell_k) - _bs_put2(spot, buy_k), 0)
                        call_cr = max(_bs_call2(spot, ic_cs) - _bs_call2(spot, ic_cb), 0)
                        credit = (put_cr + call_cr) * 0.98

                    pos_pct = cfg.get("IC_POSITION_SIZE_PCT", 0.15) if tt == "iron_condor" else cfg.get("POSITION_SIZE_PCT", 0.20)
                    eff_pct = pos_pct * sm
                    margin = 13_750
                    max_cap = state["capital"] * eff_pct
                    num_lots = max(1, int(max_cap / margin))
                    total_credit = credit * num_lots * 25
                    capital_deployed = num_lots * margin

                    trade_id = f"{display_ver.replace('.', '')}-{trade_date.isoformat()}-{signal['signal']}"

                    trade = Trade(
                        trade_id=trade_id, version=display_ver, date=trade_date,
                        signal_type=signal["signal"], trade_type=tt, entry_mode="normal",
                        entry_date=trade_date,
                        entry_time=datetime.combine(trade_date, datetime.min.time().replace(hour=9, minute=20)),
                        entry_spot=spot, expiry=expiry_date,
                        sell_strike=sell_k, buy_strike=buy_k,
                        ic_call_sell=ic_cs, ic_call_buy=ic_cb,
                        num_lots=num_lots, credit_received=credit,
                        total_credit=total_credit, status="open",
                        current_pnl=0, current_pnl_pct=0, unrealized_pnl=0,
                        position_size_pct=eff_pct, graduated_mult=sm,
                        capital_deployed=capital_deployed,
                        stop_loss_breach_days=0, peak_pnl_pct=0.0,
                    )
                    db_session.add(trade)

                    state["open_trades"].append({
                        "trade_id": trade_id, "trade_type": tt,
                        "entry_date": trade_date, "expiry": expiry_date,
                        "sell_strike": sell_k, "buy_strike": buy_k,
                        "ic_call_sell": ic_cs, "ic_call_buy": ic_cb,
                        "credit": credit, "num_lots": num_lots,
                        "capital_deployed": capital_deployed,
                        "peak_pnl_pct": 0.0, "stop_loss_breach_days": 0,
                    })
                    stats["trades_opened"] += 1

            # Daily PnL
            daily_pnl_record = DailyPnl(
                date=trade_date, version=display_ver,
                starting_capital=2_500_000, ending_capital=state["capital"],
                daily_pnl=0, cumulative_pnl=state["cumulative_pnl"],
                cumulative_return_pct=round(state["cumulative_pnl"] / 2_500_000 * 100, 2),
                open_positions=len(state["open_trades"]),
                trades_opened=0, trades_closed=0,
            )
            db_session.add(daily_pnl_record)

        stats["days_processed"] += 1

        if stats["days_processed"] % 10 == 0:
            await db_session.commit()
            logger.info(f"Backfill progress: {stats['days_processed']} days, "
                        f"{stats['predictions_created']} predictions, "
                        f"{stats['trades_opened']} trades opened")

    await db_session.commit()
    logger.info(f"=== Backfill complete: {stats} ===")
    return stats
