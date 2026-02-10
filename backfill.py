"""
Historical backfill script — populates the database with predictions and trades
from Jan 1, 2026 to today by running the ML model against historical Nifty data.

Uses shared/feature_compute.py — the single source of truth for feature computation.
Features are loaded from merged_daily.parquet (same as backtest and live system).

Usage: Called via POST /api/backfill endpoint on Render.
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

from core.timezone import today_ist
from db.models import Prediction, Trade, DailyPnl, DailyFeature
from core.model_runner import ModelRunner
from core.signal_mapper import map_signal, get_classification_breakdown
from core.option_pricer import (
    select_strikes, price_bull_put_spread, price_iron_condor,
    get_next_weekly_expiry, compute_time_to_expiry_years,
    compute_spread_value,
)
from config import (
    ACTIVE_VERSIONS, VERSION_CONFIGS, INITIAL_CAPITAL,
    DOWNSIDE_MODEL_PATH, SCALER_PATH, FEATURE_NAMES_PATH,
    NIFTY_LOT_SIZE, RISK_FREE_RATE,
    MARGIN_PER_LOT_BULL, MARGIN_PER_LOT_IC,
    BULL_OTM_SELL, BULL_OTM_BUY,
    VIX_MIN_ENTRY, VIX_MAX_ENTRY,
    DTE_MIN_ENTRY, DTE_MAX_ENTRY,
)

logger = logging.getLogger(__name__)

BACKFILL_START = date(2026, 1, 1)

# Paths to the ground truth data (shipped in backend/data/)
MERGED_DAILY_PATH = os.path.join(BACKEND_ROOT, "data", "merged_daily.parquet")
DHAN_RAW_PATH = os.path.join(BACKEND_ROOT, "data", "dhan_raw_options.parquet")
DHAN_SKEW_PATH = os.path.join(BACKEND_ROOT, "data", "daily_iv_skew_params.parquet")

# The 37 training feature names
TRAINING_FEATURES = BASE_FEATURE_COLS + [
    "deep_otm_oi_ratio", "deep_otm_oi_ratio_change_5d",
    "put_oi_buildup_ratio", "put_volume_surge_ratio",
    "iv_skew_steepness", "iv_skew_change_5d", "atm_iv",
    "atm_iv_percentile_252d", "atm_put_intraday_range",
]


# ── main backfill ────────────────────────────────────────────────────

async def run_backfill(db_session) -> dict:
    """
    Run full historical backfill from Jan 2026 to today.
    Clears existing data first, then re-populates.
    Returns summary stats.
    """
    from sqlalchemy import text

    logger.info("=== Starting historical backfill ===")

    # Clear existing backfill data so re-runs don't hit unique constraint errors
    logger.info("Clearing existing backfill data...")
    await db_session.execute(text("DELETE FROM daily_pnl"))
    await db_session.execute(text("DELETE FROM trades"))
    await db_session.execute(text("DELETE FROM predictions"))
    await db_session.execute(text("DELETE FROM daily_features"))
    await db_session.commit()
    logger.info("Existing data cleared")

    # Load ground truth data from merged_daily.parquet (same source as backtest)
    if not os.path.exists(MERGED_DAILY_PATH):
        return {
            "error": f"merged_daily.parquet not found at {MERGED_DAILY_PATH}",
            "hint": "Run data_collection.py to create it, or deploy with the data.",
        }

    merged_daily = pd.read_parquet(MERGED_DAILY_PATH)
    logger.info(f"Loaded merged_daily: {len(merged_daily)} rows, "
                f"last date: {merged_daily.index[-1].date()}")

    # Load Dhan features if available
    dhan_features = None
    if os.path.exists(DHAN_RAW_PATH) and os.path.exists(DHAN_SKEW_PATH):
        dhan_features = load_dhan_options_features(DHAN_RAW_PATH, DHAN_SKEW_PATH)
        if dhan_features is not None:
            logger.info(f"Loaded Dhan features: {len(dhan_features)} days")

    # Pre-compute features for all backfill dates using the shared module
    logger.info("Computing features for backfill dates using shared module...")
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

    # Filter to trading days that have features
    backfill_dates_with_features = sorted(feature_cache.keys())
    logger.info(f"Backfilling {len(backfill_dates_with_features)} trading days from {BACKFILL_START}")

    # Track per-version state
    version_state = {}
    for v in ACTIVE_VERSIONS:
        version_state[v] = {
            "capital": INITIAL_CAPITAL,
            "cumulative_pnl": 0.0,
            "open_trades": [],
        }

    stats = {
        "days_processed": 0,
        "predictions_created": 0,
        "trades_opened": 0,
        "trades_closed": 0,
    }

    for ts in backfill_dates_with_features:
        trade_date = ts.date() if hasattr(ts, 'date') else ts

        features = feature_cache[ts]

        spot = float(merged_daily.loc[ts, 'nifty_close'])
        vix = features["india_vix"]

        if pd.isna(spot) or spot <= 0:
            continue

        # Run model prediction
        prediction_value = model_runner.predict(features)

        # Store daily features (extra display-friendly names for the frontend)
        daily_feature = DailyFeature(
            date=trade_date,
            features=features,
            vix=vix,
            vix_20d_avg=features.get("india_vix", vix),
            nifty_20d_return=features.get("nifty_return_20d"),
            nifty_50d_return=features.get("nifty_distance_50dma"),
            iv_skew=features.get("iv_skew_steepness"),
            fii_net=0,
            dii_net=0,
            put_call_ratio=features.get("pcr_proxy"),
            rsi_14=features.get("rsi_14"),
            adx_14=0,  # not in training features
        )
        db_session.add(daily_feature)

        # Process each version
        for version in ACTIVE_VERSIONS:
            cfg = VERSION_CONFIGS[version]
            state = version_state[version]
            signal = map_signal(prediction_value, cfg)

            # Confidence
            breakdown = get_classification_breakdown(prediction_value)
            active_zone = next(
                (z for z in breakdown["zones"] if z.get("confidence") == "active"), None
            )
            confidence = active_zone["distance_to_boundary"] if active_zone else 0

            # Store prediction
            pred = Prediction(
                date=trade_date,
                timestamp=datetime.combine(trade_date, datetime.min.time().replace(hour=9, minute=20)),
                predicted_drawdown=prediction_value,
                signal_type=signal["signal"],
                version=version,
                nifty_spot=spot,
                vix=vix,
                confidence_score=confidence,
                graduated_mult=signal["size_mult"],
                features=features,
            )
            db_session.add(pred)
            stats["predictions_created"] += 1

            # Check exits on open trades first
            trades_to_close = []
            for ot in state["open_trades"]:
                dte = (ot["expiry"] - trade_date).days
                T = max(dte / 365.0, 1 / 365.0)
                sigma = vix / 100.0 if vix > 0 else 0.15

                current_value = compute_spread_value(
                    ot["trade_type"], spot,
                    ot["sell_strike"], ot["buy_strike"],
                    ot.get("ic_call_sell"), ot.get("ic_call_buy"),
                    T, RISK_FREE_RATE, sigma,
                )

                pnl_per_unit = ot["credit"] - current_value
                pnl_pct = pnl_per_unit / ot["credit"] if ot["credit"] > 0 else 0

                exit_reason = None

                # 1. Expiry exit
                if dte <= cfg.get("MIN_DTE_EXIT", 1):
                    exit_reason = "expiry"

                # 2. Profit target — day-count stages matching backtest
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

                # 3. Trailing stop — track peak PnL per trade
                if not exit_reason:
                    trailing_activate = cfg.get("TRAILING_STOP_ACTIVATE", 0.40)
                    trailing_level = cfg.get("TRAILING_STOP_LEVEL", 0.10)

                    if pnl_pct >= trailing_activate:
                        peak = ot.get("peak_pnl_pct", 0.0)
                        if peak > 0 and pnl_pct < peak - trailing_level:
                            exit_reason = "trailing_stop"
                        ot["peak_pnl_pct"] = max(peak, pnl_pct)

                # 4. Stop loss — with N-day confirmation matching backtest
                if not exit_reason:
                    sl_mult = cfg.get("STOP_LOSS_MULTIPLIER", 3.0)
                    if ot["trade_type"] == "iron_condor":
                        sl_mult = cfg.get("IC_STOP_LOSS_MULTIPLIER", 3.0)

                    confirm_days = cfg.get("STOP_LOSS_CONFIRM_DAYS", 1)
                    if ot["trade_type"] == "iron_condor":
                        confirm_days = cfg.get("IC_STOP_LOSS_CONFIRM_DAYS", 1)

                    max_loss = ot["credit"] * sl_mult
                    if current_value >= max_loss + ot["credit"]:
                        ot["stop_loss_breach_days"] = ot.get("stop_loss_breach_days", 0) + 1
                        if ot["stop_loss_breach_days"] >= confirm_days:
                            exit_reason = "stop_loss"
                    else:
                        ot["stop_loss_breach_days"] = 0

                if exit_reason:
                    realized_pnl = pnl_per_unit * ot["num_lots"] * NIFTY_LOT_SIZE
                    rpnl_pct = realized_pnl / ot["capital_deployed"] * 100 if ot["capital_deployed"] > 0 else 0

                    from sqlalchemy import update
                    await db_session.execute(
                        update(Trade).where(Trade.trade_id == ot["trade_id"]).values(
                            status="closed",
                            exit_date=trade_date,
                            exit_time=datetime.combine(trade_date, datetime.min.time().replace(hour=15, minute=30)),
                            exit_spot=spot,
                            exit_reason=exit_reason,
                            realized_pnl=realized_pnl,
                            current_pnl=realized_pnl,
                            current_pnl_pct=rpnl_pct,
                        )
                    )
                    state["cumulative_pnl"] += realized_pnl
                    state["capital"] = INITIAL_CAPITAL + state["cumulative_pnl"]
                    trades_to_close.append(ot["trade_id"])
                    stats["trades_closed"] += 1

            # Remove closed trades
            state["open_trades"] = [
                t for t in state["open_trades"] if t["trade_id"] not in trades_to_close
            ]

            # Open new trade if signal != no_trade
            if signal["signal"] != "no_trade" and signal["trade_type"]:
                trade_type = signal["trade_type"]
                size_mult = signal["size_mult"]

                # Check limits
                open_same_type = sum(
                    1 for t in state["open_trades"] if t["trade_type"] == trade_type
                )
                max_conc = cfg.get("MAX_CONCURRENT_POSITIONS", 3)
                if trade_type == "iron_condor":
                    max_conc = cfg.get("IC_MAX_CONCURRENT", 2)

                # Check entry gap
                last_entry = max(
                    (t["entry_date"] for t in state["open_trades"]), default=None
                )
                min_gap = cfg.get("MIN_ENTRY_GAP_DAYS", 2)
                gap_ok = last_entry is None or (trade_date - last_entry).days >= min_gap

                # VIX filter
                vix_ok = VIX_MIN_ENTRY <= vix <= VIX_MAX_ENTRY

                if open_same_type < max_conc and gap_ok and vix_ok:
                    strikes = select_strikes(spot, trade_type, {
                        "BULL_OTM_SELL": BULL_OTM_SELL,
                        "BULL_OTM_BUY": BULL_OTM_BUY,
                        "IC_PUT_OTM_SELL": cfg.get("IC_PUT_OTM_SELL", 0.03),
                        "IC_PUT_OTM_BUY": cfg.get("IC_PUT_OTM_BUY", 0.055),
                        "IC_CALL_OTM_SELL": cfg.get("IC_CALL_OTM_SELL", 0.04),
                        "IC_CALL_OTM_BUY": cfg.get("IC_CALL_OTM_BUY", 0.065),
                    })

                    expiry = get_next_weekly_expiry(trade_date)
                    T = compute_time_to_expiry_years(trade_date, expiry)
                    sigma = vix / 100.0 if vix > 0 else 0.15

                    if trade_type == "bull_put":
                        credit = price_bull_put_spread(
                            spot, strikes["sell_strike"], strikes["buy_strike"],
                            T, RISK_FREE_RATE, sigma, apply_slippage=True,
                        )
                    else:
                        credit = price_iron_condor(
                            spot, strikes["sell_strike"], strikes["buy_strike"],
                            strikes["ic_call_sell"], strikes["ic_call_buy"],
                            T, RISK_FREE_RATE, sigma, apply_slippage=True,
                        )

                    pos_pct = cfg.get("POSITION_SIZE_PCT", 0.20)
                    if trade_type == "iron_condor":
                        pos_pct = cfg.get("IC_POSITION_SIZE_PCT", 0.15)

                    effective_pct = pos_pct * size_mult
                    margin = MARGIN_PER_LOT_BULL if trade_type == "bull_put" else MARGIN_PER_LOT_IC
                    max_cap = state["capital"] * effective_pct
                    num_lots = max(1, int(max_cap / margin))

                    total_credit = credit * num_lots * NIFTY_LOT_SIZE
                    capital_deployed = num_lots * margin

                    trade_id = f"{version.replace('.', '')}-{trade_date.isoformat()}-{signal['signal']}"

                    trade = Trade(
                        trade_id=trade_id,
                        version=version,
                        date=trade_date,
                        signal_type=signal["signal"],
                        trade_type=trade_type,
                        entry_mode="normal",
                        entry_date=trade_date,
                        entry_time=datetime.combine(trade_date, datetime.min.time().replace(hour=9, minute=20)),
                        entry_spot=spot,
                        expiry=expiry,
                        sell_strike=strikes["sell_strike"],
                        buy_strike=strikes["buy_strike"],
                        ic_call_sell=strikes.get("ic_call_sell"),
                        ic_call_buy=strikes.get("ic_call_buy"),
                        num_lots=num_lots,
                        credit_received=credit,
                        total_credit=total_credit,
                        status="open",
                        current_pnl=0,
                        current_pnl_pct=0,
                        unrealized_pnl=0,
                        position_size_pct=effective_pct,
                        graduated_mult=size_mult,
                        capital_deployed=capital_deployed,
                        stop_loss_breach_days=0,
                        peak_pnl_pct=0.0,
                    )
                    db_session.add(trade)

                    state["open_trades"].append({
                        "trade_id": trade_id,
                        "trade_type": trade_type,
                        "entry_date": trade_date,
                        "expiry": expiry,
                        "sell_strike": strikes["sell_strike"],
                        "buy_strike": strikes["buy_strike"],
                        "ic_call_sell": strikes.get("ic_call_sell"),
                        "ic_call_buy": strikes.get("ic_call_buy"),
                        "credit": credit,
                        "num_lots": num_lots,
                        "capital_deployed": capital_deployed,
                        "peak_pnl_pct": 0.0,
                        "stop_loss_breach_days": 0,
                    })
                    stats["trades_opened"] += 1

            # Record daily PnL
            daily_pnl_record = DailyPnl(
                date=trade_date,
                version=version,
                starting_capital=INITIAL_CAPITAL,
                ending_capital=state["capital"],
                daily_pnl=0,  # simplified
                cumulative_pnl=state["cumulative_pnl"],
                cumulative_return_pct=round(state["cumulative_pnl"] / INITIAL_CAPITAL * 100, 2),
                open_positions=len(state["open_trades"]),
                trades_opened=0,
                trades_closed=0,
            )
            db_session.add(daily_pnl_record)

        stats["days_processed"] += 1

        # Commit in batches of 10 days
        if stats["days_processed"] % 10 == 0:
            await db_session.commit()
            logger.info(f"Backfill progress: {stats['days_processed']} days, "
                        f"{stats['predictions_created']} predictions, "
                        f"{stats['trades_opened']} trades opened")

    # Final commit
    await db_session.commit()
    logger.info(f"=== Backfill complete: {stats} ===")
    return stats
