"""
Run the backfill locally against the remote Neon database.
Uses raw asyncpg (no ORM) to bypass Python 3.10+ requirement for SQLAlchemy Mapped types.

Features are computed via shared/feature_compute.py — the single source of truth.

Usage:
    cd /Users/Anmol/Desktop/Options\ Trading/backend
    python3 run_backfill_local.py
"""

import asyncio
import os
import ssl
import json
import logging
import math
import time
import numpy as np
import pandas as pd
import joblib
from datetime import date, datetime, timedelta

import sys
BACKEND_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BACKEND_ROOT)

from shared.feature_compute import (
    compute_features_for_date,
    load_dhan_options_features,
    BASE_FEATURE_COLS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Config ───────────────────────────────────────────────────────────
DB_URL = "postgresql://neondb_owner:npg_g8DMmJ3XByNF@ep-flat-tree-a1bahirx-pooler.ap-southeast-1.aws.neon.tech/neondb"
INITIAL_CAPITAL = 2_500_000
NIFTY_LOT_SIZE = 25
RISK_FREE_RATE = 0.07
MARGIN_PER_LOT = 13_750
BULL_OTM_SELL = 0.03
BULL_OTM_BUY = 0.055
VIX_MIN_ENTRY = 8
VIX_MAX_ENTRY = 25
BACKFILL_START = date(2026, 1, 1)

ACTIVE_VERSIONS = ["v5.4.2", "v5.4.3", "v5.4.4"]
VERSION_CONFIGS = {
    "v5.4.2": {
        "SIGNAL_MAPPING": "sharp",
        "POSITION_SIZE_PCT": 0.20, "IC_POSITION_SIZE_PCT": 0.20,
        "MAX_CONCURRENT_POSITIONS": 3, "IC_MAX_CONCURRENT": 3,
        "MIN_ENTRY_GAP_DAYS": 2, "MIN_DTE_EXIT": 1,
        "PROFIT_TARGET_EARLY": 0.50, "PROFIT_TARGET_MID": 0.65, "PROFIT_TARGET_LATE": 0.80,
        "STOP_LOSS_MULTIPLIER": 3.0, "IC_STOP_LOSS_MULTIPLIER": 3.0,
        "STOP_LOSS_CONFIRM_DAYS": 2, "IC_STOP_LOSS_CONFIRM_DAYS": 2,
        "TRAILING_STOP_ACTIVATE": 0.40, "TRAILING_STOP_LEVEL": 0.10,
        "IC_PUT_OTM_SELL": 0.03, "IC_PUT_OTM_BUY": 0.055,
        "IC_CALL_OTM_SELL": 0.04, "IC_CALL_OTM_BUY": 0.065,
    },
    "v5.4.3": {
        "SIGNAL_MAPPING": "graduated", "GRADUATED_FLOOR": 0.50, "GRADUATED_HW": 0.50,
        "POSITION_SIZE_PCT": 0.15, "IC_POSITION_SIZE_PCT": 0.15,
        "MAX_CONCURRENT_POSITIONS": 3, "IC_MAX_CONCURRENT": 2,
        "MIN_ENTRY_GAP_DAYS": 3, "MIN_DTE_EXIT": 1,
        "PROFIT_TARGET_EARLY": 0.45, "PROFIT_TARGET_MID": 0.60, "PROFIT_TARGET_LATE": 0.75,
        "STOP_LOSS_MULTIPLIER": 2.5, "IC_STOP_LOSS_MULTIPLIER": 2.5,
        "STOP_LOSS_CONFIRM_DAYS": 1, "IC_STOP_LOSS_CONFIRM_DAYS": 1,
        "TRAILING_STOP_ACTIVATE": 0.40, "TRAILING_STOP_LEVEL": 0.10,
        "IC_PUT_OTM_SELL": 0.03, "IC_PUT_OTM_BUY": 0.055,
        "IC_CALL_OTM_SELL": 0.04, "IC_CALL_OTM_BUY": 0.065,
    },
    "v5.4.4": {
        "SIGNAL_MAPPING": "graduated_gentle", "GRADUATED_FLOOR": 0.80, "GRADUATED_HW": 0.25,
        "POSITION_SIZE_PCT": 0.20, "IC_POSITION_SIZE_PCT": 0.20,
        "MAX_CONCURRENT_POSITIONS": 3, "IC_MAX_CONCURRENT": 3,
        "MIN_ENTRY_GAP_DAYS": 2, "MIN_DTE_EXIT": 1,
        "PROFIT_TARGET_EARLY": 0.45, "PROFIT_TARGET_MID": 0.60, "PROFIT_TARGET_LATE": 0.75,
        "STOP_LOSS_MULTIPLIER": 2.5, "IC_STOP_LOSS_MULTIPLIER": 2.5,
        "STOP_LOSS_CONFIRM_DAYS": 1, "IC_STOP_LOSS_CONFIRM_DAYS": 1,
        "TRAILING_STOP_ACTIVATE": 0.40, "TRAILING_STOP_LEVEL": 0.10,
        "IC_PUT_OTM_SELL": 0.03, "IC_PUT_OTM_BUY": 0.055,
        "IC_CALL_OTM_SELL": 0.04, "IC_CALL_OTM_BUY": 0.065,
    },
}

TRAINING_FEATURES = BASE_FEATURE_COLS + [
    "deep_otm_oi_ratio", "deep_otm_oi_ratio_change_5d",
    "put_oi_buildup_ratio", "put_volume_surge_ratio",
    "iv_skew_steepness", "iv_skew_change_5d", "atm_iv",
    "atm_iv_percentile_252d", "atm_put_intraday_range",
]

# Paths to ground truth data (shipped in backend/data/)
MERGED_DAILY_PATH = os.path.join(BACKEND_ROOT, "data", "merged_daily.parquet")
DHAN_RAW_PATH = os.path.join(BACKEND_ROOT, "data", "dhan_raw_options.parquet")
DHAN_SKEW_PATH = os.path.join(BACKEND_ROOT, "data", "daily_iv_skew_params.parquet")


# ── Signal mapping (replicated from signal_mapper.py) ────────────────

def map_signal_sharp(pred):
    pred_pct = pred * 100
    if pred_pct > -3.8:
        return {"signal": "bull_full", "size_mult": 1.0, "trade_type": "bull_put"}
    elif pred_pct > -5.0:
        return {"signal": "bull_half", "size_mult": 1.0, "trade_type": "bull_put"}
    elif pred_pct > -6.5:
        return {"signal": "iron_condor", "size_mult": 1.0, "trade_type": "iron_condor"}
    else:
        return {"signal": "no_trade", "size_mult": 0.0, "trade_type": None}


def map_signal_graduated(pred, floor=0.50, hw=0.50):
    pred_pct = pred * 100
    t1, t2, t3 = -3.8, -5.0, -6.5
    if pred_pct > t1 + hw:
        return {"signal": "bull_full", "size_mult": 1.0, "trade_type": "bull_put"}
    elif pred_pct > t1 - hw:
        progress = (t1 + hw - pred_pct) / (2 * hw)
        mult = 1.0 - progress * (1.0 - floor)
        return {"signal": "bull_full", "size_mult": round(mult, 3), "trade_type": "bull_put"}
    elif pred_pct > t2 + hw:
        return {"signal": "bull_half", "size_mult": floor, "trade_type": "bull_put"}
    elif pred_pct > t2 - hw:
        return {"signal": "iron_condor", "size_mult": round(max(floor, 0), 3), "trade_type": "iron_condor"}
    elif pred_pct > t3 + hw:
        return {"signal": "iron_condor", "size_mult": floor, "trade_type": "iron_condor"}
    elif pred_pct > t3 - hw:
        progress = (t3 + hw - pred_pct) / (2 * hw)
        mult = floor * (1.0 - progress)
        if mult > 0.1:
            return {"signal": "iron_condor", "size_mult": round(mult, 3), "trade_type": "iron_condor"}
        return {"signal": "no_trade", "size_mult": 0.0, "trade_type": None}
    else:
        return {"signal": "no_trade", "size_mult": 0.0, "trade_type": None}


def map_signal(pred, cfg):
    mapping = cfg.get("SIGNAL_MAPPING", "sharp")
    if mapping == "sharp":
        return map_signal_sharp(pred)
    elif mapping == "graduated":
        return map_signal_graduated(pred, cfg.get("GRADUATED_FLOOR", 0.50), cfg.get("GRADUATED_HW", 0.50))
    elif mapping == "graduated_gentle":
        return map_signal_graduated(pred, cfg.get("GRADUATED_FLOOR", 0.80), cfg.get("GRADUATED_HW", 0.25))
    return map_signal_sharp(pred)


# ── Option pricing (B-S with IV skew + market premium) ──────────────

from scipy.stats import norm

MARKET_PREMIUM = 1.25   # 25% markup on all option prices (matches backtest)
ENTRY_SLIPPAGE = 0.02   # 2% adverse fill on entry

def _put_sigma_with_skew(S, K, base_iv):
    """Put IV with skew: moneyness = (S-K)/S, skew = 1 + moneyness*3."""
    moneyness = (S - K) / S if S > 0 else 0
    skew = 1.0 + moneyness * 3.0
    return base_iv * max(skew, 1.0)

def _call_sigma_with_skew(S, K, base_iv):
    """Call IV with skew: moneyness = (K-S)/S, sigma = base*(1 + moneyness*1.5)."""
    moneyness = (K - S) / S if S > 0 else 0
    sigma = base_iv * (1.0 + moneyness * 1.5)
    return max(sigma, base_iv * 0.8)

def _bs_put(S, K, T, r, sigma):
    base_iv = max(sigma, 0.05)
    adj_sigma = _put_sigma_with_skew(S, K, base_iv)
    if T <= 0 or adj_sigma <= 0:
        return max(K - S, 0) * MARKET_PREMIUM
    d1 = (np.log(S / K) + (r + 0.5 * adj_sigma ** 2) * T) / (adj_sigma * np.sqrt(T))
    d2 = d1 - adj_sigma * np.sqrt(T)
    price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return max(price, 0) * MARKET_PREMIUM

def _bs_call(S, K, T, r, sigma):
    base_iv = max(sigma, 0.05)
    adj_sigma = _call_sigma_with_skew(S, K, base_iv)
    if T <= 0 or adj_sigma <= 0:
        return max(S - K, 0) * MARKET_PREMIUM
    d1 = (np.log(S / K) + (r + 0.5 * adj_sigma ** 2) * T) / (adj_sigma * np.sqrt(T))
    d2 = d1 - adj_sigma * np.sqrt(T)
    price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return max(price, 0) * MARKET_PREMIUM

def price_bull_put(S, sell_K, buy_K, T, r, sigma, apply_slippage=False):
    credit = _bs_put(S, sell_K, T, r, sigma) - _bs_put(S, buy_K, T, r, sigma)
    credit = max(credit, 0)
    if apply_slippage:
        credit *= (1 - ENTRY_SLIPPAGE)
    return credit

def price_iron_condor_func(S, put_sell, put_buy, call_sell, call_buy, T, r, sigma, apply_slippage=False):
    put_credit = _bs_put(S, put_sell, T, r, sigma) - _bs_put(S, put_buy, T, r, sigma)
    call_credit = _bs_call(S, call_sell, T, r, sigma) - _bs_call(S, call_buy, T, r, sigma)
    total = put_credit + call_credit
    total = max(total, 0)
    if apply_slippage:
        total *= (1 - ENTRY_SLIPPAGE)
    return total

def compute_spread_value(trade_type, S, sell_K, buy_K, ic_cs, ic_cb, T, r, sigma):
    """Compute current spread value (no slippage — that's only for entry)."""
    if trade_type == "bull_put":
        return price_bull_put(S, sell_K, buy_K, T, r, sigma)
    elif trade_type == "iron_condor" and ic_cs and ic_cb:
        return price_iron_condor_func(S, sell_K, buy_K, ic_cs, ic_cb, T, r, sigma)
    return 0

def get_next_weekly_expiry(from_date, min_dte=5):
    """Get next weekly expiry (Thursday) with at least min_dte days remaining."""
    d = from_date
    while d.weekday() != 3:  # Thursday
        d += timedelta(days=1)
    if d == from_date:
        d += timedelta(days=7)
    # Ensure minimum DTE
    while (d - from_date).days < min_dte:
        d += timedelta(days=7)
    return d


# ── Main backfill ────────────────────────────────────────────────────

async def run_backfill():
    import asyncpg

    ssl_ctx = ssl.create_default_context()
    ssl_ctx.check_hostname = False
    ssl_ctx.verify_mode = ssl.CERT_NONE

    logger.info("Connecting to Neon DB...")
    conn = await asyncpg.connect(DB_URL, ssl=ssl_ctx)
    logger.info("Connected!")

    # Clear existing data
    logger.info("Clearing existing data...")
    await conn.execute("DELETE FROM daily_pnl")
    await conn.execute("DELETE FROM trades")
    await conn.execute("DELETE FROM predictions")
    await conn.execute("DELETE FROM daily_features")
    logger.info("Data cleared")

    # Load ground truth data from merged_daily.parquet (same source as backtest)
    if not os.path.exists(MERGED_DAILY_PATH):
        logger.error(f"merged_daily.parquet not found at {MERGED_DAILY_PATH}")
        await conn.close()
        return

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
    logger.info("Computing features via shared module...")
    all_dates = [d for d in merged_daily.index if d.date() >= BACKFILL_START]
    feature_cache = {}
    for d in all_dates:
        features = compute_features_for_date(
            merged_daily, d, dhan_features,
            data_start="2014-01-01", data_end="2026-12-31",
        )
        if features is not None:
            feature_cache[d] = features
    logger.info(f"Computed features for {len(feature_cache)} trading days")
    backfill_dates = sorted(feature_cache.keys())

    # Load model
    model = joblib.load("ml/downside_model.pkl")
    feature_names = joblib.load("ml/feature_names.pkl")
    logger.info(f"Model loaded, {len(feature_names)} features")

    # Version state
    version_state = {v: {"capital": INITIAL_CAPITAL, "cumulative_pnl": 0.0, "open_trades": []}
                      for v in ACTIVE_VERSIONS}

    # Trade log for comparison
    trade_log = {v: [] for v in ACTIVE_VERSIONS}

    stats = {"days": 0, "predictions": 0, "trades_opened": 0, "trades_closed": 0}

    logger.info(f"Backfilling {len(backfill_dates)} trading days")

    for ts in backfill_dates:
        trade_date = ts.date()

        features_dict = feature_cache[ts]

        spot = float(merged_daily.loc[ts, 'nifty_close'])
        vix = features_dict["india_vix"]
        if pd.isna(spot) or spot <= 0:
            continue

        # Build ordered feature vector for model prediction
        feat_values = [features_dict.get(c, 0.0) for c in feature_names]

        # Predict
        X = np.array(feat_values).reshape(1, -1)
        prediction_value = float(model.predict(X)[0])

        # Store daily features
        await conn.execute("""
            INSERT INTO daily_features (date, features, vix, vix_20d_avg, nifty_20d_return, nifty_50d_return,
                iv_skew, fii_net, dii_net, put_call_ratio, rsi_14, adx_14)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
            ON CONFLICT (date) DO NOTHING
        """, trade_date, json.dumps(features_dict), vix, vix,
            features_dict.get("nifty_return_20d", 0), features_dict.get("nifty_distance_50dma", 0),
            features_dict.get("iv_skew_steepness", 0), 0.0, 0.0,
            features_dict.get("pcr_proxy", 0), features_dict.get("rsi_14", 50), 0.0)

        # Process each version
        for version in ACTIVE_VERSIONS:
            cfg = VERSION_CONFIGS[version]
            state = version_state[version]
            signal = map_signal(prediction_value, cfg)

            # Confidence score
            pred_pct = prediction_value * 100
            zones = [(-0.5, 0), (-1.0, -0.5), (-1.5, -1.0), (-2.5, -1.5), (-3.5, -2.5), (-5.0, -3.5)]
            confidence = 0.5
            for zmin, zmax in zones:
                if zmin < pred_pct <= zmax:
                    confidence = min(abs(pred_pct - zmin), abs(pred_pct - zmax))
                    break

            # Store prediction
            ts_val = datetime.combine(trade_date, datetime.min.time().replace(hour=9, minute=20))
            await conn.execute("""
                INSERT INTO predictions (date, timestamp, predicted_drawdown, signal_type, version,
                    nifty_spot, vix, confidence_score, graduated_mult, features)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                ON CONFLICT ON CONSTRAINT uq_prediction_date_version DO NOTHING
            """, trade_date, ts_val, prediction_value, signal["signal"], version,
                spot, vix, confidence, signal["size_mult"], json.dumps(features_dict))
            stats["predictions"] += 1

            # Check exits — aligned with backtest logic
            trades_to_close = []
            for ot in state["open_trades"]:
                dte = (ot["expiry"] - trade_date).days
                T = max(dte / 365.0, 1 / 365.0)
                sigma = vix / 100.0 if vix > 0 else 0.15

                cv = compute_spread_value(
                    ot["trade_type"], spot, ot["sell_strike"], ot["buy_strike"],
                    ot.get("ic_call_sell"), ot.get("ic_call_buy"), T, RISK_FREE_RATE, sigma)

                pnl_per_unit = ot["credit"] - cv
                pnl_pct = pnl_per_unit / ot["credit"] if ot["credit"] > 0 else 0

                exit_reason = None

                # 1. Expiry exit
                if dte <= cfg.get("MIN_DTE_EXIT", 1):
                    exit_reason = "expiry"

                # 2. Profit target — day-count stages matching backtest
                if not exit_reason:
                    days_held = (trade_date - ot["entry_date"]).days
                    if days_held <= 3:
                        pt = cfg["PROFIT_TARGET_EARLY"]
                    elif days_held <= 7:
                        pt = cfg["PROFIT_TARGET_MID"]
                    else:
                        pt = cfg["PROFIT_TARGET_LATE"]
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
                    sl_mult = cfg["IC_STOP_LOSS_MULTIPLIER"] if ot["trade_type"] == "iron_condor" else cfg["STOP_LOSS_MULTIPLIER"]
                    confirm_days = cfg.get("IC_STOP_LOSS_CONFIRM_DAYS", 1) if ot["trade_type"] == "iron_condor" else cfg.get("STOP_LOSS_CONFIRM_DAYS", 1)
                    if cv >= ot["credit"] * sl_mult + ot["credit"]:
                        ot["stop_loss_breach_days"] = ot.get("stop_loss_breach_days", 0) + 1
                        if ot["stop_loss_breach_days"] >= confirm_days:
                            exit_reason = "stop_loss"
                    else:
                        ot["stop_loss_breach_days"] = 0

                if exit_reason:
                    rpnl = pnl_per_unit * ot["num_lots"] * NIFTY_LOT_SIZE
                    rpnl_pct = rpnl / ot["capital_deployed"] * 100 if ot["capital_deployed"] > 0 else 0
                    await conn.execute("""
                        UPDATE trades SET status='closed', exit_date=$1, exit_spot=$2,
                        exit_reason=$3, realized_pnl=$4, current_pnl=$4, current_pnl_pct=$5,
                        exit_time=$6 WHERE trade_id=$7
                    """, trade_date, spot, exit_reason, rpnl, rpnl_pct,
                        datetime.combine(trade_date, datetime.min.time().replace(hour=15, minute=30)),
                        ot["trade_id"])
                    state["cumulative_pnl"] += rpnl
                    state["capital"] = INITIAL_CAPITAL + state["cumulative_pnl"]
                    trades_to_close.append(ot["trade_id"])
                    trade_log[version].append({
                        "type": ot["trade_type"], "entry": ot["entry_date"].isoformat(),
                        "exit": trade_date.isoformat(), "pnl": round(rpnl),
                        "exit_reason": exit_reason, "days": (trade_date - ot["entry_date"]).days,
                    })
                    stats["trades_closed"] += 1

            state["open_trades"] = [t for t in state["open_trades"] if t["trade_id"] not in trades_to_close]

            # Open new trade
            if signal["signal"] != "no_trade" and signal["trade_type"]:
                tt = signal["trade_type"]
                sm = signal["size_mult"]

                open_count = sum(1 for t in state["open_trades"] if t["trade_type"] == tt)
                max_c = cfg["IC_MAX_CONCURRENT"] if tt == "iron_condor" else cfg["MAX_CONCURRENT_POSITIONS"]
                last_entry = max((t["entry_date"] for t in state["open_trades"]), default=None)
                gap_ok = last_entry is None or (trade_date - last_entry).days >= cfg["MIN_ENTRY_GAP_DAYS"]
                vix_ok = VIX_MIN_ENTRY <= vix <= VIX_MAX_ENTRY

                if open_count < max_c and gap_ok and vix_ok:
                    # Select strikes
                    sell_k = round(spot * (1 - BULL_OTM_SELL) / 50) * 50
                    buy_k = round(spot * (1 - BULL_OTM_BUY) / 50) * 50
                    ic_cs = round(spot * (1 + cfg["IC_CALL_OTM_SELL"]) / 50) * 50 if tt == "iron_condor" else None
                    ic_cb = round(spot * (1 + cfg["IC_CALL_OTM_BUY"]) / 50) * 50 if tt == "iron_condor" else None

                    expiry = get_next_weekly_expiry(trade_date)
                    T = max((expiry - trade_date).days / 365.0, 1 / 365.0)
                    sigma = vix / 100.0 if vix > 0 else 0.15

                    if tt == "bull_put":
                        credit = price_bull_put(spot, sell_k, buy_k, T, RISK_FREE_RATE, sigma, apply_slippage=True)
                    else:
                        credit = price_iron_condor_func(spot, sell_k, buy_k, ic_cs, ic_cb, T, RISK_FREE_RATE, sigma, apply_slippage=True)

                    pos_pct = cfg["IC_POSITION_SIZE_PCT"] if tt == "iron_condor" else cfg["POSITION_SIZE_PCT"]
                    eff_pct = pos_pct * sm
                    max_cap = state["capital"] * eff_pct
                    num_lots = max(1, int(max_cap / MARGIN_PER_LOT))
                    total_credit = credit * num_lots * NIFTY_LOT_SIZE
                    cap_deployed = num_lots * MARGIN_PER_LOT

                    trade_id = f"{version.replace('.', '')}-{trade_date.isoformat()}-{signal['signal']}"
                    entry_time = datetime.combine(trade_date, datetime.min.time().replace(hour=9, minute=20))

                    await conn.execute("""
                        INSERT INTO trades (trade_id, version, date, signal_type, trade_type, entry_mode,
                            entry_date, entry_time, entry_spot, expiry, sell_strike, buy_strike,
                            ic_call_sell, ic_call_buy, num_lots, credit_received, total_credit,
                            status, current_pnl, current_pnl_pct, unrealized_pnl,
                            position_size_pct, graduated_mult, capital_deployed)
                        VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19,$20,$21,$22,$23,$24)
                    """, trade_id, version, trade_date, signal["signal"], tt, "normal",
                        trade_date, entry_time, spot, expiry, sell_k, buy_k, ic_cs, ic_cb,
                        num_lots, credit, total_credit, "open", 0.0, 0.0, 0.0,
                        eff_pct, sm, cap_deployed)

                    state["open_trades"].append({
                        "trade_id": trade_id, "trade_type": tt, "entry_date": trade_date,
                        "expiry": expiry, "sell_strike": sell_k, "buy_strike": buy_k,
                        "ic_call_sell": ic_cs, "ic_call_buy": ic_cb,
                        "credit": credit, "num_lots": num_lots, "capital_deployed": cap_deployed,
                        "peak_pnl_pct": 0.0, "stop_loss_breach_days": 0,
                    })
                    stats["trades_opened"] += 1

            # Calculate unrealized PnL for open positions
            unrealized_pnl = 0.0
            for ot in state["open_trades"]:
                dte = (ot["expiry"] - trade_date).days
                T = max(dte / 365.0, 1 / 365.0)
                sigma = vix / 100.0 if vix > 0 else 0.15
                cv = compute_spread_value(
                    ot["trade_type"], spot, ot["sell_strike"], ot["buy_strike"],
                    ot.get("ic_call_sell"), ot.get("ic_call_buy"), T, RISK_FREE_RATE, sigma)
                unrealized_pnl += (ot["credit"] - cv) * ot["num_lots"] * NIFTY_LOT_SIZE

            total_pnl = state["cumulative_pnl"] + unrealized_pnl
            prev_total = state.get("prev_total_pnl", 0.0)
            daily_change = total_pnl - prev_total
            state["prev_total_pnl"] = total_pnl

            # Daily PnL
            await conn.execute("""
                INSERT INTO daily_pnl (date, version, starting_capital, ending_capital,
                    daily_pnl, cumulative_pnl, cumulative_return_pct, open_positions, trades_opened, trades_closed)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10)
                ON CONFLICT ON CONSTRAINT uq_daily_pnl_date_version DO NOTHING
            """, trade_date, version, float(INITIAL_CAPITAL), float(INITIAL_CAPITAL + total_pnl),
                float(daily_change), float(total_pnl),
                round(total_pnl / INITIAL_CAPITAL * 100, 4),
                len(state["open_trades"]), 0, 0)

        stats["days"] += 1
        if stats["days"] % 5 == 0:
            logger.info(f"Progress: {stats['days']} days, {stats['predictions']} preds, "
                        f"{stats['trades_opened']} opened, {stats['trades_closed']} closed")

    await conn.close()
    logger.info(f"=== BACKFILL COMPLETE: {stats} ===")

    # Print trade-by-trade comparison
    print("\n" + "=" * 90)
    print("BACKFILL TRADE LOG — Compare against backtest strategy_comparison.txt")
    print("=" * 90)
    for v in ACTIVE_VERSIONS:
        trades = trade_log[v]
        total_pnl = sum(t["pnl"] for t in trades)
        print(f"\n--- {v} ({len(trades)} trades, Total PnL: Rs {total_pnl:,.0f}) ---")
        print(f"  {'Type':<15} {'Entry':<12} {'Exit':<12} {'PnL':>10} {'Exit Reason':<20} {'Days':>5}")
        print(f"  {'-'*80}")
        for t in trades:
            print(f"  {t['type']:<15} {t['entry']:<12} {t['exit']:<12} Rs {t['pnl']:>8,} {t['exit_reason']:<20} {t['days']:>5}")
    print()

    return stats


if __name__ == "__main__":
    result = asyncio.run(run_backfill())
    print(f"\nResult: {result}")
