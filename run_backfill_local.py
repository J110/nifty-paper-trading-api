"""
Run the backfill locally against the remote Neon database.
Uses raw asyncpg (no ORM) to bypass Python 3.10+ requirement for SQLAlchemy Mapped types.

Usage:
    cd /Users/Anmol/Desktop/Options\ Trading/backend
    python3 run_backfill_local.py
"""

import asyncio
import ssl
import json
import logging
import math
import time
import numpy as np
import pandas as pd
import ta as ta_lib
import yfinance as yf
import joblib
from datetime import date, datetime, timedelta

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
        "IC_PUT_OTM_SELL": 0.03, "IC_PUT_OTM_BUY": 0.055,
        "IC_CALL_OTM_SELL": 0.04, "IC_CALL_OTM_BUY": 0.065,
    },
}

TRAINING_FEATURES = [
    "nifty_return_5d", "nifty_return_10d", "nifty_return_20d",
    "nifty_distance_50dma", "nifty_distance_200dma", "golden_cross",
    "rsi_14", "higher_highs_5w", "india_vix", "vix_change_5d",
    "vix_percentile_252d", "realized_vol_20d", "variance_risk_premium",
    "vix_rising_streak", "pcr_proxy", "sp500_return_5d", "us_vix",
    "us_vix_change_5d", "dxy_level", "dxy_change_5d", "us10y_level",
    "us10y_change_5d", "us10y_change_20d", "day_of_month", "month",
    "is_expiry_week", "days_to_monthly_expiry", "is_volatile_month",
    "deep_otm_oi_ratio", "deep_otm_oi_ratio_change_5d",
    "put_oi_buildup_ratio", "put_volume_surge_ratio",
    "iv_skew_steepness", "iv_skew_change_5d", "atm_iv",
    "atm_iv_percentile_252d", "atm_put_intraday_range",
]


# ── Signal mapping (replicated from signal_mapper.py) ────────────────

def map_signal_sharp(pred):
    pred_pct = pred * 100
    if pred_pct > -1.5:
        return {"signal": "bull_full", "size_mult": 1.0, "trade_type": "bull_put"}
    elif pred_pct > -2.5:
        return {"signal": "bull_half", "size_mult": 1.0, "trade_type": "bull_put"}
    elif pred_pct > -3.5:
        return {"signal": "iron_condor", "size_mult": 1.0, "trade_type": "iron_condor"}
    else:
        return {"signal": "no_trade", "size_mult": 0.0, "trade_type": None}


def map_signal_graduated(pred, floor=0.50, hw=0.50):
    pred_pct = pred * 100
    t1, t2, t3 = -1.5, -2.5, -3.5
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


# ── Option pricing (B-S) ────────────────────────────────────────────

from scipy.stats import norm

def _bs_put(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return max(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def _bs_call(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def price_bull_put(S, sell_K, buy_K, T, r, sigma):
    return _bs_put(S, sell_K, T, r, sigma) - _bs_put(S, buy_K, T, r, sigma)

def price_iron_condor(S, put_sell, put_buy, call_sell, call_buy, T, r, sigma):
    put_credit = _bs_put(S, put_sell, T, r, sigma) - _bs_put(S, put_buy, T, r, sigma)
    call_credit = _bs_call(S, call_sell, T, r, sigma) - _bs_call(S, call_buy, T, r, sigma)
    return put_credit + call_credit

def compute_spread_value(trade_type, S, sell_K, buy_K, ic_cs, ic_cb, T, r, sigma):
    if trade_type == "bull_put":
        return price_bull_put(S, sell_K, buy_K, T, r, sigma)
    elif trade_type == "iron_condor" and ic_cs and ic_cb:
        return price_iron_condor(S, sell_K, buy_K, ic_cs, ic_cb, T, r, sigma)
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


# ── Feature helpers ──────────────────────────────────────────────────

def _get_last_thursday(year, month):
    if month == 12:
        nf = date(year + 1, 1, 1)
    else:
        nf = date(year, month + 1, 1)
    last_day = nf - timedelta(days=1)
    offset = (last_day.weekday() - 3) % 7
    return last_day - timedelta(days=offset)

def _get_monthly_expiries(start, end):
    expiries = []
    y, m = start.year, start.month
    while True:
        exp = _get_last_thursday(y, m)
        if exp > end:
            break
        if exp >= start:
            expiries.append(exp)
        m += 1
        if m > 12:
            m = 1
            y += 1
    return expiries


def download_all_data():
    """Download all market data needed."""
    start = date(2025, 1, 1)
    end = date.today()

    tickers = {
        "nifty": "^NSEI",
        "india_vix": "^INDIAVIX",
        "sp500": "^GSPC",
        "us_vix": "^VIX",
        "dxy": "DX-Y.NYB",
        "us10y": "^TNX",
    }

    results = {}
    for name, ticker in tickers.items():
        logger.info(f"Downloading {name} ({ticker})...")
        for attempt in range(3):
            data = yf.download(ticker, start=start.isoformat(), end=end.isoformat(), progress=False)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            if not data.empty:
                logger.info(f"  {name}: {len(data)} days")
                results[name] = data
                break
            time.sleep(2)
        else:
            logger.warning(f"  {name}: FAILED, using empty DataFrame")
            results[name] = pd.DataFrame()
        time.sleep(0.5)

    return results


def build_feature_matrix(data):
    """Build full feature matrix from downloaded data."""
    nifty = data["nifty"]
    india_vix_df = data["india_vix"]
    sp500 = data["sp500"]
    us_vix_df = data["us_vix"]
    dxy = data["dxy"]
    us10y = data["us10y"]

    df = pd.DataFrame(index=nifty.index)
    close = nifty["Close"]
    high = nifty["High"]

    # Trend & Momentum
    df["nifty_return_5d"] = close.pct_change(5)
    df["nifty_return_10d"] = close.pct_change(10)
    df["nifty_return_20d"] = close.pct_change(20)

    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    df["nifty_distance_50dma"] = (close - sma50) / sma50 * 100
    df["nifty_distance_200dma"] = (close - sma200) / sma200 * 100
    df["golden_cross"] = (sma50 > sma200).astype(int)

    rsi = ta_lib.momentum.RSIIndicator(close, window=14)
    df["rsi_14"] = rsi.rsi()

    weekly_high = high.resample("W").max()
    weekly_hh = (weekly_high > weekly_high.shift(1)).astype(int)
    weekly_hh_count = weekly_hh.rolling(5, min_periods=1).sum()
    df["higher_highs_5w"] = weekly_hh_count.reindex(df.index, method="ffill").fillna(0)

    # Volatility
    vix_close = india_vix_df["Close"].reindex(df.index).ffill().fillna(14.0)
    df["india_vix"] = vix_close
    df["vix_change_5d"] = vix_close - vix_close.shift(5)

    df["vix_percentile_252d"] = vix_close.rolling(252, min_periods=60).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    )

    log_returns = np.log(close / close.shift(1))
    df["realized_vol_20d"] = log_returns.rolling(20).std() * np.sqrt(252) * 100
    df["variance_risk_premium"] = df["india_vix"] - df["realized_vol_20d"]

    vix_rising = (vix_close > vix_close.shift(1)).astype(int)
    vix_streak = pd.Series(0, index=df.index, dtype=int)
    for i in range(1, len(df)):
        if vix_rising.iloc[i] == 1:
            vix_streak.iloc[i] = vix_streak.iloc[i - 1] + 1
    df["vix_rising_streak"] = vix_streak

    # Sentiment
    df["pcr_proxy"] = df["vix_change_5d"] / (df["nifty_return_5d"] * 100 + 0.001)

    # Global context
    sp_close = sp500["Close"].reindex(df.index).ffill() if len(sp500) > 0 else pd.Series(0, index=df.index)
    df["sp500_return_5d"] = sp_close.pct_change(5) if sp_close.notna().sum() >= 100 else 0.0

    usvix_close = us_vix_df["Close"].reindex(df.index).ffill() if len(us_vix_df) > 0 else pd.Series(0, index=df.index)
    df["us_vix"] = usvix_close.fillna(0)
    df["us_vix_change_5d"] = (usvix_close - usvix_close.shift(5)) if usvix_close.notna().sum() >= 100 else 0.0

    dxy_close = dxy["Close"].reindex(df.index).ffill() if len(dxy) > 0 else pd.Series(0, index=df.index)
    df["dxy_level"] = dxy_close.fillna(0)
    df["dxy_change_5d"] = (dxy_close - dxy_close.shift(5)) if dxy_close.notna().sum() >= 100 else 0.0

    us10y_close = us10y["Close"].reindex(df.index).ffill() if len(us10y) > 0 else pd.Series(0, index=df.index)
    df["us10y_level"] = us10y_close.fillna(0)
    if us10y_close.notna().sum() >= 100:
        df["us10y_change_5d"] = us10y_close - us10y_close.shift(5)
        df["us10y_change_20d"] = us10y_close - us10y_close.shift(20)
    else:
        df["us10y_change_5d"] = 0.0
        df["us10y_change_20d"] = 0.0

    # Calendar
    df["day_of_month"] = df.index.day
    df["month"] = df.index.month
    df["is_volatile_month"] = df["month"].isin([9, 10]).astype(int)

    first_date = df.index[0].date()
    last_date = df.index[-1].date()
    expiries = _get_monthly_expiries(first_date, last_date + timedelta(days=60))
    trading_dates = df.index.date

    dte_list, iew_list = [], []
    for d in df.index:
        d_date = d.date()
        next_exp = next((e for e in expiries if e >= d_date), None)
        if next_exp:
            dte = len([td for td in trading_dates if d_date <= td <= next_exp])
            dte_list.append(dte)
            iew_list.append(1 if dte <= 5 else 0)
        else:
            dte_list.append(30)
            iew_list.append(0)
    df["is_expiry_week"] = iew_list
    df["days_to_monthly_expiry"] = dte_list

    # Options placeholders
    df["deep_otm_oi_ratio"] = 0.0
    df["deep_otm_oi_ratio_change_5d"] = 0.0
    df["put_oi_buildup_ratio"] = 0.0
    df["put_volume_surge_ratio"] = 1.0
    df["iv_skew_steepness"] = 0.0
    df["iv_skew_change_5d"] = 0.0
    df["atm_iv"] = df["india_vix"] / 100.0
    df["atm_iv_percentile_252d"] = df["vix_percentile_252d"]
    df["atm_put_intraday_range"] = 0.0

    df = df.fillna(0.0).replace([np.inf, -np.inf], 0.0)
    return df


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

    # Download data
    data = download_all_data()
    if data["nifty"].empty:
        logger.error("Failed to download Nifty data!")
        await conn.close()
        return

    # Build features
    logger.info("Building feature matrix...")
    feature_df = build_feature_matrix(data)
    logger.info(f"Feature matrix: {feature_df.shape}")

    # Load model
    model = joblib.load("ml/downside_model.pkl")
    feature_names = joblib.load("ml/feature_names.pkl")
    logger.info(f"Model loaded, {len(feature_names)} features")

    # Get backfill dates
    nifty_close = data["nifty"]["Close"]
    backfill_dates = [d for d in data["nifty"].index if d.date() >= BACKFILL_START]
    logger.info(f"Backfilling {len(backfill_dates)} trading days")

    # Version state
    version_state = {v: {"capital": INITIAL_CAPITAL, "cumulative_pnl": 0.0, "open_trades": []}
                      for v in ACTIVE_VERSIONS}

    stats = {"days": 0, "predictions": 0, "trades_opened": 0, "trades_closed": 0}

    for ts in backfill_dates:
        trade_date = ts.date()
        if ts not in feature_df.index:
            continue

        spot = float(nifty_close.loc[ts])
        vix = float(feature_df.loc[ts, "india_vix"])
        if pd.isna(spot) or spot <= 0:
            continue

        # Build feature vector
        feat_values = [float(feature_df.loc[ts, c]) if c in feature_df.columns and not pd.isna(feature_df.loc[ts, c]) else 0.0 for c in feature_names]
        features_dict = {c: feat_values[i] for i, c in enumerate(feature_names)}

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

            # Check exits
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
                if dte <= cfg.get("MIN_DTE_EXIT", 1):
                    exit_reason = "expiry"

                entry_dte = (ot["expiry"] - ot["entry_date"]).days
                elapsed = 1 - (dte / max(entry_dte, 1))
                pt = cfg["PROFIT_TARGET_EARLY"] if elapsed < 0.4 else (
                    cfg["PROFIT_TARGET_MID"] if elapsed < 0.7 else cfg["PROFIT_TARGET_LATE"])
                if pnl_pct >= pt:
                    exit_reason = "profit_target"

                sl_mult = cfg["IC_STOP_LOSS_MULTIPLIER"] if ot["trade_type"] == "iron_condor" else cfg["STOP_LOSS_MULTIPLIER"]
                if cv >= ot["credit"] * sl_mult + ot["credit"]:
                    exit_reason = "stop_loss"

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
                        credit = price_bull_put(spot, sell_k, buy_k, T, RISK_FREE_RATE, sigma)
                    else:
                        credit = price_iron_condor(spot, sell_k, buy_k, ic_cs, ic_cb, T, RISK_FREE_RATE, sigma)

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
    return stats


if __name__ == "__main__":
    result = asyncio.run(run_backfill())
    print(f"\nResult: {result}")
