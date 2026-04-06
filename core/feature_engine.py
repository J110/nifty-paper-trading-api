"""
Build features for live predictions using the SAME code as the backtest.

Delegates to shared/feature_compute.py — the single source of truth.
This file handles data loading, the Dhan live option chain overlay,
and display formatting. Feature computation itself lives in the shared module.
"""

import logging
import os
import numpy as np
import pandas as pd
from datetime import timedelta
from typing import Optional

logger = logging.getLogger(__name__)


class StaleDataError(Exception):
    """Raised when merged_daily.parquet is too stale for reliable predictions."""
    pass

# BACKEND_ROOT = directory containing core/, shared/, data/, ml/
# Locally: /Users/.../Options Trading/backend
# In Docker: /app
import sys
BACKEND_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND_ROOT)

from shared.feature_compute import (
    compute_features_for_date,
    load_dhan_options_features,
    BASE_FEATURE_COLS,
)

# Path to parquet data files (shipped in backend/data/)
MERGED_DAILY_PATH = os.path.join(BACKEND_ROOT, "data", "merged_daily.parquet")
DHAN_RAW_PATH = os.path.join(BACKEND_ROOT, "data", "dhan_raw_options.parquet")
DHAN_SKEW_PATH = os.path.join(BACKEND_ROOT, "data", "daily_iv_skew_params.parquet")

# The 37 training feature names — kept for reference / display code
TRAINING_FEATURES = BASE_FEATURE_COLS + [
    "deep_otm_oi_ratio", "deep_otm_oi_ratio_change_5d",
    "put_oi_buildup_ratio", "put_volume_surge_ratio",
    "iv_skew_steepness", "iv_skew_change_5d", "atm_iv",
    "atm_iv_percentile_252d", "atm_put_intraday_range",
]

# Display features — shown on the Market Signals page
DISPLAY_FEATURES = {
    "india_vix": {
        "label": "India VIX",
        "description": "Volatility index — fear gauge",
        "bullish_when": "Below 15",
        "bearish_when": "Above 20",
        "format": "{:.1f}",
    },
    "vix_percentile_252d": {
        "label": "VIX Percentile (252d)",
        "description": "Current VIX relative to last year",
        "bullish_when": "Below 30",
        "bearish_when": "Above 70",
        "format": "{:.0f}",
    },
    "nifty_return_20d": {
        "label": "Nifty 20-Day Return",
        "description": "Trailing 20-day price change",
        "bullish_when": "Above +3%",
        "bearish_when": "Below -3%",
        "format": "{:+.1%}",
    },
    "nifty_distance_50dma": {
        "label": "Distance to 50-DMA",
        "description": "% distance from 50-day moving average",
        "bullish_when": "Above +2%",
        "bearish_when": "Below -2%",
        "format": "{:+.1f}%",
    },
    "iv_skew_steepness": {
        "label": "IV Skew Steepness",
        "description": "Put vs Call implied vol difference",
        "bullish_when": "Below 2%",
        "bearish_when": "Above 5%",
        "format": "{:+.2f}",
    },
    "pcr_proxy": {
        "label": "PCR Proxy",
        "description": "VIX change per unit of Nifty return",
        "bullish_when": "Low values (fear receding)",
        "bearish_when": "High values (fear building)",
        "format": "{:.2f}",
    },
    "rsi_14": {
        "label": "RSI (14-Day)",
        "description": "Relative strength — overbought/oversold",
        "bullish_when": "Between 40-60 (healthy)",
        "bearish_when": "Above 75 (overbought) or Below 30 (oversold)",
        "format": "{:.1f}",
    },
    "sp500_return_5d": {
        "label": "S&P 500 (5-Day)",
        "description": "US market trend — global risk indicator",
        "bullish_when": "Positive",
        "bearish_when": "Negative",
        "format": "{:+.1%}",
    },
    "us_vix": {
        "label": "US VIX",
        "description": "CBOE Volatility Index — global fear gauge",
        "bullish_when": "Below 15",
        "bearish_when": "Above 25",
        "format": "{:.1f}",
    },
    "variance_risk_premium": {
        "label": "Variance Risk Premium",
        "description": "Implied vol minus realized vol",
        "bullish_when": "High (options expensive = premium sellers benefit)",
        "bearish_when": "Low or negative (realized catching up)",
        "format": "{:+.1f}",
    },
}


def _classify_indicator(name: str, value: float) -> str:
    """Classify an indicator as bullish/bearish/neutral for display."""
    rules = {
        "india_vix": lambda v: "bullish" if v < 15 else ("bearish" if v > 20 else "neutral"),
        "vix_percentile_252d": lambda v: "bullish" if v < 30 else (
            "bearish" if v > 70 else "neutral"),
        "nifty_return_20d": lambda v: "bullish" if v > 0.03 else (
            "bearish" if v < -0.03 else "neutral"),
        "nifty_distance_50dma": lambda v: "bullish" if v > 2 else (
            "bearish" if v < -2 else "neutral"),
        "iv_skew_steepness": lambda v: "bullish" if v < 0.02 else (
            "bearish" if v > 0.05 else "neutral"),
        "pcr_proxy": lambda v: "neutral",
        "rsi_14": lambda v: "bearish" if v > 75 else (
            "bearish" if v < 30 else "bullish" if 40 <= v <= 60 else "neutral"),
        "sp500_return_5d": lambda v: "bullish" if v > 0 else (
            "bearish" if v < -0.02 else "neutral"),
        "us_vix": lambda v: "bullish" if v < 15 else (
            "bearish" if v > 25 else "neutral"),
        "variance_risk_premium": lambda v: "bullish" if v > 3 else (
            "bearish" if v < 0 else "neutral"),
    }
    classifier = rules.get(name)
    if classifier:
        return classifier(value)
    return "neutral"


async def build_live_features(dhan_client, historical_df: pd.DataFrame,
                               option_chain: Optional[dict] = None) -> dict:
    """
    Build feature vector for today using the shared backtest feature code.

    Loads merged_daily.parquet (which must be kept up-to-date by the data
    collection pipeline) and calls compute_features_for_date() from the
    shared module — the EXACT same code that produced the backtest results.

    Args:
        dhan_client: DhanClient instance (used for live spot/VIX display values)
        historical_df: Not used for feature computation (kept for API compat).
                       Features come from merged_daily.parquet.
        option_chain: Optional pre-fetched option chain (not used for features)
    """
    from core.timezone import now_ist

    # Load the ground truth data source
    if not os.path.exists(MERGED_DAILY_PATH):
        logger.error(f"merged_daily.parquet not found at {MERGED_DAILY_PATH}")
        raise FileNotFoundError(
            f"Cannot compute features: {MERGED_DAILY_PATH} missing. "
            "Run data_collection.py to update it."
        )

    merged_daily = pd.read_parquet(MERGED_DAILY_PATH)
    last_parquet_date = merged_daily.index[-1].date()
    today = now_ist().date()
    logger.info(f"Loaded merged_daily: {len(merged_daily)} rows, "
                f"last date: {last_parquet_date}")

    # STALENESS GUARD: HARD BLOCK if parquet is more than 2 trading days behind.
    # (1 day behind is normal — we compute features at 9:20 AM using
    # yesterday's close. But 2+ days means the data update failed.)
    # Uses actual trading-day count (excl. weekends + NSE holidays) to avoid
    # false positives on long weekends like Good Friday + Sat/Sun.
    from core.market_holidays import trading_days_between
    trading_days_behind = trading_days_between(last_parquet_date, today)
    if trading_days_behind > 2:
        raise StaleDataError(
            f"merged_daily.parquet is {trading_days_behind} trading days stale "
            f"(last: {last_parquet_date}, today: {today}). "
            f"Data update likely failed — refusing to generate predictions."
        )

    # Load Dhan features if available
    dhan_features = None
    if os.path.exists(DHAN_RAW_PATH) and os.path.exists(DHAN_SKEW_PATH):
        dhan_features = load_dhan_options_features(DHAN_RAW_PATH, DHAN_SKEW_PATH)
        if dhan_features is not None:
            logger.info(f"Loaded Dhan features: {len(dhan_features)} days")

    # Use the most recent date in merged_daily as the target
    target_date = merged_daily.index[-1]
    logger.info(f"Computing features for target_date={target_date.date()}")

    # Call the shared module — EXACT same code as backtest
    features = compute_features_for_date(
        merged_daily, target_date, dhan_features,
        data_start="2014-01-01", data_end="2026-12-31",
    )

    if features is None:
        logger.error(f"compute_features_for_date returned None for {target_date.date()}")
        raise ValueError(f"Feature computation failed for {target_date.date()} — likely NaN warmup")

    # Get live spot/VIX for display (not used in model features)
    try:
        spot = await dhan_client.get_nifty_ltp()
        vix = await dhan_client.get_india_vix()
    except Exception as e:
        logger.warning(f"Failed to fetch live prices for display: {e}")
        spot = float(merged_daily['nifty_close'].iloc[-1])
        vix = features.get("india_vix", 14.0)

    # Add display-friendly aliases for the frontend (not model inputs)
    features["vix"] = features["india_vix"]
    features["nifty_close"] = spot if spot else float(merged_daily['nifty_close'].iloc[-1])
    features["vix_20d_avg"] = features["india_vix"]
    features["vix_percentile_20d"] = features["vix_percentile_252d"] / 100.0

    logger.info(f"Built {len(features)} features for {target_date.date()}, "
                f"spot={spot}, vix={vix}")

    return features


def format_indicators_for_display(features: dict) -> list:
    """
    Format feature dict into display-ready indicator list for frontend.
    Returns list of indicator dicts with label, value, formatted_value,
    classification (bullish/bearish/neutral).
    """
    indicators = []
    for key, meta in DISPLAY_FEATURES.items():
        value = features.get(key)
        if value is None:
            continue

        try:
            formatted = meta["format"].format(value)
        except (ValueError, KeyError):
            formatted = str(round(value, 2))

        classification = _classify_indicator(key, value)

        indicators.append({
            "key": key,
            "label": meta["label"],
            "description": meta["description"],
            "value": value,
            "formatted_value": formatted,
            "classification": classification,
            "bullish_when": meta.get("bullish_when", ""),
            "bearish_when": meta.get("bearish_when", ""),
        })

    return indicators
