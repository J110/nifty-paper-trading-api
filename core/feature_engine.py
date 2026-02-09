"""
Build the same feature vector as the backtest model, but from live data.

CRITICAL: Features must EXACTLY match the training features.
The model expects specific column names and scaling.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

# Display features — shown on the Market Signals page with explanations
DISPLAY_FEATURES = {
    "vix": {
        "label": "India VIX",
        "description": "Volatility index — fear gauge",
        "bullish_when": "Below 15",
        "bearish_when": "Above 20",
        "format": "{:.1f}",
    },
    "vix_percentile_20d": {
        "label": "VIX Percentile (20d)",
        "description": "Current VIX relative to last 20 days",
        "bullish_when": "Below 30%",
        "bearish_when": "Above 70%",
        "format": "{:.0%}",
    },
    "nifty_20d_return": {
        "label": "Nifty 20-Day Return",
        "description": "Trailing 20-day price change",
        "bullish_when": "Above +3%",
        "bearish_when": "Below -3%",
        "format": "{:+.1%}",
    },
    "nifty_50d_return": {
        "label": "Nifty 50-Day Return",
        "description": "Trailing 50-day trend direction",
        "bullish_when": "Above +5%",
        "bearish_when": "Below -5%",
        "format": "{:+.1%}",
    },
    "iv_skew": {
        "label": "IV Skew (Put-Call)",
        "description": "Put IV minus Call IV — measures fear premium",
        "bullish_when": "Below 2%",
        "bearish_when": "Above 5%",
        "format": "{:+.1%}",
    },
    "put_call_ratio": {
        "label": "Put/Call OI Ratio",
        "description": "Open interest ratio — sentiment gauge",
        "bullish_when": "Above 1.2 (excess put writing = support)",
        "bearish_when": "Below 0.8 (excess call writing = resistance)",
        "format": "{:.2f}",
    },
    "fii_net_5d": {
        "label": "FII Net Flow (5-Day)",
        "description": "Foreign institutional investor buying/selling",
        "bullish_when": "Positive (buying)",
        "bearish_when": "Negative (selling)",
        "format": "₹{:.0f}Cr",
    },
    "rsi_14": {
        "label": "RSI (14-Day)",
        "description": "Relative strength — overbought/oversold",
        "bullish_when": "Between 40-60 (healthy)",
        "bearish_when": "Above 75 (overbought) or Below 30 (oversold)",
        "format": "{:.1f}",
    },
    "adx_14": {
        "label": "ADX (14-Day)",
        "description": "Trend strength (direction-agnostic)",
        "bullish_when": "Above 25 with positive DI+",
        "bearish_when": "Above 25 with positive DI-",
        "format": "{:.1f}",
    },
    "max_oi_put_strike": {
        "label": "Max OI Put Strike",
        "description": "Highest put open interest — likely support level",
        "format": "{:.0f}",
    },
    "max_oi_call_strike": {
        "label": "Max OI Call Strike",
        "description": "Highest call open interest — likely resistance level",
        "format": "{:.0f}",
    },
}


def _classify_indicator(name: str, value: float) -> str:
    """Classify an indicator as bullish/bearish/neutral for display."""
    rules = {
        "vix": lambda v: "bullish" if v < 15 else ("bearish" if v > 20 else "neutral"),
        "vix_percentile_20d": lambda v: "bullish" if v < 0.3 else (
            "bearish" if v > 0.7 else "neutral"),
        "nifty_20d_return": lambda v: "bullish" if v > 0.03 else (
            "bearish" if v < -0.03 else "neutral"),
        "nifty_50d_return": lambda v: "bullish" if v > 0.05 else (
            "bearish" if v < -0.05 else "neutral"),
        "iv_skew": lambda v: "bullish" if v < 0.02 else (
            "bearish" if v > 0.05 else "neutral"),
        "put_call_ratio": lambda v: "bullish" if v > 1.2 else (
            "bearish" if v < 0.8 else "neutral"),
        "fii_net_5d": lambda v: "bullish" if v > 0 else (
            "bearish" if v < 0 else "neutral"),
        "rsi_14": lambda v: "bearish" if v > 75 else (
            "bearish" if v < 30 else "bullish" if 40 <= v <= 60 else "neutral"),
        "adx_14": lambda v: "neutral",
    }
    classifier = rules.get(name)
    if classifier:
        return classifier(value)
    return "neutral"


async def build_live_features(dhan_client, historical_df: pd.DataFrame,
                               option_chain: Optional[dict] = None) -> dict:
    """
    Build feature vector for today using live data.
    Returns dict with all features matching training format.

    Args:
        dhan_client: DhanClient instance for live data
        historical_df: DataFrame with [date, open, high, low, close, volume]
                       for at least 60 trading days back
        option_chain: Optional pre-fetched option chain data
    """
    import ta as ta_lib

    try:
        spot = await dhan_client.get_nifty_ltp()
        vix = await dhan_client.get_india_vix()
    except Exception as e:
        logger.error(f"Failed to fetch live prices: {e}")
        spot = historical_df["close"].iloc[-1] if len(historical_df) > 0 else 24000
        vix = 14.0

    df = historical_df.copy()

    # Ensure we have enough data
    if len(df) < 60:
        logger.warning(f"Only {len(df)} days of history, need 60+")

    close = df["close"]

    features = {}

    # === Price-based features ===
    features["nifty_close"] = spot
    features["nifty_5d_return"] = (spot / close.iloc[-5] - 1) if len(close) >= 5 else 0
    features["nifty_10d_return"] = (spot / close.iloc[-10] - 1) if len(close) >= 10 else 0
    features["nifty_20d_return"] = (spot / close.iloc[-20] - 1) if len(close) >= 20 else 0
    features["nifty_50d_return"] = (spot / close.iloc[-50] - 1) if len(close) >= 50 else 0

    # Moving averages distance
    if len(close) >= 20:
        sma20 = close.rolling(20).mean().iloc[-1]
        features["dist_sma20"] = (spot - sma20) / sma20
    else:
        features["dist_sma20"] = 0

    if len(close) >= 50:
        sma50 = close.rolling(50).mean().iloc[-1]
        features["dist_sma50"] = (spot - sma50) / sma50
    else:
        features["dist_sma50"] = 0

    # Volatility features
    if len(close) >= 20:
        log_returns = np.log(close / close.shift(1)).dropna()
        features["realized_vol_20d"] = log_returns.tail(20).std() * np.sqrt(252)
        features["realized_vol_10d"] = log_returns.tail(10).std() * np.sqrt(252)
    else:
        features["realized_vol_20d"] = 0.15
        features["realized_vol_10d"] = 0.15

    # === VIX features ===
    features["vix"] = vix
    features["vix_20d_avg"] = vix  # Will be overwritten if we have VIX history
    features["vix_percentile_20d"] = 0.5

    # === Technical indicators ===
    if len(df) >= 14:
        rsi = ta_lib.momentum.RSIIndicator(close, window=14)
        features["rsi_14"] = rsi.rsi().iloc[-1]
    else:
        features["rsi_14"] = 50.0

    if len(df) >= 14:
        adx = ta_lib.trend.ADXIndicator(df["high"], df["low"], close, window=14)
        features["adx_14"] = adx.adx().iloc[-1]
        features["di_plus"] = adx.adx_pos().iloc[-1]
        features["di_minus"] = adx.adx_neg().iloc[-1]
    else:
        features["adx_14"] = 20.0
        features["di_plus"] = 15.0
        features["di_minus"] = 15.0

    if len(df) >= 26:
        macd = ta_lib.trend.MACD(close)
        features["macd_hist"] = macd.macd_diff().iloc[-1]
    else:
        features["macd_hist"] = 0.0

    # Bollinger Bands
    if len(close) >= 20:
        bb = ta_lib.volatility.BollingerBands(close, window=20, window_dev=2)
        bb_upper = bb.bollinger_hband().iloc[-1]
        bb_lower = bb.bollinger_lband().iloc[-1]
        bb_width = (bb_upper - bb_lower) / close.iloc[-1]
        features["bb_width_20d"] = bb_width
        features["bb_position"] = (spot - bb_lower) / (bb_upper - bb_lower) if (
            bb_upper - bb_lower) > 0 else 0.5
    else:
        features["bb_width_20d"] = 0.05
        features["bb_position"] = 0.5

    # === Date/calendar features ===
    today = datetime.now()
    features["day_of_week"] = today.weekday()
    features["day_of_month"] = today.day
    features["month"] = today.month

    # Days to monthly expiry (last Thursday)
    features["days_to_expiry"] = 10  # approximate

    # === Drawdown features (max drawdown so far) ===
    if len(close) >= 20:
        rolling_max = close.rolling(20).max()
        drawdown_20d = (close / rolling_max - 1).iloc[-1]
        features["max_drawdown_20d_sofar"] = drawdown_20d
    else:
        features["max_drawdown_20d_sofar"] = 0

    if len(close) >= 10:
        rolling_max_10 = close.rolling(10).max()
        drawdown_10d = (close / rolling_max_10 - 1).iloc[-1]
        features["max_drawdown_10d_sofar"] = drawdown_10d
    else:
        features["max_drawdown_10d_sofar"] = 0

    # === Option-derived features (from Dhan option chain) ===
    features["iv_skew"] = 0.03  # default
    features["put_call_ratio"] = 1.0  # default
    features["max_oi_put_strike"] = spot * 0.97
    features["max_oi_call_strike"] = spot * 1.03

    if option_chain:
        try:
            _extract_option_features(features, option_chain, spot)
        except Exception as e:
            logger.warning(f"Failed to extract option features: {e}")

    # === Placeholder features (filled from DB if available) ===
    features.setdefault("fii_net_5d", 0)
    features.setdefault("dii_net_5d", 0)
    features.setdefault("deep_otm_oi_ratio", 0)
    features.setdefault("iv_skew_steepness", 0)
    features.setdefault("atm_iv", vix / 100)
    features.setdefault("atm_iv_percentile_252d", 50)

    # Clean NaN/inf
    for k, v in features.items():
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            features[k] = 0.0

    logger.info(f"Built {len(features)} features, spot={spot}, vix={vix}")
    return features


def _extract_option_features(features: dict, option_chain: dict,
                              spot: float):
    """Extract option-derived features from Dhan option chain response."""
    data = option_chain.get("data", [])
    if not data:
        return

    max_put_oi = 0
    max_put_strike = spot * 0.97
    max_call_oi = 0
    max_call_strike = spot * 1.03
    total_put_oi = 0
    total_call_oi = 0
    atm_put_iv = None
    atm_call_iv = None

    for item in data:
        strike = item.get("strikePrice", 0)
        put_oi = item.get("putOI", 0) or 0
        call_oi = item.get("callOI", 0) or 0
        put_iv = item.get("putIV", 0) or 0
        call_iv = item.get("callIV", 0) or 0

        total_put_oi += put_oi
        total_call_oi += call_oi

        if put_oi > max_put_oi:
            max_put_oi = put_oi
            max_put_strike = strike

        if call_oi > max_call_oi:
            max_call_oi = call_oi
            max_call_strike = strike

        # Find ATM strike
        if abs(strike - spot) < 50:
            atm_put_iv = put_iv
            atm_call_iv = call_iv

    features["max_oi_put_strike"] = max_put_strike
    features["max_oi_call_strike"] = max_call_strike
    features["put_call_ratio"] = (
        total_put_oi / total_call_oi if total_call_oi > 0 else 1.0
    )

    if atm_put_iv and atm_call_iv:
        features["iv_skew"] = (atm_put_iv - atm_call_iv) / 100.0
        features["atm_iv"] = (atm_put_iv + atm_call_iv) / 200.0


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
