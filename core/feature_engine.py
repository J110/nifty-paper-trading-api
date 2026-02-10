"""
Build the same feature vector as the backtest model, but from live data.

CRITICAL: Features must EXACTLY match the 37 training features.
The model expects specific column names in specific order.
Feature names are loaded from ml/feature_names.pkl.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, date
from typing import Optional
from core.timezone import now_ist

logger = logging.getLogger(__name__)

# The 37 training feature names — MUST match feature_names.json / feature_names.pkl
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

# Display features — shown on the Market Signals page with explanations
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


def _get_last_thursday(year: int, month: int) -> date:
    """Get the last Thursday of a given month (monthly expiry)."""
    if month == 12:
        next_month_first = date(year + 1, 1, 1)
    else:
        next_month_first = date(year, month + 1, 1)
    last_day = next_month_first - timedelta(days=1)
    offset = (last_day.weekday() - 3) % 7
    return last_day - timedelta(days=offset)


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
    Build feature vector for today using live + historical data.
    Returns dict with all 37 training features matching model input.

    Args:
        dhan_client: DhanClient instance for live data
        historical_df: DataFrame with [date, open, high, low, close, volume]
                       for at least 250 trading days back
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
    if len(df) < 60:
        logger.warning(f"Only {len(df)} days of history, need 250+")

    close = df["close"]
    high = df.get("high", close)

    features = {}

    # ── 1. Trend & Momentum ────────────────────────────────────────
    features["nifty_return_5d"] = (spot / float(close.iloc[-5]) - 1) if len(close) >= 5 else 0
    features["nifty_return_10d"] = (spot / float(close.iloc[-10]) - 1) if len(close) >= 10 else 0
    features["nifty_return_20d"] = (spot / float(close.iloc[-20]) - 1) if len(close) >= 20 else 0

    if len(close) >= 50:
        sma50 = float(close.rolling(50).mean().iloc[-1])
        features["nifty_distance_50dma"] = (spot - sma50) / sma50 * 100
    else:
        features["nifty_distance_50dma"] = 0

    if len(close) >= 200:
        sma200 = float(close.rolling(200).mean().iloc[-1])
        features["nifty_distance_200dma"] = (spot - sma200) / sma200 * 100
        sma50_val = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else 0
        features["golden_cross"] = 1 if sma50_val > sma200 else 0
    else:
        features["nifty_distance_200dma"] = 0
        features["golden_cross"] = 0

    if len(close) >= 14:
        rsi = ta_lib.momentum.RSIIndicator(close, window=14)
        rsi_val = rsi.rsi().iloc[-1]
        features["rsi_14"] = float(rsi_val) if not pd.isna(rsi_val) else 50.0
    else:
        features["rsi_14"] = 50.0

    # Higher highs over 5 weeks (simplified)
    if len(high) >= 35:
        weekly_highs = []
        for w in range(5):
            start_idx = -(w + 1) * 5
            end_idx = -w * 5 if w > 0 else None
            segment = high.iloc[start_idx:end_idx]
            weekly_highs.append(float(segment.max()) if len(segment) > 0 else 0)
        weekly_highs.reverse()
        hh_count = sum(1 for i in range(1, len(weekly_highs)) if weekly_highs[i] > weekly_highs[i - 1])
        features["higher_highs_5w"] = hh_count
    else:
        features["higher_highs_5w"] = 0

    # ── 2. Volatility ──────────────────────────────────────────────
    features["india_vix"] = float(vix) if vix else 14.0

    # VIX change 5d (approximate from current vix)
    features["vix_change_5d"] = 0.0  # no VIX history in live, default to 0

    # VIX percentile 252d
    features["vix_percentile_252d"] = 50.0  # default, override if VIX history available

    # Realized vol
    if len(close) >= 20:
        log_returns = np.log(close / close.shift(1)).dropna()
        features["realized_vol_20d"] = float(log_returns.tail(20).std() * np.sqrt(252) * 100)
    else:
        features["realized_vol_20d"] = 15.0

    features["variance_risk_premium"] = features["india_vix"] - features["realized_vol_20d"]

    features["vix_rising_streak"] = 0  # no VIX history for streak

    # ── 3. Sentiment ───────────────────────────────────────────────
    ret_5d = features.get("nifty_return_5d", 0)
    features["pcr_proxy"] = features["vix_change_5d"] / (ret_5d * 100 + 0.001)

    # ── 4. Global context (try yfinance for latest) ────────────────
    try:
        import yfinance as yf
        end_dt = now_ist()
        start_dt = end_dt - timedelta(days=30)

        sp500 = yf.download("^GSPC", start=start_dt.strftime("%Y-%m-%d"),
                             end=end_dt.strftime("%Y-%m-%d"), progress=False)
        if isinstance(sp500.columns, pd.MultiIndex):
            sp500.columns = sp500.columns.get_level_values(0)
        if len(sp500) >= 5:
            features["sp500_return_5d"] = float(sp500["Close"].pct_change(5).iloc[-1])
        else:
            features["sp500_return_5d"] = 0.0

        us_vix_data = yf.download("^VIX", start=start_dt.strftime("%Y-%m-%d"),
                                    end=end_dt.strftime("%Y-%m-%d"), progress=False)
        if isinstance(us_vix_data.columns, pd.MultiIndex):
            us_vix_data.columns = us_vix_data.columns.get_level_values(0)
        if len(us_vix_data) >= 1:
            features["us_vix"] = float(us_vix_data["Close"].iloc[-1])
            if len(us_vix_data) >= 5:
                features["us_vix_change_5d"] = float(
                    us_vix_data["Close"].iloc[-1] - us_vix_data["Close"].iloc[-5]
                )
            else:
                features["us_vix_change_5d"] = 0.0
        else:
            features["us_vix"] = 0.0
            features["us_vix_change_5d"] = 0.0

        dxy = yf.download("DX-Y.NYB", start=start_dt.strftime("%Y-%m-%d"),
                            end=end_dt.strftime("%Y-%m-%d"), progress=False)
        if isinstance(dxy.columns, pd.MultiIndex):
            dxy.columns = dxy.columns.get_level_values(0)
        if len(dxy) >= 1:
            features["dxy_level"] = float(dxy["Close"].iloc[-1])
            if len(dxy) >= 5:
                features["dxy_change_5d"] = float(dxy["Close"].iloc[-1] - dxy["Close"].iloc[-5])
            else:
                features["dxy_change_5d"] = 0.0
        else:
            features["dxy_level"] = 0.0
            features["dxy_change_5d"] = 0.0

        us10y = yf.download("^TNX", start=start_dt.strftime("%Y-%m-%d"),
                              end=end_dt.strftime("%Y-%m-%d"), progress=False)
        if isinstance(us10y.columns, pd.MultiIndex):
            us10y.columns = us10y.columns.get_level_values(0)
        if len(us10y) >= 1:
            features["us10y_level"] = float(us10y["Close"].iloc[-1])
            if len(us10y) >= 5:
                features["us10y_change_5d"] = float(us10y["Close"].iloc[-1] - us10y["Close"].iloc[-5])
            else:
                features["us10y_change_5d"] = 0.0
            if len(us10y) >= 20:
                features["us10y_change_20d"] = float(us10y["Close"].iloc[-1] - us10y["Close"].iloc[-20])
            else:
                features["us10y_change_20d"] = 0.0
        else:
            features["us10y_level"] = 0.0
            features["us10y_change_5d"] = 0.0
            features["us10y_change_20d"] = 0.0

    except Exception as e:
        logger.warning(f"Failed to get global market data: {e}")
        features.setdefault("sp500_return_5d", 0.0)
        features.setdefault("us_vix", 0.0)
        features.setdefault("us_vix_change_5d", 0.0)
        features.setdefault("dxy_level", 0.0)
        features.setdefault("dxy_change_5d", 0.0)
        features.setdefault("us10y_level", 0.0)
        features.setdefault("us10y_change_5d", 0.0)
        features.setdefault("us10y_change_20d", 0.0)

    # ── 5. Calendar ────────────────────────────────────────────────
    today = now_ist()
    features["day_of_month"] = today.day
    features["month"] = today.month
    features["is_volatile_month"] = 1 if today.month in [9, 10] else 0

    # Days to monthly expiry
    exp = _get_last_thursday(today.year, today.month)
    if exp < today.date():
        # Expiry already passed this month, use next month
        nm = today.month + 1
        ny = today.year
        if nm > 12:
            nm = 1
            ny += 1
        exp = _get_last_thursday(ny, nm)
    dte = (exp - today.date()).days
    features["days_to_monthly_expiry"] = max(dte, 1)
    features["is_expiry_week"] = 1 if dte <= 5 else 0

    # ── 6. Options-derived (from Dhan option chain or defaults) ────
    features["deep_otm_oi_ratio"] = 0.0
    features["deep_otm_oi_ratio_change_5d"] = 0.0
    features["put_oi_buildup_ratio"] = 0.0
    features["put_volume_surge_ratio"] = 1.0
    features["iv_skew_steepness"] = 0.0
    features["iv_skew_change_5d"] = 0.0
    features["atm_iv"] = (vix / 100.0) if vix else 0.14
    features["atm_iv_percentile_252d"] = features["vix_percentile_252d"]
    features["atm_put_intraday_range"] = 0.0

    if option_chain:
        try:
            _extract_option_features(features, option_chain, spot)
        except Exception as e:
            logger.warning(f"Failed to extract option features: {e}")

    # ── Clean NaN/inf ──────────────────────────────────────────────
    for k, v in features.items():
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            features[k] = 0.0

    logger.info(f"Built {len(features)} features, spot={spot}, vix={vix}")

    # Also store display-friendly aliases for the frontend
    features["vix"] = features["india_vix"]
    features["nifty_close"] = spot
    features["vix_20d_avg"] = features["india_vix"]
    features["vix_percentile_20d"] = features["vix_percentile_252d"] / 100.0

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
    deep_otm_oi = 0
    near_otm_oi = 0

    for item in data:
        strike = item.get("strikePrice", 0)
        put_oi = item.get("putOI", 0) or 0
        call_oi = item.get("callOI", 0) or 0
        put_iv = item.get("putIV", 0) or 0
        call_iv = item.get("callIV", 0) or 0

        total_put_oi += put_oi
        total_call_oi += call_oi

        otm_pct = (spot - strike) / spot if spot > 0 else 0

        # Deep OTM (7-10% below spot) vs Near OTM (0-3% below)
        if 0.07 <= otm_pct <= 0.10:
            deep_otm_oi += put_oi
        elif 0 <= otm_pct <= 0.03:
            near_otm_oi += put_oi

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

    # Deep OTM OI ratio
    if near_otm_oi > 0:
        features["deep_otm_oi_ratio"] = deep_otm_oi / near_otm_oi

    # PCR proxy (already computed from VIX, but we can refine here)
    if total_call_oi > 0:
        features["put_call_ratio_raw"] = total_put_oi / total_call_oi

    # IV skew
    if atm_put_iv and atm_call_iv:
        features["iv_skew_steepness"] = (atm_put_iv - atm_call_iv) / 100.0
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
