"""
Generate human-readable prediction reasoning from model features.

Takes the model's feature importances + current feature values and produces
a ranked list of top reasons explaining why the model predicted a specific
drawdown. Displayed on the dashboard's "Prediction Reasoning" card.
"""

import calendar
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Month names for display
_MONTH_NAMES = {i: calendar.month_name[i] for i in range(1, 13)}

# Historically strong / weak months for Nifty
_BULLISH_MONTHS = {1, 4, 10, 11, 12}
_BEARISH_MONTHS = {2, 3, 5, 6}

# ---------------------------------------------------------------------------
# Feature metadata registry — one entry per training feature
# ---------------------------------------------------------------------------
# Each entry:
#   label:    Human-readable name
#   fmt:      Python format string for value display
#   classify: callable(value) -> "bullish" | "bearish" | "neutral"
#   reason:   callable(value, direction) -> short explanation string
# ---------------------------------------------------------------------------

FEATURE_META = {
    # ---- Nifty momentum / trend ----
    "nifty_return_5d": {
        "label": "Nifty 5-Day Return",
        "fmt": "{:+.1%}",
        "classify": lambda v: "bullish" if v > 0.01 else ("bearish" if v < -0.01 else "neutral"),
        "reason": lambda v, d: f"5-day momentum {'positive' if v > 0 else 'negative'} at {v*100:+.1f}%",
    },
    "nifty_return_10d": {
        "label": "Nifty 10-Day Return",
        "fmt": "{:+.1%}",
        "classify": lambda v: "bullish" if v > 0.02 else ("bearish" if v < -0.02 else "neutral"),
        "reason": lambda v, d: f"10-day trend {'up' if v > 0 else 'down'} {abs(v)*100:.1f}%",
    },
    "nifty_return_20d": {
        "label": "Nifty 20-Day Return",
        "fmt": "{:+.1%}",
        "classify": lambda v: "bullish" if v > 0.03 else ("bearish" if v < -0.03 else "neutral"),
        "reason": lambda v, d: f"Monthly trend {'positive' if v > 0 else 'negative'}",
    },
    "nifty_distance_50dma": {
        "label": "Distance to 50-DMA",
        "fmt": "{:+.1f}%",
        "classify": lambda v: "bullish" if v > 2 else ("bearish" if v < -2 else "neutral"),
        "reason": lambda v, d: f"{'Above' if v > 0 else 'Below'} 50-day moving average",
    },
    "nifty_distance_200dma": {
        "label": "Distance to 200-DMA",
        "fmt": "{:+.1f}%",
        "classify": lambda v: "bullish" if v > 3 else ("bearish" if v < -3 else "neutral"),
        "reason": lambda v, d: f"{'Above' if v > 0 else 'Below'} 200-day moving average",
    },
    "golden_cross": {
        "label": "Golden Cross",
        "fmt": "{}",
        "classify": lambda v: "bullish" if v >= 1 else "bearish",
        "reason": lambda v, d: "50-DMA above 200-DMA (bullish structure)" if v >= 1 else "50-DMA below 200-DMA (death cross)",
        "_format_override": lambda v: "Yes" if v >= 1 else "No",
    },
    "rsi_14": {
        "label": "RSI (14-Day)",
        "fmt": "{:.1f}",
        "classify": lambda v: "bearish" if v > 75 or v < 30 else ("bullish" if 40 <= v <= 60 else "neutral"),
        "reason": lambda v, d: "Overbought territory" if v > 75 else ("Oversold territory" if v < 30 else "Healthy range"),
    },
    "higher_highs_5w": {
        "label": "Higher Highs (5W)",
        "fmt": "{:.0f}",
        "classify": lambda v: "bullish" if v >= 1 else "neutral",
        "reason": lambda v, d: f"{int(v)} of 5 weeks made higher highs",
    },

    # ---- India VIX / Volatility ----
    "india_vix": {
        "label": "India VIX",
        "fmt": "{:.1f}",
        "classify": lambda v: "bullish" if v < 15 else ("bearish" if v > 20 else "neutral"),
        "reason": lambda v, d: f"Fear gauge at {v:.1f} ({'low — calm market' if v < 15 else 'elevated — caution' if v > 20 else 'moderate'})",
    },
    "vix_change_5d": {
        "label": "VIX 5-Day Change",
        "fmt": "{:+.2f}",
        "classify": lambda v: "bullish" if v < -1 else ("bearish" if v > 2 else "neutral"),
        "reason": lambda v, d: f"VIX {'spiked' if v > 2 else 'dropped' if v < -1 else 'stable'} over 5 days",
    },
    "vix_percentile_252d": {
        "label": "VIX Percentile (1Y)",
        "fmt": "{:.0f}th",
        "classify": lambda v: "bullish" if v < 30 else ("bearish" if v > 70 else "neutral"),
        "reason": lambda v, d: f"VIX at {v:.0f}th percentile vs last year",
    },
    "realized_vol_20d": {
        "label": "Realized Volatility (20d)",
        "fmt": "{:.1f}%",
        "classify": lambda v: "bullish" if v < 12 else ("bearish" if v > 20 else "neutral"),
        "reason": lambda v, d: f"Actual market swings {'low' if v < 12 else 'high' if v > 20 else 'moderate'}",
    },
    "variance_risk_premium": {
        "label": "Variance Risk Premium",
        "fmt": "{:+.1f}",
        "classify": lambda v: "bullish" if v > 3 else ("bearish" if v < 0 else "neutral"),
        "reason": lambda v, d: "Implied vol exceeds realized (options expensive — good for sellers)" if v > 3 else ("Realized vol catching up — tail risk" if v < 0 else "Normal vol spread"),
    },
    "vix_rising_streak": {
        "label": "VIX Rising Streak",
        "fmt": "{:.0f} days",
        "classify": lambda v: "neutral" if v == 0 else ("bearish" if v > 3 else "neutral"),
        "reason": lambda v, d: f"VIX rising for {int(v)} consecutive days" if v > 0 else "No VIX uptrend",
    },
    "pcr_proxy": {
        "label": "PCR Proxy",
        "fmt": "{:.2f}",
        "classify": lambda v: "neutral",
        "reason": lambda v, d: f"Put-call proxy at {v:.2f}",
    },

    # ---- US / Global ----
    "sp500_return_5d": {
        "label": "S&P 500 (5-Day)",
        "fmt": "{:+.1%}",
        "classify": lambda v: "bullish" if v > 0 else ("bearish" if v < -0.02 else "neutral"),
        "reason": lambda v, d: f"US market {'up' if v > 0 else 'down'} {abs(v)*100:.1f}% this week",
    },
    "us_vix": {
        "label": "US VIX (CBOE)",
        "fmt": "{:.1f}",
        "classify": lambda v: "bullish" if v < 15 else ("bearish" if v > 25 else "neutral"),
        "reason": lambda v, d: f"Global fear gauge at {v:.1f} ({'calm' if v < 15 else 'elevated' if v > 25 else 'moderate'})",
    },
    "us_vix_change_5d": {
        "label": "US VIX 5-Day Change",
        "fmt": "{:+.2f}",
        "classify": lambda v: "bullish" if v < -1 else ("bearish" if v > 2 else "neutral"),
        "reason": lambda v, d: f"US VIX {'spiked' if v > 2 else 'dropped' if v < -1 else 'stable'} over 5 days",
    },

    # ---- Macro ----
    "dxy_level": {
        "label": "US Dollar Index (DXY)",
        "fmt": "{:.1f}",
        "classify": lambda v: "bullish" if v < 100 else ("bearish" if v > 106 else "neutral"),
        "reason": lambda v, d: f"Dollar at {v:.1f} ({'strong — pressures EM' if v > 106 else 'weak — supports EM' if v < 100 else 'moderate'})",
    },
    "dxy_change_5d": {
        "label": "DXY 5-Day Change",
        "fmt": "{:+.2f}",
        "classify": lambda v: "bullish" if v < -0.5 else ("bearish" if v > 0.5 else "neutral"),
        "reason": lambda v, d: f"Dollar {'strengthening' if v > 0.5 else 'weakening' if v < -0.5 else 'stable'}",
    },
    "us10y_level": {
        "label": "US 10Y Yield",
        "fmt": "{:.2f}%",
        "classify": lambda v: "bullish" if v < 3.5 else ("bearish" if v > 4.5 else "neutral"),
        "reason": lambda v, d: f"US 10Y at {v:.2f}% ({'high — risk-off pressure' if v > 4.5 else 'low — supports risk' if v < 3.5 else 'moderate'})",
    },
    "us10y_change_5d": {
        "label": "US 10Y Change (5d)",
        "fmt": "{:+.3f}",
        "classify": lambda v: "bullish" if v < -0.1 else ("bearish" if v > 0.1 else "neutral"),
        "reason": lambda v, d: f"Yields {'rising' if v > 0.1 else 'falling' if v < -0.1 else 'stable'} short-term",
    },
    "us10y_change_20d": {
        "label": "US 10Y Change (20d)",
        "fmt": "{:+.3f}",
        "classify": lambda v: "bullish" if v < -0.2 else ("bearish" if v > 0.2 else "neutral"),
        "reason": lambda v, d: f"Yields {'rising' if v > 0.2 else 'falling' if v < -0.2 else 'stable'} over month",
    },

    # ---- Calendar ----
    "day_of_month": {
        "label": "Day of Month",
        "fmt": "{:.0f}",
        "classify": lambda v: "neutral",
        "reason": lambda v, d: f"Day {int(v)} of month",
    },
    "month": {
        "label": "Month",
        "fmt": "{}",
        "classify": lambda v: "bullish" if int(v) in _BULLISH_MONTHS else ("bearish" if int(v) in _BEARISH_MONTHS else "neutral"),
        "reason": lambda v, d: f"{_MONTH_NAMES.get(int(v), '?')} — {'historically strong' if int(v) in _BULLISH_MONTHS else 'historically weaker' if int(v) in _BEARISH_MONTHS else 'neutral seasonality'}",
        "_format_override": lambda v: _MONTH_NAMES.get(int(v), str(v)),
    },
    "is_expiry_week": {
        "label": "Expiry Week",
        "fmt": "{}",
        "classify": lambda v: "bearish" if v >= 1 else "neutral",
        "reason": lambda v, d: "Options expiry week — higher gamma risk" if v >= 1 else "Not expiry week",
        "_format_override": lambda v: "Yes" if v >= 1 else "No",
    },
    "days_to_monthly_expiry": {
        "label": "Days to Expiry",
        "fmt": "{:.0f}",
        "classify": lambda v: "bearish" if v < 5 else ("bullish" if v > 15 else "neutral"),
        "reason": lambda v, d: f"{int(v)} trading days to expiry" + (" — near-expiry risk" if v < 5 else ""),
    },
    "is_volatile_month": {
        "label": "Volatile Month",
        "fmt": "{}",
        "classify": lambda v: "bearish" if v >= 1 else "neutral",
        "reason": lambda v, d: "Sep/Oct — historically volatile period" if v >= 1 else "Not in volatile period",
        "_format_override": lambda v: "Yes" if v >= 1 else "No",
    },

    # ---- Dhan Options-Derived ----
    "deep_otm_oi_ratio": {
        "label": "Deep OTM OI Ratio",
        "fmt": "{:.2f}",
        "classify": lambda v: "bullish" if v < 1.0 else ("bearish" if v > 2.0 else "neutral"),
        "reason": lambda v, d: f"Deep OTM put OI {'concentrated' if v > 2 else 'light' if v < 1 else 'normal'}",
    },
    "deep_otm_oi_ratio_change_5d": {
        "label": "Deep OTM OI Change (5d)",
        "fmt": "{:+.2f}",
        "classify": lambda v: "bullish" if v < -0.1 else ("bearish" if v > 0.2 else "neutral"),
        "reason": lambda v, d: f"Deep OTM put OI {'building' if v > 0.2 else 'declining' if v < -0.1 else 'stable'}",
    },
    "put_oi_buildup_ratio": {
        "label": "Put OI Buildup",
        "fmt": "{:.2f}",
        "classify": lambda v: "bullish" if v < 0.8 else ("bearish" if v > 1.5 else "neutral"),
        "reason": lambda v, d: f"Put OI buildup {'heavy' if v > 1.5 else 'light' if v < 0.8 else 'normal'}",
    },
    "put_volume_surge_ratio": {
        "label": "Put Volume Surge",
        "fmt": "{:.2f}x",
        "classify": lambda v: "bullish" if v < 1.0 else ("bearish" if v > 2.0 else "neutral"),
        "reason": lambda v, d: f"Put volume {'surging {v:.1f}x' if v > 2 else 'below normal' if v < 1 else 'normal'}",
    },
    "iv_skew_steepness": {
        "label": "IV Skew Steepness",
        "fmt": "{:.2f}",
        "classify": lambda v: "bullish" if v < 0.02 else ("bearish" if v > 0.05 else "neutral"),
        "reason": lambda v, d: f"Put skew {'steep — tail risk priced in' if v > 0.05 else 'flat — calm options market' if v < 0.02 else 'moderate'}",
    },
    "iv_skew_change_5d": {
        "label": "IV Skew Change (5d)",
        "fmt": "{:+.2f}",
        "classify": lambda v: "bullish" if v < -0.5 else ("bearish" if v > 0.5 else "neutral"),
        "reason": lambda v, d: f"Skew {'steepening' if v > 0.5 else 'flattening' if v < -0.5 else 'stable'}",
    },
    "atm_iv": {
        "label": "ATM Implied Volatility",
        "fmt": "{:.1f}%",
        "classify": lambda v: "bullish" if v < 12 else ("bearish" if v > 20 else "neutral"),
        "reason": lambda v, d: f"ATM IV at {v:.1f}% ({'low' if v < 12 else 'high' if v > 20 else 'moderate'})",
    },
    "atm_iv_percentile_252d": {
        "label": "ATM IV Percentile (1Y)",
        "fmt": "{:.0f}th",
        "classify": lambda v: "bullish" if v < 30 else ("bearish" if v > 70 else "neutral"),
        "reason": lambda v, d: f"ATM IV at {v:.0f}th percentile vs last year",
    },
    "atm_put_intraday_range": {
        "label": "ATM Put Intraday Range",
        "fmt": "{:.2f}",
        "classify": lambda v: "bullish" if v < 0.5 else ("bearish" if v > 1.5 else "neutral"),
        "reason": lambda v, d: f"ATM put range {'wide — high activity' if v > 1.5 else 'narrow — quiet' if v < 0.5 else 'normal'}",
    },
}


def _format_value(feature_name: str, value) -> str:
    """Format a feature value for display, handling special cases."""
    if value is None:
        return "N/A"

    meta = FEATURE_META.get(feature_name)
    if not meta:
        return str(round(value, 2))

    # Use override formatter if provided (month names, yes/no, etc.)
    override = meta.get("_format_override")
    if override:
        return override(value)

    try:
        return meta["fmt"].format(value)
    except (ValueError, TypeError):
        return str(round(value, 2))


def _generate_summary(
    reasons: list,
    predicted_drawdown: Optional[float],
) -> str:
    """
    Generate a plain-English summary explaining the prediction.

    The model's prediction is driven by feature importances — a feature
    classified as "neutral" can still push the prediction negative if the
    model learned that value range correlates with drawdowns. This summary
    explains that nuance in laymen's terms.
    """
    if not reasons:
        return ""

    pct = abs(predicted_drawdown * 100) if predicted_drawdown else 0

    # Weighted direction: sum importance for each direction
    weighted = {"bullish": 0.0, "bearish": 0.0, "neutral": 0.0}
    for r in reasons:
        weighted[r["direction"]] += r["importance_pct"]

    # Identify the dominant driver(s)
    top = reasons[0]  # most important feature
    top2 = reasons[1] if len(reasons) > 1 else None

    # Build the narrative
    parts = []

    # Overall assessment
    if pct >= 6.5:
        parts.append(
            f"The model predicts a significant -{pct:.1f}% potential drawdown."
        )
    elif pct >= 5.0:
        parts.append(
            f"The model predicts a moderate -{pct:.1f}% potential drawdown."
        )
    elif pct >= 3.8:
        parts.append(
            f"The model predicts a mild -{pct:.1f}% potential drawdown."
        )
    else:
        parts.append(
            f"The model predicts a low -{pct:.1f}% potential drawdown."
        )

    # Explain the top driver
    parts.append(
        f"The biggest driver is {top['label']} ({top['formatted_value']}), "
        f"which carries {top['importance_pct']:.0f}% of the model's weight."
    )

    # Explain why neutral features still matter
    if weighted["neutral"] > weighted["bearish"] and weighted["neutral"] > weighted["bullish"]:
        neutral_reasons = [r for r in reasons if r["direction"] == "neutral"]
        neutral_weight = sum(r["importance_pct"] for r in neutral_reasons)
        parts.append(
            f"Most top features ({len(neutral_reasons)} of {len(reasons)}) are in neutral ranges, "
            f"but they still account for {neutral_weight:.0f}% of the model's weight. "
            f"The model learned that these 'in-between' levels historically precede moderate drawdowns."
        )

    # Mention bullish factors as counterbalance
    bullish_reasons = [r for r in reasons if r["direction"] == "bullish"]
    if bullish_reasons:
        bullish_labels = [r["label"] for r in bullish_reasons[:2]]
        bullish_weight = sum(r["importance_pct"] for r in bullish_reasons)
        parts.append(
            f"{' and '.join(bullish_labels)} {'are' if len(bullish_labels) > 1 else 'is'} "
            f"providing a bullish offset ({bullish_weight:.0f}% weight), "
            f"which is why the prediction isn't even more negative."
        )

    # Mention bearish factors
    bearish_reasons = [r for r in reasons if r["direction"] == "bearish"]
    if bearish_reasons:
        bearish_labels = [r["label"] for r in bearish_reasons[:2]]
        parts.append(
            f"{' and '.join(bearish_labels)} {'are' if len(bearish_labels) > 1 else 'is'} "
            f"actively bearish, adding downside pressure."
        )

    return " ".join(parts)


def generate_prediction_reasons(
    features: dict,
    feature_importances: dict,
    top_n: int = 7,
    predicted_drawdown: Optional[float] = None,
) -> dict:
    """
    Generate ranked list of reasons explaining the prediction, plus a
    plain-English summary narrative.

    Args:
        features:            Dict of 37 feature values (from Prediction.features)
        feature_importances: Dict of {feature_name: importance_weight} from model
        top_n:               Number of top reasons to return
        predicted_drawdown:  The model's predicted drawdown (decimal, e.g. -0.0777)

    Returns:
        Dict with:
          reasons: list of dicts (label, formatted_value, direction, importance_pct, reason)
          summary: plain-English narrative string
    """
    if not features or not feature_importances:
        return {"reasons": [], "summary": ""}

    # Total importance for percentage calculation
    total_importance = sum(feature_importances.values()) or 1.0

    reasons = []
    for feat_name, importance in feature_importances.items():
        value = features.get(feat_name)
        if value is None:
            continue

        meta = FEATURE_META.get(feat_name)
        if not meta:
            continue

        # Skip features with near-zero values that are likely missing/placeholder
        # (Dhan features default to 0 when unavailable)
        importance_pct = (importance / total_importance) * 100
        if importance_pct < 0.5:
            continue  # Skip negligible features

        direction = meta["classify"](value)
        formatted = _format_value(feat_name, value)
        reason_text = meta["reason"](value, direction)

        reasons.append({
            "feature": feat_name,
            "label": meta["label"],
            "value": round(float(value), 4) if value is not None else 0,
            "formatted_value": formatted,
            "direction": direction,
            "importance_pct": round(importance_pct, 1),
            "reason": reason_text,
        })

    # Sort by importance (descending) and take top N
    reasons.sort(key=lambda r: r["importance_pct"], reverse=True)
    top_reasons = reasons[:top_n]

    summary = _generate_summary(top_reasons, predicted_drawdown)

    return {"reasons": top_reasons, "summary": summary}
