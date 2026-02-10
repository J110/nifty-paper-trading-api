"""
Maps model prediction to trade signal for each version.
Replicates signal mapping from the backtest code exactly.
"""

import logging

logger = logging.getLogger(__name__)

# Thresholds (as decimals: -0.015 = -1.5%)
THRESH_BULL_FULL = -0.015
THRESH_BULL_HALF = -0.025
THRESH_IC = -0.035


def map_signal_sharp(pred: float) -> dict:
    """
    v5.4.2: Sharp threshold signal mapping.
    No transition zones — hard cutoffs at each boundary.
    """
    pred_pct = pred * 100  # e.g., -0.012 → -1.2

    if pred_pct > -1.5:
        return {
            "signal": "bull_full",
            "size_mult": 1.0,
            "trade_type": "bull_put",
        }
    elif pred_pct > -2.5:
        return {
            "signal": "bull_half",
            "size_mult": 1.0,
            "trade_type": "bull_put",
        }
    elif pred_pct > -3.5:
        return {
            "signal": "iron_condor",
            "size_mult": 1.0,
            "trade_type": "iron_condor",
        }
    else:
        return {
            "signal": "no_trade",
            "size_mult": 0.0,
            "trade_type": None,
        }


def map_signal_graduated(pred: float, floor: float = 0.50,
                          hw: float = 0.50) -> dict:
    """
    v5.4.3: Graduated signal mapping with configurable floor and half-width.
    Smooth transitions between zones instead of hard cutoffs.

    floor: minimum size multiplier (0.5 = always at least 50% position)
    hw: transition half-width in percentage points (0.5 = ±0.5% transition zone)
    """
    pred_pct = pred * 100  # -0.012 → -1.2

    # Thresholds in % terms
    t1 = -1.5  # bull_full / bull_half boundary
    t2 = -2.5  # bull_half / iron_condor boundary
    t3 = -3.5  # iron_condor / no_trade boundary

    if pred_pct > t1 + hw:
        # Fully in bull_full zone
        return {
            "signal": "bull_full",
            "size_mult": 1.0,
            "trade_type": "bull_put",
        }
    elif pred_pct > t1 - hw:
        # Transition: bull_full → bull_half
        # Linear interpolation from 1.0 to floor
        progress = (t1 + hw - pred_pct) / (2 * hw)
        mult = 1.0 - progress * (1.0 - floor)
        return {
            "signal": "bull_full",
            "size_mult": round(mult, 3),
            "trade_type": "bull_put",
        }
    elif pred_pct > t2 + hw:
        # Fully in bull_half zone
        return {
            "signal": "bull_half",
            "size_mult": floor,
            "trade_type": "bull_put",
        }
    elif pred_pct > t2 - hw:
        # Transition: bull_half → iron_condor
        progress = (t2 + hw - pred_pct) / (2 * hw)
        mult = floor - progress * (floor - floor)
        return {
            "signal": "iron_condor",
            "size_mult": round(max(mult, floor), 3),
            "trade_type": "iron_condor",
        }
    elif pred_pct > t3 + hw:
        # Fully in iron_condor zone
        return {
            "signal": "iron_condor",
            "size_mult": floor,
            "trade_type": "iron_condor",
        }
    elif pred_pct > t3 - hw:
        # Transition: iron_condor → no_trade
        progress = (t3 + hw - pred_pct) / (2 * hw)
        mult = floor * (1.0 - progress)
        return {
            "signal": "iron_condor" if mult > 0.1 else "no_trade",
            "size_mult": round(max(mult, 0.0), 3),
            "trade_type": "iron_condor" if mult > 0.1 else None,
        }
    else:
        return {
            "signal": "no_trade",
            "size_mult": 0.0,
            "trade_type": None,
        }


def map_signal(pred: float, version_cfg: dict) -> dict:
    """
    Map prediction to signal using version-specific configuration.
    """
    mapping_type = version_cfg.get("SIGNAL_MAPPING", "sharp")

    if mapping_type == "sharp":
        return map_signal_sharp(pred)
    elif mapping_type == "graduated":
        floor = version_cfg.get("GRADUATED_FLOOR", 0.50)
        hw = version_cfg.get("GRADUATED_HW", 0.50)
        return map_signal_graduated(pred, floor=floor, hw=hw)
    elif mapping_type == "graduated_gentle":
        floor = version_cfg.get("GRADUATED_FLOOR", 0.80)
        hw = version_cfg.get("GRADUATED_HW", 0.25)
        return map_signal_graduated(pred, floor=floor, hw=hw)
    else:
        logger.warning(f"Unknown mapping type: {mapping_type}, using sharp")
        return map_signal_sharp(pred)


def get_classification_breakdown(pred: float) -> dict:
    """
    For the Market Signals page: show how far the prediction is
    from each classification boundary.
    Returns confidence scores for each zone.
    """
    pred_pct = pred * 100  # e.g., -1.23

    zones = [
        {
            "name": "Strong Bull",
            "range": "0.0% to -0.5%",
            "color": "#00E676",
            "active": pred_pct > -0.5,
            "min": -0.5,
            "max": 0.0,
        },
        {
            "name": "Moderate Bull",
            "range": "-0.5% to -1.0%",
            "color": "#66BB6A",
            "active": -1.0 < pred_pct <= -0.5,
            "min": -1.0,
            "max": -0.5,
        },
        {
            "name": "Bull (Full Position)",
            "range": "-1.0% to -1.5%",
            "color": "#A5D6A7",
            "active": -1.5 < pred_pct <= -1.0,
            "min": -1.5,
            "max": -1.0,
        },
        {
            "name": "Bull (Half Position)",
            "range": "-1.5% to -2.5%",
            "color": "#FFD54F",
            "active": -2.5 < pred_pct <= -1.5,
            "min": -2.5,
            "max": -1.5,
        },
        {
            "name": "Iron Condor",
            "range": "-2.5% to -3.5%",
            "color": "#FF9800",
            "active": -3.5 < pred_pct <= -2.5,
            "min": -3.5,
            "max": -2.5,
        },
        {
            "name": "No Trade (Bear)",
            "range": "-3.5% to -8.0%",
            "color": "#EF5350",
            "active": pred_pct <= -3.5,
            "min": -8.0,
            "max": -3.5,
        },
    ]

    # Find active zone and compute distances
    current_zone = "Unknown"
    for z in zones:
        if z["active"]:
            current_zone = z["name"]
            # Distance to nearest boundary
            dist_min = abs(pred_pct - z["min"])
            dist_max = abs(pred_pct - z["max"])
            z["confidence"] = "active"
            z["distance_to_boundary"] = min(dist_min, dist_max)
        else:
            # Distance to this zone
            if pred_pct > z["max"]:
                z["distance_to_boundary"] = pred_pct - z["max"]
            else:
                z["distance_to_boundary"] = z["min"] - pred_pct
            z["confidence"] = f"{z['distance_to_boundary']:.2f}% away"

    return {
        "predicted_drawdown": round(pred_pct, 2),
        "predicted_drawdown_decimal": round(pred, 4),
        "zones": zones,
        "current_zone": current_zone,
    }
