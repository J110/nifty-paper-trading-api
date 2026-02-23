"""Signal mapping functions — extracted from nifty_options_model/src/model.py.

Only the signal mapping functions (not training code).
Self-contained: no dependencies on nifty_options_model/.
"""

import numpy as np


def rules_based_signal(row, config_dict=None):
    """Rules-based entry signal (credit-only baseline)."""
    vix = row.get('india_vix', 20)
    vix_pct = row.get('vix_percentile_252d', 50)
    dist_50dma = row.get('nifty_distance_50dma', 0)
    is_exp_week = row.get('is_expiry_week', 0)

    cfg = config_dict or {}
    vix_min = cfg.get('VIX_MIN_ENTRY', 11)
    vix_max = cfg.get('VIX_MAX_ENTRY', 25)
    vix_pct_max = cfg.get('VIX_PERCENTILE_MAX', 80)

    if 22 < vix <= 25 and vix_pct < vix_pct_max and dist_50dma > 0 and is_exp_week == 0:
        return "enter_half"
    if (vix_min <= vix <= vix_max and vix_pct < vix_pct_max and
            dist_50dma > 0 and is_exp_week == 0):
        return "enter_full"
    return "no_entry"


def model_signal(model, features_row, feature_cols, config_dict=None):
    """Regression model entry signal based on predicted max drawdown.
    Returns: (signal, predicted_drawdown)"""
    X = features_row[feature_cols].values.reshape(1, -1)
    predicted_dd = model.predict(X)[0]

    cfg = config_dict or {}
    bull_full_t = cfg.get('DRAWDOWN_BULL_FULL', -0.020)
    bull_half_t = cfg.get('DRAWDOWN_BULL_HALF', -0.047)
    ic_t = cfg.get('DRAWDOWN_IRON_CONDOR', -0.076)
    bear_half_t = cfg.get('DRAWDOWN_BEAR_HALF', -0.140)
    bear_full_t = cfg.get('DRAWDOWN_BEAR_FULL', -0.170)
    bear_enabled = cfg.get('BEAR_ENABLED', True)

    if predicted_dd > bull_full_t:
        signal = "bull_full"
    elif predicted_dd > bull_half_t:
        signal = "bull_half"
    elif predicted_dd > ic_t:
        signal = "iron_condor"
    elif predicted_dd > bear_half_t:
        signal = "no_trade"
    elif predicted_dd > bear_full_t:
        signal = "bear_half"
    else:
        signal = "bear_full"

    if not bear_enabled and signal in ("bear_half", "bear_full"):
        signal = "no_trade"
    return signal, predicted_dd


def model_signal_graduated(model, features_row, feature_cols, config_dict=None):
    """Graduated signal mapping (v5.4.3).
    Returns: (signal, predicted_drawdown, size_multiplier)"""
    X = features_row[feature_cols].values.reshape(1, -1)
    predicted_dd = model.predict(X)[0]
    bear_enabled = (config_dict or {}).get('BEAR_ENABLED', True)

    if predicted_dd > -0.01:
        return "bull_full", predicted_dd, 1.0
    elif predicted_dd > -0.02:
        frac = (predicted_dd - (-0.02)) / ((-0.01) - (-0.02))
        return "bull_full", predicted_dd, 0.7 + 0.3 * frac
    elif predicted_dd > -0.03:
        frac = (predicted_dd - (-0.03)) / ((-0.02) - (-0.03))
        if frac > 0.5:
            return "bull_half", predicted_dd, 0.5 + 0.5 * frac
        else:
            return "iron_condor", predicted_dd, 0.5 + 0.5 * (1 - frac)
    elif predicted_dd > -0.04:
        frac = (predicted_dd - (-0.04)) / ((-0.03) - (-0.04))
        return "iron_condor", predicted_dd, max(0.25, 0.5 * frac)
    else:
        return "no_trade", predicted_dd, 0.0


def model_signal_graduated_gentle(model, features_row, feature_cols, config_dict=None):
    """Gentle graduated signal mapping (v5.4.4).
    Returns: (signal, predicted_drawdown, size_multiplier)"""
    X = features_row[feature_cols].values.reshape(1, -1)
    predicted_dd = model.predict(X)[0]

    cfg = config_dict or {}
    floor = cfg.get('GRADUATED_FLOOR', 0.80)
    hw = cfg.get('GRADUATED_HALF_WIDTH', 0.0025)
    bull_full_t = cfg.get('DRAWDOWN_BULL_FULL', -0.020)
    bull_half_t = cfg.get('DRAWDOWN_BULL_HALF', -0.047)
    ic_t = cfg.get('DRAWDOWN_IRON_CONDOR', -0.076)

    def _lerp(val, lo, hi, out_lo, out_hi):
        if hi == lo:
            return (out_lo + out_hi) / 2
        frac = (val - lo) / (hi - lo)
        frac = max(0.0, min(1.0, frac))
        return out_lo + frac * (out_hi - out_lo)

    if predicted_dd > bull_full_t + hw:
        return "bull_full", predicted_dd, 1.0
    if predicted_dd > bull_full_t - hw:
        mult = _lerp(predicted_dd, bull_full_t - hw, bull_full_t + hw, floor, 1.0)
        return "bull_full", predicted_dd, mult
    if predicted_dd > bull_half_t + hw:
        return "bull_half", predicted_dd, 1.0
    if predicted_dd > bull_half_t - hw:
        mult = _lerp(predicted_dd, bull_half_t - hw, bull_half_t + hw, floor, 1.0)
        return "bull_half", predicted_dd, mult
    if predicted_dd > ic_t + hw:
        return "iron_condor", predicted_dd, 1.0
    if predicted_dd > ic_t - hw:
        mult = _lerp(predicted_dd, ic_t - hw, ic_t + hw, floor, 1.0)
        return "iron_condor", predicted_dd, mult
    return "no_trade", predicted_dd, 0.0


def model_signal_directional(model, features_row, feature_cols, config_dict=None):
    """v6.0: Directional signal mapping.
    Returns: (signal_type, predicted_dd, size_multiplier, call_tier)"""
    X = features_row[feature_cols].values.reshape(1, -1)
    predicted_dd = model.predict(X)[0]

    cfg = config_dict or {}
    floor = cfg.get('GRADUATED_FLOOR', 0.80)
    hw = cfg.get('GRADUATED_HALF_WIDTH', 0.0025)
    strong_bull_t = cfg.get('STRONG_BULL_THRESHOLD', -0.005)
    moderate_bull_t = cfg.get('MODERATE_BULL_THRESHOLD', -0.01)
    bull_full_t = cfg.get('DRAWDOWN_BULL_FULL', -0.020)
    bull_half_t = cfg.get('DRAWDOWN_BULL_HALF', -0.047)
    ic_t = cfg.get('DRAWDOWN_IRON_CONDOR', -0.076)

    def _lerp(val, lo, hi, out_lo, out_hi):
        if hi == lo:
            return (out_lo + out_hi) / 2
        frac = (val - lo) / (hi - lo)
        frac = max(0.0, min(1.0, frac))
        return out_lo + frac * (out_hi - out_lo)

    if predicted_dd > strong_bull_t:
        return "bull_full", predicted_dd, 1.0, 1
    elif predicted_dd > moderate_bull_t:
        return "bull_full", predicted_dd, 1.0, 2
    elif predicted_dd > bull_full_t + hw:
        return "bull_full", predicted_dd, 1.0, 0
    elif predicted_dd > bull_full_t - hw:
        mult = _lerp(predicted_dd, bull_full_t - hw, bull_full_t + hw, floor, 1.0)
        return "bull_full", predicted_dd, mult, 0
    elif predicted_dd > bull_half_t + hw:
        return "bull_half", predicted_dd, 1.0, 0
    elif predicted_dd > bull_half_t - hw:
        mult = _lerp(predicted_dd, bull_half_t - hw, bull_half_t + hw, floor, 1.0)
        return "bull_half", predicted_dd, mult, 0
    elif predicted_dd > ic_t + hw:
        return "iron_condor", predicted_dd, 1.0, 0
    elif predicted_dd > ic_t - hw:
        mult = _lerp(predicted_dd, ic_t - hw, ic_t + hw, floor, 1.0)
        return "iron_condor", predicted_dd, mult, 0
    else:
        return "no_trade", predicted_dd, 0.0, 0


def model_signal_directional_bear(model, features_row, feature_cols, config_dict=None):
    """v6.2: Extends graduated_gentle with bear debit zone.
    Returns: (signal_type, predicted_dd, size_multiplier, bear_tier)"""
    X = features_row[feature_cols].values.reshape(1, -1)
    predicted_dd = model.predict(X)[0]

    cfg = config_dict or {}
    floor = cfg.get('GRADUATED_FLOOR', 0.80)
    hw = cfg.get('GRADUATED_HALF_WIDTH', 0.0025)
    bull_full_t = cfg.get('DRAWDOWN_BULL_FULL', -0.020)
    bull_half_t = cfg.get('DRAWDOWN_BULL_HALF', -0.047)
    ic_t = cfg.get('DRAWDOWN_IRON_CONDOR', -0.076)
    bear_moderate_t = cfg.get('BEAR_MODERATE_THRESHOLD', -0.065)
    bear_strong_t = cfg.get('BEAR_STRONG_THRESHOLD', -0.090)

    def _lerp(val, lo, hi, out_lo, out_hi):
        if hi == lo:
            return (out_lo + out_hi) / 2
        frac = (val - lo) / (hi - lo)
        frac = max(0.0, min(1.0, frac))
        return out_lo + frac * (out_hi - out_lo)

    if predicted_dd > bull_full_t + hw:
        return "bull_full", predicted_dd, 1.0, 0
    if predicted_dd > bull_full_t - hw:
        mult = _lerp(predicted_dd, bull_full_t - hw, bull_full_t + hw, floor, 1.0)
        return "bull_full", predicted_dd, mult, 0
    if predicted_dd > bull_half_t + hw:
        return "bull_half", predicted_dd, 1.0, 0
    if predicted_dd > bull_half_t - hw:
        mult = _lerp(predicted_dd, bull_half_t - hw, bull_half_t + hw, floor, 1.0)
        return "bull_half", predicted_dd, mult, 0
    if predicted_dd > ic_t + hw:
        return "iron_condor", predicted_dd, 1.0, 0
    if predicted_dd > ic_t - hw:
        mult = _lerp(predicted_dd, ic_t - hw, ic_t + hw, floor, 1.0)
        return "iron_condor", predicted_dd, mult, 0
    if predicted_dd > bear_strong_t:
        return "bear_debit", predicted_dd, 1.0, 2
    return "bear_debit", predicted_dd, 1.0, 1


def predict_upside(upside_model, features_row, feature_cols):
    """Predict upside rally for single day. Returns predicted max_rally_10d."""
    if upside_model is None:
        return 0.0
    X = features_row[feature_cols].values.reshape(1, -1)
    return float(upside_model.predict(X)[0])


def upside_signal(predicted_rally, config_dict=None):
    """Map rally prediction to call debit decision (fixed thresholds).
    Returns: (call_tier, size_mult)"""
    cfg = config_dict or {}
    strong_t = cfg.get('UPSIDE_STRONG_THRESHOLD', 0.04)
    moderate_t = cfg.get('UPSIDE_MODERATE_THRESHOLD', 0.03)
    min_t = cfg.get('UPSIDE_MIN_THRESHOLD', 0.02)

    if predicted_rally >= strong_t:
        return 1, cfg.get('CALL_SIZE_MULT_T1', 1.0)
    elif predicted_rally >= moderate_t:
        return 2, cfg.get('CALL_SIZE_MULT_T2', 0.50)
    elif predicted_rally >= min_t:
        return 2, cfg.get('CALL_SIZE_MULT_T2_LIGHT', 0.30)
    return 0, 0.0


def upside_signal_adaptive(predicted_rally, history, config_dict=None):
    """Adaptive thresholds based on rolling distribution.
    Returns: (call_tier, size_mult)"""
    cfg = config_dict or {}
    lookback = cfg.get('UPSIDE_PERCENTILE_LOOKBACK', 60)

    if len(history) < lookback:
        return upside_signal(predicted_rally, cfg)

    recent = np.array(history[-lookback:])
    strong_pct = cfg.get('UPSIDE_STRONG_PERCENTILE', 85)
    moderate_pct = cfg.get('UPSIDE_MODERATE_PERCENTILE', 70)
    light_pct = cfg.get('UPSIDE_LIGHT_PERCENTILE', 50)

    strong_t = np.percentile(recent, strong_pct)
    moderate_t = np.percentile(recent, moderate_pct)
    light_t = np.percentile(recent, light_pct)

    if predicted_rally >= strong_t:
        return 1, cfg.get('CALL_SIZE_MULT_T1', 1.0)
    elif predicted_rally >= moderate_t:
        return 2, cfg.get('CALL_SIZE_MULT_T2', 0.50)
    elif predicted_rally >= light_t:
        return 2, cfg.get('CALL_SIZE_MULT_T2_LIGHT', 0.30)
    return 0, 0.0
