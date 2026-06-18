"""
Backtest comparison: WITH vs WITHOUT 'month' feature.

This script uses the existing trained model but zeroes out the 'month'
(and optionally 'is_volatile_month') features to see how predictions change.

Since we can't retrain the model without month, we instead:
1. Compute features normally → get predictions (baseline)
2. Set month=6 (neutral mid-year value) for all dates → get predictions (no-month)
3. Compare signal distributions and simulated trade outcomes for 2024-2025

The key insight: if 'month' is causing Feb to always predict large drawdowns,
zeroing it out will shift predictions toward more trades.
"""

import sys
import os
import json
import pickle
import numpy as np
import pandas as pd
from datetime import date

BACKEND_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND_ROOT)

from shared.feature_compute import compute_all_features, load_dhan_options_features, BASE_FEATURE_COLS

# ─── Load model and feature names ───
import joblib

model = joblib.load(os.path.join(BACKEND_ROOT, "ml", "downside_model.pkl"))
feature_names = joblib.load(os.path.join(BACKEND_ROOT, "ml", "feature_names.pkl"))

print(f"Model: {type(model).__name__}")
print(f"Features: {len(feature_names)}")

# Feature importance for calendar features
importances = dict(zip(feature_names, model.feature_importances_))
calendar_feats = ['month', 'day_of_month', 'is_volatile_month', 'is_expiry_week', 'days_to_monthly_expiry']
print("\n=== Calendar Feature Importances ===")
total_imp = sum(model.feature_importances_)
for f in calendar_feats:
    imp = importances.get(f, 0)
    print(f"  {f:30s}  {imp:.4f}  ({imp/total_imp*100:.1f}%)")

# Top 10 features
ranked = sorted(importances.items(), key=lambda x: x[1], reverse=True)
print("\n=== Top 10 Features ===")
for name, imp in ranked[:10]:
    print(f"  {name:35s}  {imp:.4f}  ({imp/total_imp*100:.1f}%)")

# ─── Load data ───
merged_path = os.path.join(BACKEND_ROOT, "data", "merged_daily.parquet")
merged_daily = pd.read_parquet(merged_path)
print(f"\nMerged daily: {len(merged_daily)} rows, {merged_daily.index[0].date()} to {merged_daily.index[-1].date()}")

# Load Dhan features if available
dhan_raw = os.path.join(BACKEND_ROOT, "data", "dhan_raw_options.parquet")
dhan_skew = os.path.join(BACKEND_ROOT, "data", "daily_iv_skew_params.parquet")
dhan_features = load_dhan_options_features(dhan_raw, dhan_skew)

# ─── Compute full feature matrix ───
print("\nComputing features...")
df, feature_cols = compute_all_features(merged_daily, dhan_features,
                                         data_start="2014-01-01", data_end="2026-12-31")
print(f"Feature matrix: {len(df)} rows, {len(feature_cols)} features")

# ─── Filter to 2024-2025 ───
mask = (df.index >= "2024-01-01") & (df.index <= "2025-12-31")
df_period = df[mask].copy()
print(f"Analysis period: {len(df_period)} trading days ({df_period.index[0].date()} to {df_period.index[-1].date()})")

# ─── Signal mapping thresholds (from config.py) ───
THRESHOLDS = {
    "v5.4.2": {  # sharp
        "bull_full": -0.038,
        "bull_half": -0.050,
        "iron_condor": -0.065,
    },
    "v5.4.3": {  # graduated
        "zones": [
            (-0.01, "bull_full", 1.0),
            (-0.02, "bull_full", 0.7),
            (-0.025, "bull_half", 0.75),
            (-0.03, "iron_condor", 0.75),
            (-0.04, "iron_condor", 0.25),
        ]
    },
    "v5.4.4": {  # graduated_gentle
        "bull_full": -0.038,
        "bull_half": -0.050,
        "iron_condor": -0.065,
        "hw": 0.0025,
        "floor": 0.80,
    },
}


def map_signal_sharp(pred_dd):
    """v5.4.2 sharp thresholds"""
    if pred_dd > -0.038:
        return "bull_full", 1.0
    elif pred_dd > -0.050:
        return "bull_half", 1.0
    elif pred_dd > -0.065:
        return "iron_condor", 1.0
    else:
        return "no_trade", 0.0


def map_signal_graduated(pred_dd):
    """v5.4.3 graduated"""
    if pred_dd > -0.01:
        return "bull_full", 1.0
    elif pred_dd > -0.02:
        frac = (pred_dd - (-0.02)) / ((-0.01) - (-0.02))
        return "bull_full", 0.7 + 0.3 * frac
    elif pred_dd > -0.03:
        frac = (pred_dd - (-0.03)) / ((-0.02) - (-0.03))
        if frac > 0.5:
            return "bull_half", 0.5 + 0.5 * frac
        else:
            return "iron_condor", 0.5 + 0.5 * (1 - frac)
    elif pred_dd > -0.04:
        frac = (pred_dd - (-0.04)) / ((-0.03) - (-0.04))
        return "iron_condor", max(0.25, 0.5 * frac)
    else:
        return "no_trade", 0.0


def map_signal_graduated_gentle(pred_dd):
    """v5.4.4 graduated gentle"""
    hw = 0.0025
    floor = 0.80
    bull_full_t = -0.038
    bull_half_t = -0.050
    ic_t = -0.065

    def _lerp(val, lo, hi, out_lo, out_hi):
        if hi == lo:
            return (out_lo + out_hi) / 2
        frac = max(0.0, min(1.0, (val - lo) / (hi - lo)))
        return out_lo + frac * (out_hi - out_lo)

    if pred_dd > bull_full_t + hw:
        return "bull_full", 1.0
    if pred_dd > bull_full_t - hw:
        return "bull_full", _lerp(pred_dd, bull_full_t - hw, bull_full_t + hw, floor, 1.0)
    if pred_dd > bull_half_t + hw:
        return "bull_half", 1.0
    if pred_dd > bull_half_t - hw:
        return "bull_half", _lerp(pred_dd, bull_half_t - hw, bull_half_t + hw, floor, 1.0)
    if pred_dd > ic_t + hw:
        return "iron_condor", 1.0
    if pred_dd > ic_t - hw:
        return "iron_condor", _lerp(pred_dd, ic_t - hw, ic_t + hw, floor, 1.0)
    return "no_trade", 0.0


SIGNAL_MAPPERS = {
    "v5.4.2": map_signal_sharp,
    "v5.4.3": map_signal_graduated,
    "v5.4.4": map_signal_graduated_gentle,
}


def predict_all(df_subset, feature_cols, month_override=None, is_volatile_override=None):
    """Run model predictions, optionally overriding month feature."""
    X = df_subset[feature_cols].copy()

    if month_override is not None and 'month' in X.columns:
        X['month'] = month_override
    if is_volatile_override is not None and 'is_volatile_month' in X.columns:
        X['is_volatile_month'] = is_volatile_override

    predictions = model.predict(X.values)
    return predictions


# ─── Run both scenarios ───
print("\n" + "="*80)
print("RUNNING PREDICTIONS: BASELINE vs NO-MONTH")
print("="*80)

preds_baseline = predict_all(df_period, feature_cols)
# Set month=6 (June — neutral, mid-year) and is_volatile_month=0
preds_no_month = predict_all(df_period, feature_cols, month_override=6, is_volatile_override=0)

df_period = df_period.copy()
df_period['pred_baseline'] = preds_baseline
df_period['pred_no_month'] = preds_no_month
df_period['pred_diff'] = preds_no_month - preds_baseline
df_period['actual_month'] = df_period.index.month

# ─── Monthly breakdown of predictions ───
print("\n=== MONTHLY AVERAGE PREDICTIONS (2024-2025) ===")
print(f"{'Month':>6s}  {'Baseline':>10s}  {'No-Month':>10s}  {'Diff':>10s}  {'Days':>5s}")
print("-" * 50)
for m in range(1, 13):
    mask_m = df_period['actual_month'] == m
    if mask_m.sum() == 0:
        continue
    bl = df_period.loc[mask_m, 'pred_baseline'].mean()
    nm = df_period.loc[mask_m, 'pred_no_month'].mean()
    diff = nm - bl
    count = mask_m.sum()
    # Highlight months where month feature makes things worse
    arrow = "  <<<" if diff > 0.005 else ""
    print(f"  {m:4d}  {bl*100:>9.2f}%  {nm*100:>9.2f}%  {diff*100:>+9.2f}%  {count:>5d}{arrow}")

# ─── Signal distribution comparison for each version ───
for version, mapper in SIGNAL_MAPPERS.items():
    print(f"\n{'='*80}")
    print(f"VERSION: {version}")
    print(f"{'='*80}")

    signals_bl = []
    signals_nm = []

    for i in range(len(df_period)):
        sig_bl, mult_bl = mapper(preds_baseline[i])
        sig_nm, mult_nm = mapper(preds_no_month[i])
        signals_bl.append(sig_bl)
        signals_nm.append(sig_nm)

    df_period[f'signal_bl_{version}'] = signals_bl
    df_period[f'signal_nm_{version}'] = signals_nm

    # Overall signal counts
    from collections import Counter
    counts_bl = Counter(signals_bl)
    counts_nm = Counter(signals_nm)

    print(f"\n--- Overall Signal Distribution (2024-2025) ---")
    print(f"{'Signal':>15s}  {'Baseline':>10s}  {'No-Month':>10s}  {'Diff':>10s}")
    print("-" * 50)
    for sig in ["bull_full", "bull_half", "iron_condor", "no_trade"]:
        bl = counts_bl.get(sig, 0)
        nm = counts_nm.get(sig, 0)
        print(f"  {sig:>13s}  {bl:>10d}  {nm:>10d}  {nm-bl:>+10d}")

    trade_days_bl = len(df_period) - counts_bl.get("no_trade", 0)
    trade_days_nm = len(df_period) - counts_nm.get("no_trade", 0)
    print(f"\n  Total trade days:  {trade_days_bl:>4d}  →  {trade_days_nm:>4d}  ({trade_days_nm - trade_days_bl:+d})")

    # Monthly breakdown for this version
    print(f"\n--- Monthly No-Trade Days: Baseline vs No-Month ---")
    print(f"{'Month':>6s}  {'BL no_trade':>12s}  {'NM no_trade':>12s}  {'BL trade%':>10s}  {'NM trade%':>10s}")
    print("-" * 60)
    for m in range(1, 13):
        mask_m = df_period['actual_month'] == m
        if mask_m.sum() == 0:
            continue
        total_m = mask_m.sum()
        bl_nt = (df_period.loc[mask_m, f'signal_bl_{version}'] == 'no_trade').sum()
        nm_nt = (df_period.loc[mask_m, f'signal_nm_{version}'] == 'no_trade').sum()
        bl_pct = (total_m - bl_nt) / total_m * 100
        nm_pct = (total_m - nm_nt) / total_m * 100
        arrow = "  <<<" if nm_pct - bl_pct > 10 else ""
        print(f"  {m:4d}  {bl_nt:>12d}  {nm_nt:>12d}  {bl_pct:>9.0f}%  {nm_pct:>9.0f}%{arrow}")

# ─── February focus ───
print(f"\n{'='*80}")
print(f"FEBRUARY DEEP DIVE")
print(f"{'='*80}")

feb_mask = df_period['actual_month'] == 2
feb_data = df_period[feb_mask]
print(f"\nFeb trading days: {len(feb_data)}")
print(f"Baseline prediction range: {feb_data['pred_baseline'].min()*100:.2f}% to {feb_data['pred_baseline'].max()*100:.2f}%")
print(f"No-Month prediction range: {feb_data['pred_no_month'].min()*100:.2f}% to {feb_data['pred_no_month'].max()*100:.2f}%")
print(f"Average shift: {feb_data['pred_diff'].mean()*100:+.2f}% (positive = less bearish)")

for version in SIGNAL_MAPPERS:
    bl_nt = (feb_data[f'signal_bl_{version}'] == 'no_trade').sum()
    nm_nt = (feb_data[f'signal_nm_{version}'] == 'no_trade').sum()
    print(f"\n  {version}:")
    print(f"    Baseline no_trade days: {bl_nt}/{len(feb_data)} ({bl_nt/len(feb_data)*100:.0f}%)")
    print(f"    No-Month no_trade days: {nm_nt}/{len(feb_data)} ({nm_nt/len(feb_data)*100:.0f}%)")

    # Show what actual drawdowns were
    if 'max_drawdown_30d' in feb_data.columns:
        actual_dd = feb_data['max_drawdown_30d'].dropna()
        if len(actual_dd) > 0:
            print(f"    Actual max drawdowns: mean={actual_dd.mean()*100:.2f}%, worst={actual_dd.min()*100:.2f}%")

# ─── Approximate PnL comparison (simplified) ───
print(f"\n{'='*80}")
print(f"SIMPLIFIED PNL COMPARISON (2024-2025)")
print(f"{'='*80}")
print(f"(Assumes: +Rs 2,500 credit per bull_put trade that doesn't hit stop,")
print(f" +Rs 4,000 credit per iron_condor, actual P&L based on max_drawdown_30d)")
print()

for version, mapper in SIGNAL_MAPPERS.items():
    for label, pred_col in [("Baseline", "pred_baseline"), ("No-Month", "pred_no_month")]:
        trades = 0
        wins = 0
        losses = 0
        total_pnl = 0

        for i, (idx, row) in enumerate(df_period.iterrows()):
            pred = row[pred_col]
            signal, mult = mapper(pred)

            if signal == "no_trade":
                continue

            trades += 1
            actual_dd = row.get('max_drawdown_30d', 0)

            if actual_dd is None or pd.isna(actual_dd):
                continue

            # Simplified: if actual drawdown exceeded strike distance, it's a loss
            if signal in ("bull_full", "bull_half"):
                credit = 2500 * mult
                if actual_dd < -0.03:  # breached sell strike
                    loss = min(credit * 3, 7500)  # stop loss at 3x
                    total_pnl -= loss
                    losses += 1
                else:
                    total_pnl += credit * 0.7  # profit target ~70%
                    wins += 1
            elif signal == "iron_condor":
                credit = 4000 * mult
                if actual_dd < -0.03 or (hasattr(row, 'max_rally_10d') and row.get('max_rally_10d', 0) > 0.03):
                    loss = min(credit * 3, 12000)
                    total_pnl -= loss
                    losses += 1
                else:
                    total_pnl += credit * 0.7
                    wins += 1

        win_rate = wins / trades * 100 if trades > 0 else 0
        print(f"  {version} ({label:>8s}): {trades:>4d} trades, "
              f"{wins}W/{losses}L ({win_rate:.0f}%), "
              f"PnL: Rs {total_pnl:>+10,.0f}")
    print()

# ─── Show prediction shift by year-month ───
print(f"\n{'='*80}")
print(f"DETAILED MONTH-BY-MONTH VIEW (v5.4.2 sharp)")
print(f"{'='*80}")
print(f"{'Year-Month':>10s}  {'Avg Pred BL':>12s}  {'Avg Pred NM':>12s}  {'Shift':>8s}  {'BL Trades':>10s}  {'NM Trades':>10s}")
print("-" * 70)

df_period['ym'] = df_period.index.to_period('M')
for ym in sorted(df_period['ym'].unique()):
    mask_ym = df_period['ym'] == ym
    sub = df_period[mask_ym]
    bl_avg = sub['pred_baseline'].mean()
    nm_avg = sub['pred_no_month'].mean()
    shift = nm_avg - bl_avg
    bl_trades = (sub['signal_bl_v5.4.2'] != 'no_trade').sum()
    nm_trades = (sub['signal_nm_v5.4.2'] != 'no_trade').sum()
    total = len(sub)
    print(f"  {str(ym):>8s}  {bl_avg*100:>11.2f}%  {nm_avg*100:>11.2f}%  {shift*100:>+7.2f}%  {bl_trades:>4d}/{total:<4d}  {nm_trades:>4d}/{total:<4d}")
