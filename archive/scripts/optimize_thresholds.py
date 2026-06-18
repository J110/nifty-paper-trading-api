"""
Comprehensive threshold optimization for the retrained model (no month features).

Loads the new model (35 features, trained 2014-2023), generates predictions for
2024-2025, then sweeps threshold combinations to find optimal signal boundaries
for each version (v5.4.2, v5.4.3, v5.4.4).

Evaluation criteria:
  - Signal accuracy vs actual drawdown outcomes
  - Trade frequency (penalize excessive no-trade days)
  - Safety: avoid bull_full signals during actual crashes
  - Opportunity: avoid no_trade signals during mild drawdown periods
"""

import sys
import os
import numpy as np
import pandas as pd
import joblib
from collections import Counter
from itertools import product

BACKEND_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND_ROOT)

from shared.feature_compute import compute_all_features, load_dhan_options_features

# ─── Config ───
TEST_START = "2024-01-01"
FEATURES_TO_DROP = ['month', 'is_volatile_month']

# ─── Load data & model ───
print("=" * 80)
print("THRESHOLD OPTIMIZATION FOR NEW MODEL (no month features)")
print("=" * 80)

print("\nLoading data...")
merged_path = os.path.join(BACKEND_ROOT, "data", "merged_daily.parquet")
merged_daily = pd.read_parquet(merged_path)

dhan_raw = os.path.join(BACKEND_ROOT, "data", "dhan_raw_options.parquet")
dhan_skew = os.path.join(BACKEND_ROOT, "data", "daily_iv_skew_params.parquet")
dhan_features = load_dhan_options_features(dhan_raw, dhan_skew)

print("Computing features...")
df, feature_cols = compute_all_features(merged_daily, dhan_features,
                                         data_start="2014-01-01", data_end="2026-12-31")

# Load new model
model_path = os.path.join(BACKEND_ROOT, "ml", "downside_model_no_month.pkl")
feat_path = os.path.join(BACKEND_ROOT, "ml", "feature_names_no_month.pkl")
model = joblib.load(model_path)
new_feature_names = joblib.load(feat_path)
print(f"Loaded model: {len(new_feature_names)} features")

# Also load original model for comparison
orig_model = joblib.load(os.path.join(BACKEND_ROOT, "ml", "downside_model.pkl"))
orig_features = joblib.load(os.path.join(BACKEND_ROOT, "ml", "feature_names.pkl"))

# ─── Filter to test period ───
test_df = df[df.index >= TEST_START].copy()
print(f"\nTest period: {test_df.index[0].date()} to {test_df.index[-1].date()} ({len(test_df)} days)")

target_col = 'max_drawdown_30d'
valid_mask = ~test_df[target_col].isna()
test_df = test_df[valid_mask].copy()
print(f"Valid test days (with target): {len(test_df)}")

# ─── Generate predictions ───
X_test_new = test_df[new_feature_names].values
X_test_orig = test_df[orig_features].values

preds_new = model.predict(X_test_new)
preds_orig = orig_model.predict(X_test_orig)
actuals = test_df[target_col].values

test_df['pred_new'] = preds_new
test_df['pred_orig'] = preds_orig
test_df['actual_dd'] = actuals

# ─── Section 1: Prediction Distribution Analysis ───
print("\n" + "=" * 80)
print("SECTION 1: NEW MODEL PREDICTION DISTRIBUTION")
print("=" * 80)

print(f"\n{'Metric':<30} {'New Model':>12} {'Old Model':>12}")
print("-" * 56)
for label, vals in [("Mean", [np.mean(preds_new), np.mean(preds_orig)]),
                     ("Std", [np.std(preds_new), np.std(preds_orig)]),
                     ("Min", [np.min(preds_new), np.min(preds_orig)]),
                     ("Max", [np.max(preds_new), np.max(preds_orig)]),
                     ("Median", [np.median(preds_new), np.median(preds_orig)])]:
    print(f"{label:<30} {vals[0]:>11.4f}  {vals[1]:>11.4f}")

print(f"\n{'Percentile':<30} {'New Model':>12} {'Old Model':>12}")
print("-" * 56)
for pct in [5, 10, 25, 50, 75, 90, 95]:
    v_new = np.percentile(preds_new, pct)
    v_orig = np.percentile(preds_orig, pct)
    print(f"  {pct}th percentile            {v_new:>11.4f}  {v_orig:>11.4f}")

# Monthly breakdown
print(f"\n{'Month':<8} {'New Mean':>10} {'Old Mean':>10} {'Actual Mean':>12} {'Days':>6}")
print("-" * 50)
test_df['month_num'] = test_df.index.month
for m in sorted(test_df['month_num'].unique()):
    mask = test_df['month_num'] == m
    mn = test_df.loc[mask, 'pred_new'].mean()
    mo = test_df.loc[mask, 'pred_orig'].mean()
    ma = test_df.loc[mask, 'actual_dd'].mean()
    nd = mask.sum()
    print(f"  {m:<6} {mn:>10.4f} {mo:>10.4f} {ma:>12.4f} {nd:>6}")


# ─── Signal mapping functions (simplified) ───
def signal_sharp(pred, t1, t2, t3):
    """Sharp cutoffs: bull_full > t1 > bull_half > t2 > iron_condor > t3 > no_trade"""
    if pred > t1:
        return "bull_full", 1.0
    elif pred > t2:
        return "bull_half", 1.0
    elif pred > t3:
        return "iron_condor", 1.0
    else:
        return "no_trade", 0.0


def signal_graduated(pred, t1, t2, t3, floor=0.50, hw_pct=0.50):
    """Graduated with configurable floor and half-width (in percentage points)."""
    pred_pct = pred * 100
    t1p, t2p, t3p = t1 * 100, t2 * 100, t3 * 100

    if pred_pct > t1p + hw_pct:
        return "bull_full", 1.0
    elif pred_pct > t1p - hw_pct:
        progress = (t1p + hw_pct - pred_pct) / (2 * hw_pct)
        return "bull_full", round(1.0 - progress * (1.0 - floor), 3)
    elif pred_pct > t2p + hw_pct:
        return "bull_half", floor
    elif pred_pct > t2p - hw_pct:
        return "iron_condor", floor
    elif pred_pct > t3p + hw_pct:
        return "iron_condor", floor
    elif pred_pct > t3p - hw_pct:
        progress = (t3p + hw_pct - pred_pct) / (2 * hw_pct)
        mult = floor * (1.0 - progress)
        if mult > 0.1:
            return "iron_condor", round(mult, 3)
        return "no_trade", 0.0
    else:
        return "no_trade", 0.0


def signal_graduated_gentle(pred, t1, t2, t3, floor=0.80, hw_pct=0.25):
    """Graduated gentle — same as graduated but with different defaults."""
    return signal_graduated(pred, t1, t2, t3, floor=floor, hw_pct=hw_pct)


# ─── Section 2: Current thresholds evaluation ───
print("\n" + "=" * 80)
print("SECTION 2: CURRENT THRESHOLDS WITH NEW MODEL")
print("=" * 80)

OLD_T1, OLD_T2, OLD_T3 = -0.038, -0.050, -0.065

versions = {
    "v5.4.2": {"fn": signal_sharp, "kwargs": {}},
    "v5.4.3": {"fn": signal_graduated, "kwargs": {"floor": 0.50, "hw_pct": 0.50}},
    "v5.4.4": {"fn": signal_graduated_gentle, "kwargs": {"floor": 0.80, "hw_pct": 0.25}},
}

for ver, cfg in versions.items():
    signals = []
    for pred in preds_new:
        sig, mult = cfg["fn"](pred, OLD_T1, OLD_T2, OLD_T3, **cfg["kwargs"])
        signals.append(sig)

    counts = Counter(signals)
    total = len(signals)
    trade_days = total - counts.get("no_trade", 0)

    print(f"\n{ver} (old thresholds -3.8%/-5.0%/-6.5%):")
    for sig in ["bull_full", "bull_half", "iron_condor", "no_trade"]:
        c = counts.get(sig, 0)
        print(f"  {sig:<15} {c:>4} ({c/total*100:>5.1f}%)")
    print(f"  Trade days:    {trade_days}/{total} ({trade_days/total*100:.1f}%)")

    # Monthly trade rate
    test_df[f'sig_{ver}'] = signals
    print(f"\n  Monthly trade rate:")
    for m in sorted(test_df['month_num'].unique()):
        mask = test_df['month_num'] == m
        month_sigs = test_df.loc[mask, f'sig_{ver}']
        trade_rate = (month_sigs != "no_trade").sum() / len(month_sigs) * 100
        print(f"    Month {m:>2}: {trade_rate:>5.1f}%")


# ─── Section 3: Threshold Sweep ───
print("\n" + "=" * 80)
print("SECTION 3: THRESHOLD OPTIMIZATION SWEEP")
print("=" * 80)

# Define the actual drawdown severity categories
def actual_severity(dd):
    """Classify actual drawdown into severity bucket."""
    if dd > -0.02:
        return "mild"       # Less than 2% drawdown — very safe to trade
    elif dd > -0.04:
        return "moderate"   # 2-4% drawdown — tradeable with caution
    elif dd > -0.06:
        return "elevated"   # 4-6% — risky for bull, ok for IC
    elif dd > -0.10:
        return "severe"     # 6-10% — should be no_trade or bear
    else:
        return "crash"      # 10%+ — definitely no trade


actual_severities = [actual_severity(a) for a in actuals]
test_df['severity'] = actual_severities

print(f"\nActual drawdown severity distribution (2024-2025):")
sev_counts = Counter(actual_severities)
for sev in ["mild", "moderate", "elevated", "severe", "crash"]:
    c = sev_counts.get(sev, 0)
    print(f"  {sev:<12} {c:>4} ({c/len(actuals)*100:>5.1f}%)")


def evaluate_thresholds(preds, actuals, t1, t2, t3, signal_fn, signal_kwargs):
    """Evaluate a threshold combination. Returns metrics dict."""
    signals = []
    mults = []
    for pred in preds:
        sig, mult = signal_fn(pred, t1, t2, t3, **signal_kwargs)
        signals.append(sig)
        mults.append(mult)

    total = len(signals)
    counts = Counter(signals)
    trade_days = total - counts.get("no_trade", 0)
    trade_pct = trade_days / total

    # Severity-matched scoring
    correct = 0
    false_bull_in_crash = 0
    false_notrade_in_mild = 0
    weighted_risk = 0.0

    for sig, actual_dd in zip(signals, actuals):
        sev = actual_severity(actual_dd)

        # Correct signal assessment
        if sig == "bull_full" and sev in ("mild", "moderate"):
            correct += 1
        elif sig == "bull_half" and sev in ("moderate", "elevated"):
            correct += 1
        elif sig == "iron_condor" and sev in ("moderate", "elevated"):
            correct += 1
        elif sig == "no_trade" and sev in ("severe", "crash"):
            correct += 1

        # Dangerous: bull_full during severe drawdown
        if sig == "bull_full" and sev in ("severe", "crash"):
            false_bull_in_crash += 1
            weighted_risk += 3.0 if sev == "crash" else 1.5

        # Missed opportunity: no_trade during mild period
        if sig == "no_trade" and sev == "mild":
            false_notrade_in_mild += 1

    accuracy = correct / total
    danger_rate = false_bull_in_crash / total
    missed_opp_rate = false_notrade_in_mild / total

    # Composite score (higher is better)
    # Heavily penalize danger, moderately penalize missed opportunities
    # Reward trade frequency up to ~80%, diminishing above
    trade_score = min(trade_pct, 0.85) / 0.85  # saturates at 85%
    safety_score = 1.0 - danger_rate * 10  # each 1% danger costs 10% score
    opportunity_score = 1.0 - missed_opp_rate * 2  # each 1% miss costs 2%

    composite = (
        0.30 * accuracy +
        0.30 * safety_score +
        0.20 * trade_score +
        0.20 * opportunity_score
    )

    return {
        "t1": t1, "t2": t2, "t3": t3,
        "accuracy": accuracy,
        "trade_pct": trade_pct,
        "trade_days": trade_days,
        "danger_rate": danger_rate,
        "missed_opp_rate": missed_opp_rate,
        "weighted_risk": weighted_risk,
        "composite": composite,
        "counts": dict(counts),
    }


# Sweep ranges
t1_range = np.arange(-0.010, -0.062, -0.002)  # bull_full threshold
gap_min, gap_max, gap_step = 0.005, 0.030, 0.002

print(f"\nSweep parameters:")
print(f"  bull_full (t1): {t1_range[0]:.3f} to {t1_range[-1]:.3f} ({len(t1_range)} values)")
print(f"  gap to t2: {gap_min:.3f} to {gap_max:.3f}")
print(f"  gap to t3: {gap_min:.3f} to {gap_max:.3f}")

for ver, cfg in versions.items():
    print(f"\n{'─' * 70}")
    print(f"  Optimizing {ver} ({cfg['fn'].__name__})")
    print(f"{'─' * 70}")

    results = []
    for t1 in t1_range:
        for gap2 in np.arange(gap_min, gap_max + 0.001, gap_step):
            t2 = t1 - gap2
            for gap3 in np.arange(gap_min, gap_max + 0.001, gap_step):
                t3 = t2 - gap3
                if t3 < -0.15:  # skip unreasonable
                    continue
                r = evaluate_thresholds(preds_new, actuals, t1, t2, t3,
                                         cfg["fn"], cfg["kwargs"])
                results.append(r)

    # Sort by composite score
    results.sort(key=lambda x: x["composite"], reverse=True)

    print(f"\n  Total combos evaluated: {len(results)}")
    print(f"\n  TOP 10 THRESHOLD COMBINATIONS:")
    print(f"  {'Rank':<5} {'T1':>7} {'T2':>7} {'T3':>7} {'Composite':>10} "
          f"{'Accuracy':>9} {'Trade%':>7} {'Danger%':>8} {'MissOpp%':>9}")
    print(f"  {'-'*75}")

    for i, r in enumerate(results[:10]):
        print(f"  {i+1:<5} {r['t1']:>7.3f} {r['t2']:>7.3f} {r['t3']:>7.3f} "
              f"{r['composite']:>10.4f} {r['accuracy']:>8.1%} "
              f"{r['trade_pct']:>6.1%} {r['danger_rate']:>7.1%} "
              f"{r['missed_opp_rate']:>8.1%}")

    # Best result detail
    best = results[0]
    print(f"\n  BEST for {ver}:")
    print(f"    Thresholds: bull_full={best['t1']:.3f} ({best['t1']*100:.1f}%), "
          f"bull_half={best['t2']:.3f} ({best['t2']*100:.1f}%), "
          f"ic={best['t3']:.3f} ({best['t3']*100:.1f}%)")
    print(f"    Signal distribution: {best['counts']}")
    print(f"    Trade days: {best['trade_days']}/{len(preds_new)} ({best['trade_pct']:.1%})")
    print(f"    Accuracy: {best['accuracy']:.1%}")
    print(f"    Danger rate: {best['danger_rate']:.1%}")
    print(f"    Missed opportunities: {best['missed_opp_rate']:.1%}")

    # Compare with old thresholds
    old_r = evaluate_thresholds(preds_new, actuals, OLD_T1, OLD_T2, OLD_T3,
                                 cfg["fn"], cfg["kwargs"])
    print(f"\n  OLD thresholds (-3.8%/-5.0%/-6.5%):")
    print(f"    Composite: {old_r['composite']:.4f}")
    print(f"    Trade%: {old_r['trade_pct']:.1%}, Accuracy: {old_r['accuracy']:.1%}, "
          f"Danger: {old_r['danger_rate']:.1%}, MissOpp: {old_r['missed_opp_rate']:.1%}")
    delta = best['composite'] - old_r['composite']
    print(f"    Improvement: {delta:+.4f} ({delta/max(old_r['composite'],0.001)*100:+.1f}%)")


# ─── Section 4: Cross-version recommendation ───
print("\n" + "=" * 80)
print("SECTION 4: FINAL RECOMMENDATIONS")
print("=" * 80)

# Find if there's a single threshold set that works well across all versions
print("\nChecking if a single threshold set works across all versions...")
best_universal = None
best_universal_score = -999

for t1 in t1_range:
    for gap2 in np.arange(gap_min, gap_max + 0.001, gap_step):
        t2 = t1 - gap2
        for gap3 in np.arange(gap_min, gap_max + 0.001, gap_step):
            t3 = t2 - gap3
            if t3 < -0.15:
                continue

            scores = []
            for ver, cfg in versions.items():
                r = evaluate_thresholds(preds_new, actuals, t1, t2, t3,
                                         cfg["fn"], cfg["kwargs"])
                scores.append(r["composite"])

            avg_score = np.mean(scores)
            min_score = np.min(scores)
            # Universal score: average with penalty for worst version
            universal = 0.7 * avg_score + 0.3 * min_score

            if universal > best_universal_score:
                best_universal_score = universal
                best_universal = {"t1": t1, "t2": t2, "t3": t3,
                                  "avg": avg_score, "min": min_score,
                                  "scores": scores}

if best_universal:
    t1, t2, t3 = best_universal["t1"], best_universal["t2"], best_universal["t3"]
    print(f"\n  Best universal thresholds:")
    print(f"    bull_full = {t1:.3f} ({t1*100:.1f}%)")
    print(f"    bull_half = {t2:.3f} ({t2*100:.1f}%)")
    print(f"    iron_condor = {t3:.3f} ({t3*100:.1f}%)")
    print(f"    Universal score: {best_universal_score:.4f} (avg={best_universal['avg']:.4f}, min={best_universal['min']:.4f})")

    for i, (ver, cfg) in enumerate(versions.items()):
        r = evaluate_thresholds(preds_new, actuals, t1, t2, t3,
                                 cfg["fn"], cfg["kwargs"])
        print(f"\n    {ver}: composite={r['composite']:.4f}, "
              f"trade%={r['trade_pct']:.1%}, accuracy={r['accuracy']:.1%}, "
              f"danger={r['danger_rate']:.1%}")
        print(f"      Signals: {r['counts']}")

        # Monthly trade rate
        month_sigs = []
        for pred in preds_new:
            sig, mult = cfg["fn"](pred, t1, t2, t3, **cfg["kwargs"])
            month_sigs.append(sig)
        test_df[f'opt_sig_{ver}'] = month_sigs
        print(f"      Monthly trade rates:")
        for m in sorted(test_df['month_num'].unique()):
            mask = test_df['month_num'] == m
            ms = test_df.loc[mask, f'opt_sig_{ver}']
            tr = (ms != "no_trade").sum() / len(ms) * 100
            print(f"        Month {m:>2}: {tr:>5.1f}%")


print("\n" + "=" * 80)
print("DONE — Use these thresholds to update config files")
print("=" * 80)
