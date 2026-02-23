"""
Shared feature computation — THE single source of truth.

Extracted verbatim from nifty_options_model/src/feature_engineering.py
(the code that produced the 59.9% backtest return). Both the backtest
pipeline and the live backend MUST call this module. No reimplementations.

Usage (live):
    from shared.feature_compute import compute_features_for_date
    features = compute_features_for_date(merged_daily_df, target_date)

Usage (backtest):
    from shared.feature_compute import compute_all_features
    df, feature_cols = compute_all_features(merged_daily_df, dhan_features_df)
"""

import numpy as np
import pandas as pd
import ta


# ─── Monthly expiry helpers (from nifty_options_model/src/utils.py) ───

def _get_last_thursday(year, month):
    """Get the last Thursday of a given month."""
    from datetime import datetime, timedelta
    if month == 12:
        next_month = datetime(year + 1, 1, 1)
    else:
        next_month = datetime(year, month + 1, 1)
    last_day = next_month - timedelta(days=1)
    while last_day.weekday() != 3:
        last_day -= timedelta(days=1)
    return last_day.date()


def _get_monthly_expiries(start_date, end_date):
    """Generate all monthly expiry dates (last Thursday) in range."""
    expiries = []
    current = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    while current <= end:
        exp = _get_last_thursday(current.year, current.month)
        if pd.Timestamp(exp) >= pd.Timestamp(start_date):
            expiries.append(pd.Timestamp(exp))
        if current.month == 12:
            from datetime import datetime
            current = pd.Timestamp(datetime(current.year + 1, 1, 1))
        else:
            from datetime import datetime
            current = pd.Timestamp(datetime(current.year, current.month + 1, 1))
    return expiries


# ─── Dhan options features loader ───

def load_dhan_options_features(raw_path, skew_path):
    """
    Load Dhan raw options data and IV skew params to compute options-derived features.
    Exact copy of _load_dhan_options_features() from the backtest.
    Returns DataFrame indexed by date, or None if data unavailable.
    """
    import os
    if not os.path.exists(raw_path) or not os.path.exists(skew_path):
        return None

    try:
        raw = pd.read_parquet(raw_path)
        skew = pd.read_parquet(skew_path)
    except Exception:
        return None

    if "date" in raw.columns:
        raw["date"] = pd.to_datetime(raw["date"])
    if "date" in skew.columns:
        skew["date"] = pd.to_datetime(skew["date"])
        skew = skew.set_index("date")

    # Feature 1: deep_otm_oi_ratio
    deep_otm_labels = [f"ATM-{i}" for i in range(7, 11)]
    near_otm_labels = ["ATM"] + [f"ATM-{i}" for i in range(1, 4)]
    daily_oi = raw.groupby(["date", "strike_label"])["oi"].last().unstack(fill_value=0)
    deep_oi = daily_oi[[c for c in deep_otm_labels if c in daily_oi.columns]].sum(axis=1)
    near_oi = daily_oi[[c for c in near_otm_labels if c in daily_oi.columns]].sum(axis=1)
    deep_otm_oi_ratio = (deep_oi / near_oi.replace(0, np.nan)).fillna(0)

    # Feature 2: deep_otm_oi_ratio_change_5d
    deep_otm_oi_ratio_change_5d = deep_otm_oi_ratio - deep_otm_oi_ratio.shift(5)

    # Feature 3: put_oi_buildup_ratio
    total_oi = daily_oi.sum(axis=1)
    oi_5d_ago = total_oi.shift(5)
    put_oi_buildup_ratio = ((total_oi - oi_5d_ago) / oi_5d_ago.replace(0, np.nan)).fillna(0)

    # Feature 4: iv_skew_steepness
    iv_skew_steepness = pd.Series(dtype=float)
    if "iv_8pct_otm" in skew.columns and "iv_atm" in skew.columns:
        iv_skew_steepness = skew["iv_8pct_otm"] - skew["iv_atm"]

    # Feature 5: iv_skew_change_5d
    iv_skew_change_5d = iv_skew_steepness - iv_skew_steepness.shift(5)

    # Feature 6: atm_iv
    atm_iv = skew["iv_atm"] if "iv_atm" in skew.columns else pd.Series(dtype=float)

    # Feature 7: atm_iv_percentile_252d
    atm_iv_percentile_252d = atm_iv.rolling(252, min_periods=60).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    )

    # Feature 8: put_volume_surge_ratio
    daily_vol = raw.groupby(["date", "strike_label"])["volume"].last().unstack(fill_value=0)
    total_vol = daily_vol.sum(axis=1)
    avg_vol_20d = total_vol.rolling(20, min_periods=5).mean()
    put_volume_surge_ratio = (total_vol / avg_vol_20d.replace(0, np.nan)).fillna(1.0)

    # Feature 9: atm_put_intraday_range
    atm_raw = raw[raw["strike_label"] == "ATM"].copy()
    if len(atm_raw) > 0:
        atm_daily = atm_raw.groupby("date").agg(
            atm_high=("high", "max"),
            atm_low=("low", "min"),
            atm_close=("close", "last"),
        )
        atm_put_intraday_range = (
            (atm_daily["atm_high"] - atm_daily["atm_low"])
            / atm_daily["atm_close"].replace(0, np.nan)
        ).fillna(0)
    else:
        atm_put_intraday_range = pd.Series(dtype=float)

    # Combine
    features = pd.DataFrame({
        "deep_otm_oi_ratio": deep_otm_oi_ratio,
        "deep_otm_oi_ratio_change_5d": deep_otm_oi_ratio_change_5d,
        "put_oi_buildup_ratio": put_oi_buildup_ratio,
        "put_volume_surge_ratio": put_volume_surge_ratio,
    })
    for name, series in [
        ("iv_skew_steepness", iv_skew_steepness),
        ("iv_skew_change_5d", iv_skew_change_5d),
        ("atm_iv", atm_iv),
        ("atm_iv_percentile_252d", atm_iv_percentile_252d),
    ]:
        if len(series) > 0:
            features[name] = series
    if len(atm_put_intraday_range) > 0:
        features["atm_put_intraday_range"] = atm_put_intraday_range

    features.index = pd.to_datetime(features.index)
    return features


# ─── The 26 base feature names (order matters for the model) ───
# Note: 'month' and 'is_volatile_month' removed — seasonality is captured
# by VIX/volatility features; month caused systematic bearish bias in Jan/Feb.

BASE_FEATURE_COLS = [
    'nifty_return_5d', 'nifty_return_10d', 'nifty_return_20d',
    'nifty_distance_50dma', 'nifty_distance_200dma', 'golden_cross',
    'rsi_14', 'higher_highs_5w',
    'india_vix', 'vix_change_5d', 'vix_percentile_252d',
    'realized_vol_20d', 'variance_risk_premium', 'vix_rising_streak',
    'pcr_proxy',
    'sp500_return_5d', 'us_vix', 'us_vix_change_5d',
    'dxy_level', 'dxy_change_5d',
    'us10y_level', 'us10y_change_5d', 'us10y_change_20d',
    'day_of_month', 'is_expiry_week',
    'days_to_monthly_expiry',
]


def compute_all_features(df, dhan_features=None, data_start="2014-01-01",
                          data_end="2025-12-31"):
    """
    Compute the full feature matrix from merged_daily data.
    This is the EXACT logic from the backtest's build_feature_matrix().

    Args:
        df: DataFrame with DatetimeIndex and columns:
            nifty_open, nifty_high, nifty_low, nifty_close, nifty_volume,
            india_vix, sp500_close, us_vix, dxy_close, us10y_yield
        dhan_features: Optional DataFrame of Dhan-derived features (from load_dhan_options_features)
        data_start, data_end: Date range for expiry computation

    Returns:
        (df_with_features, feature_col_names)
        - df_with_features has all features + targets as columns
        - feature_col_names is the ordered list of feature column names
    """
    df = df.copy()

    # ── 1. Trend & Momentum ──
    close = df['nifty_close']

    df['nifty_return_5d'] = close.pct_change(5)
    df['nifty_return_10d'] = close.pct_change(10)
    df['nifty_return_20d'] = close.pct_change(20)

    df['nifty_50dma'] = close.rolling(50).mean()
    df['nifty_200dma'] = close.rolling(200).mean()
    df['nifty_distance_50dma'] = (close - df['nifty_50dma']) / df['nifty_50dma'] * 100
    df['nifty_distance_200dma'] = (close - df['nifty_200dma']) / df['nifty_200dma'] * 100
    df['golden_cross'] = (df['nifty_50dma'] > df['nifty_200dma']).astype(int)

    df['rsi_14'] = ta.momentum.RSIIndicator(close, window=14).rsi()

    # Higher highs (weekly) — exact backtest logic
    weekly_high = df['nifty_high'].resample('W').max()
    weekly_hh = (weekly_high > weekly_high.shift(1)).astype(int)
    weekly_hh_count = weekly_hh.rolling(5, min_periods=1).sum()
    df['higher_highs_5w'] = weekly_hh_count.reindex(df.index, method='ffill').fillna(0)

    # ── 2. Volatility ──
    df['vix_change_5d'] = df['india_vix'] - df['india_vix'].shift(5)

    df['vix_percentile_252d'] = df['india_vix'].rolling(252, min_periods=60).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    )

    log_returns = np.log(close / close.shift(1))
    df['realized_vol_20d'] = log_returns.rolling(20).std() * np.sqrt(252) * 100

    df['variance_risk_premium'] = df['india_vix'] - df['realized_vol_20d']

    # VIX rising streak — exact backtest loop
    vix = df['india_vix']
    vix_rising = (vix > vix.shift(1)).astype(int)
    vix_streak = pd.Series(0, index=df.index, dtype=int)
    for i in range(1, len(df)):
        if vix_rising.iloc[i] == 1:
            vix_streak.iloc[i] = vix_streak.iloc[i - 1] + 1
        else:
            vix_streak.iloc[i] = 0
    df['vix_rising_streak'] = vix_streak

    # ── 3. Sentiment ──
    df['pcr_proxy'] = df['vix_change_5d'] / (df['nifty_return_5d'] * 100 + 0.001)

    # ── 4. Global Context ──
    if df['sp500_close'].notna().sum() > 100:
        df['sp500_return_5d'] = df['sp500_close'].pct_change(5)
    else:
        df['sp500_return_5d'] = 0.0

    if df['us_vix'].notna().sum() > 100:
        df['us_vix_change_5d'] = df['us_vix'] - df['us_vix'].shift(5)
    else:
        df['us_vix_change_5d'] = 0.0

    df['dxy_level'] = df['dxy_close']
    if df['dxy_close'].notna().sum() > 100:
        df['dxy_change_5d'] = df['dxy_close'] - df['dxy_close'].shift(5)
    else:
        df['dxy_change_5d'] = 0.0

    if 'us10y_yield' in df.columns and df['us10y_yield'].notna().sum() > 100:
        df['us10y_level'] = df['us10y_yield']
        df['us10y_change_5d'] = df['us10y_yield'] - df['us10y_yield'].shift(5)
        df['us10y_change_20d'] = df['us10y_yield'] - df['us10y_yield'].shift(20)
    else:
        df['us10y_level'] = 0.0
        df['us10y_change_5d'] = 0.0
        df['us10y_change_20d'] = 0.0

    # ── 5. Calendar ──
    df['day_of_month'] = df.index.day
    df['month'] = df.index.month
    df['is_volatile_month'] = df['month'].isin([9, 10]).astype(int)

    expiries = _get_monthly_expiries(data_start, data_end)
    trading_dates = df.index

    days_to_expiry = []
    is_expiry_week = []
    for date in df.index:
        next_exp = None
        for exp in expiries:
            if exp >= date:
                next_exp = exp
                break
        if next_exp is not None:
            mask = (trading_dates >= date) & (trading_dates <= next_exp)
            dte = mask.sum()
            days_to_expiry.append(dte)
            is_expiry_week.append(1 if dte <= 5 else 0)
        else:
            days_to_expiry.append(30)
            is_expiry_week.append(0)

    df['days_to_monthly_expiry'] = days_to_expiry
    df['is_expiry_week'] = is_expiry_week

    # ── 6. Dhan Options-Derived Features ──
    dhan_feature_cols = []
    if dhan_features is not None and len(dhan_features) > 0:
        for col in dhan_features.columns:
            df[col] = dhan_features[col].reindex(df.index, method='ffill')
            dhan_feature_cols.append(col)

    # ── 7. Targets (only for backtest — live callers ignore these) ──
    close_values = df['nifty_close'].values
    max_drawdown_30d = np.full(len(df), np.nan)
    for i in range(len(df) - 1):
        end_idx = min(i + 31, len(df))
        future_window = close_values[i + 1:end_idx]
        if len(future_window) > 0:
            min_future = future_window.min()
            max_drawdown_30d[i] = (min_future - close_values[i]) / close_values[i]
    df['max_drawdown_30d'] = max_drawdown_30d
    df['label_binary'] = (df['max_drawdown_30d'] < -0.05).astype(int)
    df['label'] = df['label_binary']

    high_values = df['nifty_high'].values if 'nifty_high' in df.columns else close_values
    max_rally_10d = np.full(len(df), np.nan)
    for i in range(len(df) - 1):
        end_idx = min(i + 11, len(df))
        future_highs = high_values[i + 1:end_idx]
        if len(future_highs) > 0:
            max_future_high = future_highs.max()
            max_rally_10d[i] = (max_future_high - close_values[i]) / close_values[i]
    df['max_rally_10d'] = max_rally_10d

    # ── 8. Cleanup ──
    feature_cols = BASE_FEATURE_COLS + dhan_feature_cols

    keep_cols = feature_cols + [
        'label', 'label_binary', 'max_drawdown_30d', 'max_rally_10d',
        'nifty_close', 'nifty_open', 'nifty_high', 'nifty_low', 'nifty_volume',
        'nifty_50dma', 'nifty_200dma',
    ]
    df = df[[c for c in keep_cols if c in df.columns]]

    required_cols = BASE_FEATURE_COLS + ['max_drawdown_30d']
    df = df.dropna(subset=[c for c in required_cols if c in df.columns])

    for col in dhan_feature_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    return df, feature_cols


def compute_features_for_date(merged_daily_df, target_date=None,
                                dhan_features=None,
                                data_start="2014-01-01", data_end="2026-12-31"):
    """
    Compute the feature vector for a single date using the EXACT same logic
    as the backtest. This is what the live system must call.

    Args:
        merged_daily_df: Full history DataFrame with DatetimeIndex and columns:
            nifty_open, nifty_high, nifty_low, nifty_close, nifty_volume,
            india_vix, sp500_close, us_vix, dxy_close, us10y_yield
        target_date: The date to compute features for (default: last row).
                     Must exist in merged_daily_df.index.
        dhan_features: Optional DataFrame from load_dhan_options_features()
        data_start, data_end: For expiry list generation.

    Returns:
        dict of {feature_name: value} with all 37 training features,
        ordered to match the model's expected input.
        Returns None if target_date has NaN in required features.
    """
    if target_date is not None:
        target_date = pd.Timestamp(target_date)
        # Only keep data up to and including target_date
        df = merged_daily_df.loc[:target_date].copy()
    else:
        df = merged_daily_df.copy()
        target_date = df.index[-1]

    if target_date not in df.index:
        raise ValueError(f"target_date {target_date} not found in merged_daily_df")

    # ── Compute all features using the exact same vectorized code ──
    close = df['nifty_close']

    # Trend & momentum
    nifty_return_5d = close.pct_change(5)
    nifty_return_10d = close.pct_change(10)
    nifty_return_20d = close.pct_change(20)

    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    nifty_distance_50dma = (close - sma50) / sma50 * 100
    nifty_distance_200dma = (close - sma200) / sma200 * 100
    golden_cross = (sma50 > sma200).astype(int)

    rsi_14 = ta.momentum.RSIIndicator(close, window=14).rsi()

    # Higher highs — exact backtest logic
    weekly_high = df['nifty_high'].resample('W').max()
    weekly_hh = (weekly_high > weekly_high.shift(1)).astype(int)
    weekly_hh_count = weekly_hh.rolling(5, min_periods=1).sum()
    higher_highs_5w = weekly_hh_count.reindex(df.index, method='ffill').fillna(0)

    # Volatility
    india_vix = df['india_vix']
    vix_change_5d = india_vix - india_vix.shift(5)

    vix_percentile_252d = india_vix.rolling(252, min_periods=60).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    )

    log_returns = np.log(close / close.shift(1))
    realized_vol_20d = log_returns.rolling(20).std() * np.sqrt(252) * 100

    variance_risk_premium = india_vix - realized_vol_20d

    # VIX rising streak — exact loop
    vix_rising = (india_vix > india_vix.shift(1)).astype(int)
    vix_streak = pd.Series(0, index=df.index, dtype=int)
    for i in range(1, len(df)):
        if vix_rising.iloc[i] == 1:
            vix_streak.iloc[i] = vix_streak.iloc[i - 1] + 1
        else:
            vix_streak.iloc[i] = 0

    # Sentiment
    pcr_proxy = vix_change_5d / (nifty_return_5d * 100 + 0.001)

    # Global context
    if df['sp500_close'].notna().sum() > 100:
        sp500_return_5d = df['sp500_close'].pct_change(5)
    else:
        sp500_return_5d = pd.Series(0.0, index=df.index)

    us_vix_series = df['us_vix']
    if us_vix_series.notna().sum() > 100:
        us_vix_change_5d = us_vix_series - us_vix_series.shift(5)
    else:
        us_vix_change_5d = pd.Series(0.0, index=df.index)

    dxy_level = df['dxy_close']
    if df['dxy_close'].notna().sum() > 100:
        dxy_change_5d = df['dxy_close'] - df['dxy_close'].shift(5)
    else:
        dxy_change_5d = pd.Series(0.0, index=df.index)

    if 'us10y_yield' in df.columns and df['us10y_yield'].notna().sum() > 100:
        us10y_level = df['us10y_yield']
        us10y_change_5d = df['us10y_yield'] - df['us10y_yield'].shift(5)
        us10y_change_20d = df['us10y_yield'] - df['us10y_yield'].shift(20)
    else:
        us10y_level = pd.Series(0.0, index=df.index)
        us10y_change_5d = pd.Series(0.0, index=df.index)
        us10y_change_20d = pd.Series(0.0, index=df.index)

    # Calendar
    day_of_month = target_date.day
    month = target_date.month
    is_volatile_month = 1 if month in [9, 10] else 0

    # Days to expiry — exact backtest logic (trading days, not calendar).
    # IMPORTANT: use the FULL merged_daily_df index as the trading calendar,
    # not the sliced df.index, because the backtest computes this over the
    # entire DataFrame and future trading dates must be visible.
    expiries = _get_monthly_expiries(data_start, data_end)
    all_trading_dates = merged_daily_df.index
    next_exp = None
    for exp in expiries:
        if exp >= target_date:
            next_exp = exp
            break
    if next_exp is not None:
        mask = (all_trading_dates >= target_date) & (all_trading_dates <= next_exp)
        dte = mask.sum()
    else:
        dte = 30
    days_to_monthly_expiry = dte
    is_expiry_week = 1 if dte <= 5 else 0

    # ── Extract values at target_date ──
    def _val(series):
        """Get value at target_date, handling NaN."""
        if isinstance(series, (int, float)):
            return series
        try:
            v = series.loc[target_date]
            if pd.isna(v):
                return np.nan
            return float(v)
        except KeyError:
            return np.nan

    features = {
        'nifty_return_5d': _val(nifty_return_5d),
        'nifty_return_10d': _val(nifty_return_10d),
        'nifty_return_20d': _val(nifty_return_20d),
        'nifty_distance_50dma': _val(nifty_distance_50dma),
        'nifty_distance_200dma': _val(nifty_distance_200dma),
        'golden_cross': _val(golden_cross),
        'rsi_14': _val(rsi_14),
        'higher_highs_5w': _val(higher_highs_5w),
        'india_vix': _val(india_vix),
        'vix_change_5d': _val(vix_change_5d),
        'vix_percentile_252d': _val(vix_percentile_252d),
        'realized_vol_20d': _val(realized_vol_20d),
        'variance_risk_premium': _val(variance_risk_premium),
        'vix_rising_streak': _val(vix_streak),
        'pcr_proxy': _val(pcr_proxy),
        'sp500_return_5d': _val(sp500_return_5d),
        'us_vix': _val(us_vix_series),
        'us_vix_change_5d': _val(us_vix_change_5d),
        'dxy_level': _val(dxy_level),
        'dxy_change_5d': _val(dxy_change_5d),
        'us10y_level': _val(us10y_level),
        'us10y_change_5d': _val(us10y_change_5d),
        'us10y_change_20d': _val(us10y_change_20d),
        'day_of_month': day_of_month,
        'month': month,
        'is_expiry_week': is_expiry_week,
        'days_to_monthly_expiry': days_to_monthly_expiry,
        'is_volatile_month': is_volatile_month,
    }

    # Dhan features
    dhan_feature_names = [
        'deep_otm_oi_ratio', 'deep_otm_oi_ratio_change_5d',
        'put_oi_buildup_ratio', 'put_volume_surge_ratio',
        'iv_skew_steepness', 'iv_skew_change_5d', 'atm_iv',
        'atm_iv_percentile_252d', 'atm_put_intraday_range',
    ]

    if dhan_features is not None and len(dhan_features) > 0:
        dhan_reindexed = dhan_features.reindex(df.index, method='ffill')
        for col in dhan_feature_names:
            if col in dhan_reindexed.columns:
                v = dhan_reindexed[col].loc[target_date] if target_date in dhan_reindexed.index else np.nan
                features[col] = float(v) if not pd.isna(v) else 0.0
            else:
                features[col] = 0.0
    else:
        for col in dhan_feature_names:
            features[col] = 0.0

    # Check for NaN in base features
    for col in BASE_FEATURE_COLS:
        v = features.get(col)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None  # This date doesn't have enough history for warmup

    # Ensure all values are native Python types (not numpy) for JSON serialization
    for k, v in features.items():
        if isinstance(v, (np.integer,)):
            features[k] = int(v)
        elif isinstance(v, (np.floating,)):
            features[k] = float(v)

    return features
