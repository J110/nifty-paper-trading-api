"""Utility functions for backtest engine — extracted from nifty_options_model/src/utils.py.

Self-contained: no dependencies on nifty_options_model/.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def get_last_thursday(year, month):
    """Get the last Thursday of a given month (monthly expiry)."""
    if month == 12:
        next_month = datetime(year + 1, 1, 1)
    else:
        next_month = datetime(year, month + 1, 1)
    last_day = next_month - timedelta(days=1)
    while last_day.weekday() != 3:
        last_day -= timedelta(days=1)
    return last_day.date()


def get_monthly_expiries(start_date, end_date):
    """Generate all monthly expiry dates (last Thursday) in range."""
    expiries = []
    current = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    while current <= end:
        exp = get_last_thursday(current.year, current.month)
        if pd.Timestamp(exp) >= pd.Timestamp(start_date):
            expiries.append(pd.Timestamp(exp))
        if current.month == 12:
            current = pd.Timestamp(datetime(current.year + 1, 1, 1))
        else:
            current = pd.Timestamp(datetime(current.year, current.month + 1, 1))
    return expiries


def next_monthly_expiry(date, expiries):
    """Find the next monthly expiry on or after date."""
    date = pd.Timestamp(date)
    for exp in expiries:
        if exp >= date:
            return exp
    return None


def round_to_nearest(value, base=50):
    """Round a value to the nearest multiple of base."""
    return int(round(value / base) * base)


def trading_days_between(date1, date2, trading_dates):
    """Count trading days between two dates."""
    d1 = pd.Timestamp(date1)
    d2 = pd.Timestamp(date2)
    mask = (trading_dates >= min(d1, d2)) & (trading_dates <= max(d1, d2))
    return mask.sum()


def annualized_return(total_return, days):
    """Compute annualized return from total return and number of trading days."""
    if days <= 0:
        return 0.0
    years = days / 252.0
    if total_return <= -1:
        return -1.0
    return (1 + total_return) ** (1 / years) - 1


def max_drawdown(equity_series):
    """Compute maximum drawdown from an equity series."""
    peak = equity_series.cummax()
    drawdown = (equity_series - peak) / peak
    return drawdown.min()


def max_drawdown_recovery_days(equity_series):
    """Compute the longest recovery period (in trading days) from a drawdown."""
    peak = equity_series.cummax()
    in_drawdown = equity_series < peak
    if not in_drawdown.any():
        return 0
    max_recovery = 0
    current_recovery = 0
    for i in range(len(equity_series)):
        if equity_series.iloc[i] < peak.iloc[i]:
            current_recovery += 1
            max_recovery = max(max_recovery, current_recovery)
        else:
            current_recovery = 0
    return max_recovery


def sharpe_ratio(returns, risk_free_daily=0.07 / 252):
    """Compute annualized Sharpe ratio."""
    excess = returns - risk_free_daily
    if len(excess) == 0 or excess.std() == 0:
        return 0.0
    return np.sqrt(252) * excess.mean() / excess.std()


def find_next_weekly_expiry(date):
    """Find the next weekly Nifty expiry.

    NSE moved Nifty's weekly expiry from Thursday to Tuesday effective 2025-09-01,
    so use Tuesday (weekday 1) on/after the cutover and Thursday (3) before it. If
    *date* is the expiry day itself, roll to next week (don't enter on expiry day).
    NOTE: weekday-only (no holiday roll-back) — matches this module's original
    behaviour; the historical backtest did not holiday-adjust weekly expiries.
    """
    date = pd.Timestamp(date)
    target = 1 if date >= pd.Timestamp("2025-09-01") else 3  # Tue on/after cutover, else Thu
    days_ahead = target - date.weekday()
    if days_ahead <= 0:
        days_ahead += 7
    return date + timedelta(days=days_ahead)


def get_weekly_expiries(start_date, end_date):
    """Generate all weekly expiry dates (every Thursday) in range."""
    expiries = []
    current = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    days_ahead = (3 - current.weekday()) % 7
    current = current + timedelta(days=days_ahead)
    while current <= end:
        expiries.append(current)
        current += timedelta(days=7)
    return expiries


def find_next_monthly_expiry(date):
    """Find the next monthly expiry (last Thursday of the month) on or after date."""
    date = pd.Timestamp(date)
    exp = pd.Timestamp(get_last_thursday(date.year, date.month))
    if exp >= date:
        return exp
    if date.month == 12:
        return pd.Timestamp(get_last_thursday(date.year + 1, 1))
    return pd.Timestamp(get_last_thursday(date.year, date.month + 1))


def compute_realized_vol_5d(close_series, idx, window=5):
    """Compute 5-day annualized realized volatility at index position idx."""
    if idx < window:
        return None
    prices = close_series.iloc[max(0, idx - window):idx + 1]
    if len(prices) < 2:
        return None
    log_returns = np.log(prices / prices.shift(1)).dropna()
    if len(log_returns) == 0 or log_returns.std() == 0:
        return 0.0
    return log_returns.std() * np.sqrt(252) * 100
