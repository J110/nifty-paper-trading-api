"""Option pricing module — extracted from nifty_options_model/src/option_pricing.py.

Two-tier hybrid pricing: Dhan IV skew (primary) + B-S fallback.
Self-contained: no dependencies on nifty_options_model/.
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import norm

RISK_FREE_RATE = 0.07


class OptionPricer:
    """Two-tier option pricer supporting puts and calls."""

    def __init__(self, iv_skew_path=None):
        self.iv_skew = None
        self.dhan_price_count = 0
        self.bs_fallback_count = 0

        if iv_skew_path and os.path.exists(iv_skew_path):
            try:
                self.iv_skew = pd.read_parquet(iv_skew_path)
                if "date" in self.iv_skew.columns:
                    self.iv_skew["date"] = pd.to_datetime(self.iv_skew["date"])
                    self.iv_skew = self.iv_skew.set_index("date")
                elif not isinstance(self.iv_skew.index, pd.DatetimeIndex):
                    self.iv_skew.index = pd.to_datetime(self.iv_skew.index)
                print(f"  Tier 1 loaded: Dhan IV skew curves for {len(self.iv_skew)} trading days")
            except Exception as e:
                print(f"  WARNING: Failed to load IV skew params: {e}")

    def get_put_price(self, spot, strike, dte_days, vix, trade_date=None, expiry_date=None):
        """Get put option price using two-tier hierarchy. Returns: (price, source)"""
        if self.iv_skew is not None and trade_date is not None:
            td = pd.Timestamp(trade_date)
            skew_row = self._find_nearest_skew(td)
            if skew_row is not None:
                moneyness = (spot - strike) / spot
                a = skew_row["skew_a"]
                b = skew_row["skew_b"]
                c = skew_row["skew_c"]
                iv_extrapolated = a + b * moneyness + c * moneyness ** 2
                iv_extrapolated = min(max(iv_extrapolated, 5), 100)
                price = self._bs_put_price(spot, strike, dte_days, iv_extrapolated / 100.0)
                price *= 1.10
                self.dhan_price_count += 1
                return price, "dhan"

        price = self._bs_put_with_fixed_skew(spot, strike, dte_days, vix)
        self.bs_fallback_count += 1
        return price, "bs"

    def _bs_put_price(self, S, K, dte_days, sigma):
        T = max(dte_days / 365.0, 1.0 / 365.0)
        r = RISK_FREE_RATE
        if sigma <= 0:
            sigma = 0.01
        d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return max(K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1), 0)

    def _bs_put_with_fixed_skew(self, S, K, dte_days, vix):
        T = max(dte_days / 365.0, 1.0 / 365.0)
        r = RISK_FREE_RATE
        base_iv = max(vix / 100.0, 0.05)
        moneyness = (S - K) / S
        skew_adjustment = 1.0 + (moneyness * 3.0)
        sigma = base_iv * max(skew_adjustment, 1.0)
        if sigma <= 0:
            sigma = 0.01
        d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return max(K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1), 0) * 1.25

    def get_call_price(self, spot, strike, dte_days, vix, trade_date=None, expiry_date=None):
        """Get call option price. Returns: (price, source)"""
        if self.iv_skew is not None and trade_date is not None:
            td = pd.Timestamp(trade_date)
            skew_row = self._find_nearest_skew(td)
            if skew_row is not None:
                call_moneyness = (strike - spot) / spot
                a = skew_row["skew_a"]
                b = skew_row["skew_b"]
                c = skew_row["skew_c"]
                iv_call = a + b * call_moneyness * 0.5 + c * (call_moneyness * 0.5) ** 2
                iv_call *= 0.90
                iv_call = min(max(iv_call, 5), 100)
                price = self._bs_call_price(spot, strike, dte_days, iv_call / 100.0)
                price *= 1.10
                self.dhan_price_count += 1
                return price, "dhan"

        price = self._bs_call_with_fixed_skew(spot, strike, dte_days, vix)
        self.bs_fallback_count += 1
        return price, "bs"

    def _bs_call_price(self, S, K, dte_days, sigma):
        T = max(dte_days / 365.0, 1.0 / 365.0)
        r = RISK_FREE_RATE
        if sigma <= 0:
            sigma = 0.01
        d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return max(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2), 0)

    def _bs_call_with_fixed_skew(self, S, K, dte_days, vix):
        T = max(dte_days / 365.0, 1.0 / 365.0)
        r = RISK_FREE_RATE
        base_iv = max(vix / 100.0, 0.05)
        call_moneyness = (K - S) / S
        sigma = base_iv * (1.0 + call_moneyness * 1.5)
        sigma = max(sigma, base_iv * 0.8)
        if sigma <= 0:
            sigma = 0.01
        d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return max(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2), 0) * 1.25

    def _find_nearest_skew(self, td):
        td = td.normalize()
        if td in self.iv_skew.index:
            return self.iv_skew.loc[td]
        mask = abs(self.iv_skew.index - td) <= pd.Timedelta(days=3)
        if mask.any():
            candidates = self.iv_skew.index[mask]
            nearest = candidates[abs(candidates - td).argmin()]
            return self.iv_skew.loc[nearest]
        return None

    def print_pricing_summary(self):
        total = self.dhan_price_count + self.bs_fallback_count
        if total > 0:
            dhan_pct = self.dhan_price_count / total * 100
            print(f"\n  Pricing Summary:")
            print(f"    Dhan IV skew (data-driven): {self.dhan_price_count:>6} ({dhan_pct:>5.1f}%)")
            print(f"    Fixed B-S fallback:         {self.bs_fallback_count:>6} ({100-dhan_pct:>5.1f}%)")


def price_bull_put_spread(pricer, spot, sell_strike, buy_strike, dte, vix,
                          trade_date=None, expiry_date=None):
    """Price a bull put spread (CREDIT). Returns net credit received."""
    sell_price, _ = pricer.get_put_price(spot, sell_strike, dte, vix, trade_date, expiry_date)
    buy_price, _ = pricer.get_put_price(spot, buy_strike, dte, vix, trade_date, expiry_date)
    return max(sell_price - buy_price, 0)


def price_bear_put_debit(pricer, spot, buy_strike, sell_strike, dte, vix,
                         trade_date=None, expiry_date=None):
    """Price a bear put debit spread (DEBIT). Returns net debit paid."""
    buy_price, _ = pricer.get_put_price(spot, buy_strike, dte, vix, trade_date, expiry_date)
    sell_price, _ = pricer.get_put_price(spot, sell_strike, dte, vix, trade_date, expiry_date)
    return max(buy_price - sell_price, 0)


def price_bear_call_spread(pricer, spot, sell_strike, buy_strike, dte, vix,
                           trade_date=None, expiry_date=None):
    """Price the call side of an iron condor (CREDIT). Returns net credit received."""
    sell_price, _ = pricer.get_call_price(spot, sell_strike, dte, vix, trade_date, expiry_date)
    buy_price, _ = pricer.get_call_price(spot, buy_strike, dte, vix, trade_date, expiry_date)
    return max(sell_price - buy_price, 0)


def compute_position_value(pricer, trade, current_spot, dte, current_vix, date):
    """Mark-to-market value of any open position."""
    if trade.trade_type == "bull_put":
        sell_val, _ = pricer.get_put_price(current_spot, trade.put_sell_strike, dte, current_vix, date)
        buy_val, _ = pricer.get_put_price(current_spot, trade.put_buy_strike, dte, current_vix, date)
        return sell_val - buy_val

    elif trade.trade_type == "bear_debit":
        buy_val, _ = pricer.get_put_price(current_spot, trade.put_buy_strike, dte, current_vix, date)
        sell_val, _ = pricer.get_put_price(current_spot, trade.put_sell_strike, dte, current_vix, date)
        return buy_val - sell_val

    elif trade.trade_type == "call_debit":
        buy_val, _ = pricer.get_call_price(current_spot, trade.call_buy_strike, dte, current_vix, date)
        sell_val, _ = pricer.get_call_price(current_spot, trade.call_sell_strike, dte, current_vix, date)
        return buy_val - sell_val

    elif trade.trade_type == "iron_condor":
        p_sell_val, _ = pricer.get_put_price(current_spot, trade.put_sell_strike, dte, current_vix, date)
        p_buy_val, _ = pricer.get_put_price(current_spot, trade.put_buy_strike, dte, current_vix, date)
        c_sell_val, _ = pricer.get_call_price(current_spot, trade.call_sell_strike, dte, current_vix, date)
        c_buy_val, _ = pricer.get_call_price(current_spot, trade.call_buy_strike, dte, current_vix, date)
        return (p_sell_val - p_buy_val) + (c_sell_val - c_buy_val)

    return 0
