"""
Option pricing utilities for paper trades.
Uses Black-Scholes for theoretical prices, Dhan for live IV/prices.
"""

import logging
import math
from datetime import datetime, date, timedelta
from scipy.stats import norm
from core.timezone import today_ist

logger = logging.getLogger(__name__)


# Market premium markup — matches backtest B-S fallback tier
MARKET_PREMIUM = 1.25  # 25% markup on all option prices
# Slippage on entry — matches backtest (2% adverse fill)
ENTRY_SLIPPAGE = 0.02


def _put_sigma_with_skew(S: float, K: float, base_iv: float) -> float:
    """
    Compute put IV with skew adjustment matching backtest Tier 2:
      moneyness = (S - K) / S
      skew = 1.0 + moneyness * 3.0
      sigma = base_iv * max(skew, 1.0)
    """
    moneyness = (S - K) / S if S > 0 else 0
    skew_adjustment = 1.0 + moneyness * 3.0
    return base_iv * max(skew_adjustment, 1.0)


def _call_sigma_with_skew(S: float, K: float, base_iv: float) -> float:
    """
    Compute call IV with skew adjustment matching backtest Tier 2:
      moneyness = (K - S) / S
      sigma = base_iv * (1.0 + moneyness * 1.5)
      sigma = max(sigma, base_iv * 0.8)
    """
    moneyness = (K - S) / S if S > 0 else 0
    sigma = base_iv * (1.0 + moneyness * 1.5)
    return max(sigma, base_iv * 0.8)


def bs_put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Black-Scholes put option price with IV skew and market premium.
    Matches backtest Tier 2 (B-S fallback) pricing.
    """
    # Apply skew
    base_iv = max(sigma, 0.05)
    adj_sigma = _put_sigma_with_skew(S, K, base_iv)

    if T <= 0 or adj_sigma <= 0:
        return max(K - S, 0) * MARKET_PREMIUM

    d1 = (math.log(S / K) + (r + 0.5 * adj_sigma**2) * T) / (adj_sigma * math.sqrt(T))
    d2 = d1 - adj_sigma * math.sqrt(T)

    price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return max(price, 0) * MARKET_PREMIUM


def bs_call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Black-Scholes call option price with IV skew and market premium.
    Matches backtest Tier 2 (B-S fallback) pricing.
    """
    # Apply skew
    base_iv = max(sigma, 0.05)
    adj_sigma = _call_sigma_with_skew(S, K, base_iv)

    if T <= 0 or adj_sigma <= 0:
        return max(S - K, 0) * MARKET_PREMIUM

    d1 = (math.log(S / K) + (r + 0.5 * adj_sigma**2) * T) / (adj_sigma * math.sqrt(T))
    d2 = d1 - adj_sigma * math.sqrt(T)

    price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    return max(price, 0) * MARKET_PREMIUM


def price_bull_put_spread(S: float, sell_strike: float, buy_strike: float,
                           T: float, r: float, sigma: float,
                           apply_slippage: bool = False) -> float:
    """
    Price a bull put spread (sell put at higher strike, buy put at lower strike).
    Returns net credit received per share.
    If apply_slippage=True, applies 2% adverse slippage (less credit at entry).
    """
    sell_put = bs_put_price(S, sell_strike, T, r, sigma)
    buy_put = bs_put_price(S, buy_strike, T, r, sigma)
    credit = sell_put - buy_put
    credit = max(credit, 0)
    if apply_slippage:
        credit *= (1 - ENTRY_SLIPPAGE)  # Receive slightly less credit
    return credit


def price_iron_condor(S: float, put_sell: float, put_buy: float,
                       call_sell: float, call_buy: float,
                       T: float, r: float, sigma: float,
                       apply_slippage: bool = False) -> float:
    """
    Price an iron condor spread.
    Returns net credit received per share.
    If apply_slippage=True, applies 2% adverse slippage (less credit at entry).
    """
    put_credit = bs_put_price(S, put_sell, T, r, sigma) - \
                 bs_put_price(S, put_buy, T, r, sigma)
    call_credit = bs_call_price(S, call_sell, T, r, sigma) - \
                  bs_call_price(S, call_buy, T, r, sigma)
    total_credit = put_credit + call_credit
    total_credit = max(total_credit, 0)
    if apply_slippage:
        total_credit *= (1 - ENTRY_SLIPPAGE)
    return total_credit


def price_bear_put_debit(S: float, buy_strike: float, sell_strike: float,
                          T: float, r: float, sigma: float,
                          apply_slippage: bool = False) -> float:
    """
    Price a bear put debit spread (buy higher-strike put, sell lower-strike put).
    Returns net debit paid per share.
    buy_strike > sell_strike (buy is closer to ATM).
    If apply_slippage=True, applies 2% adverse slippage (more debit at entry).
    """
    buy_put = bs_put_price(S, buy_strike, T, r, sigma)
    sell_put = bs_put_price(S, sell_strike, T, r, sigma)
    debit = buy_put - sell_put
    debit = max(debit, 0)
    if apply_slippage:
        debit *= (1 + ENTRY_SLIPPAGE)  # Pay slightly more debit
    return debit


def price_bear_call_spread(S: float, sell_strike: float, buy_strike: float,
                            T: float, r: float, sigma: float,
                            apply_slippage: bool = False) -> float:
    """
    Price a bear call spread (sell call at lower strike, buy call at higher strike).
    Returns net credit received per share.
    """
    sell_call = bs_call_price(S, sell_strike, T, r, sigma)
    buy_call = bs_call_price(S, buy_strike, T, r, sigma)
    credit = sell_call - buy_call
    credit = max(credit, 0)
    if apply_slippage:
        credit *= (1 - ENTRY_SLIPPAGE)
    return credit


def compute_spread_value(trade_type: str, S: float,
                         sell_strike: float, buy_strike: float,
                         ic_call_sell: float = None, ic_call_buy: float = None,
                         T: float = 0.03, r: float = 0.07,
                         sigma: float = 0.15) -> float:
    """
    Compute current value of a spread given current spot price.
    For credit spreads: what it costs to close (buy back).
    For bear_put_debit: current value of the debit spread (what you could sell for).
    """
    if trade_type == "bull_put":
        return price_bull_put_spread(S, sell_strike, buy_strike, T, r, sigma)
    elif trade_type == "iron_condor" and ic_call_sell and ic_call_buy:
        return price_iron_condor(S, sell_strike, buy_strike,
                                  ic_call_sell, ic_call_buy, T, r, sigma)
    elif trade_type == "bear_put_debit":
        # For bear debit: buy_strike is the higher strike (bought put)
        # sell_strike is the lower strike (sold put)
        return price_bear_put_debit(S, buy_strike, sell_strike, T, r, sigma)
    elif trade_type == "bear_call":
        # Bear call credit: sell_strike=ic_call_sell, buy_strike=ic_call_buy
        call_sell = ic_call_sell or sell_strike
        call_buy = ic_call_buy or buy_strike
        return price_bear_call_spread(S, call_sell, call_buy, T, r, sigma)
    else:
        logger.warning(f"Unknown trade type: {trade_type}")
        return 0.0


def select_strikes(spot: float, trade_type: str, version_cfg: dict) -> dict:
    """
    Select option strikes based on spot price and OTM percentages.
    Returns dict of strike prices rounded to nearest 50.
    """
    lot_size = 25

    def round_strike(price: float) -> float:
        return round(price / 50) * 50

    if trade_type == "bull_put":
        sell_strike = round_strike(spot * (1 - version_cfg.get("BULL_OTM_SELL", 0.03)))
        buy_strike = round_strike(spot * (1 - version_cfg.get("BULL_OTM_BUY", 0.055)))
        return {
            "sell_strike": sell_strike,
            "buy_strike": buy_strike,
            "ic_call_sell": None,
            "ic_call_buy": None,
        }
    elif trade_type == "iron_condor":
        put_sell = round_strike(spot * (1 - version_cfg.get("IC_PUT_OTM_SELL", 0.03)))
        put_buy = round_strike(spot * (1 - version_cfg.get("IC_PUT_OTM_BUY", 0.055)))
        call_sell_otm = version_cfg.get("IC_CALL_OTM_SELL", 0.04)
        call_buy_otm = version_cfg.get("IC_CALL_OTM_BUY", 0.065)
        call_sell = round_strike(spot * (1 + call_sell_otm))
        call_buy = round_strike(spot * (1 + call_buy_otm))
        return {
            "sell_strike": put_sell,
            "buy_strike": put_buy,
            "ic_call_sell": call_sell,
            "ic_call_buy": call_buy,
        }
    elif trade_type == "bear_put_debit":
        # Bear put debit: buy near-ATM put (higher strike), sell OTM put (lower strike)
        buy_otm = version_cfg.get("BEAR_PUT_BUY_OTM", 0.01)
        sell_otm = version_cfg.get("BEAR_PUT_SELL_OTM", 0.04)
        buy_strike = round_strike(spot * (1 - buy_otm))   # near ATM
        sell_strike = round_strike(spot * (1 - sell_otm))  # further OTM
        return {
            "sell_strike": sell_strike,
            "buy_strike": buy_strike,
            "ic_call_sell": None,
            "ic_call_buy": None,
        }
    elif trade_type == "bear_call":
        # Bear call credit: sell OTM call, buy further OTM call
        call_sell_otm = version_cfg.get("BEAR_CALL_OTM_SELL", version_cfg.get("IC_CALL_OTM_SELL", 0.04))
        call_buy_otm = version_cfg.get("BEAR_CALL_OTM_BUY", version_cfg.get("IC_CALL_OTM_BUY", 0.065))
        call_sell = round_strike(spot * (1 + call_sell_otm))
        call_buy = round_strike(spot * (1 + call_buy_otm))
        return {
            "sell_strike": None,
            "buy_strike": None,
            "ic_call_sell": call_sell,
            "ic_call_buy": call_buy,
        }
    else:
        return {
            "sell_strike": None,
            "buy_strike": None,
            "ic_call_sell": None,
            "ic_call_buy": None,
        }


def get_next_weekly_expiry(from_date: date = None) -> date:
    """
    Get next weekly Nifty expiry (Thursday).
    If today is Thursday, returns next Thursday.
    """
    if from_date is None:
        from_date = today_ist()

    days_until_thursday = (3 - from_date.weekday()) % 7
    if days_until_thursday == 0:
        days_until_thursday = 7  # next Thursday, not today

    return from_date + timedelta(days=days_until_thursday)


def compute_dte(entry_date: date, expiry_date: date) -> int:
    """Compute days to expiry."""
    return (expiry_date - entry_date).days


def compute_time_to_expiry_years(entry_date: date, expiry_date: date) -> float:
    """Compute time to expiry in years for BS formula."""
    dte = (expiry_date - entry_date).days
    return max(dte / 365.0, 1 / 365.0)
