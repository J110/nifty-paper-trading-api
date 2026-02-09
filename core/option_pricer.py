"""
Option pricing utilities for paper trades.
Uses Black-Scholes for theoretical prices, Dhan for live IV/prices.
"""

import logging
import math
from datetime import datetime, date, timedelta
from scipy.stats import norm

logger = logging.getLogger(__name__)


def bs_put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes put option price."""
    if T <= 0 or sigma <= 0:
        return max(K - S, 0)

    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return max(price, 0)


def bs_call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes call option price."""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0)

    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    return max(price, 0)


def price_bull_put_spread(S: float, sell_strike: float, buy_strike: float,
                           T: float, r: float, sigma: float) -> float:
    """
    Price a bull put spread (sell put at higher strike, buy put at lower strike).
    Returns net credit received per share.
    """
    sell_put = bs_put_price(S, sell_strike, T, r, sigma)
    buy_put = bs_put_price(S, buy_strike, T, r, sigma)
    credit = sell_put - buy_put
    return max(credit, 0)


def price_iron_condor(S: float, put_sell: float, put_buy: float,
                       call_sell: float, call_buy: float,
                       T: float, r: float, sigma: float) -> float:
    """
    Price an iron condor spread.
    Returns net credit received per share.
    """
    put_credit = bs_put_price(S, put_sell, T, r, sigma) - \
                 bs_put_price(S, put_buy, T, r, sigma)
    call_credit = bs_call_price(S, call_sell, T, r, sigma) - \
                  bs_call_price(S, call_buy, T, r, sigma)
    total_credit = put_credit + call_credit
    return max(total_credit, 0)


def compute_spread_value(trade_type: str, S: float,
                         sell_strike: float, buy_strike: float,
                         ic_call_sell: float = None, ic_call_buy: float = None,
                         T: float = 0.03, r: float = 0.07,
                         sigma: float = 0.15) -> float:
    """
    Compute current value of a spread given current spot price.
    This is what it would cost to close the position (buy back).
    """
    if trade_type == "bull_put":
        return price_bull_put_spread(S, sell_strike, buy_strike, T, r, sigma)
    elif trade_type == "iron_condor" and ic_call_sell and ic_call_buy:
        return price_iron_condor(S, sell_strike, buy_strike,
                                  ic_call_sell, ic_call_buy, T, r, sigma)
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
        from_date = date.today()

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
