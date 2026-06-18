"""
Real option pricing from the Dhan option chain — replaces synthetic Black-Scholes
premiums with actual market quotes where available, and falls back to BS otherwise.

The whole point is to replicate the REAL trade environment: you sell at the bid and
buy at the ask, on the actual option legs. When the chain can't give a clean quote
(illiquid far-OTM wing, missing strike, chain unavailable), we fall back to the BS
estimate and **flag it loudly** — the caller surfaces that in the log, the trade
record, and the daily email so a fallback is never silent.

Returned `source` values:
  "real"        — every leg priced from real bid/ask quotes.
  "real_ltp"    — priced from real quotes but ≥1 leg used last-traded price (no
                  depth) — real, but the fill is less certain (illiquid).
  "bs_fallback" — at least one leg had no usable quote → BS used for the whole
                  spread. HIGHLIGHT THIS.
"""

import logging

logger = logging.getLogger(__name__)

# Legs per trade type: (option_type, action, strike_key_in_strikes_dict)
_SPREAD_LEGS = {
    "bull_put": [("pe", "sell", "sell_strike"), ("pe", "buy", "buy_strike")],
    "iron_condor": [
        ("pe", "sell", "sell_strike"), ("pe", "buy", "buy_strike"),
        ("ce", "sell", "ic_call_sell"), ("ce", "buy", "ic_call_buy"),
    ],
    "bear_call": [("ce", "sell", "ic_call_sell"), ("ce", "buy", "ic_call_buy")],
}


def parse_chain(chain_response: dict) -> dict:
    """
    Normalise a Dhan /optionchain response into
    {strike_float: {"ce": {...}, "pe": {...}}} with ltp/bid/ask/iv/oi per leg.

    Defensive about nesting (`data.oc` vs `oc`) and field names so a schema tweak
    degrades to a fallback rather than an exception.
    """
    out = {}
    if not isinstance(chain_response, dict):
        return out
    data = chain_response.get("data", chain_response)
    oc = data.get("oc") or data.get("option_chain") or {}
    if not isinstance(oc, dict):
        return out

    def num(d, *keys):
        for k in keys:
            v = d.get(k)
            if v is not None:
                try:
                    return float(v)
                except (TypeError, ValueError):
                    continue
        return None

    for strike_str, legs in oc.items():
        try:
            strike = float(strike_str)
        except (TypeError, ValueError):
            continue
        if not isinstance(legs, dict):
            continue
        entry = {}
        for opt in ("ce", "pe"):
            leg = legs.get(opt) or legs.get(opt.upper())
            if not isinstance(leg, dict):
                continue
            entry[opt] = {
                "ltp": num(leg, "last_price", "ltp", "lastPrice"),
                "bid": num(leg, "top_bid_price", "bid", "bidPrice"),
                "ask": num(leg, "top_ask_price", "ask", "askPrice"),
                "iv": num(leg, "implied_volatility", "iv"),
                "oi": num(leg, "oi", "openInterest"),
            }
        if entry:
            out[strike] = entry
    return out


def _match_strike(parsed: dict, target: float):
    """Find the chain entry for *target* strike (keys are exact-ish floats)."""
    if target in parsed:
        return parsed[target]
    for k in parsed:
        if abs(k - target) < 0.5:
            return parsed[k]
    return None


def _leg_fill_price(parsed: dict, strike: float, opt: str, action: str,
                    allow_ltp: bool = True):
    """
    Price one leg at a realistic fill: SELL → bid (you receive), BUY → ask (you pay).
    Falls back to LTP if there's no depth. Returns (price, price_source) or (None, reason).
    """
    entry = _match_strike(parsed, strike)
    if not entry or opt not in entry:
        return None, f"missing_{opt}_{int(strike)}"
    leg = entry[opt]
    want = "bid" if action == "sell" else "ask"
    px = leg.get(want)
    if px and px > 0:
        return px, want
    if allow_ltp:
        ltp = leg.get("ltp")
        if ltp and ltp > 0:
            return ltp, "ltp"
    return None, f"no_quote_{opt}_{int(strike)}"


def price_spread_real(chain_response: dict, trade_type: str, strikes: dict,
                      bs_credit: float, *, allow_ltp_fallback: bool = True) -> dict:
    """
    Compute the real net credit for *trade_type* from the chain.

    Net credit = Σ(sell-leg fill) − Σ(buy-leg fill), with sells at bid and buys at ask.
    If any leg has no usable quote, fall back to *bs_credit* for the whole spread and
    flag `source="bs_fallback"`. Always returns both real_credit and bs_credit so the
    gap is logged on every trade.
    """
    result = {
        "credit": bs_credit,
        "source": "bs_fallback",
        "fallback_reason": None,
        "real_credit": None,
        "bs_credit": round(bs_credit, 2),
        "legs": [],
    }

    legs = _SPREAD_LEGS.get(trade_type)
    if not legs:
        result["fallback_reason"] = f"unsupported_trade_type:{trade_type}"
        return result

    parsed = parse_chain(chain_response)
    if not parsed:
        result["fallback_reason"] = "chain_unavailable_or_unparseable"
        return result

    net = 0.0
    used_ltp = False
    leg_detail = []
    for opt, action, key in legs:
        strike = strikes.get(key)
        if strike is None:
            result["fallback_reason"] = f"missing_strike:{key}"
            return result
        px, src = _leg_fill_price(parsed, float(strike), opt, action, allow_ltp_fallback)
        if px is None:
            result["fallback_reason"] = src  # e.g. no_quote_pe_22650
            result["legs"] = leg_detail
            return result
        if src == "ltp":
            used_ltp = True
        net += px if action == "sell" else -px
        leg_detail.append({"opt": opt, "action": action, "strike": float(strike),
                           "price": round(px, 2), "px_source": src})

    real_credit = max(net, 0.0)
    result.update(
        credit=real_credit,
        source="real_ltp" if used_ltp else "real",
        real_credit=round(real_credit, 2),
        legs=leg_detail,
    )
    return result


def value_spread_real(chain_response: dict, trade_type: str, strikes: dict,
                      bs_value: float, *, allow_ltp_fallback: bool = True) -> dict:
    """
    Real cost-to-CLOSE the spread (the exit/MTM value) — the mirror of entry:
    you BUY BACK the short legs at the ask and SELL the long legs at the bid.

    Cost = Σ(short-leg ask) − Σ(long-leg bid). Using the same basis as the real
    entry credit means a freshly-opened trade shows only the round-trip bid/ask
    spread as a loss (realistic), not a synthetic BS mis-mark. Falls back to
    *bs_value* (flagged) when any leg has no usable quote.
    """
    result = {
        "value": bs_value, "source": "bs_fallback", "fallback_reason": None,
        "real_value": None, "bs_value": round(bs_value, 2), "legs": [],
    }
    legs = _SPREAD_LEGS.get(trade_type)
    if not legs:
        result["fallback_reason"] = f"unsupported_trade_type:{trade_type}"
        return result
    parsed = parse_chain(chain_response)
    if not parsed:
        result["fallback_reason"] = "chain_unavailable_or_unparseable"
        return result

    cost = 0.0
    used_ltp = False
    leg_detail = []
    for opt, entry_action, key in legs:
        strike = strikes.get(key)
        if strike is None:
            result["fallback_reason"] = f"missing_strike:{key}"
            return result
        # Closing is the opposite of entry: short→buy back (ask), long→sell (bid).
        exit_action = "buy" if entry_action == "sell" else "sell"
        px, src = _leg_fill_price(parsed, float(strike), opt, exit_action, allow_ltp_fallback)
        if px is None:
            result["fallback_reason"] = src
            return result
        if src == "ltp":
            used_ltp = True
        cost += px if exit_action == "buy" else -px
        leg_detail.append({"opt": opt, "action": exit_action, "strike": float(strike),
                           "price": round(px, 2), "px_source": src})

    real_value = max(cost, 0.0)
    result.update(value=real_value, source="real_ltp" if used_ltp else "real",
                  real_value=round(real_value, 2), legs=leg_detail)
    return result


def log_pricing(version: str, trade_type: str, pricing: dict) -> None:
    """Emit a HIGHLIGHTED log line — loud on BS fallback, plain on real."""
    src = pricing.get("source")
    real, bs = pricing.get("real_credit"), pricing.get("bs_credit")
    if src == "bs_fallback":
        logger.warning(
            "⚠️ BS-FALLBACK PRICING [%s %s]: no real quote (%s) — using BS credit ₹%s/unit. "
            "Real fill unknown; paper P&L is synthetic for this trade.",
            version, trade_type, pricing.get("fallback_reason"), bs,
        )
    else:
        gap = (real - bs) if (real is not None and bs is not None) else None
        flag = " (LTP, no depth — illiquid)" if src == "real_ltp" else ""
        logger.info(
            "REAL PRICING [%s %s]%s: real ₹%s vs BS ₹%s/unit (gap %s)",
            version, trade_type, flag, real, bs,
            f"{gap:+.2f}" if gap is not None else "n/a",
        )
