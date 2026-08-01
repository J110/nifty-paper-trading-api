"""
Manages paper trade lifecycle:
1. Open trades when signals fire
2. Track current PnL using live option prices
3. Check exit conditions every 5 min
4. Close trades and record results
"""

import logging
import uuid
from datetime import datetime, date, timedelta
from typing import Optional

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from core.timezone import now_ist, today_ist
from config import (
    INITIAL_CAPITAL, NIFTY_LOT_SIZE, RISK_FREE_RATE,
    MARGIN_PER_LOT_BULL, MARGIN_PER_LOT_IC, MARGIN_PER_LOT_BEAR,
    VERSION_CONFIGS,
    BULL_OTM_SELL, BULL_OTM_BUY,
    MIN_TOTAL_CREDIT, MIN_ENTRY_DTE, USE_REAL_OPTION_PRICING,
    COMPOUND_SIZING, MAX_MARGIN_UTILISATION, EXPOSURE_MARGIN_PCT,
)
from core.option_pricer import (
    select_strikes, price_bull_put_spread, price_iron_condor,
    price_bear_put_debit, price_bear_call_spread,
    get_next_weekly_expiry, compute_time_to_expiry_years,
    compute_spread_value,
)
from core.real_option_pricer import price_spread_real, value_spread_real, log_pricing
from db.models import Trade, DailyPnl, Prediction

logger = logging.getLogger(__name__)


class TradeManager:
    """Manages paper trade lifecycle for all versions."""

    def __init__(self, dhan_client=None):
        self.dhan_client = dhan_client

    def _spread_value(self, trade, spot, vix, chain=None):
        """Cost-to-close the spread: real (short legs @ ask, long legs @ bid) from
        the chain when available, else Black-Scholes — the SAME basis as the real
        entry credit, so no entry-real/exit-BS mis-mark."""
        dte = (trade.expiry - today_ist()).days
        T = max(dte / 365.0, 1 / 365.0)
        sigma = vix / 100.0 if vix else 0.15
        bs = compute_spread_value(
            trade.trade_type, spot, trade.sell_strike, trade.buy_strike,
            trade.ic_call_sell, trade.ic_call_buy, T, RISK_FREE_RATE, sigma,
        )
        if USE_REAL_OPTION_PRICING and chain and not trade.is_bear_debit:
            v = value_spread_real(chain, trade.trade_type, {
                "sell_strike": trade.sell_strike, "buy_strike": trade.buy_strike,
                "ic_call_sell": trade.ic_call_sell, "ic_call_buy": trade.ic_call_buy,
            }, bs)
            return v["value"]
        return bs

    async def open_trade(
        self,
        signal: dict,
        version: str,
        spot: float,
        vix: float,
        db: AsyncSession,
        dhan_client=None,
        predicted_drawdown: float = None,
    ) -> Optional[dict]:
        """
        Construct and record a new paper trade (credit or bear debit).
        Returns trade dict or None if trade can't be opened.
        """
        if signal["signal"] == "no_trade" or signal["trade_type"] is None:
            return None

        cfg = VERSION_CONFIGS.get(version, {})
        trade_type = signal["trade_type"]
        size_mult = signal["size_mult"]
        is_bear_debit = (trade_type == "bear_put_debit")
        is_bear_call = (trade_type == "bear_call")
        bear_tier = signal.get("bear_tier", 0)

        # Check concurrent position limits
        if is_bear_debit:
            open_count = await self._count_open_trades(db, version, "bear_put_debit")
            max_concurrent = cfg.get("BEAR_DEBIT_MAX_CONCURRENT", 2)
        elif is_bear_call:
            open_count = await self._count_open_trades(db, version, "bear_call")
            max_concurrent = cfg.get("BEAR_CALL_MAX_CONCURRENT", 1)
        else:
            open_count = await self._count_open_trades(db, version, trade_type)
            max_concurrent = cfg.get("MAX_CONCURRENT_POSITIONS", 3)
            if trade_type == "iron_condor":
                max_concurrent = cfg.get("IC_MAX_CONCURRENT", 2)

        if open_count >= max_concurrent:
            logger.info(f"[{version}] Max concurrent {trade_type} reached ({open_count})")
            return None

        # Check minimum entry gap (only for bull/IC credit trades, not bear_call)
        if not is_bear_debit and not is_bear_call:
            min_gap = cfg.get("MIN_ENTRY_GAP_DAYS", 2)
            last_entry = await self._get_last_entry_date(db, version)
            if last_entry and (today_ist() - last_entry).days < min_gap:
                logger.info(f"[{version}] Entry gap too small ({(today_ist() - last_entry).days} < {min_gap})")
                return None

        # Select strikes
        strike_cfg = {
            "BULL_OTM_SELL": BULL_OTM_SELL,
            "BULL_OTM_BUY": BULL_OTM_BUY,
            "IC_PUT_OTM_SELL": cfg.get("IC_PUT_OTM_SELL", 0.03),
            "IC_PUT_OTM_BUY": cfg.get("IC_PUT_OTM_BUY", 0.055),
            "IC_CALL_OTM_SELL": cfg.get("IC_CALL_OTM_SELL", 0.04),
            "IC_CALL_OTM_BUY": cfg.get("IC_CALL_OTM_BUY", 0.065),
            "BEAR_PUT_BUY_OTM": cfg.get("BEAR_PUT_BUY_OTM", 0.01),
            "BEAR_PUT_SELL_OTM": cfg.get("BEAR_PUT_SELL_OTM", 0.04),
            "BEAR_CALL_OTM_SELL": cfg.get("BEAR_CALL_OTM_SELL", 0.04),
            "BEAR_CALL_OTM_BUY": cfg.get("BEAR_CALL_OTM_BUY", 0.065),
        }
        strikes = select_strikes(spot, trade_type, strike_cfg)

        # Compute expiry
        expiry = get_next_weekly_expiry()
        T = compute_time_to_expiry_years(today_ist(), expiry)
        sigma = vix / 100.0 if vix else 0.15

        # Pricing-source metadata (overridden below for credit trades when real
        # option pricing is on); defaults keep the bear-debit + disabled paths safe.
        pricing_source, bs_credit_unit, real_credit_unit, pricing_reason = "bs_synthetic", None, None, None
        pricing_legs = []  # per-leg fill detail (sell@bid / buy@ask) for the signal email

        if is_bear_debit:
            # Bear put debit: buy higher-strike put, sell lower-strike put
            debit = price_bear_put_debit(
                spot, strikes["buy_strike"], strikes["sell_strike"],
                T, RISK_FREE_RATE, sigma, apply_slippage=True
            )
            credit = 0.0

            # Compute lots
            position_size_pct = cfg.get("POSITION_SIZE_PCT", 0.20)
            effective_size_pct = position_size_pct * size_mult
            max_capital = INITIAL_CAPITAL * effective_size_pct
            num_lots = max(1, int(max_capital / MARGIN_PER_LOT_BEAR))
            num_lots = min(num_lots, 3)  # cap at 3 lots for bear debits

            total_debit = debit * num_lots * NIFTY_LOT_SIZE
            capital_deployed = num_lots * MARGIN_PER_LOT_BEAR

            # Max profit/loss for bear debit spread
            spread_width = strikes["buy_strike"] - strikes["sell_strike"]
            max_profit_per_lot = (spread_width - debit) * NIFTY_LOT_SIZE
            max_loss_per_lot = debit * NIFTY_LOT_SIZE
            total_max_profit = max_profit_per_lot * num_lots
            total_max_loss = max_loss_per_lot * num_lots

            total_credit = 0.0
            entry_mode = "bear_debit"
        else:
            # Credit trades (bull_put, iron_condor, bear_call)
            debit = 0.0
            if trade_type == "bull_put":
                credit = price_bull_put_spread(
                    spot, strikes["sell_strike"], strikes["buy_strike"],
                    T, RISK_FREE_RATE, sigma, apply_slippage=True
                )
            elif trade_type == "iron_condor":
                credit = price_iron_condor(
                    spot, strikes["sell_strike"], strikes["buy_strike"],
                    strikes["ic_call_sell"], strikes["ic_call_buy"],
                    T, RISK_FREE_RATE, sigma, apply_slippage=True
                )
            elif trade_type == "bear_call":
                credit = price_bear_call_spread(
                    spot, strikes["ic_call_sell"], strikes["ic_call_buy"],
                    T, RISK_FREE_RATE, sigma, apply_slippage=True
                )
            else:
                credit = 0

            # --- Real option pricing: use the actual Dhan chain quote (sell@bid,
            # buy@ask) instead of synthetic BS where available; fall back to BS and
            # FLAG it loudly otherwise. Controlled by USE_REAL_OPTION_PRICING.
            bs_credit_unit = round(credit, 2)
            if USE_REAL_OPTION_PRICING and dhan_client is not None:
                try:
                    chain = await dhan_client.get_option_chain(expiry.isoformat())
                except Exception as e:
                    chain = None
                    logger.warning(f"[{version}] Option-chain fetch failed for real pricing: {e}")
                pr = price_spread_real(chain, trade_type, strikes, credit)
                log_pricing(version, trade_type, pr)
                credit = pr["credit"]
                pricing_source = pr["source"]
                bs_credit_unit = pr["bs_credit"]
                real_credit_unit = pr["real_credit"]
                pricing_reason = pr["fallback_reason"]
                pricing_legs = pr.get("legs") or []

            position_size_pct = cfg.get("POSITION_SIZE_PCT", 0.20)
            if trade_type == "iron_condor":
                position_size_pct = cfg.get("IC_POSITION_SIZE_PCT", 0.15)
            elif trade_type == "bear_call":
                position_size_pct = cfg.get("BEAR_CALL_SIZE_PCT", 0.20)

            effective_size_pct = position_size_pct * size_mult
            margin_per_lot = MARGIN_PER_LOT_BULL if trade_type == "bull_put" else MARGIN_PER_LOT_IC

            # Compounding: size off CURRENT equity so profits grow position size.
            # Falls back to INITIAL_CAPITAL if the lookup fails, so a DB hiccup can
            # never silently up-size a trade.
            sizing_base = INITIAL_CAPITAL
            if COMPOUND_SIZING:
                try:
                    sizing_base = await self._get_current_capital(db, version)
                except Exception as e:
                    logger.warning(f"[{version}] equity lookup failed, sizing off "
                                   f"INITIAL_CAPITAL: {e}")
                if not sizing_base or sizing_base <= 0:
                    sizing_base = INITIAL_CAPITAL

            max_capital = sizing_base * effective_size_pct
            num_lots = max(1, int(max_capital / margin_per_lot))

            # --- Margin cap -------------------------------------------------
            # Never let TOTAL real margin across open positions exceed
            # MAX_MARGIN_UTILISATION of equity. Without this, compounding peaks at
            # ~127% of equity and goes margin-short (forced liquidation). Downsize
            # to fit; skip the trade entirely if there is not room for one lot.
            per_unit = self._real_margin_per_unit(trade_type, strikes, credit, spot)
            if per_unit > 0:
                used = await self._open_margin_used(db, version, spot)
                headroom = sizing_base * MAX_MARGIN_UTILISATION - used
                affordable = int(headroom / (per_unit * NIFTY_LOT_SIZE))
                if affordable < 1:
                    logger.info(
                        f"[{version}] Skipping entry: margin cap reached "
                        f"(used ₹{used:,.0f} of ₹{sizing_base * MAX_MARGIN_UTILISATION:,.0f} "
                        f"allowed at {MAX_MARGIN_UTILISATION:.0%} of ₹{sizing_base:,.0f})"
                    )
                    return None
                if affordable < num_lots:
                    logger.info(
                        f"[{version}] Margin cap: downsizing {num_lots} -> {affordable} lots "
                        f"(used ₹{used:,.0f}, headroom ₹{headroom:,.0f})"
                    )
                    num_lots = affordable

            total_credit = credit * num_lots * NIFTY_LOT_SIZE
            capital_deployed = num_lots * margin_per_lot
            total_max_profit = None
            total_max_loss = None
            total_debit = 0.0

            entry_mode = "normal"
            if cfg.get("VIX_HARVEST_ENABLED") and vix and vix >= cfg.get("VIX_HARVEST_TRIGGER", 23):
                entry_mode = "vix_harvest"

        # Entry quality filter: skip trades where the premium offered is too small
        # relative to the max loss exposure. Empirically (Mar–May 2026) the 9 trades
        # opened on Wednesdays for next-day Thursday expiry had total credit
        # ₹12–₹1970 and net P&L −₹6.7K — small wins offset by two max-loss expiries.
        # Same filter expressed as DTE catches the same trades.
        dte_at_entry = (expiry - today_ist()).days
        if not is_bear_debit:
            if total_credit < MIN_TOTAL_CREDIT or dte_at_entry < MIN_ENTRY_DTE:
                logger.info(
                    f"[{version}] Skipping low-quality entry: "
                    f"total_credit=₹{total_credit:.0f} (min ₹{MIN_TOTAL_CREDIT}), "
                    f"DTE={dte_at_entry} (min {MIN_ENTRY_DTE})"
                )
                return None

        # Create trade ID
        trade_id = f"{version.replace('.', '')}-{today_ist().isoformat()}-{signal['signal']}"

        now = now_ist()
        trade = Trade(
            trade_id=trade_id,
            version=version,
            date=today_ist(),
            signal_type=signal["signal"],
            trade_type=trade_type,
            entry_mode=entry_mode,
            entry_date=today_ist(),
            entry_time=now,
            entry_spot=spot,
            expiry=expiry,
            sell_strike=strikes["sell_strike"],
            buy_strike=strikes["buy_strike"],
            ic_call_sell=strikes.get("ic_call_sell"),
            ic_call_buy=strikes.get("ic_call_buy"),
            num_lots=num_lots,
            lot_size=NIFTY_LOT_SIZE,
            credit_received=credit,
            total_credit=total_credit,
            status="open",
            current_pnl=0,
            current_pnl_pct=0,
            unrealized_pnl=0,
            position_size_pct=effective_size_pct if not is_bear_debit else size_mult,
            graduated_mult=size_mult,
            capital_deployed=capital_deployed,
            # Bear debit fields
            is_bear_debit=is_bear_debit,
            bear_tier=bear_tier,
            entry_debit=debit if is_bear_debit else None,
            predicted_drawdown=predicted_drawdown,
            max_profit=total_max_profit,
            max_loss_amount=total_max_loss,
            bear_trail_high=0.0,
            entry_vix=vix,
            pricing_source=pricing_source,
            bs_credit=bs_credit_unit,
            real_credit=real_credit_unit,
        )

        # Guard against duplicate trades (e.g. pipeline re-run on same day)
        existing = await db.execute(
            select(Trade).where(Trade.trade_id == trade_id)
        )
        if existing.scalar_one_or_none() is not None:
            logger.warning(f"[{version}] Trade {trade_id} already exists — skipping duplicate")
            return trade_id

        db.add(trade)
        await db.commit()

        logger.info(
            f"[{version}] Opened {trade_type} trade: {trade_id}, "
            f"strikes={strikes}, {'debit' if is_bear_debit else 'credit'}="
            f"{debit if is_bear_debit else credit:.2f}, lots={num_lots}"
        )

        result = {
            "trade_id": trade_id,
            "trade_type": trade_type,
            "strikes": strikes,
            "num_lots": num_lots,
            "entry_mode": entry_mode,
            "expiry": expiry.isoformat(),
        }
        if is_bear_debit:
            result.update({
                "debit": debit,
                "total_debit": total_debit,
                "bear_tier": bear_tier,
                "max_profit": total_max_profit,
                "max_loss": total_max_loss,
            })
        else:
            result.update({
                "credit": credit,
                "total_credit": total_credit,
                "pricing_source": pricing_source,
                "bs_credit": bs_credit_unit,
                "real_credit": real_credit_unit,
                "pricing_reason": pricing_reason,
                "pricing_legs": pricing_legs,
            })
        return result

    async def check_exits(self, version: str, spot: float, vix: float,
                           db: AsyncSession, loss_only: bool = False,
                           is_eod: bool = False) -> list[dict]:
        """
        Check all open positions for exit conditions.

        loss_only=True  → only check stop loss (runs every 5 min intraday)
        loss_only=False → check profit target + trailing stop too
        is_eod=True     → also check expiry (backtest evaluates at daily close)

        Returns list of closed trade dicts (for email notifications).
        """
        cfg = VERSION_CONFIGS.get(version, {})
        closed_trades = []

        result = await db.execute(
            select(Trade).where(
                Trade.version == version,
                Trade.status == "open"
            )
        )
        open_trades = result.scalars().all()

        # Fetch the live chain(s) once for real exit valuation (cached ~20s in the
        # client, so the 3 versions' calls within a cycle dedupe). BS fallback if
        # unavailable. Keyed by expiry.
        chains = {}
        if USE_REAL_OPTION_PRICING and getattr(self, "dhan_client", None):
            for exp in {t.expiry for t in open_trades if not t.is_bear_debit}:
                try:
                    chains[exp] = await self.dhan_client.get_option_chain(exp.isoformat())
                except Exception:
                    chains[exp] = None
                if chains.get(exp) is None:
                    logger.warning(
                        f"⚠️ [{version}] Exit pricing fell back to BS — "
                        f"option chain unavailable for expiry {exp}."
                    )

        for trade in open_trades:
            chain = chains.get(trade.expiry)
            if trade.is_bear_debit:
                exit_reason = await self._check_bear_debit_exit(
                    trade, spot, vix, cfg, loss_only=loss_only, is_eod=is_eod)
            else:
                exit_reason = await self._check_single_exit(
                    trade, spot, vix, cfg, loss_only=loss_only, is_eod=is_eod, chain=chain)

            if exit_reason:
                closed_info = await self._close_trade(trade, spot, exit_reason, db, vix=vix, chain=chain)
                if closed_info:
                    closed_trades.append(closed_info)
            else:
                # Update current PnL
                await self._update_trade_pnl(trade, spot, vix, db, chain=chain)

        return closed_trades

    async def _check_single_exit(self, trade: Trade, spot: float,
                                  vix: float, cfg: dict,
                                  loss_only: bool = False,
                                  is_eod: bool = False,
                                  chain: dict = None) -> Optional[str]:
        """
        Check if a single credit trade should be exited. Returns exit reason or None.

        loss_only=True:  only check stop loss (intraday 5-min checks)
        loss_only=False: also check profit target + trailing stop
        is_eod=True:     also check expiry (backtest evaluates at daily close)

        Exit order matches backtest (check_exit_credit):
        1. Profit target               — EOD only
        2. Trailing stop               — EOD only
        3. Stop loss (N-day confirm)   — always checked
        4. VIX spike                   — always checked
        5. Expiry (DTE ≤ 1)            — EOD only (matches backtest daily-close eval)
        """
        today = today_ist()
        dte = (trade.expiry - today).days

        # Current cost-to-close (real from the chain, else BS — same basis as entry)
        current_value = self._spread_value(trade, spot, vix, chain)

        # PnL: credit_received - current_value (what it costs to close)
        pnl_per_unit = trade.credit_received - current_value
        pnl_pct = pnl_per_unit / trade.credit_received if trade.credit_received > 0 else 0

        # Always update peak tracking unconditionally (backtest does this before all checks)
        trade.peak_pnl_pct = max(trade.peak_pnl_pct or 0.0, pnl_pct)

        # --- 1. Profit target — EOD only (let winners run intraday) ---
        if not loss_only:
            days_held = (today - trade.entry_date).days
            if days_held <= 3:
                pt = cfg.get("PROFIT_TARGET_EARLY", 0.50)
            elif days_held <= 7:
                pt = cfg.get("PROFIT_TARGET_MID", 0.65)
            else:
                pt = cfg.get("PROFIT_TARGET_LATE", 0.80)

            if pnl_pct >= pt:
                return "profit_target"

        # --- 2. Trailing stop — activation is persistent, exit is EOD only ---
        trailing_activate = cfg.get("TRAILING_STOP_ACTIVATE", 0.40)
        trailing_level = cfg.get("TRAILING_STOP_LEVEL", 0.10)

        # Persistent activation: once True, stays True for the trade (matches backtest)
        if pnl_pct >= trailing_activate:
            trade.trailing_stop_active = True

        # Check trailing stop even if pnl dropped below activation (backtest behaviour)
        if not loss_only and trade.trailing_stop_active:
            peak = trade.peak_pnl_pct or 0.0
            if peak > 0:
                retrace = peak - pnl_pct
                # Backtest: retrace > peak * level AND still profitable
                if retrace > peak * trailing_level and pnl_pct > 0:
                    return "trailing_stop"

        # --- 3. Stop loss — always checked (with N-day confirmation) ---
        sl_mult = cfg.get("STOP_LOSS_MULTIPLIER", 3.0)
        if trade.trade_type == "iron_condor":
            sl_mult = cfg.get("IC_STOP_LOSS_MULTIPLIER", 3.0)

        # Profit-arming gate (v5.4.4+): if configured, use a looser pre-armed
        # multiplier until the trade has shown ARMING_PCT profit. The tight
        # stop is meant to protect *gained* profit, not punish unlucky opens
        # where the trade may still mean-revert.
        arming_pct = cfg.get("STOP_LOSS_ARMING_PCT")
        if arming_pct is not None and (trade.peak_pnl_pct or 0.0) < arming_pct:
            sl_mult = cfg.get("STOP_LOSS_ARMING_MULT", 3.0)

        confirm_days = cfg.get("STOP_LOSS_CONFIRM_DAYS", 2)
        if trade.trade_type == "iron_condor":
            confirm_days = cfg.get("IC_STOP_LOSS_CONFIRM_DAYS", 2)

        max_loss = trade.credit_received * sl_mult
        if current_value >= max_loss + trade.credit_received:
            # Breach detected — only increment once per calendar day
            last_breach = trade.stop_loss_last_breach_date
            if last_breach != today:
                trade.stop_loss_breach_days = (trade.stop_loss_breach_days or 0) + 1
                trade.stop_loss_last_breach_date = today
            # Backtest uses strict > (needs confirm_days+1 iterations to trigger)
            if trade.stop_loss_breach_days > confirm_days:
                return "stop_loss"
        else:
            # Recovery — reset breach counter
            trade.stop_loss_breach_days = 0
            trade.stop_loss_last_breach_date = None

        # --- 4. VIX spike exit (matches backtest: VIX > threshold AND VIX > entry * 1.3) ---
        vix_spike_threshold = cfg.get("VIX_SPIKE_EXIT")
        if vix_spike_threshold and vix:
            entry_vix = trade.entry_vix
            if entry_vix and vix > vix_spike_threshold and vix > entry_vix * 1.3:
                return "vix_spike"

        # --- 5. Expiry exit — EOD only (backtest evaluates at daily close) ---
        if is_eod:
            min_dte = cfg.get("MIN_DTE_EXIT", 1)
            if dte <= min_dte:
                return "expiry"

        return None

    async def _check_bear_debit_exit(self, trade: Trade, spot: float,
                                      vix: float, cfg: dict,
                                      loss_only: bool = False,
                                      is_eod: bool = False) -> Optional[str]:
        """Check if a bear debit trade should be exited. Returns exit reason or None."""
        today = today_ist()
        dte = (trade.expiry - today).days

        # 1. Expiry exit — EOD only (matches backtest daily-close evaluation)
        if is_eod and dte <= 1:
            return "expiry"

        # 2. Max hold days
        hold_days = (today - trade.entry_date).days
        max_hold = cfg.get("BEAR_DEBIT_MAX_HOLD_DAYS", 8)
        if hold_days >= max_hold:
            return "max_hold"

        # 3. Compute current spread value
        T = max(dte / 365.0, 1 / 365.0)
        sigma = vix / 100.0 if vix else 0.15
        current_value = compute_spread_value(
            "bear_put_debit", spot,
            trade.sell_strike, trade.buy_strike,
            None, None, T, RISK_FREE_RATE, sigma
        )

        entry_debit = trade.entry_debit or 0
        if entry_debit <= 0:
            return None

        # PnL ratio: (current_value - entry_debit) / entry_debit
        pnl_ratio = (current_value - entry_debit) / entry_debit

        # 4. Profit target — EOD only
        if not loss_only:
            profit_target = cfg.get("BEAR_DEBIT_PROFIT_TARGET", 2.0)
            if pnl_ratio >= profit_target - 1.0:  # 2.0 target means 100% gain
                return "profit_target"

        # 5. Stop loss — always checked
        stop_loss = cfg.get("BEAR_DEBIT_STOP_LOSS", 0.70)
        if pnl_ratio <= -stop_loss:
            return "stop_loss"

        # 6. Trailing stop — EOD only (trail high always updated)
        trailing_activate = cfg.get("BEAR_DEBIT_TRAILING_ACTIVATE", 1.0)
        trailing_level = cfg.get("BEAR_DEBIT_TRAILING_LEVEL", 0.30)

        if pnl_ratio >= trailing_activate - 1.0:
            # Always update trail high
            if current_value > trade.bear_trail_high:
                trade.bear_trail_high = current_value
            # Only exit on trailing stop at EOD
            if not loss_only and trade.bear_trail_high > 0:
                pullback = (trade.bear_trail_high - current_value) / trade.bear_trail_high
                if pullback >= trailing_level:
                    return "trailing_stop"

        return None

    async def _close_trade(self, trade: Trade, spot: float,
                            exit_reason: str, db: AsyncSession,
                            vix: float = None, chain: dict = None) -> Optional[dict]:
        """Close a trade and record final PnL. Returns trade info dict for notifications."""
        now = now_ist()

        # Final cost-to-close (real from the chain, else BS — same basis as entry)
        current_value = self._spread_value(trade, spot, vix, chain)

        if trade.is_bear_debit:
            # Bear debit PnL: (current_value - entry_debit) * lots * lot_size
            entry_debit = trade.entry_debit or 0
            realized_pnl = (current_value - entry_debit) * trade.num_lots * (trade.lot_size or 25)
        else:
            # Credit trade PnL: (credit_received - current_value) * lots * lot_size
            realized_pnl = (trade.credit_received - current_value) * \
                            trade.num_lots * (trade.lot_size or 25)

        pnl_pct = (
            realized_pnl / trade.capital_deployed * 100
            if trade.capital_deployed > 0 else 0
        )

        trade.status = "closed"
        trade.exit_date = today_ist()
        trade.exit_time = now
        trade.exit_spot = spot
        trade.exit_reason = exit_reason
        trade.realized_pnl = realized_pnl
        trade.current_pnl = realized_pnl
        trade.current_pnl_pct = pnl_pct
        trade.updated_at = now

        await db.commit()
        logger.info(
            f"[{trade.version}] Closed {trade.trade_id}: "
            f"reason={exit_reason}, pnl={realized_pnl:.0f}"
        )

        # Return trade info for email notifications
        result = {
            "trade_id": trade.trade_id,
            "exit_reason": exit_reason,
            "trade_type": trade.trade_type,
            "entry_spot": trade.entry_spot,
            "realized_pnl": realized_pnl,
            "pnl_pct": pnl_pct,
            "sell_strike": trade.sell_strike,
            "buy_strike": trade.buy_strike,
            "is_bear_debit": trade.is_bear_debit,
            "bear_tier": trade.bear_tier,
        }
        return result

    async def _update_trade_pnl(self, trade: Trade, spot: float,
                                 vix: float, db: AsyncSession, chain: dict = None):
        """Update unrealized PnL for an open trade."""
        current_value = self._spread_value(trade, spot, vix, chain)

        if trade.is_bear_debit:
            entry_debit = trade.entry_debit or 0
            unrealized_pnl = (current_value - entry_debit) * \
                              trade.num_lots * (trade.lot_size or 25)
            # Update trail high for trailing stop
            if current_value > trade.bear_trail_high:
                trade.bear_trail_high = current_value
        else:
            unrealized_pnl = (trade.credit_received - current_value) * \
                              trade.num_lots * (trade.lot_size or 25)

        trade.current_spread_value = current_value
        trade.unrealized_pnl = unrealized_pnl
        trade.current_pnl = unrealized_pnl
        trade.current_pnl_pct = (
            unrealized_pnl / trade.capital_deployed * 100
            if trade.capital_deployed > 0 else 0
        )
        trade.updated_at = now_ist()
        await db.commit()

    async def get_portfolio_state(self, version: str,
                                   db: AsyncSession) -> dict:
        """Compute current portfolio state for a version."""
        # Get all trades
        result = await db.execute(
            select(Trade).where(Trade.version == version)
        )
        all_trades = result.scalars().all()

        open_trades = [t for t in all_trades if t.status == "open"]
        closed_trades = [t for t in all_trades if t.status == "closed"]

        total_realized = sum(t.realized_pnl or 0 for t in closed_trades)
        total_unrealized = sum(t.unrealized_pnl or 0 for t in open_trades)
        total_pnl = total_realized + total_unrealized
        deployed = sum(t.capital_deployed or 0 for t in open_trades)

        current_capital = INITIAL_CAPITAL + total_pnl

        return {
            "starting_capital": INITIAL_CAPITAL,
            "current_capital": round(current_capital, 2),
            "total_pnl": round(total_pnl, 2),
            "total_return_pct": round(total_pnl / INITIAL_CAPITAL * 100, 2),
            "realized_pnl": round(total_realized, 2),
            "unrealized_pnl": round(total_unrealized, 2),
            "deployed_capital": round(deployed, 2),
            "available_capital": round(current_capital - deployed, 2),
            "open_positions": len(open_trades),
            "closed_positions": len(closed_trades),
            "total_trades": len(all_trades),
            "win_rate": self._compute_win_rate(closed_trades),
            "avg_pnl": round(
                total_realized / len(closed_trades), 2
            ) if closed_trades else 0,
            "profit_factor": self._compute_profit_factor(closed_trades),
        }

    def _compute_win_rate(self, closed_trades: list) -> float:
        """Compute win rate from closed trades."""
        if not closed_trades:
            return 0.0
        winners = sum(1 for t in closed_trades if (t.realized_pnl or 0) > 0)
        return round(winners / len(closed_trades), 3)

    def _compute_profit_factor(self, closed_trades: list) -> float:
        """Compute profit factor (gross profits / gross losses)."""
        gross_profit = sum(
            t.realized_pnl for t in closed_trades if (t.realized_pnl or 0) > 0
        )
        gross_loss = abs(sum(
            t.realized_pnl for t in closed_trades if (t.realized_pnl or 0) < 0
        ))
        if gross_loss == 0:
            return 999.0 if gross_profit > 0 else 0.0
        return round(gross_profit / gross_loss, 2)

    async def _count_open_trades(self, db: AsyncSession, version: str,
                                  trade_type: str = None) -> int:
        """Count open trades for version, optionally filtered by type."""
        stmt = select(Trade).where(
            Trade.version == version,
            Trade.status == "open"
        )
        if trade_type:
            stmt = stmt.where(Trade.trade_type == trade_type)
        result = await db.execute(stmt)
        return len(result.scalars().all())

    async def _get_last_entry_date(self, db: AsyncSession,
                                    version: str) -> Optional[date]:
        """Get date of most recent trade entry for version."""
        result = await db.execute(
            select(Trade.entry_date).where(
                Trade.version == version
            ).order_by(Trade.entry_date.desc()).limit(1)
        )
        row = result.scalar_one_or_none()
        return row

    async def _get_current_capital(self, db: AsyncSession,
                                    version: str) -> float:
        """Get current capital for version."""
        portfolio = await self.get_portfolio_state(version, db)
        return portfolio["current_capital"]

    @staticmethod
    def _real_margin_per_unit(trade_type: str, strikes: dict,
                              credit: float, spot: float) -> float:
        """
        Real broker margin per UNIT, using the formula verified against live Kite
        basket quotes on 2026-07-30 (predicted 4 structures within 1.2%):

            margin = max_loss + EXPOSURE_MARGIN_PCT * spot * n_short_legs

        The exposure term is ~62% of the total and does NOT shrink with narrower
        wings, so narrowing spreads hurts capital efficiency rather than helping.
        Returns 0.0 for debit spreads (margin there is just the premium paid).
        """
        if trade_type == "bear_put_debit":
            return 0.0
        sell = strikes.get("sell_strike")
        buy = strikes.get("buy_strike")
        if sell is None or buy is None:
            return 0.0
        width = abs(sell - buy)
        n_short = 1
        if trade_type == "iron_condor":
            n_short = 2
            # Only one wing can be breached, so max loss is the WIDER wing. Both are
            # 2.5% of spot today, but don't assume that stays true.
            c_sell, c_buy = strikes.get("ic_call_sell"), strikes.get("ic_call_buy")
            if c_sell is not None and c_buy is not None:
                width = max(width, abs(c_buy - c_sell))
        max_loss = max(0.0, width - (credit or 0.0))
        return max_loss + EXPOSURE_MARGIN_PCT * spot * n_short

    async def _open_margin_used(self, db: AsyncSession, version: str,
                                spot: float) -> float:
        """Total real margin currently blocked by this version's open trades."""
        result = await db.execute(
            select(Trade).where(Trade.version == version, Trade.status == "open")
        )
        total = 0.0
        for t in result.scalars().all():
            per_unit = self._real_margin_per_unit(
                t.trade_type,
                {"sell_strike": t.sell_strike, "buy_strike": t.buy_strike,
                 "ic_call_sell": t.ic_call_sell, "ic_call_buy": t.ic_call_buy},
                t.credit_received, spot,
            )
            total += per_unit * t.num_lots * (t.lot_size or 25)
        return total
