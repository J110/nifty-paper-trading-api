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
)
from core.option_pricer import (
    select_strikes, price_bull_put_spread, price_iron_condor,
    price_bear_put_debit,
    get_next_weekly_expiry, compute_time_to_expiry_years,
    compute_spread_value,
)
from db.models import Trade, DailyPnl, Prediction

logger = logging.getLogger(__name__)


class TradeManager:
    """Manages paper trade lifecycle for all versions."""

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
        bear_tier = signal.get("bear_tier", 0)

        # Check concurrent position limits
        if is_bear_debit:
            open_count = await self._count_open_trades(db, version, "bear_put_debit")
            max_concurrent = cfg.get("BEAR_DEBIT_MAX_CONCURRENT", 2)
        else:
            open_count = await self._count_open_trades(db, version, trade_type)
            max_concurrent = cfg.get("MAX_CONCURRENT_POSITIONS", 3)
            if trade_type == "iron_condor":
                max_concurrent = cfg.get("IC_MAX_CONCURRENT", 2)

        if open_count >= max_concurrent:
            logger.info(f"[{version}] Max concurrent {trade_type} reached ({open_count})")
            return None

        # Check minimum entry gap (only for credit trades)
        if not is_bear_debit:
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
        }
        strikes = select_strikes(spot, trade_type, strike_cfg)

        # Compute expiry
        expiry = get_next_weekly_expiry()
        T = compute_time_to_expiry_years(today_ist(), expiry)
        sigma = vix / 100.0 if vix else 0.15

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
            current_capital = await self._get_current_capital(db, version)
            max_capital = current_capital * effective_size_pct
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
            # Credit trades (bull_put, iron_condor)
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
            else:
                credit = 0

            position_size_pct = cfg.get("POSITION_SIZE_PCT", 0.20)
            if trade_type == "iron_condor":
                position_size_pct = cfg.get("IC_POSITION_SIZE_PCT", 0.15)

            effective_size_pct = position_size_pct * size_mult
            margin_per_lot = MARGIN_PER_LOT_BULL if trade_type == "bull_put" else MARGIN_PER_LOT_IC

            current_capital = await self._get_current_capital(db, version)
            max_capital = current_capital * effective_size_pct
            num_lots = max(1, int(max_capital / margin_per_lot))

            total_credit = credit * num_lots * NIFTY_LOT_SIZE
            capital_deployed = num_lots * margin_per_lot
            total_max_profit = None
            total_max_loss = None
            total_debit = 0.0

            entry_mode = "normal"
            if cfg.get("VIX_HARVEST_ENABLED") and vix and vix >= cfg.get("VIX_HARVEST_TRIGGER", 23):
                entry_mode = "vix_harvest"

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
        )

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
            })
        return result

    async def check_exits(self, version: str, spot: float, vix: float,
                           db: AsyncSession) -> list[dict]:
        """
        Check all open positions for exit conditions.
        Called every 5 minutes during market hours.
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

        for trade in open_trades:
            if trade.is_bear_debit:
                exit_reason = await self._check_bear_debit_exit(trade, spot, vix, cfg)
            else:
                exit_reason = await self._check_single_exit(trade, spot, vix, cfg)

            if exit_reason:
                closed_info = await self._close_trade(trade, spot, exit_reason, db, vix=vix)
                if closed_info:
                    closed_trades.append(closed_info)
            else:
                # Update current PnL
                await self._update_trade_pnl(trade, spot, vix, db)

        return closed_trades

    async def _check_single_exit(self, trade: Trade, spot: float,
                                  vix: float, cfg: dict) -> Optional[str]:
        """
        Check if a single credit trade should be exited. Returns exit reason or None.

        Matches backtest exit hierarchy:
        1. Expiry (DTE ≤ 1)
        2. Profit target (day-count stages: ≤3 / 4-7 / ≥8)
        3. Trailing stop (DB-persisted peak_pnl_pct)
        4. Stop loss (with N-day confirmation, DB-persisted breach counter)
        5. DTE expiry
        """
        today = today_ist()
        dte = (trade.expiry - today).days

        # 1. Expiry exit
        min_dte = cfg.get("MIN_DTE_EXIT", 1)
        if dte <= min_dte:
            return "expiry"

        # 2. Compute current spread value
        T = max(dte / 365.0, 1 / 365.0)
        sigma = vix / 100.0 if vix else 0.15
        current_value = compute_spread_value(
            trade.trade_type, spot,
            trade.sell_strike, trade.buy_strike,
            trade.ic_call_sell, trade.ic_call_buy,
            T, RISK_FREE_RATE, sigma
        )

        # PnL: credit_received - current_value (what it costs to close)
        pnl_per_unit = trade.credit_received - current_value
        pnl_pct = pnl_per_unit / trade.credit_received if trade.credit_received > 0 else 0

        # 3. Profit target — day-count stages matching backtest
        days_held = (today - trade.entry_date).days
        if days_held <= 3:
            pt = cfg.get("PROFIT_TARGET_EARLY", 0.50)
        elif days_held <= 7:
            pt = cfg.get("PROFIT_TARGET_MID", 0.65)
        else:
            pt = cfg.get("PROFIT_TARGET_LATE", 0.80)

        if pnl_pct >= pt:
            return "profit_target"

        # 4. Trailing stop — DB-persisted peak tracking
        trailing_activate = cfg.get("TRAILING_STOP_ACTIVATE", 0.40)
        trailing_level = cfg.get("TRAILING_STOP_LEVEL", 0.10)

        if pnl_pct >= trailing_activate:
            peak = trade.peak_pnl_pct or 0.0
            if peak > 0 and pnl_pct < peak - trailing_level:
                return "trailing_stop"
            trade.peak_pnl_pct = max(peak, pnl_pct)

        # 5. Stop loss — with N-day confirmation matching backtest
        sl_mult = cfg.get("STOP_LOSS_MULTIPLIER", 3.0)
        if trade.trade_type == "iron_condor":
            sl_mult = cfg.get("IC_STOP_LOSS_MULTIPLIER", 3.0)

        confirm_days = cfg.get("STOP_LOSS_CONFIRM_DAYS", 1)
        if trade.trade_type == "iron_condor":
            confirm_days = cfg.get("IC_STOP_LOSS_CONFIRM_DAYS", 1)

        max_loss = trade.credit_received * sl_mult
        if current_value >= max_loss + trade.credit_received:
            # Breach detected — increment counter
            trade.stop_loss_breach_days = (trade.stop_loss_breach_days or 0) + 1
            if trade.stop_loss_breach_days >= confirm_days:
                return "stop_loss"
        else:
            # Recovery — reset breach counter
            trade.stop_loss_breach_days = 0

        return None

    async def _check_bear_debit_exit(self, trade: Trade, spot: float,
                                      vix: float, cfg: dict) -> Optional[str]:
        """Check if a bear debit trade should be exited. Returns exit reason or None."""
        today = today_ist()
        dte = (trade.expiry - today).days

        # 1. Expiry exit
        if dte <= 1:
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

        # 4. Profit target (e.g., 2x = 100% gain)
        profit_target = cfg.get("BEAR_DEBIT_PROFIT_TARGET", 2.0)
        if pnl_ratio >= profit_target - 1.0:  # 2.0 target means 100% gain
            return "profit_target"

        # 5. Stop loss (e.g., 70% loss of debit)
        stop_loss = cfg.get("BEAR_DEBIT_STOP_LOSS", 0.70)
        if pnl_ratio <= -stop_loss:
            return "stop_loss"

        # 6. Trailing stop
        trailing_activate = cfg.get("BEAR_DEBIT_TRAILING_ACTIVATE", 1.0)
        trailing_level = cfg.get("BEAR_DEBIT_TRAILING_LEVEL", 0.30)

        if pnl_ratio >= trailing_activate - 1.0:
            # Update trail high
            if current_value > trade.bear_trail_high:
                trade.bear_trail_high = current_value
            # Check if pulled back from peak
            if trade.bear_trail_high > 0:
                pullback = (trade.bear_trail_high - current_value) / trade.bear_trail_high
                if pullback >= trailing_level:
                    return "trailing_stop"

        return None

    async def _close_trade(self, trade: Trade, spot: float,
                            exit_reason: str, db: AsyncSession,
                            vix: float = None) -> Optional[dict]:
        """Close a trade and record final PnL. Returns trade info dict for notifications."""
        now = now_ist()

        # Compute final PnL
        T = max((trade.expiry - today_ist()).days / 365.0, 1 / 365.0)
        sigma = vix / 100.0 if vix else 0.15
        current_value = compute_spread_value(
            trade.trade_type, spot,
            trade.sell_strike, trade.buy_strike,
            trade.ic_call_sell, trade.ic_call_buy,
            T, RISK_FREE_RATE, sigma
        )

        if trade.is_bear_debit:
            # Bear debit PnL: (current_value - entry_debit) * lots * lot_size
            entry_debit = trade.entry_debit or 0
            realized_pnl = (current_value - entry_debit) * trade.num_lots * NIFTY_LOT_SIZE
        else:
            # Credit trade PnL: (credit_received - current_value) * lots * lot_size
            realized_pnl = (trade.credit_received - current_value) * \
                            trade.num_lots * NIFTY_LOT_SIZE

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
                                 vix: float, db: AsyncSession):
        """Update unrealized PnL for an open trade."""
        T = max((trade.expiry - today_ist()).days / 365.0, 1 / 365.0)
        sigma = vix / 100.0 if vix else 0.15
        current_value = compute_spread_value(
            trade.trade_type, spot,
            trade.sell_strike, trade.buy_strike,
            trade.ic_call_sell, trade.ic_call_buy,
            T, RISK_FREE_RATE, sigma
        )

        if trade.is_bear_debit:
            entry_debit = trade.entry_debit or 0
            unrealized_pnl = (current_value - entry_debit) * \
                              trade.num_lots * NIFTY_LOT_SIZE
            # Update trail high for trailing stop
            if current_value > trade.bear_trail_high:
                trade.bear_trail_high = current_value
        else:
            unrealized_pnl = (trade.credit_received - current_value) * \
                              trade.num_lots * NIFTY_LOT_SIZE

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
