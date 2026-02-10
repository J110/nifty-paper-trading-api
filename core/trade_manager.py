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

from config import (
    INITIAL_CAPITAL, NIFTY_LOT_SIZE, RISK_FREE_RATE,
    MARGIN_PER_LOT_BULL, MARGIN_PER_LOT_IC, VERSION_CONFIGS,
    BULL_OTM_SELL, BULL_OTM_BUY,
)
from core.option_pricer import (
    select_strikes, price_bull_put_spread, price_iron_condor,
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
    ) -> Optional[dict]:
        """
        Construct and record a new paper trade.
        Returns trade dict or None if trade can't be opened.
        """
        if signal["signal"] == "no_trade" or signal["trade_type"] is None:
            return None

        cfg = VERSION_CONFIGS.get(version, {})
        trade_type = signal["trade_type"]
        size_mult = signal["size_mult"]

        # Check concurrent position limits
        open_count = await self._count_open_trades(db, version, trade_type)
        max_concurrent = cfg.get("MAX_CONCURRENT_POSITIONS", 3)
        if trade_type == "iron_condor":
            max_concurrent = cfg.get("IC_MAX_CONCURRENT", 2)

        if open_count >= max_concurrent:
            logger.info(f"[{version}] Max concurrent {trade_type} reached ({open_count})")
            return None

        # Check minimum entry gap
        min_gap = cfg.get("MIN_ENTRY_GAP_DAYS", 2)
        last_entry = await self._get_last_entry_date(db, version)
        if last_entry and (date.today() - last_entry).days < min_gap:
            logger.info(f"[{version}] Entry gap too small ({(date.today() - last_entry).days} < {min_gap})")
            return None

        # Select strikes
        strikes = select_strikes(spot, trade_type, {
            "BULL_OTM_SELL": BULL_OTM_SELL,
            "BULL_OTM_BUY": BULL_OTM_BUY,
            "IC_PUT_OTM_SELL": cfg.get("IC_PUT_OTM_SELL", 0.03),
            "IC_PUT_OTM_BUY": cfg.get("IC_PUT_OTM_BUY", 0.055),
            "IC_CALL_OTM_SELL": cfg.get("IC_CALL_OTM_SELL", 0.04),
            "IC_CALL_OTM_BUY": cfg.get("IC_CALL_OTM_BUY", 0.065),
        })

        # Compute expiry
        expiry = get_next_weekly_expiry()
        T = compute_time_to_expiry_years(date.today(), expiry)
        sigma = vix / 100.0 if vix else 0.15

        # Compute credit
        if trade_type == "bull_put":
            credit = price_bull_put_spread(
                spot, strikes["sell_strike"], strikes["buy_strike"],
                T, RISK_FREE_RATE, sigma
            )
        elif trade_type == "iron_condor":
            credit = price_iron_condor(
                spot, strikes["sell_strike"], strikes["buy_strike"],
                strikes["ic_call_sell"], strikes["ic_call_buy"],
                T, RISK_FREE_RATE, sigma
            )
        else:
            credit = 0

        # Compute lots
        position_size_pct = cfg.get("POSITION_SIZE_PCT", 0.20)
        if trade_type == "iron_condor":
            position_size_pct = cfg.get("IC_POSITION_SIZE_PCT", 0.15)

        effective_size_pct = position_size_pct * size_mult
        margin_per_lot = MARGIN_PER_LOT_BULL if trade_type == "bull_put" else MARGIN_PER_LOT_IC

        # Get current capital
        current_capital = await self._get_current_capital(db, version)
        max_capital = current_capital * effective_size_pct
        num_lots = max(1, int(max_capital / margin_per_lot))

        total_credit = credit * num_lots * NIFTY_LOT_SIZE
        capital_deployed = num_lots * margin_per_lot

        # Determine entry mode
        entry_mode = "normal"
        if cfg.get("VIX_HARVEST_ENABLED") and vix and vix >= cfg.get("VIX_HARVEST_TRIGGER", 23):
            entry_mode = "vix_harvest"
        elif cfg.get("EVENT_CRUSH_ENABLED"):
            entry_mode = "normal"  # Event detection would be more complex

        # Create trade ID
        trade_id = f"{version.replace('.', '')}-{date.today().isoformat()}-{signal['signal']}"

        now = datetime.now()
        trade = Trade(
            trade_id=trade_id,
            version=version,
            date=date.today(),
            signal_type=signal["signal"],
            trade_type=trade_type,
            entry_mode=entry_mode,
            entry_date=date.today(),
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
            position_size_pct=effective_size_pct,
            graduated_mult=size_mult,
            capital_deployed=capital_deployed,
        )

        db.add(trade)
        await db.commit()

        logger.info(
            f"[{version}] Opened {trade_type} trade: {trade_id}, "
            f"strikes={strikes}, credit={credit:.2f}, lots={num_lots}"
        )

        return {
            "trade_id": trade_id,
            "trade_type": trade_type,
            "strikes": strikes,
            "credit": credit,
            "total_credit": total_credit,
            "num_lots": num_lots,
            "entry_mode": entry_mode,
            "expiry": expiry.isoformat(),
        }

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
            exit_reason = await self._check_single_exit(trade, spot, vix, cfg)

            if exit_reason:
                closed_info = await self._close_trade(trade, spot, exit_reason, db)
                if closed_info:
                    closed_trades.append(closed_info)
            else:
                # Update current PnL
                await self._update_trade_pnl(trade, spot, vix, db)

        return closed_trades

    async def _check_single_exit(self, trade: Trade, spot: float,
                                  vix: float, cfg: dict) -> Optional[str]:
        """Check if a single trade should be exited. Returns exit reason or None."""
        today = date.today()
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

        # 3. Profit target (varies by DTE stage)
        entry_dte = (trade.expiry - trade.entry_date).days
        elapsed_pct = 1 - (dte / max(entry_dte, 1))

        if elapsed_pct < 0.4:
            pt = cfg.get("PROFIT_TARGET_EARLY", 0.50)
        elif elapsed_pct < 0.7:
            pt = cfg.get("PROFIT_TARGET_MID", 0.65)
        else:
            pt = cfg.get("PROFIT_TARGET_LATE", 0.80)

        if pnl_pct >= pt:
            return "profit_target"

        # 4. Stop loss
        sl_mult = cfg.get("STOP_LOSS_MULTIPLIER", 3.0)
        if trade.trade_type == "iron_condor":
            sl_mult = cfg.get("IC_STOP_LOSS_MULTIPLIER", 3.0)

        max_loss = trade.credit_received * sl_mult
        if current_value >= max_loss + trade.credit_received:
            return "stop_loss"

        # 5. Trailing stop
        trailing_activate = cfg.get("TRAILING_STOP_ACTIVATE", 0.40)
        trailing_level = cfg.get("TRAILING_STOP_LEVEL", 0.10)

        if pnl_pct >= trailing_activate:
            # Check if price has pulled back enough from peak
            if hasattr(trade, '_peak_pnl_pct'):
                if pnl_pct < trade._peak_pnl_pct - trailing_level:
                    return "trailing_stop"
            trade._peak_pnl_pct = max(
                getattr(trade, '_peak_pnl_pct', 0), pnl_pct
            )

        return None

    async def _close_trade(self, trade: Trade, spot: float,
                            exit_reason: str, db: AsyncSession) -> Optional[dict]:
        """Close a trade and record final PnL. Returns trade info dict for notifications."""
        now = datetime.now()

        # Compute final PnL
        T = max((trade.expiry - date.today()).days / 365.0, 1 / 365.0)
        sigma = 0.15  # Approximate; in production use live VIX
        current_value = compute_spread_value(
            trade.trade_type, spot,
            trade.sell_strike, trade.buy_strike,
            trade.ic_call_sell, trade.ic_call_buy,
            T, RISK_FREE_RATE, sigma
        )

        realized_pnl = (trade.credit_received - current_value) * \
                        trade.num_lots * NIFTY_LOT_SIZE

        pnl_pct = (
            realized_pnl / trade.capital_deployed * 100
            if trade.capital_deployed > 0 else 0
        )

        trade.status = "closed"
        trade.exit_date = date.today()
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
        return {
            "trade_id": trade.trade_id,
            "exit_reason": exit_reason,
            "trade_type": trade.trade_type,
            "entry_spot": trade.entry_spot,
            "realized_pnl": realized_pnl,
            "pnl_pct": pnl_pct,
            "sell_strike": trade.sell_strike,
            "buy_strike": trade.buy_strike,
        }

    async def _update_trade_pnl(self, trade: Trade, spot: float,
                                 vix: float, db: AsyncSession):
        """Update unrealized PnL for an open trade."""
        T = max((trade.expiry - date.today()).days / 365.0, 1 / 365.0)
        sigma = vix / 100.0 if vix else 0.15
        current_value = compute_spread_value(
            trade.trade_type, spot,
            trade.sell_strike, trade.buy_strike,
            trade.ic_call_sell, trade.ic_call_buy,
            T, RISK_FREE_RATE, sigma
        )

        unrealized_pnl = (trade.credit_received - current_value) * \
                          trade.num_lots * NIFTY_LOT_SIZE

        trade.current_spread_value = current_value
        trade.unrealized_pnl = unrealized_pnl
        trade.current_pnl = unrealized_pnl
        trade.current_pnl_pct = (
            unrealized_pnl / trade.capital_deployed * 100
            if trade.capital_deployed > 0 else 0
        )
        trade.updated_at = datetime.now()
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
