"""API routes for trades, portfolio, and delay analysis."""

from datetime import date, timedelta
from fastapi import APIRouter, Depends, Query
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from db.database import get_db
from db.models import Trade, DelayPrice, Prediction
from core.trade_manager import TradeManager
from core.price_tracker import PriceTracker
from core.signal_mapper import map_signal
from config import ACTIVE_VERSIONS, VERSION_CONFIGS

router = APIRouter(prefix="/api")
trade_manager = TradeManager()
price_tracker = PriceTracker()


@router.get("/trades/{version}")
async def get_trades(
    version: str,
    status: str = Query("all", regex="^(all|open|closed)$"),
    db: AsyncSession = Depends(get_db),
):
    """Returns trade list and portfolio summary for a version."""
    if version not in ACTIVE_VERSIONS:
        return {"error": f"Unknown version: {version}"}

    # Get portfolio state
    portfolio = await trade_manager.get_portfolio_state(version, db)

    # Get trades
    stmt = select(Trade).where(Trade.version == version)
    if status == "open":
        stmt = stmt.where(Trade.status == "open")
    elif status == "closed":
        stmt = stmt.where(Trade.status == "closed")
    stmt = stmt.order_by(desc(Trade.entry_date))

    result = await db.execute(stmt)
    trades = result.scalars().all()

    open_trades = []
    closed_trades = []
    for t in trades:
        trade_dict = _trade_to_dict(t)
        if t.status == "open":
            open_trades.append(trade_dict)
        else:
            closed_trades.append(trade_dict)

    cfg = VERSION_CONFIGS.get(version, {})

    return {
        "version": version,
        "label": cfg.get("label", version),
        "color": cfg.get("color", "#888"),
        "portfolio": portfolio,
        "open_trades": open_trades,
        "closed_trades": closed_trades,
        "summary": {
            "total_trades": portfolio["total_trades"],
            "win_rate": portfolio["win_rate"],
            "avg_pnl": portfolio["avg_pnl"],
            "profit_factor": portfolio["profit_factor"],
        },
    }


@router.get("/trades/{version}/delay-analysis")
async def get_delay_analysis(
    version: str,
    data_mode: str = Query("combined", regex="^(backtest|forwardtest|combined)$"),
    db: AsyncSession = Depends(get_db),
):
    """Returns execution delay impact analysis."""
    if version not in ACTIVE_VERSIONS:
        return {"error": f"Unknown version: {version}"}

    analysis = await price_tracker.compute_delay_analysis(
        version, db, data_mode=data_mode, forward_test_start=FORWARD_TEST_START,
    )

    return {
        "version": version,
        "data_mode": data_mode,
        "analysis": analysis,
    }


FORWARD_TEST_START = date(2026, 2, 13)


@router.get("/trades/{version}/returns")
async def get_returns(
    version: str,
    period: str = Query("weekly", regex="^(daily|weekly|monthly)$"),
    data_mode: str = Query("combined", regex="^(backtest|forwardtest|combined)$"),
    from_date: str | None = Query(None, regex=r"^\d{4}-\d{2}-\d{2}$"),
    to_date: str | None = Query(None, regex=r"^\d{4}-\d{2}-\d{2}$"),
    db: AsyncSession = Depends(get_db),
):
    """
    Returns periodic returns.

    data_mode:
      - backtest: trades with entry_date < 2026-02-13
      - forwardtest: trades with entry_date >= 2026-02-13
      - combined: all trades

    from_date / to_date: optional ISO date strings to further restrict the range.
    """
    if version not in ACTIVE_VERSIONS:
        return {"error": f"Unknown version: {version}"}

    # Build query
    stmt = select(Trade).where(
        Trade.version == version,
        Trade.status == "closed",
    )

    # Apply data_mode filter
    if data_mode == "backtest":
        stmt = stmt.where(Trade.entry_date < FORWARD_TEST_START)
    elif data_mode == "forwardtest":
        stmt = stmt.where(Trade.entry_date >= FORWARD_TEST_START)

    # Apply optional date range filters
    if from_date:
        stmt = stmt.where(Trade.exit_date >= date.fromisoformat(from_date))
    if to_date:
        stmt = stmt.where(Trade.exit_date <= date.fromisoformat(to_date))

    stmt = stmt.order_by(Trade.exit_date)
    result = await db.execute(stmt)
    trades = result.scalars().all()

    # Compute aggregate summary for the filtered trade set
    # Assumes fresh INITIAL_CAPITAL base (all positions closed before this set)
    from config import INITIAL_CAPITAL

    total_pnl = sum(t.realized_pnl for t in trades if t.realized_pnl is not None)
    total_trades = len([t for t in trades if t.realized_pnl is not None])
    winning = len([t for t in trades if t.realized_pnl is not None and t.realized_pnl > 0])
    losing_pnl = sum(
        t.realized_pnl for t in trades
        if t.realized_pnl is not None and t.realized_pnl < 0
    )
    winning_pnl = sum(
        t.realized_pnl for t in trades
        if t.realized_pnl is not None and t.realized_pnl > 0
    )

    summary = {
        "total_pnl": round(total_pnl, 2),
        "total_return_pct": round(total_pnl / INITIAL_CAPITAL * 100, 2),
        "total_trades": total_trades,
        "win_rate": round(winning / total_trades * 100, 1) if total_trades > 0 else 0,
        "profit_factor": round(winning_pnl / abs(losing_pnl), 2) if losing_pnl != 0 else 999.0,
        "avg_pnl": round(total_pnl / total_trades, 2) if total_trades > 0 else 0,
        "starting_capital": INITIAL_CAPITAL,
    }

    if not trades:
        return {
            "version": version,
            "period": period,
            "data_mode": data_mode,
            "summary": summary,
            "returns": [],
            "cumulative": [],
        }

    # Group by period
    returns = []
    cumulative = []
    cumulative_pnl = 0.0

    from collections import defaultdict
    period_groups = defaultdict(lambda: {"pnl": 0, "trades": 0, "wins": 0})

    for t in trades:
        if not t.exit_date or t.realized_pnl is None:
            continue

        if period == "daily":
            key = t.exit_date.isoformat()
        elif period == "weekly":
            # ISO week
            iso = t.exit_date.isocalendar()
            key = f"{iso[0]}-W{iso[1]:02d}"
        else:  # monthly
            key = f"{t.exit_date.year}-{t.exit_date.month:02d}"

        period_groups[key]["pnl"] += t.realized_pnl
        period_groups[key]["trades"] += 1
        if t.realized_pnl > 0:
            period_groups[key]["wins"] += 1

    for key in sorted(period_groups.keys()):
        data = period_groups[key]
        cumulative_pnl += data["pnl"]
        win_rate = data["wins"] / data["trades"] if data["trades"] > 0 else 0

        returns.append({
            "period": key,
            "pnl": round(data["pnl"], 2),
            "return_pct": round(data["pnl"] / INITIAL_CAPITAL * 100, 2),
            "trades": data["trades"],
            "win_rate": round(win_rate, 3),
        })

        cumulative.append({
            "period": key,
            "cumulative_pnl": round(cumulative_pnl, 2),
            "cumulative_return": round(cumulative_pnl / INITIAL_CAPITAL * 100, 2),
        })

    return {
        "version": version,
        "period": period,
        "data_mode": data_mode,
        "summary": summary,
        "returns": returns,
        "cumulative": cumulative,
    }


@router.get("/trades/{version}/recommendations")
async def get_recommendations(
    version: str,
    from_date: str | None = Query(None, regex=r"^\d{4}-\d{2}-\d{2}$"),
    to_date: str | None = Query(None, regex=r"^\d{4}-\d{2}-\d{2}$"),
    db: AsyncSession = Depends(get_db),
):
    """
    Returns trade recommendations (daily predictions mapped to version-specific signals).
    Each row = one trading day with the model's prediction and the resulting trade recommendation.
    """
    if version not in ACTIVE_VERSIONS:
        return {"error": f"Unknown version: {version}"}

    cfg = VERSION_CONFIGS[version]

    # Get predictions — one prediction per date (any version, same model)
    stmt = select(Prediction).order_by(Prediction.date)

    if from_date:
        stmt = stmt.where(Prediction.date >= date.fromisoformat(from_date))
    if to_date:
        stmt = stmt.where(Prediction.date <= date.fromisoformat(to_date))

    result = await db.execute(stmt)
    all_predictions = result.scalars().all()

    # De-duplicate by date (keep first per date since all versions have same prediction)
    seen_dates = set()
    predictions = []
    for p in all_predictions:
        if p.date not in seen_dates:
            seen_dates.add(p.date)
            predictions.append(p)

    recommendations = []
    for p in predictions:
        signal = map_signal(p.predicted_drawdown, cfg)
        recommendations.append({
            "date": p.date.isoformat(),
            "predicted_drawdown_pct": round(p.predicted_drawdown * 100, 2),
            "signal": signal["signal"],
            "trade_type": signal["trade_type"],
            "size_mult": signal["size_mult"],
            "nifty_spot": p.nifty_spot,
            "vix": p.vix,
            "confidence_score": p.confidence_score,
        })

    return {
        "version": version,
        "label": cfg.get("label", version),
        "count": len(recommendations),
        "recommendations": recommendations,
    }


def _trade_to_dict(t: Trade) -> dict:
    """Convert Trade ORM object to dict for API response."""
    d = {
        "trade_id": t.trade_id,
        "version": t.version,
        "date": t.date.isoformat() if t.date else None,
        "signal_type": t.signal_type,
        "trade_type": t.trade_type,
        "entry_mode": t.entry_mode,
        "entry_date": t.entry_date.isoformat() if t.entry_date else None,
        "entry_time": t.entry_time.isoformat() if t.entry_time else None,
        "entry_spot": t.entry_spot,
        "expiry": t.expiry.isoformat() if t.expiry else None,
        "sell_strike": t.sell_strike,
        "buy_strike": t.buy_strike,
        "ic_call_sell": t.ic_call_sell,
        "ic_call_buy": t.ic_call_buy,
        "num_lots": t.num_lots,
        "credit_received": t.credit_received,
        "total_credit": t.total_credit,
        "status": t.status,
        "current_spread_value": t.current_spread_value,
        "current_pnl": t.current_pnl,
        "current_pnl_pct": t.current_pnl_pct,
        "unrealized_pnl": t.unrealized_pnl,
        "exit_date": t.exit_date.isoformat() if t.exit_date else None,
        "exit_time": t.exit_time.isoformat() if t.exit_time else None,
        "exit_spot": t.exit_spot,
        "exit_reason": t.exit_reason,
        "realized_pnl": t.realized_pnl,
        "position_size_pct": t.position_size_pct,
        "graduated_mult": t.graduated_mult,
        "capital_deployed": t.capital_deployed,
        "holding_days": (
            (t.exit_date - t.entry_date).days
            if t.exit_date and t.entry_date else None
        ),
        # Bear debit fields (v6.2+)
        "is_bear_debit": t.is_bear_debit,
        "bear_tier": t.bear_tier,
        "entry_debit": t.entry_debit,
        "predicted_drawdown": t.predicted_drawdown,
        "max_profit": t.max_profit,
        "max_loss_amount": t.max_loss_amount,
    }
    return d
