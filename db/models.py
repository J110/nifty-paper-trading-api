"""
SQLAlchemy 2.0 ORM models for the Nifty paper-trading system.
"""

from datetime import date, datetime
from typing import Optional

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Float,
    Integer,
    JSON,
    String,
    UniqueConstraint,
    ForeignKey,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


# ------------------------------------------------------------------
# 1. Prediction
# ------------------------------------------------------------------
class Prediction(Base):
    __tablename__ = "predictions"
    __table_args__ = (UniqueConstraint("date", "version", name="uq_prediction_date_version"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    date: Mapped[date] = mapped_column(Date, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    predicted_drawdown: Mapped[float] = mapped_column(Float, nullable=False)
    signal_type: Mapped[str] = mapped_column(String(20), nullable=False)
    version: Mapped[str] = mapped_column(String(10), nullable=False)
    nifty_spot: Mapped[float] = mapped_column(Float, nullable=False)
    vix: Mapped[float] = mapped_column(Float, nullable=False)
    confidence_score: Mapped[float] = mapped_column(Float, nullable=False)
    graduated_mult: Mapped[float] = mapped_column(Float, default=1.0)
    features: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)


# ------------------------------------------------------------------
# 2. Trade
# ------------------------------------------------------------------
class Trade(Base):
    __tablename__ = "trades"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    trade_id: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    version: Mapped[str] = mapped_column(String(10), nullable=False)
    date: Mapped[date] = mapped_column(Date, nullable=False)
    signal_type: Mapped[str] = mapped_column(String(20), nullable=False)
    trade_type: Mapped[str] = mapped_column(String(20), nullable=False)
    entry_mode: Mapped[str] = mapped_column(String(30), nullable=False)
    entry_date: Mapped[date] = mapped_column(Date, nullable=False)
    entry_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    entry_spot: Mapped[float] = mapped_column(Float, nullable=False)
    expiry: Mapped[date] = mapped_column(Date, nullable=False)
    sell_strike: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    buy_strike: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    ic_call_sell: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    ic_call_buy: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    num_lots: Mapped[int] = mapped_column(Integer, nullable=False)
    credit_received: Mapped[float] = mapped_column(Float, nullable=False)
    total_credit: Mapped[float] = mapped_column(Float, nullable=False)
    status: Mapped[str] = mapped_column(String(15), default="open")
    current_spread_value: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    current_pnl: Mapped[float] = mapped_column(Float, default=0)
    current_pnl_pct: Mapped[float] = mapped_column(Float, default=0)
    unrealized_pnl: Mapped[float] = mapped_column(Float, default=0)
    exit_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    exit_time: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    exit_spot: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    exit_reason: Mapped[Optional[str]] = mapped_column(String(30), nullable=True)
    realized_pnl: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    position_size_pct: Mapped[float] = mapped_column(Float, nullable=False)
    graduated_mult: Mapped[float] = mapped_column(Float, default=1.0)
    capital_deployed: Mapped[float] = mapped_column(Float, nullable=False)
    # Bear debit spread fields (v6.2+)
    is_bear_debit: Mapped[bool] = mapped_column(Boolean, default=False)
    bear_tier: Mapped[int] = mapped_column(Integer, default=0)
    entry_debit: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    predicted_drawdown: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    max_profit: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    max_loss_amount: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    bear_trail_high: Mapped[float] = mapped_column(Float, default=0.0)
    # Stop loss confirmation (require N consecutive CALENDAR days of breach)
    stop_loss_breach_days: Mapped[int] = mapped_column(Integer, default=0)
    stop_loss_last_breach_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    # Trailing stop peak tracking (persisted, not in-memory)
    peak_pnl_pct: Mapped[float] = mapped_column(Float, default=0.0)
    # Trailing stop activation (persistent — once True, stays True for the trade)
    trailing_stop_active: Mapped[bool] = mapped_column(Boolean, default=False)
    # VIX at entry (for VIX spike exit: exit if VIX > threshold AND VIX > entry_vix * 1.3)
    entry_vix: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


# ------------------------------------------------------------------
# 3. DelayPrice
# ------------------------------------------------------------------
class DelayPrice(Base):
    __tablename__ = "delay_prices"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    trade_id: Mapped[str] = mapped_column(
        String(50), ForeignKey("trades.trade_id"), nullable=False
    )
    version: Mapped[str] = mapped_column(String(10), nullable=False)
    signal_date: Mapped[date] = mapped_column(Date, nullable=False)
    signal_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    price_at_signal: Mapped[float] = mapped_column(Float, nullable=False)
    price_at_10min: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    price_at_1hr: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    price_at_3hr: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    price_at_6hr: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    price_at_12hr: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    spread_price_at_signal: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    spread_price_at_10min: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    spread_price_at_1hr: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    spread_price_at_3hr: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    spread_price_at_6hr: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    spread_price_at_12hr: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    pnl_delta_10min: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    pnl_delta_1hr: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    pnl_delta_3hr: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    pnl_delta_6hr: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    pnl_delta_12hr: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


# ------------------------------------------------------------------
# 4. PriceSnapshot
# ------------------------------------------------------------------
class PriceSnapshot(Base):
    __tablename__ = "price_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), unique=True, nullable=False
    )
    nifty_spot: Mapped[float] = mapped_column(Float, nullable=False)
    nifty_high: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    nifty_low: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    vix: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    total_oi_put: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    total_oi_call: Mapped[Optional[float]] = mapped_column(Float, nullable=True)


# ------------------------------------------------------------------
# 5. DailyFeature
# ------------------------------------------------------------------
class DailyFeature(Base):
    __tablename__ = "daily_features"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    date: Mapped[date] = mapped_column(Date, unique=True, nullable=False)
    features: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    vix: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    vix_20d_avg: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    nifty_20d_return: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    nifty_50d_return: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    iv_skew: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    fii_net: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    dii_net: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    put_call_ratio: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    rsi_14: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    adx_14: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


# ------------------------------------------------------------------
# 6. DailyPnl
# ------------------------------------------------------------------
class DailyPnl(Base):
    __tablename__ = "daily_pnl"
    __table_args__ = (UniqueConstraint("date", "version", name="uq_daily_pnl_date_version"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    date: Mapped[date] = mapped_column(Date, nullable=False)
    version: Mapped[str] = mapped_column(String(10), nullable=False)
    starting_capital: Mapped[float] = mapped_column(Float, nullable=False)
    ending_capital: Mapped[float] = mapped_column(Float, nullable=False)
    daily_pnl: Mapped[float] = mapped_column(Float, nullable=False)
    cumulative_pnl: Mapped[float] = mapped_column(Float, nullable=False)
    cumulative_return_pct: Mapped[float] = mapped_column(Float, nullable=False)
    open_positions: Mapped[int] = mapped_column(Integer, nullable=False)
    trades_opened: Mapped[int] = mapped_column(Integer, default=0)
    trades_closed: Mapped[int] = mapped_column(Integer, default=0)
