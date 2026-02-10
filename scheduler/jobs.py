"""
Scheduled tasks for the paper trading system.
Uses APScheduler for cron-based job execution.
Includes startup recovery: if the server restarts after 9:20 AM IST
and today's prediction was missed, it auto-runs the pipeline.
"""

import logging
from datetime import datetime, date, time as dtime
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from sqlalchemy import select, func

from db.database import async_session_factory
from db.models import Prediction, DailyFeature, DailyPnl, PriceSnapshot
from core.timezone import now_ist, today_ist
from core.dhan_client import DhanClient
from core.model_runner import ModelRunner
from core.signal_mapper import map_signal, get_classification_breakdown
from core.feature_engine import build_live_features
from core.trade_manager import TradeManager
from core.price_tracker import PriceTracker
from core.email_notifier import send_trade_alert, send_no_trade_alert, send_exit_alert
from config import (
    ACTIVE_VERSIONS, VERSION_CONFIGS, INITIAL_CAPITAL,
    DOWNSIDE_MODEL_PATH, SCALER_PATH, FEATURE_NAMES_PATH,
)

logger = logging.getLogger(__name__)

# Global instances (initialized on startup)
dhan_client = DhanClient()
model_runner = ModelRunner(DOWNSIDE_MODEL_PATH, SCALER_PATH, FEATURE_NAMES_PATH)
trade_manager = TradeManager()
price_tracker = PriceTracker()


def setup_scheduler() -> AsyncIOScheduler:
    """Configure and return the APScheduler instance."""
    scheduler = AsyncIOScheduler(timezone="Asia/Kolkata")

    # 1. Fetch live prices every 5 minutes during market hours
    scheduler.add_job(
        record_price_snapshot,
        CronTrigger(
            day_of_week="mon-fri",
            hour="9-15",
            minute="*/5",
            timezone="Asia/Kolkata",
        ),
        id="price_snapshot",
        replace_existing=True,
    )

    # 2. Generate daily predictions at 9:20 AM IST
    scheduler.add_job(
        generate_daily_predictions,
        CronTrigger(
            day_of_week="mon-fri",
            hour=9, minute=20,
            timezone="Asia/Kolkata",
        ),
        id="daily_predictions",
        replace_existing=True,
    )

    # 3. Check exits every 5 minutes during market hours
    scheduler.add_job(
        check_all_exits,
        CronTrigger(
            day_of_week="mon-fri",
            hour="9-15",
            minute="*/5",
            timezone="Asia/Kolkata",
        ),
        id="check_exits",
        replace_existing=True,
    )

    # 4. EOD processing at 3:35 PM IST
    scheduler.add_job(
        eod_processing,
        CronTrigger(
            day_of_week="mon-fri",
            hour=15, minute=35,
            timezone="Asia/Kolkata",
        ),
        id="eod_processing",
        replace_existing=True,
    )

    logger.info("Scheduler configured with all jobs")
    return scheduler


async def check_and_recover_missed_prediction():
    """
    Startup recovery: check if today's daily prediction was missed.
    If the server restarted after 9:20 AM IST on a weekday and no
    prediction exists for today, auto-run the prediction pipeline.
    """
    try:
        current_time = now_ist()

        # Only recover on weekdays during market hours (9:20 AM - 3:30 PM)
        if current_time.weekday() >= 5:  # Weekend
            logger.info("Startup recovery: weekend — skipping")
            return

        if current_time.time() < dtime(9, 20):
            logger.info("Startup recovery: before 9:20 AM — scheduler will handle it")
            return

        if current_time.time() > dtime(15, 30):
            logger.info("Startup recovery: after market close — too late to recover")
            return

        # Check if today's prediction exists in DB
        today = current_time.date()
        async with async_session_factory() as db:
            result = await db.execute(
                select(func.count()).select_from(Prediction).where(
                    Prediction.date == today
                )
            )
            count = result.scalar() or 0

        if count > 0:
            logger.info(
                f"Startup recovery: {count} predictions found for {today} — no action needed"
            )
            return

        # No predictions for today — run the pipeline now
        logger.warning(
            f"Startup recovery: NO predictions for {today} and it's {current_time.strftime('%H:%M')} IST. "
            f"Running missed prediction pipeline..."
        )
        await generate_daily_predictions()
        logger.info("Startup recovery: prediction pipeline completed")

    except Exception as e:
        logger.error(f"Startup recovery check failed: {e}", exc_info=True)


async def record_price_snapshot():
    """Record Nifty spot + VIX snapshot every 5 minutes."""
    try:
        spot = await dhan_client.get_nifty_ltp()
        vix = await dhan_client.get_india_vix()

        if spot is None:
            logger.warning("Failed to get Nifty LTP for snapshot")
            return

        async with async_session_factory() as db:
            await price_tracker.record_price_snapshot(
                spot=spot, vix=vix, high=spot, low=spot, db=db
            )

        logger.debug(f"Price snapshot: Nifty={spot}, VIX={vix}")
    except Exception as e:
        logger.error(f"Price snapshot failed: {e}")


async def generate_daily_predictions():
    """
    Daily prediction pipeline at 9:20 AM IST:
    1. Fetch current Nifty, VIX, option chain from Dhan
    2. Build feature vector
    3. Run model prediction
    4. Map prediction to signal for each version
    5. If signal != no_trade: open paper trade + send email alert
    6. Schedule delay price recordings
    7. Store prediction + features in DB
    8. If all versions = no_trade: send no-trade email
    """
    logger.info("=== Starting daily prediction pipeline ===")

    try:
        # Fetch live data
        spot = await dhan_client.get_nifty_ltp()
        vix = await dhan_client.get_india_vix()

        if spot is None:
            logger.error("Cannot generate predictions: no Nifty price")
            return

        logger.info(f"Live data: Nifty={spot}, VIX={vix}")

        # Get historical data for feature computation
        import pandas as pd
        historical_raw = await dhan_client.get_historical_ohlc(days=365)

        # Parse historical data into DataFrame
        if historical_raw and isinstance(historical_raw, dict):
            ohlc_data = historical_raw.get("data", [])
        elif isinstance(historical_raw, list):
            ohlc_data = historical_raw
        else:
            ohlc_data = []

        if ohlc_data:
            historical_df = pd.DataFrame(ohlc_data)
            # Normalize column names
            col_map = {
                "open": "open", "high": "high", "low": "low",
                "close": "close", "volume": "volume",
                "start_Time": "date", "timestamp": "date",
            }
            historical_df.rename(columns=col_map, inplace=True)
        else:
            # Fallback: use yfinance
            import yfinance as yf
            nifty = yf.download("^NSEI", period="1y", progress=False)
            historical_df = nifty.reset_index()
            historical_df.columns = [c.lower() for c in historical_df.columns]

        # Build features
        option_chain = None
        try:
            from core.option_pricer import get_next_weekly_expiry
            expiry = get_next_weekly_expiry()
            option_chain = await dhan_client.get_option_chain(expiry.isoformat())
        except Exception as e:
            logger.warning(f"Failed to get option chain: {e}")

        features = await build_live_features(
            dhan_client, historical_df, option_chain
        )

        # Run model prediction
        prediction_value = model_runner.predict(features)
        logger.info(f"Model prediction: {prediction_value:.4f} ({prediction_value*100:.2f}%)")

        # Track trades opened and all signals (for no-trade email)
        trades_opened = []
        all_version_signals = []

        # Store in DB
        async with async_session_factory() as db:
            # Store features
            daily_feature = DailyFeature(
                date=today_ist(),
                features=features,
                vix=features.get("vix"),
                vix_20d_avg=features.get("vix_20d_avg"),
                nifty_20d_return=features.get("nifty_20d_return"),
                nifty_50d_return=features.get("nifty_50d_return"),
                iv_skew=features.get("iv_skew"),
                fii_net=features.get("fii_net_5d"),
                dii_net=features.get("dii_net_5d"),
                put_call_ratio=features.get("put_call_ratio"),
                rsi_14=features.get("rsi_14"),
                adx_14=features.get("adx_14"),
            )
            db.add(daily_feature)

            # Process each version
            for version in ACTIVE_VERSIONS:
                cfg = VERSION_CONFIGS[version]
                signal = map_signal(prediction_value, cfg)

                # Track for no-trade check
                all_version_signals.append({
                    "version": version,
                    "signal": signal["signal"],
                })

                # Compute confidence score (distance from nearest boundary)
                breakdown = get_classification_breakdown(prediction_value, version=version)
                active_zone = next(
                    (z for z in breakdown["zones"] if z.get("confidence") == "active"),
                    None
                )
                confidence = active_zone["distance_to_boundary"] if active_zone else 0

                # Store prediction
                pred = Prediction(
                    date=today_ist(),
                    timestamp=now_ist(),
                    predicted_drawdown=prediction_value,
                    signal_type=signal["signal"],
                    version=version,
                    nifty_spot=spot,
                    vix=vix,
                    confidence_score=confidence,
                    graduated_mult=signal["size_mult"],
                    features=features,
                )
                db.add(pred)

                # Open trade if signal says so
                if signal["signal"] != "no_trade":
                    trade_result = await trade_manager.open_trade(
                        signal=signal,
                        version=version,
                        spot=spot,
                        vix=vix,
                        db=db,
                        dhan_client=dhan_client,
                        predicted_drawdown=prediction_value,
                    )

                    if trade_result:
                        # Record initial delay price
                        spread_price = trade_result.get("credit", 0) or trade_result.get("debit", 0)
                        await price_tracker.record_initial_price(
                            trade_id=trade_result["trade_id"],
                            version=version,
                            signal_time=now_ist(),
                            spot=spot,
                            spread_price=spread_price,
                            db=db,
                        )
                        logger.info(
                            f"[{version}] Opened trade: {trade_result['trade_id']}"
                        )

                        # ===== EMAIL: Trade opened =====
                        trades_opened.append((version, signal, trade_result))

            await db.commit()

        # ===== Send email notifications (outside DB transaction) =====
        if trades_opened:
            for version, signal, trade_result in trades_opened:
                try:
                    await send_trade_alert(
                        version=version,
                        signal=signal,
                        trade_result=trade_result,
                        spot=spot,
                        vix=vix,
                        prediction=prediction_value,
                    )
                except Exception as e:
                    logger.error(f"Failed to send trade alert email for {version}: {e}")
        else:
            # All versions produced no_trade — send daily no-trade summary
            try:
                await send_no_trade_alert(
                    prediction=prediction_value,
                    spot=spot,
                    vix=vix,
                    version_signals=all_version_signals,
                )
            except Exception as e:
                logger.error(f"Failed to send no-trade alert email: {e}")

        logger.info("=== Daily prediction pipeline complete ===")

    except Exception as e:
        logger.error(f"Daily prediction pipeline failed: {e}", exc_info=True)


async def check_all_exits():
    """Check all open positions for exit conditions."""
    try:
        spot = await dhan_client.get_nifty_ltp()
        vix = await dhan_client.get_india_vix()

        if spot is None:
            return

        async with async_session_factory() as db:
            for version in ACTIVE_VERSIONS:
                closed_trades = await trade_manager.check_exits(
                    version, spot, vix, db
                )

                # ===== EMAIL: Trade closed =====
                if closed_trades:
                    for ct in closed_trades:
                        try:
                            await send_exit_alert(
                                version=version,
                                trade_id=ct["trade_id"],
                                exit_reason=ct["exit_reason"],
                                trade_type=ct["trade_type"],
                                entry_spot=ct["entry_spot"],
                                exit_spot=spot,
                                realized_pnl=ct["realized_pnl"],
                                pnl_pct=ct["pnl_pct"],
                                sell_strike=ct["sell_strike"],
                                buy_strike=ct["buy_strike"],
                            )
                        except Exception as e:
                            logger.error(
                                f"Failed to send exit alert for {ct['trade_id']}: {e}"
                            )

    except Exception as e:
        logger.error(f"Exit check failed: {e}")


async def eod_processing():
    """End-of-day processing: final mark-to-market, record daily PnL."""
    logger.info("=== Starting EOD processing ===")

    try:
        spot = await dhan_client.get_nifty_ltp()
        vix = await dhan_client.get_india_vix()

        if spot is None:
            logger.error("Cannot do EOD: no Nifty price")
            return

        async with async_session_factory() as db:
            for version in ACTIVE_VERSIONS:
                # Final exit check
                closed_trades = await trade_manager.check_exits(
                    version, spot, vix, db
                )

                # Send exit emails for EOD closures too
                if closed_trades:
                    for ct in closed_trades:
                        try:
                            await send_exit_alert(
                                version=version,
                                trade_id=ct["trade_id"],
                                exit_reason=ct["exit_reason"],
                                trade_type=ct["trade_type"],
                                entry_spot=ct["entry_spot"],
                                exit_spot=spot,
                                realized_pnl=ct["realized_pnl"],
                                pnl_pct=ct["pnl_pct"],
                                sell_strike=ct["sell_strike"],
                                buy_strike=ct["buy_strike"],
                            )
                        except Exception as e:
                            logger.error(
                                f"Failed to send EOD exit alert for {ct['trade_id']}: {e}"
                            )

                # Get portfolio state
                portfolio = await trade_manager.get_portfolio_state(version, db)

                # Record daily PnL
                from sqlalchemy import select
                result = await db.execute(
                    select(DailyPnl).where(
                        DailyPnl.date == today_ist(),
                        DailyPnl.version == version,
                    )
                )
                existing = result.scalar_one_or_none()

                if existing:
                    existing.ending_capital = portfolio["current_capital"]
                    existing.daily_pnl = portfolio["total_pnl"]
                    existing.cumulative_pnl = portfolio["total_pnl"]
                    existing.cumulative_return_pct = portfolio["total_return_pct"]
                    existing.open_positions = portfolio["open_positions"]
                else:
                    daily_pnl = DailyPnl(
                        date=today_ist(),
                        version=version,
                        starting_capital=INITIAL_CAPITAL,
                        ending_capital=portfolio["current_capital"],
                        daily_pnl=portfolio["total_pnl"],
                        cumulative_pnl=portfolio["total_pnl"],
                        cumulative_return_pct=portfolio["total_return_pct"],
                        open_positions=portfolio["open_positions"],
                    )
                    db.add(daily_pnl)

            await db.commit()

        logger.info("=== EOD processing complete ===")

    except Exception as e:
        logger.error(f"EOD processing failed: {e}", exc_info=True)
