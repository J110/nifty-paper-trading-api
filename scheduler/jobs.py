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
from sqlalchemy import select, func, delete

from db.database import async_session_factory
from db.models import Prediction, DailyFeature, DailyPnl, PriceSnapshot
from core.timezone import now_ist, today_ist
from core.dhan_client import DhanClient
from core.model_runner import ModelRunner
from core.signal_mapper import map_signal, get_classification_breakdown
from core.feature_engine import build_live_features, StaleDataError
from core.trade_manager import TradeManager
from core.price_tracker import PriceTracker
from core.email_notifier import (
    send_trade_alert, send_no_trade_alert, send_exit_alert,
    send_data_stale_alert, send_pipeline_failure_alert,
)
from core.data_updater import update_merged_daily
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

    # 0a. Renew Dhan token at 8:30 AM IST (before any API calls)
    #     Token expires every 24h — this extends it for another 24h.
    scheduler.add_job(
        renew_dhan_token,
        CronTrigger(
            day_of_week="mon-fri",
            hour=8, minute=30,
            timezone="Asia/Kolkata",
        ),
        id="token_renewal",
        replace_existing=True,
    )

    # 0b. Update merged_daily.parquet at 9:05 AM IST (before predictions at 9:20)
    #     Downloads latest data from yfinance so features use yesterday's close.
    scheduler.add_job(
        run_data_update,
        CronTrigger(
            day_of_week="mon-fri",
            hour=9, minute=5,
            timezone="Asia/Kolkata",
        ),
        id="data_update",
        replace_existing=True,
    )

    # 1. Fetch live prices every 15 minutes during market hours (reduced from 5min to avoid Dhan 429)
    scheduler.add_job(
        record_price_snapshot,
        CronTrigger(
            day_of_week="mon-fri",
            hour="9-15",
            minute="*/15",
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

    # 3. Check exits once daily at 3:25 PM IST (matches backtest EOD check)
    # Backtest checks once per day at close. Using 3:25 PM (5 min before close)
    # to match the backtest's daily exit evaluation frequency.
    scheduler.add_job(
        check_all_exits,
        CronTrigger(
            day_of_week="mon-fri",
            hour=15, minute=25,
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

    logger.info("Scheduler configured with all jobs (token→data→predictions→exits→eod)")
    return scheduler


async def renew_dhan_token():
    """
    Scheduled job: renew Dhan API token for another 24 hours.

    Runs at 8:30 AM IST daily (before any market data calls).
    The token must still be active (not expired) for renewal to work.
    If renewal fails, sends an alert — live spot/VIX/option chain
    will be unavailable for the rest of the day.
    """
    logger.info("=== Starting Dhan token renewal ===")
    try:
        result = await dhan_client.renew_token()
        expiry = result.get("expiryTime", "unknown")
        logger.info(f"Token renewal SUCCESS — new token expires: {expiry}")
    except Exception as e:
        logger.error(f"Token renewal FAILED: {e}")
        try:
            await send_pipeline_failure_alert(
                pipeline_name="Dhan Token Renewal",
                error_msg=str(e),
                stage="token_renewal",
            )
        except Exception:
            pass
    logger.info("=== Dhan token renewal complete ===")


async def run_data_update():
    """
    Scheduled job: update merged_daily.parquet with latest Yahoo Finance data.

    Runs at 9:05 AM IST, 15 minutes before the prediction pipeline.
    This ensures yesterday's close prices are available for feature computation.
    Failures are logged and trigger an email alert. If the parquet remains
    >3 calendar days stale, the prediction pipeline will REFUSE to run
    (raises StaleDataError) to prevent opening trades on wrong signals.
    """
    logger.info("=== Starting daily data update ===")
    try:
        result = await update_merged_daily()
        if result["status"] == "updated":
            logger.info(
                f"Data update SUCCESS: added {result['rows_added']} rows, "
                f"last date now {result['last_date_after']}"
            )
        elif result["status"] == "already_current":
            logger.info(f"Data already current (last: {result['last_date_after']})")
        elif result["status"] in ("no_new_data", "no_new_dates"):
            # Data update didn't add anything — check if this means stale data
            last_date = result.get("last_date_after", "unknown")
            today = today_ist()
            if last_date != "unknown":
                from datetime import date
                last_dt = date.fromisoformat(last_date)
                days_behind = (today - last_dt).days
                if days_behind > 3:
                    logger.error(
                        f"STALE DATA: parquet at {last_date}, {days_behind} days behind!"
                    )
                    await send_data_stale_alert(
                        last_parquet_date=last_date,
                        update_status=result["status"],
                    )
                else:
                    logger.info(f"No new data (last: {last_date}) — normal for weekends/holidays")
        else:
            # Unexpected status (including "error") — send alert
            logger.error(f"Data update unexpected result: {result}")
            await send_data_stale_alert(
                last_parquet_date=result.get("last_date_after", "unknown"),
                update_status=result.get("status", "unknown"),
                error_msg=f"Unexpected data update result: {result}",
            )
    except Exception as e:
        logger.error(f"Data update failed: {e}", exc_info=True)
        # Send stale data alert on exception
        try:
            await send_data_stale_alert(
                last_parquet_date="unknown",
                update_status="exception",
                error_msg=str(e),
            )
        except Exception:
            pass  # Don't let alert failure mask the original error
    logger.info("=== Daily data update complete ===")


async def check_and_recover_missed_prediction():
    """
    Startup recovery: check if today's daily prediction was missed.
    If the server restarted after 9:20 AM IST on a weekday and no
    prediction exists for today, auto-run the prediction pipeline.

    Also runs data update first to ensure parquet is fresh.
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

        # Renew Dhan token before any API calls
        logger.info("Startup recovery: renewing Dhan token...")
        try:
            await dhan_client.renew_token()
            logger.info("Startup token renewal succeeded")
        except Exception as e:
            logger.error(f"Startup token renewal failed: {e}")
            try:
                await send_pipeline_failure_alert(
                    pipeline_name="Startup Token Renewal",
                    error_msg=str(e),
                    stage="startup_recovery",
                )
            except Exception:
                pass

        # Update parquet data before running prediction recovery
        logger.info("Startup recovery: updating parquet data first...")
        try:
            update_result = await update_merged_daily()
            logger.info(f"Startup data update: {update_result.get('status')}, "
                        f"last date: {update_result.get('last_date_after')}")
        except Exception as e:
            logger.error(f"Startup data update failed: {e}")
            try:
                await send_data_stale_alert(
                    last_parquet_date="unknown",
                    update_status="startup_exception",
                    error_msg=f"Startup data update failed: {e}",
                )
            except Exception:
                pass

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
        try:
            await generate_daily_predictions()
            logger.info("Startup recovery: prediction pipeline completed")
        except StaleDataError as e:
            logger.error(
                f"Startup recovery BLOCKED — data too stale: {e}. "
                f"No predictions generated. Fix data pipeline and trigger manually."
            )
            # StaleDataError alert is already sent inside generate_daily_predictions()

    except Exception as e:
        logger.error(f"Startup recovery check failed: {e}", exc_info=True)
        try:
            await send_pipeline_failure_alert(
                pipeline_name="Startup Recovery",
                error_msg=str(e),
                stage="startup_recovery",
            )
        except Exception:
            pass


async def record_price_snapshot():
    """Record Nifty spot + VIX snapshot every 15 minutes."""
    try:
        spot = await dhan_client.get_nifty_ltp()
        vix = await dhan_client.get_india_vix()

        if spot is None:
            logger.warning("Failed to get Nifty LTP for snapshot — Dhan API may be down")
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
        # Fetch live data (best-effort — features come from parquet, not live prices)
        spot = await dhan_client.get_nifty_ltp()
        vix = await dhan_client.get_india_vix()

        logger.info(f"Live data: Nifty={spot}, VIX={vix}")

        # Build features from parquet (historical_df is not used for features)
        # The feature engine reads directly from merged_daily.parquet
        import pandas as pd
        historical_df = pd.DataFrame()  # Not used by build_live_features

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

        # Fallback: use parquet data if live prices unavailable
        if spot is None:
            spot = features.get("nifty_close", 0)
            logger.warning(f"Using parquet close as spot fallback: {spot}")
        if vix is None:
            vix = features.get("india_vix", 0)
            logger.warning(f"Using parquet VIX as vix fallback: {vix}")

        # Run model prediction
        prediction_value = model_runner.predict(features)
        logger.info(f"Model prediction: {prediction_value:.4f} ({prediction_value*100:.2f}%)")

        # Track trades opened and all signals (for no-trade email)
        trades_opened = []
        all_version_signals = []

        # Store in DB
        async with async_session_factory() as db:
            try:
                # Upsert daily features (delete old row if exists, then insert)
                await db.execute(
                    delete(DailyFeature).where(DailyFeature.date == today_ist())
                )
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
                await db.flush()  # Flush features first to detect errors early
                logger.info("Daily features stored")

                # Clean up any existing predictions for today (from failed earlier runs)
                await db.execute(
                    delete(Prediction).where(Prediction.date == today_ist())
                )

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
                    logger.info(f"[{version}] signal={signal['signal']}, size={signal['size_mult']}")

                await db.flush()  # Flush all predictions
                logger.info(f"Stored {len(ACTIVE_VERSIONS)} predictions")

                # Open trades (in separate try-except so prediction storage is not affected)
                for version in ACTIVE_VERSIONS:
                    cfg = VERSION_CONFIGS[version]
                    signal = map_signal(prediction_value, cfg)

                    if signal["signal"] != "no_trade":
                        try:
                            trade_result = await trade_manager.open_trade(
                                signal=signal,
                                version=version,
                                spot=spot,
                                vix=vix,
                                db=db,
                                dhan_client=dhan_client,
                                predicted_drawdown=prediction_value,
                            )

                            if trade_result and isinstance(trade_result, dict):
                                spread_price = trade_result.get("credit", 0) or trade_result.get("debit", 0)
                                await price_tracker.record_initial_price(
                                    trade_id=trade_result["trade_id"],
                                    version=version,
                                    signal_time=now_ist(),
                                    spot=spot,
                                    spread_price=spread_price,
                                    db=db,
                                )
                                logger.info(f"[{version}] Opened trade: {trade_result['trade_id']}")
                                trades_opened.append((version, signal, trade_result))
                        except Exception as e:
                            logger.error(f"[{version}] Failed to open trade: {e}")
                            try:
                                await send_pipeline_failure_alert(
                                    pipeline_name="Trade Opening",
                                    error_msg=f"[{version}] {e}",
                                    stage="open_trade",
                                )
                            except Exception:
                                pass

                await db.commit()
                logger.info("DB commit successful")

            except Exception as e:
                logger.error(f"DB transaction failed: {e}", exc_info=True)
                await db.rollback()
                raise

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

    except StaleDataError as e:
        # HARD BLOCK: data is too stale — do NOT generate predictions or open trades
        logger.error(f"PREDICTION BLOCKED — STALE DATA: {e}")
        try:
            await send_data_stale_alert(
                last_parquet_date=str(e),
                update_status="prediction_blocked",
                error_msg=(
                    "The prediction pipeline was BLOCKED because the data is too stale. "
                    "No predictions were generated and no trades were opened. "
                    "Fix the data update pipeline (check Yahoo API / Dhan token) "
                    "and trigger a manual data update + prediction via the API."
                ),
            )
        except Exception:
            pass
        raise  # Re-raise so callers can see the error

    except Exception as e:
        logger.error(f"Daily prediction pipeline failed: {e}", exc_info=True)
        # Send pipeline failure email
        try:
            await send_pipeline_failure_alert(
                pipeline_name="Daily Prediction Pipeline",
                error_msg=str(e),
                stage="prediction_generation",
            )
        except Exception:
            pass  # Don't let alert failure mask the original error
        raise  # Re-raise so callers (manual trigger) can see the error


async def check_all_exits():
    """Check all open positions for exit conditions."""
    try:
        spot = await dhan_client.get_nifty_ltp()
        vix = await dhan_client.get_india_vix()

        if spot is None:
            logger.error("EXIT CHECK FAILED: Cannot get Nifty spot price — Dhan API down")
            await send_pipeline_failure_alert(
                pipeline_name="Exit Check",
                error_msg="Cannot get Nifty spot price (Dhan API returned None). Exit checks SKIPPED — open trades were NOT evaluated for stop-loss or expiry.",
                stage="check_exits",
            )
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
        logger.error(f"Exit check failed: {e}", exc_info=True)
        try:
            await send_pipeline_failure_alert(
                pipeline_name="Exit Check",
                error_msg=str(e),
                stage="check_exits",
            )
        except Exception:
            pass


async def eod_processing():
    """End-of-day processing: final mark-to-market, record daily PnL."""
    logger.info("=== Starting EOD processing ===")

    try:
        spot = await dhan_client.get_nifty_ltp()
        vix = await dhan_client.get_india_vix()

        if spot is None:
            logger.error("EOD FAILED: Cannot get Nifty spot price — Dhan API down")
            await send_pipeline_failure_alert(
                pipeline_name="EOD Processing",
                error_msg="Cannot get Nifty spot price (Dhan API returned None). EOD processing SKIPPED — no PnL recorded, no final exit checks.",
                stage="eod_processing",
            )
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
        try:
            await send_pipeline_failure_alert(
                pipeline_name="EOD Processing",
                error_msg=str(e),
                stage="eod_processing",
            )
        except Exception:
            pass
