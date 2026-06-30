"""
Nifty Paper Trading API — FastAPI Application

Serves predictions, trades, and analytics for 3 model versions (v5.4.2, v5.4.3, v5.4.4).
Runs scheduled jobs for daily predictions and trade management.
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from db.database import init_db
from api.signals import router as signals_router
from api.trades import router as trades_router
from api.charts import router as charts_router
from api.returns import router as returns_router
from scheduler.jobs import setup_scheduler, check_and_recover_missed_prediction, run_data_update

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Scheduler instance (global for cleanup)
scheduler = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    global scheduler

    # Startup
    logger.info("Starting Nifty Paper Trading API...")

    # Initialize database tables
    await init_db()
    logger.info("Database initialized")

    # Ensure real-option-pricing columns exist on the trades table (idempotent).
    try:
        from sqlalchemy import text as _text
        from db.database import async_session_factory as _asf
        async with _asf() as _db:
            for _sql in (
                "ALTER TABLE trades ADD COLUMN IF NOT EXISTS pricing_source VARCHAR(20)",
                "ALTER TABLE trades ADD COLUMN IF NOT EXISTS bs_credit FLOAT",
                "ALTER TABLE trades ADD COLUMN IF NOT EXISTS real_credit FLOAT",
            ):
                await _db.execute(_text(_sql))
            await _db.commit()
        logger.info("Real-pricing columns ensured")
    except Exception as _e:
        logger.warning(f"Pricing-column migration note: {_e}")

    # Eagerly initialize DhanClient so the DB-backed token (which may
    # be fresher than the bootstrap env var) is loaded into memory now,
    # rather than lazily on the first market-data call. Without this,
    # /health and the 06:00/18:00 renewal cron see stale env-var data
    # immediately after a redeploy.
    from scheduler.jobs import dhan_client
    try:
        await dhan_client.start()
    except Exception as exc:
        logger.error(f"Eager DhanClient.start() failed: {exc}")

    # Start scheduler
    scheduler = setup_scheduler()
    scheduler.start()
    logger.info("APScheduler started")

    # On every startup: refresh parquet data immediately, then check
    # for missed predictions. This is critical because Render's ephemeral
    # filesystem resets the parquet to the git version on each deploy,
    # and the Dhan token requires daily redeployment.
    import asyncio

    async def _startup_data_refresh():
        # Wait for port binding first (Render times out if port isn't open quickly)
        await asyncio.sleep(30)
        # Always update parquet on startup — regardless of time/day
        await run_data_update()
        # Then check if predictions were missed
        await check_and_recover_missed_prediction()

    asyncio.create_task(_startup_data_refresh())

    yield

    # Shutdown
    if scheduler:
        scheduler.shutdown()
        logger.info("APScheduler stopped")
    logger.info("Nifty Paper Trading API stopped")


app = FastAPI(
    title="Nifty Paper Trading API",
    description="Paper trading system for Nifty options with 3 model versions",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow Flutter web frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to Vercel domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(signals_router)
app.include_router(trades_router)
app.include_router(charts_router)
app.include_router(returns_router)


@app.api_route("/health", methods=["GET", "HEAD"])
async def health_check():
    """Health check endpoint for Render + UptimeRobot pinger."""
    import os
    from sqlalchemy import select, func
    from db.database import async_session_factory
    from db.models import Prediction
    from core.timezone import now_ist, today_ist

    # Quick DB check: is today's prediction present?
    today_predictions = 0
    last_prediction_date = None
    db_healthy = False
    db_error = None
    try:
        async with async_session_factory() as db:
            result = await db.execute(
                select(func.count()).select_from(Prediction).where(
                    Prediction.date == today_ist()
                )
            )
            today_predictions = result.scalar() or 0

            result2 = await db.execute(
                select(func.max(Prediction.date))
            )
            last_prediction_date = result2.scalar()
            db_healthy = True
    except Exception as e:
        db_error = str(e)
        logger.error(f"Health check DB query failed: {e}")

    scheduler_running = scheduler is not None and scheduler.running
    overall_healthy = db_healthy and scheduler_running

    # Surface Dhan token expiry so we can spot stale tokens before they break trades.
    from datetime import datetime
    from scheduler.jobs import dhan_client
    token_exp = dhan_client.token_expiry()
    token_exp_iso = token_exp.isoformat() if token_exp else None
    token_hours_left = (
        round((token_exp - datetime.now(token_exp.tzinfo)).total_seconds() / 3600, 1)
        if token_exp else None
    )

    return {
        "status": "healthy" if overall_healthy else "degraded",
        "service": "nifty-paper-trading-api",
        "version": "1.0.0",
        "db_healthy": db_healthy,
        "db_error": db_error,
        "scheduler_running": scheduler_running,
        "today_predictions": today_predictions,
        "last_prediction_date": last_prediction_date.isoformat() if last_prediction_date else None,
        "dhan_token_expiry_utc": token_exp_iso,
        "dhan_token_hours_left": token_hours_left,
        "server_time_ist": now_ist().isoformat(),
        "deployed_commit": (os.environ.get("RENDER_GIT_COMMIT") or "local")[:8],
    }


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Nifty Paper Trading API",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "signals": "/api/signals/current",
            "trades": "/api/trades/{version}",
            "delay_analysis": "/api/trades/{version}/delay-analysis",
            "returns": "/api/trades/{version}/returns",
            "chart_nifty": "/api/chart-data/nifty",
            "chart_equity": "/api/chart-data/equity/{version}",
            "indicators": "/api/indicators",
            "trigger": "/api/trigger-prediction (POST)",
            "renew_token": "/api/renew-token (POST)",
            "update_data": "/api/update-data (POST)",
        },
    }


@app.post("/api/trigger-prediction")
async def trigger_prediction():
    """
    Manually trigger the daily prediction pipeline.
    Runs synchronously so errors are returned in the response.
    """
    from scheduler.jobs import generate_daily_predictions
    from db.database import async_session_factory
    from db.models import Prediction
    from core.timezone import today_ist, now_ist
    from sqlalchemy import select, func
    import traceback

    logger.info("Manual prediction trigger received")

    # Count predictions before
    before_count = 0
    try:
        async with async_session_factory() as db:
            result = await db.execute(
                select(func.count()).select_from(Prediction).where(
                    Prediction.date == today_ist()
                )
            )
            before_count = result.scalar() or 0
    except Exception:
        pass

    error_info = None
    try:
        await generate_daily_predictions()
    except Exception as e:
        logger.error(f"Manual trigger failed: {e}", exc_info=True)
        error_info = {"error": str(e), "traceback": traceback.format_exc()}

    # Count predictions after
    after_count = 0
    try:
        async with async_session_factory() as db:
            result = await db.execute(
                select(func.count()).select_from(Prediction).where(
                    Prediction.date == today_ist()
                )
            )
            after_count = result.scalar() or 0
    except Exception:
        pass

    result = {
        "status": "error" if error_info else ("success" if after_count > before_count else "warning"),
        "message": f"Pipeline completed. Predictions for {today_ist()}: {before_count} -> {after_count}",
        "today_date": str(today_ist()),
        "server_time": str(now_ist()),
        "predictions_before": before_count,
        "predictions_after": after_count,
    }
    if error_info:
        result.update(error_info)
    return result


@app.post("/api/debug/run-pipeline")
async def debug_run_pipeline():
    """Run the exact pipeline code inline with full error capture at every step."""
    import traceback
    from db.database import async_session_factory
    from db.models import Prediction, DailyFeature
    from core.timezone import today_ist, now_ist
    from core.feature_engine import build_live_features
    from core.model_runner import ModelRunner
    from core.signal_mapper import map_signal
    from core.dhan_client import DhanClient
    from config import DOWNSIDE_MODEL_PATH, SCALER_PATH, FEATURE_NAMES_PATH, ACTIVE_VERSIONS, VERSION_CONFIGS
    from sqlalchemy import delete
    import pandas as pd

    steps = {}
    try:
        dc = DhanClient()
        spot = await dc.get_nifty_ltp()
        vix_live = await dc.get_india_vix()
        steps["1_dhan"] = {"spot": spot, "vix": vix_live}

        features = await build_live_features(dc, pd.DataFrame(), None)
        steps["2_features"] = {"count": len(features)}

        if spot is None:
            spot = features.get("nifty_close", 0)
        if vix_live is None:
            vix_live = features.get("india_vix", 0)
        steps["3_fallback"] = {"spot": float(spot), "vix": float(vix_live)}

        mr = ModelRunner(DOWNSIDE_MODEL_PATH, SCALER_PATH, FEATURE_NAMES_PATH)
        prediction_value = mr.predict(features)
        steps["4_prediction"] = float(prediction_value)

        async with async_session_factory() as db:
            try:
                await db.execute(delete(DailyFeature).where(DailyFeature.date == today_ist()))
                df = DailyFeature(date=today_ist(), features=features, vix=features.get("vix"), rsi_14=features.get("rsi_14"))
                db.add(df)
                await db.flush()
                steps["5a_features_flush"] = "OK"

                await db.execute(delete(Prediction).where(Prediction.date == today_ist()))
                for version in ACTIVE_VERSIONS:
                    cfg = VERSION_CONFIGS[version]
                    signal = map_signal(prediction_value, cfg)
                    pred = Prediction(date=today_ist(), timestamp=now_ist(), predicted_drawdown=prediction_value,
                        signal_type=signal["signal"], version=version, nifty_spot=spot, vix=vix_live,
                        confidence_score=0.25, graduated_mult=signal["size_mult"], features=features)
                    db.add(pred)

                await db.flush()
                steps["5b_predictions_flush"] = "OK"

                await db.commit()
                steps["5c_commit"] = "OK"

            except Exception as e:
                await db.rollback()
                steps["5_db_error"] = str(e)

        steps["status"] = "success"
    except Exception as e:
        steps["fatal_error"] = f"{e}"
        steps["status"] = "failed"

    steps["today"] = str(today_ist())
    return steps


@app.get("/api/debug/pipeline-test")
async def debug_pipeline_test():
    """Test each step of the prediction pipeline independently."""
    import traceback
    import os
    results = {}

    # Step 1: Check parquet file
    try:
        from core.feature_engine import MERGED_DAILY_PATH, DHAN_RAW_PATH, DHAN_SKEW_PATH
        import pandas as pd
        results["parquet_exists"] = os.path.exists(MERGED_DAILY_PATH)
        results["parquet_path"] = MERGED_DAILY_PATH
        if results["parquet_exists"]:
            df = pd.read_parquet(MERGED_DAILY_PATH)
            results["parquet_rows"] = len(df)
            results["parquet_last_date"] = str(df.index[-1].date())
        results["dhan_raw_exists"] = os.path.exists(DHAN_RAW_PATH)
        results["dhan_skew_exists"] = os.path.exists(DHAN_SKEW_PATH)
    except Exception as e:
        results["parquet_error"] = f"{e}\n{traceback.format_exc()}"

    # Step 2: Check Dhan API
    try:
        from core.dhan_client import DhanClient
        dc = DhanClient()
        spot = await dc.get_nifty_ltp()
        vix = await dc.get_india_vix()
        results["dhan_spot"] = spot
        results["dhan_vix"] = vix
    except Exception as e:
        results["dhan_error"] = f"{e}\n{traceback.format_exc()}"

    # Step 3: Build features
    try:
        from core.feature_engine import build_live_features
        features = await build_live_features(dc, None, None)
        results["features_count"] = len(features) if features else 0
        results["features_sample"] = {k: features[k] for k in list(features.keys())[:5]} if features else {}
    except Exception as e:
        results["features_error"] = f"{e}\n{traceback.format_exc()}"

    # Step 4: Model prediction
    try:
        from core.model_runner import ModelRunner
        from config import DOWNSIDE_MODEL_PATH, SCALER_PATH, FEATURE_NAMES_PATH
        mr = ModelRunner(DOWNSIDE_MODEL_PATH, SCALER_PATH, FEATURE_NAMES_PATH)
        if 'features_error' not in results:
            pred = mr.predict(features)
            results["prediction"] = float(pred)
            results["prediction_pct"] = f"{pred*100:.2f}%"
        else:
            results["model_skipped"] = "Features failed"
    except Exception as e:
        results["model_error"] = f"{e}\n{traceback.format_exc()}"

    # Step 5: Check today_ist
    from core.timezone import today_ist, now_ist
    results["today_ist"] = str(today_ist())
    results["now_ist"] = str(now_ist())

    return results


@app.post("/api/renew-token")
async def renew_token_endpoint():
    """
    Manually trigger Dhan token renewal.
    Extends the current token for another 24 hours.
    Token must still be active (not expired) for this to work.
    """
    from scheduler.jobs import dhan_client
    from core.timezone import now_ist
    import traceback

    logger.info("Manual token renewal triggered")
    try:
        result = await dhan_client.renew_token()
        return {
            "status": "renewed",
            "expiry": result.get("expiryTime"),
            "client_id": result.get("dhanClientId"),
            "server_time": str(now_ist()),
        }
    except Exception as e:
        logger.error(f"Manual token renewal failed: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
            "server_time": str(now_ist()),
        }


@app.post("/api/update-data")
async def update_data_endpoint():
    """
    Manually trigger data update (yfinance → merged_daily.parquet).
    Useful for ensuring latest market data is available before predictions.
    Runs automatically at 9:05 AM IST, but can be triggered manually.
    """
    from core.data_updater import update_merged_daily
    from core.timezone import now_ist
    import traceback

    logger.info("Manual data update triggered")
    try:
        result = await update_merged_daily()
        result["server_time"] = str(now_ist())
        return result
    except Exception as e:
        logger.error(f"Manual data update failed: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
            "server_time": str(now_ist()),
        }


@app.get("/api/debug/yahoo-test")
async def debug_yahoo_test():
    """Diagnostic: test Yahoo Finance direct API download."""
    import traceback
    from core.timezone import now_ist
    from datetime import timedelta
    from core.data_updater import _download_yahoo

    results = {"method": "direct_yahoo_api", "server_time": str(now_ist())}
    try:
        start = (now_ist() - timedelta(days=7)).strftime("%Y-%m-%d")
        end = (now_ist() + timedelta(days=1)).strftime("%Y-%m-%d")
        results["download_range"] = f"{start} to {end}"

        df = _download_yahoo("^NSEI", "Nifty 50", start, end)
        if df is not None and not df.empty:
            results["rows"] = len(df)
            results["dates"] = [str(d.date()) for d in df.index]
            results["last_close"] = float(df["Close"].iloc[-1])
        else:
            results["error"] = "Yahoo API returned no data"
    except Exception as e:
        results["error"] = str(e)
        results["traceback"] = traceback.format_exc()

    return results


@app.post("/api/backfill")
async def run_backfill_endpoint():
    """
    Run historical backfill from Jan 2026 to today.
    Populates predictions, trades, and daily PnL for all versions.
    Takes ~60-120 seconds. Run once after initial deployment.
    Runs synchronously so errors are returned in the response.
    """
    from backfill import run_backfill
    from db.database import async_session_factory

    logger.info("Backfill endpoint triggered — running synchronously")
    try:
        async with async_session_factory() as db:
            result = await run_backfill(db)
            logger.info(f"Backfill result: {result}")
            return {"status": "backfill_complete", "result": result}
    except Exception as e:
        logger.error(f"Backfill failed: {e}", exc_info=True)
        import traceback
        return {
            "status": "backfill_failed",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


@app.post("/api/debug/recalculate-forward-test")
async def recalculate_forward_test():
    """
    Recalculate all forward test trades (entry_date >= 2026-02-13) with:
    1. INITIAL_CAPITAL for position sizing (no compounding)
    2. Per-calendar-day stop loss breach tracking

    Preserves backtest trades. Uses stored predictions for signals,
    merged_daily.parquet close prices for exit evaluation.
    """
    import traceback
    import math
    import numpy as np
    import pandas as pd
    from datetime import date as date_cls, datetime, timedelta
    from sqlalchemy import text, select, delete, update
    from db.database import async_session_factory
    from db.models import Trade, DailyPnl, DelayPrice, Prediction
    from core.signal_mapper import map_signal
    from core.option_pricer import (
        select_strikes, price_bull_put_spread, price_iron_condor,
        compute_spread_value, get_next_weekly_expiry,
    )
    from config import (
        ACTIVE_VERSIONS, VERSION_CONFIGS, INITIAL_CAPITAL, NIFTY_LOT_SIZE,
        RISK_FREE_RATE, MARGIN_PER_LOT_BULL, MARGIN_PER_LOT_IC,
        BULL_OTM_SELL, BULL_OTM_BUY, MIN_TOTAL_CREDIT, MIN_ENTRY_DTE,
    )

    FORWARD_TEST_START = date_cls(2026, 2, 13)
    # The recompute is a Black-Scholes analysis tool. It is confined to the BS era
    # [FORWARD_TEST_START, REAL_PRICING_START) and MUST NOT delete or rebuild the
    # real-priced live trades on/after REAL_PRICING_START — those are the actual
    # tracked record. See config.REAL_PRICING_START.
    from config import REAL_PRICING_START

    logger.info("=== Starting forward test recalculation ===")

    # Ensure new columns exist
    try:
        async with async_session_factory() as migrate_db:
            from sqlalchemy import text as _text
            for col_sql in [
                "ALTER TABLE trades ADD COLUMN IF NOT EXISTS stop_loss_last_breach_date DATE",
                "ALTER TABLE trades ADD COLUMN IF NOT EXISTS trailing_stop_active BOOLEAN DEFAULT FALSE",
                "ALTER TABLE trades ADD COLUMN IF NOT EXISTS entry_vix FLOAT",
            ]:
                await migrate_db.execute(_text(col_sql))
            await migrate_db.commit()
            logger.info("Ensured all required columns exist")
    except Exception as e:
        logger.warning(f"Column migration note: {e}")

    try:
        async with async_session_factory() as db:
            # 1. Load all forward test predictions from DB
            result = await db.execute(
                select(Prediction).where(
                    Prediction.date >= FORWARD_TEST_START,
                    Prediction.date < REAL_PRICING_START,
                ).order_by(Prediction.date, Prediction.version)
            )
            all_predictions = result.scalars().all()

            # Build per-date prediction data (use first version's spot/vix)
            pred_by_date = {}
            pred_by_date_version = {}
            for p in all_predictions:
                if p.date not in pred_by_date:
                    pred_by_date[p.date] = {
                        "spot": p.nifty_spot,
                        "vix": p.vix,
                        "predicted_drawdown": p.predicted_drawdown,
                    }
                pred_by_date_version[(p.date, p.version)] = p

            logger.info(f"Loaded {len(all_predictions)} predictions for {len(pred_by_date)} dates")

            # 2. Load merged_daily.parquet for daily close prices (exit evaluation)
            parquet_path = "/app/data/merged_daily.parquet"
            import os
            if not os.path.exists(parquet_path):
                # Try local path
                local_path = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "data", "merged_daily.parquet"
                )
                if os.path.exists(local_path):
                    parquet_path = local_path
                else:
                    return {"status": "error", "message": f"merged_daily.parquet not found"}

            merged_daily = pd.read_parquet(parquet_path)
            logger.info(f"Loaded merged_daily: {len(merged_daily)} rows, last={merged_daily.index[-1].date()}")

            # Build close price lookup from parquet
            close_prices = {}
            for ts in merged_daily.index:
                d = ts.date()
                if FORWARD_TEST_START <= d < REAL_PRICING_START:
                    close_val = merged_daily.loc[ts, "nifty_close"]
                    vix_val = merged_daily.loc[ts].get("india_vix", 15.0)
                    if not pd.isna(close_val) and close_val > 0:
                        close_prices[d] = {
                            "spot": float(close_val),
                            "vix": float(vix_val) if not pd.isna(vix_val) else 15.0,
                        }

            # Merge prediction spots into close_prices (prefer live Dhan spot for entry days)
            for d, data in pred_by_date.items():
                if data["spot"] and data["spot"] > 0:
                    close_prices[d] = {
                        "spot": data["spot"],
                        "vix": data["vix"] if data["vix"] else close_prices.get(d, {}).get("vix", 15.0),
                    }

            all_dates = sorted(close_prices.keys())
            logger.info(f"Trading dates for simulation: {len(all_dates)} ({all_dates[0]} to {all_dates[-1]})")

            # 3. Delete all forward test trades, daily_pnl, delay_prices
            # First get trade_ids for delay_prices cleanup
            fwd_trades_result = await db.execute(
                select(Trade.trade_id).where(
                    Trade.entry_date >= FORWARD_TEST_START,
                    Trade.entry_date < REAL_PRICING_START,
                )
            )
            fwd_trade_ids = [r[0] for r in fwd_trades_result.fetchall()]

            if fwd_trade_ids:
                await db.execute(
                    delete(DelayPrice).where(DelayPrice.trade_id.in_(fwd_trade_ids))
                )
            await db.execute(
                delete(Trade).where(
                    Trade.entry_date >= FORWARD_TEST_START,
                    Trade.entry_date < REAL_PRICING_START,
                )
            )
            await db.execute(
                delete(DailyPnl).where(
                    DailyPnl.date >= FORWARD_TEST_START,
                    DailyPnl.date < REAL_PRICING_START,
                )
            )
            await db.commit()
            logger.info(f"Deleted {len(fwd_trade_ids)} forward test trades + related data")

            # 4. Re-simulate trades
            stats = {"trades_opened": 0, "trades_closed": 0, "days_processed": 0}

            # Per-version state
            version_state = {}
            for v in ACTIVE_VERSIONS:
                version_state[v] = {
                    "open_trades": [],
                    "cumulative_pnl": 0.0,
                }

            for trade_date in all_dates:
                day_data = close_prices[trade_date]
                spot = day_data["spot"]
                vix = day_data["vix"]

                if spot <= 0:
                    continue

                for version in ACTIVE_VERSIONS:
                    cfg = VERSION_CONFIGS[version]
                    state = version_state[version]

                    # --- CHECK EXITS ---
                    trades_to_close = []
                    for ot in state["open_trades"]:
                        dte = (ot["expiry"] - trade_date).days
                        T = max(dte / 365.0, 1 / 365.0)
                        sigma = vix / 100.0 if vix > 0 else 0.15

                        current_value = compute_spread_value(
                            ot["trade_type"], spot,
                            ot["sell_strike"], ot["buy_strike"],
                            ot.get("ic_call_sell"), ot.get("ic_call_buy"),
                            T, RISK_FREE_RATE, sigma
                        )

                        pnl_per_unit = ot["credit"] - current_value
                        pnl_pct = pnl_per_unit / ot["credit"] if ot["credit"] > 0 else 0

                        exit_reason = None

                        # Always update peak tracking unconditionally (matches backtest)
                        ot["peak_pnl_pct"] = max(ot.get("peak_pnl_pct", 0.0), pnl_pct)

                        # Exit order matches backtest: profit_target → trailing_stop → stop_loss → vix_spike → expiry

                        # 1. Profit target
                        if not exit_reason:
                            days_held = (trade_date - ot["entry_date"]).days
                            if days_held <= 3:
                                pt = cfg.get("PROFIT_TARGET_EARLY", 0.50)
                            elif days_held <= 7:
                                pt = cfg.get("PROFIT_TARGET_MID", 0.65)
                            else:
                                pt = cfg.get("PROFIT_TARGET_LATE", 0.80)
                            if pnl_pct >= pt:
                                exit_reason = "profit_target"

                        # 2. Trailing stop — persistent activation, proportional retracement
                        if not exit_reason:
                            ta = cfg.get("TRAILING_STOP_ACTIVATE", 0.40)
                            tl = cfg.get("TRAILING_STOP_LEVEL", 0.10)
                            if pnl_pct >= ta:
                                ot["trailing_stop_active"] = True
                            if ot.get("trailing_stop_active"):
                                peak = ot.get("peak_pnl_pct", 0.0)
                                if peak > 0:
                                    retrace = peak - pnl_pct
                                    if retrace > peak * tl and pnl_pct > 0:
                                        exit_reason = "trailing_stop"

                        # 3. Stop loss — per-CALENDAR-DAY breach, strict > (matches backtest)
                        if not exit_reason:
                            sl_mult = cfg.get("IC_STOP_LOSS_MULTIPLIER", 3.0) if ot["trade_type"] == "iron_condor" else cfg.get("STOP_LOSS_MULTIPLIER", 3.0)
                            cd = cfg.get("IC_STOP_LOSS_CONFIRM_DAYS", 2) if ot["trade_type"] == "iron_condor" else cfg.get("STOP_LOSS_CONFIRM_DAYS", 2)

                            if current_value >= ot["credit"] * sl_mult + ot["credit"]:
                                last_breach = ot.get("stop_loss_last_breach_date")
                                if last_breach != trade_date:
                                    ot["stop_loss_breach_days"] = ot.get("stop_loss_breach_days", 0) + 1
                                    ot["stop_loss_last_breach_date"] = trade_date
                                if ot["stop_loss_breach_days"] > cd:
                                    exit_reason = "stop_loss"
                            else:
                                ot["stop_loss_breach_days"] = 0
                                ot["stop_loss_last_breach_date"] = None

                        # 4. VIX spike exit (matches backtest: VIX > threshold AND VIX > entry * 1.3)
                        if not exit_reason:
                            vix_spike_threshold = cfg.get("VIX_SPIKE_EXIT")
                            if vix_spike_threshold and vix > 0:
                                entry_vix = ot.get("entry_vix", 0)
                                if entry_vix and vix > vix_spike_threshold and vix > entry_vix * 1.3:
                                    exit_reason = "vix_spike"

                        # 5. Expiry (last — matches backtest priority)
                        if not exit_reason:
                            if dte <= cfg.get("MIN_DTE_EXIT", 1):
                                exit_reason = "expiry"

                        if exit_reason:
                            realized_pnl = pnl_per_unit * ot["num_lots"] * NIFTY_LOT_SIZE
                            rpnl_pct = realized_pnl / ot["capital_deployed"] * 100 if ot["capital_deployed"] > 0 else 0

                            await db.execute(
                                update(Trade).where(Trade.trade_id == ot["trade_id"]).values(
                                    status="closed",
                                    exit_date=trade_date,
                                    exit_time=datetime.combine(trade_date, datetime.min.time().replace(hour=15, minute=25)),
                                    exit_spot=spot,
                                    exit_reason=exit_reason,
                                    realized_pnl=realized_pnl,
                                    current_pnl=realized_pnl,
                                    current_pnl_pct=rpnl_pct,
                                    current_spread_value=current_value,
                                    unrealized_pnl=0,
                                )
                            )
                            state["cumulative_pnl"] += realized_pnl
                            trades_to_close.append(ot["trade_id"])
                            stats["trades_closed"] += 1
                        else:
                            # Update unrealized PnL
                            unrealized_pnl = pnl_per_unit * ot["num_lots"] * NIFTY_LOT_SIZE
                            await db.execute(
                                update(Trade).where(Trade.trade_id == ot["trade_id"]).values(
                                    current_spread_value=current_value,
                                    unrealized_pnl=unrealized_pnl,
                                    current_pnl=unrealized_pnl,
                                    current_pnl_pct=unrealized_pnl / ot["capital_deployed"] * 100 if ot["capital_deployed"] > 0 else 0,
                                )
                            )

                    state["open_trades"] = [t for t in state["open_trades"] if t["trade_id"] not in trades_to_close]

                    # --- OPEN NEW TRADE ---
                    pred = pred_by_date_version.get((trade_date, version))
                    if pred:
                        signal = map_signal(pred.predicted_drawdown, cfg)

                        if signal["signal"] != "no_trade" and signal["trade_type"]:
                            tt = signal["trade_type"]
                            sm = signal["size_mult"]

                            # Check concurrent limits
                            open_count = sum(1 for t in state["open_trades"] if t["trade_type"] == tt)
                            max_c = cfg.get("IC_MAX_CONCURRENT", 2) if tt == "iron_condor" else cfg.get("MAX_CONCURRENT_POSITIONS", 3)
                            last_entry = max((t["entry_date"] for t in state["open_trades"]), default=None)
                            gap_ok = last_entry is None or (trade_date - last_entry).days >= cfg.get("MIN_ENTRY_GAP_DAYS", 2)

                            # VIX harvest mode
                            entry_mode = "normal"
                            if cfg.get("VIX_HARVEST_ENABLED") and vix and vix >= cfg.get("VIX_HARVEST_TRIGGER", 23):
                                entry_mode = "vix_harvest"

                            if open_count < max_c and gap_ok:
                                entry_spot = pred.nifty_spot if pred.nifty_spot and pred.nifty_spot > 0 else spot

                                # Select strikes
                                strike_cfg = {
                                    "BULL_OTM_SELL": BULL_OTM_SELL,
                                    "BULL_OTM_BUY": BULL_OTM_BUY,
                                    "IC_PUT_OTM_SELL": cfg.get("IC_PUT_OTM_SELL", 0.03),
                                    "IC_PUT_OTM_BUY": cfg.get("IC_PUT_OTM_BUY", 0.055),
                                    "IC_CALL_OTM_SELL": cfg.get("IC_CALL_OTM_SELL", 0.04),
                                    "IC_CALL_OTM_BUY": cfg.get("IC_CALL_OTM_BUY", 0.065),
                                }
                                strikes = select_strikes(entry_spot, tt, strike_cfg)

                                # Price the spread
                                expiry = get_next_weekly_expiry(trade_date)
                                T_entry = max((expiry - trade_date).days / 365.0, 1 / 365.0)
                                sigma_entry = vix / 100.0 if vix > 0 else 0.15

                                if tt == "bull_put":
                                    credit = price_bull_put_spread(
                                        entry_spot, strikes["sell_strike"], strikes["buy_strike"],
                                        T_entry, RISK_FREE_RATE, sigma_entry, apply_slippage=True
                                    )
                                elif tt == "iron_condor":
                                    credit = price_iron_condor(
                                        entry_spot, strikes["sell_strike"], strikes["buy_strike"],
                                        strikes["ic_call_sell"], strikes["ic_call_buy"],
                                        T_entry, RISK_FREE_RATE, sigma_entry, apply_slippage=True
                                    )
                                else:
                                    credit = 0

                                # Position sizing: INITIAL_CAPITAL (no compounding)
                                pos_pct = cfg.get("IC_POSITION_SIZE_PCT", 0.15) if tt == "iron_condor" else cfg.get("POSITION_SIZE_PCT", 0.20)
                                eff_pct = pos_pct * sm
                                margin = MARGIN_PER_LOT_IC if tt == "iron_condor" else MARGIN_PER_LOT_BULL
                                max_cap = INITIAL_CAPITAL * eff_pct
                                num_lots = max(1, int(max_cap / margin))

                                total_credit = credit * num_lots * NIFTY_LOT_SIZE
                                capital_deployed = num_lots * margin

                                # Live open_trade quality gate (previously missing here):
                                # skip nano-credit / near-expiry entries so the forward
                                # test matches what live trading actually takes. On the
                                # Tuesday calendar this drops the 1-DTE Monday condors.
                                dte_at_entry = (expiry - trade_date).days
                                if total_credit < MIN_TOTAL_CREDIT or dte_at_entry < MIN_ENTRY_DTE:
                                    continue

                                trade_id = f"{version.replace('.', '')}-{trade_date.isoformat()}-{signal['signal']}"

                                # Create trade
                                trade = Trade(
                                    trade_id=trade_id,
                                    version=version,
                                    date=trade_date,
                                    signal_type=signal["signal"],
                                    trade_type=tt,
                                    entry_mode=entry_mode,
                                    entry_date=trade_date,
                                    entry_time=datetime.combine(trade_date, datetime.min.time().replace(hour=9, minute=20)),
                                    entry_spot=entry_spot,
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
                                    position_size_pct=eff_pct,
                                    graduated_mult=sm,
                                    capital_deployed=capital_deployed,
                                    stop_loss_breach_days=0,
                                    stop_loss_last_breach_date=None,
                                    peak_pnl_pct=0.0,
                                    trailing_stop_active=False,
                                    entry_vix=vix,
                                    predicted_drawdown=pred.predicted_drawdown,
                                )
                                db.add(trade)

                                state["open_trades"].append({
                                    "trade_id": trade_id,
                                    "trade_type": tt,
                                    "entry_date": trade_date,
                                    "expiry": expiry,
                                    "sell_strike": strikes["sell_strike"],
                                    "buy_strike": strikes["buy_strike"],
                                    "ic_call_sell": strikes.get("ic_call_sell"),
                                    "ic_call_buy": strikes.get("ic_call_buy"),
                                    "credit": credit,
                                    "num_lots": num_lots,
                                    "capital_deployed": capital_deployed,
                                    "peak_pnl_pct": 0.0,
                                    "trailing_stop_active": False,
                                    "entry_vix": vix,
                                    "stop_loss_breach_days": 0,
                                    "stop_loss_last_breach_date": None,
                                })
                                stats["trades_opened"] += 1

                stats["days_processed"] += 1

                if stats["days_processed"] % 5 == 0:
                    await db.commit()

            await db.commit()

            # 5. Build summary
            result = await db.execute(
                select(Trade).where(Trade.entry_date >= FORWARD_TEST_START).order_by(Trade.entry_date)
            )
            new_trades = result.scalars().all()

            trade_summary = []
            for t in new_trades:
                trade_summary.append({
                    "trade_id": t.trade_id,
                    "entry_date": str(t.entry_date),
                    "exit_date": str(t.exit_date) if t.exit_date else "open",
                    "num_lots": t.num_lots,
                    "realized_pnl": round(t.realized_pnl, 2) if t.realized_pnl else None,
                    "exit_reason": t.exit_reason,
                    "status": t.status,
                })

            logger.info(f"=== Forward test recalculation complete: {stats} ===")

            return {
                "status": "recalculated",
                "stats": stats,
                "initial_capital_used": INITIAL_CAPITAL,
                "forward_test_start": str(FORWARD_TEST_START),
                "dates_simulated": len(all_dates),
                "trades": trade_summary,
            }

    except Exception as e:
        logger.error(f"Forward test recalculation failed: {e}", exc_info=True)
        return {
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc(),
        }


@app.get("/api/debug/db-counts")
async def db_counts():
    """Check how many rows are in each table — useful for debugging backfill."""
    from sqlalchemy import text
    from db.database import async_session_factory

    async with async_session_factory() as db:
        tables = ["predictions", "trades", "daily_pnl", "daily_features"]
        counts = {}
        for table in tables:
            try:
                result = await db.execute(text(f"SELECT COUNT(*) FROM {table}"))
                counts[table] = result.scalar()
            except Exception as e:
                counts[table] = f"error: {e}"
        return counts


@app.get("/api/debug/snapshot-coverage")
async def debug_snapshot_coverage(start: str = "2026-02-13"):
    """Read-only: coverage of price_snapshots over [start, now] — gates the
    intraday exit-replay harness. Reports totals, populated low/high/vix, and
    per-day gaps (IST date grouping)."""
    import traceback
    from datetime import date as _date
    from sqlalchemy import text
    from db.database import async_session_factory

    try:
        start_dt = _date.fromisoformat(start)
        async with async_session_factory() as db:
            summ = (await db.execute(text(
                """
                SELECT COUNT(*) AS total,
                       MIN("timestamp") AS earliest,
                       MAX("timestamp") AS latest,
                       COUNT(nifty_low) AS n_low,
                       COUNT(nifty_high) AS n_high,
                       COUNT(vix) AS n_vix,
                       COUNT(DISTINCT ("timestamp" AT TIME ZONE 'Asia/Kolkata')::date) AS distinct_days
                FROM price_snapshots
                WHERE "timestamp" >= CAST(:start AS timestamptz)
                """
            ), {"start": start_dt})).mappings().first()

            per_day = (await db.execute(text(
                """
                SELECT ("timestamp" AT TIME ZONE 'Asia/Kolkata')::date AS d,
                       COUNT(*) AS c, COUNT(nifty_low) AS lo,
                       COUNT(nifty_high) AS hi, COUNT(vix) AS vx
                FROM price_snapshots
                WHERE "timestamp" >= CAST(:start AS timestamptz)
                GROUP BY d ORDER BY d
                """
            ), {"start": start_dt})).mappings().all()

        total = summ["total"] or 0
        counts = [r["c"] for r in per_day]
        days = [{"date": str(r["d"]), "n": r["c"], "low": r["lo"], "high": r["hi"], "vix": r["vx"]}
                for r in per_day]

        def pct(n):
            return round(100.0 * (n or 0) / total, 1) if total else 0.0

        return {
            "window_start": start,
            "total_snapshots": total,
            "earliest": str(summ["earliest"]) if summ["earliest"] else None,
            "latest": str(summ["latest"]) if summ["latest"] else None,
            "distinct_days": summ["distinct_days"],
            "populated_pct": {
                "nifty_low": pct(summ["n_low"]),
                "nifty_high": pct(summ["n_high"]),
                "vix": pct(summ["n_vix"]),
            },
            "per_day_snapshots": {
                "avg": round(sum(counts) / len(counts), 1) if counts else 0,
                "min": min(counts) if counts else 0,
                "max": max(counts) if counts else 0,
            },
            "sparse_days_lt10": [d for d in days if d["n"] < 10],
            "days_missing_low": [d["date"] for d in days if d["low"] == 0],
            "days_missing_vix": [d["date"] for d in days if d["vix"] == 0],
            "days": days,
        }
    except Exception as e:
        return {"status": "error", "error": str(e), "traceback": traceback.format_exc()}


@app.get("/api/debug/intraday-replay")
async def debug_intraday_replay(start: str = "2026-02-13"):
    """Read-only: replay each version's exits at intraday (15-min snapshot)
    resolution and compare to the daily-close recompute. breach_mode
    close/spot/extreme brackets the live answer. Does NOT persist anything —
    'close' should reproduce the stored forward-test (correctness check)."""
    import traceback, os
    from datetime import date as _date, datetime as _dt, timezone as _tz
    import pandas as pd
    from sqlalchemy import select
    from db.database import async_session_factory
    from db.models import Prediction, PriceSnapshot
    from core.timezone import IST
    from core.intraday_replay import replay
    from config import ACTIVE_VERSIONS
    try:
        start_d = _date.fromisoformat(start)
        start_dt = _dt(start_d.year, start_d.month, start_d.day, tzinfo=_tz.utc)

        async with async_session_factory() as db:
            prows = (await db.execute(
                select(Prediction).where(Prediction.date >= start_d)
                .order_by(Prediction.date, Prediction.version)
            )).scalars().all()
            srows = (await db.execute(
                select(PriceSnapshot).where(PriceSnapshot.timestamp >= start_dt)
                .order_by(PriceSnapshot.timestamp)
            )).scalars().all()

        # One prediction per date (same drawdown/spot/vix across versions).
        predictions = {}
        for p in prows:
            if p.date not in predictions and p.nifty_spot and p.nifty_spot > 0:
                predictions[p.date] = {
                    "predicted_drawdown": p.predicted_drawdown,
                    "spot": float(p.nifty_spot),
                    "vix": float(p.vix) if p.vix else 15.0,
                }

        # Intraday marks grouped by IST date.
        intraday_by_day = {}
        for s in srows:
            ist = s.timestamp.astimezone(IST)
            intraday_by_day.setdefault(ist.date(), []).append((
                ist.time(), float(s.nifty_spot),
                float(s.nifty_low) if s.nifty_low else float(s.nifty_spot),
                float(s.nifty_high) if s.nifty_high else float(s.nifty_spot),
                float(s.vix) if s.vix else 15.0,
            ))

        # Daily close grid (the trading calendar + EOD/fallback valuation).
        parquet_path = "/app/data/merged_daily.parquet"
        if not os.path.exists(parquet_path):
            parquet_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "data", "merged_daily.parquet")
        md = pd.read_parquet(parquet_path)
        daily_close = {}
        for ts in md.index:
            d = ts.date()
            if d >= start_d:
                c = md.loc[ts, "nifty_close"]
                v = md.loc[ts].get("india_vix", 15.0)
                if not pd.isna(c) and c > 0:
                    daily_close[d] = {"spot": float(c),
                                      "vix": float(v) if not pd.isna(v) else 15.0}

        # Overlay prediction spots — matches the recompute's close_prices: it prefers
        # the live 9:20 Dhan spot on entry days, and predictions cover trading days
        # that merged_daily (sparse in this window) lacks. Without this the day grid
        # is too sparse and far fewer trades open.
        for d, pr in predictions.items():
            if pr["spot"] and pr["spot"] > 0:
                daily_close[d] = {"spot": pr["spot"],
                                  "vix": pr["vix"] if pr.get("vix") else daily_close.get(d, {}).get("vix", 15.0)}

        results = {}
        for ver in ACTIVE_VERSIONS:
            results[ver] = {
                mode: replay(ver, predictions, intraday_by_day, daily_close, breach_mode=mode)
                for mode in ["close", "spot", "extreme"]
            }

        return {
            "window_start": start,
            "n_pred_dates": len(predictions),
            "n_snapshot_days": len(intraday_by_day),
            "n_trading_days": len(daily_close),
            "results": results,
        }
    except Exception as e:
        return {"status": "error", "error": str(e), "traceback": traceback.format_exc()}


@app.get("/api/debug/pricing-comparison")
async def debug_pricing_comparison(limit: int = 50):
    """Read-only: per-trade real-vs-BS credit + pricing source, newest first.
    Highlights any trade that fell back to synthetic BS pricing."""
    import traceback
    from collections import Counter
    from sqlalchemy import select, desc
    from db.database import async_session_factory
    from db.models import Trade
    try:
        async with async_session_factory() as db:
            rows = (await db.execute(
                select(Trade).order_by(desc(Trade.entry_date), desc(Trade.id)).limit(limit)
            )).scalars().all()
        src_counts = Counter()
        trades = []
        for t in rows:
            src = t.pricing_source or "bs_synthetic"
            src_counts[src] += 1
            real, bs = t.real_credit, t.bs_credit
            trades.append({
                "trade_id": t.trade_id, "version": t.version,
                "entry_date": t.entry_date.isoformat() if t.entry_date else None,
                "trade_type": t.trade_type, "pricing_source": src,
                "bs_credit": bs, "real_credit": real, "credit_used": t.credit_received,
                "gap": round(real - bs, 2) if (real is not None and bs is not None) else None,
                "BS_FALLBACK": src == "bs_fallback",
            })
        return {
            "count": len(trades),
            "by_source": dict(src_counts),
            "bs_fallbacks": src_counts.get("bs_fallback", 0),
            "trades": trades,
        }
    except Exception as e:
        return {"status": "error", "error": str(e), "traceback": traceback.format_exc()}


@app.get("/api/debug/test-real-pricing")
async def debug_test_real_pricing():
    """Read-only: fetch the live option chain for the next expiry, parse it, and
    price a sample iron condor — verifies the chain parser against real Dhan data
    and shows real-vs-BS. Returns a raw sample if parsing finds nothing."""
    import traceback
    from core.option_pricer import (
        get_next_weekly_expiry, select_strikes, price_iron_condor,
        compute_time_to_expiry_years,
    )
    from core.real_option_pricer import parse_chain, price_spread_real, value_spread_real
    from core.timezone import today_ist
    from config import RISK_FREE_RATE
    from scheduler.jobs import dhan_client
    try:
        expiry = get_next_weekly_expiry()
        chain = await dhan_client.get_option_chain(expiry.isoformat())
        if not chain:
            return {"status": "no_chain", "expiry": expiry.isoformat()}
        data = chain.get("data", chain)
        spot = data.get("last_price") or await dhan_client.get_nifty_ltp()
        parsed = parse_chain(chain)
        strikes = select_strikes(float(spot), "iron_condor", {
            "IC_PUT_OTM_SELL": 0.03, "IC_PUT_OTM_BUY": 0.055,
            "IC_CALL_OTM_SELL": 0.04, "IC_CALL_OTM_BUY": 0.065})
        vix = await dhan_client.get_india_vix() or 14.0
        T = compute_time_to_expiry_years(today_ist(), expiry)
        bs = price_iron_condor(float(spot), strikes["sell_strike"], strikes["buy_strike"],
                               strikes["ic_call_sell"], strikes["ic_call_buy"],
                               T, RISK_FREE_RATE, vix / 100.0, apply_slippage=True)
        pricing = price_spread_real(chain, "iron_condor", strikes, bs)
        exit_val = value_spread_real(chain, "iron_condor", strikes, bs)
        immediate_pnl = None
        if pricing.get("source", "").startswith("real") and exit_val.get("source", "").startswith("real"):
            immediate_pnl = round((pricing["real_credit"] or 0) - (exit_val["real_value"] or 0), 2)
        sample = None
        if not parsed:
            oc = data.get("oc") or data.get("option_chain") or {}
            k = next(iter(oc), None)
            sample = {k: oc.get(k)} if k else {"data_keys": list(data.keys())[:10]}
        return {
            "expiry": expiry.isoformat(), "spot": spot, "vix": vix,
            "strikes": strikes, "parsed_strikes": len(parsed),
            "entry_pricing": pricing, "exit_pricing": exit_val,
            "immediate_pnl_per_unit": immediate_pnl, "raw_sample_if_unparsed": sample,
        }
    except Exception as e:
        return {"status": "error", "error": str(e), "traceback": traceback.format_exc()}


@app.get("/api/debug/daily-marks")
async def debug_daily_marks(limit: int = 40):
    """Read-only: recent daily real-vs-BS cost-to-close marks per open trade —
    the accumulating real-priced history. Highlights BS fallbacks."""
    import traceback
    from collections import Counter
    from sqlalchemy import select, desc
    from db.database import async_session_factory
    from db.models import DailyTradeMark
    try:
        async with async_session_factory() as db:
            rows = (await db.execute(
                select(DailyTradeMark)
                .order_by(desc(DailyTradeMark.date), desc(DailyTradeMark.id))
                .limit(limit)
            )).scalars().all()
        src = Counter()
        marks = []
        for m in rows:
            src[m.pricing_source] += 1
            r, b = m.real_spread_value, m.bs_spread_value
            marks.append({
                "date": m.date.isoformat(), "trade_id": m.trade_id, "version": m.version,
                "spot": m.spot, "source": m.pricing_source,
                "real_value": r, "bs_value": b,
                "gap": round(r - b, 2) if (r is not None and b is not None) else None,
                "unrealized_pnl": m.unrealized_pnl,
                "BS_FALLBACK": m.pricing_source == "bs_fallback",
            })
        return {"count": len(marks), "by_source": dict(src),
                "bs_fallbacks": src.get("bs_fallback", 0), "marks": marks}
    except Exception as e:
        return {"status": "error", "error": str(e), "traceback": traceback.format_exc()}


@app.post("/api/debug/run-daily-marks")
async def debug_run_daily_marks():
    """Manually trigger the daily real-price marks job (normally 15:32 IST)."""
    import traceback
    from scheduler.jobs import record_daily_marks
    try:
        await record_daily_marks()
        return {"status": "ok"}
    except Exception as e:
        return {"status": "error", "error": str(e), "traceback": traceback.format_exc()}


@app.post("/api/debug/test-email")
async def debug_test_email():
    """Send a test email to verify Resend API key and email delivery."""
    import os
    import httpx

    api_key = os.environ.get("RESEND_API_KEY", "")
    notify_email = os.environ.get("NOTIFY_EMAIL", "anmol@turings.xyz")
    from_email = "Nifty Trading Bot <onboarding@resend.dev>"

    if not api_key:
        return {"status": "error", "message": "RESEND_API_KEY not set in environment"}

    # Show config (mask API key)
    config_info = {
        "api_key_set": bool(api_key),
        "api_key_prefix": api_key[:10] + "..." if len(api_key) > 10 else "too_short",
        "api_key_length": len(api_key),
        "notify_email": notify_email,
        "from_email": from_email,
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.resend.com/emails",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "from": from_email,
                    "to": [notify_email],
                    "subject": "Test Email from Nifty Trading Bot",
                    "html": "<h2>Email delivery is working!</h2><p>This is a test email from your Nifty Trading Bot.</p>",
                },
                timeout=10,
            )

            return {
                "status": "sent" if response.status_code in (200, 202) else "failed",
                "resend_status_code": response.status_code,
                "resend_response": response.json() if response.status_code < 500 else response.text,
                "config": config_info,
            }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "config": config_info,
        }


@app.post("/api/debug/recompute-date")
async def debug_recompute_date(target_date: str):
    """
    Recompute predictions for a specific past date using current parquet data.
    Useful for fixing stale predictions caused by data pipeline failures.
    Does NOT touch trades — only updates Prediction and DailyFeature rows.
    """
    import traceback
    from datetime import date as date_cls, datetime
    from db.database import async_session_factory
    from db.models import Prediction, DailyFeature
    from core.model_runner import ModelRunner
    from core.signal_mapper import map_signal, get_classification_breakdown
    from core.timezone import IST
    from config import (
        DOWNSIDE_MODEL_PATH, SCALER_PATH, FEATURE_NAMES_PATH,
        ACTIVE_VERSIONS, VERSION_CONFIGS,
    )
    from sqlalchemy import delete
    import pandas as pd

    try:
        target = date_cls.fromisoformat(target_date)
    except ValueError:
        return {"status": "error", "message": f"Invalid date format: {target_date}. Use YYYY-MM-DD."}

    try:
        # Load model
        mr = ModelRunner(DOWNSIDE_MODEL_PATH, SCALER_PATH, FEATURE_NAMES_PATH)

        # Load merged_daily (server's current data)
        merged = pd.read_parquet("/app/data/merged_daily.parquet")
        logger.info(f"Recompute: merged_daily has {len(merged)} rows, last={merged.index[-1].date()}")

        if pd.Timestamp(target) > merged.index[-1]:
            return {"status": "error", "message": f"Target date {target} is beyond parquet data (last: {merged.index[-1].date()})"}

        # Compute features using the PREVIOUS trading day's data.
        # This matches how the live system works: at 9:20 AM on date X,
        # only data through X-1 (yesterday's close) is available.
        from shared.feature_compute import compute_features_for_date, load_dhan_options_features

        target_ts = pd.Timestamp(target)
        dates_before = merged.index[merged.index < target_ts]
        if len(dates_before) == 0:
            return {"status": "error", "message": f"No data before {target} to compute features from"}
        prev_trading_day = dates_before[-1]
        logger.info(f"Recompute {target}: using data through {prev_trading_day.date()} (previous trading day)")

        dhan = None
        try:
            dhan = load_dhan_options_features(
                "/app/data/dhan_raw_options.parquet",
                "/app/data/daily_iv_skew_params.parquet",
            )
        except Exception as e:
            logger.warning(f"Recompute: Dhan features unavailable: {e}")

        features = compute_features_for_date(merged, prev_trading_day, dhan)
        if features is None:
            return {"status": "error", "message": f"Could not compute features for {target} (missing data?)"}

        # Run model prediction
        prediction_value = mr.predict(features)
        logger.info(f"Recompute {target}: prediction={prediction_value*100:.4f}%")

        # Store in DB (replace old predictions + features for this date)
        async with async_session_factory() as db:
            await db.execute(delete(Prediction).where(Prediction.date == target))
            await db.execute(delete(DailyFeature).where(DailyFeature.date == target))

            ts = datetime(target.year, target.month, target.day, 9, 20, 0, tzinfo=IST)

            version_signals = {}
            for version in ACTIVE_VERSIONS:
                cfg = VERSION_CONFIGS[version]
                signal = map_signal(prediction_value, cfg)

                breakdown = get_classification_breakdown(prediction_value, version=version)
                active_zone = next(
                    (z for z in breakdown["zones"] if z.get("confidence") == "active"),
                    None,
                )
                confidence = active_zone["distance_to_boundary"] if active_zone else 0

                pred = Prediction(
                    date=target,
                    timestamp=ts,
                    predicted_drawdown=prediction_value,
                    signal_type=signal["signal"],
                    version=version,
                    nifty_spot=features.get("nifty_close", 0),
                    vix=features.get("india_vix", 0),
                    confidence_score=confidence,
                    graduated_mult=signal["size_mult"],
                    features=features,
                )
                db.add(pred)
                version_signals[version] = signal["signal"]

            daily_feature = DailyFeature(
                date=target,
                features=features,
                vix=features.get("india_vix"),
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
            await db.commit()

        return {
            "status": "recomputed",
            "date": target_date,
            "data_through": str(prev_trading_day.date()),
            "prediction": prediction_value,
            "prediction_pct": f"{prediction_value*100:.2f}%",
            "versions": version_signals,
            "features_count": len(features),
            "nifty_close": features.get("nifty_close"),
        }

    except Exception as e:
        logger.error(f"Recompute failed for {target_date}: {e}", exc_info=True)
        return {
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc(),
        }


@app.post("/api/debug/delete-trade")
async def debug_delete_trade(trade_id: str):
    """
    Delete a specific trade by trade_id.
    Use this to remove trades that were opened on stale/incorrect data.
    Returns the deleted trade details for confirmation.
    """
    from db.database import async_session_factory
    from db.models import Trade, DelayPrice
    from sqlalchemy import select, delete

    try:
        async with async_session_factory() as db:
            # First, fetch the trade to show what's being deleted
            result = await db.execute(
                select(Trade).where(Trade.trade_id == trade_id)
            )
            trade = result.scalar_one_or_none()

            if not trade:
                return {"status": "not_found", "message": f"No trade found with id: {trade_id}"}

            trade_info = {
                "trade_id": trade.trade_id,
                "version": trade.version,
                "entry_date": str(trade.entry_date),
                "trade_type": trade.trade_type,
                "status": trade.status,
                "entry_spot": trade.entry_spot,
                "predicted_drawdown": trade.predicted_drawdown,
                "realized_pnl": trade.realized_pnl,
            }

            # Delete related rows first (foreign key constraints)
            await db.execute(delete(DelayPrice).where(DelayPrice.trade_id == trade_id))
            # Delete the trade
            await db.execute(delete(Trade).where(Trade.trade_id == trade_id))
            await db.commit()

            return {
                "status": "deleted",
                "deleted_trade": trade_info,
            }

    except Exception as e:
        logger.error(f"Delete trade failed for {trade_id}: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}


@app.post("/api/debug/clone-v542-to-v543")
async def clone_v542_to_v543():
    """
    Clone ALL v5.4.2 data (backtest + forward test) into v5.4.3.

    Deletes all existing v5.4.3 trades, daily_pnl, delay_prices,
    and predictions. Then copies everything from v5.4.2 into v5.4.3
    with updated version and trade_id fields.

    This makes v5.4.3 identical to v5.4.2 so they can diverge only based
    on exit frequency going forward (v5.4.2 = hybrid, v5.4.3 = all 5-min).
    """
    from datetime import date as date_cls
    from sqlalchemy import select, delete, text
    from db.database import async_session_factory
    from db.models import Trade, DailyPnl, DelayPrice, Prediction

    try:
        async with async_session_factory() as db:
            # ── Step 1: Delete ALL v5.4.3 data ──

            # Get all v5.4.3 trade_ids (for delay_prices FK)
            result = await db.execute(
                select(Trade.trade_id).where(Trade.version == "v5.4.3")
            )
            old_trade_ids = [r[0] for r in result.all()]

            # Delete delay_prices for those trades
            if old_trade_ids:
                # Delete in batches to avoid too-long IN clause
                batch_size = 500
                for i in range(0, len(old_trade_ids), batch_size):
                    batch = old_trade_ids[i:i + batch_size]
                    await db.execute(
                        delete(DelayPrice).where(DelayPrice.trade_id.in_(batch))
                    )

            # Delete all v5.4.3 trades
            del_trades = await db.execute(
                delete(Trade).where(Trade.version == "v5.4.3")
            )

            # Delete all v5.4.3 daily_pnl
            del_pnl = await db.execute(
                delete(DailyPnl).where(DailyPnl.version == "v5.4.3")
            )

            # Delete all v5.4.3 predictions
            del_preds = await db.execute(
                delete(Prediction).where(Prediction.version == "v5.4.3")
            )

            deleted_summary = {
                "trades": del_trades.rowcount,
                "daily_pnl": del_pnl.rowcount,
                "predictions": del_preds.rowcount,
                "delay_prices": len(old_trade_ids),
            }

            # ── Step 2: Load ALL v5.4.2 data ──

            result = await db.execute(
                select(Trade).where(
                    Trade.version == "v5.4.2",
                ).order_by(Trade.entry_date)
            )
            v542_trades = result.scalars().all()

            result = await db.execute(
                select(DailyPnl).where(
                    DailyPnl.version == "v5.4.2",
                ).order_by(DailyPnl.date)
            )
            v542_pnl = result.scalars().all()

            result = await db.execute(
                select(Prediction).where(
                    Prediction.version == "v5.4.2",
                ).order_by(Prediction.date)
            )
            v542_preds = result.scalars().all()

            # ── Step 3: Clone trades ──
            cloned_trades = []
            trade_id_map = {}  # old v542 trade_id -> new v543 trade_id

            for t in v542_trades:
                new_trade_id = t.trade_id.replace("v542-", "v543-")
                trade_id_map[t.trade_id] = new_trade_id

                new_trade = Trade(
                    trade_id=new_trade_id,
                    version="v5.4.3",
                    date=t.date,
                    signal_type=t.signal_type,
                    trade_type=t.trade_type,
                    entry_mode=t.entry_mode,
                    entry_date=t.entry_date,
                    entry_time=t.entry_time,
                    entry_spot=t.entry_spot,
                    expiry=t.expiry,
                    sell_strike=t.sell_strike,
                    buy_strike=t.buy_strike,
                    ic_call_sell=t.ic_call_sell,
                    ic_call_buy=t.ic_call_buy,
                    num_lots=t.num_lots,
                    credit_received=t.credit_received,
                    total_credit=t.total_credit,
                    status=t.status,
                    current_spread_value=t.current_spread_value,
                    current_pnl=t.current_pnl,
                    current_pnl_pct=t.current_pnl_pct,
                    unrealized_pnl=t.unrealized_pnl,
                    exit_date=t.exit_date,
                    exit_time=t.exit_time,
                    exit_spot=t.exit_spot,
                    exit_reason=t.exit_reason,
                    realized_pnl=t.realized_pnl,
                    position_size_pct=t.position_size_pct,
                    graduated_mult=t.graduated_mult,
                    capital_deployed=t.capital_deployed,
                    is_bear_debit=t.is_bear_debit,
                    bear_tier=t.bear_tier,
                    entry_debit=t.entry_debit,
                    predicted_drawdown=t.predicted_drawdown,
                    max_profit=t.max_profit,
                    max_loss_amount=t.max_loss_amount,
                    bear_trail_high=t.bear_trail_high,
                    stop_loss_breach_days=t.stop_loss_breach_days,
                    stop_loss_last_breach_date=t.stop_loss_last_breach_date,
                    peak_pnl_pct=t.peak_pnl_pct,
                )
                db.add(new_trade)
                cloned_trades.append(new_trade_id)

            await db.flush()

            # ── Step 4: Clone delay_prices ──
            v542_trade_ids = [t.trade_id for t in v542_trades]
            if v542_trade_ids:
                result = await db.execute(
                    select(DelayPrice).where(
                        DelayPrice.trade_id.in_(v542_trade_ids)
                    )
                )
                v542_delays = result.scalars().all()

                for dp in v542_delays:
                    new_dp = DelayPrice(
                        trade_id=trade_id_map.get(dp.trade_id, dp.trade_id),
                        version="v5.4.3",
                        signal_date=dp.signal_date,
                        signal_time=dp.signal_time,
                        price_at_signal=dp.price_at_signal,
                        price_at_10min=dp.price_at_10min,
                        price_at_1hr=dp.price_at_1hr,
                        price_at_3hr=dp.price_at_3hr,
                        price_at_6hr=dp.price_at_6hr,
                        price_at_12hr=dp.price_at_12hr,
                        spread_price_at_signal=dp.spread_price_at_signal,
                        spread_price_at_10min=dp.spread_price_at_10min,
                        spread_price_at_1hr=dp.spread_price_at_1hr,
                        spread_price_at_3hr=dp.spread_price_at_3hr,
                        spread_price_at_6hr=dp.spread_price_at_6hr,
                        spread_price_at_12hr=dp.spread_price_at_12hr,
                    )
                    db.add(new_dp)

            # ── Step 5: Clone daily_pnl ──
            for p in v542_pnl:
                new_pnl = DailyPnl(
                    date=p.date,
                    version="v5.4.3",
                    starting_capital=p.starting_capital,
                    ending_capital=p.ending_capital,
                    daily_pnl=p.daily_pnl,
                    cumulative_pnl=p.cumulative_pnl,
                    cumulative_return_pct=p.cumulative_return_pct,
                    open_positions=p.open_positions,
                    trades_opened=p.trades_opened,
                    trades_closed=p.trades_closed,
                )
                db.add(new_pnl)

            # ── Step 6: Clone predictions ──
            for pred in v542_preds:
                new_pred = Prediction(
                    date=pred.date,
                    timestamp=pred.timestamp,
                    predicted_drawdown=pred.predicted_drawdown,
                    signal_type=pred.signal_type,
                    version="v5.4.3",
                    nifty_spot=pred.nifty_spot,
                    vix=pred.vix,
                    confidence_score=pred.confidence_score,
                    graduated_mult=pred.graduated_mult,
                    features=pred.features,
                )
                db.add(new_pred)

            await db.commit()

            return {
                "status": "cloned",
                "deleted_v543": deleted_summary,
                "cloned_from_v542": {
                    "trades": len(cloned_trades),
                    "daily_pnl": len(v542_pnl),
                    "predictions": len(v542_preds),
                    "trade_ids": cloned_trades,
                },
            }

    except Exception as e:
        logger.error(f"Clone v542->v543 failed: {e}", exc_info=True)
        import traceback
        return {"status": "error", "message": str(e), "traceback": traceback.format_exc()}


@app.post("/api/debug/correct-stale-trades")
async def correct_stale_trades(correct_spot: float, correct_vix: float):
    """
    Correct today's trades and predictions that were opened with stale
    fallback prices (e.g. after a Dhan token expiry). Recalculates strikes
    and credit using the correct spot/VIX values.
    """
    from datetime import date as date_cls
    from sqlalchemy import select
    from db.database import async_session_factory
    from db.models import Trade, Prediction
    from core.option_pricer import (
        select_strikes, price_bull_put_spread, price_iron_condor,
        price_bear_put_debit, price_bear_call_spread,
        get_next_weekly_expiry, compute_time_to_expiry_years,
    )
    from config import (
        RISK_FREE_RATE, NIFTY_LOT_SIZE, BULL_OTM_SELL, BULL_OTM_BUY,
        VERSION_CONFIGS,
    )
    from core.timezone import today_ist

    today = today_ist()
    sigma = correct_vix / 100.0

    try:
        async with async_session_factory() as db:
            # ── Fix trades (both open AND closed from today) ──
            from core.option_pricer import compute_spread_value
            result = await db.execute(
                select(Trade).where(Trade.entry_date == today)
            )
            trades = result.scalars().all()

            trade_corrections = []
            for trade in trades:
                old = {
                    "trade_id": trade.trade_id,
                    "status": trade.status,
                    "entry_spot": trade.entry_spot,
                    "entry_vix": trade.entry_vix,
                    "sell_strike": trade.sell_strike,
                    "buy_strike": trade.buy_strike,
                    "credit_received": trade.credit_received,
                    "total_credit": trade.total_credit,
                    "realized_pnl": trade.realized_pnl,
                    "current_pnl": trade.current_pnl,
                    "current_pnl_pct": trade.current_pnl_pct,
                }

                cfg = VERSION_CONFIGS.get(trade.version, {})
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

                new_strikes = select_strikes(correct_spot, trade.trade_type, strike_cfg)
                T_entry = compute_time_to_expiry_years(today, trade.expiry)

                if trade.trade_type == "bull_put":
                    new_credit = price_bull_put_spread(
                        correct_spot, new_strikes["sell_strike"], new_strikes["buy_strike"],
                        T_entry, RISK_FREE_RATE, sigma, apply_slippage=True,
                    )
                elif trade.trade_type == "iron_condor":
                    new_credit = price_iron_condor(
                        correct_spot, new_strikes["sell_strike"], new_strikes["buy_strike"],
                        new_strikes["ic_call_sell"], new_strikes["ic_call_buy"],
                        T_entry, RISK_FREE_RATE, sigma, apply_slippage=True,
                    )
                elif trade.trade_type == "bear_call":
                    new_credit = price_bear_call_spread(
                        correct_spot, new_strikes["ic_call_sell"], new_strikes["ic_call_buy"],
                        T_entry, RISK_FREE_RATE, sigma, apply_slippage=True,
                    )
                else:
                    new_credit = trade.credit_received

                new_total_credit = new_credit * trade.num_lots * NIFTY_LOT_SIZE

                # Update entry fields
                trade.entry_spot = correct_spot
                trade.entry_vix = correct_vix
                trade.sell_strike = new_strikes["sell_strike"]
                trade.buy_strike = new_strikes["buy_strike"]
                trade.ic_call_sell = new_strikes.get("ic_call_sell")
                trade.ic_call_buy = new_strikes.get("ic_call_buy")
                trade.credit_received = new_credit
                trade.total_credit = new_total_credit

                new_info = {
                    "entry_spot": correct_spot,
                    "entry_vix": correct_vix,
                    "sell_strike": new_strikes["sell_strike"],
                    "buy_strike": new_strikes["buy_strike"],
                    "credit_received": round(new_credit, 4),
                    "total_credit": round(new_total_credit, 2),
                }

                # For closed trades, also recompute exit PnL
                if trade.status == "closed" and trade.exit_spot:
                    T_exit = max((trade.expiry - today).days / 365.0, 1 / 365.0)
                    exit_value = compute_spread_value(
                        trade.trade_type, trade.exit_spot,
                        new_strikes["sell_strike"], new_strikes["buy_strike"],
                        new_strikes.get("ic_call_sell"), new_strikes.get("ic_call_buy"),
                        T_exit, RISK_FREE_RATE, sigma,
                    )
                    new_realized = (new_credit - exit_value) * trade.num_lots * NIFTY_LOT_SIZE
                    new_pnl_pct = (
                        new_realized / trade.capital_deployed * 100
                        if trade.capital_deployed > 0 else 0
                    )
                    trade.realized_pnl = new_realized
                    trade.current_pnl = new_realized
                    trade.current_pnl_pct = new_pnl_pct
                    trade.unrealized_pnl = 0.0
                    new_info.update({
                        "exit_value": round(exit_value, 4),
                        "realized_pnl": round(new_realized, 2),
                        "current_pnl_pct": round(new_pnl_pct, 4),
                    })

                trade_corrections.append({
                    "trade_id": trade.trade_id,
                    "old": old,
                    "new": new_info,
                })

            # ── Fix predictions ──
            result = await db.execute(
                select(Prediction).where(Prediction.date == today)
            )
            predictions = result.scalars().all()

            pred_corrections = []
            for pred in predictions:
                old_pred = {
                    "version": pred.version,
                    "nifty_spot": pred.nifty_spot,
                    "vix": pred.vix,
                    "signal_type": pred.signal_type,
                }

                pred.nifty_spot = correct_spot
                pred.vix = correct_vix
                # Remove * suffix if present (stale price flag)
                if pred.signal_type.endswith("*"):
                    pred.signal_type = pred.signal_type[:-1]

                pred_corrections.append({
                    "version": pred.version,
                    "old": old_pred,
                    "new": {
                        "nifty_spot": correct_spot,
                        "vix": correct_vix,
                        "signal_type": pred.signal_type,
                    },
                })

            await db.commit()

            return {
                "status": "corrected",
                "correct_spot": correct_spot,
                "correct_vix": correct_vix,
                "trades_corrected": len(trade_corrections),
                "predictions_corrected": len(pred_corrections),
                "trade_details": trade_corrections,
                "prediction_details": pred_corrections,
            }

    except Exception as e:
        logger.error(f"Correct stale trades failed: {e}", exc_info=True)
        import traceback
        return {"status": "error", "message": str(e), "traceback": traceback.format_exc()}


@app.post("/api/debug/reopen-trade")
async def reopen_trade(trade_id: str):
    """
    Reopen a trade that was incorrectly closed (e.g. exit check ran against
    stale entry data). Clears all exit fields and resets exit-tracking state
    (peak_pnl_pct, trailing_stop_active, stop_loss counters) so the trade
    resumes as if it was just opened with its current entry parameters.
    """
    from sqlalchemy import select
    from db.database import async_session_factory
    from db.models import Trade

    try:
        async with async_session_factory() as db:
            result = await db.execute(
                select(Trade).where(Trade.trade_id == trade_id)
            )
            trade = result.scalar_one_or_none()

            if not trade:
                return {"status": "not_found", "message": f"No trade: {trade_id}"}

            if trade.status != "closed":
                return {"status": "skipped", "message": f"Trade is already {trade.status}"}

            old = {
                "trade_id": trade.trade_id,
                "status": trade.status,
                "exit_date": str(trade.exit_date),
                "exit_time": str(trade.exit_time),
                "exit_spot": trade.exit_spot,
                "exit_reason": trade.exit_reason,
                "realized_pnl": trade.realized_pnl,
                "current_pnl": trade.current_pnl,
                "current_pnl_pct": trade.current_pnl_pct,
                "peak_pnl_pct": trade.peak_pnl_pct,
                "trailing_stop_active": trade.trailing_stop_active,
                "stop_loss_breach_days": trade.stop_loss_breach_days,
            }

            # Clear exit fields
            trade.status = "open"
            trade.exit_date = None
            trade.exit_time = None
            trade.exit_spot = None
            trade.exit_reason = None
            trade.realized_pnl = None

            # Reset PnL to zero (next exit check will recompute)
            trade.current_pnl = 0
            trade.current_pnl_pct = 0
            trade.unrealized_pnl = 0
            trade.current_spread_value = None

            # Reset exit-tracking state that was corrupted by stale data
            trade.peak_pnl_pct = 0.0
            trade.trailing_stop_active = False
            trade.stop_loss_breach_days = 0
            trade.stop_loss_last_breach_date = None

            await db.commit()

            return {
                "status": "reopened",
                "trade_id": trade_id,
                "old": old,
                "new": {
                    "status": "open",
                    "exit_date": None,
                    "exit_reason": None,
                    "realized_pnl": None,
                    "peak_pnl_pct": 0.0,
                    "trailing_stop_active": False,
                    "stop_loss_breach_days": 0,
                    "entry_spot": trade.entry_spot,
                    "entry_vix": trade.entry_vix,
                    "sell_strike": trade.sell_strike,
                    "buy_strike": trade.buy_strike,
                    "credit_received": trade.credit_received,
                },
            }

    except Exception as e:
        logger.error(f"Reopen trade failed for {trade_id}: {e}", exc_info=True)
        import traceback
        return {"status": "error", "message": str(e), "traceback": traceback.format_exc()}
