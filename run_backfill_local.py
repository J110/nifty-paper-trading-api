"""
Run the backfill locally against the remote Neon database.
Uses raw asyncpg (no ORM) to bypass Python 3.10+ requirement for SQLAlchemy Mapped types.

KEY DESIGN: This script calls run_backtest() from nifty_options_model/src/backtest.py
directly — the EXACT same code that produces the backtest results. No reimplemented
trade logic. The only code here is data loading, DB writes, and trade-to-DB mapping.

Features are computed via shared/feature_compute.py — the single source of truth.

Usage:
    cd /Users/Anmol/Desktop/Options\ Trading/backend
    python3 run_backfill_local.py
"""

import asyncio
import os
import ssl
import json
import logging
import pickle
import time
import numpy as np
import pandas as pd
from datetime import date, datetime, timedelta

import sys
BACKEND_ROOT = os.path.dirname(os.path.abspath(__file__))

# Add nifty_options_model to path so we can import backtest, config, etc.
NIFTY_MODEL_ROOT = os.path.join(os.path.dirname(BACKEND_ROOT), "nifty_options_model")
sys.path.insert(0, NIFTY_MODEL_ROOT)

import config as backtest_config
from config import get_config_profile
from src.backtest import run_backtest
from src.option_pricing import OptionPricer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Config ───────────────────────────────────────────────────────────
DB_URL = "postgresql://neondb_owner:npg_g8DMmJ3XByNF@ep-flat-tree-a1bahirx-pooler.ap-southeast-1.aws.neon.tech/neondb"
BACKFILL_START = "2024-01-01"
FORWARD_TEST_START = "2026-02-13"  # Protect live trades from backfill wipe

# Version mapping: display version → config profile name
VERSION_MAP = {
    "v5.4.2": "v5.4",
    "v5.4.3": "v5.4.3",
    "v5.4.4": "v5.4.4",
}


async def run_backfill():
    import asyncpg

    ssl_ctx = ssl.create_default_context()
    ssl_ctx.check_hostname = False
    ssl_ctx.verify_mode = ssl.CERT_NONE

    logger.info("Connecting to Neon DB...")
    conn = await asyncpg.connect(DB_URL, ssl=ssl_ctx)
    logger.info("Connected!")

    # Clear only backtest-period data, preserve live/forward-test trades
    logger.info(f"Clearing backtest data (before {FORWARD_TEST_START}), preserving live trades...")
    await conn.execute(f"DELETE FROM delay_prices WHERE signal_date < '{FORWARD_TEST_START}'")
    await conn.execute(f"DELETE FROM price_snapshots WHERE timestamp < '{FORWARD_TEST_START}'")
    await conn.execute(f"DELETE FROM daily_pnl WHERE date < '{FORWARD_TEST_START}'")
    await conn.execute(f"DELETE FROM trades WHERE entry_date < '{FORWARD_TEST_START}'")
    await conn.execute(f"DELETE FROM predictions WHERE date < '{FORWARD_TEST_START}'")
    await conn.execute(f"DELETE FROM daily_features WHERE date < '{FORWARD_TEST_START}'")
    live_trades = await conn.fetchval(f"SELECT count(*) FROM trades WHERE entry_date >= '{FORWARD_TEST_START}'")
    live_preds = await conn.fetchval(f"SELECT count(*) FROM predictions WHERE date >= '{FORWARD_TEST_START}'")
    logger.info(f"Backtest data cleared. Preserved {live_trades} live trades, {live_preds} live predictions")

    # Load feature matrix (already built by main.py pipeline)
    feature_matrix_path = os.path.join(backtest_config.DATA_FEATURES, "feature_matrix.parquet")
    feature_names_path = os.path.join(backtest_config.DATA_FEATURES, "feature_names.json")
    model_path = os.path.join(backtest_config.MODELS_DIR, "regression_model.pkl")

    logger.info(f"Loading feature matrix from {feature_matrix_path}")
    df = pd.read_parquet(feature_matrix_path)
    logger.info(f"Feature matrix: {len(df)} rows, {df.index[0].date()} to {df.index[-1].date()}")

    with open(feature_names_path, 'r') as f:
        feature_cols = json.load(f)
    logger.info(f"Feature columns: {len(feature_cols)}")

    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    logger.info("Model loaded")

    pricer = OptionPricer()
    logger.info("OptionPricer initialized")

    # Load merged_daily for daily feature storage (used for daily_features table)
    merged_daily_path = os.path.join(BACKEND_ROOT, "data", "merged_daily.parquet")
    merged_daily = pd.read_parquet(merged_daily_path) if os.path.exists(merged_daily_path) else None

    stats = {"predictions": 0, "trades_opened": 0, "trades_closed": 0}

    for display_ver, profile_name in VERSION_MAP.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Running backtest for {display_ver} (profile={profile_name})")
        logger.info(f"{'='*60}")

        # Get config and override TEST_START for backfill period
        cfg = get_config_profile(profile_name)
        cfg['TEST_START'] = BACKFILL_START

        # Run the EXACT same backtest code
        result = run_backtest(df, feature_cols, model=model, pricer=pricer, config_dict=cfg)

        trades = result['trades']
        daily_preds = result.get('daily_predictions', [])
        equity_curve = result.get('equity_curve', pd.DataFrame())

        logger.info(f"{display_ver}: {len(trades)} trades, {len(daily_preds)} daily predictions")

        # ── Write predictions to DB ──────────────────────────────
        for pred in daily_preds:
            pred_date = pred['date']
            if isinstance(pred_date, pd.Timestamp):
                pred_date = pred_date.date()
            elif isinstance(pred_date, str):
                pred_date = datetime.strptime(pred_date, "%Y-%m-%d").date()

            ts_val = datetime.combine(pred_date, datetime.min.time().replace(hour=9, minute=20))

            # Build features dict for storage
            features_dict = {}
            if merged_daily is not None:
                ts_key = pd.Timestamp(pred_date)
                if ts_key in merged_daily.index:
                    row = merged_daily.loc[ts_key]
                    features_dict = {
                        "india_vix": float(row.get("india_vix", 0)),
                        "nifty_close": float(row.get("nifty_close", 0)),
                    }

            # Confidence score (simple: distance from nearest threshold)
            pred_dd = pred.get('predicted_drawdown', 0)
            thresholds = [-0.038, -0.050, -0.065]
            confidence = min(abs(pred_dd - t) for t in thresholds) if thresholds else 0.5

            await conn.execute("""
                INSERT INTO predictions (date, timestamp, predicted_drawdown, signal_type, version,
                    nifty_spot, vix, confidence_score, graduated_mult, features)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                ON CONFLICT ON CONSTRAINT uq_prediction_date_version DO NOTHING
            """, pred_date, ts_val,
                float(pred_dd),
                str(pred.get('signal', 'unknown')),
                display_ver,
                float(pred.get('spot', 0)),
                float(pred.get('vix', 0)),
                float(confidence),
                float(pred.get('graduated_mult', 1.0)),
                json.dumps(features_dict))
            stats["predictions"] += 1

        # ── Write daily features to DB (once, not per version) ───
        if display_ver == list(VERSION_MAP.keys())[0]:
            for pred in daily_preds:
                pred_date = pred['date']
                if isinstance(pred_date, pd.Timestamp):
                    pred_date = pred_date.date()
                elif isinstance(pred_date, str):
                    pred_date = datetime.strptime(pred_date, "%Y-%m-%d").date()

                vix = float(pred.get('vix', 0))
                # Use feature matrix (computed features) for display indicators
                features_dict = {}
                ts_key = pd.Timestamp(pred_date)
                if ts_key in df.index:
                    row = df.loc[ts_key]
                    features_dict = {c: float(row[c]) for c in row.index if not pd.isna(row[c])}

                await conn.execute("""
                    INSERT INTO daily_features (date, features, vix, vix_20d_avg, nifty_20d_return,
                        nifty_50d_return, iv_skew, fii_net, dii_net, put_call_ratio, rsi_14, adx_14)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                    ON CONFLICT (date) DO NOTHING
                """, pred_date, json.dumps(features_dict), vix, vix,
                    features_dict.get("nifty_return_20d", 0),
                    features_dict.get("nifty_distance_50dma", 0),
                    features_dict.get("iv_skew_steepness", 0),
                    0.0, 0.0,
                    features_dict.get("pcr_proxy", 0),
                    features_dict.get("rsi_14", 50),
                    0.0)

        # ── Write trades to DB ───────────────────────────────────
        for trade in trades:
            entry_date = trade.entry_date
            if isinstance(entry_date, str):
                entry_date = datetime.strptime(entry_date, "%Y-%m-%d").date()
            elif isinstance(entry_date, pd.Timestamp):
                entry_date = entry_date.date()

            exit_date = trade.exit_date
            if isinstance(exit_date, str) and exit_date:
                exit_date = datetime.strptime(exit_date, "%Y-%m-%d").date()
            elif isinstance(exit_date, pd.Timestamp):
                exit_date = exit_date.date()
            else:
                exit_date = None

            # Determine trade status
            if trade.exit_reason == "end_of_period":
                status = "open"
                exit_date = None
                exit_reason = None
            else:
                status = "closed"
                exit_reason = trade.exit_reason

            # Build trade_id
            trade_id = f"{display_ver.replace('.', '')}-{entry_date.isoformat()}-{trade.signal_type}-{trade.trade_id}"

            entry_time = datetime.combine(entry_date, datetime.min.time().replace(hour=9, minute=20))
            exit_time = datetime.combine(exit_date, datetime.min.time().replace(hour=15, minute=30)) if exit_date else None

            # Expiry date
            expiry_date = trade.expiry_date
            if isinstance(expiry_date, str) and expiry_date:
                expiry_date = datetime.strptime(expiry_date, "%Y-%m-%d").date()
            elif isinstance(expiry_date, pd.Timestamp):
                expiry_date = expiry_date.date()

            # Capital deployed
            margin_per_lot = cfg.get('MARGIN_PER_LOT_IC', 13750) if trade.trade_type == "iron_condor" else cfg.get('MARGIN_PER_LOT_BULL', 13750)
            capital_deployed = trade.num_lots * margin_per_lot

            # PnL percentage
            pnl_pct = (trade.pnl / capital_deployed * 100) if capital_deployed > 0 else 0

            await conn.execute("""
                INSERT INTO trades (trade_id, version, date, signal_type, trade_type, entry_mode,
                    entry_date, entry_time, entry_spot, expiry, sell_strike, buy_strike,
                    ic_call_sell, ic_call_buy, num_lots, credit_received, total_credit,
                    status, current_pnl, current_pnl_pct, unrealized_pnl,
                    position_size_pct, graduated_mult, capital_deployed,
                    exit_date, exit_spot, exit_reason, realized_pnl, exit_time)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19,$20,$21,$22,$23,$24,$25,$26,$27,$28,$29)
            """,
                trade_id, display_ver, entry_date,
                trade.signal_type, trade.trade_type, trade.entry_mode or "normal",
                entry_date, entry_time, trade.spot_at_entry,
                expiry_date,
                trade.put_sell_strike, trade.put_buy_strike,
                trade.call_sell_strike if trade.call_sell_strike > 0 else None,
                trade.call_buy_strike if trade.call_buy_strike > 0 else None,
                trade.num_lots, trade.net_premium,
                trade.net_premium * trade.num_lots * cfg.get('NIFTY_LOT_SIZE', 25),
                status,
                trade.pnl if status == "closed" else 0.0,
                pnl_pct if status == "closed" else 0.0,
                0.0 if status == "closed" else trade.pnl,
                trade.position_size, trade.graduated_mult, float(capital_deployed),
                exit_date,
                trade.spot_at_entry if exit_date else None,  # exit_spot not tracked in Trade dataclass; using entry as placeholder
                exit_reason,
                trade.pnl if status == "closed" else None,
                exit_time)

            if status == "closed":
                stats["trades_closed"] += 1
            else:
                stats["trades_opened"] += 1

        # ── Write daily PnL to DB ───────────────────────────────
        if not equity_curve.empty:
            initial_capital = cfg.get('INITIAL_CAPITAL', 2_500_000)
            prev_capital = initial_capital
            for idx, row in equity_curve.iterrows():
                eq_date = idx
                if isinstance(eq_date, pd.Timestamp):
                    eq_date = eq_date.date()
                elif isinstance(eq_date, str):
                    eq_date = datetime.strptime(eq_date, "%Y-%m-%d").date()

                current_capital = float(row.get('capital', initial_capital))
                daily_change = current_capital - prev_capital
                cumulative_pnl = current_capital - initial_capital
                cumulative_return_pct = round(cumulative_pnl / initial_capital * 100, 4)

                await conn.execute("""
                    INSERT INTO daily_pnl (date, version, starting_capital, ending_capital,
                        daily_pnl, cumulative_pnl, cumulative_return_pct, open_positions,
                        trades_opened, trades_closed)
                    VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10)
                    ON CONFLICT ON CONSTRAINT uq_daily_pnl_date_version DO NOTHING
                """, eq_date, display_ver, float(prev_capital), float(current_capital),
                    float(daily_change), float(cumulative_pnl), float(cumulative_return_pct),
                    int(row.get('num_open_positions', 0)), 0, 0)

                prev_capital = current_capital

        # Print trade summary
        closed_trades = [t for t in trades if t.exit_reason != "end_of_period"]
        open_trades = [t for t in trades if t.exit_reason == "end_of_period"]
        total_pnl = sum(t.pnl for t in closed_trades)
        logger.info(f"{display_ver}: {len(closed_trades)} closed trades, "
                    f"{len(open_trades)} open, Total PnL: Rs {total_pnl:,.0f}")

        # Detailed trade log
        print(f"\n--- {display_ver} ({len(closed_trades)} closed, {len(open_trades)} open, "
              f"PnL: Rs {total_pnl:,.0f}) ---")
        print(f"  {'Type':<15} {'Entry':<12} {'Exit':<12} {'PnL':>10} {'Exit Reason':<20} {'Mode':<15}")
        print(f"  {'-'*85}")
        for t in trades:
            exit_d = t.exit_date if t.exit_date else "open"
            pnl_str = f"Rs {t.pnl:>8,.0f}" if t.pnl != 0 else "Rs       0"
            print(f"  {t.trade_type:<15} {t.entry_date:<12} {str(exit_d):<12} "
                  f"{pnl_str} {t.exit_reason:<20} {t.entry_mode or 'normal':<15}")

    await conn.close()
    logger.info(f"\n=== BACKFILL COMPLETE: {stats} ===")
    return stats


if __name__ == "__main__":
    result = asyncio.run(run_backfill())
    print(f"\nResult: {result}")
