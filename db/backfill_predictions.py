"""
Backfill historical predictions from the backtest feature matrix + trained model.

Generates one prediction per trading day from 2024-01-01 to 2025-12-31,
using the same GradientBoosting model used in live predictions.

Usage (from backend/ directory):
    DATABASE_URL="postgresql://..." python -m db.backfill_predictions

Requires the nifty_options_model directory to be at the same level as backend/.
"""

import os
import sys
import asyncio
from datetime import datetime, timezone

import pandas as pd
import joblib
from sqlalchemy import select, text


# Resolve paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_DIR = os.path.dirname(BACKEND_DIR)
MODEL_DIR = os.path.join(PROJECT_DIR, "nifty_options_model")

FEATURE_MATRIX_PATH = os.path.join(MODEL_DIR, "data", "features", "feature_matrix.parquet")
MODEL_PATH = os.path.join(MODEL_DIR, "models", "regression_model.pkl")

# Columns that are NOT features (targets + raw price data + labels)
NON_FEATURE_COLS = [
    "max_drawdown_30d", "nifty_close", "nifty_open", "nifty_high",
    "nifty_low", "nifty_volume", "nifty_50dma", "nifty_200dma",
    "max_rally_10d", "label", "label_binary",
]

# Signal thresholds (v5.4.4 graduated_gentle — same for all versions)
DRAWDOWN_BULL_FULL = -0.015
DRAWDOWN_BULL_HALF = -0.025
DRAWDOWN_IRON_CONDOR = -0.035


def classify_signal(pred_dd: float) -> str:
    """Map predicted drawdown to signal type."""
    if pred_dd > DRAWDOWN_BULL_FULL:
        return "bull_full"
    elif pred_dd > DRAWDOWN_BULL_HALF:
        return "bull_half"
    elif pred_dd > DRAWDOWN_IRON_CONDOR:
        return "iron_condor"
    else:
        return "no_trade"


async def main():
    # Check paths
    if not os.path.exists(FEATURE_MATRIX_PATH):
        print(f"ERROR: Feature matrix not found at {FEATURE_MATRIX_PATH}")
        sys.exit(1)
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found at {MODEL_PATH}")
        sys.exit(1)

    # Load model and features
    print("Loading model and feature matrix...")
    model = joblib.load(MODEL_PATH)
    df = pd.read_parquet(FEATURE_MATRIX_PATH)

    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    assert len(feature_cols) == model.n_features_in_, (
        f"Feature mismatch: {len(feature_cols)} cols vs {model.n_features_in_} expected"
    )

    # Filter to backfill period (test period onwards — before live system started)
    # Live system started ~2026-01-01, so backfill 2024-01-01 to 2025-12-31
    backfill_df = df[(df.index >= "2024-01-01") & (df.index <= "2025-12-31")]
    print(f"Backfill period: {backfill_df.index[0].date()} to {backfill_df.index[-1].date()}")
    print(f"Trading days to backfill: {len(backfill_df)}")

    # Generate predictions
    print("Generating predictions...")
    predictions_data = []

    X = backfill_df[feature_cols].values
    preds = model.predict(X)

    for i, (date_idx, row) in enumerate(backfill_df.iterrows()):
        pred_dd = float(preds[i])
        nifty_close = float(row.get("nifty_close", 0))
        vix = float(row.get("india_vix", 0))
        signal = classify_signal(pred_dd)

        predictions_data.append({
            "date": date_idx.date(),
            "timestamp": datetime(date_idx.year, date_idx.month, date_idx.day,
                                  9, 20, 0, tzinfo=timezone.utc),
            "predicted_drawdown": pred_dd,
            "signal_type": signal,
            "nifty_spot": nifty_close,
            "vix": vix,
            "confidence_score": 0.0,
            "graduated_mult": 1.0,
        })

    print(f"Generated {len(predictions_data)} predictions")

    # Preview
    print("\nSample predictions:")
    for p in predictions_data[:3] + predictions_data[-3:]:
        print(f"  {p['date']}  pred={p['predicted_drawdown']*100:.2f}%  "
              f"signal={p['signal_type']}  spot={p['nifty_spot']:.0f}  vix={p['vix']:.1f}")

    # Connect to DB and insert using raw SQL (avoids ORM Python 3.9 issues)
    import ssl
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

    db_url = os.environ.get("DATABASE_URL", "")
    if "postgresql://" in db_url and "+asyncpg" not in db_url:
        db_url = db_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    # Strip params that asyncpg doesn't understand (it uses ssl context instead)
    from urllib.parse import urlparse, urlencode, parse_qs, urlunparse
    parsed = urlparse(db_url)
    params = parse_qs(parsed.query)
    params.pop("sslmode", None)
    params.pop("channel_binding", None)
    clean_query = urlencode({k: v[0] for k, v in params.items()})
    db_url = urlunparse(parsed._replace(query=clean_query))

    ssl_ctx = ssl.create_default_context()
    ssl_ctx.check_hostname = False
    ssl_ctx.verify_mode = ssl.CERT_NONE

    eng = create_async_engine(db_url, connect_args={"ssl": ssl_ctx})
    session_factory = async_sessionmaker(eng, class_=AsyncSession, expire_on_commit=False)

    # Use a single version for backfill (v5.4.2 — the first active version)
    # All versions share the same predicted_drawdown, so one is enough
    VERSION = "v5.4.2"

    async with session_factory() as session:
        # Check existing
        result = await session.execute(
            text("SELECT date FROM predictions WHERE version = :v AND date >= :d1 AND date <= :d2"),
            {"v": VERSION, "d1": backfill_df.index[0].date(), "d2": backfill_df.index[-1].date()},
        )
        existing_dates = {row[0] for row in result.fetchall()}
        print(f"\nExisting predictions for {VERSION}: {len(existing_dates)}")

        # Insert only new dates
        inserted = 0
        for p in predictions_data:
            if p["date"] in existing_dates:
                continue

            await session.execute(
                text("""
                    INSERT INTO predictions
                        (date, timestamp, predicted_drawdown, signal_type, version,
                         nifty_spot, vix, confidence_score, graduated_mult)
                    VALUES
                        (:date, :ts, :pred, :signal, :version,
                         :spot, :vix, :conf, :grad)
                    ON CONFLICT (date, version) DO NOTHING
                """),
                {
                    "date": p["date"],
                    "ts": p["timestamp"],
                    "pred": p["predicted_drawdown"],
                    "signal": p["signal_type"],
                    "version": VERSION,
                    "spot": p["nifty_spot"],
                    "vix": p["vix"],
                    "conf": p["confidence_score"],
                    "grad": p["graduated_mult"],
                },
            )
            inserted += 1

            if inserted % 100 == 0:
                await session.commit()
                print(f"  Inserted {inserted} so far...")

        await session.commit()
        print(f"\nInserted {inserted} new predictions (skipped {len(existing_dates)} existing)")

    # Final verification
    async with session_factory() as session:
        result = await session.execute(
            text("SELECT COUNT(*), MIN(date), MAX(date) FROM predictions WHERE version = :v"),
            {"v": VERSION},
        )
        row = result.fetchone()
        print(f"\nTotal predictions in DB for {VERSION}: {row[0]}")
        print(f"Date range: {row[1]} to {row[2]}")

    await eng.dispose()
    print("\nBackfill complete!")


if __name__ == "__main__":
    asyncio.run(main())
