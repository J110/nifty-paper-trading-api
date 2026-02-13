"""
Automated daily data updater for merged_daily.parquet.

Downloads the latest market data from yfinance and appends it to the
existing parquet file. Runs as a scheduled job at 9:05 AM IST (before
the 9:20 AM prediction pipeline).

This ensures the server always has yesterday's close prices available
for feature computation — without requiring manual parquet updates
and redeployment.

Sources (same as nifty_options_model/src/data_collection.py):
  1. Nifty 50       (^NSEI)
  2. India VIX      (^INDIAVIX)
  3. S&P 500        (^GSPC)
  4. US VIX         (^VIX)
  5. DXY            (DX-Y.NYB)
  6. US 10Y Yield   (^TNX)
"""

import logging
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

BACKEND_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MERGED_DAILY_PATH = os.path.join(BACKEND_ROOT, "data", "merged_daily.parquet")

# Tickers to download — must match nifty_options_model/src/data_collection.py
TICKERS = {
    "nifty": {"symbol": "^NSEI", "col": None},       # OHLCV (special handling)
    "india_vix": {"symbol": "^INDIAVIX", "col": "india_vix"},
    "sp500": {"symbol": "^GSPC", "col": "sp500_close"},
    "us_vix": {"symbol": "^VIX", "col": "us_vix"},
    "dxy": {"symbol": "DX-Y.NYB", "col": "dxy_close"},
    "us10y": {"symbol": "^TNX", "col": "us10y_yield"},
}


async def update_merged_daily() -> dict:
    """
    Update merged_daily.parquet with the latest data from yfinance.

    Strategy:
    - Read existing parquet to find last date
    - Download data from (last_date - 5 days) to today (overlap for safety)
    - Merge new data with existing, deduplicating on date index
    - Save updated parquet back to disk

    Returns a status dict with details about what was updated.
    """
    import yfinance as yf

    result = {
        "status": "unknown",
        "last_date_before": None,
        "last_date_after": None,
        "rows_added": 0,
    }

    try:
        # Load existing parquet
        if not os.path.exists(MERGED_DAILY_PATH):
            logger.error(f"merged_daily.parquet not found at {MERGED_DAILY_PATH}")
            result["status"] = "error"
            result["error"] = "Parquet file not found"
            return result

        existing = pd.read_parquet(MERGED_DAILY_PATH)
        last_date = existing.index[-1]
        result["last_date_before"] = str(last_date.date())
        result["rows_before"] = len(existing)

        logger.info(f"Existing parquet: {len(existing)} rows, last date: {last_date.date()}")

        # Check if we actually need to update
        from core.timezone import now_ist
        today = now_ist().date()

        # If last date is today or yesterday (data not yet available), skip
        if last_date.date() >= today:
            logger.info(f"Parquet already up to date (last: {last_date.date()}, today: {today})")
            result["status"] = "already_current"
            result["last_date_after"] = str(last_date.date())
            return result

        # Download from a few days before last date (overlap for data corrections)
        start_date = (last_date - timedelta(days=5)).strftime("%Y-%m-%d")
        end_date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

        logger.info(f"Downloading data from {start_date} to {end_date}")

        # Download Nifty (primary — determines trading calendar)
        nifty = _download_safe(yf, "^NSEI", "Nifty 50", start_date, end_date)
        if nifty is None or nifty.empty:
            logger.warning("No new Nifty data available — market may be closed")
            result["status"] = "no_new_data"
            result["last_date_after"] = str(last_date.date())
            return result

        # Build new rows on Nifty's trading calendar
        new_data = pd.DataFrame(index=nifty.index)
        new_data["nifty_open"] = nifty["Open"]
        new_data["nifty_high"] = nifty["High"]
        new_data["nifty_low"] = nifty["Low"]
        new_data["nifty_close"] = nifty["Close"]
        new_data["nifty_volume"] = nifty["Volume"]

        # Download other tickers
        for name, info in TICKERS.items():
            if name == "nifty":
                continue
            col = info["col"]
            ticker_data = _download_safe(yf, info["symbol"], name, start_date, end_date)
            if ticker_data is not None and not ticker_data.empty:
                new_data[col] = ticker_data["Close"].reindex(new_data.index, method="ffill")
            else:
                logger.warning(f"No data for {name} — will forward-fill from existing")

        # Forward-fill any remaining NaNs from within the new data
        new_data = new_data.ffill()

        # Find truly new rows (dates not already in existing)
        new_dates = new_data.index.difference(existing.index)

        if len(new_dates) == 0:
            logger.info("No new trading dates to add")
            result["status"] = "no_new_dates"
            result["last_date_after"] = str(last_date.date())
            return result

        new_rows = new_data.loc[new_dates].copy()

        # Forward-fill from existing data for any columns that are NaN
        # (e.g., if India VIX wasn't available today, use last known value)
        for col in existing.columns:
            if col in new_rows.columns:
                mask = new_rows[col].isna()
                if mask.any():
                    last_val = existing[col].iloc[-1]
                    new_rows.loc[mask, col] = last_val

        # Ensure column order matches existing
        new_rows = new_rows[existing.columns]

        # Append and sort
        updated = pd.concat([existing, new_rows])
        updated = updated.sort_index()
        updated = updated[~updated.index.duplicated(keep="last")]

        # Validate: no huge gaps
        date_diff = (updated.index[-1] - updated.index[-2]).days
        if date_diff > 10:
            logger.warning(f"Large gap detected: {date_diff} days between last two dates")

        # Save
        updated.to_parquet(MERGED_DAILY_PATH)

        result["status"] = "updated"
        result["last_date_after"] = str(updated.index[-1].date())
        result["rows_after"] = len(updated)
        result["rows_added"] = len(new_rows)
        result["new_dates"] = [str(d.date()) for d in new_dates.sort_values()]

        logger.info(
            f"Parquet updated: {len(existing)} → {len(updated)} rows, "
            f"added {len(new_rows)} new dates: {result['new_dates']}"
        )

        return result

    except Exception as e:
        logger.error(f"Data update failed: {e}", exc_info=True)
        result["status"] = "error"
        result["error"] = str(e)
        return result


def _download_safe(yf, symbol: str, name: str, start: str, end: str):
    """Download a ticker from yfinance with error handling."""
    try:
        df = yf.download(symbol, start=start, end=end, progress=False)
        if df.empty:
            logger.warning(f"No data returned for {name} ({symbol})")
            return None
        # Handle MultiIndex columns (yfinance sometimes returns these)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        logger.info(f"Downloaded {name}: {len(df)} rows, "
                    f"{df.index[0].date()} to {df.index[-1].date()}")
        return df
    except Exception as e:
        logger.warning(f"Failed to download {name} ({symbol}): {e}")
        return None
