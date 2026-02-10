"""
Historical backfill script — populates the database with predictions and trades
from Jan 1, 2026 to today by running the ML model against historical Nifty data.

CRITICAL: Features must match the exact 37 training feature names and computation
from nifty_options_model/src/feature_engineering.py.

Usage: Called via POST /api/backfill endpoint on Render.
"""

import logging
import math
import numpy as np
import pandas as pd
import ta as ta_lib
from datetime import date, datetime, timedelta

from core.timezone import today_ist
from db.models import Prediction, Trade, DailyPnl, DailyFeature
from core.model_runner import ModelRunner
from core.signal_mapper import map_signal, get_classification_breakdown
from core.option_pricer import (
    select_strikes, price_bull_put_spread, price_iron_condor,
    get_next_weekly_expiry, compute_time_to_expiry_years,
    compute_spread_value,
)
from config import (
    ACTIVE_VERSIONS, VERSION_CONFIGS, INITIAL_CAPITAL,
    DOWNSIDE_MODEL_PATH, SCALER_PATH, FEATURE_NAMES_PATH,
    NIFTY_LOT_SIZE, RISK_FREE_RATE,
    MARGIN_PER_LOT_BULL, MARGIN_PER_LOT_IC,
    BULL_OTM_SELL, BULL_OTM_BUY,
    VIX_MIN_ENTRY, VIX_MAX_ENTRY,
    DTE_MIN_ENTRY, DTE_MAX_ENTRY,
)

logger = logging.getLogger(__name__)

BACKFILL_START = date(2026, 1, 1)


# ── helpers ──────────────────────────────────────────────────────────

def _get_last_thursday(year: int, month: int) -> date:
    """Get the last Thursday of a given month (monthly expiry)."""
    if month == 12:
        next_month_first = date(year + 1, 1, 1)
    else:
        next_month_first = date(year, month + 1, 1)
    last_day = next_month_first - timedelta(days=1)
    # Walk back to Thursday (weekday 3)
    offset = (last_day.weekday() - 3) % 7
    return last_day - timedelta(days=offset)


def _get_monthly_expiries(start: date, end: date) -> list:
    """Generate list of monthly expiry dates (last Thursday) in range."""
    expiries = []
    y, m = start.year, start.month
    while True:
        exp = _get_last_thursday(y, m)
        if exp > end:
            break
        if exp >= start:
            expiries.append(exp)
        m += 1
        if m > 12:
            m = 1
            y += 1
    return expiries


def _safe_yf_download(ticker: str, start: str, end: str, retries: int = 3) -> pd.DataFrame:
    """Download a single ticker from yfinance with retries."""
    import yfinance as yf
    import time

    for attempt in range(retries):
        try:
            data = yf.download(ticker, start=start, end=end, progress=False)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            if not data.empty:
                logger.info(f"  {ticker}: {len(data)} days (attempt {attempt + 1})")
                return data
            logger.warning(f"  {ticker}: empty response (attempt {attempt + 1})")
        except Exception as e:
            logger.warning(f"  {ticker}: error on attempt {attempt + 1}: {e}")

        if attempt < retries - 1:
            time.sleep(2 * (attempt + 1))  # increasing delay between retries

    logger.error(f"  {ticker}: all {retries} attempts failed, returning empty DataFrame")
    return pd.DataFrame()


def _download_market_data(start: date, end: date):
    """
    Download all market data needed for the 37 training features:
    Nifty, India VIX, S&P 500, US VIX, DXY, US 10Y Treasury.
    Downloads each ticker with retries and delay to avoid rate limiting.
    Returns aligned DataFrames.
    """
    import time

    start_str = start.isoformat()
    end_str = end.isoformat()
    logger.info(f"Downloading market data from {start} to {end}")

    tickers = {
        "nifty": "^NSEI",
        "india_vix": "^INDIAVIX",
        "sp500": "^GSPC",
        "us_vix": "^VIX",
        "dxy": "DX-Y.NYB",
        "us10y": "^TNX",
    }

    results = {}
    for name, ticker in tickers.items():
        results[name] = _safe_yf_download(ticker, start_str, end_str)
        time.sleep(1)  # small delay between downloads to avoid rate limiting

    return (results["nifty"], results["india_vix"], results["sp500"],
            results["us_vix"], results["dxy"], results["us10y"])


def _build_feature_matrix(nifty: pd.DataFrame, india_vix_df: pd.DataFrame,
                           sp500: pd.DataFrame, us_vix_df: pd.DataFrame,
                           dxy: pd.DataFrame, us10y: pd.DataFrame) -> pd.DataFrame:
    """
    Build the full feature matrix matching the exact 37 training features.
    Returns DataFrame indexed by date with columns matching feature_names.json.
    """
    # Work with Nifty as the base index
    df = pd.DataFrame(index=nifty.index)
    close = nifty["Close"]
    high = nifty["High"]

    # ── 1. Trend & Momentum ─────────────────────────────────────────
    df["nifty_return_5d"] = close.pct_change(5)
    df["nifty_return_10d"] = close.pct_change(10)
    df["nifty_return_20d"] = close.pct_change(20)

    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    df["nifty_distance_50dma"] = (close - sma50) / sma50 * 100
    df["nifty_distance_200dma"] = (close - sma200) / sma200 * 100
    df["golden_cross"] = (sma50 > sma200).astype(int)

    rsi = ta_lib.momentum.RSIIndicator(close, window=14)
    df["rsi_14"] = rsi.rsi()

    # Higher highs over 5 weeks
    weekly_high = high.resample("W").max()
    weekly_hh = (weekly_high > weekly_high.shift(1)).astype(int)
    weekly_hh_count = weekly_hh.rolling(5, min_periods=1).sum()
    df["higher_highs_5w"] = weekly_hh_count.reindex(df.index, method="ffill").fillna(0)

    # ── 2. Volatility ───────────────────────────────────────────────
    # Align India VIX to Nifty dates
    vix_close = india_vix_df["Close"].reindex(df.index).ffill().fillna(14.0)
    df["india_vix"] = vix_close

    df["vix_change_5d"] = vix_close - vix_close.shift(5)

    # VIX percentile over 252 days
    df["vix_percentile_252d"] = vix_close.rolling(252, min_periods=60).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    )

    # Realized volatility (annualized, in percentage)
    log_returns = np.log(close / close.shift(1))
    df["realized_vol_20d"] = log_returns.rolling(20).std() * np.sqrt(252) * 100

    # Variance risk premium
    df["variance_risk_premium"] = df["india_vix"] - df["realized_vol_20d"]

    # VIX rising streak
    vix_rising = (vix_close > vix_close.shift(1)).astype(int)
    vix_streak = pd.Series(0, index=df.index, dtype=int)
    for i in range(1, len(df)):
        if vix_rising.iloc[i] == 1:
            vix_streak.iloc[i] = vix_streak.iloc[i - 1] + 1
        else:
            vix_streak.iloc[i] = 0
    df["vix_rising_streak"] = vix_streak

    # ── 3. Sentiment ────────────────────────────────────────────────
    df["pcr_proxy"] = df["vix_change_5d"] / (df["nifty_return_5d"] * 100 + 0.001)

    # ── 4. Global context ───────────────────────────────────────────
    # S&P 500
    sp_close = sp500["Close"].reindex(df.index).ffill() if len(sp500) > 0 else pd.Series(0, index=df.index)
    if sp_close.notna().sum() >= 100:
        df["sp500_return_5d"] = sp_close.pct_change(5)
    else:
        df["sp500_return_5d"] = 0.0

    # US VIX
    usvix_close = us_vix_df["Close"].reindex(df.index).ffill() if len(us_vix_df) > 0 else pd.Series(0, index=df.index)
    df["us_vix"] = usvix_close.fillna(0)
    if usvix_close.notna().sum() >= 100:
        df["us_vix_change_5d"] = usvix_close - usvix_close.shift(5)
    else:
        df["us_vix_change_5d"] = 0.0

    # DXY
    dxy_close = dxy["Close"].reindex(df.index).ffill() if len(dxy) > 0 else pd.Series(0, index=df.index)
    df["dxy_level"] = dxy_close.fillna(0)
    if dxy_close.notna().sum() >= 100:
        df["dxy_change_5d"] = dxy_close - dxy_close.shift(5)
    else:
        df["dxy_change_5d"] = 0.0

    # US 10Y Treasury
    us10y_close = us10y["Close"].reindex(df.index).ffill() if len(us10y) > 0 else pd.Series(0, index=df.index)
    df["us10y_level"] = us10y_close.fillna(0)
    if us10y_close.notna().sum() >= 100:
        df["us10y_change_5d"] = us10y_close - us10y_close.shift(5)
        df["us10y_change_20d"] = us10y_close - us10y_close.shift(20)
    else:
        df["us10y_change_5d"] = 0.0
        df["us10y_change_20d"] = 0.0

    # ── 5. Calendar ─────────────────────────────────────────────────
    df["day_of_month"] = df.index.day
    df["month"] = df.index.month
    df["is_volatile_month"] = df["month"].isin([9, 10]).astype(int)

    # Monthly expiry features
    first_date = df.index[0].date() if hasattr(df.index[0], 'date') else df.index[0]
    last_date = df.index[-1].date() if hasattr(df.index[-1], 'date') else df.index[-1]
    expiries = _get_monthly_expiries(first_date, last_date + timedelta(days=60))
    trading_dates = df.index.date if hasattr(df.index[0], 'date') else df.index

    days_to_expiry_list = []
    is_expiry_week_list = []
    for d in df.index:
        d_date = d.date() if hasattr(d, 'date') else d
        next_exp = None
        for exp in expiries:
            if exp >= d_date:
                next_exp = exp
                break
        if next_exp is not None:
            # Count trading days between d and expiry
            mask = [td for td in trading_dates if d_date <= td <= next_exp]
            dte = len(mask)
            days_to_expiry_list.append(dte)
            is_expiry_week_list.append(1 if dte <= 5 else 0)
        else:
            days_to_expiry_list.append(30)
            is_expiry_week_list.append(0)

    df["is_expiry_week"] = is_expiry_week_list
    df["days_to_monthly_expiry"] = days_to_expiry_list

    # ── 6. Options-derived (placeholders — no historical option chain) ──
    df["deep_otm_oi_ratio"] = 0.0
    df["deep_otm_oi_ratio_change_5d"] = 0.0
    df["put_oi_buildup_ratio"] = 0.0
    df["put_volume_surge_ratio"] = 1.0
    df["iv_skew_steepness"] = 0.0
    df["iv_skew_change_5d"] = 0.0
    df["atm_iv"] = df["india_vix"] / 100.0  # rough proxy
    df["atm_iv_percentile_252d"] = df["vix_percentile_252d"]  # same proxy
    df["atm_put_intraday_range"] = 0.0

    # ── Clean up ────────────────────────────────────────────────────
    df = df.fillna(0.0)

    # Replace inf/-inf
    df = df.replace([np.inf, -np.inf], 0.0)

    return df


# ── main backfill ────────────────────────────────────────────────────

async def run_backfill(db_session) -> dict:
    """
    Run full historical backfill from Jan 2026 to today.
    Clears existing data first, then re-populates.
    Returns summary stats.
    """
    from sqlalchemy import text

    logger.info("=== Starting historical backfill ===")

    # Clear existing backfill data so re-runs don't hit unique constraint errors
    logger.info("Clearing existing backfill data...")
    await db_session.execute(text("DELETE FROM daily_pnl"))
    await db_session.execute(text("DELETE FROM trades"))
    await db_session.execute(text("DELETE FROM predictions"))
    await db_session.execute(text("DELETE FROM daily_features"))
    await db_session.commit()
    logger.info("Existing data cleared")

    # Download historical data (need extra history for feature computation: 200-day MA, 252-day VIX)
    start_download = date(2025, 1, 1)  # ~1 year before backfill start for warmup
    end_download = today_ist()

    nifty, india_vix_df, sp500, us_vix_df, dxy, us10y = _download_market_data(
        start_download, end_download
    )

    if nifty.empty:
        return {
            "error": "Failed to download Nifty data from yfinance",
            "hint": "yfinance may be blocked from this server's IP. Try running backfill locally.",
            "other_data": {
                "india_vix": len(india_vix_df),
                "sp500": len(sp500),
                "us_vix": len(us_vix_df),
            },
        }

    # Build full feature matrix
    logger.info("Building feature matrix...")
    feature_df = _build_feature_matrix(nifty, india_vix_df, sp500, us_vix_df, dxy, us10y)
    logger.info(f"Feature matrix: {feature_df.shape}")

    # The 37 training feature names (order matters for model prediction)
    TRAINING_FEATURES = [
        "nifty_return_5d", "nifty_return_10d", "nifty_return_20d",
        "nifty_distance_50dma", "nifty_distance_200dma", "golden_cross",
        "rsi_14", "higher_highs_5w", "india_vix", "vix_change_5d",
        "vix_percentile_252d", "realized_vol_20d", "variance_risk_premium",
        "vix_rising_streak", "pcr_proxy", "sp500_return_5d", "us_vix",
        "us_vix_change_5d", "dxy_level", "dxy_change_5d", "us10y_level",
        "us10y_change_5d", "us10y_change_20d", "day_of_month", "month",
        "is_expiry_week", "days_to_monthly_expiry", "is_volatile_month",
        "deep_otm_oi_ratio", "deep_otm_oi_ratio_change_5d",
        "put_oi_buildup_ratio", "put_volume_surge_ratio",
        "iv_skew_steepness", "iv_skew_change_5d", "atm_iv",
        "atm_iv_percentile_252d", "atm_put_intraday_range",
    ]

    # Load model
    model_runner = ModelRunner(DOWNSIDE_MODEL_PATH, SCALER_PATH, FEATURE_NAMES_PATH)
    if model_runner.model is None:
        return {"error": "Model not loaded"}

    # Filter to trading days from BACKFILL_START onwards
    nifty_close = nifty["Close"]
    nifty_dates = nifty.index
    backfill_dates = [d for d in nifty_dates if d.date() >= BACKFILL_START]

    logger.info(f"Backfilling {len(backfill_dates)} trading days from {BACKFILL_START}")

    # Track per-version state
    version_state = {}
    for v in ACTIVE_VERSIONS:
        version_state[v] = {
            "capital": INITIAL_CAPITAL,
            "cumulative_pnl": 0.0,
            "open_trades": [],
        }

    stats = {
        "days_processed": 0,
        "predictions_created": 0,
        "trades_opened": 0,
        "trades_closed": 0,
    }

    for ts in backfill_dates:
        trade_date = ts.date() if hasattr(ts, 'date') else ts

        # Check feature row exists
        if ts not in feature_df.index:
            continue

        spot = float(nifty_close.loc[ts])
        vix = float(feature_df.loc[ts, "india_vix"])

        if pd.isna(spot) or spot <= 0:
            continue

        # Build feature dict from feature matrix (exact 37 training features)
        features = {}
        for col in TRAINING_FEATURES:
            val = feature_df.loc[ts, col] if col in feature_df.columns else 0.0
            features[col] = float(val) if not (pd.isna(val)) else 0.0

        # Run model prediction
        prediction_value = model_runner.predict(features)

        # Store daily features (extra display-friendly names for the frontend)
        daily_feature = DailyFeature(
            date=trade_date,
            features=features,
            vix=vix,
            vix_20d_avg=features.get("india_vix", vix),
            nifty_20d_return=features.get("nifty_return_20d"),
            nifty_50d_return=features.get("nifty_distance_50dma"),
            iv_skew=features.get("iv_skew_steepness"),
            fii_net=0,
            dii_net=0,
            put_call_ratio=features.get("pcr_proxy"),
            rsi_14=features.get("rsi_14"),
            adx_14=0,  # not in training features
        )
        db_session.add(daily_feature)

        # Process each version
        for version in ACTIVE_VERSIONS:
            cfg = VERSION_CONFIGS[version]
            state = version_state[version]
            signal = map_signal(prediction_value, cfg)

            # Confidence
            breakdown = get_classification_breakdown(prediction_value)
            active_zone = next(
                (z for z in breakdown["zones"] if z.get("confidence") == "active"), None
            )
            confidence = active_zone["distance_to_boundary"] if active_zone else 0

            # Store prediction
            pred = Prediction(
                date=trade_date,
                timestamp=datetime.combine(trade_date, datetime.min.time().replace(hour=9, minute=20)),
                predicted_drawdown=prediction_value,
                signal_type=signal["signal"],
                version=version,
                nifty_spot=spot,
                vix=vix,
                confidence_score=confidence,
                graduated_mult=signal["size_mult"],
                features=features,
            )
            db_session.add(pred)
            stats["predictions_created"] += 1

            # Check exits on open trades first
            trades_to_close = []
            for ot in state["open_trades"]:
                dte = (ot["expiry"] - trade_date).days
                T = max(dte / 365.0, 1 / 365.0)
                sigma = vix / 100.0 if vix > 0 else 0.15

                current_value = compute_spread_value(
                    ot["trade_type"], spot,
                    ot["sell_strike"], ot["buy_strike"],
                    ot.get("ic_call_sell"), ot.get("ic_call_buy"),
                    T, RISK_FREE_RATE, sigma,
                )

                pnl_per_unit = ot["credit"] - current_value
                pnl_pct = pnl_per_unit / ot["credit"] if ot["credit"] > 0 else 0

                exit_reason = None

                # Expiry exit
                if dte <= cfg.get("MIN_DTE_EXIT", 1):
                    exit_reason = "expiry"

                # Profit target
                entry_dte = (ot["expiry"] - ot["entry_date"]).days
                elapsed_pct = 1 - (dte / max(entry_dte, 1))
                if elapsed_pct < 0.4:
                    pt = cfg.get("PROFIT_TARGET_EARLY", 0.50)
                elif elapsed_pct < 0.7:
                    pt = cfg.get("PROFIT_TARGET_MID", 0.65)
                else:
                    pt = cfg.get("PROFIT_TARGET_LATE", 0.80)

                if pnl_pct >= pt:
                    exit_reason = "profit_target"

                # Stop loss
                sl_mult = cfg.get("STOP_LOSS_MULTIPLIER", 3.0)
                if ot["trade_type"] == "iron_condor":
                    sl_mult = cfg.get("IC_STOP_LOSS_MULTIPLIER", 3.0)
                max_loss = ot["credit"] * sl_mult
                if current_value >= max_loss + ot["credit"]:
                    exit_reason = "stop_loss"

                if exit_reason:
                    realized_pnl = pnl_per_unit * ot["num_lots"] * NIFTY_LOT_SIZE
                    rpnl_pct = realized_pnl / ot["capital_deployed"] * 100 if ot["capital_deployed"] > 0 else 0

                    from sqlalchemy import update
                    await db_session.execute(
                        update(Trade).where(Trade.trade_id == ot["trade_id"]).values(
                            status="closed",
                            exit_date=trade_date,
                            exit_time=datetime.combine(trade_date, datetime.min.time().replace(hour=15, minute=30)),
                            exit_spot=spot,
                            exit_reason=exit_reason,
                            realized_pnl=realized_pnl,
                            current_pnl=realized_pnl,
                            current_pnl_pct=rpnl_pct,
                        )
                    )
                    state["cumulative_pnl"] += realized_pnl
                    state["capital"] = INITIAL_CAPITAL + state["cumulative_pnl"]
                    trades_to_close.append(ot["trade_id"])
                    stats["trades_closed"] += 1

            # Remove closed trades
            state["open_trades"] = [
                t for t in state["open_trades"] if t["trade_id"] not in trades_to_close
            ]

            # Open new trade if signal != no_trade
            if signal["signal"] != "no_trade" and signal["trade_type"]:
                trade_type = signal["trade_type"]
                size_mult = signal["size_mult"]

                # Check limits
                open_same_type = sum(
                    1 for t in state["open_trades"] if t["trade_type"] == trade_type
                )
                max_conc = cfg.get("MAX_CONCURRENT_POSITIONS", 3)
                if trade_type == "iron_condor":
                    max_conc = cfg.get("IC_MAX_CONCURRENT", 2)

                # Check entry gap
                last_entry = max(
                    (t["entry_date"] for t in state["open_trades"]), default=None
                )
                min_gap = cfg.get("MIN_ENTRY_GAP_DAYS", 2)
                gap_ok = last_entry is None or (trade_date - last_entry).days >= min_gap

                # VIX filter
                vix_ok = VIX_MIN_ENTRY <= vix <= VIX_MAX_ENTRY

                if open_same_type < max_conc and gap_ok and vix_ok:
                    strikes = select_strikes(spot, trade_type, {
                        "BULL_OTM_SELL": BULL_OTM_SELL,
                        "BULL_OTM_BUY": BULL_OTM_BUY,
                        "IC_PUT_OTM_SELL": cfg.get("IC_PUT_OTM_SELL", 0.03),
                        "IC_PUT_OTM_BUY": cfg.get("IC_PUT_OTM_BUY", 0.055),
                        "IC_CALL_OTM_SELL": cfg.get("IC_CALL_OTM_SELL", 0.04),
                        "IC_CALL_OTM_BUY": cfg.get("IC_CALL_OTM_BUY", 0.065),
                    })

                    expiry = get_next_weekly_expiry(trade_date)
                    T = compute_time_to_expiry_years(trade_date, expiry)
                    sigma = vix / 100.0 if vix > 0 else 0.15

                    if trade_type == "bull_put":
                        credit = price_bull_put_spread(
                            spot, strikes["sell_strike"], strikes["buy_strike"],
                            T, RISK_FREE_RATE, sigma,
                        )
                    else:
                        credit = price_iron_condor(
                            spot, strikes["sell_strike"], strikes["buy_strike"],
                            strikes["ic_call_sell"], strikes["ic_call_buy"],
                            T, RISK_FREE_RATE, sigma,
                        )

                    pos_pct = cfg.get("POSITION_SIZE_PCT", 0.20)
                    if trade_type == "iron_condor":
                        pos_pct = cfg.get("IC_POSITION_SIZE_PCT", 0.15)

                    effective_pct = pos_pct * size_mult
                    margin = MARGIN_PER_LOT_BULL if trade_type == "bull_put" else MARGIN_PER_LOT_IC
                    max_cap = state["capital"] * effective_pct
                    num_lots = max(1, int(max_cap / margin))

                    total_credit = credit * num_lots * NIFTY_LOT_SIZE
                    capital_deployed = num_lots * margin

                    trade_id = f"{version.replace('.', '')}-{trade_date.isoformat()}-{signal['signal']}"

                    trade = Trade(
                        trade_id=trade_id,
                        version=version,
                        date=trade_date,
                        signal_type=signal["signal"],
                        trade_type=trade_type,
                        entry_mode="normal",
                        entry_date=trade_date,
                        entry_time=datetime.combine(trade_date, datetime.min.time().replace(hour=9, minute=20)),
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
                        position_size_pct=effective_pct,
                        graduated_mult=size_mult,
                        capital_deployed=capital_deployed,
                    )
                    db_session.add(trade)

                    state["open_trades"].append({
                        "trade_id": trade_id,
                        "trade_type": trade_type,
                        "entry_date": trade_date,
                        "expiry": expiry,
                        "sell_strike": strikes["sell_strike"],
                        "buy_strike": strikes["buy_strike"],
                        "ic_call_sell": strikes.get("ic_call_sell"),
                        "ic_call_buy": strikes.get("ic_call_buy"),
                        "credit": credit,
                        "num_lots": num_lots,
                        "capital_deployed": capital_deployed,
                    })
                    stats["trades_opened"] += 1

            # Record daily PnL
            daily_pnl_record = DailyPnl(
                date=trade_date,
                version=version,
                starting_capital=INITIAL_CAPITAL,
                ending_capital=state["capital"],
                daily_pnl=0,  # simplified
                cumulative_pnl=state["cumulative_pnl"],
                cumulative_return_pct=round(state["cumulative_pnl"] / INITIAL_CAPITAL * 100, 2),
                open_positions=len(state["open_trades"]),
                trades_opened=0,
                trades_closed=0,
            )
            db_session.add(daily_pnl_record)

        stats["days_processed"] += 1

        # Commit in batches of 10 days
        if stats["days_processed"] % 10 == 0:
            await db_session.commit()
            logger.info(f"Backfill progress: {stats['days_processed']} days, "
                        f"{stats['predictions_created']} predictions, "
                        f"{stats['trades_opened']} trades opened")

    # Final commit
    await db_session.commit()
    logger.info(f"=== Backfill complete: {stats} ===")
    return stats
