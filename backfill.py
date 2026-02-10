"""
Historical backfill script — populates the database with predictions and trades
from Jan 1, 2026 to today by running the ML model against historical Nifty data.

Usage: Called via POST /api/backfill endpoint on Render.
"""

import logging
import math
import numpy as np
import pandas as pd
import ta as ta_lib
from datetime import date, datetime, timedelta

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


def _build_features_for_date(idx: int, df: pd.DataFrame, vix_series: pd.Series,
                               trade_date: date) -> dict:
    """Build feature vector from historical data at a given index."""
    close = df["Close"].iloc[:idx + 1]
    spot = float(close.iloc[-1])
    vix = float(vix_series.iloc[idx]) if idx < len(vix_series) else 14.0

    if pd.isna(vix) or vix <= 0:
        vix = 14.0

    features = {}

    # Price features
    features["nifty_close"] = spot
    features["nifty_5d_return"] = (spot / float(close.iloc[-5]) - 1) if len(close) >= 5 else 0
    features["nifty_10d_return"] = (spot / float(close.iloc[-10]) - 1) if len(close) >= 10 else 0
    features["nifty_20d_return"] = (spot / float(close.iloc[-20]) - 1) if len(close) >= 20 else 0
    features["nifty_50d_return"] = (spot / float(close.iloc[-50]) - 1) if len(close) >= 50 else 0

    # SMA distance
    if len(close) >= 20:
        sma20 = float(close.rolling(20).mean().iloc[-1])
        features["dist_sma20"] = (spot - sma20) / sma20
    else:
        features["dist_sma20"] = 0

    if len(close) >= 50:
        sma50 = float(close.rolling(50).mean().iloc[-1])
        features["dist_sma50"] = (spot - sma50) / sma50
    else:
        features["dist_sma50"] = 0

    # Volatility
    if len(close) >= 20:
        log_ret = np.log(close / close.shift(1)).dropna()
        features["realized_vol_20d"] = float(log_ret.tail(20).std() * np.sqrt(252))
        features["realized_vol_10d"] = float(log_ret.tail(10).std() * np.sqrt(252))
    else:
        features["realized_vol_20d"] = 0.15
        features["realized_vol_10d"] = 0.15

    # VIX
    features["vix"] = vix
    features["vix_20d_avg"] = vix
    features["vix_percentile_20d"] = 0.5
    if idx >= 20 and idx < len(vix_series):
        vix_window = vix_series.iloc[max(0, idx - 20):idx + 1].dropna()
        if len(vix_window) > 5:
            features["vix_20d_avg"] = float(vix_window.mean())
            features["vix_percentile_20d"] = float(
                (vix_window < vix).sum() / len(vix_window)
            )

    # Technical indicators
    if len(close) >= 14:
        rsi = ta_lib.momentum.RSIIndicator(close, window=14)
        rsi_val = rsi.rsi().iloc[-1]
        features["rsi_14"] = float(rsi_val) if not pd.isna(rsi_val) else 50.0
    else:
        features["rsi_14"] = 50.0

    high_series = df["High"].iloc[:idx + 1]
    low_series = df["Low"].iloc[:idx + 1]

    if len(close) >= 14:
        adx = ta_lib.trend.ADXIndicator(high_series, low_series, close, window=14)
        adx_val = adx.adx().iloc[-1]
        features["adx_14"] = float(adx_val) if not pd.isna(adx_val) else 20.0
        di_p = adx.adx_pos().iloc[-1]
        di_m = adx.adx_neg().iloc[-1]
        features["di_plus"] = float(di_p) if not pd.isna(di_p) else 15.0
        features["di_minus"] = float(di_m) if not pd.isna(di_m) else 15.0
    else:
        features["adx_14"] = 20.0
        features["di_plus"] = 15.0
        features["di_minus"] = 15.0

    if len(close) >= 26:
        macd = ta_lib.trend.MACD(close)
        mh = macd.macd_diff().iloc[-1]
        features["macd_hist"] = float(mh) if not pd.isna(mh) else 0.0
    else:
        features["macd_hist"] = 0.0

    if len(close) >= 20:
        bb = ta_lib.volatility.BollingerBands(close, window=20, window_dev=2)
        bb_u = float(bb.bollinger_hband().iloc[-1])
        bb_l = float(bb.bollinger_lband().iloc[-1])
        bb_w = (bb_u - bb_l) / spot
        features["bb_width_20d"] = bb_w
        features["bb_position"] = (spot - bb_l) / (bb_u - bb_l) if (bb_u - bb_l) > 0 else 0.5
    else:
        features["bb_width_20d"] = 0.05
        features["bb_position"] = 0.5

    # Calendar
    features["day_of_week"] = trade_date.weekday()
    features["day_of_month"] = trade_date.day
    features["month"] = trade_date.month
    features["days_to_expiry"] = 10

    # Drawdown
    if len(close) >= 20:
        rm = close.rolling(20).max()
        features["max_drawdown_20d_sofar"] = float((close / rm - 1).iloc[-1])
    else:
        features["max_drawdown_20d_sofar"] = 0

    if len(close) >= 10:
        rm10 = close.rolling(10).max()
        features["max_drawdown_10d_sofar"] = float((close / rm10 - 1).iloc[-1])
    else:
        features["max_drawdown_10d_sofar"] = 0

    # Option placeholders (no historical option chain)
    features["iv_skew"] = 0.03
    features["put_call_ratio"] = 1.0
    features["max_oi_put_strike"] = spot * 0.97
    features["max_oi_call_strike"] = spot * 1.03
    features["fii_net_5d"] = 0
    features["dii_net_5d"] = 0
    features["deep_otm_oi_ratio"] = 0
    features["iv_skew_steepness"] = 0
    features["atm_iv"] = vix / 100
    features["atm_iv_percentile_252d"] = 50

    # Clean NaN/inf
    for k, v in features.items():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            features[k] = 0.0

    return features


async def run_backfill(db_session) -> dict:
    """
    Run full historical backfill from Jan 2026 to today.
    Returns summary stats.
    """
    import yfinance as yf

    logger.info("=== Starting historical backfill ===")

    # Download historical data (need extra history for feature computation)
    start_download = date(2025, 6, 1)  # 6 months before backfill start for features
    end_download = date.today()

    logger.info(f"Downloading Nifty data from {start_download} to {end_download}")
    nifty = yf.download("^NSEI", start=start_download.isoformat(),
                         end=end_download.isoformat(), progress=False)

    if nifty.empty:
        return {"error": "Failed to download Nifty data from yfinance"}

    # Flatten multi-level columns if present
    if isinstance(nifty.columns, pd.MultiIndex):
        nifty.columns = nifty.columns.get_level_values(0)

    logger.info(f"Downloaded {len(nifty)} days of Nifty data")

    # Download VIX
    vix_df = yf.download("^INDIAVIX", start=start_download.isoformat(),
                          end=end_download.isoformat(), progress=False)
    if isinstance(vix_df.columns, pd.MultiIndex):
        vix_df.columns = vix_df.columns.get_level_values(0)

    vix_series = vix_df["Close"] if not vix_df.empty else pd.Series(dtype=float)

    # Align VIX with Nifty dates
    vix_aligned = vix_series.reindex(nifty.index).ffill().fillna(14.0)

    # Load model
    model_runner = ModelRunner(DOWNSIDE_MODEL_PATH, SCALER_PATH, FEATURE_NAMES_PATH)
    if model_runner.model is None:
        return {"error": "Model not loaded"}

    # Filter to trading days from BACKFILL_START onwards
    nifty_dates = nifty.index.date
    backfill_mask = nifty_dates >= BACKFILL_START
    backfill_indices = [i for i, m in enumerate(backfill_mask) if m]

    logger.info(f"Backfilling {len(backfill_indices)} trading days from {BACKFILL_START}")

    # Track per-version state
    version_state = {}
    for v in ACTIVE_VERSIONS:
        version_state[v] = {
            "capital": INITIAL_CAPITAL,
            "cumulative_pnl": 0.0,
            "open_trades": [],  # list of trade dicts
        }

    stats = {
        "days_processed": 0,
        "predictions_created": 0,
        "trades_opened": 0,
        "trades_closed": 0,
    }

    for idx in backfill_indices:
        trade_date = nifty_dates[idx]
        spot = float(nifty["Close"].iloc[idx])
        vix = float(vix_aligned.iloc[idx])

        if pd.isna(spot) or spot <= 0:
            continue

        # Build features
        features = _build_features_for_date(idx, nifty, vix_aligned, trade_date)

        # Run model
        prediction_value = model_runner.predict(features)

        # Store daily features
        daily_feature = DailyFeature(
            date=trade_date,
            features=features,
            vix=vix,
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

                    # Update trade in DB (find by trade_id)
                    from sqlalchemy import select, update
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
