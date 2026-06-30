# ARCHITECTURE — READ THIS FIRST

This is a **live Nifty options paper-trading system** deployed on Render. This file
is the single source of truth for *what runs in production*, *what is analysis/debug*,
and *what is archived reference only*. Read it before changing anything.

---

## ⚠️ CRITICAL FACTS — do not revert, do not confuse

1. **Live trades are priced on REAL Dhan option-chain quotes, NOT Black-Scholes.**
   - Entry credit = sell legs @ bid, buy legs @ ask. Exit/MTM = real cost-to-close
     (buy back shorts @ ask, sell longs @ bid). Both via `core/real_option_pricer.py`,
     wired into `core/trade_manager.py` (`open_trade`, `check_exits`).
   - Black-Scholes (`core/option_pricer.py`) is **only a flagged FALLBACK** for when a
     leg has no quote. Controlled by `USE_REAL_OPTION_PRICING` (env, default **ON**).
   - **DO NOT** claim or assume the live system prices on BS. It does not.

2. **The analysis/debug tools DO use Black-Scholes or coarse daily data — they are NOT
   how live trades are priced.** This is the #1 source of confusion. Specifically:
   - `/api/debug/recalculate-forward-test` (the "+X% returns" recompute) → **BS, daily close.**
     **Confined to the BS era `[2026-02-13, config.REAL_PRICING_START)`** — it deletes and
     rebuilds only pre-cutover trades and will NOT touch the real-priced live trades on/after
     `REAL_PRICING_START` (2026-06-19). It used to delete the whole table from Feb 13, which
     clobbered the live record; that is now fixed. Running it is safe for the live data.
   - `core/intraday_replay.py` (`/api/debug/intraday-replay`) → **BS, 15-min snapshots.**
   - `archive/scripts/bhavcopy_backtest.py` → **real NSE Bhavcopy daily prices** (this one is real, but daily/no-spread).
   - When you report numbers, state which engine produced them. Do not equate
     analysis-tool output with live behaviour.
   - **Dashboard data modes pivot on `config.REAL_PRICING_START` (2026-06-19):**
     `forwardtest` = the real-priced live era (`entry_date >= REAL_PRICING_START`) — this is
     the **ACTUAL tracked profit**; `backtest` = the older BS period before it; `combined` = all.
     Read `forwardtest` for real performance (`api/trades.py`, `api/charts.py`).

3. **NSE Nifty expiry is TUESDAY** (changed from Thursday on 2025-09-01).
   `core/option_pricer.py::get_next_weekly_expiry` is date-aware (Thursday before the
   cutover, Tuesday after) with holiday roll-back. **DO NOT revert to Thursday.**

4. **The entry quality gate is intentional:** skip credit trades with
   `total_credit < MIN_TOTAL_CREDIT` (₹2000) or `DTE < MIN_ENTRY_DTE` (2). Removing it
   re-introduces the worthless 1-DTE entries. Keep it (live `open_trade` + the recompute).

5. **Version to trade = v5.4.4.** On a *real-priced* backtest (Bhavcopy, Jan–Jun 2026)
   v5.4.4 ≈ v5.4.2 (~+23%, ~−9% max drawdown, ~88% win, robust). **v5.4.3 looks best on
   the synthetic numbers but is a fragile intraday-execution bet (−41% drawdown on a
   fair/conservative basis).** Do not pick v5.4.3 off BS numbers.

6. **NIFTY lot size:** `config.py` has `NIFTY_LOT_SIZE = 25`, but the real NSE lot is
   larger (~75). The live paper sizing uses 25; **real-money order sizing must read the
   real lot from the broker.** Don't trust config's 25 for live size/margin.

---

## What runs LIVE (the trading loop)

Deployed app = `main.py` (FastAPI) + `scheduler/jobs.py` (APScheduler crons). Daily flow (IST):

| Time | Job (`scheduler/jobs.py`) | Does |
|---|---|---|
| 09:05 | `run_data_update` | refresh `data/merged_daily.parquet` (yfinance) |
| 09:20 | `generate_daily_predictions` | features → model → signal → **open trades (real entry pricing)** |
| 09:20–15:30 /5min | `check_all_exits` | **exit/stop/trailing checks (real pricing)** |
| 15:32 | daily real-price marks | record each open trade's real cost-to-close |
| 06:00 / 18:00 | `renew_dhan_token` | auto-renew Dhan token |
| EOD | daily P&L | per-version equity |

**Live trading files** (all under the deployed import graph — see "Verifying" below):
`scheduler/jobs.py`, `core/trade_manager.py`, `core/dhan_client.py`,
`core/real_option_pricer.py`, `core/option_pricer.py` (BS fallback), `core/signal_mapper.py`,
`core/model_runner.py`, `core/feature_engine.py`, `shared/feature_compute.py`,
`core/data_updater.py`, `core/price_tracker.py`, `core/market_holidays.py`,
`core/timezone.py`, `core/prediction_reasoning.py`, `core/email_notifier.py`,
`config.py`, `db/database.py`, `db/models.py`, and the dashboard routers
`api/{signals,trades,charts,returns}.py`.

---

## Deployed but ANALYSIS / DEBUG (not the trading loop)

These ship with the app but are diagnostics/backtests, **not live trading**:
- `core/intraday_replay.py` — intraday exit-replay harness (**BS**, 15-min snapshots).
- `backfill.py` — historical backtest/backfill (imports `shared/{utils_backtest,
  config_profiles,model_signals,option_pricing_engine}`); run via `/api/debug` only.
- `main.py` debug endpoints: `recalculate-forward-test` (BS daily recompute),
  `intraday-replay`, `pricing-comparison`, `test-real-pricing`, `daily-marks`,
  `snapshot-coverage`, `db-counts`, etc.

---

## Reference (archived in `archive/`, NOT imported by the app)

Verified not in the deployed import graph; kept for reference only.
- `archive/scripts/bhavcopy_backtest.py` — **the real-priced backtest** (NSE F&O Bhavcopy,
  zips in `~/Downloads/2026-*.zip`) that produced the v5.4.4 decision. Re-run with more
  data: `python3 archive/scripts/bhavcopy_backtest.py` (paths hardcoded inside).
- `archive/scripts/{backtest_no_month,optimize_thresholds,replay_forward_test,
  retrain_no_month,simulate_returns}.py` — old model/threshold experiments (BS).
- `archive/db_migrations/*` — already-applied one-off schema migrations.
- `archive/run_backfill_local.py` — local backtest backfill.

---

## Strategy & theory (current understanding)

- **Strategy:** daily ML model predicts Nifty 30-day drawdown → mapped to a signal →
  sell OTM **credit spreads** (bull put ~3%/5.5% OTM, or iron condor + a call spread).
  Defined-risk; high win rate (~87%) with occasional large losers; the edge is premium
  decay, so **time-to-expiry and real fills matter.**
- **Versions** = parameter sets in `config.py::VERSION_CONFIGS`. `ACTIVE_VERSIONS =
  [v5.4.2, v5.4.3, v5.4.4]`. v5.4.2 sharp thresholds; v5.4.3 = clone with **5-min
  intraday exits + gap=1** (trades more; its edge depends entirely on intraday
  profit-taking); v5.4.4 balanced (tighter 2.5× stop + profit-arming).
- **Pricing reality:** live = real Dhan chain. BS over-values these OTM spreads
  (1.25× markup) — it overstates entry credit AND invents phantom stop-losses, so BS
  *under*-states real profitability. Real-priced backtest > BS for v5.4.2/v5.4.4.
- **Data sources:** Dhan API (live spot/VIX/option chain + token); yfinance →
  `merged_daily.parquet` (model features — has gaps); NSE F&O Bhavcopy (real historical
  option prices, daily OHLC+settle, in `~/Downloads/2026-*.zip`).
- **Known gaps to a true real replica:** transaction costs not modelled; fills assume
  the bid/ask touch (no slippage/partial-fill/market-impact — still paper, no real
  orders); the BS analysis tools (recompute, intraday-replay); `shared/feature_compute.py`
  monthly-expiry features still use last-Thursday (kept for **train/serve consistency** —
  do NOT naively flip to Tuesday without retraining the model).

---

## Operational

- **Hosting:** Render (free tier), auto-deploys from GitHub `J110/nifty-paper-trading-api`
  on push to `main`. `/health` exposes `deployed_commit`, token expiry, DB/scheduler health.
- **Push access:** the `J110` GitHub account owns the repo (the local `anmol-turings`
  account is pull-only). Push as J110.
- **Env vars (Render):** `DHAN_ACCESS_TOKEN`, `DHAN_CLIENT_ID`, `DATABASE_URL` (Postgres),
  `RESEND_API_KEY`, `NOTIFY_EMAIL`, `USE_REAL_OPTION_PRICING` (default on).
- **Dhan token:** auto-renews 06:00/18:00 IST → `system_config` table. A failed
  *data feed* is usually the Data API subscription, NOT the token (renewing won't fix it).

---

## Verifying what is "deployed" (don't guess — compute it)

The deployed set is everything reachable from `main.py` + `scheduler/jobs.py`. To
re-derive it (e.g. before archiving more code), walk the import graph from those two
entry points; anything not reachable (excluding package `__init__.py` markers) is safe
to treat as reference. As of this writing the deployed graph is **29 files**.

## Detailed history (agent memory)

Deeper rationale for the decisions above lives in the project memory:
`project_nifty_expiry_tuesday_fix_2026-06-18`, `project_real_option_pricing_2026-06-18`,
`project_realprice_backtest_2026-06-18`, `reference_dhan_token`,
`project_dhan_data_api_outage_2026-06-15`, `project_dhan_token_trade_guard`.
