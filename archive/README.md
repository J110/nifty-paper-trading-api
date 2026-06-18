# archive/ — reference only (NOT deployed)

Nothing in this folder is imported by the live app (`main.py` / `scheduler/jobs.py`).
It is kept for reference. See `../ARCHITECTURE.md` for what actually runs in production.

- `scripts/bhavcopy_backtest.py` — the **real-priced** backtest (NSE Bhavcopy daily prices)
  that produced the **v5.4.4** decision. The current/best analysis tool here. Re-run with
  more data; paths are hardcoded inside.
- `scripts/{backtest_no_month,optimize_thresholds,replay_forward_test,retrain_no_month,simulate_returns}.py`
  — older model/threshold experiments, priced on **Black-Scholes** (do not mistake for live).
- `db_migrations/` — already-applied one-off schema migrations.
- `run_backfill_local.py` — local backtest backfill helper.

Before moving anything back into the deployed tree, re-derive the import graph from the
two entry points (see ARCHITECTURE.md → "Verifying what is deployed").
