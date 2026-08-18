"""
Configuration for all model versions (v5.4.2, v5.4.3, v5.4.4).
Defines version-specific parameters for signal mapping, exits, and sizing.
"""

import os
from datetime import date
from dotenv import load_dotenv

load_dotenv()

# Real-pricing cutover: the first date trades were priced on REAL Dhan quotes
# (sell@bid / buy@ask) instead of Black-Scholes. The dashboard tracks ACTUAL live
# profit from this date, and the BS forward-test recompute must NEVER delete or
# rebuild trades on/after it (see main.py recalculate_forward_test). First src=real
# trade was 2026-06-19; 2026-06-18 and earlier opened on BS (pre-deploy).
REAL_PRICING_START = date(2026, 6, 19)

# ============================================================
# Environment
# ============================================================
# Supabase PostgreSQL (free tier) — set DATABASE_URL in Render env vars
# Format: postgresql+asyncpg://user:pass@host.pooler.supabase.com:5432/postgres
_raw_db_url = os.environ.get("DATABASE_URL", "")

# Render sets postgres:// but SQLAlchemy needs postgresql+asyncpg://
if _raw_db_url.startswith("postgres://"):
    _raw_db_url = _raw_db_url.replace("postgres://", "postgresql+asyncpg://", 1)
elif _raw_db_url.startswith("postgresql://"):
    _raw_db_url = _raw_db_url.replace("postgresql://", "postgresql+asyncpg://", 1)
elif _raw_db_url and not _raw_db_url.startswith("postgresql+asyncpg://"):
    _raw_db_url = "postgresql+asyncpg://" + _raw_db_url

# Strip ALL query params from URL — asyncpg doesn't accept URL query params
# like sslmode, channel_binding, etc. SSL is handled via connect_args in database.py.
if _raw_db_url and "?" in _raw_db_url:
    _raw_db_url = _raw_db_url.split("?")[0]

DATABASE_URL = _raw_db_url or "postgresql+asyncpg://localhost/nifty_paper"
DHAN_ACCESS_TOKEN = os.environ.get("DHAN_ACCESS_TOKEN", "")
DHAN_CLIENT_ID = os.environ.get("DHAN_CLIENT_ID", "")
DHAN_API_BASE = "https://api.dhan.co/v2"

# Email notifications via Resend (free tier: 100 emails/day)
RESEND_API_KEY = os.environ.get("RESEND_API_KEY", "")
NOTIFY_EMAIL = os.environ.get("NOTIFY_EMAIL", "anmol@turings.xyz")

# Dhan security IDs
DHAN_NIFTY_SECURITY_ID = "13"
DHAN_VIX_SECURITY_ID = "21"  # Confirmed from Dhan scrip master (was 26, which was wrong)
DHAN_RATE_LIMIT_DELAY = 0.25

# Price trades from the real Dhan option chain (sell@bid / buy@ask) instead of
# synthetic Black-Scholes, falling back to BS (flagged) when the chain has no
# usable quote. Toggle off via env to revert to pure BS without a redeploy.
USE_REAL_OPTION_PRICING = os.environ.get("USE_REAL_OPTION_PRICING", "true").lower() == "true"

# ============================================================
# Shared Trade Parameters
# ============================================================
INITIAL_CAPITAL = 2_500_000
# Broker-verified on Kite 2026-07-30 (order screen minimum qty). Was wrongly 25.
# NOTE: exit P&L = (credit - value) * num_lots * NIFTY_LOT_SIZE, so this constant is
# applied at EXIT time. Never change it while trades are open — their num_lots was
# sized under the old value and they would be mis-valued.
NIFTY_LOT_SIZE = 65
RISK_FREE_RATE = 0.07
DTE_TARGET = 10
DTE_MIN_ENTRY = 5
DTE_MAX_ENTRY = 18
SLIPPAGE_PCT = 0.02
BROKERAGE_PER_ORDER = 20

# Strike selection (OTM percentages)
BULL_OTM_SELL = 0.03
BULL_OTM_BUY = 0.055
IC_PUT_OTM_SELL = 0.03
IC_PUT_OTM_BUY = 0.055
IC_CALL_OTM_BUY = 0.055

# Margin per lot — measured on Kite 2026-07-30 at spot 24,265 with 600-pt wings.
# Verified formula (within 1.2% on 4 baskets):
#     margin = max_loss + 0.02 * spot * qty * n_short_legs
# The exposure term (~Rs31.5k per short contract) is ~62% of the total and does NOT
# shrink with narrower wings. These are SNAPSHOTS: the exposure part scales with spot,
# so they drift as Nifty moves (~+4% at 26,000). Recompute if spot moves a lot.
MARGIN_PER_LOT_BULL = 69_501     # 1 short leg  (measured: put spread only)
MARGIN_PER_LOT_IC = 100_888      # 2 short legs (measured: 600-wide iron condor)
MARGIN_PER_LOT_BEAR = 18_750     # UNVERIFIED — bear debit spreads never measured on Kite

# Exposure-margin rate in the verified formula above (2% of notional per short leg).
EXPOSURE_MARGIN_PCT = 0.02

# The lot size that applied BEFORE the 2026-07-30 correction. Trades opened then were
# sized against 25, so anything that rebuilds or replays that era (the BS forward-test
# recompute, intraday_replay) must keep using this, NOT the live NIFTY_LOT_SIZE.
HISTORICAL_LOT_SIZE = 25

# --- Position sizing mode -------------------------------------------------
# COMPOUND_SIZING: size off CURRENT equity instead of the static INITIAL_CAPITAL, so
# profits increase position size. Backtested on the real-priced 2.5y (v5.4.4):
# fixed 71.17% -> compounded 97-100%.
# MAX_MARGIN_UTILISATION is what makes it safe. Uncapped compounding peaks at 126.8%
# of equity and goes margin-short on 5 days (forced liquidation). Capping TOTAL real
# margin keeps essentially all the upside:
#     cap 95% -> 100.29% ret, peak 99.4%, survives only a  0.6% margin hike
#     cap 90% ->  99.81% ret, peak 94.0%, survives only a  6.3% margin hike
#     cap 85% ->  98.70% ret, peak 89.0%, survives a      12.4% margin hike   <-- chosen
#     cap 80% ->  97.22% ret, peak 82.3%, survives a      21.5% margin hike
# 85% keeps ~all the return while leaving room for an exchange margin hike mid-crisis.
COMPOUND_SIZING = True
MAX_MARGIN_UTILISATION = 0.85

# SIZING_CAPITAL_OVERRIDE: the REAL broker balance to size against.
#
# Compounding sizes off _get_current_capital(), which sums realized P&L from the
# trades table -- and that table has been accumulating PAPER trades since Mar 2024.
# On 2026-08-18 that was Rs12.47L of simulated profit on top of the Rs25L start, so
# the model sized off Rs37.5L and emailed 21 lots against a real account of Rs25.12L
# -- 1.49x the money that actually exists. The gap widens as paper equity compounds.
#
# Set this to the real balance and both the lot count AND the 85% margin cap size
# off it. Set to None to fall back to the paper equity curve (backtests/paper only).
# Bump it as the real account actually grows.
SIZING_CAPITAL_OVERRIDE = 2_512_000

# Classification thresholds (shared by all versions)
DRAWDOWN_BULL_FULL = -0.038     # > -3.8%: bull full
DRAWDOWN_BULL_HALF = -0.050     # -3.8% to -5.0%: bull half
DRAWDOWN_IRON_CONDOR = -0.065   # -5.0% to -6.5%: iron condor
DRAWDOWN_NO_TRADE = -0.065      # < -6.5%: no trade

# VIX filters
VIX_MIN_ENTRY = 8
VIX_MAX_ENTRY = 25

# Credit/DTE quality floor for credit trades (skip nano-credit Wednesday entries
# that have 50:1-against risk:reward at entry).
MIN_TOTAL_CREDIT = 2000     # ₹ — skip credit trades with total credit below this
MIN_ENTRY_DTE = 2           # skip credit trades opened with ≤1 day to expiry

# ============================================================
# Version Profiles
# ============================================================

VERSION_CONFIGS = {
    "v5.4.2": {
        "label": "Sharp Threshold",
        "color": "#4A90D9",  # blue

        # Sizing
        # Calibration constants, NOT a literal % of capital: after dividing by the
        # (now real) MARGIN_PER_LOT and rounding to whole 65-unit lots these
        # reproduce the same position size as the old 0.20 / lot-25 / margin-13750
        # setup. Retuned 2026-07-30 with NIFTY_LOT_SIZE 25->65.
        "POSITION_SIZE_PCT": 0.3972,
        "IC_POSITION_SIZE_PCT": 0.5766,
        "MAX_CONCURRENT_POSITIONS": 3,
        "IC_MAX_CONCURRENT": 3,
        "MIN_ENTRY_GAP_DAYS": 2,

        # Exits
        "PROFIT_TARGET_EARLY": 0.50,
        "PROFIT_TARGET_MID": 0.65,
        "PROFIT_TARGET_LATE": 0.80,
        "STOP_LOSS_MULTIPLIER": 3.0,
        "STOP_LOSS_CONFIRM_DAYS": 2,
        "IC_STOP_LOSS_MULTIPLIER": 3.0,
        "IC_STOP_LOSS_CONFIRM_DAYS": 2,
        "TRAILING_STOP_ACTIVATE": 0.40,
        "TRAILING_STOP_LEVEL": 0.10,
        "MIN_DTE_EXIT": 1,
        "VIX_SPIKE_EXIT": 25,

        # Signal mapping type
        "SIGNAL_MAPPING": "sharp",  # sharp thresholds

        # Advanced features
        "IC_CALL_OTM_SELL": 0.04,
        "IC_CALL_OTM_BUY": 0.065,

        # --- The bear side, and the bull_half->condor swap (added 2026-08-12) ---
        # Both default OFF so this deploys inert; flip to True to go live.
        #
        # BEAR_CALL: days the model scores below -6.5% are currently sat out
        # entirely. Selling a call spread on them instead adds ~+9.6pp of annual
        # return with drawdown flat. Sized, gapped and margined exactly like a
        # bull_put because it is the same one-sided credit structure.
        # Strikes are 3.0%/5.5% — the IC defaults of 4.0%/6.5% are too thin to
        # clear the minimum-credit gate and produced almost no trades in test.
        "BEAR_CALL_ENABLED": True,
        "BEAR_CALL_OTM_SELL": 0.030,
        "BEAR_CALL_OTM_BUY": 0.055,
        "BEAR_CALL_SIZE_PCT": 0.3972,      # matches POSITION_SIZE_PCT
        "BEAR_CALL_MAX_CONCURRENT": 3,     # matches MAX_CONCURRENT_POSITIONS
        #
        # BULL_HALF_AS_IC: trade the -3.8%..-5.0% band as an iron condor rather
        # than a put spread identical to bull_full. ~+4pp annual return.
        # NOTE: condors need ~45% more margin per lot, so this pushes 95th-pct
        # margin utilisation from 76% to 83% (cap still never breached).
        "BULL_HALF_AS_IC": True,

        "OI_WALL_ENABLED": True,
        "OI_WALL_MIN_RATIO": 1.5,
        "VIX_HARVEST_ENABLED": True,
        "VIX_HARVEST_TRIGGER": 23,
        "VIX_HARVEST_REVERT_DROP": 2,
        "EVENT_CRUSH_ENABLED": True,
        "EVENT_CRUSH_IV_INFLATE": 0.10,
        "SCALE_INTO_WINNERS_ENABLED": True,
        "SCALE_PROFIT_TRIGGER": 0.30,
        "POST_EVENT_FOLLOWUP_ENABLED": True,
        "POST_EVENT_IV_THRESHOLD": 0.03,
    },

    "v5.4.3": {
        "label": "Sharp Threshold (5-min exits)",
        "color": "#E8833A",  # orange

        # Clone of v5.4.2 config, but all exit checks run every 5 min (no hybrid)
        # Sizing
        # Calibration constants, NOT a literal % of capital: after dividing by the
        # (now real) MARGIN_PER_LOT and rounding to whole 65-unit lots these
        # reproduce the same position size as the old 0.20 / lot-25 / margin-13750
        # setup. Retuned 2026-07-30 with NIFTY_LOT_SIZE 25->65.
        "POSITION_SIZE_PCT": 0.3972,
        "IC_POSITION_SIZE_PCT": 0.5766,
        "MAX_CONCURRENT_POSITIONS": 3,
        "IC_MAX_CONCURRENT": 3,
        # Gap relaxed to 1 (vs 2 on v5.4.2/4) — Mar–May 2026 analysis showed
        # 17/42 trading days were skipped purely by the 2-day gap, with skipped-
        # day conditions statistically identical to taken-day ones (~84% win
        # rate would apply). 5-min exits mean v5.4.3 can react fast if a
        # crash starts mid-week, so it's the safest version to test this on.
        "MIN_ENTRY_GAP_DAYS": 1,

        # Exits (same as v5.4.2 except tighter trail — with 5-min polling we can
        # lock in more of peak P&L without whipsaw risk).
        "PROFIT_TARGET_EARLY": 0.50,
        "PROFIT_TARGET_MID": 0.65,
        "PROFIT_TARGET_LATE": 0.80,
        "STOP_LOSS_MULTIPLIER": 3.0,
        "STOP_LOSS_CONFIRM_DAYS": 2,
        "IC_STOP_LOSS_MULTIPLIER": 3.0,
        "IC_STOP_LOSS_CONFIRM_DAYS": 2,
        "TRAILING_STOP_ACTIVATE": 0.50,
        "TRAILING_STOP_LEVEL": 0.05,
        "MIN_DTE_EXIT": 1,
        "VIX_SPIKE_EXIT": 25,

        # Signal mapping type (same as v5.4.2)
        "SIGNAL_MAPPING": "sharp",

        # Advanced features (same as v5.4.2)
        "IC_CALL_OTM_SELL": 0.04,
        "IC_CALL_OTM_BUY": 0.065,
        "OI_WALL_ENABLED": True,
        "OI_WALL_MIN_RATIO": 1.5,
        "VIX_HARVEST_ENABLED": True,
        "VIX_HARVEST_TRIGGER": 23,
        "VIX_HARVEST_REVERT_DROP": 2,
        "EVENT_CRUSH_ENABLED": True,
        "EVENT_CRUSH_IV_INFLATE": 0.10,
        "SCALE_INTO_WINNERS_ENABLED": True,
        "SCALE_PROFIT_TRIGGER": 0.30,
        "POST_EVENT_FOLLOWUP_ENABLED": True,
        "POST_EVENT_IV_THRESHOLD": 0.03,
    },

    "v5.4.4": {
        "label": "Balanced",
        "color": "#50C878",  # green

        # Sizing: same as v5.4.2 (take every trade)
        # Calibration constants, NOT a literal % of capital: after dividing by the
        # (now real) MARGIN_PER_LOT and rounding to whole 65-unit lots these
        # reproduce the same position size as the old 0.20 / lot-25 / margin-13750
        # setup. Retuned 2026-07-30 with NIFTY_LOT_SIZE 25->65.
        "POSITION_SIZE_PCT": 0.3972,
        "IC_POSITION_SIZE_PCT": 0.5766,
        "MAX_CONCURRENT_POSITIONS": 3,
        "IC_MAX_CONCURRENT": 3,
        "MIN_ENTRY_GAP_DAYS": 2,

        # Exits: tighter risk than v5.4.2/3, but with two guards:
        #   - 2-day confirm (was 1) so a single ugly close doesn't lock in the bottom
        #   - STOP_LOSS_ARMING_PCT: until the trade has earned ARMING_PCT in profit,
        #     fall back to the looser v5.4.2 stop (3.0×). The tight 2.5× stop is
        #     meant to protect *gained* profit, not punish an unlucky open.
        "PROFIT_TARGET_EARLY": 0.45,
        "PROFIT_TARGET_MID": 0.60,
        "PROFIT_TARGET_LATE": 0.75,
        "STOP_LOSS_MULTIPLIER": 2.5,
        "STOP_LOSS_CONFIRM_DAYS": 2,
        "IC_STOP_LOSS_MULTIPLIER": 2.5,
        "IC_STOP_LOSS_CONFIRM_DAYS": 2,
        "STOP_LOSS_ARMING_PCT": 0.10,        # trade must reach +10% before tight stop arms
        "STOP_LOSS_ARMING_MULT": 3.0,         # pre-arming multiplier (matches v5.4.2)
        "TRAILING_STOP_ACTIVATE": 0.45,
        "TRAILING_STOP_LEVEL": 0.15,
        "MIN_DTE_EXIT": 1,

        # Signal mapping type — narrower transition (HW 0.25→0.10) so borderline
        # predictions like -3.86% size as bull_half (not bull_full), matching the
        # sharp mapping's behavior on the boundary.
        "SIGNAL_MAPPING": "graduated_gentle",
        "GRADUATED_FLOOR": 0.80,
        "GRADUATED_HW": 0.10,

        # Advanced features: same entry params as v5.4.2
        "IC_CALL_OTM_SELL": 0.04,
        "IC_CALL_OTM_BUY": 0.065,
        "OI_WALL_ENABLED": True,
        "OI_WALL_MIN_RATIO": 1.5,
        "VIX_HARVEST_ENABLED": True,
        "VIX_HARVEST_TRIGGER": 23,
        "VIX_HARVEST_REVERT_DROP": 2,
        "EVENT_CRUSH_ENABLED": True,
        "EVENT_CRUSH_IV_INFLATE": 0.10,
        "SCALE_INTO_WINNERS_ENABLED": True,
        "SCALE_PROFIT_TRIGGER": 0.30,
        "POST_EVENT_FOLLOWUP_ENABLED": True,
        "POST_EVENT_IV_THRESHOLD": 0.03,
    },

    "v6.2": {
        "label": "Bear Alpha",
        "color": "#E5534B",  # red

        # Sizing: same as v5.4.4
        # Calibration constants, NOT a literal % of capital: after dividing by the
        # (now real) MARGIN_PER_LOT and rounding to whole 65-unit lots these
        # reproduce the same position size as the old 0.20 / lot-25 / margin-13750
        # setup. Retuned 2026-07-30 with NIFTY_LOT_SIZE 25->65.
        "POSITION_SIZE_PCT": 0.3972,
        "IC_POSITION_SIZE_PCT": 0.5766,
        "MAX_CONCURRENT_POSITIONS": 3,
        "IC_MAX_CONCURRENT": 3,
        "MIN_ENTRY_GAP_DAYS": 2,

        # Exits: same as v5.4.4
        "PROFIT_TARGET_EARLY": 0.45,
        "PROFIT_TARGET_MID": 0.60,
        "PROFIT_TARGET_LATE": 0.75,
        "STOP_LOSS_MULTIPLIER": 2.5,
        "STOP_LOSS_CONFIRM_DAYS": 1,
        "IC_STOP_LOSS_MULTIPLIER": 2.5,
        "IC_STOP_LOSS_CONFIRM_DAYS": 1,
        "TRAILING_STOP_ACTIVATE": 0.45,
        "TRAILING_STOP_LEVEL": 0.15,
        "MIN_DTE_EXIT": 1,

        # Signal mapping: directional_bear (graduated_gentle above IC, bear debit below)
        "SIGNAL_MAPPING": "directional_bear",
        "GRADUATED_FLOOR": 0.80,
        "GRADUATED_HW": 0.25,

        # Advanced features: same as v5.4.4
        "IC_CALL_OTM_SELL": 0.04,
        "IC_CALL_OTM_BUY": 0.065,
        "OI_WALL_ENABLED": True,
        "OI_WALL_MIN_RATIO": 1.5,
        "VIX_HARVEST_ENABLED": True,
        "VIX_HARVEST_TRIGGER": 23,
        "VIX_HARVEST_REVERT_DROP": 2,
        "EVENT_CRUSH_ENABLED": True,
        "EVENT_CRUSH_IV_INFLATE": 0.10,
        "SCALE_INTO_WINNERS_ENABLED": True,
        "SCALE_PROFIT_TRIGGER": 0.30,
        "POST_EVENT_FOLLOWUP_ENABLED": True,
        "POST_EVENT_IV_THRESHOLD": 0.03,

        # Bear debit spread parameters
        "BEAR_DEBIT_ENABLED": True,
        "BEAR_DEBIT_THRESHOLD": -0.065,
        "BEAR_STRONG_THRESHOLD": -0.090,
        "BEAR_MODERATE_THRESHOLD": -0.065,
        "BEAR_PUT_BUY_OTM": 0.01,
        "BEAR_PUT_SELL_OTM": 0.04,
        "BEAR_SIZE_MULT_T1": 0.50,
        "BEAR_SIZE_MULT_T2": 0.25,
        "BEAR_DEBIT_PROFIT_TARGET": 2.0,
        "BEAR_DEBIT_STOP_LOSS": 0.70,
        "BEAR_DEBIT_TRAILING_ACTIVATE": 1.0,
        "BEAR_DEBIT_TRAILING_LEVEL": 0.30,
        "BEAR_DEBIT_MAX_CONCURRENT": 2,
        "BEAR_DEBIT_MAX_HOLD_DAYS": 8,
        "BEAR_DEBIT_BLOCKS_CREDIT": False,
    },
}

# All versions that are actively paper traded
ACTIVE_VERSIONS = ["v5.4.2", "v5.4.3", "v5.4.4"]

# Model file paths
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ml")
DOWNSIDE_MODEL_PATH = os.path.join(MODEL_DIR, "downside_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
FEATURE_NAMES_PATH = os.path.join(MODEL_DIR, "feature_names.pkl")
