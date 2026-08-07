"""
Email notifications for trade signals using Resend (free tier: 100 emails/day).
Sends beautifully formatted HTML emails when trades are ready to execute.
"""

import logging
import time
import httpx
from datetime import datetime
from core.timezone import now_ist
from config import VERSION_CONFIGS

logger = logging.getLogger(__name__)

# Resend API config (set via env var on Render)
import os
RESEND_API_KEY = os.environ.get("RESEND_API_KEY", "")
NOTIFY_EMAIL = os.environ.get("NOTIFY_EMAIL", "anmol@turings.xyz")
# MUST be a Resend-VERIFIED domain. The old default (onboarding@resend.dev — Resend's
# shared sandbox sender) is rejected with "Domain is not verified" under load: on
# 2026-08-06 three exit alerts fired within 4s at 15:35 and ALL failed, while a lone
# alert at 14:40 got through. Failures are silent (see _send_email — it returns False
# and the caller discards it), so nobody noticed the trades had closed.
# dreamvalley.app is verified on this Resend account; override via env if that changes.
FROM_EMAIL = os.environ.get(
    "RESEND_FROM_EMAIL", "Nifty Trading Bot <nifty@dreamvalley.app>"
)

# ---------------------------------------------------------------------------
# Alert deduplication: suppress repeated failure alerts for the same pipeline.
# Only one alert per pipeline per ALERT_COOLDOWN_SECONDS (default 1 hour).
# ---------------------------------------------------------------------------
ALERT_COOLDOWN_SECONDS = 3600  # 1 hour
_last_alert_ts: dict[str, float] = {}  # pipeline_name -> monotonic timestamp


def _signal_emoji(signal: str) -> str:
    """Map signal type to emoji."""
    return {
        "bull_full": "\U0001F7E2",      # 🟢
        "bull_half": "\U0001F7E1",      # 🟡
        "iron_condor": "\U0001F7E0",    # 🟠
        "bear_call": "\U0001F7E3",      # 🟣
        "no_trade": "\U0001F534",       # 🔴
    }.get(signal, "\u26AA")              # ⚪


def _format_currency(value: float) -> str:
    """Format as Indian currency."""
    if abs(value) >= 100_000:
        return f"\u20B9{value:,.0f}"
    return f"\u20B9{value:,.2f}"


def _build_trade_html(
    version: str,
    signal: dict,
    trade_result: dict,
    spot: float,
    vix: float,
    prediction: float,
) -> str:
    """Build a clean HTML email body for a trade alert."""
    cfg = VERSION_CONFIGS.get(version, {})
    version_label = cfg.get("label", version)
    version_color = cfg.get("color", "#4A90D9")
    signal_type = signal["signal"]
    trade_type = trade_result.get("trade_type", signal.get("trade_type", ""))
    strikes = trade_result.get("strikes", {})
    emoji = _signal_emoji(signal_type)

    # Build strikes display
    if trade_type == "bull_put":
        strikes_html = f"""
        <tr><td style="color:#8b949e;padding:4px 12px;">Sell Put</td>
            <td style="color:#e6edf3;padding:4px 12px;font-weight:600;">{strikes.get('sell_strike', 'N/A')}</td></tr>
        <tr><td style="color:#8b949e;padding:4px 12px;">Buy Put</td>
            <td style="color:#e6edf3;padding:4px 12px;font-weight:600;">{strikes.get('buy_strike', 'N/A')}</td></tr>
        """
    elif trade_type == "iron_condor":
        strikes_html = f"""
        <tr><td style="color:#8b949e;padding:4px 12px;">Sell Put</td>
            <td style="color:#e6edf3;padding:4px 12px;font-weight:600;">{strikes.get('sell_strike', 'N/A')}</td></tr>
        <tr><td style="color:#8b949e;padding:4px 12px;">Buy Put</td>
            <td style="color:#e6edf3;padding:4px 12px;font-weight:600;">{strikes.get('buy_strike', 'N/A')}</td></tr>
        <tr><td style="color:#8b949e;padding:4px 12px;">Sell Call</td>
            <td style="color:#e6edf3;padding:4px 12px;font-weight:600;">{strikes.get('ic_call_sell', 'N/A')}</td></tr>
        <tr><td style="color:#8b949e;padding:4px 12px;">Buy Call</td>
            <td style="color:#e6edf3;padding:4px 12px;font-weight:600;">{strikes.get('ic_call_buy', 'N/A')}</td></tr>
        """
    elif trade_type == "bear_call":
        strikes_html = f"""
        <tr><td style="color:#8b949e;padding:4px 12px;">Sell Call</td>
            <td style="color:#e6edf3;padding:4px 12px;font-weight:600;">{strikes.get('ic_call_sell', 'N/A')}</td></tr>
        <tr><td style="color:#8b949e;padding:4px 12px;">Buy Call</td>
            <td style="color:#e6edf3;padding:4px 12px;font-weight:600;">{strikes.get('ic_call_buy', 'N/A')}</td></tr>
        """
    else:
        strikes_html = ""

    # Per-leg expected fills (sell@bid / buy@ask) — lets you compare each leg against
    # the live Kite bid/ask at 9:20 before placing (the manual fill-validation check).
    legs = trade_result.get("pricing_legs") or []
    if legs and trade_result.get("pricing_source") in ("real", "real_ltp"):
        leg_rows = ""
        for lg in legs:
            act = str(lg.get("action", "")).upper()
            opt = str(lg.get("opt", "")).upper()
            stk = lg.get("strike")
            stk_str = f"{stk:,.0f}" if isinstance(stk, (int, float)) else str(stk)
            src = lg.get("px_source", "")
            src_label = {"bid": "bid", "ask": "ask", "ltp": "LTP · illiquid"}.get(src, src)
            act_color = "#3fb950" if lg.get("action") == "sell" else "#f85149"
            leg_rows += f"""
                <tr>
                    <td style="padding:4px 12px;"><span style="color:{act_color};font-weight:600;">{act}</span> <span style="color:#e6edf3;">{stk_str} {opt}</span></td>
                    <td style="color:#e6edf3;padding:4px 12px;font-weight:600;text-align:right;">{_format_currency(lg.get('price', 0))} <span style="color:#8b949e;font-weight:400;font-size:11px;">({src_label})</span></td>
                </tr>"""
        legs_html = f"""
        <div style="padding:0 24px 16px;">
            <h3 style="color:#8b949e;font-size:11px;text-transform:uppercase;letter-spacing:1px;margin:0 0 8px;">
                Expected Leg Fills (model)
            </h3>
            <table style="width:100%;border-collapse:collapse;background:#161b22;border-radius:8px;">
                {leg_rows}
                <tr>
                    <td style="border-top:1px solid #30363d;color:#8b949e;padding:6px 12px;">Net credit / unit</td>
                    <td style="border-top:1px solid #30363d;color:#3fb950;padding:6px 12px;font-weight:700;text-align:right;">{_format_currency(trade_result.get('credit', 0))}</td>
                </tr>
            </table>
            <p style="color:#6e7681;font-size:11px;margin:8px 12px 0;line-height:1.5;">
                Compare each leg to the live bid/ask on Kite at 9:20 before placing — you sell near the <b>bid</b>, buy near the <b>ask</b>. If your basket's net credit lands close to the model's, fills are tracking.
            </p>
        </div>"""
    else:
        legs_html = ""

    return f"""
    <div style="font-family:'Segoe UI',Roboto,Helvetica,Arial,sans-serif;max-width:520px;margin:0 auto;background:#0d1117;border:1px solid #30363d;border-radius:12px;overflow:hidden;">

        <!-- Header -->
        <div style="background:{version_color};padding:20px 24px;">
            <h1 style="margin:0;color:#fff;font-size:20px;font-weight:600;">
                {emoji} Trade Signal — {version_label}
            </h1>
            <p style="margin:4px 0 0;color:rgba(255,255,255,0.85);font-size:13px;">
                {now_ist().strftime('%d %b %Y, %I:%M %p IST')}
            </p>
        </div>

        <!-- Signal Badge -->
        <div style="padding:20px 24px 0;">
            <div style="display:inline-block;background:#161b22;border:1px solid #30363d;border-radius:8px;padding:10px 20px;">
                <span style="color:#8b949e;font-size:12px;text-transform:uppercase;letter-spacing:1px;">Signal</span>
                <br/>
                <span style="color:{version_color};font-size:22px;font-weight:700;text-transform:uppercase;">
                    {signal_type.replace('_', ' ')}
                </span>
            </div>
        </div>

        <!-- Market Context -->
        <div style="padding:16px 24px;">
            <h3 style="color:#8b949e;font-size:11px;text-transform:uppercase;letter-spacing:1px;margin:0 0 8px;">
                Market Snapshot
            </h3>
            <table style="width:100%;border-collapse:collapse;">
                <tr>
                    <td style="color:#8b949e;padding:4px 12px;">Nifty Spot</td>
                    <td style="color:#e6edf3;padding:4px 12px;font-weight:600;">{spot:,.1f}</td>
                </tr>
                <tr>
                    <td style="color:#8b949e;padding:4px 12px;">India VIX</td>
                    <td style="color:#e6edf3;padding:4px 12px;font-weight:600;">{vix:.2f}</td>
                </tr>
                <tr>
                    <td style="color:#8b949e;padding:4px 12px;">Predicted Drawdown</td>
                    <td style="color:{'#f85149' if prediction < -0.025 else '#ffa657' if prediction < -0.015 else '#3fb950'};padding:4px 12px;font-weight:600;">
                        {prediction*100:.2f}%
                    </td>
                </tr>
            </table>
        </div>

        <!-- Trade Details -->
        <div style="padding:0 24px 16px;">
            <h3 style="color:#8b949e;font-size:11px;text-transform:uppercase;letter-spacing:1px;margin:0 0 8px;">
                Trade Details
            </h3>
            <table style="width:100%;border-collapse:collapse;background:#161b22;border-radius:8px;">
                <tr>
                    <td style="color:#8b949e;padding:4px 12px;">Strategy</td>
                    <td style="color:#e6edf3;padding:4px 12px;font-weight:600;">{trade_type.replace('_', ' ').title()}</td>
                </tr>
                <tr>
                    <td style="color:#8b949e;padding:4px 12px;">Version</td>
                    <td style="color:#e6edf3;padding:4px 12px;font-weight:600;">{version} ({version_label})</td>
                </tr>
                {strikes_html}
                <tr>
                    <td style="color:#8b949e;padding:4px 12px;">Lots</td>
                    <td style="color:#e6edf3;padding:4px 12px;font-weight:600;">{trade_result.get('num_lots', 'N/A')}</td>
                </tr>
                <tr>
                    <td style="color:#8b949e;padding:4px 12px;">Credit (per unit)</td>
                    <td style="color:#3fb950;padding:4px 12px;font-weight:600;">{_format_currency(trade_result.get('credit', 0))}</td>
                </tr>
                <tr>
                    <td style="color:#8b949e;padding:4px 12px;">Total Credit</td>
                    <td style="color:#3fb950;padding:4px 12px;font-weight:600;">{_format_currency(trade_result.get('total_credit', 0))}</td>
                </tr>
                <tr>
                    <td style="color:#8b949e;padding:4px 12px;">Pricing</td>
                    <td style="color:{'#f85149' if trade_result.get('pricing_source') == 'bs_fallback' else '#d29922' if trade_result.get('pricing_source') == 'real_ltp' else '#3fb950'};padding:4px 12px;font-weight:600;">
                        {'⚠️ BS FALLBACK — ' + str(trade_result.get('pricing_reason')) if trade_result.get('pricing_source') == 'bs_fallback' else 'REAL · LTP/illiquid' if trade_result.get('pricing_source') == 'real_ltp' else 'REAL (Dhan chain)' if trade_result.get('pricing_source') == 'real' else 'BS (synthetic)'} &nbsp;(real {_format_currency(trade_result.get('real_credit') or 0)} vs BS {_format_currency(trade_result.get('bs_credit') or 0)})
                    </td>
                </tr>
                <tr>
                    <td style="color:#8b949e;padding:4px 12px;">Expiry</td>
                    <td style="color:#e6edf3;padding:4px 12px;font-weight:600;">{trade_result.get('expiry', 'N/A')}</td>
                </tr>
                <tr>
                    <td style="color:#8b949e;padding:4px 12px;">Size Multiplier</td>
                    <td style="color:#e6edf3;padding:4px 12px;font-weight:600;">{signal.get('size_mult', 1.0):.1%}</td>
                </tr>
                <tr>
                    <td style="color:#8b949e;padding:4px 12px;">Entry Mode</td>
                    <td style="color:#e6edf3;padding:4px 12px;font-weight:600;">{trade_result.get('entry_mode', 'normal').title()}</td>
                </tr>
            </table>
        </div>

        {legs_html}

        <!-- Footer -->
        <div style="padding:12px 24px 16px;border-top:1px solid #21262d;">
            <p style="color:#484f58;font-size:11px;margin:0;text-align:center;">
                Paper Trading Bot &bull; {version} &bull; Auto-generated alert
            </p>
        </div>
    </div>
    """


def _build_no_trade_html(
    prediction: float,
    spot: float,
    vix: float,
    version_signals: list[dict],
    blocked_versions: list[dict] = None,
) -> str:
    """Build HTML for when no trades were opened.

    blocked_versions: versions where signal was bullish but trade was blocked
    (entry gap, max concurrent, etc.). If non-empty, the email explains the
    block reason instead of saying "high risk".
    """
    blocked_versions = blocked_versions or []
    blocked_set = {bv["version"] for bv in blocked_versions}
    is_blocked = len(blocked_set) > 0

    rows = ""
    for vs in version_signals:
        cfg = VERSION_CONFIGS.get(vs["version"], {})
        signal = vs["signal"]
        if vs["version"] in blocked_set:
            # Signal was bullish but trade was blocked
            signal_display = f"{signal.replace('_', ' ').upper()} (blocked)"
            signal_color = "#ffa657"  # orange
        elif signal == "no_trade":
            signal_display = "NO TRADE"
            signal_color = "#f85149"  # red
        else:
            signal_display = signal.replace("_", " ").upper()
            signal_color = "#3fb950"  # green
        rows += f"""
        <tr>
            <td style="color:#e6edf3;padding:6px 12px;">{vs['version']} ({cfg.get('label', '')})</td>
            <td style="color:{signal_color};padding:6px 12px;font-weight:600;">{signal_display}</td>
        </tr>"""

    if is_blocked:
        header_color = "#ffa657"  # orange for blocked
        header_title = "\u26A0\uFE0F Trade Blocked"
        footer_text = (
            "Signal was bullish but trades blocked by entry gap or position limits. "
            "Existing positions may need to close first."
        )
    else:
        header_color = "#f85149"  # red for no_trade
        header_title = "\U0001F534 No Trade Today"
        footer_text = "Model predicts high risk \u2014 sitting out today."

    drawdown_color = '#f85149' if prediction < -0.05 else '#ffa657' if prediction < -0.025 else '#3fb950'

    return f"""
    <div style="font-family:'Segoe UI',Roboto,Helvetica,Arial,sans-serif;max-width:520px;margin:0 auto;background:#0d1117;border:1px solid #30363d;border-radius:12px;overflow:hidden;">
        <div style="background:{header_color};padding:20px 24px;">
            <h1 style="margin:0;color:#fff;font-size:20px;font-weight:600;">
                {header_title}
            </h1>
            <p style="margin:4px 0 0;color:rgba(255,255,255,0.85);font-size:13px;">
                {now_ist().strftime('%d %b %Y, %I:%M %p IST')}
            </p>
        </div>
        <div style="padding:20px 24px;">
            <table style="width:100%;border-collapse:collapse;">
                <tr><td style="color:#8b949e;padding:4px 12px;">Nifty</td>
                    <td style="color:#e6edf3;padding:4px 12px;font-weight:600;">{spot:,.1f}</td></tr>
                <tr><td style="color:#8b949e;padding:4px 12px;">VIX</td>
                    <td style="color:#e6edf3;padding:4px 12px;font-weight:600;">{vix:.2f}</td></tr>
                <tr><td style="color:#8b949e;padding:4px 12px;">Predicted Drawdown</td>
                    <td style="color:{drawdown_color};padding:4px 12px;font-weight:600;">{prediction*100:.2f}%</td></tr>
            </table>
            <h3 style="color:#8b949e;font-size:11px;text-transform:uppercase;letter-spacing:1px;margin:16px 0 8px;">
                All Versions
            </h3>
            <table style="width:100%;border-collapse:collapse;background:#161b22;border-radius:8px;">
                {rows}
            </table>
        </div>
        <div style="padding:12px 24px 16px;border-top:1px solid #21262d;">
            <p style="color:#484f58;font-size:11px;margin:0;text-align:center;">
                {footer_text}
            </p>
        </div>
    </div>
    """


def _build_exit_html(
    version: str,
    trade_id: str,
    exit_reason: str,
    trade_type: str,
    entry_spot: float,
    exit_spot: float,
    realized_pnl: float,
    pnl_pct: float,
    sell_strike: float,
    buy_strike: float,
) -> str:
    """Build HTML for a trade exit notification."""
    cfg = VERSION_CONFIGS.get(version, {})
    version_label = cfg.get("label", version)
    is_profit = realized_pnl > 0
    pnl_color = "#3fb950" if is_profit else "#f85149"
    header_color = "#3fb950" if is_profit else "#f85149"
    emoji = "\u2705" if is_profit else "\u274C"

    return f"""
    <div style="font-family:'Segoe UI',Roboto,Helvetica,Arial,sans-serif;max-width:520px;margin:0 auto;background:#0d1117;border:1px solid #30363d;border-radius:12px;overflow:hidden;">
        <div style="background:{header_color};padding:20px 24px;">
            <h1 style="margin:0;color:#fff;font-size:20px;font-weight:600;">
                {emoji} Trade Closed — {version_label}
            </h1>
            <p style="margin:4px 0 0;color:rgba(255,255,255,0.85);font-size:13px;">
                {now_ist().strftime('%d %b %Y, %I:%M %p IST')}
            </p>
        </div>
        <div style="padding:20px 24px;">
            <table style="width:100%;border-collapse:collapse;background:#161b22;border-radius:8px;">
                <tr><td style="color:#8b949e;padding:6px 12px;">Trade ID</td>
                    <td style="color:#e6edf3;padding:6px 12px;font-weight:600;font-size:12px;">{trade_id}</td></tr>
                <tr><td style="color:#8b949e;padding:6px 12px;">Strategy</td>
                    <td style="color:#e6edf3;padding:6px 12px;font-weight:600;">{trade_type.replace('_', ' ').title()}</td></tr>
                <tr><td style="color:#8b949e;padding:6px 12px;">Exit Reason</td>
                    <td style="color:#e6edf3;padding:6px 12px;font-weight:600;">{exit_reason.replace('_', ' ').title()}</td></tr>
                <tr><td style="color:#8b949e;padding:6px 12px;">Entry Spot</td>
                    <td style="color:#e6edf3;padding:6px 12px;font-weight:600;">{entry_spot:,.1f}</td></tr>
                <tr><td style="color:#8b949e;padding:6px 12px;">Exit Spot</td>
                    <td style="color:#e6edf3;padding:6px 12px;font-weight:600;">{exit_spot:,.1f}</td></tr>
                <tr><td style="color:#8b949e;padding:6px 12px;">Strikes</td>
                    <td style="color:#e6edf3;padding:6px 12px;font-weight:600;">{sell_strike} / {buy_strike}</td></tr>
                <tr><td style="color:#8b949e;padding:6px 12px;">Realized P&L</td>
                    <td style="color:{pnl_color};padding:6px 12px;font-weight:700;font-size:16px;">
                        {'+' if is_profit else ''}{_format_currency(realized_pnl)} ({pnl_pct:+.1f}%)
                    </td></tr>
            </table>
        </div>
        <div style="padding:12px 24px 16px;border-top:1px solid #21262d;">
            <p style="color:#484f58;font-size:11px;margin:0;text-align:center;">
                Paper Trading Bot &bull; {version} &bull; Auto-generated alert
            </p>
        </div>
    </div>
    """


async def send_trade_alert(
    version: str,
    signal: dict,
    trade_result: dict,
    spot: float,
    vix: float,
    prediction: float,
) -> bool:
    """
    Send email alert when a trade is opened.
    Returns True if sent successfully.
    """
    if not RESEND_API_KEY:
        logger.warning("RESEND_API_KEY not set — skipping email notification")
        return False

    cfg = VERSION_CONFIGS.get(version, {})
    signal_type = signal["signal"]
    emoji = _signal_emoji(signal_type)

    subject = (
        f"{emoji} {signal_type.replace('_', ' ').upper()} — "
        f"{version} ({cfg.get('label', '')}) | "
        f"Nifty {spot:,.0f} | "
        f"Credit {_format_currency(trade_result.get('total_credit', 0))}"
    )

    html = _build_trade_html(version, signal, trade_result, spot, vix, prediction)

    return await _send_email(subject, html)


async def send_no_trade_alert(
    prediction: float,
    spot: float,
    vix: float,
    version_signals: list[dict],
    blocked_versions: list[dict] = None,
) -> bool:
    """Send email when no trades were opened.

    blocked_versions: if non-empty, signals were bullish but trades were
    blocked by entry gap / position limits (not by model signal).
    """
    if not RESEND_API_KEY:
        logger.warning("RESEND_API_KEY not set — skipping email notification")
        return False

    blocked_versions = blocked_versions or []
    if blocked_versions:
        signals = ", ".join(bv["signal"].replace("_", " ") for bv in blocked_versions)
        subject = (
            f"\u26A0\uFE0F TRADE BLOCKED | "
            f"Signal: {signals} | "
            f"Nifty {spot:,.0f} | Drawdown {prediction*100:.2f}%"
        )
    else:
        subject = (
            f"\U0001F534 NO TRADE TODAY | "
            f"Nifty {spot:,.0f} | Drawdown {prediction*100:.2f}%"
        )

    html = _build_no_trade_html(prediction, spot, vix, version_signals, blocked_versions)
    return await _send_email(subject, html)


async def send_exit_alert(
    version: str,
    trade_id: str,
    exit_reason: str,
    trade_type: str,
    entry_spot: float,
    exit_spot: float,
    realized_pnl: float,
    pnl_pct: float,
    sell_strike: float,
    buy_strike: float,
) -> bool:
    """Send email when a trade is closed."""
    if not RESEND_API_KEY:
        logger.warning("RESEND_API_KEY not set — skipping email notification")
        return False

    is_profit = realized_pnl > 0
    emoji = "\u2705" if is_profit else "\u274C"
    cfg = VERSION_CONFIGS.get(version, {})

    subject = (
        f"{emoji} TRADE CLOSED — {version} ({cfg.get('label', '')}) | "
        f"P&L {'+' if is_profit else ''}{_format_currency(realized_pnl)} | "
        f"{exit_reason.replace('_', ' ').title()}"
    )

    html = _build_exit_html(
        version, trade_id, exit_reason, trade_type,
        entry_spot, exit_spot, realized_pnl, pnl_pct,
        sell_strike, buy_strike,
    )
    return await _send_email(subject, html)


async def _send_email(subject: str, html: str) -> bool:
    """Send email via Resend API."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.resend.com/emails",
                headers={
                    "Authorization": f"Bearer {RESEND_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "from": FROM_EMAIL,
                    "to": [NOTIFY_EMAIL],
                    "subject": subject,
                    "html": html,
                },
                timeout=10,
            )

            if response.status_code in (200, 202):
                data = response.json()
                logger.info(f"Email sent successfully: {data.get('id', 'unknown')}")
                return True
            else:
                logger.error(f"Email send failed: {response.status_code} — {response.text}")
                return False

    except Exception as e:
        logger.error(f"Email send error: {e}")
        return False


async def send_pipeline_failure_alert(
    pipeline_name: str,
    error_msg: str,
    stage: str = "",
) -> bool:
    """Send alert when a scheduled pipeline (prediction, exit check, EOD) fails.

    Suppresses duplicate alerts for the same pipeline within ALERT_COOLDOWN_SECONDS
    to avoid flooding the inbox when the Dhan API is down.
    """
    if not RESEND_API_KEY:
        logger.warning("RESEND_API_KEY not set — skipping pipeline failure alert")
        return False

    # Dedup: skip if we already sent an alert for this pipeline recently
    now = time.monotonic()
    last_sent = _last_alert_ts.get(pipeline_name, 0)
    if now - last_sent < ALERT_COOLDOWN_SECONDS:
        logger.warning(
            f"Pipeline failure alert for '{pipeline_name}' suppressed "
            f"(cooldown: {int(ALERT_COOLDOWN_SECONDS - (now - last_sent))}s remaining)"
        )
        return False

    emoji = "\U0001F6A8"  # 🚨
    subject = (
        f"{emoji} PIPELINE FAILED — {pipeline_name} | "
        f"{now_ist().strftime('%d %b %Y, %I:%M %p IST')}"
    )

    html = f"""
    <div style="font-family:'Segoe UI',Roboto,Helvetica,Arial,sans-serif;max-width:520px;margin:0 auto;background:#0d1117;border:1px solid #30363d;border-radius:12px;overflow:hidden;">
        <div style="background:#da3633;padding:20px 24px;">
            <h1 style="margin:0;color:#fff;font-size:20px;font-weight:600;">
                {emoji} Pipeline Failure
            </h1>
            <p style="margin:4px 0 0;color:rgba(255,255,255,0.85);font-size:13px;">
                {now_ist().strftime('%d %b %Y, %I:%M %p IST')}
            </p>
        </div>
        <div style="padding:20px 24px;">
            <table style="width:100%;border-collapse:collapse;background:#161b22;border-radius:8px;">
                <tr>
                    <td style="color:#8b949e;padding:6px 12px;">Pipeline</td>
                    <td style="color:#e6edf3;padding:6px 12px;font-weight:600;">{pipeline_name}</td>
                </tr>
                {"<tr><td style='color:#8b949e;padding:6px 12px;'>Stage</td><td style='color:#e6edf3;padding:6px 12px;font-weight:600;'>" + stage + "</td></tr>" if stage else ""}
                <tr>
                    <td style="color:#8b949e;padding:6px 12px;">Error</td>
                    <td style="color:#f85149;padding:6px 12px;font-weight:600;font-size:12px;word-break:break-all;">
                        {error_msg[:500]}
                    </td>
                </tr>
            </table>
            <p style="color:#8b949e;font-size:13px;margin-top:16px;">
                Check server logs on
                <a href="https://dashboard.render.com/" style="color:#58a6ff;">Render Dashboard</a>
                for full error details.
            </p>
            <p style="color:#8b949e;font-size:12px;margin-top:8px;">
                Manual trigger: POST <a href="https://nifty-paper-trading-api.onrender.com/api/trigger-prediction"
                style="color:#58a6ff;">/api/trigger-prediction</a>
            </p>
        </div>
        <div style="padding:12px 24px 16px;border-top:1px solid #21262d;">
            <p style="color:#484f58;font-size:11px;margin:0;text-align:center;">
                Paper Trading Bot &bull; Pipeline Failure Alert
            </p>
        </div>
    </div>
    """

    sent = await _send_email(subject, html)
    if sent:
        _last_alert_ts[pipeline_name] = time.monotonic()
    return sent


async def send_data_stale_alert(
    last_parquet_date: str,
    update_status: str,
    error_msg: str = "",
) -> bool:
    """Send alert when daily data update fails and parquet is stale."""
    if not RESEND_API_KEY:
        logger.warning("RESEND_API_KEY not set — skipping stale data alert")
        return False

    subject = (
        f"\u26A0\uFE0F DATA STALE — Parquet stuck at {last_parquet_date} | "
        f"Predictions may be wrong!"
    )

    html = f"""
    <div style="font-family:'Segoe UI',Roboto,Helvetica,Arial,sans-serif;max-width:520px;margin:0 auto;background:#0d1117;border:1px solid #30363d;border-radius:12px;overflow:hidden;">
        <div style="background:#da3633;padding:20px 24px;">
            <h1 style="margin:0;color:#fff;font-size:20px;font-weight:600;">
                \u26A0\uFE0F Stale Data Alert
            </h1>
            <p style="margin:4px 0 0;color:rgba(255,255,255,0.85);font-size:13px;">
                {now_ist().strftime('%d %b %Y, %I:%M %p IST')}
            </p>
        </div>
        <div style="padding:20px 24px;">
            <p style="color:#f85149;font-size:16px;font-weight:600;">
                merged_daily.parquet is stuck at {last_parquet_date}
            </p>
            <p style="color:#8b949e;font-size:14px;">
                The daily data update returned: <strong style="color:#e6edf3;">{update_status}</strong>
            </p>
            {"<p style='color:#8b949e;font-size:13px;'>Error: " + error_msg + "</p>" if error_msg else ""}
            <p style="color:#f85149;font-size:14px;margin-top:16px;">
                \u26A0\uFE0F Today's predictions will use stale features and may be incorrect.
                Check the Yahoo Finance API or server logs.
            </p>
            <p style="color:#8b949e;font-size:12px;margin-top:16px;">
                Debug: <a href="https://nifty-paper-trading-api.onrender.com/api/debug/yahoo-test"
                style="color:#58a6ff;">/api/debug/yahoo-test</a> |
                Trigger: POST <a href="https://nifty-paper-trading-api.onrender.com/api/update-data"
                style="color:#58a6ff;">/api/update-data</a>
            </p>
        </div>
    </div>
    """

    return await _send_email(subject, html)
