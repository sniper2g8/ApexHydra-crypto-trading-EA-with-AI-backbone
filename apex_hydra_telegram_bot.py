"""
╔══════════════════════════════════════════════════════════════════════╗
║         ApexHydra Crypto — Telegram Management Bot                  ║
║  Full remote control + live alerts via Telegram                      ║
║                                                                      ║
║  Setup:                                                              ║
║    pip install python-telegram-bot>=20.0 supabase python-dotenv     ║
║                                                                      ║
║  1. Create bot: message @BotFather → /newbot                         ║
║  2. Get your chat ID: message @userinfobot                           ║
║  3. Set env vars (see .env.example below) or use .env file           ║
║  4. Run: python apex_hydra_telegram_bot.py                           ║
║                                                                      ║
║  .env.example:                                                       ║
║    TELEGRAM_BOT_TOKEN=123456:ABC...                                  ║
║    TELEGRAM_ALLOWED_IDS=123456789,987654321                          ║
║    SUPABASE_URL=https://xxx.supabase.co                              ║
║    SUPABASE_KEY=your_service_role_key                                ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import os
import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional
from functools import wraps

from dotenv import load_dotenv
from supabase import create_client, Client
from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    BotCommand,
)
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
    JobQueue,
)
from telegram.constants import ParseMode

load_dotenv()
logging.basicConfig(
    format="%(asctime)s │ %(name)s │ %(levelname)s │ %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("ApexHydra-Bot")

# ──────────────────────────────────────────────────────────────────────
#  CONFIG
# ──────────────────────────────────────────────────────────────────────

BOT_TOKEN    = os.environ["TELEGRAM_BOT_TOKEN"]
ALLOWED_IDS  = set(int(x) for x in os.environ.get("TELEGRAM_ALLOWED_IDS", "").split(",") if x.strip())
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]

# Alert thresholds
DD_ALERT_PCT       = float(os.getenv("DD_ALERT_PCT",       "10.0"))  # Alert at 10% drawdown
DD_CRITICAL_PCT    = float(os.getenv("DD_CRITICAL_PCT",    "18.0"))  # Critical at 18%
MONITOR_INTERVAL_S = int(os.getenv("MONITOR_INTERVAL_S",  "60"))    # Check every 60s

# ──────────────────────────────────────────────────────────────────────
#  SUPABASE CLIENT (module-level singleton)
# ──────────────────────────────────────────────────────────────────────

sb: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ── State tracking for alerts (avoid repeat spam) ────────────────────
_alert_state: dict = {
    "last_dd_alert":   None,
    "dd_alerted_pct":  0.0,
    "last_trade_alert": None,
    "halted_alerted":  False,
    "last_perf_snap":  None,
}

# ──────────────────────────────────────────────────────────────────────
#  HELPERS
# ──────────────────────────────────────────────────────────────────────

REGIME_EMOJI = {
    "Trend Bull":       "🟢",
    "Trend Bear":       "🔴",
    "Ranging":          "🟡",
    "High Volatility":  "🟠",
    "Breakout":         "🟣",
    "Undefined":        "⚪",
}

SIGNAL_EMOJI = {2: "🚀", 1: "📈", 0: "⏳", -1: "📉", -2: "💥"}

def fmt_pnl(v) -> str:
    try:
        f = float(v)
        return f"✅ +${f:,.2f}" if f >= 0 else f"❌ -${abs(f):,.2f}"
    except:
        return str(v)

def fmt_pct(v) -> str:
    try:
        return f"{float(v):.1f}%"
    except:
        return "N/A"


def db_get_config() -> dict:
    r = sb.table("ea_config").select("*").limit(1).execute()
    return r.data[0] if r.data else {}


def db_push_config(updates: dict) -> bool:
    try:
        r = sb.table("ea_config").select("id").limit(1).execute()
        if r.data:
            sb.table("ea_config").update({**updates, "updated_by": "telegram"}).eq("id", r.data[0]["id"]).execute()
        else:
            sb.table("ea_config").insert({**updates, "magic": 20250228, "updated_by": "telegram"}).execute()
        return True
    except Exception as e:
        logger.error(f"Config push failed: {e}")
        return False


def db_get_latest_performance() -> dict:
    """
    Returns the latest performance snapshot.
    If the performance table has data, uses that.
    Falls back to live_* fields patched into ea_config by the EA every 15s
    so Telegram always shows current balance/equity even before the first
    full performance row is written.
    """
    # Try performance table first (has full history)
    r = sb.table("performance").select("*").order("timestamp", desc=True).limit(1).execute()
    if r.data:
        row = r.data[0]
        # Merge live fields from ea_config if they're fresher
        try:
            cfg = sb.table("ea_config").select(
                "live_balance,live_equity,live_dd_pct,live_pnl,"
                "live_trades,live_wins,live_losses,live_ts"
            ).limit(1).execute()
            if cfg.data and cfg.data[0].get("live_balance"):
                c = cfg.data[0]
                # Use live values if ea_config was updated more recently
                row["balance"]      = c.get("live_balance",  row.get("balance", 0))
                row["equity"]       = c.get("live_equity",   row.get("equity", 0))
                row["drawdown"]     = (c.get("live_dd_pct", 0) or 0) / 100.0
                row["total_pnl"]    = c.get("live_pnl",      row.get("total_pnl", 0))
                row["total_trades"] = c.get("live_trades",   row.get("total_trades", 0))
                row["wins"]         = c.get("live_wins",     row.get("wins", 0))
                row["losses"]       = c.get("live_losses",   row.get("losses", 0))
        except Exception:
            pass
        return row

    # No performance rows yet — build a synthetic row from ea_config live fields
    try:
        cfg = sb.table("ea_config").select("*").limit(1).execute()
        if cfg.data:
            c = cfg.data[0]
            if c.get("live_balance"):
                return {
                    "balance":       c.get("live_balance", 0),
                    "equity":        c.get("live_equity",  0),
                    "drawdown":      (c.get("live_dd_pct", 0) or 0) / 100.0,
                    "total_pnl":     c.get("live_pnl",     0),
                    "total_trades":  c.get("live_trades",  0),
                    "wins":          c.get("live_wins",    0),
                    "losses":        c.get("live_losses",  0),
                    "global_accuracy": 0.0,
                    "timestamp":     c.get("live_ts",      "N/A"),
                }
    except Exception:
        pass
    return {}


def db_get_recent_trades(limit: int = 10) -> list:
    r = sb.table("trades").select("*").order("timestamp", desc=True).limit(limit).execute()
    return r.data or []


def db_get_recent_regime_changes(limit: int = 5) -> list:
    r = sb.table("regime_changes").select("*").order("timestamp", desc=True).limit(limit).execute()
    return r.data or []


def db_get_current_regimes() -> list:
    """Latest regime per symbol via the SQL view."""
    try:
        r = sb.table("current_regimes").select("*").execute()
        return r.data or []
    except:
        # Fallback: manual latest-per-symbol
        r = sb.table("regime_changes").select("*").order("timestamp", desc=True).limit(50).execute()
        seen, results = set(), []
        for row in (r.data or []):
            if row["symbol"] not in seen:
                seen.add(row["symbol"])
                results.append(row)
        return results


def db_get_trade_summary() -> list:
    try:
        r = sb.table("trade_summary").select("*").execute()
        return r.data or []
    except:
        return []


def db_get_recent_events(limit: int = 10) -> list:
    r = sb.table("events").select("*").order("timestamp", desc=True).limit(limit).execute()
    return r.data or []


# ──────────────────────────────────────────────────────────────────────
#  AUTH DECORATOR
# ──────────────────────────────────────────────────────────────────────

def restricted(func):
    @wraps(func)
    async def wrapper(update: Update, ctx: ContextTypes.DEFAULT_TYPE, *args, **kwargs):
        user_id = update.effective_user.id
        if ALLOWED_IDS and user_id not in ALLOWED_IDS:
            await update.message.reply_text("⛔ Unauthorized. Your ID: " + str(user_id))
            logger.warning(f"Unauthorized access attempt from {user_id}")
            return
        return await func(update, ctx, *args, **kwargs)
    return wrapper


def restricted_callback(func):
    @wraps(func)
    async def wrapper(update: Update, ctx: ContextTypes.DEFAULT_TYPE, *args, **kwargs):
        user_id = update.effective_user.id
        if ALLOWED_IDS and user_id not in ALLOWED_IDS:
            await update.callback_query.answer("⛔ Unauthorized")
            return
        return await func(update, ctx, *args, **kwargs)
    return wrapper


# ──────────────────────────────────────────────────────────────────────
#  COMMAND HANDLERS
# ──────────────────────────────────────────────────────────────────────

@restricted
async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    text = (
        "⚡ <b>ApexHydra Crypto Bot</b> v4.0\n\n"
        "📊 <b>Monitoring Commands:</b>\n"
        "/status — EA status + account summary\n"
        "/perf — Performance metrics\n"
        "/trades — Last 10 trades\n"
        "/regimes — Current market regimes\n"
        "/summary — Per-symbol P&amp;L summary\n"
        "/events — Recent event log\n\n"
        "🎛 <b>Control Commands:</b>\n"
        "/resume — Resume EA trading\n"
        "/pause — Pause EA (no new trades)\n"
        "/stop — ⚠️ Emergency halt\n"
        "/config — View current settings\n"
        "/setcapital &lt;amount&gt; — Set allocated capital\n"
        "/setrisk &lt;pct&gt; — Set risk % per trade\n"
        "/setconf &lt;0.40-0.90&gt; — Set min AI confidence\n"
        "/setmaxdd &lt;pct&gt; — Set max drawdown % halt\n"
        "/setmaxpos &lt;n&gt; — Set max simultaneous positions\n\n"
        "🔔 <b>Alerts:</b>\n"
        f"Drawdown alert: &gt;{DD_ALERT_PCT}%\n"
        f"Drawdown critical: &gt;{DD_CRITICAL_PCT}%\n"
        f"Monitor interval: every {MONITOR_INTERVAL_S}s\n\n"
        "Use /help for detailed descriptions."
    )
    await update.message.reply_text(text, parse_mode=ParseMode.HTML)


@restricted
async def cmd_status(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    config = db_get_config()
    perf   = db_get_latest_performance()

    is_halted = config.get("halted", False)
    is_paused = config.get("paused", False)

    if is_halted:
        status_icon = "⛔ HALTED"
    elif is_paused:
        status_icon = "⏸ PAUSED"
    else:
        status_icon = "✅ ACTIVE"

    balance    = float(perf.get("balance",       0))
    equity     = float(perf.get("equity",        0))
    dd_frac    = float(perf.get("drawdown",      0))
    dd         = dd_frac * 100.0          # stored as fraction (0.05 = 5%), convert to %
    tot_trades = int(perf.get("total_trades",    0))
    wins       = int(perf.get("wins",            0))
    losses     = int(perf.get("losses",          0))
    tot_pnl    = float(perf.get("total_pnl",     0))
    wr         = wins / tot_trades * 100 if tot_trades > 0 else 0
    alloc      = float(config.get("allocated_capital", 0))
    updated_at = config.get("updated_at", "N/A")

    keyboard = InlineKeyboardMarkup([
        [
            InlineKeyboardButton("▶ Resume", callback_data="ctrl_resume"),
            InlineKeyboardButton("⏸ Pause",  callback_data="ctrl_pause"),
        ],
        [InlineKeyboardButton("⛔ Emergency Stop", callback_data="ctrl_stop")],
        [InlineKeyboardButton("🔄 Refresh",          callback_data="status_refresh")],
    ])

    text = (
        f"<b>⚡ ApexHydra Status</b>\n"
        f"{'─'*28}\n"
        f"<b>Status:</b> {status_icon}\n"
        f"<b>Config sync:</b> {updated_at[:16] if isinstance(updated_at, str) else 'N/A'}\n\n"
        f"<b>💰 Account</b>\n"
        f"Balance:  <code>${balance:,.2f}</code>\n"
        f"Equity:   <code>${equity:,.2f}</code>  ({equity-balance:+.2f})\n"
        f"Total P&amp;L: <code>{fmt_pnl(tot_pnl)}</code>\n"
        f"Drawdown: <code>{dd:.1f}%</code>{'  ⚠️' if dd > DD_ALERT_PCT else ''}\n\n"
        f"<b>📊 Performance</b>\n"
        f"Trades: <code>{tot_trades}</code> (W:{wins} / L:{losses})\n"
        f"Win Rate: <code>{wr:.1f}%</code>\n\n"
        f"<b>⚙️ Settings</b>\n"
        f"Risk/trade:   <code>{config.get('risk_pct', '?')}%</code>\n"
        f"Max DD:       <code>{config.get('max_dd_pct', '?')}%</code>\n"
        f"Max Pos:      <code>{config.get('max_positions', '?')}</code>\n"
        f"Min Conf:     <code>{float(config.get('min_confidence', 0))*100:.0f}%</code>\n"
        f"Capital:      <code>{'$'+f'{alloc:,.0f}' if alloc > 0 else 'Full balance'}</code>\n"
    )

    await update.message.reply_text(text, parse_mode=ParseMode.HTML, reply_markup=keyboard)


@restricted
async def cmd_perf(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    perf = db_get_latest_performance()
    if not perf:
        await update.message.reply_text("📭 No performance data yet.")
        return

    balance   = float(perf.get("balance", 0))
    equity    = float(perf.get("equity", 0))
    dd        = float(perf.get("drawdown", 0)) * 100.0   # stored as fraction, convert to %
    tot_t     = int(perf.get("total_trades", 0))
    wins      = int(perf.get("wins", 0))
    losses    = int(perf.get("losses", 0))
    tot_pnl   = float(perf.get("total_pnl", 0))
    ai_acc    = float(perf.get("global_accuracy", 0)) * 100
    wr        = wins / tot_t * 100 if tot_t > 0 else 0
    ts        = perf.get("timestamp", "")[:16]

    # Profit factor approximation from trade summary
    summary  = db_get_trade_summary()
    gross_win = sum(float(r.get("total_pnl", 0)) for r in summary if float(r.get("total_pnl", 0)) > 0)
    gross_los = abs(sum(float(r.get("total_pnl", 0)) for r in summary if float(r.get("total_pnl", 0)) < 0))
    pf = gross_win / gross_los if gross_los > 0 else 0.0

    dd_color = "🟥" if dd > DD_CRITICAL_PCT else "🟧" if dd > DD_ALERT_PCT else "🟩"
    pnl_str  = f"✅ +${tot_pnl:,.2f}" if tot_pnl >= 0 else f"❌ -${abs(tot_pnl):,.2f}"

    text = (
        f"<b>📊 Performance Report</b>\n"
        f"<i>{ts} UTC</i>\n"
        f"{'─'*28}\n"
        f"Balance:      <code>${balance:,.2f}</code>\n"
        f"Equity:       <code>${equity:,.2f}</code>\n"
        f"Total P&amp;L:    {pnl_str}\n"
        f"Drawdown:     {dd_color} <code>{dd:.2f}%</code>\n\n"
        f"<b>Trades</b>\n"
        f"Total:        <code>{tot_t}</code>\n"
        f"Wins/Losses:  <code>{wins} / {losses}</code>\n"
        f"Win Rate:     <code>{wr:.1f}%</code>\n"
        f"Profit Factor:<code>{pf:.2f}</code>\n"
        f"AI Accuracy:  <code>{ai_acc:.1f}%</code>\n"
    )
    if summary:
        text += f"\n<b>Per Symbol</b>\n"
        for row in summary[:5]:
            sym = row.get("symbol", "")
            sym_pnl = float(row.get("total_pnl", 0))
            sym_wr  = float(row.get("win_rate_pct", 0))
            pnl_icon = "✅" if sym_pnl >= 0 else "❌"
            text += f"{pnl_icon} <code>{sym:<8}</code> P&amp;L: <code>${sym_pnl:+,.2f}</code> WR: <code>{sym_wr:.0f}%</code>\n"

    await update.message.reply_text(text, parse_mode=ParseMode.HTML)


@restricted
async def cmd_trades(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    trades = db_get_recent_trades(10)
    if not trades:
        await update.message.reply_text("📭 No trades recorded yet.")
        return

    text = "<b>📋 Last 10 Trades</b>\n" + "─" * 28 + "\n"
    for t in trades:
        ts     = str(t.get("timestamp", ""))[:16]
        sym    = t.get("symbol", "")
        action = t.get("action", "")
        regime = t.get("regime", "?")
        pnl    = t.get("pnl")
        conf   = float(t.get("confidence", 0)) * 100
        lots   = t.get("lots", 0)

        if action == "CLOSE" and pnl is not None:
            pnl_v = float(pnl)
            icon  = "✅" if pnl_v >= 0 else "❌"
            pnl_s = f"  P&amp;L: <code>{icon}{pnl_v:+.2f}</code>"
        else:
            pnl_s = ""
        reg_e = REGIME_EMOJI.get(str(regime), "⚪")
        text += (
            f"<code>{ts}</code> {reg_e} <b>{sym}</b> <code>{action}</code> "
            f"lots:<code>{lots}</code> conf:<code>{conf:.0f}%</code>{pnl_s}\n"
        )

    await update.message.reply_text(text, parse_mode=ParseMode.HTML)


@restricted
async def cmd_regimes(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    regimes = db_get_current_regimes()
    if not regimes:
        await update.message.reply_text("📭 No regime data yet.")
        return

    text = "<b>🌐 Current Market Regimes</b>\n" + "─" * 28 + "\n"
    for r in regimes:
        sym   = r.get("symbol", "")
        reg   = r.get("regime", "Undefined")
        conf  = float(r.get("confidence", 0)) * 100
        adx   = r.get("adx", "?")
        rsi   = r.get("rsi", "?")
        ai    = float(r.get("ai_score", 0)) * 100
        ts    = str(r.get("timestamp", ""))[:16]
        icon  = REGIME_EMOJI.get(str(reg), "⚪")
        ai_s  = f"AI:<code>{ai:+.1f}%</code>"
        text += (
            f"{icon} <b>{sym}</b> — <i>{reg}</i>\n"
            f"  conf:<code>{conf:.0f}%</code> ADX:<code>{adx}</code> RSI:<code>{rsi}</code> {ai_s}\n"
            f"  <i>{ts} UTC</i>\n\n"
        )

    await update.message.reply_text(text, parse_mode=ParseMode.HTML)


@restricted
async def cmd_summary(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    summary = db_get_trade_summary()
    if not summary:
        await update.message.reply_text("📭 No trade summary available yet.")
        return

    text = "<b>💰 Per-Symbol P&amp;L Summary</b>\n" + "─" * 28 + "\n"
    total_pnl = 0.0
    for row in summary:
        sym     = row.get("symbol", "")
        pnl     = float(row.get("total_pnl", 0))
        wr      = float(row.get("win_rate_pct", 0))
        trades  = int(row.get("total_trades", 0))
        w       = int(row.get("wins", 0))
        l       = int(row.get("losses", 0))
        icon    = "✅" if pnl >= 0 else "❌"
        total_pnl += pnl
        text += (
            f"{icon} <b>{sym}</b>\n"
            f"  P&amp;L:<code>${pnl:+,.2f}</code> WR:<code>{wr:.0f}%</code> T:<code>{trades}</code> (W:{w}/L:{l})\n"
        )

    total_icon = "✅" if total_pnl >= 0 else "❌"
    text += f"{'─'*28}\n{total_icon} <b>TOTAL: <code>${total_pnl:+,.2f}</code></b>\n"
    await update.message.reply_text(text, parse_mode=ParseMode.HTML)


@restricted
async def cmd_events(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    events = db_get_recent_events(15)
    if not events:
        await update.message.reply_text("📭 No events logged yet.")
        return

    text = "<b>📟 Recent Events</b>\n" + "─" * 28 + "\n"
    TYPE_ICON = {"HALT": "⛔", "RESUME": "▶️", "OPEN": "📂", "CLOSE": "📁",
                 "ERROR": "🔴", "INFO": "ℹ️", "DEINIT": "🔌", "WARN": "⚠️"}
    for ev in events:
        ts  = str(ev.get("timestamp", ""))[:16]
        typ = ev.get("type", "INFO")
        msg = str(ev.get("message", ""))[:80]
        ico = TYPE_ICON.get(typ, "•")
        # Escape HTML special chars in msg
        msg = msg.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        text += f"<code>{ts}</code> {ico} <b>{typ}</b> {msg}\n"

    await update.message.reply_text(text, parse_mode=ParseMode.HTML)


@restricted
async def cmd_config(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    cfg = db_get_config()
    alloc = float(cfg.get("allocated_capital", 0))
    text = (
        f"<b>⚙️ EA Configuration</b>\n"
        f"{'─'*28}\n"
        f"Allocated Capital: <code>{'$'+f'{alloc:,.0f}' if alloc > 0 else 'Full balance'}</code>\n"
        f"Risk % / trade:    <code>{cfg.get('risk_pct', '?')}%</code>\n"
        f"Max Drawdown:      <code>{cfg.get('max_dd_pct', '?')}%</code>\n"
        f"Max Positions:     <code>{cfg.get('max_positions', '?')}</code>\n"
        f"Min Confidence:    <code>{float(cfg.get('min_confidence', 0))*100:.0f}%</code>\n"
        f"Halted:            <code>{cfg.get('halted', False)}</code>\n"
        f"Paused:            <code>{cfg.get('paused', False)}</code>\n"
        f"Updated by:        <code>{cfg.get('updated_by', '?')}</code>\n"
        f"Updated at:        <code>{str(cfg.get('updated_at', 'N/A'))[:16]}</code>\n\n"
        f"<b>Quick edit commands:</b>\n"
        f"<code>/setcapital 5000</code> — allocate $5,000\n"
        f"<code>/setrisk 1.5</code> — 1.5% risk per trade\n"
        f"<code>/setconf 0.65</code> — 65% min confidence\n"
        f"<code>/setmaxdd 15</code> — halt at 15% drawdown\n"
        f"<code>/setmaxpos 5</code> — max 5 open positions\n"
    )
    await update.message.reply_text(text, parse_mode=ParseMode.HTML)


# ── Control commands ──────────────────────────────────────────────────

@restricted
async def cmd_resume(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if db_push_config({"halted": False, "paused": False}):
        await update.message.reply_text("▶️ EA <b>Resumed</b> — will apply on next config sync.", parse_mode=ParseMode.HTML)
    else:
        await update.message.reply_text("❌ Failed to update config.")


@restricted
async def cmd_pause(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if db_push_config({"paused": True}):
        await update.message.reply_text("⏸ EA <b>Paused</b> — no new trades will open.", parse_mode=ParseMode.HTML)
    else:
        await update.message.reply_text("❌ Failed to update config.")


@restricted
async def cmd_stop(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Two-step confirmation for emergency stop."""
    keyboard = InlineKeyboardMarkup([[
        InlineKeyboardButton("✅ CONFIRM HALT", callback_data="confirm_halt"),
        InlineKeyboardButton("❌ Cancel",        callback_data="cancel_halt"),
    ]])
    await update.message.reply_text(
        "⚠️ <b>Confirm Emergency Stop?</b>\nThis will halt the EA immediately. Use /resume to restart.",
        parse_mode=ParseMode.HTML,
        reply_markup=keyboard,
    )


@restricted
async def cmd_setcapital(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    try:
        amount = float(ctx.args[0]) if ctx.args else -1
        if amount < 0:
            raise ValueError
    except (ValueError, IndexError):
        await update.message.reply_text("Usage: <code>/setcapital 5000</code> (0 = use full balance)", parse_mode=ParseMode.HTML)
        return
    if db_push_config({"allocated_capital": amount}):
        if amount == 0:
            await update.message.reply_text("✅ Allocated capital cleared — using <b>full account balance</b>.", parse_mode=ParseMode.HTML)
        else:
            await update.message.reply_text(f"✅ Allocated capital set to <code>${amount:,.2f}</code>. EA will size lots based on this amount only.", parse_mode=ParseMode.HTML)
    else:
        await update.message.reply_text("❌ Failed to update config.")


@restricted
async def cmd_setrisk(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    try:
        val = float(ctx.args[0])
        assert 0.1 <= val <= 5.0
    except:
        await update.message.reply_text("Usage: <code>/setrisk 1.5</code> (range: 0.1 – 5.0)", parse_mode=ParseMode.HTML)
        return
    if db_push_config({"risk_pct": val}):
        await update.message.reply_text(f"✅ Risk per trade set to <code>{val}%</code>", parse_mode=ParseMode.HTML)
    else:
        await update.message.reply_text("❌ Failed to update config.")


@restricted
async def cmd_setconf(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    try:
        val = float(ctx.args[0])
        assert 0.40 <= val <= 0.95
    except:
        await update.message.reply_text("Usage: <code>/setconf 0.65</code> (range: 0.40 – 0.95)", parse_mode=ParseMode.HTML)
        return
    if db_push_config({"min_confidence": val}):
        await update.message.reply_text(f"✅ Min AI confidence set to <code>{val*100:.0f}%</code>", parse_mode=ParseMode.HTML)
    else:
        await update.message.reply_text("❌ Failed to update config.")


@restricted
async def cmd_setmaxdd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    try:
        val = float(ctx.args[0])
        assert 5.0 <= val <= 50.0
    except:
        await update.message.reply_text("Usage: <code>/setmaxdd 15</code> (range: 5 – 50)", parse_mode=ParseMode.HTML)
        return
    if db_push_config({"max_dd_pct": val}):
        await update.message.reply_text(f"✅ Max drawdown halt set to <code>{val}%</code>", parse_mode=ParseMode.HTML)
    else:
        await update.message.reply_text("❌ Failed to update config.")


@restricted
async def cmd_setmaxpos(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    try:
        val = int(ctx.args[0])
        assert 1 <= val <= 20
    except:
        await update.message.reply_text("Usage: <code>/setmaxpos 5</code> (range: 1 – 20)", parse_mode=ParseMode.HTML)
        return
    if db_push_config({"max_positions": val}):
        await update.message.reply_text(f"✅ Max positions set to <code>{val}</code>", parse_mode=ParseMode.HTML)
    else:
        await update.message.reply_text("❌ Failed to update config.")


@restricted
async def cmd_help(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    text = (
        "<b>📖 ApexHydra Bot — Help</b>\n\n"
        "<b>Monitoring:</b>\n"
        "<code>/status</code> — Full status with inline controls\n"
        "<code>/perf</code> — Performance metrics + per-symbol breakdown\n"
        "<code>/trades</code> — Last 10 trade entries/exits\n"
        "<code>/regimes</code> — Live market regime per symbol\n"
        "<code>/summary</code> — Per-symbol P&amp;L table\n"
        "<code>/events</code> — Recent EA event log\n"
        "<code>/config</code> — View all current settings\n\n"
        "<b>Control:</b>\n"
        "<code>/resume</code> — Resume EA trading\n"
        "<code>/pause</code> — Pause EA (no new trades)\n"
        "<code>/stop</code> — ⚠️ Emergency halt (with confirmation)\n\n"
        "<b>Risk Settings:</b>\n"
        "<code>/setcapital &lt;$&gt;</code> — Allocated capital (0 = full balance)\n"
        "<code>/setrisk &lt;pct&gt;</code> — Risk % per trade (0.1–5.0)\n"
        "<code>/setconf &lt;0.4–0.9&gt;</code> — Min AI confidence\n"
        "<code>/setmaxdd &lt;pct&gt;</code> — Max drawdown halt threshold\n"
        "<code>/setmaxpos &lt;n&gt;</code> — Max simultaneous positions\n\n"
        "<b>Auto-Alerts:</b>\n"
        f"• Drawdown exceeds <code>{DD_ALERT_PCT}%</code> or <code>{DD_CRITICAL_PCT}%</code>\n"
        f"• EA halted by drawdown limit\n"
        f"• Every trade open/close (with P&amp;L)\n"
        f"• Daily performance summary at 00:00 UTC\n"
    )
    await update.message.reply_text(text, parse_mode=ParseMode.HTML)


# ──────────────────────────────────────────────────────────────────────
#  INLINE BUTTON CALLBACKS
# ──────────────────────────────────────────────────────────────────────

@restricted_callback
async def button_handler(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data

    if data == "ctrl_resume":
        ok = db_push_config({"halted": False, "paused": False})
        await query.edit_message_text("▶️ EA <b>Resumed</b>.", parse_mode=ParseMode.HTML)

    elif data == "ctrl_pause":
        ok = db_push_config({"paused": True})
        await query.edit_message_text("⏸ EA <b>Paused</b>.", parse_mode=ParseMode.HTML)

    elif data == "ctrl_stop":
        keyboard = InlineKeyboardMarkup([[
            InlineKeyboardButton("✅ CONFIRM HALT", callback_data="confirm_halt"),
            InlineKeyboardButton("❌ Cancel",        callback_data="cancel_halt"),
        ]])
        await query.edit_message_text(
            "⚠️ <b>Confirm Emergency Stop?</b>",
            parse_mode=ParseMode.HTML,
            reply_markup=keyboard,
        )

    elif data == "confirm_halt":
        db_push_config({"halted": True, "paused": True})
        await query.edit_message_text("⛔ EA <b>HALTED</b>. Use /resume to restart.", parse_mode=ParseMode.HTML)

    elif data == "cancel_halt":
        await query.edit_message_text("✅ Halt cancelled.", parse_mode=ParseMode.HTML)

    elif data == "status_refresh":
        # Re-run status inline
        config = db_get_config()
        perf   = db_get_latest_performance()
        balance = float(perf.get("balance", 0))
        equity  = float(perf.get("equity", 0))
        dd      = float(perf.get("drawdown", 0)) * 100.0   # stored as fraction, convert to %
        tot_t   = int(perf.get("total_trades", 0))
        wins    = int(perf.get("wins", 0))
        losses  = int(perf.get("losses", 0))
        tot_pnl = float(perf.get("total_pnl", 0))
        wr      = wins / tot_t * 100 if tot_t > 0 else 0
        alloc   = float(config.get("allocated_capital", 0))
        is_halted = config.get("halted", False)
        is_paused = config.get("paused", False)
        status_icon = "⛔ HALTED" if is_halted else ("⏸ PAUSED" if is_paused else "✅ ACTIVE")
        pnl_s = f"+${tot_pnl:,.2f}" if tot_pnl >= 0 else f"-${abs(tot_pnl):,.2f}"
        text = (
            f"<b>⚡ ApexHydra Status</b>  <i>(refreshed)</i>\n"
            f"<b>Status:</b> {status_icon}\n"
            f"Balance: <code>${balance:,.2f}</code>  Equity: <code>${equity:,.2f}</code>\n"
            f"P&amp;L: <code>{pnl_s}</code>  DD: <code>{dd:.1f}%</code>\n"
            f"Trades: <code>{tot_t}</code> WR:<code>{wr:.1f}%</code>\n"
            f"Capital: <code>{'$'+f'{alloc:,.0f}' if alloc > 0 else 'Full balance'}</code>\n"
        )
        keyboard = InlineKeyboardMarkup([
            [
                InlineKeyboardButton("▶ Resume", callback_data="ctrl_resume"),
                InlineKeyboardButton("⏸ Pause",  callback_data="ctrl_pause"),
            ],
            [InlineKeyboardButton("⛔ Emergency Stop", callback_data="ctrl_stop")],
            [InlineKeyboardButton("🔄 Refresh",         callback_data="status_refresh")],
        ])
        await query.edit_message_text(text, parse_mode=ParseMode.HTML, reply_markup=keyboard)


# ──────────────────────────────────────────────────────────────────────
#  BACKGROUND MONITOR — runs every MONITOR_INTERVAL_S seconds
# ──────────────────────────────────────────────────────────────────────

async def monitor_job(ctx: ContextTypes.DEFAULT_TYPE):
    """Background task: checks for alerts and sends notifications."""
    if not ALLOWED_IDS:
        return
    chat_ids = list(ALLOWED_IDS)

    try:
        perf   = db_get_latest_performance()
        config = db_get_config()
        if not perf:
            return

        dd         = float(perf.get("drawdown", 0)) * 100.0   # stored as fraction, convert to %
        is_halted  = config.get("halted", False)
        perf_ts    = perf.get("timestamp", "")

        # ── Drawdown alerts ───────────────────────────────────────────
        now = datetime.now(timezone.utc)
        last_dd_alert = _alert_state.get("last_dd_alert")
        dd_alerted    = _alert_state.get("dd_alerted_pct", 0.0)
        cooldown      = timedelta(minutes=30)

        if dd >= DD_CRITICAL_PCT:
            if dd_alerted < DD_CRITICAL_PCT or (last_dd_alert and now - last_dd_alert > cooldown):
                msg = (
                    f"🔴 <b>CRITICAL DRAWDOWN ALERT</b>\n"
                    f"Current DD: <code>{dd:.2f}%</code> (threshold: <code>{DD_CRITICAL_PCT}%</code>)\n"
                    f"Balance: <code>${float(perf.get('balance',0)):,.2f}</code>\n"
                    f"Consider: /stop"
                )
                for cid in chat_ids:
                    await ctx.bot.send_message(cid, msg, parse_mode=ParseMode.HTML)
                _alert_state["last_dd_alert"]  = now
                _alert_state["dd_alerted_pct"] = dd

        elif dd >= DD_ALERT_PCT:
            if dd_alerted < DD_ALERT_PCT or (last_dd_alert and now - last_dd_alert > cooldown):
                msg = (
                    f"🟠 <b>Drawdown Warning</b>\n"
                    f"Current DD: <code>{dd:.2f}%</code> (alert at <code>{DD_ALERT_PCT}%</code>)\n"
                    f"Balance: <code>${float(perf.get('balance',0)):,.2f}</code>"
                )
                for cid in chat_ids:
                    await ctx.bot.send_message(cid, msg, parse_mode=ParseMode.HTML)
                _alert_state["last_dd_alert"]  = now
                _alert_state["dd_alerted_pct"] = dd

        else:
            # Reset dd alert state when dd returns to normal
            _alert_state["dd_alerted_pct"] = 0.0

        # ── EA halted alert ───────────────────────────────────────────
        if is_halted and not _alert_state.get("halted_alerted"):
            msg = (
                f"⛔ <b>EA HALTED</b>\n"
                f"The EA has been halted. Current DD: <code>{dd:.2f}%</code>\n"
                f"Use /resume to restart when ready."
            )
            for cid in chat_ids:
                await ctx.bot.send_message(cid, msg, parse_mode=ParseMode.HTML)
            _alert_state["halted_alerted"] = True
        elif not is_halted:
            _alert_state["halted_alerted"] = False

        # ── New trade alerts ──────────────────────────────────────────
        last_trade_ts = _alert_state.get("last_trade_alert")
        trades = db_get_recent_trades(5)
        if trades:
            newest_ts = trades[0].get("timestamp", "")
            if last_trade_ts != newest_ts:
                _alert_state["last_trade_alert"] = newest_ts
                t = trades[0]
                action = t.get("action", "")
                sym    = t.get("symbol", "")
                regime = t.get("regime", "?")
                conf   = float(t.get("confidence", 0)) * 100
                lots   = t.get("lots", 0)
                pnl    = t.get("pnl")
                reg_e  = REGIME_EMOJI.get(str(regime), "⚪")
                ts_s   = str(newest_ts)[:16]

                if action == "OPEN":
                    price = t.get("price", 0)
                    sl    = t.get("sl", 0)
                    tp    = t.get("tp", 0)
                    msg   = (
                        f"📂 <b>Trade OPENED</b>\n"
                        f"{reg_e} <code>{sym}</code> | Lots: <code>{lots}</code> | Conf: <code>{conf:.0f}%</code>\n"
                        f"Entry: <code>{price}</code> | SL: <code>{sl}</code> | TP: <code>{tp}</code>\n"
                        f"Regime: <i>{regime}</i> | <code>{ts_s}</code>"
                    )
                elif action == "CLOSE" and pnl is not None:
                    pnl_v = float(pnl)
                    icon  = "✅" if pnl_v >= 0 else "❌"
                    msg   = (
                        f"📁 <b>Trade CLOSED</b> {icon}\n"
                        f"{reg_e} <code>{sym}</code> P&amp;L: <code>{pnl_v:+.2f}</code>\n"
                        f"Lots: <code>{lots}</code> | <code>{ts_s}</code>"
                    )
                else:
                    msg = None

                if msg:
                    for cid in chat_ids:
                        await ctx.bot.send_message(cid, msg, parse_mode=ParseMode.HTML)

    except Exception as e:
        logger.error(f"Monitor job error: {e}")


async def daily_summary_job(ctx: ContextTypes.DEFAULT_TYPE):
    """Sends a daily performance summary at midnight UTC."""
    if not ALLOWED_IDS:
        return

    try:
        perf    = db_get_latest_performance()
        summary = db_get_trade_summary()
        if not perf:
            return

        balance  = float(perf.get("balance", 0))
        equity   = float(perf.get("equity", 0))
        dd       = float(perf.get("drawdown", 0)) * 100.0   # stored as fraction, convert to %
        tot_t    = int(perf.get("total_trades", 0))
        wins     = int(perf.get("wins", 0))
        losses   = int(perf.get("losses", 0))
        tot_pnl  = float(perf.get("total_pnl", 0))
        wr       = wins / tot_t * 100 if tot_t > 0 else 0
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        pnl_icon = "✅" if tot_pnl >= 0 else "❌"

        text = (
            f"📅 <b>Daily Summary — {date_str}</b>\n"
            f"{'─'*28}\n"
            f"Balance:  <code>${balance:,.2f}</code> | Equity: <code>${equity:,.2f}</code>\n"
            f"P&amp;L:      {pnl_icon} <code>{tot_pnl:+,.2f}</code>\n"
            f"Drawdown: <code>{dd:.2f}%</code>\n"
            f"Trades:   <code>{tot_t}</code> (W:{wins}/L:{losses}) WR:<code>{wr:.1f}%</code>\n"
        )
        if summary:
            text += "\n<b>Symbols:</b>\n"
            for row in summary[:6]:
                sym     = row.get("symbol", "")
                pnl     = float(row.get("total_pnl", 0))
                wr_s    = float(row.get("win_rate_pct", 0))
                icon    = "✅" if pnl >= 0 else "❌"
                text += f"{icon} <code>{sym:<8}</code> <code>{pnl:+,.2f}</code> WR:<code>{wr_s:.0f}%</code>\n"

        for cid in ALLOWED_IDS:
            await ctx.bot.send_message(cid, text, parse_mode=ParseMode.HTML)

    except Exception as e:
        logger.error(f"Daily summary job error: {e}")


# ──────────────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────────────

async def post_init(application: Application):
    """Set bot commands menu."""
    commands = [
        BotCommand("start",       "Show welcome message"),
        BotCommand("status",      "EA status + account summary"),
        BotCommand("perf",        "Performance metrics"),
        BotCommand("trades",      "Last 10 trades"),
        BotCommand("regimes",     "Current market regimes"),
        BotCommand("summary",     "Per-symbol P&L"),
        BotCommand("events",      "Recent event log"),
        BotCommand("config",      "View current settings"),
        BotCommand("resume",      "Resume EA"),
        BotCommand("pause",       "Pause EA"),
        BotCommand("stop",        "Emergency halt"),
        BotCommand("setcapital",  "Set allocated capital"),
        BotCommand("setrisk",     "Set risk % per trade"),
        BotCommand("setconf",     "Set min AI confidence"),
        BotCommand("setmaxdd",    "Set max drawdown % halt"),
        BotCommand("setmaxpos",   "Set max positions"),
        BotCommand("help",        "Detailed help"),
    ]
    await application.bot.set_my_commands(commands)
    logger.info("ApexHydra Telegram Bot started.")


def main():
    if not BOT_TOKEN:
        raise ValueError("TELEGRAM_BOT_TOKEN not set")
    if not ALLOWED_IDS:
        logger.warning("TELEGRAM_ALLOWED_IDS not set — bot is open to everyone!")

    app = (
        Application.builder()
        .token(BOT_TOKEN)
        .post_init(post_init)
        .build()
    )

    # Command handlers
    for cmd, handler in [
        ("start",       cmd_start),
        ("help",        cmd_help),
        ("status",      cmd_status),
        ("perf",        cmd_perf),
        ("trades",      cmd_trades),
        ("regimes",     cmd_regimes),
        ("summary",     cmd_summary),
        ("events",      cmd_events),
        ("config",      cmd_config),
        ("resume",      cmd_resume),
        ("pause",       cmd_pause),
        ("stop",        cmd_stop),
        ("setcapital",  cmd_setcapital),
        ("setrisk",     cmd_setrisk),
        ("setconf",     cmd_setconf),
        ("setmaxdd",    cmd_setmaxdd),
        ("setmaxpos",   cmd_setmaxpos),
    ]:
        app.add_handler(CommandHandler(cmd, handler))

    app.add_handler(CallbackQueryHandler(button_handler))

    # Background jobs
    jq: JobQueue = app.job_queue
    jq.run_repeating(monitor_job, interval=MONITOR_INTERVAL_S, first=10)
    # Daily summary at 00:00 UTC
    jq.run_daily(daily_summary_job, time=datetime.strptime("00:00", "%H:%M").time().replace(tzinfo=timezone.utc))

    logger.info(f"Starting bot — monitoring every {MONITOR_INTERVAL_S}s — {len(ALLOWED_IDS)} authorized users")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
