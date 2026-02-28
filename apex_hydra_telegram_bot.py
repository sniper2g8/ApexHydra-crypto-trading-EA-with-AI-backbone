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
    r = sb.table("performance").select("*").order("timestamp", desc=True).limit(1).execute()
    return r.data[0] if r.data else {}


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
        "⚡ *ApexHydra Crypto Bot* v4.0\n\n"
        "📊 *Monitoring Commands:*\n"
        "/status — EA status + account summary\n"
        "/perf — Performance metrics\n"
        "/trades — Last 10 trades\n"
        "/regimes — Current market regimes\n"
        "/summary — Per-symbol P\\&L summary\n"
        "/events — Recent event log\n\n"
        "🎛 *Control Commands:*\n"
        "/resume — Resume EA trading\n"
        "/pause — Pause EA (no new trades)\n"
        "/stop — ⚠️ Emergency halt\n"
        "/config — View current settings\n"
        "/setcapital \\<amount\\> — Set allocated capital\n"
        "/setrisk \\<pct\\> — Set risk % per trade\n"
        "/setconf \\<0.40-0.90\\> — Set min AI confidence\n"
        "/setmaxdd \\<pct\\> — Set max drawdown % halt\n"
        "/setmaxpos \\<n\\> — Set max simultaneous positions\n\n"
        "🔔 *Alerts:*\n"
        f"Drawdown alert: >{DD_ALERT_PCT}%\n"
        f"Drawdown critical: >{DD_CRITICAL_PCT}%\n"
        f"Monitor interval: every {MONITOR_INTERVAL_S}s\n\n"
        "Use /help for detailed descriptions."
    )
    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN_V2)


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
    dd         = float(perf.get("drawdown",      0))
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
        f"*⚡ ApexHydra Status*\n"
        f"{'─'*28}\n"
        f"*Status:* {status_icon}\n"
        f"*Config sync:* {updated_at[:16] if isinstance(updated_at, str) else 'N/A'}\n\n"
        f"*💰 Account*\n"
        f"Balance:  `${balance:,.2f}`\n"
        f"Equity:   `${equity:,.2f}`  ({equity-balance:+.2f})\n"
        f"Total P\\&L: `{fmt_pnl(tot_pnl)}`\n"
        f"Drawdown: `{dd:.1f}%`{'  ⚠️' if dd > DD_ALERT_PCT else ''}\n\n"
        f"*📊 Performance*\n"
        f"Trades: `{tot_trades}` (W:{wins} / L:{losses})\n"
        f"Win Rate: `{wr:.1f}%`\n\n"
        f"*⚙️ Settings*\n"
        f"Risk/trade:   `{config.get('risk_pct', '?')}%`\n"
        f"Max DD:       `{config.get('max_dd_pct', '?')}%`\n"
        f"Max Pos:      `{config.get('max_positions', '?')}`\n"
        f"Min Conf:     `{float(config.get('min_confidence', 0))*100:.0f}%`\n"
        f"Capital:      `{'$'+f'{alloc:,.0f}' if alloc > 0 else 'Full balance'}`\n"
    )

    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN_V2, reply_markup=keyboard)


@restricted
async def cmd_perf(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    perf = db_get_latest_performance()
    if not perf:
        await update.message.reply_text("📭 No performance data yet.")
        return

    balance   = float(perf.get("balance", 0))
    equity    = float(perf.get("equity", 0))
    dd        = float(perf.get("drawdown", 0))
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
    pnl_str  = f"✅ \\+${tot_pnl:,.2f}" if tot_pnl >= 0 else f"❌ \\-${abs(tot_pnl):,.2f}"

    text = (
        f"*📊 Performance Report*\n"
        f"_{ts} UTC_\n"
        f"{'─'*28}\n"
        f"Balance:      `${balance:,.2f}`\n"
        f"Equity:       `${equity:,.2f}`\n"
        f"Total P\\&L:    {pnl_str}\n"
        f"Drawdown:     {dd_color} `{dd:.2f}%`\n\n"
        f"*Trades*\n"
        f"Total:        `{tot_t}`\n"
        f"Wins/Losses:  `{wins} / {losses}`\n"
        f"Win Rate:     `{wr:.1f}%`\n"
        f"Profit Factor:`{pf:.2f}`\n"
        f"AI Accuracy:  `{ai_acc:.1f}%`\n"
    )
    if summary:
        text += f"\n*Per Symbol*\n"
        for row in summary[:5]:
            sym = row.get("symbol", "")
            sym_pnl = float(row.get("total_pnl", 0))
            sym_wr  = float(row.get("win_rate_pct", 0))
            pnl_icon = "✅" if sym_pnl >= 0 else "❌"
            text += f"{pnl_icon} `{sym:<8}` P\\&L: `${sym_pnl:+,.2f}` WR: `{sym_wr:.0f}%`\n"

    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN_V2)


@restricted
async def cmd_trades(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    trades = db_get_recent_trades(10)
    if not trades:
        await update.message.reply_text("📭 No trades recorded yet.")
        return

    text = "*📋 Last 10 Trades*\n" + "─" * 28 + "\n"
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
            pnl_s = f"  P\\&L: `{icon}{pnl_v:+.2f}`"
        else:
            pnl_s = ""
        reg_e = REGIME_EMOJI.get(str(regime), "⚪")
        text += (
            f"`{ts}` {reg_e} *{sym}* `{action}` "
            f"lots:`{lots}` conf:`{conf:.0f}%`{pnl_s}\n"
        )

    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN_V2)


@restricted
async def cmd_regimes(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    regimes = db_get_current_regimes()
    if not regimes:
        await update.message.reply_text("📭 No regime data yet.")
        return

    text = "*🌐 Current Market Regimes*\n" + "─" * 28 + "\n"
    for r in regimes:
        sym   = r.get("symbol", "")
        reg   = r.get("regime", "Undefined")
        conf  = float(r.get("confidence", 0)) * 100
        adx   = r.get("adx", "?")
        rsi   = r.get("rsi", "?")
        ai    = float(r.get("ai_score", 0)) * 100
        ts    = str(r.get("timestamp", ""))[:16]
        icon  = REGIME_EMOJI.get(str(reg), "⚪")
        ai_s  = f"AI:`{ai:+.1f}%`"
        text += (
            f"{icon} *{sym}* — _{reg}_\n"
            f"  conf:`{conf:.0f}%` ADX:`{adx}` RSI:`{rsi}` {ai_s}\n"
            f"  _{ts} UTC_\n\n"
        )

    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN_V2)


@restricted
async def cmd_summary(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    summary = db_get_trade_summary()
    if not summary:
        await update.message.reply_text("📭 No trade summary available yet.")
        return

    text = "*💰 Per\\-Symbol P\\&L Summary*\n" + "─" * 28 + "\n"
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
            f"{icon} *{sym}*\n"
            f"  P\\&L:`${pnl:+,.2f}` WR:`{wr:.0f}%` T:`{trades}` \\(W:{w}/L:{l}\\)\n"
        )

    total_icon = "✅" if total_pnl >= 0 else "❌"
    text += f"{'─'*28}\n{total_icon} *TOTAL: `${total_pnl:+,.2f}`*\n"
    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN_V2)


@restricted
async def cmd_events(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    events = db_get_recent_events(15)
    if not events:
        await update.message.reply_text("📭 No events logged yet.")
        return

    text = "*📟 Recent Events*\n" + "─" * 28 + "\n"
    TYPE_ICON = {"HALT": "⛔", "RESUME": "▶️", "OPEN": "📂", "CLOSE": "📁",
                 "ERROR": "🔴", "INFO": "ℹ️", "DEINIT": "🔌", "WARN": "⚠️"}
    for ev in events:
        ts  = str(ev.get("timestamp", ""))[:16]
        typ = ev.get("type", "INFO")
        msg = str(ev.get("message", ""))[:80]
        ico = TYPE_ICON.get(typ, "•")
        # Escape markdown special chars in msg
        msg = msg.replace("_", "\\_").replace("*", "\\*").replace("`", "\\`").replace("[", "\\[")
        text += f"`{ts}` {ico} *{typ}* {msg}\n"

    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN_V2)


@restricted
async def cmd_config(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    cfg = db_get_config()
    alloc = float(cfg.get("allocated_capital", 0))
    text = (
        f"*⚙️ EA Configuration*\n"
        f"{'─'*28}\n"
        f"Allocated Capital: `{'$'+f'{alloc:,.0f}' if alloc > 0 else 'Full balance'}`\n"
        f"Risk % / trade:    `{cfg.get('risk_pct', '?')}%`\n"
        f"Max Drawdown:      `{cfg.get('max_dd_pct', '?')}%`\n"
        f"Max Positions:     `{cfg.get('max_positions', '?')}`\n"
        f"Min Confidence:    `{float(cfg.get('min_confidence', 0))*100:.0f}%`\n"
        f"Halted:            `{cfg.get('halted', False)}`\n"
        f"Paused:            `{cfg.get('paused', False)}`\n"
        f"Updated by:        `{cfg.get('updated_by', '?')}`\n"
        f"Updated at:        `{str(cfg.get('updated_at', 'N/A'))[:16]}`\n\n"
        f"*Quick edit commands:*\n"
        f"`/setcapital 5000` — allocate $5,000\n"
        f"`/setrisk 1.5` — 1\\.5% risk per trade\n"
        f"`/setconf 0.65` — 65% min confidence\n"
        f"`/setmaxdd 15` — halt at 15% drawdown\n"
        f"`/setmaxpos 5` — max 5 open positions\n"
    )
    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN_V2)


# ── Control commands ──────────────────────────────────────────────────

@restricted
async def cmd_resume(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if db_push_config({"halted": False, "paused": False}):
        await update.message.reply_text("▶️ EA *Resumed* — will apply on next config sync.", parse_mode=ParseMode.MARKDOWN_V2)
    else:
        await update.message.reply_text("❌ Failed to update config.")


@restricted
async def cmd_pause(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if db_push_config({"paused": True}):
        await update.message.reply_text("⏸ EA *Paused* — no new trades will open.", parse_mode=ParseMode.MARKDOWN_V2)
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
        "⚠️ *Confirm Emergency Stop?*\nThis will halt the EA immediately\\. Use /resume to restart\\.",
        parse_mode=ParseMode.MARKDOWN_V2,
        reply_markup=keyboard,
    )


@restricted
async def cmd_setcapital(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    try:
        amount = float(ctx.args[0]) if ctx.args else -1
        if amount < 0:
            raise ValueError
    except (ValueError, IndexError):
        await update.message.reply_text("Usage: `/setcapital 5000` (0 = use full balance)", parse_mode=ParseMode.MARKDOWN_V2)
        return
    if db_push_config({"allocated_capital": amount}):
        if amount == 0:
            await update.message.reply_text("✅ Allocated capital cleared — using *full account balance*\\.", parse_mode=ParseMode.MARKDOWN_V2)
        else:
            await update.message.reply_text(f"✅ Allocated capital set to `${amount:,.2f}`\\. EA will size lots based on this amount only\\.", parse_mode=ParseMode.MARKDOWN_V2)
    else:
        await update.message.reply_text("❌ Failed to update config.")


@restricted
async def cmd_setrisk(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    try:
        val = float(ctx.args[0])
        assert 0.1 <= val <= 5.0
    except:
        await update.message.reply_text("Usage: `/setrisk 1.5` (range: 0.1 – 5.0)", parse_mode=ParseMode.MARKDOWN_V2)
        return
    if db_push_config({"risk_pct": val}):
        await update.message.reply_text(f"✅ Risk per trade set to `{val}%`", parse_mode=ParseMode.MARKDOWN_V2)
    else:
        await update.message.reply_text("❌ Failed to update config.")


@restricted
async def cmd_setconf(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    try:
        val = float(ctx.args[0])
        assert 0.40 <= val <= 0.95
    except:
        await update.message.reply_text("Usage: `/setconf 0.65` (range: 0.40 – 0.95)", parse_mode=ParseMode.MARKDOWN_V2)
        return
    if db_push_config({"min_confidence": val}):
        await update.message.reply_text(f"✅ Min AI confidence set to `{val*100:.0f}%`", parse_mode=ParseMode.MARKDOWN_V2)
    else:
        await update.message.reply_text("❌ Failed to update config.")


@restricted
async def cmd_setmaxdd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    try:
        val = float(ctx.args[0])
        assert 5.0 <= val <= 50.0
    except:
        await update.message.reply_text("Usage: `/setmaxdd 15` (range: 5 – 50)", parse_mode=ParseMode.MARKDOWN_V2)
        return
    if db_push_config({"max_dd_pct": val}):
        await update.message.reply_text(f"✅ Max drawdown halt set to `{val}%`", parse_mode=ParseMode.MARKDOWN_V2)
    else:
        await update.message.reply_text("❌ Failed to update config.")


@restricted
async def cmd_setmaxpos(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    try:
        val = int(ctx.args[0])
        assert 1 <= val <= 20
    except:
        await update.message.reply_text("Usage: `/setmaxpos 5` (range: 1 – 20)", parse_mode=ParseMode.MARKDOWN_V2)
        return
    if db_push_config({"max_positions": val}):
        await update.message.reply_text(f"✅ Max positions set to `{val}`", parse_mode=ParseMode.MARKDOWN_V2)
    else:
        await update.message.reply_text("❌ Failed to update config.")


@restricted
async def cmd_help(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    text = (
        "*📖 ApexHydra Bot — Help*\n\n"
        "*Monitoring:*\n"
        "`/status` — Full status with inline controls\n"
        "`/perf` — Performance metrics \\+ per\\-symbol breakdown\n"
        "`/trades` — Last 10 trade entries/exits\n"
        "`/regimes` — Live market regime per symbol\n"
        "`/summary` — Per\\-symbol P\\&L table\n"
        "`/events` — Recent EA event log\n"
        "`/config` — View all current settings\n\n"
        "*Control:*\n"
        "`/resume` — Resume EA trading\n"
        "`/pause` — Pause EA \\(no new trades\\)\n"
        "`/stop` — ⚠️ Emergency halt \\(with confirmation\\)\n\n"
        "*Risk Settings:*\n"
        "`/setcapital <$>` — Allocated capital \\(0 = full balance\\)\n"
        "`/setrisk <pct>` — Risk % per trade \\(0\\.1–5\\.0\\)\n"
        "`/setconf <0.4–0.9>` — Min AI confidence\n"
        "`/setmaxdd <pct>` — Max drawdown halt threshold\n"
        "`/setmaxpos <n>` — Max simultaneous positions\n\n"
        "*Auto\\-Alerts:*\n"
        f"• Drawdown exceeds `{DD_ALERT_PCT}%` or `{DD_CRITICAL_PCT}%`\n"
        f"• EA halted by drawdown limit\n"
        f"• Every trade open/close \\(with P\\&L\\)\n"
        f"• Daily performance summary at 00:00 UTC\n"
    )
    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN_V2)


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
        await query.edit_message_text("▶️ EA *Resumed*\\.", parse_mode=ParseMode.MARKDOWN_V2)

    elif data == "ctrl_pause":
        ok = db_push_config({"paused": True})
        await query.edit_message_text("⏸ EA *Paused*\\.", parse_mode=ParseMode.MARKDOWN_V2)

    elif data == "ctrl_stop":
        keyboard = InlineKeyboardMarkup([[
            InlineKeyboardButton("✅ CONFIRM HALT", callback_data="confirm_halt"),
            InlineKeyboardButton("❌ Cancel",        callback_data="cancel_halt"),
        ]])
        await query.edit_message_text(
            "⚠️ *Confirm Emergency Stop?*",
            parse_mode=ParseMode.MARKDOWN_V2,
            reply_markup=keyboard,
        )

    elif data == "confirm_halt":
        db_push_config({"halted": True, "paused": True})
        await query.edit_message_text("⛔ EA *HALTED*\\. Use /resume to restart\\.", parse_mode=ParseMode.MARKDOWN_V2)

    elif data == "cancel_halt":
        await query.edit_message_text("✅ Halt cancelled\\.", parse_mode=ParseMode.MARKDOWN_V2)

    elif data == "status_refresh":
        # Re-run status inline
        config = db_get_config()
        perf   = db_get_latest_performance()
        balance = float(perf.get("balance", 0))
        equity  = float(perf.get("equity", 0))
        dd      = float(perf.get("drawdown", 0))
        tot_t   = int(perf.get("total_trades", 0))
        wins    = int(perf.get("wins", 0))
        losses  = int(perf.get("losses", 0))
        tot_pnl = float(perf.get("total_pnl", 0))
        wr      = wins / tot_t * 100 if tot_t > 0 else 0
        alloc   = float(config.get("allocated_capital", 0))
        is_halted = config.get("halted", False)
        is_paused = config.get("paused", False)
        status_icon = "⛔ HALTED" if is_halted else ("⏸ PAUSED" if is_paused else "✅ ACTIVE")
        pnl_s = f"\\+${tot_pnl:,.2f}" if tot_pnl >= 0 else f"\\-${abs(tot_pnl):,.2f}"
        text = (
            f"*⚡ ApexHydra Status*  _\\(refreshed\\)_\n"
            f"*Status:* {status_icon}\n"
            f"Balance: `${balance:,.2f}`  Equity: `${equity:,.2f}`\n"
            f"P\\&L: `{pnl_s}`  DD: `{dd:.1f}%`\n"
            f"Trades: `{tot_t}` WR:`{wr:.1f}%`\n"
            f"Capital: `{'$'+f'{alloc:,.0f}' if alloc > 0 else 'Full balance'}`\n"
        )
        keyboard = InlineKeyboardMarkup([
            [
                InlineKeyboardButton("▶ Resume", callback_data="ctrl_resume"),
                InlineKeyboardButton("⏸ Pause",  callback_data="ctrl_pause"),
            ],
            [InlineKeyboardButton("⛔ Emergency Stop", callback_data="ctrl_stop")],
            [InlineKeyboardButton("🔄 Refresh",         callback_data="status_refresh")],
        ])
        await query.edit_message_text(text, parse_mode=ParseMode.MARKDOWN_V2, reply_markup=keyboard)


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

        dd         = float(perf.get("drawdown", 0))
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
                    f"🔴 *CRITICAL DRAWDOWN ALERT*\n"
                    f"Current DD: `{dd:.2f}%` \\(threshold: `{DD_CRITICAL_PCT}%`\\)\n"
                    f"Balance: `${float(perf.get('balance',0)):,.2f}`\n"
                    f"Consider: /stop"
                )
                for cid in chat_ids:
                    await ctx.bot.send_message(cid, msg, parse_mode=ParseMode.MARKDOWN_V2)
                _alert_state["last_dd_alert"]  = now
                _alert_state["dd_alerted_pct"] = dd

        elif dd >= DD_ALERT_PCT:
            if dd_alerted < DD_ALERT_PCT or (last_dd_alert and now - last_dd_alert > cooldown):
                msg = (
                    f"🟠 *Drawdown Warning*\n"
                    f"Current DD: `{dd:.2f}%` \\(alert at `{DD_ALERT_PCT}%`\\)\n"
                    f"Balance: `${float(perf.get('balance',0)):,.2f}`"
                )
                for cid in chat_ids:
                    await ctx.bot.send_message(cid, msg, parse_mode=ParseMode.MARKDOWN_V2)
                _alert_state["last_dd_alert"]  = now
                _alert_state["dd_alerted_pct"] = dd

        else:
            # Reset dd alert state when dd returns to normal
            _alert_state["dd_alerted_pct"] = 0.0

        # ── EA halted alert ───────────────────────────────────────────
        if is_halted and not _alert_state.get("halted_alerted"):
            msg = (
                f"⛔ *EA HALTED*\n"
                f"The EA has been halted\\. Current DD: `{dd:.2f}%`\n"
                f"Use /resume to restart when ready\\."
            )
            for cid in chat_ids:
                await ctx.bot.send_message(cid, msg, parse_mode=ParseMode.MARKDOWN_V2)
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
                        f"📂 *Trade OPENED*\n"
                        f"{reg_e} `{sym}` | Lots: `{lots}` | Conf: `{conf:.0f}%`\n"
                        f"Entry: `{price}` | SL: `{sl}` | TP: `{tp}`\n"
                        f"Regime: _{regime}_ | `{ts_s}`"
                    )
                elif action == "CLOSE" and pnl is not None:
                    pnl_v = float(pnl)
                    icon  = "✅" if pnl_v >= 0 else "❌"
                    msg   = (
                        f"📁 *Trade CLOSED* {icon}\n"
                        f"{reg_e} `{sym}` P\\&L: `{pnl_v:+.2f}`\n"
                        f"Lots: `{lots}` | `{ts_s}`"
                    )
                else:
                    msg = None

                if msg:
                    for cid in chat_ids:
                        await ctx.bot.send_message(cid, msg, parse_mode=ParseMode.MARKDOWN_V2)

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
        dd       = float(perf.get("drawdown", 0))
        tot_t    = int(perf.get("total_trades", 0))
        wins     = int(perf.get("wins", 0))
        losses   = int(perf.get("losses", 0))
        tot_pnl  = float(perf.get("total_pnl", 0))
        wr       = wins / tot_t * 100 if tot_t > 0 else 0
        date_str = datetime.now(timezone.utc).strftime("%Y\\-%m\\-%d")
        pnl_icon = "✅" if tot_pnl >= 0 else "❌"

        text = (
            f"📅 *Daily Summary — {date_str}*\n"
            f"{'─'*28}\n"
            f"Balance:  `${balance:,.2f}` \\| Equity: `${equity:,.2f}`\n"
            f"P\\&L:      {pnl_icon} `{tot_pnl:+,.2f}`\n"
            f"Drawdown: `{dd:.2f}%`\n"
            f"Trades:   `{tot_t}` \\(W:{wins}/L:{losses}\\) WR:`{wr:.1f}%`\n"
        )
        if summary:
            text += "\n*Symbols:*\n"
            for row in summary[:6]:
                sym     = row.get("symbol", "")
                pnl     = float(row.get("total_pnl", 0))
                wr_s    = float(row.get("win_rate_pct", 0))
                icon    = "✅" if pnl >= 0 else "❌"
                text += f"{icon} `{sym:<8}` `{pnl:+,.2f}` WR:`{wr_s:.0f}%`\n"

        for cid in ALLOWED_IDS:
            await ctx.bot.send_message(cid, text, parse_mode=ParseMode.MARKDOWN_V2)

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
