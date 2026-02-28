"""
╔══════════════════════════════════════════════════════════════════════╗
║        ApexHydra Crypto v4.1 — Streamlit Dashboard                  ║
║  Live monitoring + Remote control + Capital Allocation + Telegram   ║
║  NEW: Backtest tab | ML model status | News Filter status            ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from supabase import create_client
from datetime import datetime, timezone, timedelta
import time
import requests
import json

# ──────────────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ApexHydra v4.0",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp { background-color: #f8fafc; color: #1e293b; }
    .metric-card {
        background: #ffffff; border: 1px solid #e2e8f0;
        border-radius: 10px; padding: 16px; margin: 4px 0;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }
    .metric-val-pos { color: #16a34a; font-size: 1.4rem; font-weight: 700; }
    .metric-val-neg { color: #dc2626; font-size: 1.4rem; font-weight: 700; }
    .metric-val-neu { color: #2563eb; font-size: 1.4rem; font-weight: 700; }
    .regime-badge {
        display:inline-block; padding:3px 10px; border-radius:12px;
        font-size:0.75rem; font-weight:600; margin:2px;
    }
    .status-active { background:#dcfce7; color:#16a34a; border:1px solid #16a34a; }
    .status-paused { background:#fef9c3; color:#ca8a04; border:1px solid #ca8a04; }
    .status-halted { background:#fee2e2; color:#dc2626; border:1px solid #dc2626; }
    .capital-box {
        background: #eff6ff; border: 1px solid #93c5fd;
        border-radius: 8px; padding: 12px; margin: 8px 0;
        font-size: 0.85rem; color: #475569;
    }
    .telegram-box {
        background: #f0f9ff; border: 1px solid #7dd3fc;
        border-radius: 8px; padding: 12px; margin: 8px 0;
        font-size: 0.85rem; color: #475569;
    }
    div[data-testid="stSidebarContent"] { background: #ffffff; border-right: 1px solid #e2e8f0; }
    .stTabs [data-baseweb="tab-list"] { background: #f1f5f9; border-radius: 8px; padding: 2px; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────
#  SUPABASE CLIENT
# ──────────────────────────────────────────────────────────────────────
@st.cache_resource
def get_supabase():
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

try:
    supabase = get_supabase()
    DB_OK = True
except Exception as e:
    st.error(f"Supabase connection failed: {e}")
    DB_OK = False

# ──────────────────────────────────────────────────────────────────────
#  TELEGRAM HELPERS
# ──────────────────────────────────────────────────────────────────────
def send_telegram(bot_token: str, chat_id: str, message: str) -> bool:
    if not bot_token or not chat_id:
        return False
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{bot_token}/sendMessage",
            json={"chat_id": chat_id, "text": message, "parse_mode": "HTML"},
            timeout=5
        )
        return r.status_code == 200
    except Exception:
        return False

def tg_notify(msg: str):
    token  = st.secrets.get("TG_BOT_TOKEN", "")
    chatid = st.secrets.get("TG_CHAT_ID",   "")
    if token and chatid:
        send_telegram(token, chatid, msg)

# ──────────────────────────────────────────────────────────────────────
#  DATA FETCHERS
# ──────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=15)
def fetch_config():
    if not DB_OK: return None
    try:
        r = supabase.table("ea_config").select("*").limit(1).execute()
        return r.data[0] if r.data else None
    except: return None

@st.cache_data(ttl=15)
def fetch_performance(hours=720):
    if not DB_OK: return pd.DataFrame()
    try:
        since = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        r = supabase.table("performance").select("*").gte("timestamp", since).order("timestamp").execute()
        return pd.DataFrame(r.data) if r.data else pd.DataFrame()
    except: return pd.DataFrame()

@st.cache_data(ttl=15)
def fetch_trades(limit=200):
    if not DB_OK: return pd.DataFrame()
    try:
        r = supabase.table("trades").select("*").order("timestamp", desc=True).limit(limit).execute()
        return pd.DataFrame(r.data) if r.data else pd.DataFrame()
    except: return pd.DataFrame()

@st.cache_data(ttl=15)
def fetch_regime_changes(limit=50):
    if not DB_OK: return pd.DataFrame()
    try:
        r = supabase.table("regime_changes").select("*").order("timestamp", desc=True).limit(limit).execute()
        return pd.DataFrame(r.data) if r.data else pd.DataFrame()
    except: return pd.DataFrame()

@st.cache_data(ttl=30)
def fetch_events(limit=30):
    if not DB_OK: return pd.DataFrame()
    try:
        r = supabase.table("events").select("*").order("timestamp", desc=True).limit(limit).execute()
        return pd.DataFrame(r.data) if r.data else pd.DataFrame()
    except: return pd.DataFrame()

@st.cache_data(ttl=30)
def fetch_trade_summary():
    if not DB_OK: return pd.DataFrame()
    try:
        r = supabase.table("trade_summary").select("*").execute()
        return pd.DataFrame(r.data) if r.data else pd.DataFrame()
    except: return pd.DataFrame()

@st.cache_data(ttl=30)
def fetch_regime_stats():
    if not DB_OK: return pd.DataFrame()
    try:
        r = supabase.table("regime_stats").select("*").execute()
        return pd.DataFrame(r.data) if r.data else pd.DataFrame()
    except: return pd.DataFrame()

@st.cache_data(ttl=15)
def fetch_current_regimes():
    if not DB_OK: return pd.DataFrame()
    try:
        r = supabase.table("current_regimes").select("*").execute()
        return pd.DataFrame(r.data) if r.data else pd.DataFrame()
    except: return pd.DataFrame()

def update_config(updates: dict) -> bool:
    if not DB_OK: return False
    try:
        updates["updated_by"] = "streamlit"
        supabase.table("ea_config").update(updates).eq("magic", 20250228).execute()
        st.cache_data.clear()
        return True
    except Exception as e:
        st.error(f"Config update failed: {e}")
        return False

# ──────────────────────────────────────────────────────────────────────
#  SIDEBAR
# ──────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ ApexHydra v4.0")
    st.markdown("---")

    cfg = fetch_config()

    if cfg:
        if cfg.get("halted"):
            st.markdown('<span class="regime-badge status-halted">⛔ HALTED</span>', unsafe_allow_html=True)
        elif cfg.get("paused"):
            st.markdown('<span class="regime-badge status-paused">⏸ PAUSED</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="regime-badge status-active">▶ ACTIVE</span>', unsafe_allow_html=True)

    st.markdown("### 🎛 EA Control")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶ Resume", use_container_width=True, type="primary"):
            if update_config({"paused": False, "halted": False}):
                tg_notify("▶️ <b>ApexHydra RESUMED</b> via Streamlit dashboard")
                st.success("Resumed!")
                st.rerun()
    with col2:
        if st.button("⏸ Pause", use_container_width=True):
            if update_config({"paused": True}):
                tg_notify("⏸ <b>ApexHydra PAUSED</b> via Streamlit dashboard")
                st.warning("Paused!")
                st.rerun()

    if st.button("⛔ Emergency Stop", use_container_width=True, type="secondary"):
        if update_config({"halted": True, "paused": True}):
            tg_notify("🚨 <b>EMERGENCY STOP</b> triggered via Streamlit!")
            st.error("EA Halted!")
            st.rerun()

    # ── CAPITAL ALLOCATION
    st.markdown("---")
    st.markdown("### 💰 Capital Allocation")
    st.markdown(
        '<div class="capital-box">'
        'Set a <b>capital budget</b> the EA risks against instead of the full account balance. '
        'The AI position sizer will use whichever is smaller: this budget or the actual balance. '
        'Set to <b>0</b> to use full balance (default behaviour).'
        '</div>',
        unsafe_allow_html=True
    )

    current_cap = float(cfg.get("trading_capital", 0.0)) if cfg else 0.0
    current_cpct = int(cfg.get("capital_pct", 100)) if cfg else 100

    cap_val = st.number_input(
        "Trading Capital ($)",
        min_value=0.0, max_value=1_000_000.0,
        value=current_cap, step=100.0, format="%.2f",
        help="0 = use full MT5 account balance",
    )
    cap_pct = st.slider(
        "Usable Capital (%)", min_value=10, max_value=100,
        value=current_cpct,
        help="% of the capital budget that can be actively in positions",
    )

    eff_cap = cap_val * (cap_pct / 100) if cap_val > 0 else 0
    if cap_val > 0:
        st.caption(f"Effective risk base: **${eff_cap:,.2f}**")
    else:
        st.caption("Risk base: full MT5 balance")

    if st.button("💾 Apply Capital", use_container_width=True):
        if update_config({"trading_capital": cap_val, "capital_pct": cap_pct}):
            tg_notify(
                f"💰 <b>Capital Updated</b>\n"
                f"Budget: ${cap_val:,.2f} @ {cap_pct}%  →  Effective: ${eff_cap:,.2f}"
            )
            st.success(f"Capital set: ${cap_val:,.2f} ({cap_pct}%)")
            st.rerun()

    st.markdown("---")
    # ── MODAL AI + ML MODEL STATUS ──────────────────────────────────
    st.markdown("### 🤖 Modal AI")
    modal_url = st.secrets.get("MODAL_URL", "")
    if modal_url:
        try:
            r = requests.get(f"{modal_url.rstrip('/')}/health", timeout=5)
            if r.status_code == 200:
                h = r.json()
                ml = h.get("ml_model", {})
                st.markdown(
                    f'<div class="capital-box">'
                    f'🟢 Modal OK — v{h.get("version","?")}<br>'
                    f'ML Model: {"✅ Loaded" if ml.get("loaded") else "⚠️ Not trained yet"}'
                    + (f'<br>Accuracy: <b>{ml["accuracy"]:.1%}</b> | Samples: {ml["samples"]}' if ml.get("accuracy") else "")
                    + f'<br>News filter: ✅ enabled</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown('<div class="capital-box">🟡 Modal: returned non-200</div>', unsafe_allow_html=True)
        except Exception:
            st.markdown('<div class="capital-box">🔴 Modal: unreachable</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="capital-box">Add <code>MODAL_URL</code> to secrets.toml<br><small>e.g. <code>https://YOUR_WORKSPACE--apex-hydracrypto-apexhydra.modal.run</code></small></div>',
            unsafe_allow_html=True
        )

    # ── NEWS FILTER STATUS ───────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📰 News Filter")
    news_mins = st.number_input("Block N minutes around events", min_value=5, max_value=60, value=15, step=5,
                                 help="MT5 EA uses this value as news_buffer_minutes in each request")
    st.caption("Set `Inp_News_Buffer_Min` in EA inputs to match. EA must also enable `Inp_News_Filter = true`.")
    st.markdown(
        '<div class="capital-box">📡 EA sends <code>news_minutes_away</code> to Modal.<br>'
        'Modal auto-blocks if event is within the buffer window.</div>',
        unsafe_allow_html=True
    )

    # ── RISK SETTINGS
    st.markdown("---")
    st.markdown("### ⚙️ Risk Settings")
    if cfg:
        new_risk   = st.number_input("Risk % per trade",    min_value=0.1, max_value=10.0, value=float(cfg.get("risk_pct",1.0)),     step=0.1)
        new_dd     = st.number_input("Max Drawdown %",      min_value=1.0, max_value=50.0, value=float(cfg.get("max_dd_pct",20.0)),   step=1.0)
        new_maxpos = st.number_input("Max Open Positions",  min_value=1,   max_value=20,   value=int(cfg.get("max_positions",3)),     step=1)
        new_conf   = st.number_input("Min Confidence",      min_value=0.30, max_value=0.95, value=float(cfg.get("min_confidence",0.55)), step=0.01)

        if st.button("💾 Apply Risk Settings", use_container_width=True):
            upd = {"risk_pct": new_risk, "max_dd_pct": new_dd, "max_positions": new_maxpos, "min_confidence": new_conf}
            if update_config(upd):
                tg_notify(f"⚙️ <b>Risk Settings Updated</b>\nRisk:{new_risk}% MaxDD:{new_dd}% MaxPos:{new_maxpos} MinConf:{new_conf:.2f}")
                st.success("Applied!")
                st.rerun()

    # ── TELEGRAM STATUS
    st.markdown("---")
    st.markdown("### 📱 Telegram")
    tg_token  = st.secrets.get("TG_BOT_TOKEN", "")
    tg_chatid = st.secrets.get("TG_CHAT_ID", "")
    if tg_token and tg_chatid:
        st.markdown(
            f'<div class="telegram-box">'
            f'Status: 🟢 Connected<br>'
            f'Chat ID: <code>{tg_chatid}</code>'
            f'</div>',
            unsafe_allow_html=True
        )
    else:
        missing = []
        if not tg_token:  missing.append("<code>TG_BOT_TOKEN</code>")
        if not tg_chatid: missing.append("<code>TG_CHAT_ID</code>")
        st.markdown(
            f'<div class="telegram-box">'
            f'Status: 🔴 Not configured<br>'
            f'Add {" and ".join(missing)} to secrets.toml'
            f'</div>',
            unsafe_allow_html=True
        )
    if tg_token:
        if st.button("📤 Send Test Message"):
            ok = send_telegram(tg_token, st.secrets.get("TG_CHAT_ID",""),
                               "🤖 <b>ApexHydra Test</b>\nDashboard OK ✅")
            st.success("Sent!") if ok else st.error("Failed")

    st.markdown("---")
    auto_refresh = st.checkbox("Auto-refresh (15s)", value=True)
    if st.button("↺ Refresh Now"):
        st.cache_data.clear()
        st.rerun()

# ──────────────────────────────────────────────────────────────────────
#  MAIN — HEADER + KPIs
# ──────────────────────────────────────────────────────────────────────
st.markdown("# ⚡ ApexHydra Crypto v4.0")
st.markdown(f"*{datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}*")
st.markdown("---")

perf_df   = fetch_performance(hours=720)
trades_df = fetch_trades()
cfg_data  = fetch_config()

kpi_cols = st.columns(6)

def kpi(col, label, value, color_class="metric-val-neu"):
    col.markdown(
        f'<div class="metric-card">'
        f'<div style="font-size:0.7rem;color:#64748b;">{label}</div>'
        f'<div class="{color_class}">{value}</div>'
        f'</div>',
        unsafe_allow_html=True
    )

def _render_kpis(balance, equity, dd, tot, wins, total_pnl):
    wr = (wins / tot * 100) if tot > 0 else 0
    kpi(kpi_cols[0], "Balance",   f"${balance:,.2f}")
    kpi(kpi_cols[1], "Equity",    f"${equity:,.2f}")
    kpi(kpi_cols[2], "Drawdown",  f"{dd:.2f}%",
        "metric-val-neg" if dd > 10 else ("metric-val-neu" if dd > 5 else "metric-val-pos"))
    kpi(kpi_cols[3], "Win Rate",  f"{wr:.1f}%",
        "metric-val-pos" if wr >= 50 else "metric-val-neg")
    kpi(kpi_cols[4], "Total PnL", f"${total_pnl:+,.2f}",
        "metric-val-pos" if total_pnl >= 0 else "metric-val-neg")
    kpi(kpi_cols[5], "Trades",    str(tot))
    return balance

if not perf_df.empty:
    latest    = perf_df.iloc[-1]
    balance = _render_kpis(
        float(latest.get("balance", 0)),
        float(latest.get("equity",  0)),
        float(latest.get("drawdown", 0)) * 100,   # stored as fraction in perf table
        int(latest.get("total_trades", 0)),
        int(latest.get("wins", 0)),
        float(latest.get("total_pnl", 0)),
    )

elif cfg_data and cfg_data.get("live_balance"):
    # ── Fallback: read live_* fields the EA PATCHes into ea_config every 15s ──
    live_ts = cfg_data.get("live_ts", "")
    balance = _render_kpis(
        float(cfg_data.get("live_balance", 0)),
        float(cfg_data.get("live_equity",  0)),
        float(cfg_data.get("live_dd_pct",  0)),   # already in % — EA sends g_dd_pct
        int(cfg_data.get("live_trades", 0) or 0),
        int(cfg_data.get("live_wins",   0) or 0),
        float(cfg_data.get("live_pnl",  0) or 0),
    )
    st.caption(f"⚡ Live snapshot from EA — last update: {live_ts}")

else:
    balance = 0
    for c in kpi_cols:
        kpi(c, "—", "N/A")

# Capital budget banner (shown in all branches that have data)
if cfg_data:
    tcap = float(cfg_data.get("trading_capital", 0) or 0)
    if tcap > 0:
        cpct = float(cfg_data.get("capital_pct", 100) or 100)
        eff  = tcap * (cpct / 100)
        bal_str = f"${balance:,.2f}" if balance > 0 else "full balance"
        st.info(
            f"💰 **Capital Budget**: ${tcap:,.2f} × {cpct:.0f}% = "
            f"**Effective ${eff:,.2f}**"
            + (f" (Full balance: ${balance:,.2f})" if balance > 0 else "")
        )

st.markdown("---")

# ──────────────────────────────────────────────────────────────────────
#  TABS
# ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📈 Equity Curve", "🎯 Regime Map", "🤖 AI Performance",
    "📋 Trade History", "📡 Live Feed", "📱 Telegram", "🧪 Backtest",
])

# ── TAB 1: EQUITY CURVE
with tab1:
    if not perf_df.empty:
        perf_df["timestamp"] = pd.to_datetime(perf_df["timestamp"])
        perf_df = perf_df.sort_values("timestamp")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=perf_df["timestamp"], y=perf_df["equity"],
            name="Equity", line=dict(color="#58a6ff", width=2),
            fill="tozeroy", fillcolor="rgba(88,166,255,0.07)",
        ))
        if "balance" in perf_df.columns:
            fig.add_trace(go.Scatter(
                x=perf_df["timestamp"], y=perf_df["balance"],
                name="Balance", line=dict(color="#16a34a", width=1.5, dash="dot"),
            ))
        fig.update_layout(
            template="plotly_white", paper_bgcolor="#f8fafc", plot_bgcolor="#ffffff",
            title="Equity & Balance", height=350, margin=dict(l=0,r=0,t=40,b=0),
        )
        st.plotly_chart(fig, use_container_width=True)

        if "drawdown" in perf_df.columns:
            fig2 = go.Figure(go.Scatter(
                x=perf_df["timestamp"], y=perf_df["drawdown"] * 100,
                fill="tozeroy", fillcolor="rgba(248,81,73,0.2)",
                line=dict(color="#dc2626", width=1.5), name="DD%",
            ))
            fig2.update_layout(
                template="plotly_white", paper_bgcolor="#f8fafc", plot_bgcolor="#ffffff",
                title="Drawdown %", height=200, margin=dict(l=0,r=0,t=40,b=0),
            )
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No performance data yet.")

# ── TAB 2: REGIME MAP
with tab2:
    c1, c2 = st.columns(2)
    REGIME_COLORS = {
        "Trend Bull": "#16a34a", "Trend Bear": "#dc2626",
        "Ranging": "#e3b341", "High Volatility": "#ff7c43",
        "Breakout": "#a371f7", "Undefined": "#8b949e",
    }

    with c1:
        st.subheader("Current Regimes")
        regimes_df = fetch_current_regimes()
        if not regimes_df.empty:
            for _, row in regimes_df.iterrows():
                r    = row.get("regime", "Undefined")
                col  = REGIME_COLORS.get(r, "#8b949e")
                conf = float(row.get("confidence", 0)) * 100
                ts   = pd.to_datetime(row["timestamp"]).strftime("%H:%M") if row.get("timestamp") else "—"
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:10px;padding:8px;'
                    f'background:#f8fafc;border-radius:6px;margin:3px 0;border-left:3px solid {col};">'
                    f'<b style="color:#1e293b;width:80px;">{row["symbol"]}</b>'
                    f'<span style="color:{col};font-weight:600;width:120px;">{r}</span>'
                    f'<span style="color:#64748b;font-size:0.78rem;">'
                    f'Conf:{conf:.0f}% ADX:{row.get("adx","—")} RSI:{row.get("rsi","—")} {ts}</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )
        else:
            st.info("No live regime data.")

    with c2:
        st.subheader("Regime Performance")
        rs = fetch_regime_stats()
        if not rs.empty:
            fig = px.bar(rs, x="regime", y="win_rate_pct", color="win_rate_pct",
                         color_continuous_scale="RdYlGn", text="win_rate_pct",
                         title="Win Rate % by Regime")
            fig.update_layout(template="plotly_white", paper_bgcolor="#f8fafc",
                              plot_bgcolor="#ffffff", height=280,
                              margin=dict(l=0,r=0,t=40,b=0), coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

            fig2 = px.bar(rs, x="regime", y="total_pnl", color="total_pnl",
                          color_continuous_scale="RdYlGn", text="total_pnl",
                          title="PnL by Regime")
            fig2.update_layout(template="plotly_white", paper_bgcolor="#f8fafc",
                               plot_bgcolor="#ffffff", height=250,
                               margin=dict(l=0,r=0,t=40,b=0), coloraxis_showscale=False)
            st.plotly_chart(fig2, use_container_width=True)

# ── TAB 3: AI PERFORMANCE
with tab3:
    ca, cb = st.columns([2,1])
    with ca:
        st.subheader("Per-Symbol Performance")
        ts = fetch_trade_summary()
        if not ts.empty:
            st.dataframe(
                ts.style.background_gradient(subset=["win_rate_pct"], cmap="RdYlGn")
                  .format({"win_rate_pct":"{:.1f}%","total_pnl":"${:.2f}",
                           "avg_ai_score":"{:.3f}","avg_confidence":"{:.3f}"}),
                use_container_width=True, hide_index=True,
            )
        else:
            st.info("No data yet.")
    with cb:
        st.subheader("Win Rate Trend")
        if not perf_df.empty and "global_accuracy" in perf_df.columns:
            fig = go.Figure(go.Scatter(
                x=perf_df["timestamp"], y=perf_df["global_accuracy"]*100,
                line=dict(color="#a371f7", width=2),
                fill="tozeroy", fillcolor="rgba(163,113,247,0.1)",
            ))
            fig.add_hline(y=50, line_dash="dot", line_color="#94a3b8")
            fig.update_layout(template="plotly_white", paper_bgcolor="#f8fafc",
                              plot_bgcolor="#ffffff", title="Global Win Rate %",
                              height=300, margin=dict(l=0,r=0,t=40,b=0))
            st.plotly_chart(fig, use_container_width=True)

# ── TAB 4: TRADE HISTORY
with tab4:
    st.subheader("Trade Log")
    if not trades_df.empty:
        trades_df["timestamp"] = pd.to_datetime(trades_df["timestamp"])
        cols = ["timestamp","symbol","action","regime","signal","confidence","lots","price","sl","tp","pnl","won"]
        avail = [c for c in cols if c in trades_df.columns]

        def color_pnl(val):
            if pd.isna(val) or val == 0: return "color:#64748b"
            return "color:#2dba4e" if val > 0 else "color:#f85149"

        styled = trades_df[avail].style
        if "pnl" in avail:
            styled = styled.map(color_pnl, subset=["pnl"])
        st.dataframe(styled, use_container_width=True, hide_index=True)

        close_t = trades_df[trades_df.get("action","") == "CLOSE"].copy() if "action" in trades_df.columns else pd.DataFrame()
        if not close_t.empty and "pnl" in close_t.columns:
            close_t["pnl"] = pd.to_numeric(close_t["pnl"], errors="coerce")
            fig = px.histogram(close_t, x="pnl", nbins=30, color_discrete_sequence=["#58a6ff"],
                               title="PnL Distribution")
            fig.add_vline(x=0, line_dash="dash", line_color="#dc2626")
            fig.update_layout(template="plotly_white", paper_bgcolor="#f8fafc",
                              plot_bgcolor="#ffffff", height=250, margin=dict(l=0,r=0,t=40,b=0))
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No trades yet.")

# ── TAB 5: LIVE FEED
with tab5:
    ce1, ce2 = st.columns(2)
    with ce1:
        st.subheader("📡 Event Log")
        ev = fetch_events()
        if not ev.empty:
            ICONS = {"HALT":"⛔","RESUME":"▶️","OPEN":"📈","CLOSE":"📉","ERROR":"❌","INFO":"ℹ️","DEINIT":"🔌"}
            for _, row in ev.iterrows():
                t    = row.get("type","INFO")
                icon = ICONS.get(t,"•")
                ts   = pd.to_datetime(row["timestamp"]).strftime("%H:%M:%S") if row.get("timestamp") else "—"
                col  = "#dc2626" if t in ("HALT","ERROR") else ("#16a34a" if t in ("RESUME","OPEN") else "#8b949e")
                st.markdown(
                    f'<div style="font-size:0.8rem;padding:4px 8px;border-left:3px solid {col};margin:2px 0;">'
                    f'<span style="color:#64748b;">{ts}</span> {icon} '
                    f'<span style="color:{col};">[{t}]</span> {row.get("message","")}</div>',
                    unsafe_allow_html=True
                )
    with ce2:
        st.subheader("🔄 Recent Regime Changes")
        rc = fetch_regime_changes(limit=20)
        if not rc.empty:
            rc["timestamp"] = pd.to_datetime(rc["timestamp"])
            st.dataframe(rc[["timestamp","symbol","regime","confidence","adx","rsi"]].head(20),
                         use_container_width=True, hide_index=True)

# ── TAB 6: TELEGRAM
with tab6:
    st.subheader("📱 Telegram Bot Setup & Alerts")
    st.markdown("""
**Quick Setup (3 steps):**

1. Message **@BotFather** on Telegram → `/newbot` → copy your **Bot Token**
2. Message your bot to start a chat, then get your **Chat ID** from **@userinfobot**
3. Add to `.streamlit/secrets.toml`:

```toml
TG_BOT_TOKEN = "123456789:ABCDef_your_token_here"
TG_CHAT_ID   = "123456789"          # or "-100xxxxxxx" for groups
```

**Alerts fired automatically:**
""")

    alert_tbl = pd.DataFrame({
        "Event": [
            "Trade Open","Trade Close (Win)","Trade Close (Loss)",
            "EA Halted (DD limit)","Emergency Stop","EA Paused","EA Resumed",
            "Risk Settings Changed","Capital Budget Changed",
        ],
        "Emoji": ["📈","✅","❌","🚨","⛔","⏸","▶️","⚙️","💰"],
        "Includes": [
            "Symbol, direction, lots, price, regime, confidence",
            "Symbol, PnL, cumulative win rate",
            "Symbol, PnL, regime",
            "DD %, threshold, balance",
            "Triggered by whom",
            "Timestamp",
            "Timestamp",
            "Risk%, MaxDD%, MaxPos, MinConf",
            "Budget $, %, effective capital",
        ],
    })
    st.table(alert_tbl)

    st.markdown("---")
    st.markdown("### 📤 Manual Broadcast")
    msg_input = st.text_area("Message:", placeholder="Enter message to send…")
    if st.button("Send"):
        tok = st.secrets.get("TG_BOT_TOKEN","")
        cid = st.secrets.get("TG_CHAT_ID","")
        if not tok:
            st.error("TG_BOT_TOKEN not set.")
        else:
            ok = send_telegram(tok, cid, msg_input)
            st.success("Message sent!") if ok else st.error("Failed — check credentials.")

    st.markdown("---")
    st.markdown("### 🔕 Alert Preferences")
    st.caption("These are saved in ea_config.tg_alerts (requires schema addition — see notes below)")
    a1, a2 = st.columns(2)
    with a1:
        on_open   = st.checkbox("Alert on Trade Open",    value=True)
        on_win    = st.checkbox("Alert on Win",           value=True)
        on_loss   = st.checkbox("Alert on Loss",          value=True)
    with a2:
        on_halt   = st.checkbox("Alert on EA Halt",       value=True)
        on_resume = st.checkbox("Alert on EA Resume",     value=False)
        on_cfg    = st.checkbox("Alert on Config Change", value=False)

    if st.button("💾 Save Alert Prefs"):
        prefs = json.dumps({
            "on_open":on_open,"on_win":on_win,"on_loss":on_loss,
            "on_halt":on_halt,"on_resume":on_resume,"on_cfg":on_cfg
        })
        if update_config({"tg_alerts": prefs}):
            st.success("Saved!")

    st.info(
        "💡 **Schema note**: To persist per-alert toggles, add this column to ea_config:\n"
        "```sql\nALTER TABLE ea_config ADD COLUMN IF NOT EXISTS tg_alerts JSONB DEFAULT '{}'::jsonb;\n"
        "ALTER TABLE ea_config ADD COLUMN IF NOT EXISTS trading_capital DECIMAL(16,2) DEFAULT 0;\n"
        "ALTER TABLE ea_config ADD COLUMN IF NOT EXISTS capital_pct INTEGER DEFAULT 100;\n```"
    )


# ── TAB 7: BACKTEST (NEW) ─────────────────────────────────────────────
with tab7:
    st.subheader("🧪 Strategy Backtester")
    st.markdown(
        "Upload a CSV of historical bars with pre-computed indicators and run the "
        "ApexHydra strategy on them. The same logic used live is applied here."
    )

    MODAL_URL = st.secrets.get("MODAL_URL", "")

    col_bt1, col_bt2 = st.columns([2, 1])
    with col_bt2:
        st.markdown("#### Settings")
        bt_symbol    = st.text_input("Symbol",         value="BTCUSD")
        bt_tf        = st.text_input("Timeframe",      value="H1")
        bt_balance   = st.number_input("Initial Balance ($)", value=10000.0, step=500.0)
        bt_risk      = st.number_input("Risk % per trade",    value=1.0, min_value=0.1, max_value=10.0, step=0.1)
        bt_min_rr    = st.number_input("Min R:R",             value=1.5, min_value=1.0, max_value=5.0,  step=0.1)
        bt_min_conf  = st.number_input("Min Confidence",      value=0.52, min_value=0.30, max_value=0.95, step=0.01)
        bt_spread    = st.number_input("Spread (points)",     value=20.0, min_value=0.0, max_value=500.0)
        bt_tick_val  = st.number_input("Tick Value",          value=1.0,  step=0.1)
        bt_tick_size = st.number_input("Tick Size",           value=0.01, step=0.01, format="%.4f")

    with col_bt1:
        st.markdown("#### Upload Bar Data (CSV)")
        st.markdown(
            "**Required columns**: `timestamp, open, high, low, close, volume, "
            "atr, atr_avg, adx, plus_di, minus_di, rsi, macd, macd_signal, macd_hist, "
            "ema20, ema50, ema200, htf_ema50, htf_ema200`"
        )
        st.caption("Export from MT5 using the companion MQL5 script (see repo). Minimum 250 rows recommended.")

        uploaded = st.file_uploader("Upload CSV", type=["csv"], key="backtest_csv")

        if uploaded and MODAL_URL:
            try:
                df_bt = pd.read_csv(uploaded)
                st.success(f"Loaded {len(df_bt)} rows. Preview:")
                st.dataframe(df_bt.head(5), use_container_width=True, hide_index=True)

                REQUIRED = ["timestamp","open","high","low","close","volume","atr","atr_avg",
                            "adx","plus_di","minus_di","rsi","macd","macd_signal","macd_hist",
                            "ema20","ema50","ema200","htf_ema50","htf_ema200"]
                missing = [c for c in REQUIRED if c not in df_bt.columns]
                if missing:
                    st.error(f"Missing columns: {missing}")
                else:
                    if st.button("▶ Run Backtest", type="primary"):
                        with st.spinner("Running backtest on Modal…"):
                            bars_payload = df_bt[REQUIRED].fillna(0).to_dict(orient="records")
                            payload = {
                                "symbol":          bt_symbol,
                                "timeframe":       bt_tf,
                                "bars":            bars_payload,
                                "initial_balance": bt_balance,
                                "risk_pct":        bt_risk,
                                "min_rr":          bt_min_rr,
                                "min_confidence":  bt_min_conf,
                                "spread_points":   bt_spread,
                                "tick_value":      bt_tick_val,
                                "tick_size":       bt_tick_size,
                                "min_lot":         0.01,
                                "max_lot":         100.0,
                                "lot_step":        0.01,
                                "point":           bt_tick_size,
                                "digits":          2,
                            }
                            r = requests.post(
                                f"{MODAL_URL.rstrip('/')}/backtest",
                                json=payload, timeout=120
                            )
                            if r.status_code == 200:
                                res = r.json()
                                st.session_state["bt_result"] = res
                            else:
                                st.error(f"Backtest failed: {r.status_code} — {r.text[:300]}")
            except Exception as e:
                st.error(f"Error: {e}")
        elif uploaded and not MODAL_URL:
            st.warning("Add `MODAL_URL` to secrets.toml to run backtest via Modal.")
        elif not uploaded:
            st.info("Upload a CSV to begin. You can export bar data from MT5 using the companion script in the repo.")

    # ── Backtest Results ──────────────────────────────────────────────
    if "bt_result" in st.session_state:
        res = st.session_state["bt_result"]
        st.markdown("---")
        st.markdown("### 📊 Backtest Results")

        m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
        def bt_kpi(col, label, val, good=None):
            color = "#58a6ff"
            if good is not None:
                color = "#16a34a" if good else "#dc2626"
            col.markdown(
                f'<div class="metric-card"><div style="font-size:0.65rem;color:#64748b;">{label}</div>'
                f'<div style="color:{color};font-size:1.2rem;font-weight:700;">{val}</div></div>',
                unsafe_allow_html=True
            )

        bt_kpi(m1, "Trades",     res["total_trades"])
        bt_kpi(m2, "Win Rate",   f"{res['win_rate']*100:.1f}%",        good=res['win_rate'] >= 0.5)
        bt_kpi(m3, "Total PnL",  f"${res['total_pnl']:+,.2f}",         good=res['total_pnl'] >= 0)
        bt_kpi(m4, "Max DD",     f"{res['max_drawdown_pct']:.1f}%",    good=res['max_drawdown_pct'] <= 15)
        bt_kpi(m5, "Sharpe",     f"{res['sharpe_ratio']:.2f}",         good=res['sharpe_ratio'] >= 1.0)
        bt_kpi(m6, "Prof.Factor",f"{res['profit_factor']:.2f}",        good=res['profit_factor'] >= 1.5)
        bt_kpi(m7, "Final Bal.", f"${res['final_balance']:,.2f}")

        # Equity curve
        if res.get("equity_curve"):
            fig_eq = go.Figure(go.Scatter(
                y=res["equity_curve"], mode="lines",
                line=dict(color="#58a6ff", width=2),
                fill="tozeroy", fillcolor="rgba(88,166,255,0.07)",
                name="Balance",
            ))
            fig_eq.add_hline(y=res.get("initial_balance", res["equity_curve"][0]),
                             line_dash="dot", line_color="#94a3b8",
                             annotation_text="Starting balance")
            fig_eq.update_layout(
                template="plotly_white", paper_bgcolor="#f8fafc", plot_bgcolor="#ffffff",
                title="Equity Curve (Backtest)", height=320, margin=dict(l=0,r=0,t=40,b=0),
            )
            st.plotly_chart(fig_eq, use_container_width=True)

        rc1, rc2 = st.columns(2)
        # Regime breakdown
        with rc1:
            if res.get("regime_breakdown"):
                st.markdown("**Performance by Regime**")
                rd = pd.DataFrame(res["regime_breakdown"]).T.reset_index()
                rd.columns = ["Regime","Trades","Wins","Win Rate %","Total PnL"]
                fig_r = px.bar(rd, x="Regime", y="Win Rate %", color="Win Rate %",
                               color_continuous_scale="RdYlGn", text="Win Rate %",
                               title="Win Rate by Regime (Backtest)")
                fig_r.update_layout(template="plotly_white", paper_bgcolor="#f8fafc",
                                    plot_bgcolor="#ffffff", height=280,
                                    margin=dict(l=0,r=0,t=40,b=0), coloraxis_showscale=False)
                st.plotly_chart(fig_r, use_container_width=True)

        # PnL histogram
        with rc2:
            if res.get("trades"):
                trades_bt = pd.DataFrame([t for t in res["trades"]])
                fig_h = px.histogram(trades_bt, x="pnl", nbins=25,
                                     color_discrete_sequence=["#a371f7"],
                                     title="PnL Distribution (Backtest)")
                fig_h.add_vline(x=0, line_dash="dash", line_color="#dc2626")
                fig_h.update_layout(template="plotly_white", paper_bgcolor="#f8fafc",
                                    plot_bgcolor="#ffffff", height=280,
                                    margin=dict(l=0,r=0,t=40,b=0))
                st.plotly_chart(fig_h, use_container_width=True)

        # Trade list
        if res.get("trades"):
            st.markdown("**Trade Log**")
            bt_df = pd.DataFrame(res["trades"])
            st.dataframe(bt_df, use_container_width=True, hide_index=True)

            # Download
            csv_out = bt_df.to_csv(index=False)
            st.download_button("📥 Download Trade Log (CSV)", csv_out,
                               file_name=f"backtest_{bt_symbol}_{bt_tf}.csv", mime="text/csv")

# ── AUTO-REFRESH
if auto_refresh:
    time.sleep(15)
    st.rerun()