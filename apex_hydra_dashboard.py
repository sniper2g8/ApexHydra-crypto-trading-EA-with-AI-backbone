"""
╔══════════════════════════════════════════════════════════════════════╗
║         ApexHydra Crypto — Streamlit Live Dashboard                 ║
║  Monitors the EA running on VPS via Supabase real-time data          ║
║  Also sends control commands back to EA via ea_config table          ║
║                                                                      ║
║  Run:  streamlit run apex_hydra_dashboard.py                         ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from supabase import create_client, Client
from datetime import datetime, timezone, timedelta
import time, os, json

# ──────────────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="ApexHydra Crypto",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────
#  CUSTOM CSS — Dark Trading Terminal Theme
# ──────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&display=swap');

  html, body, [class*="css"] {
    background-color: #0d1117 !important;
    color: #e6edf3;
    font-family: 'JetBrains Mono', monospace;
  }
  .main { background-color: #0d1117; }
  section[data-testid="stSidebar"] { background-color: #161b22 !important; border-right: 1px solid #30363d; }

  /* Metric cards */
  div[data-testid="metric-container"] {
    background: linear-gradient(135deg, #161b22, #1c2128);
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 12px 16px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.4);
  }
  div[data-testid="metric-container"] label { color: #8b949e !important; font-size: 11px; letter-spacing: 1px; text-transform: uppercase; }
  div[data-testid="metric-container"] div[data-testid="metric-value"] { font-size: 24px !important; font-weight: 700; }

  /* Status badges */
  .badge {
    display: inline-block; padding: 3px 10px; border-radius: 20px;
    font-size: 11px; font-weight: 600; letter-spacing: 0.5px;
  }
  .badge-active  { background: rgba(0,230,118,0.15); color: #00e676; border: 1px solid #00e676; }
  .badge-halted  { background: rgba(255,82,82,0.15);  color: #ff5252; border: 1px solid #ff5252; }
  .badge-warn    { background: rgba(255,193,7,0.15);  color: #ffc107; border: 1px solid #ffc107; }

  /* Section headers */
  .section-header {
    font-size: 13px; font-weight: 700; letter-spacing: 2px;
    text-transform: uppercase; color: #00bcd4; margin: 16px 0 8px 0;
    border-bottom: 1px solid #21262d; padding-bottom: 6px;
  }

  /* Tables */
  div[data-testid="stDataFrame"] { border: 1px solid #30363d; border-radius: 8px; overflow: hidden; }

  /* Control buttons */
  .stButton > button {
    width: 100%;
    background: #1c2128 !important;
    border: 1px solid #30363d !important;
    color: #e6edf3 !important;
    border-radius: 8px;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    transition: all 0.2s;
  }
  .stButton > button:hover { border-color: #00bcd4 !important; color: #00bcd4 !important; }

  /* Log panel */
  .log-panel {
    background: #0d1117; border: 1px solid #21262d; border-radius: 8px;
    padding: 12px; font-size: 11px; height: 280px; overflow-y: auto;
    font-family: 'JetBrains Mono', monospace;
  }
  .log-line { padding: 2px 0; border-bottom: 1px solid #161b22; }
  .log-win  { color: #00e676; }
  .log-loss { color: #ff5252; }
  .log-info { color: #8b949e; }
  .log-warn { color: #ffc107; }

  /* Regime color pills */
  .regime-trend-bull  { color: #00e676; }
  .regime-trend-bear  { color: #ff5252; }
  .regime-ranging     { color: #ffc107; }
  .regime-high-vol    { color: #ff9800; }
  .regime-breakout    { color: #9c27b0; }

  /* Hide Streamlit branding */
  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding-top: 1rem; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────
#  SUPABASE CONNECTION
# ──────────────────────────────────────────────────────────────────────

@st.cache_resource
def get_supabase() -> Client:
    url = st.secrets.get("SUPABASE_URL", os.getenv("SUPABASE_URL", ""))
    key = st.secrets.get("SUPABASE_KEY", os.getenv("SUPABASE_KEY", ""))
    if not url or not key:
        st.error("Set SUPABASE_URL and SUPABASE_KEY in .streamlit/secrets.toml or env vars.")
        st.stop()
    return create_client(url, key)


# ──────────────────────────────────────────────────────────────────────
#  DATA FETCHERS
# ──────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=10)
def fetch_performance(_sb):
    try:
        r = _sb.table("performance").select("*").order("timestamp", desc=True).limit(500).execute()
        return pd.DataFrame(r.data) if r.data else pd.DataFrame()
    except: return pd.DataFrame()

@st.cache_data(ttl=10)
def fetch_trades(_sb, limit=200):
    try:
        r = _sb.table("trades").select("*").order("timestamp", desc=True).limit(limit).execute()
        return pd.DataFrame(r.data) if r.data else pd.DataFrame()
    except: return pd.DataFrame()

@st.cache_data(ttl=10)
def fetch_regime_changes(_sb, limit=100):
    try:
        r = _sb.table("regime_changes").select("*").order("timestamp", desc=True).limit(limit).execute()
        return pd.DataFrame(r.data) if r.data else pd.DataFrame()
    except: return pd.DataFrame()

@st.cache_data(ttl=10)
def fetch_events(_sb, limit=50):
    try:
        r = _sb.table("events").select("*").order("timestamp", desc=True).limit(limit).execute()
        return pd.DataFrame(r.data) if r.data else pd.DataFrame()
    except: return pd.DataFrame()

@st.cache_data(ttl=5)
def fetch_config(_sb):
    try:
        r = _sb.table("ea_config").select("*").limit(1).execute()
        return r.data[0] if r.data else {}
    except: return {}

def push_config(_sb, updates: dict):
    try:
        r = _sb.table("ea_config").select("id").limit(1).execute()
        if r.data:
            _sb.table("ea_config").update(updates).eq("id", r.data[0]["id"]).execute()
        else:
            _sb.table("ea_config").insert({**updates, "magic": 20250228}).execute()
        return True
    except Exception as e:
        st.error(f"Config update failed: {e}")
        return False


# ──────────────────────────────────────────────────────────────────────
#  COLOUR HELPERS
# ──────────────────────────────────────────────────────────────────────

REGIME_COLORS = {
    "Trend Bull": "#00e676",
    "Trend Bear": "#ff5252",
    "Ranging":    "#ffc107",
    "High Volatility": "#ff9800",
    "Breakout":   "#9c27b0",
    "Undefined":  "#8b949e",
}

def pnl_color(v):
    if isinstance(v, (int, float)):
        return "color:#00e676" if v >= 0 else "color:#ff5252"
    return ""

def signal_badge(s):
    if not isinstance(s, int): return s
    colors = {2:"#00e676", 1:"#4caf50", 0:"#8b949e", -1:"#f44336", -2:"#ff1744"}
    labels = {2:"STR BUY", 1:"BUY", 0:"WAIT", -1:"SELL", -2:"STR SELL"}
    c = colors.get(s, "#8b949e")
    l = labels.get(s, str(s))
    return f'<span style="color:{c};font-weight:700">{l}</span>'


# ──────────────────────────────────────────────────────────────────────
#  CHARTS
# ──────────────────────────────────────────────────────────────────────

def equity_curve_chart(perf_df: pd.DataFrame):
    if perf_df.empty:
        return go.Figure()
    df = perf_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.7, 0.3], vertical_spacing=0.05)

    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["equity"],
        name="Equity", line=dict(color="#00bcd4", width=2),
        fill="tozeroy", fillcolor="rgba(0,188,212,0.08)"
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["balance"],
        name="Balance", line=dict(color="#9c27b0", width=1.5, dash="dot"),
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=df["timestamp"], y=df["drawdown"],
        name="Drawdown %", marker_color="#ff5252", opacity=0.7
    ), row=2, col=1)

    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        font_family="JetBrains Mono", font_color="#e6edf3",
        margin=dict(l=0, r=0, t=10, b=0), height=340,
        legend=dict(orientation="h", y=1.02, x=0),
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor="#21262d", showgrid=True)
    fig.update_yaxes(gridcolor="#21262d", showgrid=True)
    return fig


def pnl_by_symbol_chart(trades_df: pd.DataFrame):
    if trades_df.empty or "symbol" not in trades_df.columns:
        return go.Figure()
    closed = trades_df[trades_df["action"] == "CLOSE"].copy()
    if closed.empty: return go.Figure()
    grp = closed.groupby("symbol")["pnl"].sum().reset_index()
    grp = grp.sort_values("pnl", ascending=True)
    colors = ["#00e676" if v >= 0 else "#ff5252" for v in grp["pnl"]]
    fig = go.Figure(go.Bar(
        x=grp["pnl"], y=grp["symbol"], orientation="h",
        marker_color=colors, text=[f"${v:+.2f}" for v in grp["pnl"]],
        textposition="outside",
    ))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        font_family="JetBrains Mono", height=280, margin=dict(l=0, r=60, t=10, b=0),
        xaxis_title="P&L ($)", yaxis_title="",
    )
    return fig


def regime_donut_chart(regime_df: pd.DataFrame):
    if regime_df.empty or "regime" not in regime_df.columns:
        return go.Figure()
    counts = regime_df["regime"].value_counts().reset_index()
    counts.columns = ["regime", "count"]
    colors = [REGIME_COLORS.get(r, "#8b949e") for r in counts["regime"]]
    fig = go.Figure(go.Pie(
        labels=counts["regime"], values=counts["count"],
        hole=0.55, marker_colors=colors,
        textinfo="label+percent", textfont_size=10,
    ))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#0d1117",
        font_family="JetBrains Mono", height=260,
        margin=dict(l=0, r=0, t=10, b=0),
        showlegend=False,
    )
    return fig


def win_rate_by_regime_chart(trades_df: pd.DataFrame):
    if trades_df.empty: return go.Figure()
    closed = trades_df[(trades_df["action"] == "CLOSE") & trades_df["regime"].notna()].copy()
    if closed.empty: return go.Figure()
    def wr(grp):
        wins = (grp["pnl"] > 0).sum()
        return wins / len(grp) * 100 if len(grp) > 0 else 0
    stats = closed.groupby("regime").apply(
        lambda g: pd.Series({
            "win_rate": wr(g),
            "trades":   len(g),
            "pnl":      g["pnl"].sum(),
        })
    ).reset_index()
    fig = go.Figure(go.Bar(
        x=stats["regime"], y=stats["win_rate"],
        marker_color=[REGIME_COLORS.get(r, "#8b949e") for r in stats["regime"]],
        text=[f'{v:.1f}%<br>{int(t)} trades' for v, t in zip(stats["win_rate"], stats["trades"])],
        textposition="outside",
    ))
    fig.add_hline(y=50, line_dash="dot", line_color="#8b949e", annotation_text="50%")
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        font_family="JetBrains Mono", height=280,
        margin=dict(l=0, r=0, t=10, b=30),
        yaxis_title="Win Rate %", yaxis_range=[0, 105],
    )
    return fig


def ai_confidence_scatter(trades_df: pd.DataFrame):
    if trades_df.empty: return go.Figure()
    df = trades_df[(trades_df["action"] == "CLOSE") &
                   trades_df["ai_score"].notna() &
                   trades_df["pnl"].notna()].copy()
    if df.empty: return go.Figure()
    df["result"] = df["pnl"].apply(lambda x: "Win" if x > 0 else "Loss")
    fig = px.scatter(
        df, x="confidence", y="pnl",
        color="result",
        color_discrete_map={"Win": "#00e676", "Loss": "#ff5252"},
        size=abs(df["pnl"].fillna(1)).clip(1) if "pnl" in df.columns else None,
        hover_data=["symbol", "regime"],
        labels={"confidence": "AI Confidence", "pnl": "P&L ($)"},
    )
    fig.add_hline(y=0, line_dash="dot", line_color="#8b949e")
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        font_family="JetBrains Mono", height=280,
        margin=dict(l=0, r=0, t=10, b=0),
    )
    return fig


# ──────────────────────────────────────────────────────────────────────
#  SIDEBAR — CONNECTION & CONTROLS
# ──────────────────────────────────────────────────────────────────────

def render_sidebar(sb):
    with st.sidebar:
        st.markdown("## ⚡ ApexHydra")
        st.markdown("---")

        config = fetch_config(sb)
        is_halted  = config.get("halted", False)
        is_paused  = config.get("paused", False)

        # EA Status
        status_html = f"""
        <div style='margin-bottom:12px'>
          <div class='section-header'>EA STATUS</div>
          <span class='badge {"badge-halted" if is_halted else ("badge-warn" if is_paused else "badge-active")}'>
            {"⛔ HALTED" if is_halted else ("⏸ PAUSED" if is_paused else "✅ ACTIVE")}
          </span>
        </div>
        """
        st.markdown(status_html, unsafe_allow_html=True)

        st.markdown('<div class="section-header">REMOTE CONTROL</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("▶ Resume", use_container_width=True):
                if push_config(sb, {"paused": False, "halted": False}):
                    st.success("Resumed")
                    st.cache_data.clear()
        with col2:
            if st.button("⏸ Pause", use_container_width=True):
                if push_config(sb, {"paused": True}):
                    st.warning("Paused")
                    st.cache_data.clear()

        if st.button("⛔ Emergency Stop", use_container_width=True):
            if push_config(sb, {"halted": True, "paused": True}):
                st.error("EA Halted!")
                st.cache_data.clear()

        st.markdown("---")
        st.markdown('<div class="section-header">RISK SETTINGS</div>', unsafe_allow_html=True)

        new_risk = st.slider("Risk % Per Trade", 0.1, 5.0,
                              float(config.get("risk_pct", 1.0)), 0.1)
        new_dd   = st.slider("Max DD % (Halt)", 5.0, 50.0,
                              float(config.get("max_dd_pct", 20.0)), 1.0)
        new_pos  = st.slider("Max Positions", 1, 10,
                              int(config.get("max_positions", 3)))
        new_conf = st.slider("Min AI Confidence", 0.40, 0.90,
                              float(config.get("min_confidence", 0.60)), 0.01)

        if st.button("💾 Apply Risk Settings", use_container_width=True):
            if push_config(sb, {
                "risk_pct":       new_risk,
                "max_dd_pct":     new_dd,
                "max_positions":  new_pos,
                "min_confidence": new_conf,
            }):
                st.success("Settings saved — EA will apply on next sync")
                st.cache_data.clear()

        st.markdown("---")
        st.markdown('<div class="section-header">REFRESH</div>', unsafe_allow_html=True)
        auto_refresh = st.toggle("Auto-refresh (10s)", value=True)
        if st.button("🔄 Refresh Now", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

        st.markdown("---")
        st.caption(f"Last update: {datetime.now().strftime('%H:%M:%S')}")

    return auto_refresh


# ──────────────────────────────────────────────────────────────────────
#  MAIN LAYOUT
# ──────────────────────────────────────────────────────────────────────

def main():
    sb = get_supabase()
    auto_refresh = render_sidebar(sb)

    # ── Header ────────────────────────────────────────────────────────
    st.markdown("""
    <div style='display:flex;align-items:center;gap:16px;margin-bottom:4px'>
      <h1 style='margin:0;font-size:26px;color:#00bcd4;letter-spacing:2px'>⚡ APEXHYDRA CRYPTO</h1>
      <span class='badge badge-active'>v3.0 LIVE</span>
    </div>
    <p style='color:#8b949e;font-size:12px;margin:0'>AI-Powered Crypto EA · Multi-Symbol Scanner · Modal AI · Supabase DB</p>
    <hr style='border-color:#21262d;margin:12px 0'>
    """, unsafe_allow_html=True)

    # ── Load data ─────────────────────────────────────────────────────
    perf_df    = fetch_performance(sb)
    trades_df  = fetch_trades(sb)
    regime_df  = fetch_regime_changes(sb)
    events_df  = fetch_events(sb)

    # ── TOP KPI METRICS ───────────────────────────────────────────────
    latest = perf_df.iloc[0] if not perf_df.empty else {}

    balance  = float(latest.get("balance",  0))
    equity   = float(latest.get("equity",   0))
    dd       = float(latest.get("drawdown", 0))
    tot_t    = int(latest.get("total_trades", 0))
    wins_t   = int(latest.get("wins",  0))
    loss_t   = int(latest.get("losses", 0))
    tot_pnl  = float(latest.get("total_pnl", 0))
    ai_acc   = float(latest.get("global_accuracy", 0)) * 100
    wr_pct   = (wins_t / tot_t * 100) if tot_t > 0 else 0

    c1,c2,c3,c4,c5,c6,c7,c8 = st.columns(8)
    c1.metric("Balance",      f"${balance:,.2f}")
    c2.metric("Equity",       f"${equity:,.2f}",   delta=f"{equity-balance:+.2f}")
    c3.metric("Total P&L",    f"${tot_pnl:+,.2f}")
    c4.metric("Drawdown",     f"{dd:.1f}%",         delta=None)
    c5.metric("Win Rate",     f"{wr_pct:.1f}%")
    c6.metric("Trades",       tot_t)
    c7.metric("AI Accuracy",  f"{ai_acc:.1f}%")
    c8.metric("W / L",        f"{wins_t} / {loss_t}")

    st.markdown("")

    # ── ROW 1: Equity Curve + Symbol Heat ────────────────────────────
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.markdown('<div class="section-header">📈 EQUITY CURVE & DRAWDOWN</div>', unsafe_allow_html=True)
        st.plotly_chart(equity_curve_chart(perf_df), use_container_width=True)

    with col_right:
        st.markdown('<div class="section-header">🌐 MARKET REGIMES (Today)</div>', unsafe_allow_html=True)
        # Latest regime per symbol
        if not regime_df.empty:
            latest_regimes = regime_df.sort_values("timestamp").groupby("symbol").last().reset_index()
            for _, row in latest_regimes.iterrows():
                rc = REGIME_COLORS.get(row.get("regime",""), "#8b949e")
                conf = float(row.get("confidence", 0)) * 100
                ai   = float(row.get("ai_score", 0)) * 100
                st.markdown(f"""
                <div style='display:flex;justify-content:space-between;align-items:center;
                            padding:6px 10px;margin-bottom:4px;background:#161b22;
                            border-radius:6px;border-left:3px solid {rc}'>
                  <span style='font-weight:700;font-size:12px'>{row.get("symbol","")}</span>
                  <span style='color:{rc};font-size:11px'>{row.get("regime","")}</span>
                  <span style='color:#8b949e;font-size:10px'>{conf:.0f}% conf</span>
                  <span style='color:{"#00e676" if ai>=0 else "#ff5252"};font-size:10px'>{ai:+.1f}%</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No regime data yet.")

    # ── ROW 2: P&L by Symbol + Win Rate by Regime + Confidence Scatter ─
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown('<div class="section-header">💰 P&L BY SYMBOL</div>', unsafe_allow_html=True)
        st.plotly_chart(pnl_by_symbol_chart(trades_df), use_container_width=True)

    with c2:
        st.markdown('<div class="section-header">🎯 WIN RATE BY REGIME</div>', unsafe_allow_html=True)
        st.plotly_chart(win_rate_by_regime_chart(trades_df), use_container_width=True)

    with c3:
        st.markdown('<div class="section-header">🤖 AI CONFIDENCE vs P&L</div>', unsafe_allow_html=True)
        st.plotly_chart(ai_confidence_scatter(trades_df), use_container_width=True)

    # ── ROW 3: Open Trades + Regime Donut ────────────────────────────
    col_trades, col_donut = st.columns([3, 1])

    with col_trades:
        st.markdown('<div class="section-header">📋 RECENT TRADES</div>', unsafe_allow_html=True)
        if not trades_df.empty:
            disp = trades_df.head(25).copy()
            for col in ["timestamp"]:
                if col in disp.columns:
                    disp[col] = pd.to_datetime(disp[col]).dt.strftime("%m/%d %H:%M")
            display_cols = [c for c in
                ["timestamp","symbol","action","regime","signal","lots","price","sl","tp","pnl","confidence","ai_score"]
                if c in disp.columns]
            disp = disp[display_cols]

            def style_pnl(val):
                if isinstance(val, (int, float)) and not np.isnan(val):
                    return "color: #00e676" if val >= 0 else "color: #ff5252"
                return ""

            styled = disp.style.applymap(style_pnl, subset=["pnl"] if "pnl" in disp.columns else [])
            st.dataframe(styled, use_container_width=True, height=280)
        else:
            st.info("No trade data yet.")

    with col_donut:
        st.markdown('<div class="section-header">🔄 REGIME DISTRIBUTION</div>', unsafe_allow_html=True)
        st.plotly_chart(regime_donut_chart(regime_df), use_container_width=True)

    # ── ROW 4: AI Performance Table ───────────────────────────────────
    st.markdown('<div class="section-header">🧠 AI PERFORMANCE BY REGIME</div>', unsafe_allow_html=True)
    if not trades_df.empty:
        closed = trades_df[(trades_df["action"] == "CLOSE") & trades_df["regime"].notna()].copy()
        if not closed.empty:
            stats = closed.groupby("regime").apply(lambda g: pd.Series({
                "Trades": len(g),
                "Wins":   (g["pnl"] > 0).sum(),
                "Losses": (g["pnl"] <= 0).sum(),
                "Win Rate": f"{(g['pnl']>0).mean()*100:.1f}%",
                "Total P&L": f"${g['pnl'].sum():+.2f}",
                "Avg P&L": f"${g['pnl'].mean():+.2f}",
                "Avg Confidence": f"{g['confidence'].mean()*100:.1f}%" if "confidence" in g else "N/A",
                "Avg AI Score": f"{g['ai_score'].mean()*100:+.1f}%" if "ai_score" in g else "N/A",
            })).reset_index()
            st.dataframe(stats, use_container_width=True, hide_index=True)
        else:
            st.info("No closed trades yet.")

    # ── ROW 5: Event Log ─────────────────────────────────────────────
    col_log, col_regime = st.columns([2, 1])

    with col_log:
        st.markdown('<div class="section-header">📟 EVENT LOG</div>', unsafe_allow_html=True)
        if not events_df.empty:
            log_html = "<div class='log-panel'>"
            for _, row in events_df.head(40).iterrows():
                ts  = pd.to_datetime(row.get("timestamp","")).strftime("%H:%M:%S") if row.get("timestamp") else ""
                typ = str(row.get("type","INFO"))
                msg = str(row.get("message",""))
                cls = "log-win" if "WIN" in typ or "OPEN" in typ else \
                      "log-loss" if "LOSS" in typ or "HALT" in typ else \
                      "log-warn" if "WARN" in typ or "RESUME" in typ else "log-info"
                log_html += f"<div class='log-line {cls}'>{ts} │ <b>{typ}</b> │ {msg}</div>"
            log_html += "</div>"
            st.markdown(log_html, unsafe_allow_html=True)
        else:
            st.info("No events logged yet.")

    with col_regime:
        st.markdown('<div class="section-header">🔍 RECENT REGIME CHANGES</div>', unsafe_allow_html=True)
        if not regime_df.empty:
            rdisp = regime_df.head(15).copy()
            rdisp["timestamp"] = pd.to_datetime(rdisp["timestamp"]).dt.strftime("%m/%d %H:%M")
            rcols = [c for c in ["timestamp","symbol","regime","confidence","adx","rsi"] if c in rdisp.columns]
            st.dataframe(rdisp[rcols], use_container_width=True, height=280, hide_index=True)
        else:
            st.info("No regime changes yet.")

    # ── AUTO REFRESH ──────────────────────────────────────────────────
    if auto_refresh:
        time.sleep(10)
        st.cache_data.clear()
        st.rerun()


# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
