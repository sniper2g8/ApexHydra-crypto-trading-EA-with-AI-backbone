"""
ApexHydra Crypto v4.2 — Streamlit Dashboard (Redesign)
Dark-mode UI · Live KPIs with diagnostics · 8 tabs · EA Logs tab
"""
import streamlit as st, pandas as pd, plotly.graph_objects as go
import plotly.express as px
from supabase import create_client
from datetime import datetime, timezone, timedelta
import time, requests, json

st.set_page_config(page_title="ApexHydra v4.0", page_icon="⚡", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""<style>
html,body,.stApp{background:#0d1117;color:#e6edf3}
section[data-testid="stSidebar"]{background:#161b22 !important;border-right:1px solid #30363d}
.block-container{padding-top:1.2rem;padding-bottom:2rem}
.kpi-card{background:#161b22;border:1px solid #30363d;border-radius:12px;padding:18px 16px 14px;margin:4px 0;box-shadow:0 2px 8px rgba(0,0,0,.25)}
.kpi-label{font-size:.68rem;color:#8b949e;text-transform:uppercase;letter-spacing:.05em;margin-bottom:6px}
.kpi-val{font-size:1.45rem;font-weight:700;line-height:1.1}
.kpi-sub{font-size:.72rem;color:#8b949e;margin-top:4px}
.kpi-pos{color:#3fb950}.kpi-neg{color:#f85149}.kpi-neu{color:#58a6ff}.kpi-gold{color:#e3b341}
.badge{display:inline-block;padding:3px 12px;border-radius:20px;font-size:.72rem;font-weight:700;letter-spacing:.04em}
.badge-active{background:#1a3a2a;color:#3fb950;border:1px solid #3fb950}
.badge-paused{background:#3a3000;color:#e3b341;border:1px solid #e3b341}
.badge-halted{background:#3a0a0a;color:#f85149;border:1px solid #f85149}
.badge-offline{background:#1c1c1c;color:#8b949e;border:1px solid #30363d}
.ibox{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:12px 16px;margin:6px 0;font-size:.84rem;color:#c9d1d9}
.ibox-blue{border-left:3px solid #58a6ff;background:#0d1f33}
.ibox-green{border-left:3px solid #3fb950;background:#0d2118}
.ibox-red{border-left:3px solid #f85149;background:#2d0a0a}
.ibox-yellow{border-left:3px solid #e3b341;background:#2a2000}
.ev-row{font-size:.8rem;padding:5px 10px;border-radius:6px;margin:2px 0;background:#161b22;border-left:3px solid #30363d;display:flex;gap:10px;align-items:baseline}
.ev-ts{color:#8b949e;min-width:58px;font-size:.73rem}
.ev-type{font-weight:700;min-width:62px;font-size:.73rem}
.rrow{display:flex;align-items:center;gap:10px;padding:8px 12px;background:#161b22;border-radius:8px;margin:3px 0;border:1px solid #30363d}
.stTabs [data-baseweb="tab-list"]{background:#161b22;border-radius:8px;border:1px solid #30363d;gap:2px;padding:3px}
.stTabs [data-baseweb="tab"]{color:#8b949e;font-size:.82rem;border-radius:6px !important;padding:6px 14px}
.stTabs [aria-selected="true"]{background:#21262d !important;color:#e6edf3 !important}
hr{border-color:#30363d}
</style>""", unsafe_allow_html=True)

# ── Supabase ───────────────────────────────────────────────────────────
@st.cache_resource
def get_sb():
    return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])

DB_OK, DB_ERR, supabase = False, "", None
try:
    supabase = get_sb(); DB_OK = True
except Exception as e:
    DB_ERR = str(e)

# ── Telegram ───────────────────────────────────────────────────────────
def send_tg(token, chat_id, msg):
    if not token or not chat_id: return False
    try:
        r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                          json={"chat_id":chat_id,"text":msg,"parse_mode":"HTML"},timeout=5)
        return r.status_code == 200
    except: return False

def tg(msg): send_tg(st.secrets.get("TG_BOT_TOKEN",""), st.secrets.get("TG_CHAT_ID",""), msg)

# ── Fetchers ───────────────────────────────────────────────────────────
def _df(raw):
    return pd.DataFrame(raw.data) if raw.data else pd.DataFrame()

def _nums(df, cols):
    for c in cols:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _ts(df, col="timestamp"):
    if col in df.columns: df[col] = pd.to_datetime(df[col], utc=True)
    return df

@st.cache_data(ttl=15)
def cfg():
    if not DB_OK: return None
    try:
        r = supabase.table("ea_config").select("*").limit(1).execute()
        return r.data[0] if r.data else None
    except: return None

@st.cache_data(ttl=15)
def perf(hours=720):
    if not DB_OK: return pd.DataFrame()
    try:
        since = (datetime.now(timezone.utc)-timedelta(hours=hours)).isoformat()
        df = _df(supabase.table("performance").select("*").gte("timestamp",since).order("timestamp").execute())
        if not df.empty: _ts(df); _nums(df,["balance","equity","drawdown","total_pnl","global_accuracy","wins","total_trades"])
        return df
    except: return pd.DataFrame()

@st.cache_data(ttl=15)
def trades(limit=300):
    if not DB_OK: return pd.DataFrame()
    try:
        df = _df(supabase.table("trades").select("*").order("timestamp",desc=True).limit(limit).execute())
        if not df.empty: _ts(df); _nums(df,["confidence","lots","price","sl","tp","pnl"])
        return df
    except: return pd.DataFrame()

@st.cache_data(ttl=15)
def regimes_cur():
    if not DB_OK: return pd.DataFrame()
    try:
        df = _df(supabase.table("current_regimes").select("*").execute())
        if not df.empty: _ts(df)
        return df
    except: return pd.DataFrame()

@st.cache_data(ttl=15)
def regime_changes(limit=50):
    if not DB_OK: return pd.DataFrame()
    try:
        df = _df(supabase.table("regime_changes").select("*").order("timestamp",desc=True).limit(limit).execute())
        if not df.empty: _ts(df)
        return df
    except: return pd.DataFrame()

@st.cache_data(ttl=30)
def events(limit=40):
    if not DB_OK: return pd.DataFrame()
    try:
        df = _df(supabase.table("events").select("*").order("timestamp",desc=True).limit(limit).execute())
        if not df.empty: _ts(df)
        return df
    except: return pd.DataFrame()

@st.cache_data(ttl=30)
def trade_summary():
    if not DB_OK: return pd.DataFrame()
    try:
        df = _df(supabase.table("trade_summary").select("*").execute())
        if not df.empty: _nums(df,["win_rate_pct","total_pnl","avg_ai_score","avg_confidence"])
        return df
    except: return pd.DataFrame()

@st.cache_data(ttl=30)
def regime_stats():
    if not DB_OK: return pd.DataFrame()
    try:
        df = _df(supabase.table("regime_stats").select("*").execute())
        if not df.empty: _nums(df,["win_rate_pct","total_pnl","avg_pnl"])
        return df
    except: return pd.DataFrame()

@st.cache_data(ttl=30)
def ea_logs(limit=100):
    if not DB_OK: return pd.DataFrame()
    try:
        df = _df(supabase.table("ea_logs").select("*").order("logged_at",desc=True).limit(limit).execute())
        if not df.empty: _ts(df,"logged_at")
        return df
    except: return pd.DataFrame()

@st.cache_data(ttl=30)
def fwd_test():
    if not DB_OK: return pd.DataFrame()
    try:
        df = _df(supabase.table("forward_test_results").select("*").order("tested_at",desc=True).limit(20).execute())
        if not df.empty: _nums(df,["win_rate","trades","wins"])
        return df
    except: return pd.DataFrame()

def update_cfg(upd):
    if not DB_OK: return False
    try:
        upd["updated_by"]="streamlit"
        supabase.table("ea_config").update(upd).eq("magic",20250228).execute()
        st.cache_data.clear(); return True
    except Exception as e:
        st.error(f"Update failed: {e}"); return False

# ── Helpers ────────────────────────────────────────────────────────────
RC = {"Trend Bull":"#3fb950","Trend Bear":"#f85149","Ranging":"#e3b341",
      "High Volatility":"#ff7c43","Breakout":"#a371f7","Undefined":"#8b949e",
      "TRENDING":"#3fb950","RANGING":"#e3b341","VOLATILE":"#ff7c43"}

def kpi(col, label, value, sub="", color="kpi-neu"):
    col.markdown(f'<div class="kpi-card"><div class="kpi-label">{label}</div>'
                 f'<div class="kpi-val {color}">{value}</div>'
                 + (f'<div class="kpi-sub">{sub}</div>' if sub else "")
                 + '</div>', unsafe_allow_html=True)

def dark(fig, title="", h=340):
    fig.update_layout(template="plotly_dark",paper_bgcolor="#0d1117",plot_bgcolor="#161b22",
                      title=title,height=h,margin=dict(l=0,r=0,t=40,b=0),
                      font=dict(color="#c9d1d9"),legend=dict(bgcolor="#161b22",bordercolor="#30363d"))
    return fig

def ibox(text, kind=""):
    extra = f" ibox-{kind}" if kind else ""
    st.markdown(f'<div class="ibox{extra}">{text}</div>', unsafe_allow_html=True)

ICONS = {"HALT":"⛔","RESUME":"▶️","OPEN":"📈","CLOSE":"📉","ERROR":"❌","INFO":"ℹ️","DEINIT":"🔌"}
TCOL  = {"HALT":"#f85149","ERROR":"#f85149","RESUME":"#3fb950","OPEN":"#3fb950",
         "CLOSE":"#58a6ff","DEINIT":"#ff7c43","INFO":"#8b949e","WARN":"#e3b341"}

def ev_row(ts_str, type_str, msg_str, color="#8b949e"):
    st.markdown(
        f'<div class="ev-row" style="border-left-color:{color};">'
        f'<span class="ev-ts">{ts_str}</span>'
        f'<span class="ev-type" style="color:{color};">{ICONS.get(type_str,"•")} {type_str}</span>'
        f'<span style="color:#c9d1d9;font-size:.78rem;">{str(msg_str)[:130]}</span></div>',
        unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ ApexHydra v4.0")
    if DB_OK:
        st.markdown('<div style="background:#0d2118;border:1px solid #3fb950;border-radius:8px;padding:8px 14px;font-size:.82rem;color:#3fb950;">🟢 Supabase connected</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div style="background:#2d0a0a;border:1px solid #f85149;border-radius:8px;padding:8px 14px;font-size:.82rem;color:#f85149;">🔴 {DB_ERR[:80]}</div>', unsafe_allow_html=True)
        st.info("Check SUPABASE_URL + SUPABASE_KEY in .streamlit/secrets.toml")
        st.stop()

    c = cfg()
    st.markdown("<br>", unsafe_allow_html=True)
    if c is None:
        st.markdown('<span class="badge badge-offline">⚪ EA OFFLINE</span>', unsafe_allow_html=True)
        st.caption("No ea_config row. Set Inp_DB_Enable=true in EA inputs.")
    elif c.get("halted"):
        st.markdown('<span class="badge badge-halted">⛔ HALTED</span>', unsafe_allow_html=True)
    elif c.get("paused"):
        st.markdown('<span class="badge badge-paused">⏸ PAUSED</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge badge-active">▶ ACTIVE</span>', unsafe_allow_html=True)
    if c and c.get("live_ts"):
        st.caption(f"EA last ping: {c['live_ts']}")

    st.markdown("---")
    st.markdown("### 🎛 EA Control")
    b1,b2 = st.columns(2)
    with b1:
        if st.button("▶ Resume", use_container_width=True, type="primary"):
            if update_cfg({"paused":False,"halted":False}):
                tg("▶️ <b>ApexHydra RESUMED</b> via dashboard"); st.success("Resumed!"); st.rerun()
    with b2:
        if st.button("⏸ Pause", use_container_width=True):
            if update_cfg({"paused":True}):
                tg("⏸ <b>ApexHydra PAUSED</b> via dashboard"); st.warning("Paused!"); st.rerun()
    if st.button("⛔ Emergency Stop", use_container_width=True, type="secondary"):
        if update_cfg({"halted":True,"paused":True}):
            tg("🚨 <b>EMERGENCY STOP</b> via dashboard"); st.error("Halted!"); st.rerun()

    st.markdown("---")
    st.markdown("### 💰 Capital Allocation")
    cap_val  = float(c.get("trading_capital",0) or 0) if c else 0.0
    cap_pct  = int(  c.get("capital_pct",100) or 100) if c else 100
    new_cap  = st.number_input("Trading Capital ($)", 0.0, 1e6, cap_val, 100.0, format="%.2f", help="0 = full MT5 balance")
    new_pct  = st.slider("Usable %", 10, 100, cap_pct)
    eff      = new_cap * new_pct/100 if new_cap > 0 else 0
    if new_cap > 0:
        ibox(f"&#x1F4B0; <b>${new_cap:,.2f}</b> &times; {new_pct}% = <b>${eff:,.2f}</b> effective","blue")
    else:
        st.caption("Risk base: full MT5 balance")
    if st.button("💾 Apply Capital", use_container_width=True):
        if update_cfg({"trading_capital":new_cap,"capital_pct":new_pct}):
            tg(f"💰 <b>Capital</b>: ${new_cap:,.2f} @ {new_pct}% → ${eff:,.2f}"); st.success("Applied!"); st.rerun()

    st.markdown("---")
    st.markdown("### 🤖 Modal AI")
    mu = st.secrets.get("MODAL_URL","")
    if mu:
        try:
            r = requests.get(f"{mu.rstrip('/')}/health", timeout=5)
            if r.status_code == 200:
                h = r.json(); strats = h.get("strategies",{}); trained = sum(1 for s in strats.values() if s.get("trained"))
                ibox(f"🟢 Modal OK v{h.get('version','?')}<br>PPO: <b>{trained}/3</b> trained | {h.get('features','?')}D features","green")
            else: ibox("🟡 Modal returned non-200","yellow")
        except: ibox("🔴 Modal unreachable","red")
    else: ibox("Add <code>MODAL_URL</code> to secrets.toml")

    st.markdown("---")
    st.markdown("### ⚙️ Risk Settings")
    if c:
        nr = st.number_input("Risk %",0.1,10.0,float(c.get("risk_pct",1.0)),0.1)
        nd = st.number_input("Max DD %",1.0,50.0,float(c.get("max_dd_pct",20.0)),1.0)
        nm = st.number_input("Max Positions",1,20,int(c.get("max_positions",3)),1)
        nc = st.number_input("Min Confidence",0.30,0.95,float(c.get("min_confidence",0.55)),0.01)
        if st.button("💾 Apply Risk", use_container_width=True):
            if update_cfg({"risk_pct":nr,"max_dd_pct":nd,"max_positions":nm,"min_confidence":nc}):
                tg(f"⚙️ <b>Risk</b>: {nr}% DD:{nd}% Pos:{nm} Conf:{nc:.2f}"); st.success("Applied!"); st.rerun()
    else: st.info("EA config not loaded.")

    st.markdown("---")
    st.markdown("### 📱 Telegram")
    tok = st.secrets.get("TG_BOT_TOKEN",""); cid = st.secrets.get("TG_CHAT_ID","")
    if tok and cid:
        ibox(f"🟢 Connected<br>Chat: <code>{cid}</code>","green")
        if st.button("📤 Send Test"): st.success("Sent!") if send_tg(tok,cid,"🤖 <b>ApexHydra</b> — Dashboard OK ✅") else st.error("Failed")
    else:
        miss = [k for k in ["TG_BOT_TOKEN","TG_CHAT_ID"] if not st.secrets.get(k)]
        ibox(f"🔴 Missing: {', '.join(miss)}")

    st.markdown("---")
    auto_refresh = st.checkbox("🔄 Auto-refresh (15s)", True)
    if st.button("↺ Refresh Now"): st.cache_data.clear(); st.rerun()
    st.caption(f"Render: {datetime.now().strftime('%H:%M:%S')}")

# ── Load data ──────────────────────────────────────────────────────────
perf_df   = perf()
trades_df = trades()
cfg_data  = cfg()

# ── Header ─────────────────────────────────────────────────────────────
H1,H2 = st.columns([4,1])
with H1:
    st.markdown("# ⚡ ApexHydra Crypto v4.0")
    st.markdown(f"<span style='color:#8b949e;font-size:.85rem;'>{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}</span>", unsafe_allow_html=True)
with H2:
    if cfg_data:
        bc = "badge-halted" if cfg_data.get("halted") else ("badge-paused" if cfg_data.get("paused") else "badge-active")
        bl = "⛔ HALTED"   if cfg_data.get("halted") else ("⏸ PAUSED"    if cfg_data.get("paused") else "▶ ACTIVE")
        st.markdown(f'<br><span class="badge {bc}" style="font-size:.9rem;padding:7px 18px;">{bl}</span>', unsafe_allow_html=True)
st.markdown("---")

# ── KPI row ────────────────────────────────────────────────────────────
kpis = st.columns(6)

def render_kpis(bal,eq,dd,tot,wins,pnl,sub=""):
    wr = wins/tot*100 if tot>0 else 0
    kpi(kpis[0],"Balance",    f"${bal:,.2f}", sub)
    kpi(kpis[1],"Equity",     f"${eq:,.2f}",  f"Float {eq-bal:+,.2f}")
    kpi(kpis[2],"Drawdown",   f"{dd:.2f}%",   color="kpi-neg" if dd>10 else("kpi-gold" if dd>5 else "kpi-pos"))
    kpi(kpis[3],"Win Rate",   f"{wr:.1f}%",   f"{wins}W / {tot-wins}L", color="kpi-pos" if wr>=50 else "kpi-neg")
    kpi(kpis[4],"Total PnL",  f"${pnl:+,.2f}", color="kpi-pos" if pnl>=0 else "kpi-neg")
    kpi(kpis[5],"Trades",     str(tot))

if not perf_df.empty:
    l = perf_df.iloc[-1]
    render_kpis(float(l.get("balance",0) or 0), float(l.get("equity",0) or 0),
                float(l.get("drawdown",0) or 0)*100,
                int(l.get("total_trades",0) or 0), int(l.get("wins",0) or 0),
                float(l.get("total_pnl",0) or 0), "from performance log")
elif cfg_data and cfg_data.get("live_balance"):
    render_kpis(float(cfg_data.get("live_balance",0) or 0), float(cfg_data.get("live_equity",0) or 0),
                float(cfg_data.get("live_dd_pct",0) or 0),
                int(cfg_data.get("live_trades",0) or 0),   int(cfg_data.get("live_wins",0) or 0),
                float(cfg_data.get("live_pnl",0) or 0),
                f"⚡ live · {cfg_data.get('live_ts','?')}")
else:
    for col in kpis:
        col.markdown('<div class="kpi-card"><div class="kpi-label">—</div>'
                     '<div class="kpi-val" style="color:#8b949e;font-size:2rem;">—</div></div>',
                     unsafe_allow_html=True)
    reason = ("No ea_config row — set <code>Inp_DB_Enable=true</code> in EA inputs" if cfg_data is None
              else "EA hasn't sent live data yet — wait 15–30s after EA starts (check AutoTrading is ON)"
              if not (cfg_data or {}).get("live_balance")
              else "No performance snapshots yet")
    ibox(f"⚠️ <b>No live data yet</b> — {reason}","yellow")

# Capital budget banner
if cfg_data:
    tc = float(cfg_data.get("trading_capital",0) or 0)
    if tc > 0:
        cp = float(cfg_data.get("capital_pct",100) or 100)
        ibox(f"&#x1F4B0; <b>Capital Budget:</b> ${tc:,.2f} &times; {cp:.0f}% = <b>${tc*cp/100:,.2f} effective</b>","blue")

st.markdown("---")

# ── Tabs ───────────────────────────────────────────────────────────────
T = st.tabs(["📈 Equity Curve","🎯 Regime Map","🤖 AI Performance",
             "📋 Trade History","📡 Live Feed","📊 EA Logs","📱 Telegram","🧪 Backtest"])

# ── T0 Equity Curve ────────────────────────────────────────────────────
with T[0]:
    if not perf_df.empty:
        ps = perf_df.sort_values("timestamp")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ps["timestamp"],y=ps["equity"],name="Equity",
                                  line=dict(color="#58a6ff",width=2),fill="tozeroy",fillcolor="rgba(88,166,255,.06)"))
        if "balance" in ps.columns:
            fig.add_trace(go.Scatter(x=ps["timestamp"],y=ps["balance"],name="Balance",
                                      line=dict(color="#3fb950",width=1.5,dash="dot")))
        st.plotly_chart(dark(fig,"Equity & Balance",360), use_container_width=True)
        if "drawdown" in ps.columns:
            fig2 = go.Figure(go.Scatter(x=ps["timestamp"],y=ps["drawdown"]*100,
                                         fill="tozeroy",fillcolor="rgba(248,81,73,.15)",line=dict(color="#f85149",width=1.5)))
            st.plotly_chart(dark(fig2,"Drawdown %",200), use_container_width=True)
        s1,s2,s3,s4 = st.columns(4)
        sb=float(ps.iloc[0]["balance"] or 0); eb=float(ps.iloc[-1]["balance"] or 0)
        s1.metric("Start Balance",f"${sb:,.2f}"); s2.metric("Current",f"${eb:,.2f}")
        s3.metric("Return",f"{((eb-sb)/sb*100 if sb>0 else 0):+.2f}%"); s4.metric("Max DD",f"{float(ps['drawdown'].max() or 0)*100:.2f}%")
    else:
        ibox("📊 No performance snapshots yet. Ensure <code>Inp_DB_Enable=true</code> in EA inputs.","yellow")

# ── T1 Regime Map ──────────────────────────────────────────────────────
with T[1]:
    c1,c2 = st.columns([1,1])
    with c1:
        st.subheader("Current Regimes")
        rd = regimes_cur()
        if not rd.empty:
            for _,row in rd.iterrows():
                r=str(row.get("regime","Undefined")); col=RC.get(r,"#8b949e")
                ts=row["timestamp"].strftime("%H:%M") if pd.notna(row.get("timestamp")) else "—"
                st.markdown(f'<div class="rrow" style="border-left:3px solid {col};">'
                            f'<b style="color:#e6edf3;min-width:80px;">{row.get("symbol","?")}</b>'
                            f'<span style="color:{col};font-weight:600;min-width:130px;">{r}</span>'
                            f'<span style="color:#8b949e;font-size:.76rem;">Conf:{float(row.get("confidence",0) or 0)*100:.0f}% '
                            f'ADX:{row.get("adx","—")} RSI:{row.get("rsi","—")} {ts}</span></div>',
                            unsafe_allow_html=True)
        else: ibox("No live regime data — waiting for EA scans.")
    with c2:
        st.subheader("Regime Performance")
        rs = regime_stats()
        if not rs.empty:
            fig = px.bar(rs,x="regime",y="win_rate_pct",color="win_rate_pct",color_continuous_scale="RdYlGn",
                         text=rs["win_rate_pct"].round(1).astype(str)+"%")
            st.plotly_chart(dark(fig,"Win Rate % by Regime",260), use_container_width=True)
        else: ibox("No regime performance data yet.")
    st.subheader("Recent Regime Changes")
    rc = regime_changes(30)
    if not rc.empty:
        show=[c for c in ["timestamp","symbol","regime","confidence","adx","rsi","strategy_used"] if c in rc.columns]
        st.dataframe(rc[show].head(30), use_container_width=True, hide_index=True)
    else: ibox("No regime changes recorded yet.")

# ── T2 AI Performance ──────────────────────────────────────────────────
with T[2]:
    ca,cb = st.columns([3,2])
    with ca:
        st.subheader("Per-Symbol Performance")
        ts_df = trade_summary()
        if not ts_df.empty:
            fmt={c:(f"{{:.1f}}%" if c=="win_rate_pct" else "${:.2f}" if c=="total_pnl" else "{:.3f}")
                 for c in ["win_rate_pct","total_pnl","avg_ai_score","avg_confidence"] if c in ts_df.columns}
            st.dataframe(ts_df.style.background_gradient(
                subset=["win_rate_pct"] if "win_rate_pct" in ts_df.columns else [],cmap="RdYlGn").format(fmt),
                use_container_width=True, hide_index=True)
        else: ibox("Per-symbol stats appear after the first closed trade.")
    with cb:
        st.subheader("Win Rate Trend")
        if not perf_df.empty and "global_accuracy" in perf_df.columns:
            fig=go.Figure(go.Scatter(x=perf_df["timestamp"],y=perf_df["global_accuracy"]*100,
                                      line=dict(color="#a371f7",width=2),fill="tozeroy",fillcolor="rgba(163,113,247,.08)"))
            fig.add_hline(y=50,line_dash="dot",line_color="#8b949e")
            st.plotly_chart(dark(fig,"Global Win Rate %",260),use_container_width=True)
        else: ibox("Win rate trend appears once trades are closed.")
        st.subheader("Forward Test")
        ft=fwd_test()
        if not ft.empty:
            st.dataframe(ft[["ticker","strategy","trades","wins","win_rate"]].style.format({"win_rate":"{:.1%}"}),
                         use_container_width=True, hide_index=True)
        else: ibox("Forward test runs every 4h — no results yet.")

# ── T3 Trade History ───────────────────────────────────────────────────
with T[3]:
    if not trades_df.empty:
        show=[c for c in ["timestamp","symbol","action","regime","signal","confidence","lots","price","sl","tp","pnl","won","strategy_used"] if c in trades_df.columns]
        def cpnl(v):
            if pd.isna(v) or v==0: return "color:#8b949e"
            return "color:#3fb950" if v>0 else "color:#f85149"
        st_df = trades_df[show].style
        if "pnl" in show: st_df=st_df.map(cpnl,subset=["pnl"])
        if "confidence" in show: st_df=st_df.format({"confidence":"{:.1%}","pnl":"${:.2f}","lots":"{:.3f}"})
        st.dataframe(st_df, use_container_width=True, hide_index=True)
        closed = trades_df[trades_df["action"]=="CLOSE"] if "action" in trades_df.columns else pd.DataFrame()
        if not closed.empty and "pnl" in closed.columns:
            cp=pd.to_numeric(closed["pnl"],errors="coerce").dropna()
            s1,s2,s3,s4,s5=st.columns(5)
            s1.metric("Closed",len(closed)); s2.metric("Total PnL",f"${cp.sum():+,.2f}")
            s3.metric("Best",f"${cp.max():+,.2f}"); s4.metric("Worst",f"${cp.min():+,.2f}"); s5.metric("Avg",f"${cp.mean():+,.2f}")
            fig=px.histogram(cp,nbins=30,color_discrete_sequence=["#58a6ff"],title="PnL Distribution")
            fig.add_vline(x=0,line_dash="dash",line_color="#f85149")
            st.plotly_chart(dark(fig,"",240),use_container_width=True)
    else: ibox("📋 No trades yet — they appear here once the EA opens its first position.","yellow")

# ── T4 Live Feed ───────────────────────────────────────────────────────
with T[4]:
    f1,f2=st.columns(2)
    with f1:
        st.subheader("📡 Event Log")
        ev=events()
        if not ev.empty:
            for _,row in ev.iterrows():
                t=str(row.get("type","INFO"))
                ts=row["timestamp"].strftime("%H:%M:%S") if pd.notna(row.get("timestamp")) else "—"
                ev_row(ts,t,row.get("message",""),TCOL.get(t,"#8b949e"))
        else: ibox("No events yet — EA writes events on start/halt/resume/trades.")
    with f2:
        st.subheader("🔄 Regime Changes")
        rc2=regime_changes(20)
        if not rc2.empty:
            for _,row in rc2.iterrows():
                r=str(row.get("regime","Undefined")); col=RC.get(r,"#8b949e")
                ts=row["timestamp"].strftime("%m/%d %H:%M") if pd.notna(row.get("timestamp")) else "—"
                ev_row(ts,row.get("symbol","?"),f"{r} | ADX:{row.get('adx','—')} RSI:{row.get('rsi','—')}",col)
        else: ibox("No regime changes recorded yet.")

# ── T5 EA Logs ─────────────────────────────────────────────────────────
with T[5]:
    st.subheader("🖥️ EA Log Stream")
    LC={"ERROR":"#f85149","WARN":"#e3b341","INFO":"#58a6ff","DEBUG":"#8b949e"}
    eals=ea_logs(100)
    if not eals.empty:
        for _,row in eals.iterrows():
            lvl=str(row.get("level","INFO")).upper()
            ts=row["logged_at"].strftime("%H:%M:%S") if pd.notna(row.get("logged_at")) else "—"
            sym=str(row.get("symbol",""))
            col=LC.get(lvl,"#8b949e")
            msg=str(row.get("message",""))[:160]
            sym_s=f'<span style="color:#e3b341;min-width:65px;">{sym}</span>' if sym else ""
            st.markdown(f'<div class="ev-row" style="border-left-color:{col};">'
                        f'<span class="ev-ts">{ts}</span>'
                        f'<span class="ev-type" style="color:{col};">[{lvl}]</span>'
                        f'{sym_s}<span style="color:#c9d1d9;font-size:.78rem;">{msg}</span></div>',
                        unsafe_allow_html=True)
    else:
        ibox("ℹ️ EA logs stream here once running. EA writes via Modal <code>/log</code> endpoint.<br>"
             "Ensure <code>Inp_DB_Enable=true</code>.","blue")
        ev_fb=events(20)
        if not ev_fb.empty:
            st.subheader("Events (fallback)")
            for _,row in ev_fb.iterrows():
                t=str(row.get("type","INFO"))
                ts=row["timestamp"].strftime("%H:%M:%S") if pd.notna(row.get("timestamp")) else "—"
                ev_row(ts,t,row.get("message",""),TCOL.get(t,"#8b949e"))

# ── T6 Telegram ────────────────────────────────────────────────────────
with T[6]:
    st.subheader("📱 Telegram Bot")
    tgl,tgr=st.columns([2,1])
    with tgl:
        st.markdown("""**Quick Setup:**
1. Message **@BotFather** → `/newbot` → copy token
2. Get your Chat ID from **@userinfobot**
3. Add to `.streamlit/secrets.toml`:
```toml
TG_BOT_TOKEN = "123456789:ABCDef..."
TG_CHAT_ID   = "123456789"
```
""")
        st.table(pd.DataFrame({"Event":["Trade Open","Win","Loss","EA Halted","Emergency Stop","Paused","Resumed","Risk Changed","Capital"],"Emoji":["📈","✅","❌","🚨","⛔","⏸","▶️","⚙️","💰"]}))
    with tgr:
        mi=st.text_area("Broadcast message:")
        if st.button("📤 Send",type="primary"):
            tok2=st.secrets.get("TG_BOT_TOKEN",""); cid2=st.secrets.get("TG_CHAT_ID","")
            if not tok2: st.error("TG_BOT_TOKEN not set")
            else: st.success("Sent!") if send_tg(tok2,cid2,mi) else st.error("Failed")
        st.markdown("**Alert Preferences**")
        ao=st.checkbox("Trade Open",True,key="ao"); aw=st.checkbox("Win",True,key="aw")
        al=st.checkbox("Loss",True,key="al"); ah=st.checkbox("Halt",True,key="ah")
        ar=st.checkbox("Resume",False,key="ar"); ac=st.checkbox("Config Change",False,key="ac")
        if st.button("💾 Save Prefs"):
            if update_cfg({"tg_alerts":json.dumps({"on_open":ao,"on_win":aw,"on_loss":al,"on_halt":ah,"on_resume":ar,"on_cfg":ac})}):
                st.success("Saved!")

# ── T7 Backtest ────────────────────────────────────────────────────────
with T[7]:
    st.subheader("🧪 Strategy Backtester")
    MU=st.secrets.get("MODAL_URL","")
    REQ=["timestamp","open","high","low","close","volume","atr","atr_avg","adx","plus_di",
         "minus_di","rsi","macd","macd_signal","macd_hist","ema20","ema50","ema200","htf_ema50","htf_ema200"]
    bta,btb=st.columns([2,1])
    with btb:
        st.markdown("#### Settings")
        bsym=st.text_input("Symbol","BTCUSD"); btf=st.text_input("Timeframe","H1")
        bbal=st.number_input("Balance",min_value=100.0,value=10000.0,step=500.0); brisk=st.number_input("Risk %",min_value=0.1,max_value=10.0,value=1.0,step=0.1)
        brr=st.number_input("Min R:R",min_value=1.0,max_value=5.0,value=1.5,step=0.1); bconf=st.number_input("Min Conf",min_value=0.30,max_value=0.95,value=0.55,step=0.01)
        bsp=st.number_input("Spread pts",20.0); btv=st.number_input("Tick Value",1.0); bts=st.number_input("Tick Size",0.01,format="%.4f")
    with bta:
        st.markdown("#### Upload Bar CSV"); st.caption("Columns: "+", ".join(REQ))
        up=st.file_uploader("Upload CSV",type=["csv"])
        if up and MU:
            try:
                df_bt=pd.read_csv(up); st.success(f"{len(df_bt)} rows"); st.dataframe(df_bt.head(4),use_container_width=True,hide_index=True)
                miss=[c for c in REQ if c not in df_bt.columns]
                if miss: st.error(f"Missing: {miss}")
                elif st.button("▶ Run Backtest",type="primary"):
                    with st.spinner("Running…"):
                        payload={"symbol":bsym,"timeframe":btf,"bars":df_bt[REQ].fillna(0).to_dict("records"),
                                 "initial_balance":bbal,"risk_pct":brisk,"min_rr":brr,"min_confidence":bconf,
                                 "spread_points":bsp,"tick_value":btv,"tick_size":bts,"min_lot":.01,"max_lot":100.,"lot_step":.01,"point":bts,"digits":2}
                        try:
                            r=requests.post(f"{MU.rstrip('/')}/backtest",json=payload,timeout=120)
                            if r.status_code==200: st.session_state["bt"]=r.json()
                            else: st.error(f"{r.status_code}: {r.text[:200]}")
                        except Exception as e: st.error(str(e))
            except Exception as e: st.error(str(e))
        elif up: st.warning("Add MODAL_URL to secrets.toml")
        else: ibox("Upload a CSV to begin.")
    if "bt" in st.session_state:
        res=st.session_state["bt"]; st.markdown("---"); st.markdown("### Results")
        m1,m2,m3,m4,m5,m6,m7=st.columns(7)
        def bk(c,l,v,g=None):
            col="#3fb950" if g is True else("#f85149" if g is False else "#58a6ff")
            c.markdown(f'<div class="kpi-card"><div class="kpi-label">{l}</div><div class="kpi-val" style="color:{col};font-size:1.1rem;">{v}</div></div>',unsafe_allow_html=True)
        bk(m1,"Trades",res["total_trades"]); bk(m2,"Win Rate",f"{res['win_rate']*100:.1f}%",res['win_rate']>=.5)
        bk(m3,"PnL",f"${res['total_pnl']:+,.2f}",res['total_pnl']>=0); bk(m4,"Max DD",f"{res['max_drawdown_pct']:.1f}%",res['max_drawdown_pct']<=15)
        bk(m5,"Sharpe",f"{res['sharpe_ratio']:.2f}",res['sharpe_ratio']>=1.); bk(m6,"PF",f"{res['profit_factor']:.2f}",res['profit_factor']>=1.5)
        bk(m7,"Avg R:R",f"{res.get('avg_rr',0):.2f}",res.get('avg_rr',0)>=1.5)
        if res.get("equity_curve"):
            fig=go.Figure(go.Scatter(y=res["equity_curve"],mode="lines",line=dict(color="#58a6ff",width=2),fill="tozeroy",fillcolor="rgba(88,166,255,.06)"))
            fig.add_hline(y=res["equity_curve"][0],line_dash="dot",line_color="#8b949e")
            st.plotly_chart(dark(fig,"Backtest Equity",300),use_container_width=True)
        if res.get("trades"):
            bdf=pd.DataFrame(res["trades"]); st.dataframe(bdf,use_container_width=True,hide_index=True)
            st.download_button("📥 Download CSV",bdf.to_csv(index=False),f"bt_{bsym}_{btf}.csv","text/csv")

# ── Auto-refresh ───────────────────────────────────────────────────────
if auto_refresh:
    time.sleep(15)
    st.rerun()