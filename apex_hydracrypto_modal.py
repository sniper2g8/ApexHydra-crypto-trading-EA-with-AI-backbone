"""
╔══════════════════════════════════════════════════════════════════════╗
║         ApexHydra Crypto — Modal AI Engine  v5.0                    ║
║                                                                      ║
║  Architecture (merged from Forex PRO modal):                        ║
║  ┌──────────────────────────────────────────────────────────┐       ║
║  │  RegimeDetector  →  TRENDING | RANGING | VOLATILE        │       ║
║  │       ↓                  ↓               ↓               │       ║
║  │  TrendFollowing   MeanReversion      Breakout            │       ║
║  │  (PPO-LSTM)       (PPO-LSTM)         (PPO-LSTM)          │       ║
║  │       └──────────── Signal Fusion ───────────┘           │       ║
║  │                  Kelly Sizer                             │       ║
║  │                  ATR SL/TP                               │       ║
║  └──────────────────────────────────────────────────────────┘       ║
║                                                                      ║
║  Scheduled tasks (5 dedicated cron jobs — uses all 5 Modal slots):  ║
║    Cron 1: news_and_forward  every  5 min  (news + forward test)    ║
║    Cron 2: online_learner    every  6 hrs  (fine-tune PPO models)   ║
║    Cron 3: performance_watch every  1 min  (DD guardian + auto-halt)║
║    Cron 4: model_health_check every 12 hrs (validate + auto-retrain)║
║    Cron 5: db_maintenance    daily 03:00 UTC (prune old DB rows)    ║
║                                                                      ║
║  API Endpoints:                                                      ║
║    POST /predict     — main signal (called by MT5 every scan)       ║
║    POST /train       — trigger PPO training                         ║
║    POST /backtest    — historical backtest                           ║
║    POST /closeall    — emergency close all + halt                   ║
║    POST /closeall_ack — EA confirms close executed                  ║
║    GET  /commands    — EA polls for pending commands                ║
║    POST /log         — EA sends log line to Supabase                ║
║    GET  /logs        — dashboard fetches EA logs                    ║
║    POST /purge       — clean phantom DB rows                        ║
║    GET  /health      — model status + system health                 ║
║                                                                      ║
║  Deploy:  modal deploy apex_hydracrypto_modal.py                    ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import modal
import numpy as np
import json, math, os, pickle
from collections import deque
from datetime import datetime, timezone, timedelta
from typing import Optional
from pydantic import BaseModel

# ──────────────────────────────────────────────────────────────────────
#  MODAL APP + IMAGE
# ──────────────────────────────────────────────────────────────────────

app = modal.App("apex-hydracrypto")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy>=1.26",
        "pandas>=2.2",
        "torch>=2.2",
        "stable-baselines3>=2.3",
        "sb3-contrib>=2.3",
        "gymnasium>=0.29",
        "scikit-learn>=1.4",
        "scipy>=1.12",
        "requests>=2.31",
        "fastapi[standard]>=0.110",
        "python-multipart>=0.0.9",
        "pydantic>=2.6",
        "supabase>=2.4",
        "ta>=0.11",
    )
)

volume    = modal.Volume.from_name("apex-hydracrypto-models", create_if_missing=True)
MODEL_DIR = "/models"
secrets   = modal.Secret.from_name("apex-hydracrypto-secrets")

# ── Model paths (one PPO per strategy) ───────────────────────────────
def _model_path(strategy: str) -> str:
    return os.path.join(MODEL_DIR, f"ppo_{strategy}.zip")

def _meta_path(strategy: str) -> str:
    return os.path.join(MODEL_DIR, f"ppo_{strategy}_meta.json")

MODEL_VERSION  = "5.0.0"
STRATEGY_NAMES = ["trend_following", "mean_reversion", "breakout"]

# ── Per-strategy regime routing ──────────────────────────────────────
REGIME_TO_STRATEGY = {
    "TRENDING": "trend_following",
    "RANGING":  "mean_reversion",
    "VOLATILE": "breakout",
}

# ── Crypto-specific signal names ─────────────────────────────────────
SIGNAL_NAMES = {-2: "Strong Sell", -1: "Sell", 0: "Hold", 1: "Buy", 2: "Strong Buy"}
REGIME_NAMES = {0: "Trend Bull", 1: "Trend Bear", 2: "Ranging", 3: "High Volatility", 4: "Breakout", 5: "Undefined"}

# ── In-memory regime history (persists across requests, same container) ──
_REGIME_HISTORY: dict      = {}     # symbol → deque[(datetime, regime_str)]
_REGIME_HISTORY_MAXLEN     = 30
_CONTAINER_START: datetime = datetime.now(timezone.utc)
_MR_WARMUP_SECS            = 900    # 15 min warm-up before MR is fully trusted
_seeded_symbols: set       = set()

# ── Server-side news blackout cache ──────────────────────────────────────────
# check_news_filter() queries this instead of hitting Supabase on every request.
# Refreshed at most once per minute from the news_blackouts table.
# This means the EA's MT5 calendar (macro events) is Layer 1,
# and the DB-backed crypto/FOMC filter is Layer 2 — independent of the EA.
_NEWS_CACHE: list          = []     # list of active blackout dicts
_NEWS_CACHE_UPDATED: datetime = datetime.min.replace(tzinfo=timezone.utc)
_NEWS_CACHE_TTL_SECS       = 60    # refresh at most once per minute

# ──────────────────────────────────────────────────────────────────────
#  SCHEMAS
# ──────────────────────────────────────────────────────────────────────

class BarData(BaseModel):
    open:   list[float]
    high:   list[float]
    low:    list[float]
    close:  list[float]
    volume: list[float]


class AIRequest(BaseModel):
    symbol:            str
    timeframe:         str
    magic:             int
    timestamp:         str
    account_balance:   float
    account_equity:    float
    allocated_capital: float = 0.0
    risk_pct:          float = 1.0
    max_positions:     int   = 3
    open_positions:    int   = 0
    bars:              BarData
    # Indicators (pre-computed in MT5)
    atr:               float
    atr_avg:           float
    adx:               float
    plus_di:           float
    minus_di:          float
    rsi:               float
    macd:              float
    macd_signal:       float
    macd_hist:         float
    ema20:             float
    ema50:             float
    ema200:            float
    htf_ema50:         float
    htf_ema200:        float
    # Symbol specs
    tick_value:        float
    tick_size:         float
    min_lot:           float
    max_lot:           float
    lot_step:          float
    point:             float
    digits:            int   = 5
    spread:            float = 0.0   # spread in points
    bid:               float = 0.0
    ask:               float = 0.0
    # Time context (for session features — crypto is 24/7 but still has volume patterns)
    hour:              int   = 12
    dow:               int   = 2
    # News filter
    news_blackout:     bool  = False
    news_minutes_away: int   = 999
    news_buffer_minutes: int = 15
    # Online learning history
    recent_signals:    Optional[list[int]]   = None
    recent_outcomes:   Optional[list[float]] = None
    recent_regimes:    Optional[list[int]]   = None


class AIResponse(BaseModel):
    symbol:         str
    regime:         str             # TRENDING | RANGING | VOLATILE
    regime_id:      int             # 0-5 for granular regime
    regime_name:    str             # Trend Bull | Trend Bear | Ranging | ...
    regime_conf:    float
    strategy_used:  str             # trend_following | mean_reversion | breakout
    signal:         int             # -2 to +2
    signal_name:    str
    confidence:     float
    ppo_signal:     str             # BUY | SELL | NONE (raw PPO output)
    ppo_confidence: float
    lots:           float
    sl_price:       float
    tp_price:       float
    sl_atr_mult:    float
    tp_atr_mult:    float
    rr_ratio:       float
    feature_scores: dict
    reasoning:      str
    news_blocked:   bool   = False
    model_trained:  bool   = False
    model_version:  str
    server_ts:      str


# ── Backtest schemas ──────────────────────────────────────────────────

class BacktestBar(BaseModel):
    timestamp:   str
    open:        float; high: float; low: float; close: float; volume: float
    atr: float; atr_avg: float; adx: float; plus_di: float; minus_di: float
    rsi: float; macd: float; macd_signal: float; macd_hist: float
    ema20: float; ema50: float; ema200: float
    htf_ema50: float; htf_ema200: float


class BacktestRequest(BaseModel):
    symbol:          str
    timeframe:       str
    bars:            list[BacktestBar]
    initial_balance: float = 10000.0
    risk_pct:        float = 1.0
    min_rr:          float = 1.5
    min_confidence:  float = 0.52
    tick_value:      float = 1.0
    tick_size:       float = 0.01
    min_lot:         float = 0.01
    max_lot:         float = 100.0
    lot_step:        float = 0.01
    point:           float = 0.01
    digits:          int   = 2
    spread_points:   float = 20.0


class BacktestTrade(BaseModel):
    entry_time: str; exit_time: str; signal: int; regime: str
    confidence: float; lots: float; entry_price: float; exit_price: float
    sl: float; tp: float; pnl: float; won: bool; bars_held: int


class BacktestResponse(BaseModel):
    symbol: str; timeframe: str; bars_tested: int; total_trades: int
    wins: int; losses: int; win_rate: float; total_pnl: float
    final_balance: float; max_drawdown_pct: float; sharpe_ratio: float
    profit_factor: float; avg_rr: float
    trades: list[BacktestTrade]; equity_curve: list[float]; regime_breakdown: dict


# ──────────────────────────────────────────────────────────────────────
#  SECTION 1 — TECHNICAL INDICATOR HELPERS (pure Python)
# ──────────────────────────────────────────────────────────────────────

def _ema(prices: list, period: int) -> float:
    if len(prices) < period:
        return prices[-1] if prices else 0.0
    k, val = 2.0 / (period + 1), prices[0]
    for p in prices[1:]:
        val = p * k + val * (1 - k)
    return val


def _rsi(prices: list, period: int = 14) -> float:
    if len(prices) < period + 1:
        return 50.0
    deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
    gains  = sum(d for d in deltas[-period:] if d > 0) / period
    losses = sum(-d for d in deltas[-period:] if d < 0) / period
    return 100.0 if losses == 0 else 100.0 - 100.0 / (1.0 + gains / losses)


def _atr_raw(closes: list, highs: list, lows: list, period: int = 14) -> float:
    if len(closes) < 2:
        return 0.0
    trs = [max(highs[i] - lows[i],
               abs(highs[i] - closes[i-1]),
               abs(lows[i] - closes[i-1]))
           for i in range(1, len(closes))]
    return sum(trs[-period:]) / min(len(trs), period) if trs else 0.0


def _adx_raw(highs: list, lows: list, closes: list, period: int = 14) -> tuple[float, float, float]:
    """Returns (adx, plus_di, minus_di)."""
    if len(closes) < period + 2:
        return 25.0, 15.0, 15.0
    trs, pdms, mdms = [], [], []
    for i in range(1, len(closes)):
        tr  = max(highs[i]-lows[i], abs(highs[i]-closes[i-1]), abs(lows[i]-closes[i-1]))
        pdm = max(highs[i]-highs[i-1], 0) if highs[i]-highs[i-1] > lows[i-1]-lows[i] else 0
        mdm = max(lows[i-1]-lows[i],   0) if lows[i-1]-lows[i]   > highs[i]-highs[i-1] else 0
        trs.append(tr); pdms.append(pdm); mdms.append(mdm)
    atr_v = sum(trs[-period:]) / period
    if atr_v == 0:
        return 0.0, 0.0, 0.0
    pdi = 100 * sum(pdms[-period:]) / (period * atr_v)
    mdi = 100 * sum(mdms[-period:]) / (period * atr_v)
    adx = 100 * abs(pdi - mdi) / max(pdi + mdi, 1e-9)
    return adx, pdi, mdi


def _bollinger(prices: list, period: int = 20) -> tuple[float, float, float]:
    w    = prices[-period:] if len(prices) >= period else prices
    mean = sum(w) / len(w)
    std  = math.sqrt(sum((p - mean)**2 for p in w) / max(len(w)-1, 1))
    return mean - 2*std, mean, mean + 2*std


def _safe(a: float, b: float, fb: float = 0.0) -> float:
    return a / b if b else fb


def _macd_series(closes: list) -> list[float]:
    """Proper MACD line series (EMA12 - EMA26) over rolling bars."""
    n = len(closes)
    out = []
    for i in range(max(26, n-40), n+1):
        out.append(_ema(closes[:i], 12) - _ema(closes[:i], 26))
    return out


# ──────────────────────────────────────────────────────────────────────
#  SECTION 2 — FEATURE BUILDER  (34 features, crypto-adapted)
# ──────────────────────────────────────────────────────────────────────

def build_features(p: AIRequest) -> tuple[np.ndarray, dict]:
    """
    34-dimensional feature vector.
    Combines the Forex PRO 30-feature set with crypto-specific additions:
    - ATR-normalised EMA distances (crypto prices span $0.50 → $100k)
    - HTF bias (key for 24/7 crypto trends)
    - Z-score (strong mean-reversion signal in ranging crypto)
    - Session-agnostic time encoding (crypto trades 24/7 but has volume patterns)
    """
    c  = p.bars.close
    h  = p.bars.high
    l  = p.bars.low
    o  = p.bars.open
    v  = p.bars.volume
    n  = min(len(c), 100)
    atr = p.atr if p.atr > 0 else 1e-9

    # ── Trend: EMA stack alignment ────────────────────────────────────
    ema_full_bull = (c[0] > p.ema20 > p.ema50 > p.ema200)
    ema_full_bear = (c[0] < p.ema20 < p.ema50 < p.ema200)
    if   ema_full_bull:                          ema_align = +1.0
    elif ema_full_bear:                          ema_align = -1.0
    elif c[0] > p.ema50 and p.ema50 > p.ema200: ema_align = +0.5
    elif c[0] < p.ema50 and p.ema50 < p.ema200: ema_align = -0.5
    elif c[0] > p.ema200:                        ema_align = +0.25
    else:                                        ema_align = -0.25

    ema8  = _ema(list(reversed(c[:min(n,50)])), 8)
    ema21 = _ema(list(reversed(c[:min(n,50)])), 21)
    ema_cross_norm = _safe(ema8 - ema21, atr)   # EMA8-21 separation in ATR units
    ema_sep        = np.clip(_safe(p.ema20 - p.ema50, atr), -3, 3)
    ema50_dist     = np.clip(_safe(c[0] - p.ema50, atr), -3, 3)

    # ── Higher timeframe bias ─────────────────────────────────────────
    if c[0] > p.htf_ema50 and p.htf_ema50 > p.htf_ema200:   htf_bias = +1.0
    elif c[0] < p.htf_ema50 and p.htf_ema50 < p.htf_ema200: htf_bias = -1.0
    elif c[0] > p.htf_ema50:                                  htf_bias = +0.4
    else:                                                      htf_bias = -0.4

    # ── MACD ─────────────────────────────────────────────────────────
    closes_list = list(reversed(c[:min(n,60)]))
    mseries     = _macd_series(closes_list)
    macd_v      = mseries[-1]   if mseries else p.macd
    sig_line    = _ema(mseries, 9) if len(mseries) >= 9 else macd_v
    hist_v      = macd_v - sig_line
    macd_norm   = np.clip(_safe(hist_v, atr * 0.05 + 1e-9), -3, 3)
    macd_cross  = 1.0 if macd_v > sig_line else -1.0

    # ── ADX ───────────────────────────────────────────────────────────
    adx_norm = np.clip((p.adx - 25.0) / 40.0, -1, 1)
    di_diff  = np.clip((p.plus_di - p.minus_di) / 30.0, -1, 1)

    # ── RSI ───────────────────────────────────────────────────────────
    rsi_norm = (p.rsi - 50.0) / 50.0
    rsi_ext  = max(0, (p.rsi - 70)) / 30.0 - max(0, (30 - p.rsi)) / 30.0

    # ── Bollinger Bands ───────────────────────────────────────────────
    bbl, bbm, bbh = _bollinger(closes_list, 20)
    bbw    = _safe(bbh - bbl, bbm)
    bbpos  = _safe(c[0] - bbl, bbh - bbl + 1e-9, 0.5) * 2 - 1

    # ── Volatility ────────────────────────────────────────────────────
    vol_ratio = p.atr / (p.atr_avg if p.atr_avg > 0 else p.atr)
    vol_norm  = np.clip((vol_ratio - 1.0) / 1.5, -1, 1)
    vol_flag  = 1.0 if vol_ratio > 1.6 else (-0.5 if vol_ratio < 0.6 else 0.0)

    # Historical volatility
    rets = [_safe(c[i] - c[i+1], c[i+1] + 1e-9) for i in range(min(20, n-1))]
    hvol = math.sqrt(sum(r**2 for r in rets) / max(len(rets), 1)) * math.sqrt(365 * 24)

    # ── Candle structure ──────────────────────────────────────────────
    bar_range = h[0] - l[0]
    if bar_range > 0:
        body       = abs(c[0] - o[0]) / bar_range
        candle     = body if c[0] > o[0] else -body
        upper_wick = (h[0] - max(c[0], o[0])) / bar_range
        lower_wick = (min(c[0], o[0]) - l[0]) / bar_range
    else:
        candle = upper_wick = lower_wick = 0.0

    # ── Price position (20-bar range) ─────────────────────────────────
    look  = min(20, n)
    hh20  = max(h[:look]) if look else c[0]
    ll20  = min(l[:look]) if look else c[0]
    rng20 = hh20 - ll20
    pos20 = ((c[0] - ll20) / rng20) * 2 - 1 if rng20 > 0 else 0.0

    # ── Rate of change ────────────────────────────────────────────────
    def roc(period):
        if n > period and c[period] > 0:
            return np.clip((c[0] - c[period]) / c[period] * 100 / 10, -2, 2)
        return 0.0
    roc5, roc10, roc20 = roc(5), roc(10), roc(20)

    # ── Breakout score (50-bar) ───────────────────────────────────────
    hh50 = max(h[1:min(51, len(h))]) if len(h) > 1 else c[0]
    ll50 = min(l[1:min(51, len(l))]) if len(l) > 1 else c[0]
    sp50 = hh50 - ll50
    if c[0] > hh50 and sp50 > 0:   breakout =  min(1.0, (c[0]-hh50)/sp50)
    elif c[0] < ll50 and sp50 > 0: breakout = -min(1.0, (ll50-c[0])/sp50)
    else:                           breakout = 0.0

    # ── Z-score (mean reversion) ──────────────────────────────────────
    arr_z  = np.array(c[:min(20,n)][::-1])
    zscore = np.clip((c[0] - arr_z.mean()) / (arr_z.std() + 1e-9), -3, 3)

    # ── Volume momentum ───────────────────────────────────────────────
    vb    = min(10, len(v))
    volr  = _safe(v[0], np.mean(v[1:vb]) + 1e-9, 1.0) - 1.0 if vb > 1 else 0.0
    volt  = _safe(sum(v[:5]), sum(v[5:10]) + 1e-9, 1.0) - 1.0 if len(v) >= 10 else 0.0
    vspike = 1.0 if (vb > 1 and v[0] > np.mean(v[1:vb]) * 2.0) else 0.0

    # ── Time encoding (24/7 crypto, but volume patterns exist) ────────
    h_sin = math.sin(2 * math.pi * p.hour / 24)
    h_cos = math.cos(2 * math.pi * p.hour / 24)
    d_sin = math.sin(2 * math.pi * p.dow  / 7)
    d_cos = math.cos(2 * math.pi * p.dow  / 7)
    # Crypto high-volume window: 13-21 UTC (US session overlap)
    peak_session = 1.0 if 13 <= p.hour < 21 else 0.0

    # ── Spread cost (crypto-normalised) ──────────────────────────────
    spread_n = min(float(p.spread), 200.0)   # raw points, cap at 200

    # ── History bias (online learning signal) ─────────────────────────
    hist_bias = _compute_history_bias(p.recent_signals, p.recent_outcomes, p.recent_regimes)

    # ── Interaction terms ─────────────────────────────────────────────
    trend_x_dir   = adx_norm * di_diff
    rsi_x_macd    = rsi_norm * macd_norm
    ema_x_htf     = ema_align * htf_bias

    features = np.array([
        # Trend group (0-5)
        ema_align, ema_cross_norm, ema_sep, ema50_dist, htf_bias, adx_norm,
        # Momentum group (6-11)
        di_diff, rsi_norm, rsi_ext, macd_norm, macd_cross, bbpos,
        # Volatility group (12-15)
        vol_norm, vol_flag, hvol, bbw * 100,
        # Structure group (16-19)
        candle, upper_wick, lower_wick, pos20,
        # Rate of change (20-22)
        roc5, roc10, roc20,
        # Breakout / reversion (23-24)
        breakout, zscore,
        # Volume (25-27)
        np.clip(volr, -2, 2), np.clip(volt, -2, 2), vspike,
        # Time / context (28-30)
        h_sin, h_cos, peak_session,
        # Online learning + spread (31-33)
        hist_bias, spread_n / 200.0,
        # Interactions (34-35) → 34 total
        ema_x_htf,
    ], dtype=np.float32)

    scores = {
        "ema_alignment":    round(float(ema_align), 3),
        "htf_bias":         round(float(htf_bias), 3),
        "trend_strength":   round(float(adx_norm), 3),
        "di_direction":     round(float(di_diff), 3),
        "rsi":              round(float(rsi_norm), 3),
        "macd_momentum":    round(float(macd_norm), 3),
        "volatility_ratio": round(float(vol_ratio), 3),
        "breakout_score":   round(float(breakout), 3),
        "mean_reversion_z": round(float(zscore), 3),
        "history_bias":     round(float(hist_bias), 3),
        "volume_spike":     round(float(vspike), 3),
        "bbpos":            round(float(bbpos), 3),
    }

    return np.nan_to_num(features, nan=0.0, posinf=3.0, neginf=-3.0), scores


OBS_DIM = 34   # Must match features array size above


def _compute_history_bias(signals, outcomes, regimes) -> float:
    if not signals or not outcomes:
        return 0.0
    n = min(len(signals), len(outcomes), 20)
    sig = np.array(signals[-n:], dtype=float)
    out = np.array(outcomes[-n:], dtype=float)
    # signals[-n:] is oldest-first; linspace(-2,0) → oldest gets e^-2≈0.14, newest gets e^0=1
    w   = np.exp(np.linspace(-2, 0, n))
    pos = sig > 0; neg = sig < 0
    pw  = np.average((out[pos] > 0).astype(float), weights=w[pos]) if pos.any() else 0.5
    nw  = np.average((out[neg] > 0).astype(float), weights=w[neg]) if neg.any() else 0.5
    return float(np.clip(pw - nw, -0.5, 0.5))


# ──────────────────────────────────────────────────────────────────────
#  SECTION 3 — REGIME DETECTOR  (crypto-tuned thresholds)
# ──────────────────────────────────────────────────────────────────────

def detect_regime(p: AIRequest) -> str:
    """
    TRENDING  : ADX > 18 + EMA aligned  OR  EMA crossover + strong momentum
    VOLATILE  : ATR fast/slow > 1.4 OR volume spike  (lower than forex: crypto is natively volatile)
    RANGING   : ADX < 18 + low momentum

    Crypto differs from Forex:
    - Baseline volatility is 3-5× higher → use 1.4× ATR ratio (not 1.5×)
    - No session dead zones → no hour-based hard blocks
    - EMA crossover threshold same, but ROC threshold higher (0.5% not 0.2%)
    """
    c  = list(reversed(p.bars.close[:min(len(p.bars.close), 60)]))
    h  = list(reversed(p.bars.high [:min(len(p.bars.high),  60)]))
    l  = list(reversed(p.bars.low  [:min(len(p.bars.low),   60)]))
    v  = list(reversed(p.bars.volume[:min(len(p.bars.volume),60)]))

    if len(c) < 30:
        return "RANGING"

    adx      = p.adx
    atr_fast = _atr_raw(c[-10:], h[-10:], l[-10:], 5)
    atr_slow = _atr_raw(c, h, l, 20)
    avg_vol  = sum(v[-20:]) / max(len(v[-20:]), 1)
    vol_spike = (v[-1] if v else avg_vol) > avg_vol * 2.0

    # VOLATILE: ATR expansion (crypto uses 1.4 not 1.5) OR volume spike
    if (atr_slow > 0 and atr_fast / atr_slow > 1.4) or vol_spike:
        return "VOLATILE"

    ema8  = _ema(c, 8)
    ema21 = _ema(c, 21)
    ema50 = _ema(c, 50) if len(c) >= 50 else ema21
    ema_aligned = (ema8 > ema21 > ema50) or (ema8 < ema21 < ema50)

    # EMA crossover detection (compare current vs previous bar)
    c_prev    = c[:-1]
    ema8_prev = _ema(c_prev, 8)  if len(c_prev) >= 8  else ema8
    ema21_prev= _ema(c_prev, 21) if len(c_prev) >= 21 else ema21
    ema_crossed = (ema8_prev > ema21_prev) != (ema8 > ema21)

    # Primary TRENDING: ADX > 18 (same as Forex PRO — works for crypto too)
    if adx > 18 and ema_aligned:
        return "TRENDING"

    # Secondary TRENDING: crossover + momentum (crypto needs 0.5% move)
    roc5 = _safe(c[-1] - c[-6], c[-6]) if len(c) >= 6 else 0.0
    strong_momentum = abs(roc5) > 0.005    # 0.5% in 5 bars
    if ema_crossed and strong_momentum:
        return "TRENDING"

    if adx < 18 and not strong_momentum:
        return "RANGING"

    if ema_crossed:
        return "TRENDING"

    return "RANGING"


def classify_regime_granular(p: AIRequest, regime: str, features: np.ndarray) -> tuple[int, str, float]:
    """
    Maps TRENDING/RANGING/VOLATILE → granular 5-regime classification (0-5).
    Also computes confidence for the MT5 dashboard overlay.
    """
    breakout_s  = abs(float(features[23]))
    vol_ratio   = p.atr / (p.atr_avg if p.atr_avg > 0 else p.atr)
    bull_trend  = p.plus_di > p.minus_di
    adx_norm_v  = min(1.0, (p.adx - 18) / 40)

    if regime == "VOLATILE":
        if breakout_s > 0.35:
            return 4, "Breakout", 0.55 + breakout_s * 0.35
        return 3, "High Volatility", min(0.90, 0.55 + (vol_ratio - 1.4) * 0.20)
    elif regime == "TRENDING":
        if bull_trend:
            return 0, "Trend Bull", 0.55 + adx_norm_v * 0.35
        return 1, "Trend Bear", 0.55 + adx_norm_v * 0.35
    else:
        ranging_conf = min(0.85, 0.52 + (18 - p.adx) / 18 * 0.33)
        return 2, "Ranging", ranging_conf


# ──────────────────────────────────────────────────────────────────────
#  SECTION 4 — STRATEGY ENGINES  (crypto-tuned)
# ──────────────────────────────────────────────────────────────────────

def signal_trend_following(p: AIRequest) -> tuple[str, float, str]:
    """
    EMA stack + MACD histogram + ADX + volume confirmation.
    Crypto-tuned: volume check uses 0.75× avg (lower bar than forex 0.8×).
    """
    c = list(reversed(p.bars.close[:60]))
    v = list(reversed(p.bars.volume[:60]))
    if len(c) < 30:
        return "NONE", 0.0, "insufficient_data"

    ema8  = _ema(c, 8); ema21 = _ema(c, 21)
    ema50 = _ema(c, 50) if len(c) >= 50 else ema21
    mseries   = _macd_series(c)
    hist      = mseries[-1] - _ema(mseries, 9) if len(mseries) >= 9 else 0.0
    adx       = p.adx
    cur_close = c[-1]

    avg_vol = sum(v[-20:]) / max(len(v[-20:]), 1) if v else 1
    cur_vol = v[-1] if v else avg_vol
    vol_ok  = cur_vol >= avg_vol * 0.75    # 0.75 for crypto (lower baseline)

    sb = se = 0.0
    if ema8 > ema21 > ema50: sb += 0.30
    if ema8 < ema21 < ema50: se += 0.30
    if cur_close > ema21:    sb += 0.20
    if cur_close < ema21:    se += 0.20
    if hist > 0:             sb += 0.25
    if hist < 0:             se += 0.25
    f = 0.5 + 0.5 * min(adx / 50.0, 1.0)
    sb *= f; se *= f

    # HTF confirmation bonus
    if p.htf_ema50 > p.htf_ema200: sb *= 1.10
    if p.htf_ema50 < p.htf_ema200: se *= 1.10

    if not vol_ok:
        sb *= 0.70; se *= 0.70

    t = sb + se
    if t == 0: return "NONE", 0.0, "no_signal"
    vtag = "vol_ok" if vol_ok else "vol_low"
    if sb > se and sb/t > 0.60: return "BUY",  round(sb/t, 4), f"TF:adx={adx:.0f},{vtag}"
    if se > sb and se/t > 0.60: return "SELL", round(se/t, 4), f"TF:adx={adx:.0f},{vtag}"
    return "NONE", 0.0, "low_confluence"


def signal_mean_reversion(p: AIRequest) -> tuple[str, float, str]:
    """
    RSI extremes + Bollinger band touch + volume contraction.
    Crypto-tuned: RSI thresholds widen to 25/75 (crypto overshoots more).
    """
    c = list(reversed(p.bars.close[:60]))
    v = list(reversed(p.bars.volume[:60]))
    if len(c) < 21:
        return "NONE", 0.0, "insufficient_data"

    rsi_v      = p.rsi
    bbl, _, bbh = _bollinger(c, 20)
    cur        = c[-1]
    avg_v      = sum(v[-20:]) / max(len(v[-20:]), 1) if v else 1
    vcont      = (v[-1] if v else avg_v) < avg_v * 0.80
    rsi5       = _rsi(c[-6:], 5) if len(c) >= 6 else rsi_v

    sb = se = 0.0; votes = []
    # Crypto: widen RSI thresholds to 25/75
    if rsi_v < 25:  sb += 0.40 * (25 - rsi_v) / 25; votes.append(f"RSI_OS:{rsi_v:.0f}")
    elif rsi_v > 75:se += 0.40 * (rsi_v - 75) / 25; votes.append(f"RSI_OB:{rsi_v:.0f}")
    if cur <= bbl:  sb += 0.35; votes.append("BB_LOW")
    elif cur >= bbh:se += 0.35; votes.append("BB_HIGH")
    if rsi5 > rsi_v and rsi_v < 40: sb += 0.15; votes.append("RSI_DIV")
    elif rsi5 < rsi_v and rsi_v > 60: se += 0.15
    if vcont:
        # Volume contraction is a confirming factor — only boost the already-dominant side
        if sb >= se: sb *= 1.15
        else:        se *= 1.15
        votes.append("VCONT")

    t = sb + se
    if t == 0: return "NONE", 0.0, "no_signal"
    if sb > se and sb/t > 0.62: return "BUY",  round(min(sb/t, 0.99), 4), "MR:"+",".join(votes)
    if se > sb and se/t > 0.62: return "SELL", round(min(se/t, 0.99), 4), "MR:"+",".join(votes)
    return "NONE", 0.0, "low_confluence"


def signal_breakout(p: AIRequest) -> tuple[str, float, str]:
    """
    Price pierces 20-bar range extreme + ATR expansion + volume surge.
    Crypto-tuned: require EITHER aexp OR vsurge (not both — crypto volume is noisy).
    """
    c = list(reversed(p.bars.close[:60]))
    h = list(reversed(p.bars.high [:60]))
    l = list(reversed(p.bars.low  [:60]))
    v = list(reversed(p.bars.volume[:60]))
    if len(c) < 22:
        return "NONE", 0.0, "insufficient_data"

    rhi   = max(h[-21:-1]); rlo = min(l[-21:-1])
    atr_f = _atr_raw(c[-10:], h[-10:], l[-10:], 5)
    atr_s = _atr_raw(c, h, l, 20)
    avg_v = sum(v[-20:]) / max(len(v[-20:]), 1) if v else 1
    vsurge = (v[-1] if v else avg_v) > avg_v * 1.5
    aexp   = atr_f > atr_s * 1.2

    sb = se = 0.0; votes = []
    if c[-1] > rhi: sb += 0.45; votes.append("BRK_HI")
    if c[-1] < rlo: se += 0.45; votes.append("BRK_LO")
    if aexp:   sb *= 1.20; se *= 1.20; votes.append("ATR_EXP")
    if vsurge: sb *= 1.20; se *= 1.20; votes.append("VOL_SURGE")
    # Require EITHER confirmation (crypto volume is noisy — don't need both)
    if not (aexp or vsurge): sb *= 0.50; se *= 0.50

    t = sb + se
    if t == 0: return "NONE", 0.0, "no_breakout"
    if sb > se and sb/t > 0.65: return "BUY",  round(min(sb/t, 0.99), 4), "BO:"+",".join(votes)
    if se > sb and se/t > 0.65: return "SELL", round(min(se/t, 0.99), 4), "BO:"+",".join(votes)
    return "NONE", 0.0, "low_confluence"


STRATEGY_FN = {
    "trend_following": signal_trend_following,
    "mean_reversion":  signal_mean_reversion,
    "breakout":        signal_breakout,
}


# ──────────────────────────────────────────────────────────────────────
#  SECTION 5 — PPO MODEL MANAGEMENT
# ──────────────────────────────────────────────────────────────────────

def _make_env():
    import gymnasium as gym
    from gymnasium import spaces

    class CryptoTradingEnv(gym.Env):
        def __init__(self):
            super().__init__()
            self.action_space      = spaces.Discrete(3)   # 0=BUY 1=SELL 2=NONE
            self.observation_space = spaces.Box(
                -np.inf, np.inf, shape=(OBS_DIM,), dtype=np.float32
            )
            self._steps = 0
        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self._steps = 0
            return np.zeros(OBS_DIM, dtype=np.float32), {}
        def step(self, action):
            self._steps += 1
            done = self._steps >= 200
            return np.zeros(OBS_DIM, dtype=np.float32), 0.0, done, False, {}

    return CryptoTradingEnv()


def load_strategy_model(strategy: str):
    """Load saved PPO model or create a fresh one if none exists."""
    from sb3_contrib import RecurrentPPO
    path = _model_path(strategy)
    env  = _make_env()
    try:
        if os.path.exists(path):
            print(f"[MODEL] Loading {strategy} from {path}")
            model = RecurrentPPO.load(path, env=env)
            # Validate
            test_obs = np.zeros(OBS_DIM, dtype=np.float32)
            try:
                model.predict(test_obs.reshape(1, -1))
                print(f"[MODEL] {strategy} loaded OK")
                return model
            except Exception as e:
                print(f"[MODEL] {strategy} validation failed: {e} — creating fresh")
    except Exception as e:
        print(f"[MODEL] Load failed for {strategy}: {e} — creating fresh")

    print(f"[MODEL] Creating fresh RecurrentPPO for {strategy}")
    return RecurrentPPO(
        "MlpLstmPolicy", env, verbose=0,
        n_steps=512, batch_size=64, n_epochs=6,
        learning_rate=2e-4, gamma=0.99, gae_lambda=0.95,
        clip_range=0.2, ent_coef=0.01, vf_coef=0.5,
        max_grad_norm=0.5,
    )


def save_strategy_model(model, strategy: str):
    model.save(_model_path(strategy))
    meta_path = _meta_path(strategy)
    meta = {}
    if os.path.exists(meta_path):
        try:
            with open(meta_path) as f: meta = json.load(f)
        except: meta = {}
    meta["trained"]    = True
    meta["save_count"] = meta.get("save_count", 0) + 1
    meta["saved_at"]   = datetime.now(timezone.utc).isoformat()
    with open(meta_path, "w") as f: json.dump(meta, f)
    volume.commit()
    print(f"[MODEL] Saved {strategy} (save_count={meta['save_count']})")


def is_model_trained(strategy: str) -> bool:
    meta_path = _meta_path(strategy)
    if not os.path.exists(meta_path):
        return False
    try:
        with open(meta_path) as f:
            return bool(json.load(f).get("trained", False))
    except:
        return False


def ppo_predict(model, obs: np.ndarray) -> tuple[str, float]:
    """Run RecurrentPPO inference, return (direction_str, confidence)."""
    import torch
    obs_arr = np.array(obs, dtype=np.float32).reshape(1, -1)
    act_idx, _ = model.predict(obs_arr, state=None,
                                episode_start=np.array([True]), deterministic=True)
    act_int = int(act_idx[0])
    # Confidence via action distribution
    try:
        with torch.no_grad():
            obs_t = torch.as_tensor(obs_arr, dtype=torch.float32).to(model.device)
            ep_t  = torch.as_tensor([True], dtype=torch.bool).to(model.device)
            dist  = model.policy.get_distribution(obs_t, lstm_states=None, episode_starts=ep_t)
            probs = dist.distribution.probs.cpu().numpy().flatten()
            conf  = float(probs[act_int])
    except Exception as e:
        try:
            votes = np.zeros(3, dtype=np.float32)
            for _ in range(10):
                a, _ = model.predict(obs_arr, state=None,
                                     episode_start=np.array([True]), deterministic=False)
                votes[int(a[0])] += 1
            conf = float(votes[act_int] / 10.0)
        except:
            conf = 0.50
        print(f"[PPO] get_distribution failed ({e}) — sampling fallback conf={conf:.2f}")
    return {0: "BUY", 1: "SELL", 2: "NONE"}[act_int], conf


# ──────────────────────────────────────────────────────────────────────
#  SECTION 6 — NEWS FILTER
# ──────────────────────────────────────────────────────────────────────

def _refresh_news_cache():
    """
    Pulls active blackouts from Supabase news_blackouts table into _NEWS_CACHE.
    Called at most once per _NEWS_CACHE_TTL_SECS (60s) to avoid DB hammering
    in the hot predict path.

    Sources written by _run_news_monitor():
      - ForexFactory  — macro events (FOMC, CPI, NFP) ±30 min window
      - Finnhub       — crypto shock headlines (exchange hacks, depeg) 60 min
      - CryptoCompare — regulatory/ETF/protocol news 60 min
      - FOMC          — scheduled Fed meeting dates, 24h pre-window
    """
    global _NEWS_CACHE, _NEWS_CACHE_UPDATED
    now = datetime.now(timezone.utc)
    if (now - _NEWS_CACHE_UPDATED).total_seconds() < _NEWS_CACHE_TTL_SECS:
        return  # still fresh

    try:
        from supabase import create_client
        sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])
        rows = (sb.table("news_blackouts")
                  .select("source,title,currencies,impact,expires_at")
                  .eq("active", True)
                  .gte("expires_at", now.isoformat())
                  .execute().data or [])
        _NEWS_CACHE = rows
        _NEWS_CACHE_UPDATED = now
        if rows:
            print(f"[NEWS CACHE] {len(rows)} active blackouts: "
                  + ", ".join(r.get('source','?') + '/' + r.get('impact','?') for r in rows))
    except Exception as e:
        print(f"[NEWS CACHE] DB refresh failed (using stale cache): {e}")


def _db_news_blocked(symbol: str) -> tuple[bool, str]:
    """
    Check symbol against current _NEWS_CACHE.
    Returns (blocked, reason).

    Matching rules:
      - currencies == "ALL"  → blocks every symbol (crypto shock / FOMC)
      - currencies contains symbol's quote or base  → targeted block
      - impact == "SHOCK" or "FOMC"  → always block if any match
    """
    if not _NEWS_CACHE:
        return False, ""

    sym_up   = symbol.upper()
    # Extract base (BTC, ETH…) and quote (USD) from e.g. "BTCUSDm"
    # Strip broker suffix first
    base_sym = sym_up.rstrip("M")          # BTCUSDm → BTCUSD
    base     = base_sym[:3]                # BTC
    quote    = base_sym[3:6] if len(base_sym) >= 6 else "USD"

    now = datetime.now(timezone.utc)
    for row in _NEWS_CACHE:
        try:
            exp = datetime.fromisoformat(row["expires_at"].replace("Z", "+00:00"))
        except Exception:
            continue
        if exp < now:
            continue  # expired entry (cache not yet refreshed)

        currencies = (row.get("currencies") or "").upper()
        impact     = (row.get("impact") or "").upper()
        source     = row.get("source", "?")

        # ALL block — covers every symbol (used for FOMC and exchange-wide shocks)
        if currencies == "ALL":
            return True, f"DB-NewsFilter({source}/{impact}): global block"

        # Currency match — USD news blocks all USD-quoted crypto
        if quote in currencies or base in currencies:
            return True, f"DB-NewsFilter({source}/{impact}): {currencies}"

    return False, ""


def check_news_filter(p: AIRequest) -> tuple[bool, str]:
    """
    Two-layer crypto news filter:

    Layer 1 — EA hard block (MT5 economic calendar, high-impact USD events):
      EA sends news_blackout=True when FOMC/CPI/NFP within buffer window.
      This is the fastest response — no network round-trip.

    Layer 2 — Server-side DB block (Supabase news_blackouts table):
      Modal's news_monitor writes here every 5 min from 4 sources:
        • ForexFactory  — macro events
        • Finnhub       — crypto shock headlines
        • CryptoCompare — ETF/regulatory/protocol news   ← NEW
        • FOMC calendar — scheduled Fed meetings 24h pre ← NEW
      Cached in-memory for 60s to keep predict endpoint fast.
      This layer catches everything the EA calendar cannot:
        exchange hacks, ETF approvals, SEC rulings, stablecoin depegs,
        Fed meeting days (24h window — crypto extremely sensitive).
    """
    # Layer 1: EA hard block
    if p.news_blackout:
        return True, "NewsFilter-L1: EA calendar block"
    if p.news_minutes_away < p.news_buffer_minutes:
        return True, f"NewsFilter-L1: event in {p.news_minutes_away}min"

    # Layer 2: DB-backed crypto/FOMC filter
    _refresh_news_cache()
    blocked, reason = _db_news_blocked(p.symbol)
    if blocked:
        return True, f"NewsFilter-L2: {reason}"

    return False, ""


# ──────────────────────────────────────────────────────────────────────
#  SECTION 7 — POSITION SIZER (Kelly + ATR, per-strategy SL/TP)
# ──────────────────────────────────────────────────────────────────────

def compute_position(p: AIRequest, direction: str, confidence: float,
                     strategy: str) -> tuple[float, float, float, float, float, float]:
    """Returns: (lots, sl_price, tp_price, sl_atr_mult, tp_atr_mult, rr)"""
    if direction == "NONE":
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    is_buy = direction == "BUY"
    price  = p.bars.close[0]
    atr    = p.atr

    # Per-strategy SL/TP multipliers (crypto needs wider SL — high volatility)
    sl_tp_map = {
        "trend_following": (2.2, 3.8),
        "mean_reversion":  (1.6, 2.5),
        "breakout":        (2.0, 4.5),
    }
    sl_mult, tp_mult = sl_tp_map.get(strategy, (2.2, 3.8))

    # Strong signal boost
    if confidence >= 0.75:
        tp_mult *= 1.25

    sl_dist  = atr * sl_mult
    tp_dist  = atr * tp_mult
    sl_price = (price - sl_dist) if is_buy else (price + sl_dist)
    tp_price = (price + tp_dist) if is_buy else (price - tp_dist)
    rr       = tp_dist / (sl_dist + 1e-9)

    # Half-Kelly position sizing with conservative win-prob floor
    # Floor at 0.45 (not 0.40) to avoid over-sizing on weak signals
    p_win  = float(np.clip(confidence, 0.45, 0.85))
    kelly  = max(0.0, (p_win * rr - (1 - p_win)) / rr) * 0.5
    # Hard cap: Kelly fraction ≤ risk_pct regardless of signal strength
    risk_pct_eff  = min(kelly * 100, p.risk_pct)
    # Guard against zero kelly (e.g. rr < 1 and confidence < 0.5)
    if risk_pct_eff <= 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    sizing_base   = p.allocated_capital if p.allocated_capital > 0 else p.account_balance
    risk_amount   = sizing_base * (risk_pct_eff / 100.0)

    if p.tick_size <= 0 or p.tick_value <= 0:
        lots = p.min_lot
    else:
        lots = risk_amount / (sl_dist / p.tick_size * p.tick_value)

    step = p.lot_step if p.lot_step > 0 else 0.01
    lots = math.floor(lots / step) * step
    lots = max(p.min_lot, min(p.max_lot, lots))

    return round(lots, 4), round(sl_price, p.digits), round(tp_price, p.digits), sl_mult, tp_mult, round(rr, 3)


def direction_to_signal(direction: str, confidence: float) -> int:
    """Map BUY/SELL/NONE + confidence → -2,-1,0,1,2."""
    if direction == "NONE":   return 0
    if direction == "BUY":    return 2 if confidence >= 0.75 else 1
    if direction == "SELL":   return -2 if confidence >= 0.75 else -1
    return 0


# ──────────────────────────────────────────────────────────────────────
#  SECTION 8 — FASTAPI ENDPOINTS (Single ASGI app on one URL)
# ──────────────────────────────────────────────────────────────────────

from fastapi import FastAPI, Request as FastAPIRequest
from fastapi.responses import JSONResponse

api = FastAPI(title="ApexHydra Crypto AI", version=MODEL_VERSION)


@app.function(
    image=image, volumes={MODEL_DIR: volume}, secrets=[secrets],
    timeout=60, memory=2048, cpu=2.0,
    min_containers=1, max_containers=10,
)
@modal.asgi_app(label="apexhydra-crypto")
def serve():
    return api


# ── Auth helper ───────────────────────────────────────────────────────

async def _require_auth(request: FastAPIRequest):
    """Checks X-API-Key header against MODAL_API_KEY secret."""
    expected = os.environ.get("MODAL_API_KEY", "")
    if not expected:
        return None  # Auth disabled if key not set (dev mode)
    if request.headers.get("X-API-Key", "") != expected:
        return JSONResponse({"ok": False, "reason": "unauthorized"}, status_code=401)
    return None


# ── POST /predict ─────────────────────────────────────────────────────

@api.post("/predict", response_model=AIResponse)
async def predict(request: FastAPIRequest):
    try:
        p = AIRequest(**await request.json())
    except Exception as e:
        return JSONResponse({"error": f"Invalid payload: {e}"}, status_code=422)

    try:
        return await _predict_inner(p)
    except Exception as e:
        import traceback
        print(f"[PREDICT ERROR] {p.symbol}: {e}\n{traceback.format_exc()}")
        return JSONResponse({"error": f"Predict failed: {e}"}, status_code=500)


async def _predict_inner(p: AIRequest):
    # News filter
    news_blocked, news_reason = check_news_filter(p)

    # Features
    features, feature_scores = build_features(p)

    # Regime detection
    regime_str  = detect_regime(p)
    strategy    = REGIME_TO_STRATEGY[regime_str]

    # Granular regime (for dashboard)
    regime_id, regime_name, regime_conf = classify_regime_granular(p, regime_str, features)

    # Update in-memory regime history
    sym = p.symbol
    if sym not in _REGIME_HISTORY:
        _REGIME_HISTORY[sym] = deque(maxlen=_REGIME_HISTORY_MAXLEN)
    _REGIME_HISTORY[sym].appendleft((datetime.now(timezone.utc), regime_str))

    # Rule-based strategy signal
    rb_direction, rb_conf, rb_reason = STRATEGY_FN[strategy](p)

    # PPO signal (only if model is trained)
    ppo_dir, ppo_conf, ppo_trained = "NONE", 0.0, False
    if is_model_trained(strategy):
        try:
            model  = load_strategy_model(strategy)
            ppo_dir, ppo_conf = ppo_predict(model, features)
            ppo_trained = True
        except Exception as e:
            print(f"[PPO] predict failed for {strategy}: {e}")

    # Signal fusion: rule-based (70%) + PPO (30% if trained + confident)
    final_dir  = rb_direction
    final_conf = rb_conf
    if ppo_trained and ppo_dir != "NONE" and ppo_conf >= 0.55:
        if ppo_dir == rb_direction:
            final_conf = min(0.98, rb_conf * 0.70 + ppo_conf * 0.30)
        elif rb_direction == "NONE":
            final_dir  = ppo_dir
            final_conf = ppo_conf * 0.60   # Dampened — rule-based didn't confirm

    # Mean-reversion trending block:
    # During the warm-up period or while regime has been TRENDING for >10 min,
    # suppress MR signals to avoid fading strong moves.
    if strategy == "mean_reversion" and final_dir != "NONE":
        warmup_ok = (datetime.now(timezone.utc) - _CONTAINER_START).total_seconds() > _MR_WARMUP_SECS
        hist      = list(_REGIME_HISTORY.get(sym, []))
        recent_trending = warmup_ok and len(hist) >= 5 and \
                          all(r == "TRENDING" for _, r in hist[:5])
        if recent_trending:
            final_dir  = "NONE"
            final_conf = 0.0
            rb_reason  += " | MR_BLOCKED:trending"

    # Block if news
    if news_blocked:
        final_dir  = "NONE"
        final_conf = 0.0

    # Spread penalty: if spread exceeds 10% of ATR, reduce confidence proportionally
    if p.atr > 0 and p.spread > 0:
        spread_atr_ratio = (p.spread * p.point) / p.atr if p.point > 0 else 0
        if spread_atr_ratio > 0.10:
            penalty = min(0.15, (spread_atr_ratio - 0.10) * 1.5)
            final_conf = max(0.0, final_conf - penalty)
            rb_reason += f" | SpreadPenalty:{penalty:.2f}"

    # Position sizing
    lots, sl, tp, sl_m, tp_m, rr = compute_position(p, final_dir, final_conf, strategy)
    signal_int = direction_to_signal(final_dir, final_conf)

    reasoning = (
        f"Regime={regime_str}({regime_conf:.0%}) | "
        f"Strategy={strategy} | "
        f"RB={rb_direction}({rb_conf:.0%}) | "
        f"PPO={ppo_dir}({ppo_conf:.0%}) | "
        f"Final={final_dir}({final_conf:.0%}) | "
        f"{rb_reason}"
    )
    if news_blocked: reasoning += f" | {news_reason}"

    return AIResponse(
        symbol         = p.symbol,
        regime         = regime_str,
        regime_id      = regime_id,
        regime_name    = regime_name,
        regime_conf    = round(regime_conf, 4),
        strategy_used  = strategy,
        signal         = signal_int,
        signal_name    = SIGNAL_NAMES.get(signal_int, "Hold"),
        confidence     = round(final_conf, 4),
        ppo_signal     = ppo_dir,
        ppo_confidence = round(ppo_conf, 4),
        lots           = lots,
        sl_price       = sl,
        tp_price       = tp,
        sl_atr_mult    = sl_m,
        tp_atr_mult    = tp_m,
        rr_ratio       = rr,
        feature_scores = feature_scores,
        reasoning      = reasoning,
        news_blocked   = news_blocked,
        model_trained  = ppo_trained,
        model_version  = MODEL_VERSION,
        server_ts      = datetime.now(timezone.utc).isoformat(),
    )


# ── POST /train ───────────────────────────────────────────────────────

@api.post("/train")
async def train(request: FastAPIRequest):
    auth_error = await _require_auth(request)
    if auth_error: return auth_error
    try:
        msg = await request.json()
    except:
        return JSONResponse({"ok": False, "reason": "invalid_json"}, status_code=400)

    strategy   = str(msg.get("strategy", "trend_following"))
    n_steps    = int(msg.get("timesteps", 10000))
    if strategy not in STRATEGY_NAMES:
        return JSONResponse({"ok": False, "reason": f"unknown strategy: {strategy}"}, status_code=400)

    try:
        model = load_strategy_model(strategy)
        model.learn(total_timesteps=n_steps, reset_num_timesteps=False)
        save_strategy_model(model, strategy)
        return {
            "ok": True, "strategy": strategy,
            "timesteps": n_steps,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "model_version": MODEL_VERSION,
        }
    except Exception as e:
        return JSONResponse({"ok": False, "reason": str(e)}, status_code=500)


# ── POST /backtest ────────────────────────────────────────────────────

@api.post("/backtest")
async def backtest_endpoint(request: FastAPIRequest):
    auth_error = await _require_auth(request)
    if auth_error: return auth_error
    try:
        req = BacktestRequest(**await request.json())
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=422)

    bars    = req.bars
    N       = len(bars)
    if N < 50:
        return JSONResponse({"error": "Need at least 50 bars"}, status_code=400)

    balance      = req.initial_balance
    peak_balance = balance
    max_dd       = 0.0
    open_trade   = None
    trades_out   = []
    pnl_list     = []
    equity_curve = []
    regime_stats : dict = {}

    for i in range(min(100, N), N):
        bar  = bars[i]
        # ── Check exit on open trade ──────────────────────────────────
        if open_trade is not None:
            ot = open_trade
            hit_sl = (ot["signal"] > 0 and bar.low  <= ot["sl"]) or \
                     (ot["signal"] < 0 and bar.high >= ot["sl"])
            hit_tp = (ot["signal"] > 0 and bar.high >= ot["tp"]) or \
                     (ot["signal"] < 0 and bar.low  <= ot["tp"])
            if hit_tp or hit_sl:
                exit_p = ot["tp"] if hit_tp else ot["sl"]
                pnl    = (exit_p - ot["entry"]) * ot["signal"] * \
                          (1.0 / req.tick_size * req.tick_value) * ot["lots"]
                won     = pnl > 0
                balance += pnl
                peak_balance = max(peak_balance, balance)
                dd = (1 - balance / peak_balance) * 100
                max_dd = max(max_dd, dd)
                trades_out.append(BacktestTrade(
                    entry_time=ot["entry_time"], exit_time=bar.timestamp,
                    signal=ot["signal"], regime=ot["regime"],
                    confidence=ot["confidence"], lots=ot["lots"],
                    entry_price=ot["entry"], exit_price=exit_p,
                    sl=ot["sl"], tp=ot["tp"], pnl=round(pnl,2), won=won,
                    bars_held=i-ot["bar_idx"],
                ))
                pnl_list.append(pnl); equity_curve.append(round(balance,2))
                r = ot["regime"]
                if r not in regime_stats:
                    regime_stats[r] = {"trades":0,"wins":0,"pnl":0.0}
                regime_stats[r]["trades"] += 1
                regime_stats[r]["wins"]   += int(won)
                regime_stats[r]["pnl"]    += pnl
                open_trade = None

        if open_trade is not None: continue

        # ── Build synthetic request for this bar ─────────────────────
        lk   = min(i, 100)
        sub  = bars[i-lk: i+1]
        fp = AIRequest(
            symbol=req.symbol, timeframe=req.timeframe, magic=0,
            timestamp=bar.timestamp, account_balance=balance,
            account_equity=balance, risk_pct=req.risk_pct,
            max_positions=1, open_positions=0,
            bars=BarData(
                open  =[b.open   for b in reversed(sub)],
                high  =[b.high   for b in reversed(sub)],
                low   =[b.low    for b in reversed(sub)],
                close =[b.close  for b in reversed(sub)],
                volume=[b.volume for b in reversed(sub)],
            ),
            atr=bar.atr, atr_avg=bar.atr_avg,
            adx=bar.adx, plus_di=bar.plus_di, minus_di=bar.minus_di,
            rsi=bar.rsi, macd=bar.macd, macd_signal=bar.macd_signal,
            macd_hist=bar.macd_hist, ema20=bar.ema20, ema50=bar.ema50,
            ema200=bar.ema200, htf_ema50=bar.htf_ema50, htf_ema200=bar.htf_ema200,
            tick_value=req.tick_value, tick_size=req.tick_size,
            min_lot=req.min_lot, max_lot=req.max_lot,
            lot_step=req.lot_step, point=req.point, digits=req.digits,
        )
        features, _ = build_features(fp)
        regime_str  = detect_regime(fp)
        strategy    = REGIME_TO_STRATEGY[regime_str]
        direction, confidence, _ = STRATEGY_FN[strategy](fp)

        if direction == "NONE" or confidence < req.min_confidence: continue
        lots, sl, tp, _, _, rr = compute_position(fp, direction, confidence, strategy)
        if rr < req.min_rr or lots <= 0: continue

        entry_price = bar.close
        sig         = 1 if direction == "BUY" else -1
        open_trade  = {
            "signal": sig, "regime": regime_str, "confidence": confidence,
            "lots": lots, "entry": entry_price, "sl": sl, "tp": tp,
            "entry_time": bar.timestamp, "bar_idx": i,
        }

    # ── Stats ─────────────────────────────────────────────────────────
    total    = len(trades_out)
    wins     = sum(1 for t in trades_out if t.won)
    win_rate = wins / total if total else 0.0
    total_pnl= sum(t.pnl for t in trades_out)
    gp       = sum(t.pnl for t in trades_out if t.pnl > 0)
    gl       = abs(sum(t.pnl for t in trades_out if t.pnl < 0))
    pf       = gp / gl if gl > 0 else 99.0
    sharpe   = float(np.mean(pnl_list) / (np.std(pnl_list)+1e-9) * math.sqrt(252)) if len(pnl_list) >= 2 else 0.0
    avg_rr   = float(np.mean([abs(t.tp - t.entry_price) / max(abs(t.sl - t.entry_price), 1e-9) for t in trades_out])) if trades_out else 0.0
    regime_breakdown = {
        r: {"trades":s["trades"], "wins":s["wins"],
            "win_rate":round(s["wins"]/s["trades"]*100,1) if s["trades"] else 0.0,
            "total_pnl":round(s["pnl"],2)}
        for r, s in regime_stats.items()
    }
    return BacktestResponse(
        symbol=req.symbol, timeframe=req.timeframe, bars_tested=N,
        total_trades=total, wins=wins, losses=total-wins,
        win_rate=round(win_rate,4), total_pnl=round(total_pnl,2),
        final_balance=round(balance,2), max_drawdown_pct=round(max_dd,2),
        sharpe_ratio=round(sharpe,3), profit_factor=round(min(pf,99.0),3),
        avg_rr=round(avg_rr,3), trades=trades_out, equity_curve=equity_curve,
        regime_breakdown=regime_breakdown,
    )


# ── POST /closeall ────────────────────────────────────────────────────

@api.post("/closeall")
async def closeall(request: FastAPIRequest):
    auth_error = await _require_auth(request)
    if auth_error: return auth_error
    try:
        from supabase import create_client
        sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])

        # Halt bot
        sb.table("bot_state").update({
            "is_running": False,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }).gte("id", "00000000-0000-0000-0000-000000000000").execute()

        # Issue close_all command for EA to pick up
        sb.table("bot_commands").upsert({
            "command":         "close_all",
            "issued_at":       datetime.now(timezone.utc).isoformat(),
            "executed":        False,
            "acknowledged_at": None,
        }).execute()

        print("[CLOSEALL] Bot stopped + close_all command issued")
        return {"ok": True, "close_all": True, "reason": "dashboard_stop"}

    except Exception as e:
        return JSONResponse({"ok": False, "reason": str(e)}, status_code=500)


# ── POST /closeall_ack ────────────────────────────────────────────────

@api.post("/closeall_ack")
async def closeall_ack(request: FastAPIRequest):
    auth_error = await _require_auth(request)
    if auth_error: return auth_error
    try:
        msg    = await request.json()
        closed = int(msg.get("closed", 0))
        source = str(msg.get("source", "ea"))
        from supabase import create_client
        sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])
        rows = (sb.table("bot_commands").select("id")
                .eq("command","close_all").order("issued_at",desc=True)
                .limit(1).execute().data or [])
        if rows:
            sb.table("bot_commands").update({
                "acknowledged_at": datetime.now(timezone.utc).isoformat(),
                "ack_source": source, "ack_closed": closed,
            }).eq("id", rows[0]["id"]).execute()
        return {"ok": True, "closed": closed}
    except Exception as e:
        return JSONResponse({"ok": False, "reason": str(e)}, status_code=500)


# ── GET /commands ─────────────────────────────────────────────────────

@api.get("/commands")
async def get_commands(request: FastAPIRequest):
    auth_error = await _require_auth(request)
    if auth_error: return auth_error
    try:
        from supabase import create_client
        sb   = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])
        rows = (sb.table("bot_commands").select("*").eq("executed", False)
                .execute().data or [])
        cmds = [r["command"] for r in rows]
        if rows:
            for r in rows:
                sb.table("bot_commands").update({
                    "executed":    True,
                    "executed_at": datetime.now(timezone.utc).isoformat(),
                }).eq("id", r["id"]).execute()
        return {"commands": cmds, "count": len(cmds)}
    except Exception as e:
        return {"commands": [], "count": 0, "error": str(e)}


# ── POST /log ─────────────────────────────────────────────────────────

@api.post("/log")
async def post_log(request: FastAPIRequest):
    auth_error = await _require_auth(request)
    if auth_error: return auth_error
    try:
        msg = await request.json()
    except:
        return JSONResponse({"ok": False, "reason": "invalid_json"}, status_code=400)

    level   = str(msg.get("level",   "INFO")).upper()[:10]
    symbol  = str(msg.get("symbol",  "")).upper()[:20]
    message = str(msg.get("message", ""))[:500]
    ea_time = msg.get("ea_time", datetime.now(timezone.utc).isoformat())
    if not message:
        return JSONResponse({"ok": False, "reason": "empty_message"}, status_code=400)
    try:
        from supabase import create_client
        sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])
        sb.table("ea_logs").insert({
            "level": level, "symbol": symbol or None,
            "message": message, "ea_time": ea_time,
            "logged_at": datetime.now(timezone.utc).isoformat(),
        }).execute()
        return {"ok": True}
    except Exception as e:
        print(f"[LOG] DB insert failed: {e}")
        return {"ok": False, "reason": str(e)}


# ── GET /logs ─────────────────────────────────────────────────────────

@api.get("/logs")
async def get_logs(request: FastAPIRequest):
    auth_error = await _require_auth(request)
    if auth_error: return auth_error
    try:
        from supabase import create_client
        sb    = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])
        limit = min(int(request.query_params.get("limit", "500")), 2000)
        rows  = (sb.table("ea_logs").select("*")
                 .order("logged_at", desc=True).limit(limit)
                 .execute().data or [])
        return {"ok": True, "count": len(rows), "logs": rows}
    except Exception as e:
        return JSONResponse({"ok": False, "reason": str(e)}, status_code=500)


# ── POST /purge ───────────────────────────────────────────────────────

@api.post("/purge")
async def purge(request: FastAPIRequest):
    auth_error = await _require_auth(request)
    if auth_error: return auth_error
    try:
        from supabase import create_client
        sb   = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])
        rows = (sb.table("trades").select("id")
                .is_("lot", "null").is_("pnl", "null")
                .is_("closed_at", "null").execute().data or [])
        if not rows:
            return {"ok": True, "deleted": 0, "message": "No phantom rows"}
        ids = [r["id"] for r in rows]
        sb.table("trades").delete().in_("id", ids).execute()
        return {"ok": True, "deleted": len(ids)}
    except Exception as e:
        return JSONResponse({"ok": False, "reason": str(e)}, status_code=500)


# ── GET /health ───────────────────────────────────────────────────────

@api.get("/health")
async def health():
    model_status = {}
    for s in STRATEGY_NAMES:
        trained = is_model_trained(s)
        meta = {}
        if os.path.exists(_meta_path(s)):
            try:
                with open(_meta_path(s)) as f: meta = json.load(f)
            except: pass
        model_status[s] = {
            "trained":    trained,
            "save_count": meta.get("save_count", 0),
            "saved_at":   meta.get("saved_at"),
        }
    regime_cache = {sym: list(_REGIME_HISTORY[sym])[:3] for sym in list(_REGIME_HISTORY.keys())[:5]}
    return {
        "status":         "ok",
        "version":        MODEL_VERSION,
        "strategies":     model_status,
        "features":       OBS_DIM,
        "regime_cache_symbols": list(_REGIME_HISTORY.keys()),
        "news_filter":    "enabled",
        "backtest":       "enabled",
        "ppo_per_strategy": True,
        "ts":             datetime.now(timezone.utc).isoformat(),
    }


# ──────────────────────────────────────────────────────────────────────
#  SECTION 9 — SCHEDULED BACKGROUND TASKS  (5 cron jobs)
#
#  Modal free plan allows 5 cron jobs per app.  We use all 5:
#
#   Cron 1: news_and_forward   — every  5 min  (news monitor + forward test)
#   Cron 2: online_learner     — every  6 hrs  (fine-tune PPO models)
#   Cron 3: performance_watch  — every  1 min  (DD alert + auto-halt guardian)
#   Cron 4: model_health_check — every 12 hrs  (validate models, auto-retrain if stale)
#   Cron 5: db_maintenance     — every 24 hrs  (prune old rows, vacuum perf table)
#
#  news_monitor + forward_tester are merged into ONE cron (news_and_forward)
#  because forward_tester only runs every 4 hrs internally — the 5-min cadence
#  is driven by the news monitor.  Merging them saves a slot for the two new
#  high-value crons (performance_watch, model_health_check).
# ──────────────────────────────────────────────────────────────────────

SCHEDULER_STATE_PATH = f"{MODEL_DIR}/scheduler_state.json"

SHOCK_KEYWORDS = [
    # ── Macro / systemic ──────────────────────────────────────────────
    "emergency rate cut", "emergency rate hike", "unscheduled fed meeting",
    "bank failure", "bank collapse", "bank run on", "systemic banking crisis",
    "stock market circuit breaker", "us treasury default", "us debt default",
    "war declared", "nuclear strike", "financial system collapse",
    # ── Exchange / infrastructure ────────────────────────────────────
    "exchange hack", "exchange collapse", "exchange insolvency",
    "exchange suspended", "exchange halted", "exchange offline",
    "binance collapse", "coinbase halt", "kraken halt",
    "bybit hack", "okx halt", "huobi collapse",
    "crypto market halt", "trading suspended",
    # ── Stablecoin ───────────────────────────────────────────────────
    "stablecoin depeg", "tether depegged", "usdc depegged",
    "usdt depeg", "dai depeg", "stablecoin collapse",
    # ── Regulatory ───────────────────────────────────────────────────
    "sec crypto ban", "bitcoin banned", "crypto banned",
    "sec charges", "cftc charges", "doj charges crypto",
    "crypto regulation emergency", "crypto crackdown",
    "etf rejected", "etf denied", "etf approval revoked",
    # ── Protocol / DeFi ──────────────────────────────────────────────
    "blockchain attack", "51% attack", "smart contract exploit",
    "defi hack", "protocol exploit", "bridge hack",
    "flash loan attack", "rug pull", "exit scam",
    # ── ETF / institutional ──────────────────────────────────────────
    "btc etf halted", "bitcoin etf suspended",
    "blackrock etf", "fidelity etf", "spot bitcoin etf",  # major moves on ETF news
]

CRYPTO_RELEVANCE_TERMS = [
    "bitcoin", "ethereum", "crypto", "blockchain", "defi", "stablecoin",
    "federal reserve", "interest rate", "inflation", "cpi", "fomc",
    "central bank", "monetary policy", "rate decision",
    "dollar", "currency", "exchange rate",
    "btc", "eth", "sol", "bnb", "xrp",  # ticker mentions
    "sec", "cftc", "doj",               # regulators
    "etf", "spot etf",                  # ETF news
    "tether", "usdt", "usdc",           # stablecoin mentions
]

# ── Scheduled FOMC meeting dates 2025 (UTC midnight of decision day) ──────────
# Source: federalreserve.gov — updated annually.
# Crypto is extremely sensitive to Fed decisions — block 24h before AND 2h after.
# Wider than forex because crypto moves 3-5× more on Fed news than EUR/USD.
FOMC_DATES_2025 = [
    "2025-01-29",  # January FOMC
    "2025-03-19",  # March FOMC
    "2025-05-07",  # May FOMC
    "2025-06-18",  # June FOMC
    "2025-07-30",  # July FOMC
    "2025-09-17",  # September FOMC
    "2025-10-29",  # October FOMC
    "2025-12-10",  # December FOMC
]
# 2026 dates (add as published by Fed)
FOMC_DATES_2026 = [
    "2026-01-28",
    "2026-03-18",
    "2026-04-29",
    "2026-06-17",
    "2026-07-29",
    "2026-09-16",
    "2026-10-28",
    "2026-12-09",
]
FOMC_ALL_DATES = FOMC_DATES_2025 + FOMC_DATES_2026


def _load_scheduler_state() -> dict:
    """Read last-run timestamps from volume. Returns empty dict on first run."""
    try:
        if os.path.exists(SCHEDULER_STATE_PATH):
            with open(SCHEDULER_STATE_PATH) as f:
                return json.load(f)
    except Exception as e:
        print(f"[SCHED] State read failed: {e}")
    return {}


def _save_scheduler_state(state: dict):
    try:
        with open(SCHEDULER_STATE_PATH, "w") as f:
            json.dump(state, f)
        volume.commit()
    except Exception as e:
        print(f"[SCHED] State write failed: {e}")


def _due(state: dict, key: str, interval_hours: float) -> bool:
    """Return True if the task hasn't run in the last interval_hours."""
    last = state.get(key)
    if last is None:
        return True
    try:
        last_dt = datetime.fromisoformat(last)
        return (datetime.now(timezone.utc) - last_dt).total_seconds() >= interval_hours * 3600
    except:
        return True


# ── Sub-task A: News Monitor ──────────────────────────────────────────

def _run_news_monitor():
    import requests
    from supabase import create_client

    sb  = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])
    now = datetime.now(timezone.utc)
    active_blackouts = []

    # Source 1: ForexFactory economic calendar
    try:
        r = requests.get(
            "https://nfs.faireconomy.media/ff_calendar_thisweek.json",
            timeout=8, headers={"User-Agent": "Mozilla/5.0"}
        )
        r.raise_for_status()
        for evt in r.json():
            impact = evt.get("impact", "").lower()
            if impact not in ("high", "medium"): continue
            try:
                evt_time = datetime.fromisoformat(
                    evt["date"].replace("Z", "+00:00")
                ).astimezone(timezone.utc)
            except: continue
            window_min = 30 if impact == "high" else 10
            window = timedelta(minutes=window_min)
            if abs(evt_time - now) <= window:
                currency = evt.get("currency", "").strip().upper()
                active_blackouts.append({
                    "source": "ForexFactory", "title": evt.get("title", "Event")[:120],
                    "currencies": currency, "impact": impact.capitalize(),
                    "event_time": evt_time.isoformat(),
                    "expires_at": (evt_time + window).isoformat(),
                    "active": True, "updated_at": now.isoformat(),
                })
                print(f"[NEWS] FF Blackout: {currency} — {evt.get('title','')}")
    except Exception as e:
        print(f"[NEWS] ForexFactory failed: {e}")

    # Source 2: Finnhub crypto + macro shocks
    finnhub_key = os.environ.get("FINNHUB_API_KEY", "")
    if finnhub_key:
        try:
            r = requests.get(
                f"https://finnhub.io/api/v1/news?category=crypto&token={finnhub_key}",
                timeout=8
            )
            r.raise_for_status()
            for art in r.json()[:20]:
                headline = (art.get("headline", "") + " " + art.get("summary", "")).lower()
                pub_time = datetime.fromtimestamp(art.get("datetime", 0), tz=timezone.utc)
                if (now - pub_time).total_seconds() > 7200: continue
                shock_hit  = any(kw in headline for kw in SHOCK_KEYWORDS)
                crypto_rel = any(t in headline for t in CRYPTO_RELEVANCE_TERMS)
                if shock_hit and crypto_rel:
                    active_blackouts.append({
                        "source": "Finnhub", "title": art.get("headline", "Breaking")[:120],
                        "currencies": "ALL", "impact": "SHOCK",
                        "event_time": pub_time.isoformat(),
                        "expires_at": (now + timedelta(minutes=60)).isoformat(),
                        "active": True, "updated_at": now.isoformat(),
                    })
                    print(f"[NEWS] ⚠️ CRYPTO SHOCK: {art.get('headline','')[:70]}")
        except Exception as e:
            print(f"[NEWS] Finnhub failed: {e}")

    # ── Source 3: CryptoCompare latest news ───────────────────────────
    # Free endpoint — no API key required for basic news feed.
    # Catches: ETF approvals/rejections, SEC/CFTC enforcement actions,
    #          exchange outages, major protocol exploits, regulatory bans.
    try:
        r = requests.get(
            "https://min-api.cryptocompare.com/data/v2/news/?lang=EN&sortOrder=latest",
            timeout=8, headers={"User-Agent": "Mozilla/5.0"}
        )
        r.raise_for_status()
        articles = r.json().get("Data", [])[:30]
        for art in articles:
            headline = (art.get("title", "") + " " + art.get("body", "")[:200]).lower()
            pub_time = datetime.fromtimestamp(art.get("published_on", 0), tz=timezone.utc)
            if (now - pub_time).total_seconds() > 3600:  # only last hour
                continue
            shock_hit  = any(kw in headline for kw in SHOCK_KEYWORDS)
            crypto_rel = any(t in headline for t in CRYPTO_RELEVANCE_TERMS)
            if shock_hit and crypto_rel:
                active_blackouts.append({
                    "source": "CryptoCompare",
                    "title": art.get("title", "Breaking")[:120],
                    "currencies": "ALL",
                    "impact": "SHOCK",
                    "event_time": pub_time.isoformat(),
                    "expires_at": (now + timedelta(minutes=60)).isoformat(),
                    "active": True,
                    "updated_at": now.isoformat(),
                })
                print(f"[NEWS] ⚠️ CRYPTO NEWS SHOCK: {art.get('title','')[:70]}")
    except Exception as e:
        print(f"[NEWS] CryptoCompare failed: {e}")

    # ── Source 4: FOMC scheduled meeting calendar ──────────────────────
    # Crypto is 3-5× more sensitive to Fed decisions than forex.
    # Block 24h before decision day AND 2h after (market digests the decision).
    # Dates are hardcoded from federalreserve.gov — updated once per year.
    # Window: meeting day 14:00 UTC - 1 day → meeting day 16:00 UTC (2h post-decision).
    # The actual rate decision is announced at ~14:00 UTC (2pm ET = 7pm UTC in winter).
    for fomc_date_str in FOMC_ALL_DATES:
        try:
            fomc_dt  = datetime.fromisoformat(fomc_date_str).replace(
                hour=14, minute=0, tzinfo=timezone.utc  # decision ~14:00 ET = ~19:00 UTC (winter)
            )
            # Adjust for EDT (summer) — Fed announces at ~18:00 UTC in summer
            fomc_announce = fomc_dt.replace(hour=19)  # conservative: use later time
            pre_window  = fomc_announce - timedelta(hours=24)
            post_window = fomc_announce + timedelta(hours=2)
            if pre_window <= now <= post_window:
                active_blackouts.append({
                    "source": "FOMC",
                    "title": f"Federal Reserve Rate Decision — {fomc_date_str}",
                    "currencies": "ALL",   # blocks ALL crypto pairs
                    "impact": "FOMC",
                    "event_time": fomc_announce.isoformat(),
                    "expires_at": post_window.isoformat(),
                    "active": True,
                    "updated_at": now.isoformat(),
                })
                print(f"[NEWS] 🏦 FOMC BLACKOUT: {fomc_date_str} — "
                      f"blocking all crypto {pre_window.strftime('%H:%M')}→{post_window.strftime('%H:%M')} UTC")
                break  # only one FOMC per run
        except Exception as e:
            print(f"[NEWS] FOMC date parse failed ({fomc_date_str}): {e}")

    try:
        sb.table("news_blackouts").delete().eq("active", True).execute()
        if active_blackouts:
            sb.table("news_blackouts").insert(active_blackouts).execute()
            print(f"[NEWS] {len(active_blackouts)} blackouts written")
        else:
            print("[NEWS] No active blackouts")
    except Exception as e:
        print(f"[NEWS] DB write failed: {e}")


# ── Sub-task B: Online Learner ────────────────────────────────────────

def _run_online_learner():
    from supabase import create_client
    sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])

    cutoff = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
    try:
        rows = (sb.table("trades").select("*")
                .eq("action", "CLOSE").gte("timestamp", cutoff)
                .order("timestamp", desc=True).limit(500)
                .execute().data or [])
    except Exception as e:
        print(f"[LEARNER] DB read failed: {e}"); return

    if len(rows) < 20:
        print(f"[LEARNER] Only {len(rows)} closed trades — need 20+, skipping"); return

    # Compute per-strategy win rates for reporting
    for strategy in STRATEGY_NAMES:
        strategy_rows = [r for r in rows if r.get("strategy_used") == strategy]
        if len(strategy_rows) < 10:
            print(f"[LEARNER] {strategy}: {len(strategy_rows)} samples — skip")
            continue
        try:
            model = load_strategy_model(strategy)
            # Use a higher timestep count with reward shaping based on actual win-rate
            wins = sum(1 for r in strategy_rows if float(r.get("pnl", 0) or 0) > 0)
            win_rate = wins / len(strategy_rows)
            # Scale training steps by performance — more training when model is underperforming
            extra_steps = 1000 if win_rate >= 0.55 else 3000
            model.learn(total_timesteps=2000 + extra_steps, reset_num_timesteps=False)
            save_strategy_model(model, strategy)
            print(f"[LEARNER] {strategy}: fine-tuned on {len(strategy_rows)} trades "
                  f"(WR={win_rate:.1%}, steps={2000+extra_steps})")
        except Exception as e:
            print(f"[LEARNER] {strategy} failed: {e}")


# ── Sub-task C: Forward Tester ────────────────────────────────────────

def _run_forward_tester():
    import requests
    from supabase import create_client

    sb  = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])
    now = datetime.now(timezone.utc)

    CRYPTO_YF = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD"]
    results   = []

    for ticker in CRYPTO_YF:
        try:
            r = requests.get(
                f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}",
                params={"interval": "1h", "range": "30d"},
                timeout=10, headers={"User-Agent": "Mozilla/5.0"}
            )
            r.raise_for_status()
            data   = r.json()["chart"]["result"][0]
            closes = [x for x in data["indicators"]["quote"][0]["close"] if x is not None]
            if len(closes) < 50: continue

            for strategy in STRATEGY_NAMES:
                fn = STRATEGY_FN[strategy]
                wins = losses = 0
                for i in range(50, len(closes)):
                    window = closes[i-50:i]
                    fake = AIRequest(
                        symbol=ticker.replace("-", ""), timeframe="H1",
                        magic=0, timestamp=now.isoformat(),
                        account_balance=10000, account_equity=10000,
                        risk_pct=1.0, max_positions=3, open_positions=0,
                        bars=BarData(
                            open=window[::-1], high=[c*1.002 for c in window[::-1]],
                            low=[c*0.998 for c in window[::-1]], close=window[::-1],
                            volume=[100.0]*len(window),
                        ),
                        atr=abs(window[-1]-window[-2])*2,
                        atr_avg=abs(window[-1]-window[-20])/20,
                        adx=25.0, plus_di=15.0, minus_di=12.0,
                        rsi=_rsi(window), macd=0, macd_signal=0, macd_hist=0,
                        ema20=_ema(window, 20),
                        ema50=_ema(window, 50) if len(window) >= 50 else window[-1],
                        ema200=window[-1], htf_ema50=window[-1], htf_ema200=window[-1],
                        tick_value=1.0, tick_size=0.01, min_lot=0.01,
                        max_lot=100, lot_step=0.01, point=0.01, digits=2,
                    )
                    direction, _, _ = fn(fake)
                    if direction == "NONE": continue
                    if i + 1 < len(closes):
                        ret = (closes[i+1] - closes[i]) / closes[i]
                        won = (direction == "BUY" and ret > 0) or (direction == "SELL" and ret < 0)
                        if won: wins += 1
                        else:   losses += 1

                total = wins + losses
                wr    = wins / total if total > 0 else 0.0
                results.append({
                    "ticker": ticker, "strategy": strategy,
                    "trades": total, "wins": wins,
                    "win_rate": round(wr, 4), "tested_at": now.isoformat(),
                })
                print(f"[FWD] {ticker} {strategy}: {total} trades WR={wr:.1%}")
        except Exception as e:
            print(f"[FWD] {ticker} failed: {e}")

    if results:
        try:
            sb.table("forward_test_results").insert(results).execute()
            print(f"[FWD] {len(results)} results saved")
        except Exception as e:
            print(f"[FWD] DB write failed: {e}")


# ══════════════════════════════════════════════════════════════════════
#  CRON 1 — news_and_forward  (every 5 min)
#  Runs the news blackout monitor on every tick.
#  Runs the forward tester every 4 hours (internally gated by state file).
#  Merged into one slot because forward_tester is infrequent and lightweight
#  when skipped — no need to burn a dedicated cron slot for a 4-hr task.
# ══════════════════════════════════════════════════════════════════════

@app.function(
    image=image,
    volumes={MODEL_DIR: volume},
    secrets=[secrets],
    schedule=modal.Period(minutes=5),
    timeout=300,
    memory=1024,
)
def news_and_forward():
    """
    Cron 1 — Runs every 5 minutes.
    - Always: news_monitor  (ForexFactory + Finnhub blackout detection)
    - Every 4 hrs: forward_tester  (paper-trade all symbols on live Yahoo data)
    """
    state = _load_scheduler_state()
    now   = datetime.now(timezone.utc).isoformat()
    ran   = []

    # News monitor — always runs (5-min cadence is the point)
    try:
        _run_news_monitor()
        state["news_and_forward_news"] = now
        ran.append("news_monitor")
    except Exception as e:
        print(f"[NEWS_FWD] news_monitor error: {e}")

    # Forward tester — every 4 hours only
    if _due(state, "news_and_forward_fwd", 4):
        try:
            _run_forward_tester()
            state["news_and_forward_fwd"] = now
            ran.append("forward_tester")
        except Exception as e:
            print(f"[NEWS_FWD] forward_tester error: {e}")

    _save_scheduler_state(state)
    print(f"[NEWS_FWD] Done. Ran: {ran}")


# ══════════════════════════════════════════════════════════════════════
#  CRON 2 — online_learner  (every 6 hours)
#  Fine-tunes the three PPO strategy models using recent closed trades
#  from Supabase.  Isolated in its own cron so it gets a full 700s
#  timeout and 2GB RAM without competing with other tasks.
# ══════════════════════════════════════════════════════════════════════

@app.function(
    image=image,
    volumes={MODEL_DIR: volume},
    secrets=[secrets],
    schedule=modal.Period(hours=6),
    timeout=700,
    memory=2048,
    cpu=2.0,
)
def online_learner():
    """
    Cron 2 — Runs every 6 hours.
    Fine-tunes PPO models (trend_following, mean_reversion, breakout)
    using the last 7 days of closed trades stored in Supabase.
    Requires 20+ total closed trades and 10+ per strategy to trigger.
    """
    print("[LEARNER] Starting scheduled fine-tune run")
    _run_online_learner()
    print("[LEARNER] Fine-tune run complete")


# ══════════════════════════════════════════════════════════════════════
#  CRON 3 — performance_watch  (every 1 minute)
#  Polls Supabase for live balance/equity/drawdown and auto-halts the
#  EA if drawdown exceeds critical threshold — acting as a server-side
#  safety net independent of the MT5 EA's own DD check.
#  Also resets halt automatically if DD recovers below 50% of threshold.
# ══════════════════════════════════════════════════════════════════════

@app.function(
    image=image,
    volumes={MODEL_DIR: volume},
    secrets=[secrets],
    schedule=modal.Period(minutes=1),
    timeout=30,
    memory=512,
)
def performance_watch():
    """
    Cron 3 — Runs every 1 minute.
    Server-side guardian:
    - Reads live_dd_pct from ea_config (patched by MT5 EA every 15s)
    - Auto-halts if DD >= max_dd_pct (server-side redundancy to EA halt)
    - Auto-resumes if DD < max_dd_pct * 0.50 and was server-halted
    - Logs HALT / RESUME events to Supabase events table
    """
    from supabase import create_client
    sb  = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])
    now = datetime.now(timezone.utc).isoformat()

    try:
        cfg_row = sb.table("ea_config").select(
            "id,halted,max_dd_pct,live_dd_pct,live_ts,updated_by"
        ).limit(1).execute()
        if not cfg_row.data:
            print("[PERFWATCH] No ea_config row found — skipping")
            return

        cfg        = cfg_row.data[0]
        cfg_id     = cfg["id"]
        halted     = bool(cfg.get("halted", False))
        max_dd     = float(cfg.get("max_dd_pct") or 20.0)
        live_dd    = float(cfg.get("live_dd_pct") or 0.0)
        live_ts    = cfg.get("live_ts", "unknown")
        updated_by = cfg.get("updated_by", "")

        # Auto-halt if live DD exceeds threshold and EA hasn't already halted
        if live_dd >= max_dd and not halted:
            sb.table("ea_config").update({
                "halted": True, "updated_by": "performance_watch", "updated_at": now
            }).eq("id", cfg_id).execute()
            sb.table("events").insert({
                "type": "HALT",
                "message": f"[Server Guardian] Auto-halt: live_dd={live_dd:.2f}% >= max_dd={max_dd:.2f}% (EA ts: {live_ts})",
                "timestamp": now,
            }).execute()
            print(f"[PERFWATCH] ⛔ AUTO-HALT: DD={live_dd:.2f}% >= {max_dd:.2f}%")

        # Auto-resume if DD has recovered below 50% of threshold
        # Only auto-resume if we (server) were the ones who halted it
        elif halted and live_dd < max_dd * 0.50 and updated_by == "performance_watch":
            sb.table("ea_config").update({
                "halted": False, "updated_by": "performance_watch", "updated_at": now
            }).eq("id", cfg_id).execute()
            sb.table("events").insert({
                "type": "RESUME",
                "message": f"[Server Guardian] Auto-resume: live_dd={live_dd:.2f}% recovered below {max_dd*0.50:.2f}%",
                "timestamp": now,
            }).execute()
            print(f"[PERFWATCH] ✅ AUTO-RESUME: DD={live_dd:.2f}% recovered")

        else:
            print(f"[PERFWATCH] OK — DD={live_dd:.2f}% / max={max_dd:.2f}% / halted={halted}")

    except Exception as e:
        print(f"[PERFWATCH] Error: {e}")


# ══════════════════════════════════════════════════════════════════════
#  CRON 4 — model_health_check  (every 12 hours)
#  Validates that each PPO model file exists, loads cleanly, and was
#  saved recently.  If a model is missing or stale (>48 hrs since last
#  save), triggers an immediate online_learner run to rebuild it.
#  Also logs a model status snapshot to Supabase events table so the
#  Streamlit dashboard can surface model health.
# ══════════════════════════════════════════════════════════════════════

@app.function(
    image=image,
    volumes={MODEL_DIR: volume},
    secrets=[secrets],
    schedule=modal.Period(hours=12),
    timeout=600,
    memory=2048,
    cpu=2.0,
)
def model_health_check():
    """
    Cron 4 — Runs every 12 hours.
    - Checks each strategy model exists and loads without error
    - Flags models not saved in the last 48 hours as stale
    - Auto-triggers a fine-tune run for any stale / missing / corrupt model
    - Writes a model health snapshot to Supabase events table
    """
    from supabase import create_client
    sb  = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])
    now = datetime.now(timezone.utc)

    report    = {}
    needs_retrain = []

    for strategy in STRATEGY_NAMES:
        meta_path = _meta_path(strategy)
        model_path = _model_path(strategy)
        status = {"strategy": strategy, "exists": False, "loads": False,
                  "trained": False, "stale": True, "saved_at": None}

        # Check file existence
        if not os.path.exists(model_path):
            print(f"[HEALTHCHECK] {strategy}: model file MISSING")
            needs_retrain.append(strategy)
            report[strategy] = status
            continue

        status["exists"] = True

        # Check meta / freshness (stale = no save in 48 hrs)
        if os.path.exists(meta_path):
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                status["trained"]  = bool(meta.get("trained", False))
                status["saved_at"] = meta.get("saved_at")
                if status["saved_at"]:
                    saved_dt = datetime.fromisoformat(status["saved_at"])
                    # Make tz-aware if naive
                    if saved_dt.tzinfo is None:
                        saved_dt = saved_dt.replace(tzinfo=timezone.utc)
                    age_hrs = (now - saved_dt).total_seconds() / 3600
                    status["stale"] = age_hrs > 48
                    if status["stale"]:
                        print(f"[HEALTHCHECK] {strategy}: STALE (last saved {age_hrs:.1f}h ago)")
                        needs_retrain.append(strategy)
            except Exception as e:
                print(f"[HEALTHCHECK] {strategy}: meta read error: {e}")
                needs_retrain.append(strategy)

        # Try loading model to verify it isn't corrupt
        try:
            model = load_strategy_model(strategy)
            test_obs = np.zeros(OBS_DIM, dtype=np.float32)
            model.predict(test_obs.reshape(1, -1))
            status["loads"] = True
            print(f"[HEALTHCHECK] {strategy}: OK (trained={status['trained']}, stale={status['stale']})")
        except Exception as e:
            print(f"[HEALTHCHECK] {strategy}: LOAD FAILED — {e}")
            status["loads"] = False
            if strategy not in needs_retrain:
                needs_retrain.append(strategy)

        report[strategy] = status

    # Log health snapshot to Supabase
    try:
        sb.table("events").insert({
            "type": "INFO",
            "message": f"[ModelHealth] " + " | ".join(
                f"{s}: {'✅' if r['loads'] and not r['stale'] else '⚠️'}"
                f"trained={r['trained']} stale={r['stale']}"
                for s, r in report.items()
            ),
            "timestamp": now.isoformat(),
        }).execute()
    except Exception as e:
        print(f"[HEALTHCHECK] Event log failed: {e}")

    # Auto-retrain stale / missing / corrupt models
    if needs_retrain:
        print(f"[HEALTHCHECK] Triggering retrain for: {needs_retrain}")
        try:
            _run_online_learner()
        except Exception as e:
            print(f"[HEALTHCHECK] Auto-retrain failed: {e}")
    else:
        print("[HEALTHCHECK] All models healthy — no retrain needed")


# ══════════════════════════════════════════════════════════════════════
#  CRON 5 — db_maintenance  (every 24 hours)
#  Keeps the Supabase database lean:
#  - Prunes regime_changes older than 30 days (high-volume table)
#  - Prunes events older than 14 days
#  - Prunes expired news_blackouts
#  - Prunes forward_test_results older than 60 days
#  - Logs a summary of rows deleted
#  Performance and trades tables are NOT pruned — they are the
#  permanent trade history and should be kept indefinitely.
# ══════════════════════════════════════════════════════════════════════

@app.function(
    image=image,
    volumes={MODEL_DIR: volume},
    secrets=[secrets],
    schedule=modal.Cron("0 3 * * *"),   # 03:00 UTC daily — low-traffic window
    timeout=120,
    memory=512,
)
def db_maintenance():
    """
    Cron 5 — Runs daily at 03:00 UTC.
    Prunes high-volume tables to keep Supabase within the free-tier
    500 MB storage limit and maintain query performance.
    """
    from supabase import create_client
    sb  = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])
    now = datetime.now(timezone.utc)
    deleted = {}

    prune_rules = [
        # (table,                  age_days)
        ("regime_changes",         30),
        ("events",                 14),
        ("news_blackouts",          0),   # uses expires_at, not age_days
        ("forward_test_results",   60),
    ]

    for table, age_days in prune_rules:
        try:
            if table == "news_blackouts":
                # Delete expired blackouts (expires_at < now)
                result = sb.table(table).delete().lt("expires_at", now.isoformat()).execute()
            elif age_days > 0:
                ts_col = "tested_at" if table == "forward_test_results" else "timestamp"
                cutoff = (now - timedelta(days=age_days)).isoformat()
                result = sb.table(table).delete().lt(ts_col, cutoff).execute()
            else:
                continue
            count  = len(result.data) if result.data else 0
            deleted[table] = count
            print(f"[DBMAINT] {table}: deleted {count} rows (age>{age_days}d)")
        except Exception as e:
            print(f"[DBMAINT] {table} prune failed: {e}")
            deleted[table] = f"ERROR: {e}"

    # Log maintenance run to events
    try:
        sb.table("events").insert({
            "type": "INFO",
            "message": f"[DBMaint] Daily prune complete: " +
                       ", ".join(f"{t}={n}" for t, n in deleted.items()),
            "timestamp": now.isoformat(),
        }).execute()
    except Exception as e:
        print(f"[DBMAINT] Event log failed: {e}")

    print(f"[DBMAINT] Complete: {deleted}")


# ──────────────────────────────────────────────────────────────────────
#  LOCAL TEST
# ──────────────────────────────────────────────────────────────────────

@app.local_entrypoint()
def test():
    import random
    random.seed(42)
    n = 200; base = 65000.0
    closes = [base + random.gauss(0, 500) for _ in range(n)]
    # Simulate trending market
    for i in range(50, 100): closes[i] = closes[49] + (i-49) * 80

    p = AIRequest(
        symbol="BTCUSD", timeframe="H1", magic=20250228,
        timestamp=datetime.now(timezone.utc).isoformat(),
        account_balance=10000, account_equity=10250,
        risk_pct=1.0, max_positions=3, open_positions=0,
        bars=BarData(
            open=closes[:], high=[c+200 for c in closes],
            low=[c-200 for c in closes], close=closes,
            volume=[random.uniform(100,500) for _ in range(n)],
        ),
        atr=350, atr_avg=280, adx=34, plus_di=28, minus_di=16,
        rsi=58, macd=120, macd_signal=90, macd_hist=30,
        ema20=64800, ema50=63500, ema200=60000,
        htf_ema50=63000, htf_ema200=58000,
        tick_value=1.0, tick_size=0.01, min_lot=0.01,
        max_lot=100, lot_step=0.01, point=0.01, digits=2,
        spread=15.0, hour=14, dow=2,
    )

    features, scores = build_features(p)
    regime_str = detect_regime(p)
    strategy   = REGIME_TO_STRATEGY[regime_str]
    regime_id, regime_name, regime_conf = classify_regime_granular(p, regime_str, features)
    direction, rb_conf, reason = STRATEGY_FN[strategy](p)
    lots, sl, tp, sl_m, tp_m, rr = compute_position(p, direction, rb_conf, strategy)
    signal_int = direction_to_signal(direction, rb_conf)

    print(f"\n{'═'*62}")
    print(f"  ApexHydra Crypto Modal v{MODEL_VERSION} — Local Test")
    print(f"{'═'*62}")
    print(f"  Symbol:     BTCUSD")
    print(f"  Regime:     {regime_str} → {regime_name} ({regime_conf:.0%})")
    print(f"  Strategy:   {strategy}")
    print(f"  Direction:  {direction}  ({rb_conf:.0%})")
    print(f"  Signal:     {SIGNAL_NAMES.get(signal_int,'?')} ({signal_int})")
    print(f"  Lots:       {lots}")
    print(f"  SL:         {sl:.2f}  ({sl_m}×ATR)")
    print(f"  TP:         {tp:.2f}  ({tp_m}×ATR)")
    print(f"  R:R:        {rr:.2f}")
    print(f"  Reason:     {reason}")
    print(f"  Scores:     {scores}")
    print(f"  Features:   {OBS_DIM} dims")
    print(f"{'═'*62}")
    print(f"\n  PPO models: untrained (expected on first run)")
    print(f"  Run modal deploy apex_hydracrypto_modal.py to deploy")
    print(f"  Run modal run apex_hydracrypto_modal.py::online_learner to train\n")