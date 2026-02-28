"""
╔══════════════════════════════════════════════════════════════════════╗
║            ApexHydra Crypto — Modal AI Engine Server                ║
║  v4.1 — Added: ML Training | Backtesting | News Filter              ║
║                                                                      ║
║  Deploy:  modal deploy apex_hydracrypto_modal.py                    ║
║  Docs:    https://modal.com/docs                                     ║
╚══════════════════════════════════════════════════════════════════════╝

NEW ENDPOINTS (v4.1):
  POST /apex-hydracrypto-train    — Train / update XGBoost model from trade history
  POST /apex-hydracrypto-backtest — Run strategy backtest on historical OHLCV data
  GET  /apex-hydracrypto-health   — Health + model stats
  POST /apex-hydracrypto-predict  — Main prediction (unchanged interface + news filter)
"""

import modal
import numpy as np
import json
import math
import os
import pickle
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
        "scikit-learn>=1.4",
        "xgboost>=2.0",
        "scipy>=1.12",
        "fastapi[standard]>=0.110",
        "pydantic>=2.6",
        "supabase>=2.4",
    )
)

# Persistent volume — stores model weights & training data between calls
volume = modal.Volume.from_name("apex-hydracrypto-models", create_if_missing=True)
MODEL_DIR = "/models"

# Paths inside the volume
MODEL_PATH    = f"{MODEL_DIR}/xgb_signal.pkl"       # Trained XGBoost classifier
SCALER_PATH   = f"{MODEL_DIR}/feature_scaler.pkl"   # StandardScaler
DATASET_PATH  = f"{MODEL_DIR}/training_data.npz"    # Accumulated feature+label dataset
MODEL_META    = f"{MODEL_DIR}/model_meta.json"       # Training metadata

secrets = modal.Secret.from_name("apex-hydracrypto-secrets")

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
    # Identity
    symbol:           str
    timeframe:        str
    magic:            int
    timestamp:        str

    # Account context
    account_balance:  float
    account_equity:   float
    allocated_capital: float = 0.0
    risk_pct:         float = 1.0
    max_positions:    int   = 3
    open_positions:   int   = 0

    # OHLCV bars — index 0 = most recent closed bar
    bars:             BarData

    # Pre-computed indicators from MT5
    atr:              float
    atr_avg:          float
    adx:              float
    plus_di:          float
    minus_di:         float
    rsi:              float
    macd:             float
    macd_signal:      float
    macd_hist:        float
    ema20:            float
    ema50:            float
    ema200:           float
    htf_ema50:        float
    htf_ema200:       float

    # Symbol contract specs
    tick_value:       float
    tick_size:        float
    min_lot:          float
    max_lot:          float
    lot_step:         float
    point:            float
    digits:           int   = 5

    # Online learning history
    recent_signals:   Optional[list[int]]   = None
    recent_outcomes:  Optional[list[float]] = None
    recent_regimes:   Optional[list[int]]   = None

    # ── NEWS FILTER (NEW in v4.1) ─────────────────────────────────────
    # MT5 EA sets these based on MT5's economic calendar or your own logic.
    # news_blackout=True → block ALL trading for this tick
    # news_minutes_away  → minutes to next high-impact event (0 = happening now)
    # Set Inp_News_Filter=true in EA and it will populate these fields.
    news_blackout:        bool  = False   # Hard block from EA
    news_minutes_away:    int   = 999     # 999 = no news detected
    news_buffer_minutes:  int   = 15      # Block within N min of event (EA configurable)


class AIResponse(BaseModel):
    symbol:        str
    regime_id:     int
    regime_name:   str
    regime_conf:   float
    signal:        int
    signal_name:   str
    confidence:    float
    lots:          float
    sl_price:      float
    tp_price:      float
    sl_atr_mult:   float
    tp_atr_mult:   float
    rr_ratio:      float
    feature_scores: dict
    reasoning:     str
    model_version: str
    server_ts:     str
    # New fields
    news_blocked:  bool   = False   # True if trade was blocked by news filter
    ml_signal:     int    = 0       # XGBoost model signal (0 if no model loaded)
    ml_confidence: float  = 0.0     # XGBoost confidence


# ── Training schemas ──────────────────────────────────────────────────

class TrainSample(BaseModel):
    """A single labeled trade sample for training."""
    symbol:      str
    features:    list[float]       # 26-dim vector from build_features()
    regime_id:   int
    signal_given: int              # Signal that was executed (-2 to +2)
    outcome:     float             # PnL of the trade (positive = win)
    won:         bool
    timestamp:   str


class TrainRequest(BaseModel):
    samples:     list[TrainSample]
    force_retrain: bool = False    # Force full retrain even with few new samples


class TrainResponse(BaseModel):
    status:      str
    samples_total: int
    accuracy:    float
    f1_score:    float
    model_version: str
    trained_at:  str
    message:     str


# ── Backtest schemas ──────────────────────────────────────────────────

class BacktestBar(BaseModel):
    """A single bar with pre-computed indicators — sent from MT5 history."""
    timestamp:   str
    open:        float
    high:        float
    low:         float
    close:       float
    volume:      float
    atr:         float
    atr_avg:     float
    adx:         float
    plus_di:     float
    minus_di:    float
    rsi:         float
    macd:        float
    macd_signal: float
    macd_hist:   float
    ema20:       float
    ema50:       float
    ema200:      float
    htf_ema50:   float
    htf_ema200:  float


class BacktestRequest(BaseModel):
    symbol:          str
    timeframe:       str
    bars:            list[BacktestBar]   # Minimum 200 bars recommended
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
    # Spread in points (applied on entry)
    spread_points:   float = 20.0


class BacktestTrade(BaseModel):
    entry_time:  str
    exit_time:   str
    signal:      int
    regime:      str
    confidence:  float
    lots:        float
    entry_price: float
    exit_price:  float
    sl:          float
    tp:          float
    pnl:         float
    won:         bool
    bars_held:   int


class BacktestResponse(BaseModel):
    symbol:          str
    timeframe:       str
    bars_tested:     int
    total_trades:    int
    wins:            int
    losses:          int
    win_rate:        float
    total_pnl:       float
    final_balance:   float
    max_drawdown_pct: float
    sharpe_ratio:    float
    profit_factor:   float
    avg_rr:          float
    trades:          list[BacktestTrade]
    equity_curve:    list[float]          # Balance at each closed trade
    regime_breakdown: dict


# ──────────────────────────────────────────────────────────────────────
#  CONSTANTS
# ──────────────────────────────────────────────────────────────────────

REGIME_NAMES = {
    0: "Trend Bull",
    1: "Trend Bear",
    2: "Ranging",
    3: "High Volatility",
    4: "Breakout",
    5: "Undefined",
}

SIGNAL_NAMES = {
    -2: "Strong Sell",
    -1: "Sell",
     0: "Hold",
     1: "Buy",
     2: "Strong Buy",
}

MODEL_VERSION = "4.1.0"

# ──────────────────────────────────────────────────────────────────────
#  MODEL LOADER  (cached per container, reloaded when version changes)
# ──────────────────────────────────────────────────────────────────────

_MODEL_CACHE: dict = {"model": None, "scaler": None, "version": None}


def load_ml_model():
    """Load XGBoost model from volume into memory cache."""
    global _MODEL_CACHE
    try:
        if os.path.exists(MODEL_META):
            with open(MODEL_META) as f:
                meta = json.load(f)
            cached_v = _MODEL_CACHE.get("version")
            if cached_v == meta.get("version") and _MODEL_CACHE["model"] is not None:
                return _MODEL_CACHE["model"], _MODEL_CACHE["scaler"]  # Already current

        if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
            with open(MODEL_PATH, "rb") as f:
                model = pickle.load(f)
            with open(SCALER_PATH, "rb") as f:
                scaler = pickle.load(f)
            version = meta.get("version", "unknown") if os.path.exists(MODEL_META) else "unknown"
            _MODEL_CACHE = {"model": model, "scaler": scaler, "version": version}
            return model, scaler
    except Exception as e:
        print(f"[ML] Model load failed: {e}")
    return None, None


def ml_predict(features: np.ndarray) -> tuple[int, float]:
    """
    Run the trained XGBoost model.
    Returns (signal, confidence) or (0, 0.0) if no model.
    """
    model, scaler = load_ml_model()
    if model is None or scaler is None:
        return 0, 0.0
    try:
        x = scaler.transform(features.reshape(1, -1))
        proba = model.predict_proba(x)[0]
        class_idx = int(np.argmax(proba))
        # Classes: 0=Sell, 1=Hold, 2=Buy
        signal_map = {0: -1, 1: 0, 2: 1}
        conf = float(proba[class_idx])
        return signal_map.get(class_idx, 0), conf
    except Exception as e:
        print(f"[ML] Predict error: {e}")
        return 0, 0.0


# ──────────────────────────────────────────────────────────────────────
#  FEATURE ENGINEERING  (unchanged from v4.0)
# ──────────────────────────────────────────────────────────────────────

def build_features(p: AIRequest) -> tuple[np.ndarray, dict]:
    c  = p.bars.close
    h  = p.bars.high
    l  = p.bars.low
    o  = p.bars.open
    v  = p.bars.volume
    n  = min(len(c), 100)
    atr = p.atr if p.atr > 0 else 1e-9

    ema_full_bull = (c[0] > p.ema20 > p.ema50 > p.ema200)
    ema_full_bear = (c[0] < p.ema20 < p.ema50 < p.ema200)
    if ema_full_bull:    ema_align = +1.0
    elif ema_full_bear:  ema_align = -1.0
    elif c[0] > p.ema50 and p.ema50 > p.ema200: ema_align = +0.5
    elif c[0] < p.ema50 and p.ema50 < p.ema200: ema_align = -0.5
    elif c[0] > p.ema200: ema_align = +0.25
    else:                ema_align = -0.25

    ema_sep    = np.clip((p.ema20 - p.ema50) / atr, -3, 3)
    ema50_dist = np.clip((c[0] - p.ema50) / atr, -3, 3)

    if c[0] > p.htf_ema50 and p.htf_ema50 > p.htf_ema200:   htf_bias = +1.0
    elif c[0] < p.htf_ema50 and p.htf_ema50 < p.htf_ema200: htf_bias = -1.0
    elif c[0] > p.htf_ema50: htf_bias = +0.4
    else:                    htf_bias = -0.4

    adx_norm  = np.clip((p.adx - 25.0) / 40.0, -1, 1)
    di_diff   = np.clip((p.plus_di - p.minus_di) / 30.0, -1, 1)
    rsi_norm  = (p.rsi - 50.0) / 50.0
    rsi_ob    =  max(0, (p.rsi - 70)) / 30.0
    rsi_os    = -max(0, (30 - p.rsi)) / 30.0
    rsi_ext   = rsi_ob + rsi_os
    macd_hist_norm = np.clip(p.macd_hist / (atr * 0.05 + 1e-9), -3, 3)
    macd_cross     = 1.0 if p.macd > p.macd_signal else -1.0
    vol_ratio = p.atr / (p.atr_avg if p.atr_avg > 0 else p.atr)
    vol_norm  = np.clip((vol_ratio - 1.0) / 1.5, -1, 1)
    vol_flag  = 1.0 if vol_ratio > 1.6 else (-0.5 if vol_ratio < 0.6 else 0.0)

    bar_range = h[0] - l[0]
    if bar_range > 0:
        body     = abs(c[0] - o[0]) / bar_range
        candle   = body if c[0] > o[0] else -body
        upper_wick = (h[0] - max(c[0], o[0])) / bar_range
        lower_wick = (min(c[0], o[0]) - l[0]) / bar_range
    else:
        candle = upper_wick = lower_wick = 0.0

    look = min(20, n)
    hh20 = max(h[:look]) if look else c[0]
    ll20 = min(l[:look]) if look else c[0]
    rng20 = hh20 - ll20
    pos20 = ((c[0] - ll20) / rng20) * 2 - 1 if rng20 > 0 else 0.0

    def roc(period):
        if len(c) > period and c[period] > 0:
            return np.clip((c[0] - c[period]) / c[period] * 100 / 10, -2, 2)
        return 0.0
    roc5, roc10, roc20 = roc(5), roc(10), roc(20)

    hh50 = max(h[1:min(51, len(h))]) if len(h) > 1 else c[0]
    ll50 = min(l[1:min(51, len(l))]) if len(l) > 1 else c[0]
    spread50 = hh50 - ll50
    if c[0] > hh50 and spread50 > 0:
        breakout = min(1.0,  (c[0] - hh50) / spread50)
    elif c[0] < ll50 and spread50 > 0:
        breakout = max(-1.0, (c[0] - ll50) / spread50)
    else:
        breakout = 0.0

    look_z = min(20, n)
    arr_z  = np.array(c[:look_z][::-1])
    z_mean, z_std = arr_z.mean(), arr_z.std()
    zscore = np.clip((c[0] - z_mean) / (z_std + 1e-9), -3, 3)

    vol_bars = min(10, len(v))
    if vol_bars > 1 and v[1] > 0:
        vol_mom = np.clip((v[0] - np.mean(v[1:vol_bars])) / (np.mean(v[1:vol_bars]) + 1e-9), -2, 2)
    else:
        vol_mom = 0.0

    hist_bias = _compute_history_bias(p.recent_signals, p.recent_outcomes, p.recent_regimes)

    features = np.array([
        ema_align, ema_sep, ema50_dist, htf_bias,
        adx_norm, di_diff, rsi_norm, rsi_ext,
        macd_hist_norm, macd_cross, vol_norm, vol_flag,
        candle, upper_wick, lower_wick, pos20,
        roc5, roc10, roc20, breakout,
        zscore, vol_mom, hist_bias,
        adx_norm * di_diff,
        rsi_norm * macd_hist_norm,
        ema_align * htf_bias,
    ], dtype=np.float32)

    scores = {
        "ema_alignment":    round(float(ema_align), 3),
        "htf_bias":         round(float(htf_bias), 3),
        "trend_strength":   round(float(adx_norm), 3),
        "di_direction":     round(float(di_diff), 3),
        "rsi":              round(float(rsi_norm), 3),
        "macd_momentum":    round(float(macd_hist_norm), 3),
        "volatility_ratio": round(float(vol_ratio), 3),
        "breakout_score":   round(float(breakout), 3),
        "mean_reversion_z": round(float(zscore), 3),
        "history_bias":     round(float(hist_bias), 3),
        "volume_momentum":  round(float(vol_mom), 3),
    }
    return np.nan_to_num(features, 0), scores


def _compute_history_bias(signals, outcomes, regimes):
    if not signals or not outcomes:
        return 0.0
    n = min(len(signals), len(outcomes), 20)
    sig = np.array(signals[-n:], dtype=float)
    out = np.array(outcomes[-n:], dtype=float)
    w   = np.exp(np.linspace(-2, 0, n))
    pos = sig > 0
    neg = sig < 0
    pos_wr = np.average((out[pos] > 0).astype(float), weights=w[pos]) if pos.any() else 0.5
    neg_wr = np.average((out[neg] > 0).astype(float), weights=w[neg]) if neg.any() else 0.5
    return float(np.clip((pos_wr - neg_wr), -0.5, 0.5))


# ──────────────────────────────────────────────────────────────────────
#  NEWS FILTER  (NEW in v4.1)
# ──────────────────────────────────────────────────────────────────────

def check_news_filter(request: AIRequest) -> tuple[bool, str]:
    """
    Returns (blocked: bool, reason: str).

    Two-layer protection:
      Layer 1 — Hard block: EA sets news_blackout=True during known high-impact windows.
      Layer 2 — Soft block: Block if next high-impact event is within news_buffer_minutes.

    HOW TO USE IN MT5 EA:
    ─────────────────────
    Add to your EA inputs:
        input bool  Inp_News_Filter       = true;    // Enable news filter
        input int   Inp_News_Buffer_Min   = 15;       // Minutes to block around news

    In your AI payload builder, add:
        // Option A: Use EventCalendar() to detect events
        datetime now = TimeCurrent();
        MqlCalendarValue events[];
        int cnt = CalendarValueHistory(events, now, now + 3600); // next hour
        bool news_soon = false;
        int mins_away = 999;
        for(int i=0; i<cnt; i++) {
            if(events[i].impact_type == CALENDAR_IMPACT_HIGH) {
                int diff = (int)((events[i].time - now) / 60);
                if(abs(diff) <= Inp_News_Buffer_Min) { news_soon = true; mins_away = diff; }
            }
        }
        // Set in payload:
        // "news_blackout": Inp_News_Filter && news_soon,
        // "news_minutes_away": mins_away,
        // "news_buffer_minutes": Inp_News_Buffer_Min
    """
    # Layer 1 — EA-side hard block
    if request.news_blackout:
        return True, f"NewsFilter: EA hard block (news_blackout=true)"

    # Layer 2 — Server-side buffer check
    if request.news_minutes_away < request.news_buffer_minutes:
        return True, (
            f"NewsFilter: High-impact event in {request.news_minutes_away} min "
            f"(buffer={request.news_buffer_minutes} min)"
        )

    return False, ""


# ──────────────────────────────────────────────────────────────────────
#  REGIME CLASSIFIER
# ──────────────────────────────────────────────────────────────────────

def classify_regime(p: AIRequest, features: np.ndarray) -> tuple[int, str, float]:
    vol_ratio   = p.atr / (p.atr_avg if p.atr_avg > 0 else p.atr)
    is_trending = p.adx >= 25
    bull_trend  = is_trending and p.plus_di > p.minus_di
    bear_trend  = is_trending and p.minus_di > p.plus_di
    high_vol    = vol_ratio >= 1.6
    breakout_s  = abs(float(features[19]))

    scores = np.zeros(6)

    if breakout_s > 0.35 and high_vol:
        scores[4] = 0.50 + breakout_s * 0.40
        scores[3] = 0.20
    elif high_vol and not is_trending:
        scores[3] = 0.55 + min(0.35, (vol_ratio - 1.6) * 0.30)
        scores[2] = 0.20
    elif bull_trend:
        adx_boost = min(0.40, (p.adx - 25) / 40 * 0.40)
        htf_boost = 0.08 if float(features[3]) > 0 else 0.0
        scores[0] = 0.55 + adx_boost + htf_boost
        scores[2] = 0.20
    elif bear_trend:
        adx_boost = min(0.40, (p.adx - 25) / 40 * 0.40)
        htf_boost = 0.08 if float(features[3]) < 0 else 0.0
        scores[1] = 0.55 + adx_boost + htf_boost
        scores[2] = 0.20
    else:
        ranging_conf = min(0.88, 0.50 + (25 - p.adx) / 25 * 0.38)
        scores[2] = ranging_conf
        scores[5] = max(0, 1.0 - ranging_conf)

    total = scores.sum()
    if total > 0: scores /= total

    regime_id   = int(np.argmax(scores))
    regime_conf = float(scores[regime_id])
    return regime_id, REGIME_NAMES[regime_id], regime_conf


# ──────────────────────────────────────────────────────────────────────
#  SIGNAL ENGINE
# ──────────────────────────────────────────────────────────────────────

REGIME_WEIGHTS = {
    0: np.array([ 0.25, 0.20, 0.15, 0.15, 0.05, 0.10, 0.02, 0.00,  0.00, 0.05, 0.03]),
    1: np.array([ 0.25, 0.20, 0.15, 0.15, 0.05, 0.10, 0.02, 0.00,  0.00, 0.05, 0.03]),
    2: np.array([ 0.05, 0.10, 0.02, 0.05, 0.25, 0.20, 0.05, 0.00,  0.20, 0.05, 0.03]),
    3: np.array([ 0.10, 0.10, 0.08, 0.08, 0.10, 0.10, 0.18, 0.10,  0.08, 0.05, 0.03]),
    4: np.array([ 0.10, 0.12, 0.08, 0.10, 0.05, 0.10, 0.10, 0.30,  0.00, 0.03, 0.02]),
}

KEY_FEATURE_IDX = [0, 3, 4, 5, 6, 8, 10, 19, 20, 22, 21]


def compute_signal(features: np.ndarray, regime_id: int, regime_conf: float,
                   p: AIRequest, scores: dict,
                   ml_signal: int = 0, ml_confidence: float = 0.0
                   ) -> tuple[int, float, str]:
    """
    Regime-conditioned signal with optional ML blend.
    If a trained XGBoost model exists, its signal is blended in (20% weight).
    """
    weights  = REGIME_WEIGHTS.get(regime_id, REGIME_WEIGHTS[2])
    key_feat = features[KEY_FEATURE_IDX]
    raw_score = float(np.dot(key_feat, weights))
    raw_score = np.clip(raw_score, -1, 1)

    # ML blend — only if model available and confident
    if ml_signal != 0 and ml_confidence >= 0.55:
        ml_nudge  = ml_signal * ml_confidence * 0.20
        raw_score = np.clip(raw_score + ml_nudge, -1, 1)

    allowed_direction = _regime_filter(regime_id, p, raw_score)
    if allowed_direction == 0:
        return 0, 0.0, "Filtered by regime rules"

    if allowed_direction > 0 and raw_score < 0:
        raw_score = abs(raw_score) * 0.3
    if allowed_direction < 0 and raw_score > 0:
        raw_score = -abs(raw_score) * 0.3

    htf_agreement = 1.0 if (raw_score > 0 and float(features[3]) > 0) or \
                            (raw_score < 0 and float(features[3]) < 0) else 0.7
    confidence = float(np.clip(regime_conf * abs(raw_score) * htf_agreement, 0, 1))

    if   raw_score >=  0.35: signal = 2
    elif raw_score >=  0.15: signal = 1
    elif raw_score <= -0.35: signal = -2
    elif raw_score <= -0.15: signal = -1
    else:                    signal = 0

    reason = _build_reasoning(regime_id, raw_score, confidence, p, scores, ml_signal, ml_confidence)
    return signal, confidence, reason


def _regime_filter(regime_id: int, p: AIRequest, raw_score: float) -> int:
    if regime_id == 0:
        return +1 if p.bars.close[0] > p.ema50 else 0
    if regime_id == 1:
        return -1 if p.bars.close[0] < p.ema50 else 0
    if regime_id == 2:
        if raw_score > 0: return +1 if p.rsi < 45 else 0
        if raw_score < 0: return -1 if p.rsi > 55 else 0
        return 0
    if regime_id == 3:
        return (1 if raw_score > 0 else -1) if abs(raw_score) >= 0.40 else 0
    if regime_id == 4:
        c, h, l = p.bars.close, p.bars.high, p.bars.low
        hh50 = max(h[1:min(51, len(h))]) if len(h) > 1 else c[0]
        ll50 = min(l[1:min(51, len(l))]) if len(l) > 1 else c[0]
        if c[0] > hh50: return +1
        if c[0] < ll50: return -1
        return 0
    return 2


def _build_reasoning(regime_id, score, conf, p, feature_scores, ml_sig=0, ml_conf=0.0):
    parts = [f"Regime={REGIME_NAMES[regime_id]}"]
    parts.append(f"Score={score:+.3f}")
    parts.append(f"Conf={conf:.1%}")
    parts.append(f"ADX={p.adx:.1f}")
    parts.append(f"RSI={p.rsi:.1f}")
    if feature_scores.get("ema_alignment", 0) > 0.4:
        parts.append("EMA=BullStack")
    elif feature_scores.get("ema_alignment", 0) < -0.4:
        parts.append("EMA=BearStack")
    if abs(feature_scores.get("breakout_score", 0)) > 0.3:
        parts.append(f"Breakout={feature_scores['breakout_score']:+.2f}")
    if ml_sig != 0 and ml_conf >= 0.55:
        parts.append(f"ML={SIGNAL_NAMES.get(ml_sig,'?')}({ml_conf:.0%})")
    return " | ".join(parts)


# ──────────────────────────────────────────────────────────────────────
#  POSITION SIZER
# ──────────────────────────────────────────────────────────────────────

def compute_position(p: AIRequest, signal: int, confidence: float,
                     regime_id: int) -> tuple[float, float, float, float, float, float]:
    if signal == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    is_buy  = signal > 0
    price   = p.bars.close[0]
    atr     = p.atr

    regime_mult = {
        0: (2.0, 3.5),
        1: (2.0, 3.5),
        2: (1.5, 2.5),
        3: (2.8, 3.5),
        4: (1.8, 4.0),
    }
    sl_mult, tp_mult = regime_mult.get(regime_id, (2.0, 3.5))
    if abs(signal) == 2:
        tp_mult *= 1.25

    sl_dist  = atr * sl_mult
    tp_dist  = atr * tp_mult
    sl_price = (price - sl_dist) if is_buy else (price + sl_dist)
    tp_price = (price + tp_dist) if is_buy else (price - tp_dist)
    rr       = tp_dist / (sl_dist + 1e-9)

    p_win  = float(np.clip(confidence, 0.4, 0.85))
    q_lose = 1.0 - p_win
    kelly  = max(0.0, (p_win * rr - q_lose) / rr) * 0.5

    effective_risk_pct = min(kelly * 100, p.risk_pct)
    sizing_base = p.allocated_capital if p.allocated_capital > 0 else p.account_balance
    risk_amount = sizing_base * (effective_risk_pct / 100.0)

    if p.tick_size <= 0 or p.tick_value <= 0:
        lots = p.min_lot
    else:
        lots = risk_amount / (sl_dist / p.tick_size * p.tick_value)

    lots = _normalize_lots(lots, p)
    return lots, sl_price, tp_price, sl_mult, tp_mult, rr


def _normalize_lots(lots: float, p) -> float:
    step = p.lot_step if p.lot_step > 0 else 0.01
    lots = math.floor(lots / step) * step
    lots = max(p.min_lot, min(p.max_lot, lots))
    return round(lots, 8)


# ──────────────────────────────────────────────────────────────────────
#  MODAL ENDPOINTS
# ──────────────────────────────────────────────────────────────────────

# ── 1. PREDICT (main endpoint — unchanged interface) ──────────────────

@app.function(
    image=image,
    volumes={MODEL_DIR: volume},
    secrets=[secrets],
    timeout=30,
    memory=512,
    cpu=1.0,
    min_containers=1,
    max_containers=5,
)
@modal.fastapi_endpoint(method="POST", label="apex-hydracrypto-predict")
async def predict(request: AIRequest) -> AIResponse:
    """
    Main AI prediction endpoint.
    Called by MT5 EA on every scan cycle for each symbol.
    """
    # ── News Filter (NEW) ────────────────────────────────────────────
    news_blocked, news_reason = check_news_filter(request)
    if news_blocked:
        return AIResponse(
            symbol=request.symbol, regime_id=5, regime_name="Undefined",
            regime_conf=0.0, signal=0, signal_name="Hold", confidence=0.0,
            lots=0.0, sl_price=0.0, tp_price=0.0, sl_atr_mult=0.0,
            tp_atr_mult=0.0, rr_ratio=0.0, feature_scores={},
            reasoning=news_reason, model_version=MODEL_VERSION,
            server_ts=datetime.now(timezone.utc).isoformat(),
            news_blocked=True, ml_signal=0, ml_confidence=0.0,
        )

    # ── Feature Engineering ──────────────────────────────────────────
    features, feature_scores = build_features(request)

    # ── ML Model (optional boost) ────────────────────────────────────
    ml_sig, ml_conf = ml_predict(features)

    # ── Regime Classification ────────────────────────────────────────
    regime_id, regime_name, regime_conf = classify_regime(request, features)

    # ── Signal Generation ────────────────────────────────────────────
    signal, confidence, reasoning = compute_signal(
        features, regime_id, regime_conf, request, feature_scores, ml_sig, ml_conf
    )

    # ── Position Sizing + SL/TP ──────────────────────────────────────
    lots, sl_price, tp_price, sl_mult, tp_mult, rr = compute_position(
        request, signal, confidence, regime_id
    )

    # ── Final confidence guard ───────────────────────────────────────
    MIN_CONFIDENCE = 0.52
    if confidence < MIN_CONFIDENCE:
        signal = 0; lots = sl_price = tp_price = 0.0

    return AIResponse(
        symbol        = request.symbol,
        regime_id     = regime_id,
        regime_name   = regime_name,
        regime_conf   = round(regime_conf, 4),
        signal        = signal,
        signal_name   = SIGNAL_NAMES.get(signal, "Hold"),
        confidence    = round(confidence, 4),
        lots          = round(lots, 4),
        sl_price      = round(sl_price, request.digits),
        tp_price      = round(tp_price, request.digits),
        sl_atr_mult   = round(sl_mult, 2),
        tp_atr_mult   = round(tp_mult, 2),
        rr_ratio      = round(rr, 2),
        feature_scores= feature_scores,
        reasoning     = reasoning,
        model_version = MODEL_VERSION,
        server_ts     = datetime.now(timezone.utc).isoformat(),
        news_blocked  = False,
        ml_signal     = ml_sig,
        ml_confidence = round(ml_conf, 4),
    )


# ── 2. TRAIN endpoint (NEW) ───────────────────────────────────────────

@app.function(
    image=image,
    volumes={MODEL_DIR: volume},
    secrets=[secrets],
    timeout=120,
    memory=1024,
    cpu=2.0,
    min_containers=0,   # No need to keep warm — called infrequently
    max_containers=1,
)
@modal.fastapi_endpoint(method="POST", label="apex-hydracrypto-train")
async def train(request: TrainRequest) -> TrainResponse:
    """
    Train or update the XGBoost signal classifier.

    HOW IT WORKS:
    ─────────────
    1. Each time a trade CLOSES in MT5, your EA should POST a TrainSample
       with the 26-dim feature vector (captured at entry) + the outcome.
    2. New samples are appended to the persistent dataset on the volume.
    3. When dataset >= 100 samples (or force_retrain=True), re-trains XGBoost.
    4. Model is saved to volume — loaded automatically by /predict.

    CALLING FROM MT5:
    ─────────────────
    On TradeTransaction / trade close event, POST to:
      https://YOUR_WORKSPACE--apex-hydracrypto-train.modal.run

    Payload example:
    {
      "samples": [{
        "symbol": "BTCUSD",
        "features": [0.5, 0.3, ...],   // 26 floats from build_features
        "regime_id": 0,
        "signal_given": 2,
        "outcome": 125.50,
        "won": true,
        "timestamp": "2025-01-01T12:00:00Z"
      }],
      "force_retrain": false
    }

    NOTE: Store features at entry time in your EA (e.g. a global array keyed
    by ticket number). Send them when the trade closes.
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, f1_score
    import xgboost as xgb
    from scipy.special import expit

    # ── Load existing dataset ─────────────────────────────────────────
    X_existing = np.empty((0, 26), dtype=np.float32)
    y_existing = np.empty(0, dtype=np.int32)

    if os.path.exists(DATASET_PATH):
        try:
            data = np.load(DATASET_PATH)
            X_existing = data["X"]
            y_existing = data["y"]
        except Exception as e:
            print(f"[Train] Could not load existing data: {e}")

    # ── Append new samples ────────────────────────────────────────────
    new_X = []
    new_y = []
    for s in request.samples:
        if len(s.features) != 26:
            continue
        feat = np.array(s.features, dtype=np.float32)
        # Label: 2=Buy(won), 0=Sell(won), 1=Hold/loss
        if s.won and s.signal_given > 0:   label = 2   # Bullish win
        elif s.won and s.signal_given < 0: label = 0   # Bearish win
        else:                              label = 1   # Loss / hold
        new_X.append(feat)
        new_y.append(label)

    if new_X:
        X_new = np.array(new_X, dtype=np.float32)
        y_new = np.array(new_y, dtype=np.int32)
        X_all = np.vstack([X_existing, X_new])
        y_all = np.concatenate([y_existing, y_new])
    else:
        X_all = X_existing
        y_all = y_existing

    total_samples = len(X_all)

    # ── Save accumulated dataset ──────────────────────────────────────
    os.makedirs(MODEL_DIR, exist_ok=True)
    np.savez(DATASET_PATH, X=X_all, y=y_all)

    # ── Train only if enough data ─────────────────────────────────────
    MIN_TRAIN_SAMPLES = 50
    if total_samples < MIN_TRAIN_SAMPLES and not request.force_retrain:
        volume.commit()
        return TrainResponse(
            status="pending",
            samples_total=total_samples,
            accuracy=0.0, f1_score=0.0,
            model_version=MODEL_VERSION,
            trained_at=datetime.now(timezone.utc).isoformat(),
            message=f"Collecting data... {total_samples}/{MIN_TRAIN_SAMPLES} samples. Will train at {MIN_TRAIN_SAMPLES}.",
        )

    # ── Scale features ────────────────────────────────────────────────
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)

    # ── Train XGBoost classifier ──────────────────────────────────────
    # 3 classes: 0=Bearish win, 1=Loss/hold, 2=Bullish win
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.5,
        reg_lambda=1.0,
        use_label_encoder=False,
        eval_metric="mlogloss",
        n_jobs=-1,
        random_state=42,
        num_class=3,
        objective="multi:softprob",
    )

    # Simple train/val split (80/20)
    split = int(len(X_scaled) * 0.8)
    X_tr, X_val = X_scaled[:split], X_scaled[split:]
    y_tr, y_val = y_all[:split], y_all[split:]

    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    # ── Evaluate ──────────────────────────────────────────────────────
    if len(X_val) > 0:
        y_pred = model.predict(X_val)
        acc = float(accuracy_score(y_val, y_pred))
        f1  = float(f1_score(y_val, y_pred, average="weighted", zero_division=0))
    else:
        acc = f1 = 0.0

    # ── Save model + metadata ─────────────────────────────────────────
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    version = f"xgb-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M')}"
    meta = {
        "version": version,
        "samples": total_samples,
        "accuracy": round(acc, 4),
        "f1": round(f1, 4),
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(MODEL_META, "w") as f:
        json.dump(meta, f)

    volume.commit()  # Flush writes to persistent volume

    return TrainResponse(
        status="trained",
        samples_total=total_samples,
        accuracy=round(acc, 4),
        f1_score=round(f1, 4),
        model_version=version,
        trained_at=meta["trained_at"],
        message=f"Model trained on {total_samples} samples. Val accuracy={acc:.1%}, F1={f1:.3f}",
    )


# ── 3. BACKTEST endpoint (NEW) ────────────────────────────────────────

@app.function(
    image=image,
    volumes={MODEL_DIR: volume},
    secrets=[secrets],
    timeout=300,
    memory=1024,
    cpu=2.0,
    min_containers=0,
    max_containers=2,
)
@modal.fastapi_endpoint(method="POST", label="apex-hydracrypto-backtest")
async def backtest(request: BacktestRequest) -> BacktestResponse:
    """
    Run the ApexHydra strategy on historical bar data.

    USAGE FROM MT5:
    ───────────────
    Download history bars with CopyRates(), build the BacktestBar array,
    and POST to this endpoint. MT5 Script example is in the repo.

    USAGE FROM STREAMLIT:
    ──────────────────────
    The dashboard can let users upload a CSV of bars and trigger a backtest.
    See apex_hydra_dashboard.py for the backtest tab UI.

    MINIMUM: Send at least 250 bars (200 needed for EMA200 warmup + 50 to trade).
    RECOMMENDED: 1000+ bars for statistically significant results.
    """
    bars = request.bars
    N    = len(bars)

    if N < 100:
        return BacktestResponse(
            symbol=request.symbol, timeframe=request.timeframe, bars_tested=N,
            total_trades=0, wins=0, losses=0, win_rate=0.0, total_pnl=0.0,
            final_balance=request.initial_balance, max_drawdown_pct=0.0,
            sharpe_ratio=0.0, profit_factor=0.0, avg_rr=0.0,
            trades=[], equity_curve=[request.initial_balance], regime_breakdown={},
        )

    balance       = request.initial_balance
    peak_balance  = balance
    max_dd        = 0.0
    trades_out    = []
    equity_curve  = [balance]
    pnl_list      = []
    regime_stats: dict[str, dict] = {}

    # Simulate: walk forward bar by bar (starting from bar 200 for warmup)
    warmup = min(200, N // 4)
    open_trade = None

    for i in range(warmup, N):
        bar = bars[i]

        # ── Manage open trade ────────────────────────────────────────
        if open_trade is not None:
            ot = open_trade
            is_buy = ot["signal"] > 0

            # Check SL/TP hit (simplified: use bar high/low)
            sl_hit = (bar.low  <= ot["sl"]) if is_buy  else (bar.high >= ot["sl"])
            tp_hit = (bar.high >= ot["tp"]) if is_buy  else (bar.low  <= ot["tp"])

            if sl_hit or tp_hit:
                exit_price = ot["sl"] if sl_hit else ot["tp"]
                pnl_pts    = (exit_price - ot["entry"]) if is_buy else (ot["entry"] - exit_price)
                tick_size  = request.tick_size if request.tick_size > 0 else 0.01
                tick_val   = request.tick_value if request.tick_value > 0 else 1.0
                pnl        = pnl_pts / tick_size * tick_val * ot["lots"]
                won        = pnl > 0
                balance   += pnl
                peak_balance = max(peak_balance, balance)
                dd_now = (peak_balance - balance) / peak_balance * 100
                max_dd = max(max_dd, dd_now)

                t = BacktestTrade(
                    entry_time  = ot["entry_time"],
                    exit_time   = bar.timestamp,
                    signal      = ot["signal"],
                    regime      = ot["regime"],
                    confidence  = ot["confidence"],
                    lots        = ot["lots"],
                    entry_price = ot["entry"],
                    exit_price  = round(exit_price, request.digits),
                    sl          = ot["sl"],
                    tp          = ot["tp"],
                    pnl         = round(pnl, 2),
                    won         = won,
                    bars_held   = i - ot["bar_idx"],
                )
                trades_out.append(t)
                pnl_list.append(pnl)
                equity_curve.append(round(balance, 2))

                # Regime stats
                r = ot["regime"]
                if r not in regime_stats:
                    regime_stats[r] = {"trades": 0, "wins": 0, "pnl": 0.0}
                regime_stats[r]["trades"] += 1
                regime_stats[r]["wins"]   += int(won)
                regime_stats[r]["pnl"]    += pnl

                open_trade = None

        # ── Skip if trade already open (1 at a time for simplicity) ──
        if open_trade is not None:
            continue

        # ── Build a synthetic AIRequest from this bar ─────────────────
        look_back = min(i, 100)
        subset = bars[i - look_back: i + 1]

        fake_request = AIRequest(
            symbol=request.symbol, timeframe=request.timeframe, magic=0,
            timestamp=bar.timestamp,
            account_balance=balance, account_equity=balance,
            risk_pct=request.risk_pct, max_positions=1, open_positions=0,
            bars=BarData(
                open   = [b.open   for b in reversed(subset)],
                high   = [b.high   for b in reversed(subset)],
                low    = [b.low    for b in reversed(subset)],
                close  = [b.close  for b in reversed(subset)],
                volume = [b.volume for b in reversed(subset)],
            ),
            atr=bar.atr, atr_avg=bar.atr_avg,
            adx=bar.adx, plus_di=bar.plus_di, minus_di=bar.minus_di,
            rsi=bar.rsi, macd=bar.macd, macd_signal=bar.macd_signal,
            macd_hist=bar.macd_hist, ema20=bar.ema20, ema50=bar.ema50,
            ema200=bar.ema200, htf_ema50=bar.htf_ema50, htf_ema200=bar.htf_ema200,
            tick_value=request.tick_value, tick_size=request.tick_size,
            min_lot=request.min_lot, max_lot=request.max_lot,
            lot_step=request.lot_step, point=request.point, digits=request.digits,
        )

        features, feature_scores = build_features(fake_request)
        ml_sig, ml_conf = ml_predict(features)
        regime_id, regime_name, regime_conf = classify_regime(fake_request, features)
        signal, confidence, _ = compute_signal(
            features, regime_id, regime_conf, fake_request, feature_scores, ml_sig, ml_conf
        )

        if signal == 0 or confidence < request.min_confidence:
            continue

        lots, sl_price, tp_price, sl_m, tp_m, rr = compute_position(
            fake_request, signal, confidence, regime_id
        )

        if rr < request.min_rr or lots <= 0:
            continue

        # Spread cost on entry
        spread_cost = request.spread_points * request.point
        entry_price = bar.close + (spread_cost if signal > 0 else -spread_cost)

        open_trade = {
            "signal":     signal,
            "regime":     regime_name,
            "confidence": confidence,
            "lots":       lots,
            "entry":      entry_price,
            "sl":         sl_price,
            "tp":         tp_price,
            "entry_time": bar.timestamp,
            "bar_idx":    i,
        }

    # ── Compile results ───────────────────────────────────────────────
    total   = len(trades_out)
    wins    = sum(1 for t in trades_out if t.won)
    losses  = total - wins
    win_rate = wins / total if total else 0.0
    total_pnl = sum(t.pnl for t in trades_out)

    # Profit factor
    gross_profit = sum(t.pnl for t in trades_out if t.pnl > 0)
    gross_loss   = abs(sum(t.pnl for t in trades_out if t.pnl < 0))
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Sharpe (annualised, assuming each trade is ~1 bar)
    if len(pnl_list) >= 2:
        pnl_arr = np.array(pnl_list)
        sharpe = float(np.mean(pnl_arr) / (np.std(pnl_arr) + 1e-9) * math.sqrt(252))
    else:
        sharpe = 0.0

    avg_rr = float(np.mean([t.pnl / abs(t.pnl + 1e-9) for t in trades_out])) if trades_out else 0.0

    # Regime breakdown with win rates
    regime_breakdown = {}
    for r, s in regime_stats.items():
        regime_breakdown[r] = {
            "trades": s["trades"],
            "wins": s["wins"],
            "win_rate": round(s["wins"] / s["trades"] * 100, 1) if s["trades"] else 0.0,
            "total_pnl": round(s["pnl"], 2),
        }

    return BacktestResponse(
        symbol          = request.symbol,
        timeframe       = request.timeframe,
        bars_tested     = N,
        total_trades    = total,
        wins            = wins,
        losses          = losses,
        win_rate        = round(win_rate, 4),
        total_pnl       = round(total_pnl, 2),
        final_balance   = round(balance, 2),
        max_drawdown_pct= round(max_dd, 2),
        sharpe_ratio    = round(sharpe, 3),
        profit_factor   = round(min(pf, 99.0), 3),
        avg_rr          = round(avg_rr, 3),
        trades          = trades_out,
        equity_curve    = equity_curve,
        regime_breakdown= regime_breakdown,
    )


# ── 4. Health check ───────────────────────────────────────────────────

@app.function(image=image, volumes={MODEL_DIR: volume}, timeout=10)
@modal.fastapi_endpoint(method="GET", label="apex-hydracrypto-health")
async def health():
    model_info = {"loaded": False, "version": None, "samples": 0, "accuracy": None}
    try:
        if os.path.exists(MODEL_META):
            with open(MODEL_META) as f:
                meta = json.load(f)
            model_info = {
                "loaded":   os.path.exists(MODEL_PATH),
                "version":  meta.get("version"),
                "samples":  meta.get("samples", 0),
                "accuracy": meta.get("accuracy"),
                "trained_at": meta.get("trained_at"),
            }
    except Exception:
        pass

    return {
        "status":        "ok",
        "version":       MODEL_VERSION,
        "ts":            datetime.now(timezone.utc).isoformat(),
        "ml_model":      model_info,
        "news_filter":   "enabled",
        "backtest":      "enabled",
    }


# ──────────────────────────────────────────────────────────────────────
#  LOCAL TEST
# ──────────────────────────────────────────────────────────────────────

@app.local_entrypoint()
def test():
    import random
    random.seed(42)
    n = 200
    base = 65000.0
    closes = [base + random.gauss(0, 500) for _ in range(n)]

    payload = AIRequest(
        symbol="BTCUSD", timeframe="H1", magic=20250228,
        timestamp=datetime.now(timezone.utc).isoformat(),
        account_balance=10000, account_equity=10250,
        risk_pct=1.0, max_positions=3, open_positions=0,
        bars=BarData(
            open=closes[:], high=[c + 200 for c in closes],
            low=[c - 200 for c in closes], close=closes,
            volume=[random.uniform(100, 500) for _ in range(n)],
        ),
        atr=350, atr_avg=280, adx=32, plus_di=28, minus_di=18,
        rsi=58, macd=120, macd_signal=90, macd_hist=30,
        ema20=64800, ema50=63500, ema200=60000,
        htf_ema50=63000, htf_ema200=58000,
        tick_value=1.0, tick_size=0.01, min_lot=0.01,
        max_lot=100, lot_step=0.01, point=0.01, digits=2,
        news_blackout=False, news_minutes_away=999, news_buffer_minutes=15,
    )

    features, scores = build_features(payload)
    regime_id, regime_name, regime_conf = classify_regime(payload, features)
    ml_sig, ml_conf = ml_predict(features)
    signal, confidence, reasoning = compute_signal(features, regime_id, regime_conf, payload, scores, ml_sig, ml_conf)
    lots, sl, tp, sl_m, tp_m, rr = compute_position(payload, signal, confidence, regime_id)

    print(f"\n{'='*60}")
    print(f"  ApexHydra Modal AI v4.1 — Local Test")
    print(f"{'='*60}")
    print(f"  Symbol:    BTCUSD")
    print(f"  Regime:    {regime_name} ({regime_conf:.1%})")
    print(f"  Signal:    {SIGNAL_NAMES[signal]} ({signal})")
    print(f"  Conf:      {confidence:.1%}")
    print(f"  ML Signal: {SIGNAL_NAMES.get(ml_sig, 'n/a')} (no model yet — expected)")
    print(f"  Lots:      {lots}")
    print(f"  SL:        {sl:.2f}  ({sl_m}×ATR)")
    print(f"  TP:        {tp:.2f}  ({tp_m}×ATR)")
    print(f"  R:R:       {rr:.2f}")
    print(f"  Reasoning: {reasoning}")
    print(f"{'='*60}\n")

    # Test news filter
    payload.news_blackout = True
    blocked, reason = check_news_filter(payload)
    print(f"  News filter (blackout=True): blocked={blocked}, reason={reason}")
    payload.news_blackout = False
    payload.news_minutes_away = 5
    blocked, reason = check_news_filter(payload)
    print(f"  News filter (5 min away):   blocked={blocked}, reason={reason}")
    print()