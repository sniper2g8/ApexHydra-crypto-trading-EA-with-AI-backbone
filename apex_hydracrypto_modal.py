"""
╔══════════════════════════════════════════════════════════════════════╗
║            ApexHydra Crypto — Modal AI Engine Server                ║
║  Decides: Regime | Signal | Lot Size | Stop Loss | Take Profit       ║
║                                                                      ║
║  Deploy:  modal deploy apex_hydracrypto_modal.py                           ║
║  Docs:    https://modal.com/docs                                     ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import modal
import numpy as np
import json
import math
from datetime import datetime, timezone
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

# Persistent volume — stores model weights & trade memory between calls
volume = modal.Volume.from_name("apex-hydracrypto-models", create_if_missing=True)
MODEL_DIR = "/models"

# Supabase secret (set via: modal secret create apex-hydracrypto-secrets)
secrets = modal.Secret.from_name("apex-hydracrypto-secrets", required=False)

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
    allocated_capital: float = 0.0   # 0 = use full balance; >0 = cap sizing to this amount
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

    # Online learning history (last 20 trades)
    recent_signals:   Optional[list[int]]   = None
    recent_outcomes:  Optional[list[float]] = None
    recent_regimes:   Optional[list[int]]   = None

class AIResponse(BaseModel):
    symbol:        str
    regime_id:     int      # 0 TrendBull 1 TrendBear 2 Ranging 3 HighVol 4 Breakout 5 Undefined
    regime_name:   str
    regime_conf:   float    # 0.0–1.0
    signal:        int      # -2 StrongSell -1 Sell 0 Hold 1 Buy 2 StrongBuy
    signal_name:   str
    confidence:    float    # final trade confidence 0.0–1.0
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


# ──────────────────────────────────────────────────────────────────────
#  REGIME IDs & NAMES
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

MODEL_VERSION = "3.1.0"

# ──────────────────────────────────────────────────────────────────────
#  FEATURE ENGINEERING
# ──────────────────────────────────────────────────────────────────────

def build_features(p: AIRequest) -> tuple[np.ndarray, dict]:
    """
    Build a 26-dimensional normalised feature vector.
    Returns (features_array, scores_dict) for reasoning transparency.
    """
    c  = p.bars.close
    h  = p.bars.high
    l  = p.bars.low
    o  = p.bars.open
    v  = p.bars.volume
    n  = min(len(c), 100)
    atr = p.atr if p.atr > 0 else 1e-9

    # ── 1. Trend alignment (EMA stack) ──────────────────────────────
    ema_full_bull = (c[0] > p.ema20 > p.ema50 > p.ema200)
    ema_full_bear = (c[0] < p.ema20 < p.ema50 < p.ema200)
    if ema_full_bull:    ema_align = +1.0
    elif ema_full_bear:  ema_align = -1.0
    elif c[0] > p.ema50 and p.ema50 > p.ema200: ema_align = +0.5
    elif c[0] < p.ema50 and p.ema50 < p.ema200: ema_align = -0.5
    elif c[0] > p.ema200: ema_align = +0.25
    else:                ema_align = -0.25

    # ── 2. EMA separation (momentum of trend) ───────────────────────
    ema_sep = np.clip((p.ema20 - p.ema50) / atr, -3, 3)

    # ── 3. Price distance from EMA50 ────────────────────────────────
    ema50_dist = np.clip((c[0] - p.ema50) / atr, -3, 3)

    # ── 4. HTF bias ──────────────────────────────────────────────────
    if c[0] > p.htf_ema50 and p.htf_ema50 > p.htf_ema200:   htf_bias = +1.0
    elif c[0] < p.htf_ema50 and p.htf_ema50 < p.htf_ema200: htf_bias = -1.0
    elif c[0] > p.htf_ema50: htf_bias = +0.4
    else:                    htf_bias = -0.4

    # ── 5. ADX trend strength ────────────────────────────────────────
    adx_norm  = np.clip((p.adx - 25.0) / 40.0, -1, 1)
    di_diff   = np.clip((p.plus_di - p.minus_di) / 30.0, -1, 1)

    # ── 6. RSI ───────────────────────────────────────────────────────
    rsi_norm  = (p.rsi - 50.0) / 50.0
    rsi_ob    =  max(0, (p.rsi - 70)) / 30.0
    rsi_os    = -max(0, (30 - p.rsi)) / 30.0
    rsi_ext   = rsi_ob + rsi_os

    # ── 7. MACD ──────────────────────────────────────────────────────
    macd_hist_norm = np.clip(p.macd_hist / (atr * 0.05 + 1e-9), -3, 3)
    macd_cross     = 1.0 if p.macd > p.macd_signal else -1.0

    # ── 8. Volatility ────────────────────────────────────────────────
    vol_ratio = p.atr / (p.atr_avg if p.atr_avg > 0 else p.atr)
    vol_norm  = np.clip((vol_ratio - 1.0) / 1.5, -1, 1)
    vol_flag  = 1.0 if vol_ratio > 1.6 else (-0.5 if vol_ratio < 0.6 else 0.0)

    # ── 9. Candle structure ──────────────────────────────────────────
    bar_range = h[0] - l[0]
    if bar_range > 0:
        body     = abs(c[0] - o[0]) / bar_range
        candle   = body if c[0] > o[0] else -body
        upper_wick = (h[0] - max(c[0], o[0])) / bar_range
        lower_wick = (min(c[0], o[0]) - l[0]) / bar_range
    else:
        candle = upper_wick = lower_wick = 0.0

    # ── 10. Price position in recent range ──────────────────────────
    look = min(20, n)
    hh20 = max(h[:look]) if look else c[0]
    ll20 = min(l[:look]) if look else c[0]
    rng20 = hh20 - ll20
    pos20 = ((c[0] - ll20) / rng20) * 2 - 1 if rng20 > 0 else 0.0

    # ── 11. Rate of change ───────────────────────────────────────────
    def roc(period):
        if len(c) > period and c[period] > 0:
            return np.clip((c[0] - c[period]) / c[period] * 100 / 10, -2, 2)
        return 0.0
    roc5, roc10, roc20 = roc(5), roc(10), roc(20)

    # ── 12. Breakout score ───────────────────────────────────────────
    hh50 = max(h[1:min(51, len(h))]) if len(h) > 1 else c[0]
    ll50 = min(l[1:min(51, len(l))]) if len(l) > 1 else c[0]
    spread50 = hh50 - ll50
    if c[0] > hh50 and spread50 > 0:
        breakout = min(1.0,  (c[0] - hh50) / spread50)
    elif c[0] < ll50 and spread50 > 0:
        breakout = max(-1.0, (c[0] - ll50) / spread50)
    else:
        breakout = 0.0

    # ── 13. Z-score (mean reversion signal) ─────────────────────────
    look_z = min(20, n)
    arr_z  = np.array(c[:look_z][::-1])
    z_mean, z_std = arr_z.mean(), arr_z.std()
    zscore = np.clip((c[0] - z_mean) / (z_std + 1e-9), -3, 3)

    # ── 14. Volume momentum ──────────────────────────────────────────
    vol_bars = min(10, len(v))
    if vol_bars > 1 and v[1] > 0:
        vol_mom = np.clip((v[0] - np.mean(v[1:vol_bars])) / (np.mean(v[1:vol_bars]) + 1e-9), -2, 2)
    else:
        vol_mom = 0.0

    # ── 15. Historical learning bias ─────────────────────────────────
    hist_bias = _compute_history_bias(p.recent_signals, p.recent_outcomes, p.recent_regimes)

    features = np.array([
        ema_align,       # 0
        ema_sep,         # 1
        ema50_dist,      # 2
        htf_bias,        # 3
        adx_norm,        # 4
        di_diff,         # 5
        rsi_norm,        # 6
        rsi_ext,         # 7
        macd_hist_norm,  # 8
        macd_cross,      # 9
        vol_norm,        # 10
        vol_flag,        # 11
        candle,          # 12
        upper_wick,      # 13
        lower_wick,      # 14
        pos20,           # 15
        roc5,            # 16
        roc10,           # 17
        roc20,           # 18
        breakout,        # 19
        zscore,          # 20
        vol_mom,         # 21
        hist_bias,       # 22
        adx_norm * di_diff,         # 23 — interaction: trend strength × direction
        rsi_norm * macd_hist_norm,  # 24 — interaction: momentum confluence
        ema_align * htf_bias,       # 25 — interaction: LTF/HTF agreement
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
    w   = np.exp(np.linspace(-2, 0, n))   # recency weighting
    pos = sig > 0
    neg = sig < 0
    pos_wr = np.average((out[pos] > 0).astype(float), weights=w[pos]) if pos.any() else 0.5
    neg_wr = np.average((out[neg] > 0).astype(float), weights=w[neg]) if neg.any() else 0.5
    return float(np.clip((pos_wr - neg_wr), -0.5, 0.5))


# ──────────────────────────────────────────────────────────────────────
#  REGIME CLASSIFIER
# ──────────────────────────────────────────────────────────────────────

def classify_regime(p: AIRequest, features: np.ndarray) -> tuple[int, str, float]:
    """
    Probabilistic regime classifier.
    Returns: (regime_id, regime_name, confidence)
    """
    vol_ratio   = p.atr / (p.atr_avg if p.atr_avg > 0 else p.atr)
    is_trending = p.adx >= 25
    bull_trend  = is_trending and p.plus_di > p.minus_di
    bear_trend  = is_trending and p.minus_di > p.plus_di
    high_vol    = vol_ratio >= 1.6
    breakout_s  = abs(float(features[19]))

    scores = np.zeros(6)

    # Breakout: price pierces 50-bar range extreme WITH high vol
    if breakout_s > 0.35 and high_vol:
        scores[4] = 0.50 + breakout_s * 0.40
        scores[3] = 0.20

    # High volatility choppy
    elif high_vol and not is_trending:
        scores[3] = 0.55 + min(0.35, (vol_ratio - 1.6) * 0.30)
        scores[2] = 0.20

    # Bullish trend
    elif bull_trend:
        adx_boost = min(0.40, (p.adx - 25) / 40 * 0.40)
        htf_boost = 0.08 if float(features[3]) > 0 else 0.0
        scores[0] = 0.55 + adx_boost + htf_boost
        scores[2] = 0.20

    # Bearish trend
    elif bear_trend:
        adx_boost = min(0.40, (p.adx - 25) / 40 * 0.40)
        htf_boost = 0.08 if float(features[3]) < 0 else 0.0
        scores[1] = 0.55 + adx_boost + htf_boost
        scores[2] = 0.20

    # Ranging / undefined
    else:
        ranging_conf = min(0.88, 0.50 + (25 - p.adx) / 25 * 0.38)
        scores[2] = ranging_conf
        scores[5] = max(0, 1.0 - ranging_conf)

    # Normalise to probability distribution
    total = scores.sum()
    if total > 0:
        scores /= total

    regime_id   = int(np.argmax(scores))
    regime_conf = float(scores[regime_id])
    return regime_id, REGIME_NAMES[regime_id], regime_conf


# ──────────────────────────────────────────────────────────────────────
#  SIGNAL ENGINE  (regime-conditioned scoring)
# ──────────────────────────────────────────────────────────────────────

# Regime-specific feature weights
# shape: (num_regimes=5, num_features=11 key scores)
REGIME_WEIGHTS = {
    #              ema  htf  adx  dir  rsi  macd  vol  bkout  z-sc  hist  volm
    0: np.array([ 0.25, 0.20, 0.15, 0.15, 0.05, 0.10, 0.02, 0.00,  0.00, 0.05, 0.03]),  # Trend Bull
    1: np.array([ 0.25, 0.20, 0.15, 0.15, 0.05, 0.10, 0.02, 0.00,  0.00, 0.05, 0.03]),  # Trend Bear
    2: np.array([ 0.05, 0.10, 0.02, 0.05, 0.25, 0.20, 0.05, 0.00,  0.20, 0.05, 0.03]),  # Ranging
    3: np.array([ 0.10, 0.10, 0.08, 0.08, 0.10, 0.10, 0.18, 0.10,  0.08, 0.05, 0.03]),  # High Vol
    4: np.array([ 0.10, 0.12, 0.08, 0.10, 0.05, 0.10, 0.10, 0.30,  0.00, 0.03, 0.02]),  # Breakout
}

KEY_FEATURE_IDX = [0, 3, 4, 5, 6, 8, 10, 19, 20, 22, 21]  # indices in full feature array

def compute_signal(features: np.ndarray, regime_id: int, regime_conf: float,
                   p: AIRequest, scores: dict) -> tuple[int, float, str]:
    """
    Regime-conditioned signal: applies appropriate weights,
    filters by regime-specific rules, returns (signal_int, confidence, reasoning).
    """
    weights  = REGIME_WEIGHTS.get(regime_id, REGIME_WEIGHTS[2])
    key_feat = features[KEY_FEATURE_IDX]

    # Weighted dot product → raw score in [-1, +1]
    raw_score = float(np.dot(key_feat, weights))
    raw_score = np.clip(raw_score, -1, 1)

    # ── Regime filters ────────────────────────────────────────────
    allowed_direction = _regime_filter(regime_id, p, raw_score)
    if allowed_direction == 0:
        return 0, 0.0, "Filtered by regime rules"

    # Align score with allowed direction
    if allowed_direction > 0 and raw_score < 0:
        raw_score = abs(raw_score) * 0.3   # Weak signal against regime — reduce
    if allowed_direction < 0 and raw_score > 0:
        raw_score = -abs(raw_score) * 0.3

    # ── Confidence: regime_conf × |raw_score| × htf_agreement ────
    htf_agreement = 1.0 if (raw_score > 0 and float(features[3]) > 0) or \
                            (raw_score < 0 and float(features[3]) < 0) else 0.7
    confidence = float(np.clip(regime_conf * abs(raw_score) * htf_agreement, 0, 1))

    # ── Signal classification ────────────────────────────────────
    if   raw_score >=  0.35: signal = 2
    elif raw_score >=  0.15: signal = 1
    elif raw_score <= -0.35: signal = -2
    elif raw_score <= -0.15: signal = -1
    else:                    signal = 0

    reason = _build_reasoning(regime_id, raw_score, confidence, p, scores)
    return signal, confidence, reason


def _regime_filter(regime_id: int, p: AIRequest, raw_score: float) -> int:
    """Returns +1 (long only), -1 (short only), 0 (blocked), or 2 (both)."""
    if regime_id == 0:   # Trend Bull — longs only
        return +1 if p.bars.close[0] > p.ema50 else 0
    if regime_id == 1:   # Trend Bear — shorts only
        return -1 if p.bars.close[0] < p.ema50 else 0
    if regime_id == 2:   # Ranging — mean reversion
        if raw_score > 0: return +1 if p.rsi < 45 else 0
        if raw_score < 0: return -1 if p.rsi > 55 else 0
        return 0
    if regime_id == 3:   # High Vol — require strong conviction
        return (1 if raw_score > 0 else -1) if abs(raw_score) >= 0.40 else 0
    if regime_id == 4:   # Breakout — follow breakout direction
        c, h, l = p.bars.close, p.bars.high, p.bars.low
        hh50 = max(h[1:min(51, len(h))]) if len(h) > 1 else c[0]
        ll50 = min(l[1:min(51, len(l))]) if len(l) > 1 else c[0]
        if c[0] > hh50: return +1
        if c[0] < ll50: return -1
        return 0
    return 2  # Undefined — allow both but weak


def _build_reasoning(regime_id: int, score: float, conf: float,
                     p: AIRequest, feature_scores: dict) -> str:
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
    return " | ".join(parts)


# ──────────────────────────────────────────────────────────────────────
#  POSITION SIZER  (Kelly Criterion + ATR-based SL/TP)
# ──────────────────────────────────────────────────────────────────────

def compute_position(p: AIRequest, signal: int, confidence: float,
                     regime_id: int) -> tuple[float, float, float, float, float, float]:
    """
    Returns: (lots, sl_price, tp_price, sl_atr_mult, tp_atr_mult, rr)
    """
    if signal == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    is_buy  = signal > 0
    price   = p.bars.close[0]
    atr     = p.atr

    # ── Adaptive SL/TP multipliers per regime ───────────────────────
    regime_mult = {
        0: (2.0, 3.5),   # Trend Bull — standard
        1: (2.0, 3.5),   # Trend Bear
        2: (1.5, 2.5),   # Ranging  — tighter
        3: (2.8, 3.5),   # High Vol — wider SL
        4: (1.8, 4.0),   # Breakout — wide TP
    }
    sl_mult, tp_mult = regime_mult.get(regime_id, (2.0, 3.5))

    # Boost TP for strong signals
    if abs(signal) == 2:
        tp_mult *= 1.25

    sl_dist  = atr * sl_mult
    tp_dist  = atr * tp_mult
    sl_price = (price - sl_dist) if is_buy else (price + sl_dist)
    tp_price = (price + tp_dist) if is_buy else (price - tp_dist)
    rr       = tp_dist / (sl_dist + 1e-9)

    # ── Kelly-adjusted lot sizing ────────────────────────────────────
    # Half-Kelly for conservatism: f = (p*b - q) / b  × 0.5
    # p = confidence (est. win prob), b = R:R, q = 1-p
    p_win  = float(np.clip(confidence, 0.4, 0.85))
    q_lose = 1.0 - p_win
    kelly  = max(0.0, (p_win * rr - q_lose) / rr) * 0.5

    # Cap Kelly at user-specified risk %
    effective_risk_pct = min(kelly * 100, p.risk_pct)
    # Use allocated_capital if set, otherwise full balance
    sizing_base  = p.allocated_capital if p.allocated_capital > 0 else p.account_balance
    risk_amount  = sizing_base * (effective_risk_pct / 100.0)

    tick_value   = p.tick_value
    tick_size    = p.tick_size
    if tick_size <= 0 or tick_value <= 0:
        lots = p.min_lot
    else:
        lots = risk_amount / (sl_dist / tick_size * tick_value)

    lots = _normalize_lots(lots, p)
    return lots, sl_price, tp_price, sl_mult, tp_mult, rr


def _normalize_lots(lots: float, p: AIRequest) -> float:
    step = p.lot_step if p.lot_step > 0 else 0.01
    lots = math.floor(lots / step) * step
    lots = max(p.min_lot, min(p.max_lot, lots))
    return round(lots, 8)


# ──────────────────────────────────────────────────────────────────────
#  MODAL ENDPOINT
# ──────────────────────────────────────────────────────────────────────

@app.function(
    image=image,
    volumes={MODEL_DIR: volume},
    secrets=[secrets] if secrets else [],
    timeout=30,
    memory=512,
    cpu=1.0,
    # Keep 1 warm container — eliminates cold start latency
    keep_warm=1,
    max_containers=5,
)
@modal.fastapi_endpoint(method="POST", label="apex-hydracrypto-predict")
async def predict(request: AIRequest) -> AIResponse:
    """
    Main AI prediction endpoint.
    Called by MT5 EA on every scan cycle for each symbol.
    """
    # 1. Feature Engineering
    features, feature_scores = build_features(request)

    # 2. Regime Classification
    regime_id, regime_name, regime_conf = classify_regime(request, features)

    # 3. Signal Generation (regime-conditioned)
    signal, confidence, reasoning = compute_signal(
        features, regime_id, regime_conf, request, feature_scores
    )

    # 4. Position Sizing + SL/TP
    lots, sl_price, tp_price, sl_mult, tp_mult, rr = compute_position(
        request, signal, confidence, regime_id
    )

    # 5. Final guard — block if not enough conviction
    MIN_CONFIDENCE = 0.52
    if confidence < MIN_CONFIDENCE:
        signal  = 0
        lots    = 0.0
        sl_price = tp_price = 0.0

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
    )


# ── Health check ──────────────────────────────────────────────────────
@app.function(image=image, timeout=10)
@modal.fastapi_endpoint(method="GET", label="apex-hydracrypto-health")
async def health():
    return {
        "status":  "ok",
        "version": MODEL_VERSION,
        "ts":      datetime.now(timezone.utc).isoformat(),
    }


# ── Local test ────────────────────────────────────────────────────────
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
            open=closes[:],  high=[c + 200 for c in closes],
            low=[c - 200 for c in closes], close=closes,
            volume=[random.uniform(100, 500) for _ in range(n)],
        ),
        atr=350, atr_avg=280, adx=32, plus_di=28, minus_di=18,
        rsi=58, macd=120, macd_signal=90, macd_hist=30,
        ema20=64800, ema50=63500, ema200=60000,
        htf_ema50=63000, htf_ema200=58000,
        tick_value=1.0, tick_size=0.01, min_lot=0.01,
        max_lot=100, lot_step=0.01, point=0.01, digits=2,
    )

    features, scores = build_features(payload)
    regime_id, regime_name, regime_conf = classify_regime(payload, features)
    signal, confidence, reasoning = compute_signal(features, regime_id, regime_conf, payload, scores)
    lots, sl, tp, sl_m, tp_m, rr = compute_position(payload, signal, confidence, regime_id)

    print(f"\n{'='*55}")
    print(f"  ApexHydra Modal AI — Local Test")
    print(f"{'='*55}")
    print(f"  Symbol:    BTCUSD")
    print(f"  Regime:    {regime_name} ({regime_conf:.1%})")
    print(f"  Signal:    {SIGNAL_NAMES[signal]} ({signal})")
    print(f"  Conf:      {confidence:.1%}")
    print(f"  Lots:      {lots}")
    print(f"  SL:        {sl:.2f}  ({sl_m}×ATR)")
    print(f"  TP:        {tp:.2f}  ({tp_m}×ATR)")
    print(f"  R:R:       {rr:.2f}")
    print(f"  Reasoning: {reasoning}")
    print(f"  Scores:    {scores}")
    print(f"{'='*55}\n")
