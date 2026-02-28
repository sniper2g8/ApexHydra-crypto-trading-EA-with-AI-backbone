-- ══════════════════════════════════════════════════════════════════
--  ApexHydra Crypto v4.0 — Supabase Schema
--  Run in SQL Editor → New Query → Run All
-- ══════════════════════════════════════════════════════════════════

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ── TRADES ────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS trades (
    id           UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    symbol       TEXT NOT NULL,
    action       TEXT NOT NULL,        -- OPEN | CLOSE
    regime       TEXT,
    signal       INTEGER,              -- -2 to +2
    confidence   DECIMAL(6,4),
    ai_score     DECIMAL(8,4),
    lots         DECIMAL(10,4),
    price        DECIMAL(18,5),
    sl           DECIMAL(18,5),
    tp           DECIMAL(18,5),
    pnl          DECIMAL(14,2),
    won          BOOLEAN,
    magic        INTEGER,
    timestamp    TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX idx_trades_sym  ON trades(symbol);
CREATE INDEX idx_trades_ts   ON trades(timestamp DESC);
CREATE INDEX idx_trades_act  ON trades(action);

-- ── REGIME CHANGES ────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS regime_changes (
    id           UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    symbol       TEXT NOT NULL,
    regime       TEXT NOT NULL,
    confidence   DECIMAL(6,4),
    adx          DECIMAL(8,2),
    atr          DECIMAL(18,5),
    rsi          DECIMAL(8,2),
    ai_score     DECIMAL(8,4),
    timestamp    TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX idx_regime_sym ON regime_changes(symbol);
CREATE INDEX idx_regime_ts  ON regime_changes(timestamp DESC);

-- ── PERFORMANCE SNAPSHOTS ─────────────────────────────────────────
CREATE TABLE IF NOT EXISTS performance (
    id               UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    balance          DECIMAL(16,2),
    equity           DECIMAL(16,2),
    drawdown         DECIMAL(8,4),
    total_trades     INTEGER DEFAULT 0,
    wins             INTEGER DEFAULT 0,
    losses           INTEGER DEFAULT 0,
    total_pnl        DECIMAL(16,2),
    global_accuracy  DECIMAL(6,4),
    halted           BOOLEAN DEFAULT FALSE,
    final            BOOLEAN DEFAULT FALSE,
    timestamp        TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX idx_perf_ts ON performance(timestamp DESC);

-- ── EVENTS LOG ────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS events (
    id         UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    type       TEXT NOT NULL,          -- HALT|RESUME|OPEN|CLOSE|ERROR|INFO|DEINIT
    message    TEXT,
    timestamp  TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX idx_events_ts ON events(timestamp DESC);

-- ══════════════════════════════════════════════════════════════════
--  EA CONFIG TABLE — Streamlit writes here → EA reads every 30s
-- ══════════════════════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS ea_config (
    id              UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    magic           INTEGER DEFAULT 20250228,
    -- Risk parameters (Streamlit can update these)
    risk_pct        DECIMAL(5,2) DEFAULT 1.0,
    max_dd_pct      DECIMAL(5,2) DEFAULT 20.0,
    max_positions   INTEGER DEFAULT 3,
    min_confidence  DECIMAL(5,4) DEFAULT 0.55,
    -- Control flags (Streamlit remote control)
    halted          BOOLEAN DEFAULT FALSE,
    paused          BOOLEAN DEFAULT FALSE,
    -- Metadata
    updated_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_by      TEXT DEFAULT 'system'     -- 'streamlit' | 'ea' | 'system'
);

-- Insert default config row
INSERT INTO ea_config (magic, risk_pct, max_dd_pct, max_positions, min_confidence, halted, paused)
VALUES (20250228, 1.0, 20.0, 3, 0.55, false, false)
ON CONFLICT DO NOTHING;

-- Auto-update timestamp
CREATE OR REPLACE FUNCTION update_ea_config_ts()
RETURNS TRIGGER AS $$
BEGIN NEW.updated_at = NOW(); RETURN NEW; END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_ea_config_ts
BEFORE UPDATE ON ea_config
FOR EACH ROW EXECUTE FUNCTION update_ea_config_ts();

-- ══════════════════════════════════════════════════════════════════
--  VIEWS
-- ══════════════════════════════════════════════════════════════════

-- Per-symbol trade summary
CREATE OR REPLACE VIEW trade_summary AS
SELECT
    symbol,
    COUNT(*) FILTER (WHERE action='OPEN')              AS total_trades,
    COUNT(*) FILTER (WHERE won=TRUE)                   AS wins,
    COUNT(*) FILTER (WHERE won=FALSE)                  AS losses,
    ROUND(100.0*COUNT(*) FILTER (WHERE won=TRUE)/
          NULLIF(COUNT(*) FILTER (WHERE action='CLOSE'),0),2) AS win_rate_pct,
    ROUND(COALESCE(SUM(pnl),0),2)                      AS total_pnl,
    ROUND(AVG(ai_score),4)                             AS avg_ai_score,
    ROUND(AVG(confidence),4)                           AS avg_confidence,
    MAX(timestamp)                                     AS last_trade
FROM trades GROUP BY symbol ORDER BY total_pnl DESC;

-- Per-regime performance
CREATE OR REPLACE VIEW regime_stats AS
SELECT
    regime,
    COUNT(*) FILTER (WHERE action='CLOSE')             AS closed_trades,
    COUNT(*) FILTER (WHERE won=TRUE)                   AS wins,
    ROUND(100.0*COUNT(*) FILTER (WHERE won=TRUE)/
          NULLIF(COUNT(*) FILTER (WHERE action='CLOSE'),0),2) AS win_rate_pct,
    ROUND(COALESCE(SUM(pnl),0),2)                      AS total_pnl,
    ROUND(AVG(pnl),2)                                  AS avg_pnl,
    ROUND(AVG(confidence),4)                           AS avg_confidence
FROM trades
GROUP BY regime ORDER BY win_rate_pct DESC NULLS LAST;

-- Current regime per symbol
CREATE OR REPLACE VIEW current_regimes AS
SELECT DISTINCT ON (symbol)
    symbol, regime, confidence, adx, rsi, ai_score, timestamp
FROM regime_changes
ORDER BY symbol, timestamp DESC;

-- ══════════════════════════════════════════════════════════════════
--  ROW LEVEL SECURITY (enable when ready for production)
-- ══════════════════════════════════════════════════════════════════
-- ALTER TABLE trades          ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE regime_changes  ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE performance     ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE events          ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE ea_config       ENABLE ROW LEVEL SECURITY;
-- 
-- -- EA (anon key) can insert all tables, but only UPDATE ea_config fields it owns
-- CREATE POLICY "EA insert" ON trades         FOR INSERT WITH CHECK (true);
-- CREATE POLICY "EA insert" ON regime_changes FOR INSERT WITH CHECK (true);
-- CREATE POLICY "EA insert" ON performance    FOR INSERT WITH CHECK (true);
-- CREATE POLICY "EA insert" ON events         FOR INSERT WITH CHECK (true);
-- CREATE POLICY "EA read config"   ON ea_config FOR SELECT USING (true);
-- CREATE POLICY "EA update halt"   ON ea_config FOR UPDATE USING (true);
-- 
-- -- Streamlit (service role key) has full access
