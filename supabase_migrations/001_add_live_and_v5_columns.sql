-- ══════════════════════════════════════════════════════════════════
--  Migration 001: Add live_* and v5 columns (EA / Dashboard / Telegram)
--  Run in Supabase SQL Editor if your DB was created from an older schema.
--  Safe to run multiple times (IF NOT EXISTS).
-- ══════════════════════════════════════════════════════════════════

-- ea_config: live patch columns (EA PATCHes every tick)
ALTER TABLE ea_config ADD COLUMN IF NOT EXISTS live_balance   DECIMAL(16,2);
ALTER TABLE ea_config ADD COLUMN IF NOT EXISTS live_equity     DECIMAL(16,2);
ALTER TABLE ea_config ADD COLUMN IF NOT EXISTS live_dd_pct    DECIMAL(8,4);
ALTER TABLE ea_config ADD COLUMN IF NOT EXISTS live_pnl        DECIMAL(16,2);
ALTER TABLE ea_config ADD COLUMN IF NOT EXISTS live_trades     INTEGER;
ALTER TABLE ea_config ADD COLUMN IF NOT EXISTS live_wins       INTEGER;
ALTER TABLE ea_config ADD COLUMN IF NOT EXISTS live_losses     INTEGER;
ALTER TABLE ea_config ADD COLUMN IF NOT EXISTS live_ts         TEXT;
-- Dashboard capital fields (EA uses trading_capital * capital_pct/100 when allocated_capital not set)
ALTER TABLE ea_config ADD COLUMN IF NOT EXISTS trading_capital DECIMAL(16,2) DEFAULT 0.0;
ALTER TABLE ea_config ADD COLUMN IF NOT EXISTS capital_pct    DECIMAL(5,2) DEFAULT 100.0;

-- trades: v5 fields (regime_broad, strategy_used, closed_at, ppo_*, regime_id)
ALTER TABLE trades ADD COLUMN IF NOT EXISTS regime_broad   TEXT;
ALTER TABLE trades ADD COLUMN IF NOT EXISTS regime_id      INTEGER;
ALTER TABLE trades ADD COLUMN IF NOT EXISTS strategy_used   TEXT;
ALTER TABLE trades ADD COLUMN IF NOT EXISTS ppo_signal     TEXT;
ALTER TABLE trades ADD COLUMN IF NOT EXISTS ppo_confidence DECIMAL(6,4);
ALTER TABLE trades ADD COLUMN IF NOT EXISTS closed_at      TIMESTAMPTZ;

-- regime_changes: v5 fields
ALTER TABLE regime_changes ADD COLUMN IF NOT EXISTS regime_broad  TEXT;
ALTER TABLE regime_changes ADD COLUMN IF NOT EXISTS regime_id     INTEGER;
ALTER TABLE regime_changes ADD COLUMN IF NOT EXISTS strategy_used TEXT;
ALTER TABLE trades ADD COLUMN IF NOT EXISTS feature_vector TEXT;
