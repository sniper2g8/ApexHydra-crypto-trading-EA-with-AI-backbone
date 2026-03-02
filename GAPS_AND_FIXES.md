# ApexHydra Crypto — App Gap Analysis & Fix Outline

This document outlines what is lacking across the stack (MQ5 EA, Modal AI, Telegram bot, Streamlit dashboard, Supabase) and how to fix each item.

---

## 1. Capital allocation not applied to position sizing

**What’s lacking**  
- **Telegram** and **dashboard** let you set “allocated capital” or “trading capital” (and dashboard has “capital %”).  
- **Modal** uses `p.allocated_capital` for lot sizing when `> 0`, else `p.account_balance`.  
- **EA (MQ5)** never sends `allocated_capital` in the `/predict` payload; it only sends `account_balance` and `account_equity`.  
- So sizing in Modal always uses full account balance, and capital limits from Telegram/dashboard have no effect.

**How to fix**  
1. **EA**: In `PullConfig()`, read `allocated_capital` from the `ea_config` response (same way you read `risk_pct`, `max_dd_pct`, etc.). Store it in `SConfig` (e.g. `double allocated_capital`).  
2. **EA**: In `CallModalAI()`, add `allocated_capital` to the JSON body sent to Modal (e.g. `"allocated_capital":%.2f` using the value from config, or 0 if not set).  
3. **Optional**: If you want dashboard’s “capital %” to apply, either:  
   - Store `trading_capital` and `capital_pct` in DB and have EA send `allocated_capital = trading_capital * (capital_pct/100)` to Modal, or  
   - Standardize on one name (`allocated_capital`) and have dashboard write that instead of `trading_capital`/`capital_pct` so EA and Telegram stay in sync.

---

## 2. Supabase schema out of date

**What’s lacking**  
- **`supabase_schema.sql`** defines `ea_config` with only `allocated_capital` (no `trading_capital`, `capital_pct`, or `live_*`).  
- **EA** PATCHes `ea_config` with `live_balance`, `live_equity`, `live_dd_pct`, `live_pnl`, `live_trades`, `live_wins`, `live_losses`, `live_ts`.  
- **Dashboard** writes `trading_capital`, `capital_pct`, and reads `live_*`.  
- **Trades** table in schema has no `regime_broad`, `strategy_used`, `closed_at`, `ppo_signal`, `ppo_confidence`, `regime_id`; EA and app code send/expect these.

**How to fix**  
1. Add missing columns to **ea_config**:  
   - `live_balance`, `live_equity`, `live_dd_pct`, `live_pnl`, `live_trades`, `live_wins`, `live_losses`, `live_ts` (types that match EA: numeric + timestamp/text).  
   - If you keep dashboard’s “capital %” flow: add `trading_capital`, `capital_pct`; otherwise document that only `allocated_capital` is used.  
2. Add missing columns to **trades**:  
   - `regime_broad`, `strategy_used`, `regime_id`, `closed_at`, `ppo_signal`, `ppo_confidence` (and any other fields the EA/Modal actually send).  
3. Add missing columns to **regime_changes** if used:  
   - e.g. `regime_broad`, `strategy_used`, `regime_id` if the EA or dashboard write them.  
4. Provide a single **migration script** (e.g. `supabase_migrations/001_add_live_and_v5_columns.sql`) so existing DBs can be updated without recreating tables.

---

## 3. Config key inconsistency (allocated vs trading)

**What’s lacking**  
- **Telegram** and **schema** use `allocated_capital`.  
- **Dashboard** uses `trading_capital` and `capital_pct` and shows “Trading Capital” + “Usable %”.  
- Dashboard already has a fallback in UI (read `allocated_capital` if `trading_capital` missing), but writes only `trading_capital`/`capital_pct`, so EA/Telegram never see “allocated” updated from dashboard unless you align names.

**How to fix**  
- **Option A (simplest):** Dashboard writes only `allocated_capital` (and optionally drop `capital_pct` or map it: e.g. “Usable %” → store as a separate field and have EA compute `allocated_capital = balance * capital_pct/100` when you want “percent of balance”).  
- **Option B:** Keep both: add `trading_capital` and `capital_pct` to schema; EA pulls both and sends to Modal e.g. `allocated_capital = trading_capital * (capital_pct/100)` when `trading_capital > 0`, else use `allocated_capital` from Telegram.  
- Document in SETUP_GUIDE which source wins (e.g. “Dashboard capital = trading_capital × capital_pct; Telegram sets allocated_capital directly”).

---

## 4. EA: No retry / backoff on WebRequest failures

**What’s lacking**  
- `WebRequest` to Modal or Supabase can fail (timeout, network, 5xx).  
- EA only logs and continues; no retry or backoff.  
- Important for: first config pull on init, first performance sync, and critical PATCH (e.g. halt).

**How to fix**  
- Add a small helper, e.g. `WebRequestWithRetry(method, url, headers, timeout, req, res, res_hdr, max_attempts = 2)`.  
- On failure, sleep 1–2 seconds and retry once (or twice). Use only for idempotent or safe operations (GET config, PATCH live, POST event).  
- For `/predict`, one retry is acceptable; avoid retrying POST trade-log multiple times to prevent duplicate rows.

---

## 5. EA: Order of BE, half-TP lock, and trail

**What’s lacking**  
- In `ManagePositions()` you do: Break-even → Half-TP lock → Trailing stop.  
- If half-TP lock sets SL to midpoint, trailing stop can later move SL again (e.g. ATR trail). That’s by design, but if you want “once locked at half-TP, don’t trail beyond that”, you need a rule.

**How to fix**  
- **Option A:** Leave as is (trail can improve further after half-TP lock).  
- **Option B:** Once SL has been moved to “half-TP” (e.g. store a flag per position or compare current SL to midpoint), skip trailing or only allow trail that improves SL (e.g. for BUY only allow new SL if it’s higher than current and still below price).  
- **Option C:** Make order explicit: e.g. “only run trail if SL is not yet at half-TP” (so half-TP lock runs first and takes precedence; then trail only for positions that haven’t reached half-TP).  
- Document chosen behavior in EA comments or SETUP_GUIDE.

---

## 6. Modal: No rate limiting or circuit breaker

**What’s lacking**  
- Many symbols × scan every 30s = many requests to Modal.  
- If Modal is slow or returns 5xx, EA keeps hammering; no client-side backoff or “pause predictions for N seconds after 5xx”.

**How to fix**  
- In EA, maintain a simple “Modal cooldown” until next scan: e.g. if `WebRequest` to Modal returns 5xx or timeout, set `g_modal_cooldown_until = TimeCurrent() + 60` and skip calling Modal for any symbol until that time.  
- Optionally in Modal: add a minimal rate limit or health check (e.g. return 503 if overloaded) so EA can back off.

---

## 7. Telegram / Dashboard: No .env.example or secrets template

**What’s lacking**  
- Repo has no `.env.example` for the Telegram bot (only SETUP_GUIDE describes vars).  
- Dashboard uses `.streamlit/secrets.toml`; no `secrets.toml.example` in repo.

**How to fix**  
- Add **`.env.example`** with all Telegram vars (e.g. `TELEGRAM_BOT_TOKEN`, `TELEGRAM_ALLOWED_IDS`, `SUPABASE_URL`, `SUPABASE_KEY`, `DD_ALERT_PCT`, etc.) and placeholder values.  
- Add **`.streamlit/secrets.toml.example`** with keys for Supabase, Telegram, Modal URL, etc., and short comments.  
- Keep real secrets out of repo (e.g. `.env` and `secrets.toml` in `.gitignore`).

---

## 8. Error handling and observability

**What’s lacking**  
- EA: Failed `PositionModify` (e.g. half-TP lock) is logged but not sent to Supabase as an event.  
- No central “health” or “last error” that dashboard/Telegram can show (e.g. “Last Modal error: timeout at 12:34”).  
- Dashboard/Telegram: Supabase errors sometimes only in logs or generic message.

**How to fix**  
- EA: On critical failures (e.g. Modify failed after half-TP condition met), call `DBPost("events", BuildEventJSON("WARN", "SL lock failed: " + reason))` so they appear in event log and dashboard.  
- Optional: add a small “last_error” or “last_modal_error” field in `ea_config` that EA updates on Modal/WebRequest failure; dashboard and Telegram can show it.  
- Dashboard: On `update_cfg` or Supabase read failure, show a clear message (e.g. “Could not update config; check Supabase key and table”).

---

## 9. Half-TP lock: SELL midpoint formula

**What’s lacking**  
- For SELL, `target_sl = open + (tp - open) * 0.5`. For a short, `open > tp`, so `(tp - open)` is negative; `open + negative` = midpoint between open and tp, which is correct.  
- Code is correct; only ensure broker minimum distance and freeze level are respected (you already use `min_dist`).

**How to fix**  
- No code change needed. Optionally add a one-line comment: “For SELL, open>tp so midpoint is still open + 0.5*(tp-open).”

---

## 10. Documentation and version alignment

**What’s lacking**  
- SETUP_GUIDE mentions “apex-hydracrypto” and health endpoint; actual app may use `apexhydra-crypto` and `/health`.  
- Schema and SETUP_GUIDE don’t mention `ea_logs` or other tables if they were added later.  
- No short “CHANGELOG” or “VERSION” file so deployers know what’s in the current bundle.

**How to fix**  
- In SETUP_GUIDE, use the exact endpoint paths and app names (e.g. `/health`, label `apexhydra-crypto`).  
- Add a **CHANGELOG.md** or a “Version” section in SETUP_GUIDE (e.g. v4.2: half-TP lock, CopyBuffer checks, GetSymbolCurrencies 4-letter base).  
- In schema or a separate “Schema” section, list all tables and columns (including `ea_logs`, `live_*`, etc.) so migrations stay clear.

---

## 11. Testing and validation

**What’s lacking**  
- No automated tests for Modal (e.g. one prediction request with fixture bars), Telegram (e.g. mock Supabase), or dashboard (e.g. load + assert no crash).  
- No script to validate Supabase connectivity and required tables/columns from a PC (e.g. `python scripts/check_supabase.py`).

**How to fix**  
- Add a minimal **`scripts/check_supabase.py`**: connect with env URL/KEY, select one row from `ea_config`, `trades`, `performance`, and check for `live_balance` / `allocated_capital` if present; print OK or missing columns.  
- Optional: add a **`tests/test_modal_predict.py`** that sends one canned JSON to `/predict` and asserts 200 and required keys in the response.  
- Document in SETUP_GUIDE: “Run `python scripts/check_supabase.py` after deploying schema.”

---

## 12. Security and production hardening

**What’s lacking**  
- Schema comments show RLS (row-level security) and policies commented out.  
- EA uses anon key for Supabase; if RLS is enabled later, policies must allow EA to insert trades and PATCH ea_config.  
- No mention of rotating keys or using a dedicated “EA” key with minimal scope.

**How to fix**  
- When enabling RLS, add policies so the key used by the EA can: INSERT into `trades`, `regime_changes`, `performance`, `events`; SELECT and PATCH `ea_config` (e.g. by magic).  
- Document in SETUP_GUIDE: “For production, enable RLS and use a key with minimal policies; keep service_role only for Streamlit/backend.”  
- Optional: add a “Production checklist” (HTTPS, key rotation, TELEGRAM_ALLOWED_IDS, etc.).

---

## Priority summary

| Priority | Item | Effort | Impact |
|----------|------|--------|--------|
| High | 1. EA sends allocated_capital to Modal | Low | Capital limits actually apply |
| High | 2. Supabase schema: add live_*, trades/regime columns | Low | Stops silent failures or 4xx on PATCH/INSERT |
| Medium | 3. Unify config keys (allocated vs trading) | Low | Fewer support issues |
| Medium | 4. EA WebRequest retry for config/PATCH | Low | More robust on flaky network |
| Medium | 7. .env.example + secrets.toml.example | Low | Easier setup |
| Medium | 8. Log critical EA failures to events table | Low | Better visibility |
| Low | 5. Document or tweak BE vs half-TP vs trail order | Low | Clear behavior |
| Low | 6. Modal cooldown on 5xx in EA | Low | Avoid thundering herd |
| Low | 10. Docs and version alignment | Low | Fewer deploy mistakes |
| Low | 11. check_supabase script | Low | Faster debugging |
| Low | 12. RLS and production checklist | Medium | Security |

Implementing **1** and **2** (and aligning **3** with your chosen capital UX) will fix the main functional gaps; the rest improve robustness and operability.
