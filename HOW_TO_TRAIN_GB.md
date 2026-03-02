# How to Train the Gradient Boosting (GB) Model

The GB model improves decision accuracy by learning from your **closed trades**: it uses the same 38-dim feature vector that was present when each trade was opened, plus the outcome (PnL), to predict whether a future signal is more likely to win.

---

## 1. Automatic training (recommended)

GB is trained automatically by the **online_learner** cron job every **6 hours** on Modal.

**Requirements:**

- **Supabase:** `trades` table has a **`feature_vector`** column (run the migration below if needed).
- **EA:** Logs OPEN and CLOSE with **`feature_vector`** (the EA build that includes the feature_vector logging does this).
- **Data:** At least **20** closed trades in the last 30 days, and at least **15** closed trades **with** `feature_vector` per strategy (trend_following, mean_reversion, breakout).

After you deploy the updated Modal app and EA, new trades will store `feature_vector`. Once you have enough CLOSE rows with `feature_vector`, the next 6‑hour run of **online_learner** will train GB and save the model to the Modal volume. After that, `/predict` will use GB to confirm or reject rule-based signals (you’ll see `GB=BUY(0.6,src=gb_confirm)` etc. in the reasoning).

---

## 2. Run training manually (one-off)

To trigger training **now** without waiting for the 6‑hour schedule:

```bash
cd "d:\Web Apps\Trading files\ApexHydra crypto trading EA with AI backbone"
modal run apex_hydracrypto_modal.py::online_learner
```

Ensure:

- **Modal** is deployed and you’re logged in (`modal token new` if needed).
- **Secrets** `SUPABASE_URL` and `SUPABASE_SERVICE_KEY` are set in Modal (same as used by the app).

The run will:

- Read closed trades from Supabase (last 30 days).
- Filter to rows that have `feature_vector` (length ≥ 34).
- For each strategy with ≥15 such rows, train a GB classifier and save it under the Modal volume.
- Print something like: `[LEARNER] trend_following GB model saved (n=42, WR=58.5%)`.

If you see **"Only N trades have feature_vector"** or **"need 15+, skipping"**, you need more closed trades with `feature_vector` (see section 3).

---

## 3. Make sure `feature_vector` is stored (so GB can train)

For the learner to see data, the **EA** must send **`feature_vector`** when it logs trades, and **Supabase** must have a column for it.

**Supabase (one-time):**

- New installs: use the updated **`supabase_schema.sql`** (it already includes `feature_vector` on `trades`).
- Existing DB: run the migration that adds the column:

```sql
-- In Supabase SQL Editor (or use supabase_migrations/001_add_live_and_v5_columns.sql)
ALTER TABLE trades ADD COLUMN IF NOT EXISTS feature_vector TEXT;
```

**EA:**

- Use an EA build that:
  - Parses **`feature_vector`** from the Modal `/predict` response.
  - On **OPEN**: sends that vector in the trade log.
  - On **CLOSE**: sends the **stored** vector from the time of open (so the learner gets features at entry, not at exit).

The code changes for this are already in place: Modal returns **`feature_vector`**, and the EA stores it and includes it in **DBLogTrade** for both OPEN and CLOSE (using the snapshot at open for CLOSE).

---

## 4. Check that GB is trained and used

- **Modal logs:** When **online_learner** runs, look for lines like:
  - `[LEARNER] trend_following GB model saved (n=..., WR=...%)`
- **Predict response / EA logs:** In the reasoning string you should see:
  - `GB=BUY(0.6,src=gb_confirm)` or `GB=NONE(0%,src=none)` etc.
- **Health endpoint:** Call `GET /health` on your Modal app; the response includes **`gb_trained`** and **`gb_saved_at`** per strategy.

---

## 5. Summary

| Step | Action |
|------|--------|
| 1 | Run migration so `trades.feature_vector` exists (if not already). |
| 2 | Deploy Modal: `modal deploy apex_hydracrypto_modal.py`. |
| 3 | Use the EA that logs `feature_vector` on OPEN/CLOSE and let it run until you have many closed trades. |
| 4 | Wait for the next **online_learner** run (every 6 h) or run manually: `modal run apex_hydracrypto_modal.py::online_learner`. |
| 5 | After training, new `/predict` calls will use GB in fusion (fewer but more accurate signals when GB confirms). |

No separate “train GB” script is needed: training is done inside **online_learner**; you only need enough closed trades with **`feature_vector`** and then either wait for the cron or run **online_learner** once manually.
