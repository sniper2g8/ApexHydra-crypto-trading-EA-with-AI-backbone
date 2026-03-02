# ApexHydra Crypto v4.0 — Full Stack Setup Guide
## Modal AI · Streamlit Dashboard · Supabase DB · MT5 on VPS

---

## ARCHITECTURE OVERVIEW

```
┌─────────────────────────────────────────────────────────┐
│   YOUR VPS                                              │
│   MT5 + ApexHydra_Crypto_v4.mq5                        │
│   ├── Every 30s → POST market data to Modal AI         │
│   ├── Every 30s → Pull ea_config from Supabase         │
│   └── On events → POST to Supabase (trades/regimes)    │
└──────────────┬──────────────────────────────────────────┘
               │
    ┌──────────▼───────────┐    ┌───────────────────────┐
    │   MODAL.COM          │    │   SUPABASE            │
    │   apex_hydra_modal.py│    │   - trades            │
    │                      │    │   - regime_changes    │
    │  ① Regime Classify   │    │   - performance       │
    │  ② Signal Generate   │    │   - events            │
    │  ③ Kelly Lot Size    │    │   - ea_config ◄───────┼──┐
    │  ④ ATR SL/TP         │    └───────────┬───────────┘  │
    │                      │                │               │
    │  Returns JSON:        │    ┌───────────▼───────────┐  │
    │  { regime, signal,   │    │   STREAMLIT DASHBOARD │  │
    │    lots, sl, tp }    │    │   apex_hydra_dashboard│  │
    └──────────────────────┘    │                       │  │
                                │  📊 Equity curve      │  │
                                │  🎯 Regime heatmap    │  │
                                │  🤖 AI performance    │  │
                                │  📋 Trade history     │  │
                                │  🎛 Remote control ───┼──┘
                                └───────────────────────┘
```
Your new Modal URL after deploy will be:
https://YOUR_WORKSPACE--apex-hydracrypto-apex-hydracrypto-predict.modal.run

⚠️ Remember to update Inp_Modal_URL in the MT5 EA inputs and re-add the new URL to MT5's allowed WebRequest list.


🤖 How to Deploy the Telegram Bot
Option A — Run on your VPS (simplest, alongside MT5)
This is the recommended setup since the bot already shares the VPS with your EA.
1. Install dependencies
bashpip install "python-telegram-bot[job-queue]>=20.7" supabase python-dotenv
2. Create your bot with BotFather

Message @BotFather on Telegram → /newbot → follow prompts
Copy the token: 123456789:ABCdef...

3. Get your Telegram chat ID

Message @userinfobot on Telegram
It replies with your numeric ID e.g. 987654321

4. Create a `.env` file in the same folder as the bot script (copy from `.env.example` and fill in):
   - `TELEGRAM_BOT_TOKEN`, `TELEGRAM_ALLOWED_IDS`, `SUPABASE_URL`, `SUPABASE_KEY` (required)
   - Optional: `DD_ALERT_PCT`, `DD_CRITICAL_PCT`, `MONITOR_INTERVAL_S`
5. Test it runs
bashpython apex_hydra_telegram_bot.py
Then message your bot /start on Telegram.
6. Run it as a permanent background service
Create a systemd service so it auto-starts on reboot:
bashsudo nano /etc/systemd/system/apexhydra-bot.service
Paste this (adjust paths):
ini[Unit]
Description=ApexHydra Telegram Bot
After=network.target

[Service]
Type=simple
User=your_user
WorkingDirectory=/home/your_user/apexhydra
ExecStart=/usr/bin/python3 /home/your_user/apexhydra/apex_hydra_telegram_bot.py
Restart=always
RestartSec=10
EnvironmentFile=/home/your_user/apexhydra/.env

[Install]
WantedBy=multi-user.target
Then enable it:
bashsudo systemctl daemon-reload
sudo systemctl enable apexhydra-bot
sudo systemctl start apexhydra-bot
sudo systemctl status apexhydra-bot   # confirm it's running
```

---

### Option B — Deploy to Railway (free cloud hosting, zero VPS config)

1. Push your project to a GitHub repo (exclude `.env` — use `.gitignore`)
2. Go to [railway.app](https://railway.app) → New Project → Deploy from GitHub
3. Select your repo, set the start command to:
```
   python apex_hydra_telegram_bot.py

Add environment variables in Railway's dashboard (same keys as .env)
Railway keeps it running 24/7 on a free tier


Option C — Deploy to Fly.io (more robust, free tier)
bash# Install flyctl
curl -L https://fly.io/install.sh | sh

# Login
fly auth login

# In your project folder — create a minimal Dockerfile first:
cat > Dockerfile << 'EOF'
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install "python-telegram-bot[job-queue]>=20.7" supabase python-dotenv
COPY apex_hydra_telegram_bot.py .
CMD ["python", "apex_hydra_telegram_bot.py"]
EOF

# Launch app (follow prompts, pick the free plan)
fly launch

# Set secrets (instead of .env file)
fly secrets set TELEGRAM_BOT_TOKEN="..." TELEGRAM_ALLOWED_IDS="..." SUPABASE_URL="..." SUPABASE_KEY="..."

# Deploy
fly deploy

Recommendation: For your setup, Option A (VPS systemd service) is simplest — the bot, MT5 EA, and Streamlit dashboard all live on the same machine with no extra cloud cost.
---

## STEP 1 — SUPABASE SETUP (5 min)

1. Create free project at https://supabase.com
2. **SQL Editor** → New Query → Paste `supabase_schema.sql` → **Run**
   - If you already have an older schema, run `supabase_migrations/001_add_live_and_v5_columns.sql` instead (adds missing columns).
3. Go to **Project Settings → API**
   - Copy **Project URL**: `https://xxxxx.supabase.co`
   - Copy **anon / public key** (for MT5 EA)
   - Copy **service_role key** (for Streamlit — full access)

---

## STEP 2 — MODAL AI DEPLOYMENT (10 min)

```bash
# Install Modal
pip install modal

# Authenticate
modal token new    # Opens browser → log in

# Test locally first
modal run apex_hydracrypto_modal.py

# Deploy to cloud (gets a permanent HTTPS URL)
modal deploy apex_hydracrypto_modal.py
```

After deploy, Modal prints your endpoint URL:
```
✓ Created web endpoint: https://YOUR_WORKSPACE--apexhydra-crypto-apexhydra-predict.modal.run
```

**Copy this URL** — you'll paste it into the MT5 EA input `Inp_Modal_URL`.

### Keep containers warm (eliminate cold starts)
The server is already configured with `min_containers=1` which keeps
one warm container running 24/7. This costs ~$0.20–0.50/day on Modal.

### Add Supabase secrets to Modal (optional — for future logging from Modal)
```bash
modal secret create apexhydra-secrets \
  SUPABASE_URL="https://YOUR_PROJECT.supabase.co" \
  SUPABASE_KEY="YOUR_SERVICE_ROLE_KEY"
```

---

## STEP 3 — STREAMLIT DASHBOARD (5 min)

### Local run (on your PC or the VPS):
```bash
pip install -r requirements.txt

# Add your Supabase credentials (copy from .streamlit/secrets.toml.example)
mkdir -p .streamlit
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Edit .streamlit/secrets.toml: set SUPABASE_URL and SUPABASE_KEY

streamlit run apex_hydra_dashboard.py
```
Opens at http://localhost:8501

### Deploy to Streamlit Cloud (free, public URL):
1. Push your project to GitHub (exclude `secrets.toml` — it's in `.gitignore`)
2. Go to https://streamlit.io/cloud → New app → Select your repo
3. Add secrets in Streamlit Cloud settings:
   ```
   SUPABASE_URL = "https://xxx.supabase.co"
   SUPABASE_KEY = "your_key"
   ```
4. Your dashboard gets a public URL: `https://your-app.streamlit.app`

### Run on VPS alongside MT5:
```bash
# Install
pip install -r requirements.txt

# Run as background service
nohup streamlit run apex_hydra_dashboard.py --server.port 8501 &

# Or use screen
screen -S dashboard
streamlit run apex_hydra_dashboard.py
# Ctrl+A+D to detach
```

Access via `http://YOUR_VPS_IP:8501` (open port 8501 in firewall)

---

## STEP 4 — MT5 EA INSTALLATION (VPS)

1. Copy `ApexHydra_Crypto_v4.mq5` to:
   ```
   C:\Users\...\AppData\Roaming\MetaQuotes\Terminal\...\MQL5\Experts\ApexHydra\
   ```

2. Open **MetaEditor (F4)** → Open file → **Compile (F7)**
   - Must show 0 errors

3. **MT5 → Tools → Options → Expert Advisors**:
   - ✅ Allow automated trading
   - ✅ Allow WebRequest for listed URLs
   - Add these URLs:
     ```
     https://YOUR_WORKSPACE--apexhydra-crypto-apexhydra-predict.modal.run
     https://YOUR_PROJECT_ID.supabase.co
     ```

4. Attach EA to **any single chart** (e.g. BTCUSD H1)

5. Set inputs:
   ```
   Inp_Modal_URL  = "https://YOUR_MODAL_URL.modal.run"
   Inp_DB_URL     = "https://YOUR_PROJECT.supabase.co"
   Inp_DB_Key     = "YOUR_SUPABASE_ANON_KEY"
   Inp_DB_Enable  = true
   ```

---

## STEP 5 — REMOTE CONTROL VIA STREAMLIT

The Streamlit sidebar lets you control the EA running on your VPS
**without touching MT5**:

| Action | What happens |
|--------|-------------|
| ▶ Resume | Sets `ea_config.paused = false` → EA resumes in ≤30s |
| ⏸ Pause | Sets `ea_config.paused = true` → EA stops new trades |
| ⛔ Emergency Stop | Sets `ea_config.halted = true` → EA halts immediately |
| 💾 Apply Risk Settings | Updates risk_pct, max_dd, max_positions, min_confidence |

The EA polls `ea_config` every `Inp_Config_Sec` seconds (default 30).

---

## MODAL AI — WHAT IT DECIDES

For every symbol on every scan, Modal receives:
- 100 OHLCV bars
- All pre-calculated indicators (ATR, ADX, RSI, MACD, EMAs)
- Account context (balance, equity, open positions)
- Trade history (last 20 signals + outcomes for online learning)

And returns:

```json
{
  "regime_id":    0,
  "regime_name":  "Trend Bull",
  "regime_conf":  0.82,
  "signal":       2,
  "signal_name":  "Strong Buy",
  "confidence":   0.71,
  "lots":         0.05,
  "sl_price":     66200.0,
  "tp_price":     68500.0,
  "sl_atr_mult":  2.0,
  "tp_atr_mult":  4.375,
  "rr_ratio":     2.19,
  "feature_scores": { ... },
  "reasoning":    "Regime=Trend Bull | Score=+0.612 | Conf=71.4% | ADX=34.2 | EMA=BullStack"
}
```

The EA only executes if:
- `confidence >= min_confidence` (from Supabase config)
- `rr_ratio >= Inp_Min_RR`
- Open positions < `max_positions`
- Not halted/paused

---

## MONITORING CHECKLIST

| Check | Where |
|-------|-------|
| EA scanning? | MT5 Expert tab / Dashboard overlay |
| Modal healthy? | `GET /apexhydra-health` endpoint |
| DB writing? | Supabase Table Editor |
| Dashboard live? | http://VPS_IP:8501 |
| Regime changes? | Streamlit → Recent Regime Changes panel |
| Config syncing? | MT5 dashboard shows "Config X sec ago" |

---

## TROUBLESHOOTING

| Problem | Fix |
|---------|-----|
| Modal: cold start slow | `min_containers=1` already set — wait for warmup after first deploy |
| WebRequest 0 returned | Add Modal & Supabase URLs to MT5 allowed list |
| No trades opening | Check `min_confidence` — lower to 0.45 to test |
| Streamlit can't connect | Verify `SUPABASE_URL` and `SUPABASE_KEY` in secrets.toml |
| EA ignoring pause | Confirm `Inp_Config_Sec` is ≤60, check Supabase `ea_config` row exists |
| Modal returns 422 | Check JSON payload — usually a missing field |

---

## COSTS (Estimated)

| Service | Cost |
|---------|------|
| Supabase | Free (500MB, 2GB bandwidth) |
| Modal (1 warm container) | ~$6–15/month |
| Streamlit Cloud | Free |
| VPS (Windows MT5) | $15–40/month (depends on provider) |
| **Total** | **~$21–55/month** |
