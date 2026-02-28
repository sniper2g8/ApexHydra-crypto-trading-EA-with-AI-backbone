//+------------------------------------------------------------------+
//|                                     ApexHydra_Crypto_v4.mq5      |
//|         AI-Powered Crypto EA — Modal AI + Supabase + Streamlit   |
//|                        Production Grade v4.1 — News Filter                      |
//+------------------------------------------------------------------+
#property copyright   "ApexHydra Trading Systems"
#property version     "4.10"
#property description "Multi-symbol crypto EA with Modal AI backbone"
#property strict

#include <Trade\Trade.mqh>
#include <Trade\PositionInfo.mqh>

//═══════════════════════════════════════════════════════════════════
//  INPUT GROUPS
//═══════════════════════════════════════════════════════════════════

input group "━━━━━ SYMBOL SCANNER ━━━━━"
input string        Inp_Symbols       = "BTCUSD,ETHUSD,SOLUSD,BNBUSD,XRPUSD,ADAUSD,AVAXUSD,DOTUSD";
input ENUM_TIMEFRAMES Inp_TF          = PERIOD_H1;
input ENUM_TIMEFRAMES Inp_HTF         = PERIOD_H4;
input int           Inp_ScanSec       = 30;         // Scan every N seconds

input group "━━━━━ MODAL AI ━━━━━"
input string        Inp_Modal_URL     = "https://YOUR_MODAL_WORKSPACE--apexhydra-crypto-apexhydra-predict.modal.run";
input int           Inp_AI_Timeout    = 15000;      // HTTP timeout ms
input int           Inp_AI_Lookback   = 100;        // Bars to send to AI
input bool          Inp_AI_Fallback   = true;       // Use local signal if AI fails

input group "━━━━━ NEWS FILTER ━━━━━"
input bool          Inp_News_Filter   = true;       // Block trades around high-impact news
input int           Inp_News_Buffer_Min = 15;       // Minutes to block before/after event
input bool          Inp_News_Log      = true;       // Log news blocks to Expert tab

input group "━━━━━ INDICATORS ━━━━━"
input int           Inp_EMA_Fast      = 20;
input int           Inp_EMA_Mid       = 50;
input int           Inp_EMA_Slow      = 200;
input int           Inp_RSI_Period    = 14;
input int           Inp_ADX_Period    = 14;
input int           Inp_ATR_Period    = 14;
input int           Inp_MACD_Fast     = 12;
input int           Inp_MACD_Slow     = 26;
input int           Inp_MACD_Signal   = 9;
input int           Inp_ATR_Avg_Win   = 30;

input group "━━━━━ RISK (defaults — overridden by Supabase config) ━━━━━"
input double        Inp_Risk_Pct      = 1.0;
input double        Inp_MaxDD_Pct     = 20.0;
input int           Inp_Max_Pos       = 3;
input double        Inp_Min_Conf      = 0.55;
input double        Inp_Min_RR        = 1.4;

input group "━━━━━ POSITION MANAGEMENT ━━━━━"
input bool          Inp_UseTrail      = true;
input double        Inp_Trail_ATR     = 1.5;
input bool          Inp_UseBE         = true;
input double        Inp_BE_RR         = 1.0;
input int           Inp_Magic         = 20250228;
input int           Inp_Slippage      = 30;

input group "━━━━━ SUPABASE ━━━━━"
input bool          Inp_DB_Enable     = true;
input string        Inp_DB_URL        = "https://YOUR_PROJECT_ID.supabase.co";
input string        Inp_DB_Key        = "YOUR_ANON_KEY";
input int           Inp_DB_SyncSec    = 15;         // Performance sync interval (seconds)
input int           Inp_Config_Sec    = 30;         // Config pull interval

input group "━━━━━ DASHBOARD (MT5 Overlay) ━━━━━"
input bool          Inp_Dash          = true;
input int           Inp_Dash_X        = 15;
input int           Inp_Dash_Y        = 40;

//═══════════════════════════════════════════════════════════════════
//  STRUCTURES
//═══════════════════════════════════════════════════════════════════

struct SSymbolData {
   string   symbol;
   // Indicators
   double   atr, atr_avg, adx, plus_di, minus_di;
   double   rsi, macd, macd_sig, macd_hist;
   double   ema20, ema50, ema200, htf_ema50, htf_ema200;
   double   close, high, low;
   // AI Decision (latest from Modal v5)
   int      regime_id;
   string   regime_name;     // granular: "Trend Bull", "Ranging" etc.
   string   regime_broad;    // coarse:   "TRENDING" | "RANGING" | "VOLATILE"
   double   regime_conf;
   string   strategy_used;   // trend_following | mean_reversion | breakout
   int      signal;          // -2,-1,0,1,2
   string   signal_name;
   double   confidence;
   string   ppo_signal;      // BUY | SELL | NONE  (raw PPO output)
   double   ppo_confidence;
   double   ai_lots;
   double   ai_sl;
   double   ai_tp;
   double   ai_rr;
   string   ai_reasoning;
   // Trade state
   bool     has_pos;
   ulong    pos_ticket;
   double   pos_open, pos_sl, pos_tp;
   datetime pos_time;
   // Performance
   int      wins, losses;
   double   pnl_total, pnl_today;
   // AI history for online learning
   int      hist_signals[20];
   double   hist_pnl[20];
   int      hist_regimes[20];
   int      hist_idx;
   // Timestamps
   datetime last_scan;
   bool     ai_ok;           // last Modal call succeeded?
};

// Runtime config (pulled from Supabase ea_config table)
struct SConfig {
   double   risk_pct;
   double   max_dd_pct;
   int      max_positions;
   double   min_confidence;
   bool     halted;
   bool     paused;
   datetime last_pull;
};

//═══════════════════════════════════════════════════════════════════
//  GLOBALS
//═══════════════════════════════════════════════════════════════════

CTrade       g_trade;
CPositionInfo g_pos;

SSymbolData  g_syms[];
int          g_sym_cnt   = 0;
SConfig      g_cfg;

double       g_balance_start = 0;
double       g_peak_equity   = 0;
double       g_dd_pct        = 0;

int          g_total_trades  = 0;
int          g_total_wins    = 0;
int          g_total_losses  = 0;
double       g_total_pnl     = 0;

datetime     g_last_db_sync  = 0;
double       g_last_live_eq  = 0;    // last equity pushed to live patch

string       g_log_lines[];
int          g_log_cnt       = 0;

const string DASH = "AH4_";

//═══════════════════════════════════════════════════════════════════
//  INIT / DEINIT
//═══════════════════════════════════════════════════════════════════

int OnInit() {
   EventSetTimer(Inp_ScanSec);
   g_trade.SetExpertMagicNumber(Inp_Magic);
   g_trade.SetDeviationInPoints(Inp_Slippage);
   // Filling type is set per-symbol in ExecuteTrade — see GetFilling()
   g_balance_start = AccountInfoDouble(ACCOUNT_BALANCE);
   g_peak_equity   = AccountInfoDouble(ACCOUNT_EQUITY);

   // Default config from inputs
   g_cfg.risk_pct       = Inp_Risk_Pct;
   g_cfg.max_dd_pct     = Inp_MaxDD_Pct;
   g_cfg.max_positions  = Inp_Max_Pos;
   g_cfg.min_confidence = Inp_Min_Conf;
   g_cfg.halted         = false;
   g_cfg.paused         = false;

   if(!ParseSymbols()) { Alert("No valid symbols!"); return INIT_FAILED; }

   // Whitelist reminder
   Print("ApexHydra v4: Add to MT5 allowed URLs:");
   Print("  Modal:    ", Inp_Modal_URL);
   Print("  Supabase: ", Inp_DB_URL);

   if(Inp_Dash) BuildDashboard();

   Log("ApexHydra_Crypto v4.0 started — " + IntegerToString(g_sym_cnt) + " symbols");
   Log("Modal AI: " + Inp_Modal_URL);

   // Pull config immediately
   PullConfig();
   ScanAll();

   // Write first performance snapshot immediately so dashboard/Telegram
   // show live balance & equity from the moment the EA starts
   if(Inp_DB_Enable) {
      DBSyncPerformance(false);
      g_last_db_sync = TimeCurrent();
      DBPost("events", BuildEventJSON("INFO", "EA started — initial performance snapshot written"));
   }
   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason) {
   EventKillTimer();
   if(Inp_Dash) ObjectsDeleteAll(0, DASH);
   DBPost("events", BuildEventJSON("DEINIT", "EA stopped. Reason: " + IntegerToString(reason)));
   DBSyncPerformance(true);
}

//═══════════════════════════════════════════════════════════════════
//  TICK & TIMER
//═══════════════════════════════════════════════════════════════════

void OnTick() {
   // Always track peak equity even when halted/paused
   double eq = AccountInfoDouble(ACCOUNT_EQUITY);
   if(eq > g_peak_equity) g_peak_equity = eq;
   g_dd_pct = (g_peak_equity > 0) ? (1.0 - eq / g_peak_equity) * 100.0 : 0;

   // Realtime live patch — fires on every tick when equity changes >= $0.01
   // Lightweight PATCH only (no INSERT) — keeps dashboard & Telegram realtime
   if(Inp_DB_Enable && MathAbs(eq - g_last_live_eq) >= 0.01) {
      DBPatchLive();
      g_last_live_eq = eq;
   }

   if(g_cfg.halted || g_cfg.paused) return;
   ManagePositions();
}


void OnTimer() {
   CheckRisk();
   if(TimeCurrent() - g_cfg.last_pull >= Inp_Config_Sec) PullConfig();
   if(!g_cfg.halted && !g_cfg.paused) ScanAll();
   SyncPositions();
   if(Inp_Dash) UpdateDashboard();
   if(Inp_DB_Enable && TimeCurrent() - g_last_db_sync >= Inp_DB_SyncSec) {
      DBSyncPerformance(false);
      g_last_db_sync = TimeCurrent();
   }
}

//═══════════════════════════════════════════════════════════════════
//  SYMBOL PARSING
//═══════════════════════════════════════════════════════════════════

bool ParseSymbols() {
   string parts[];
   int cnt = StringSplit(Inp_Symbols, ',', parts);
   ArrayResize(g_syms, cnt);
   g_sym_cnt = 0;
   for(int i = 0; i < cnt; i++) {
      StringTrimLeft(parts[i]); StringTrimRight(parts[i]); StringToUpper(parts[i]);
      if(!SymbolSelect(parts[i], true)) { Print("Skip: ", parts[i]); continue; }
      ZeroMemory(g_syms[g_sym_cnt]);
      g_syms[g_sym_cnt].symbol        = parts[i];
      g_syms[g_sym_cnt].regime_id     = 5;
      g_syms[g_sym_cnt].regime_name   = "Undefined";
      g_syms[g_sym_cnt].regime_broad  = "RANGING";
      g_syms[g_sym_cnt].strategy_used = "";
      g_syms[g_sym_cnt].signal        = 0;
      g_syms[g_sym_cnt].ppo_signal    = "NONE";
      g_syms[g_sym_cnt].ppo_confidence= 0;
      g_syms[g_sym_cnt].ai_ok         = false;
      g_sym_cnt++;
   }
   ArrayResize(g_syms, g_sym_cnt);
   return g_sym_cnt > 0;
}

//═══════════════════════════════════════════════════════════════════
//  MAIN SCAN
//═══════════════════════════════════════════════════════════════════

void ScanAll() {
   for(int i = 0; i < g_sym_cnt; i++) {
      if(!CollectIndicators(g_syms[i])) continue;
      CallModalAI(g_syms[i]);
      if(g_syms[i].signal != 0 && !g_syms[i].has_pos)
         ExecuteTrade(g_syms[i]);
      g_syms[i].last_scan = TimeCurrent();
   }
}

//═══════════════════════════════════════════════════════════════════
//  INDICATOR COLLECTION
//═══════════════════════════════════════════════════════════════════

bool CollectIndicators(SSymbolData &s) {
   string sym = s.symbol;
   if(iBars(sym, Inp_TF) < Inp_AI_Lookback + 10) return false;

   s.close = iClose(sym, Inp_TF, 1);
   s.high  = iHigh(sym, Inp_TF, 1);
   s.low   = iLow(sym, Inp_TF, 1);

   int h;
   double buf[];
   ArraySetAsSeries(buf, true);

   // ATR
   h = iATR(sym, Inp_TF, Inp_ATR_Period);
   CopyBuffer(h, 0, 1, 1, buf); s.atr = buf[0];
   double atrbuf[];
   ArraySetAsSeries(atrbuf, true);
   CopyBuffer(h, 0, 1, Inp_ATR_Avg_Win, atrbuf);
   double sum=0; for(int b=0;b<Inp_ATR_Avg_Win;b++) sum+=atrbuf[b];
   s.atr_avg = sum / Inp_ATR_Avg_Win;
   IndicatorRelease(h);

   // ADX
   h = iADX(sym, Inp_TF, Inp_ADX_Period);
   CopyBuffer(h, 0, 1, 1, buf); s.adx      = buf[0];
   CopyBuffer(h, 1, 1, 1, buf); s.plus_di  = buf[0];
   CopyBuffer(h, 2, 1, 1, buf); s.minus_di = buf[0];
   IndicatorRelease(h);

   // RSI
   h = iRSI(sym, Inp_TF, Inp_RSI_Period, PRICE_CLOSE);
   CopyBuffer(h, 0, 1, 1, buf); s.rsi = buf[0];
   IndicatorRelease(h);

   // MACD
   h = iMACD(sym, Inp_TF, Inp_MACD_Fast, Inp_MACD_Slow, Inp_MACD_Signal, PRICE_CLOSE);
   CopyBuffer(h, 0, 1, 1, buf); s.macd     = buf[0];
   CopyBuffer(h, 1, 1, 1, buf); s.macd_sig = buf[0];
   s.macd_hist = s.macd - s.macd_sig;
   IndicatorRelease(h);

   // EMAs
   h = iMA(sym,Inp_TF,Inp_EMA_Fast,0,MODE_EMA,PRICE_CLOSE);
   CopyBuffer(h,0,1,1,buf); s.ema20=buf[0]; IndicatorRelease(h);
   h = iMA(sym,Inp_TF,Inp_EMA_Mid,0,MODE_EMA,PRICE_CLOSE);
   CopyBuffer(h,0,1,1,buf); s.ema50=buf[0]; IndicatorRelease(h);
   h = iMA(sym,Inp_TF,Inp_EMA_Slow,0,MODE_EMA,PRICE_CLOSE);
   CopyBuffer(h,0,1,1,buf); s.ema200=buf[0]; IndicatorRelease(h);

   // HTF EMAs
   h = iMA(sym,Inp_HTF,Inp_EMA_Mid,0,MODE_EMA,PRICE_CLOSE);
   CopyBuffer(h,0,1,1,buf); s.htf_ema50=buf[0]; IndicatorRelease(h);
   h = iMA(sym,Inp_HTF,Inp_EMA_Slow,0,MODE_EMA,PRICE_CLOSE);
   CopyBuffer(h,0,1,1,buf); s.htf_ema200=buf[0]; IndicatorRelease(h);

   return true;
}


//═══════════════════════════════════════════════════════════════════
//  NEWS FILTER  (v4.1)
//  Uses MT5's built-in economic calendar to detect high-impact events.
//  Returns minutes to the nearest high-impact event for this symbol's
//  currency pair. Returns 999 if no event found within look-ahead window.
//═══════════════════════════════════════════════════════════════════

//  Map a symbol like "BTCUSD" → {"USD"} or "EURUSD" → {"EUR","USD"}
void GetSymbolCurrencies(const string sym, string &currencies[]) {
   // Crypto: only quote currency matters (USD)
   string base = StringSubstr(sym, 0, 3);
   string quote = StringSubstr(sym, 3, 3);
   // Common crypto bases — treat as non-fiat, only flag quote events
   string cryptoBases[] = {"BTC","ETH","SOL","BNB","XRP","ADA","DOT","AVA","LTC","DOGE"};
   bool isCrypto = false;
   for(int i = 0; i < ArraySize(cryptoBases); i++)
      if(base == cryptoBases[i]) { isCrypto = true; break; }

   if(isCrypto) {
      ArrayResize(currencies, 1);
      currencies[0] = quote;
   } else {
      ArrayResize(currencies, 2);
      currencies[0] = base;
      currencies[1] = quote;
   }
}

int GetNewsMinutesAway(const string sym) {
   if(!Inp_News_Filter) return 999;

   string currencies[];
   GetSymbolCurrencies(sym, currencies);
   if(ArraySize(currencies) == 0) return 999;

   datetime now    = TimeCurrent();
   datetime look_ahead = now + (Inp_News_Buffer_Min + 5) * 60;  // Look slightly past buffer
   datetime look_back  = now - Inp_News_Buffer_Min * 60;        // Catch events just fired

   MqlCalendarValue events[];
   int cnt = CalendarValueHistory(events, look_back, look_ahead, NULL, NULL);
   if(cnt <= 0) return 999;

   int min_dist = 999;
   for(int i = 0; i < cnt; i++) {
      // Only HIGH impact events
      MqlCalendarEvent ev_info;
      if(!CalendarEventById(events[i].event_id, ev_info)) continue;
      if(ev_info.importance != CALENDAR_IMPORTANCE_HIGH) continue;

      // Check if this event's currency matches our symbol
      MqlCalendarCountry country;
      if(!CalendarCountryById(ev_info.country_id, country)) continue;
      string ev_currency = country.currency;

      bool match = false;
      for(int c = 0; c < ArraySize(currencies); c++)
         if(currencies[c] == ev_currency) { match = true; break; }
      if(!match) continue;

      // Calculate distance in minutes
      int dist = (int)((events[i].time - now) / 60);
      int abs_dist = (int)MathAbs((double)dist);
      if(abs_dist < min_dist) min_dist = abs_dist;
   }
   return min_dist;
}

bool IsNewsBlocked(const string sym, int &mins_away_out) {
   mins_away_out = GetNewsMinutesAway(sym);
   bool blocked = (mins_away_out < Inp_News_Buffer_Min);
   if(blocked && Inp_News_Log)
      Log(sym + " NEWS BLOCK: high-impact event in " + IntegerToString(mins_away_out) + " min");
   return blocked;
}

//═══════════════════════════════════════════════════════════════════
//  MODAL AI CALL
//═══════════════════════════════════════════════════════════════════

void CallModalAI(SSymbolData &s) {
   string sym = s.symbol;

   // Check news filter (compute once, passed into payload)
   int news_mins = 999;
   IsNewsBlocked(sym, news_mins);   // Sets news_mins; logging handled inside

   // Build OHLCV arrays for payload
   int n = MathMin(Inp_AI_Lookback, iBars(sym, Inp_TF) - 2);

   string closes="", opens="", highs="", lows="", vols="";
   for(int b = 1; b <= n; b++) {
      string sep = (b < n) ? "," : "";
      closes += DoubleToString(iClose(sym,Inp_TF,b),_Digits) + sep;
      opens  += DoubleToString(iOpen(sym,Inp_TF,b),_Digits)  + sep;
      highs  += DoubleToString(iHigh(sym,Inp_TF,b),_Digits)  + sep;
      lows   += DoubleToString(iLow(sym,Inp_TF,b),_Digits)   + sep;
      vols   += DoubleToString((double)iTickVolume(sym,Inp_TF,b),0) + sep;
   }

   // Build history arrays
   int h_cnt = MathMin(s.hist_idx, 20);
   string h_sigs="[", h_pnls="[", h_regs="[";
   for(int i = 0; i < h_cnt; i++) {
      string sep = (i < h_cnt-1) ? "," : "";
      h_sigs += IntegerToString(s.hist_signals[i]) + sep;
      h_pnls += DoubleToString(s.hist_pnl[i],2)   + sep;
      h_regs += IntegerToString(s.hist_regimes[i]) + sep;
   }
   h_sigs+="]"; h_pnls+="]"; h_regs+="]";

   // JSON payload
   string body = StringFormat(
      "{"
      "\"symbol\":\"%s\",\"timeframe\":\"%s\",\"magic\":%d,"
      "\"timestamp\":\"%s\","
      "\"account_balance\":%.2f,\"account_equity\":%.2f,"
      "\"risk_pct\":%.2f,\"max_positions\":%d,\"open_positions\":%d,"
      "\"bars\":{\"open\":[%s],\"high\":[%s],\"low\":[%s],\"close\":[%s],\"volume\":[%s]},"
      "\"atr\":%.5f,\"atr_avg\":%.5f,\"adx\":%.2f,\"plus_di\":%.2f,\"minus_di\":%.2f,"
      "\"rsi\":%.2f,\"macd\":%.5f,\"macd_signal\":%.5f,\"macd_hist\":%.5f,"
      "\"ema20\":%.5f,\"ema50\":%.5f,\"ema200\":%.5f,"
      "\"htf_ema50\":%.5f,\"htf_ema200\":%.5f,"
      "\"tick_value\":%.5f,\"tick_size\":%.5f,"
      "\"min_lot\":%.2f,\"max_lot\":%.2f,\"lot_step\":%.2f,"
      "\"point\":%.5f,\"digits\":%d,"
      "\"recent_signals\":%s,\"recent_outcomes\":%s,\"recent_regimes\":%s,"
      "\"news_blackout\":%s,\"news_minutes_away\":%d,\"news_buffer_minutes\":%d"
      "}",
      sym, EnumToString(Inp_TF), Inp_Magic,
      TimeToString(TimeCurrent(), TIME_DATE|TIME_SECONDS),
      AccountInfoDouble(ACCOUNT_BALANCE), AccountInfoDouble(ACCOUNT_EQUITY),
      g_cfg.risk_pct, g_cfg.max_positions, CountOpenPos(),
      opens, highs, lows, closes, vols,
      s.atr, s.atr_avg, s.adx, s.plus_di, s.minus_di,
      s.rsi, s.macd, s.macd_sig, s.macd_hist,
      s.ema20, s.ema50, s.ema200,
      s.htf_ema50, s.htf_ema200,
      SymbolInfoDouble(sym, SYMBOL_TRADE_TICK_VALUE),
      SymbolInfoDouble(sym, SYMBOL_TRADE_TICK_SIZE),
      SymbolInfoDouble(sym, SYMBOL_VOLUME_MIN),
      SymbolInfoDouble(sym, SYMBOL_VOLUME_MAX),
      SymbolInfoDouble(sym, SYMBOL_VOLUME_STEP),
      SymbolInfoDouble(sym, SYMBOL_POINT),
      (int)SymbolInfoInteger(sym, SYMBOL_DIGITS),
      h_sigs, h_pnls, h_regs,
      (news_mins < Inp_News_Buffer_Min ? "true" : "false"),
      news_mins,
      Inp_News_Buffer_Min
   );

   // HTTP POST to Modal
   char req[], res[];
   StringToCharArray(body, req, 0, StringLen(body));
   string headers = "Content-Type: application/json\r\n";
   string res_headers;
   string predict_url = Inp_Modal_URL;
   // Ensure the URL points to the /predict route
   if(StringFind(predict_url, "/predict") < 0) {
      StringTrimRight(predict_url);
      if(StringGetCharacter(predict_url, StringLen(predict_url)-1) == '/')
         predict_url = StringSubstr(predict_url, 0, StringLen(predict_url)-1);
      predict_url += "/predict";
   }
   int code = WebRequest("POST", predict_url, headers, Inp_AI_Timeout, req, res, res_headers);

   if(code != 200) {
      Log(sym + " Modal AI error: HTTP " + IntegerToString(code));
      s.ai_ok = false;
      // Always reset signal on AI failure — stale signal from previous call must not persist
      s.signal = 0;
      s.confidence = 0;
      return;
   }

   // Parse JSON response — all Modal v5 fields
   string json = CharArrayToString(res);
   s.ai_ok         = true;
   s.regime_id     = (int)JSONGetInt(json, "regime_id");
   s.regime_name   = JSONGetStr(json,      "regime_name");
   s.regime_broad  = JSONGetStr(json,      "regime");        // TRENDING|RANGING|VOLATILE
   s.regime_conf   = JSONGetDbl(json,      "regime_conf");
   s.strategy_used = JSONGetStr(json,      "strategy_used"); // trend_following|mean_reversion|breakout
   s.signal        = (int)JSONGetInt(json, "signal");
   s.signal_name   = JSONGetStr(json,      "signal_name");
   s.confidence    = JSONGetDbl(json,      "confidence");
   s.ppo_signal    = JSONGetStr(json,      "ppo_signal");
   s.ppo_confidence= JSONGetDbl(json,      "ppo_confidence");
   s.ai_lots       = JSONGetDbl(json,      "lots");
   s.ai_sl         = JSONGetDbl(json,      "sl_price");
   s.ai_tp         = JSONGetDbl(json,      "tp_price");
   s.ai_rr         = JSONGetDbl(json,      "rr_ratio");
   s.ai_reasoning  = JSONGetStr(json,      "reasoning");

   // Apply minimum confidence filter from config
   if(s.confidence < g_cfg.min_confidence) s.signal = 0;

   // Log to Supabase (regime change only — trades logged separately)
   if(Inp_DB_Enable) DBLogRegime(s);
}

//═══════════════════════════════════════════════════════════════════
//  FILLING MODE AUTO-DETECT
//  Brokers advertise which filling modes they support per symbol.
//  IOC is rarely supported on crypto — FOK or RETURN is standard.
//  Using the wrong mode → error 10017 "trade disabled".
//═══════════════════════════════════════════════════════════════════

ENUM_ORDER_TYPE_FILLING GetFilling(const string sym) {
   // SYMBOL_FILLING_MODE returns a bitmask of supported filling types
   long mode = 0;
   SymbolInfoInteger(sym, SYMBOL_FILLING_MODE, mode);
   if((mode & SYMBOL_FILLING_FOK) != 0) return ORDER_FILLING_FOK;
   if((mode & SYMBOL_FILLING_IOC) != 0) return ORDER_FILLING_IOC;
   return ORDER_FILLING_RETURN;
}

//═══════════════════════════════════════════════════════════════════
//  TRADE EXECUTION
//═══════════════════════════════════════════════════════════════════

void ExecuteTrade(SSymbolData &s) {
   if(g_cfg.halted || g_cfg.paused) return;
   if(s.signal == 0) return;

   // News filter — check before executing
   int news_mins = 999;
   if(IsNewsBlocked(s.symbol, news_mins)) {
      // Still send signal=0 payload to Modal so server stays in sync,
      // but skip the actual order placement.
      return;
   }
   if(s.ai_lots <= 0 || s.ai_sl <= 0 || s.ai_tp <= 0) return;
   if(CountOpenPos() >= g_cfg.max_positions) return;
   if(s.ai_rr < Inp_Min_RR) {
      Log(s.symbol + " skip: R:R=" + DoubleToString(s.ai_rr,2) + " < " + DoubleToString(Inp_Min_RR,2));
      return;
   }

   bool is_buy = (s.signal > 0);
   double price = is_buy ? SymbolInfoDouble(s.symbol, SYMBOL_ASK)
                         : SymbolInfoDouble(s.symbol, SYMBOL_BID);

   // Set filling mode per-symbol — avoids error 10017 on brokers that
   // don't support IOC (most crypto brokers use FOK or RETURN).
   g_trade.SetTypeFilling(GetFilling(s.symbol));

   string cmt = "AH4|" + s.regime_name + "|" + IntegerToString(s.signal) +
                "|" + DoubleToString(s.confidence,2);

   bool sent = is_buy
      ? g_trade.Buy(s.ai_lots,  s.symbol, price, s.ai_sl, s.ai_tp, cmt)
      : g_trade.Sell(s.ai_lots, s.symbol, price, s.ai_sl, s.ai_tp, cmt);

   if(sent) {
      s.has_pos   = true;
      s.pos_ticket= g_trade.ResultOrder();
      s.pos_open  = price;
      s.pos_sl    = s.ai_sl;
      s.pos_tp    = s.ai_tp;
      s.pos_time  = TimeCurrent();
      g_total_trades++;

      Log("OPEN " + s.symbol + " " + (is_buy?"BUY":"SELL") +
          " " + DoubleToString(s.ai_lots,2) + " @ " + DoubleToString(price,_Digits) +
          " SL:" + DoubleToString(s.ai_sl,_Digits) +
          " TP:" + DoubleToString(s.ai_tp,_Digits) +
          " | " + s.regime_name + " | Conf:" + DoubleToString(s.confidence*100,1) + "%" +
          " | AI: " + s.ai_reasoning);

      if(Inp_DB_Enable) DBLogTrade(s, "OPEN", price, 0);
   } else {
      Log("FAILED " + s.symbol + " err:" + IntegerToString(g_trade.ResultRetcode()) +
          " [" + g_trade.ResultRetcodeDescription() + "]" +
          " filling:" + EnumToString(GetFilling(s.symbol)));
   }
}

//═══════════════════════════════════════════════════════════════════
//  POSITION MANAGEMENT
//═══════════════════════════════════════════════════════════════════

void ManagePositions() {
   for(int i = PositionsTotal()-1; i >= 0; i--) {
      if(!g_pos.SelectByIndex(i)) continue;
      if(g_pos.Magic() != Inp_Magic) continue;

      string sym   = g_pos.Symbol();
      int    si    = FindSym(sym);
      if(si < 0) continue;

      double price = g_pos.PriceCurrent();
      double open  = g_pos.PriceOpen();
      double sl    = g_pos.StopLoss();
      double tp    = g_pos.TakeProfit();
      double atr   = g_syms[si].atr;
      bool   buy   = (g_pos.PositionType() == POSITION_TYPE_BUY);

      // Break-even
      if(Inp_UseBE && atr > 0) {
         double risk  = MathAbs(open - sl);
         double prof  = MathAbs(price - open);
         if(risk > 0 && prof/risk >= Inp_BE_RR) {
            double pt  = SymbolInfoDouble(sym, SYMBOL_POINT) * 2;
            double nsl = buy ? open+pt : open-pt;
            if(buy && nsl > sl)   g_trade.PositionModify(g_pos.Ticket(), nsl, tp);
            if(!buy && nsl < sl)  g_trade.PositionModify(g_pos.Ticket(), nsl, tp);
         }
      }

      // Trailing stop
      if(Inp_UseTrail && atr > 0) {
         double td  = atr * Inp_Trail_ATR;
         double nsl = buy ? price-td : price+td;
         if(buy && nsl > sl)       g_trade.PositionModify(g_pos.Ticket(), nsl, tp);
         if(!buy && (sl==0||nsl<sl)) g_trade.PositionModify(g_pos.Ticket(), nsl, tp);
      }
   }
}

//═══════════════════════════════════════════════════════════════════
//  POSITION SYNC + ONLINE LEARNING
//═══════════════════════════════════════════════════════════════════

void SyncPositions() {
   for(int i = 0; i < g_sym_cnt; i++) {
      if(!g_syms[i].has_pos) continue;
      bool open = false;
      for(int p = 0; p < PositionsTotal(); p++) {
         if(g_pos.SelectByIndex(p) &&
            g_pos.Ticket() == g_syms[i].pos_ticket &&
            g_pos.Magic()  == Inp_Magic) { open=true; break; }
      }
      if(open) continue;

      // Trade closed — record outcome
      double pnl = GetClosedPnL(g_syms[i].pos_ticket);
      bool won   = pnl > 0;

      g_syms[i].has_pos = false;
      if(won) { g_syms[i].wins++; g_total_wins++; }
      else    { g_syms[i].losses++; g_total_losses++; }
      g_syms[i].pnl_total += pnl;
      g_syms[i].pnl_today += pnl;
      g_total_pnl         += pnl;

      // Update online learning history for next Modal call
      int idx = g_syms[i].hist_idx % 20;
      g_syms[i].hist_signals[idx] = g_syms[i].signal;
      g_syms[i].hist_pnl[idx]     = pnl;
      g_syms[i].hist_regimes[idx] = g_syms[i].regime_id;
      g_syms[i].hist_idx++;

      Log("CLOSED " + g_syms[i].symbol +
          " PnL:$" + DoubleToString(pnl,2) + " " + (won?"WIN":"LOSS") +
          " | Regime:" + g_syms[i].regime_name);

      if(Inp_DB_Enable) DBLogTrade(g_syms[i], "CLOSE", g_syms[i].pos_open, pnl);
   }
}

//═══════════════════════════════════════════════════════════════════
//  RISK / CONFIG
//═══════════════════════════════════════════════════════════════════

void CheckRisk() {
   // Peak equity + DD% already maintained in OnTick — read g_dd_pct directly

   if(!g_cfg.halted && g_dd_pct >= g_cfg.max_dd_pct) {
      g_cfg.halted = true;
      Log("⛔ HALTED — DD:" + DoubleToString(g_dd_pct,2) + "% >= " + DoubleToString(g_cfg.max_dd_pct,2)+"%");
      DBPost("events", BuildEventJSON("HALT","DD limit hit: "+DoubleToString(g_dd_pct,2)+"%"));
      PushHaltToSupabase(true);
   }
   if(g_cfg.halted && g_dd_pct < g_cfg.max_dd_pct * 0.7) {
      g_cfg.halted = false;
      Log("✅ RESUMED — DD recovered to " + DoubleToString(g_dd_pct,2)+"%");
      PushHaltToSupabase(false);
   }
}

void PullConfig() {
   if(!Inp_DB_Enable) return;
   string url     = Inp_DB_URL + "/rest/v1/ea_config?magic=eq." + IntegerToString(Inp_Magic) + "&limit=1";
   string headers = "apikey: " + Inp_DB_Key + "\r\nAuthorization: Bearer " + Inp_DB_Key + "\r\n";
   char   req[], res[];
   string res_hdr;
   int code = WebRequest("GET", url, headers, 5000, req, res, res_hdr);
   if(code != 200) { g_cfg.last_pull = TimeCurrent(); return; }

   string json = CharArrayToString(res);
   // Parse config values from JSON array response
   if(StringFind(json,"risk_pct") >= 0) {
      double v = JSONGetDbl(json,"risk_pct");     if(v > 0) g_cfg.risk_pct = v;
      v = JSONGetDbl(json,"max_dd_pct");          if(v > 0) g_cfg.max_dd_pct = v;
      int iv = (int)JSONGetInt(json,"max_positions"); if(iv > 0) g_cfg.max_positions = iv;
      v = JSONGetDbl(json,"min_confidence");      if(v > 0) g_cfg.min_confidence = v;
      // Remote halt/pause from Streamlit dashboard
      string halt_str = JSONGetStr(json,"halted");
      if(halt_str == "true")  g_cfg.halted = true;
      if(halt_str == "false" && g_dd_pct < g_cfg.max_dd_pct) g_cfg.halted = false;
      string pause_str = JSONGetStr(json,"paused");
      g_cfg.paused = (pause_str == "true");
   }
   g_cfg.last_pull = TimeCurrent();
}

void PushHaltToSupabase(bool halted) {
   if(!Inp_DB_Enable) return;
   string url     = Inp_DB_URL + "/rest/v1/ea_config?magic=eq." + IntegerToString(Inp_Magic);
   string headers = "Content-Type: application/json\r\n"
                  + "apikey: " + Inp_DB_Key + "\r\n"
                  + "Authorization: Bearer " + Inp_DB_Key + "\r\n"
                  + "Prefer: return=minimal\r\n";
   string body    = "{\"halted\":" + (halted?"true":"false") + "}";
   char   req[], res[];
   StringToCharArray(body, req, 0, StringLen(body));
   string res_hdr;
   WebRequest("PATCH", url, headers, 5000, req, res, res_hdr);
}

//═══════════════════════════════════════════════════════════════════
//  SUPABASE DB LAYER
//═══════════════════════════════════════════════════════════════════

bool DBPost(string table, string json_body) {
   if(!Inp_DB_Enable) return false;
   string url     = Inp_DB_URL + "/rest/v1/" + table;
   string headers = "Content-Type: application/json\r\n"
                  + "apikey: " + Inp_DB_Key + "\r\n"
                  + "Authorization: Bearer " + Inp_DB_Key + "\r\n"
                  + "Prefer: return=minimal\r\n";
   char req[], res[];
   StringToCharArray(json_body, req, 0, StringLen(json_body));
   string res_hdr;
   int code = WebRequest("POST", url, headers, 5000, req, res, res_hdr);
   return (code == 200 || code == 201 || code == 204);
}

void DBLogTrade(const SSymbolData &s, string action, double price, double pnl) {
   // `regime`       = s.regime_name  (e.g. "Trend Bull") — granular label
   // `regime_broad` = s.regime_broad (e.g. "TRENDING")   — strategy-level label
   // `ai_score`     = s.ai_rr        (R:R ratio from Modal — used as quality score)
   // `strategy_used`, `ppo_signal`, `ppo_confidence`, `regime_id` are new in v5
   string closed_at_str = (action == "CLOSE")
      ? "\"" + TimeToString(TimeCurrent(), TIME_DATE|TIME_SECONDS) + "\""
      : "null";

   string json = StringFormat(
      "{\"symbol\":\"%s\",\"action\":\"%s\","
      "\"regime\":\"%s\",\"regime_broad\":\"%s\",\"regime_id\":%d,"
      "\"strategy_used\":\"%s\","
      "\"signal\":%d,\"confidence\":%.4f,\"ai_score\":%.4f,"
      "\"ppo_signal\":\"%s\",\"ppo_confidence\":%.4f,"
      "\"lots\":%.4f,\"price\":%.5f,\"sl\":%.5f,\"tp\":%.5f,"
      "\"pnl\":%.2f,\"won\":%s,\"magic\":%d,"
      "\"closed_at\":%s,\"timestamp\":\"%s\"}",
      s.symbol, action,
      s.regime_name, s.regime_broad, s.regime_id,
      s.strategy_used,
      s.signal, s.confidence, s.ai_rr,
      s.ppo_signal, s.ppo_confidence,
      s.ai_lots, price, s.ai_sl, s.ai_tp,
      pnl,
      (pnl > 0 ? "true" : (pnl < 0 ? "false" : "null")),
      Inp_Magic,
      closed_at_str,
      TimeToString(TimeCurrent(), TIME_DATE|TIME_SECONDS)
   );
   DBPost("trades", json);
}

void DBLogRegime(const SSymbolData &s) {
   string json = StringFormat(
      "{\"symbol\":\"%s\","
      "\"regime\":\"%s\",\"regime_broad\":\"%s\",\"regime_id\":%d,"
      "\"strategy_used\":\"%s\","
      "\"confidence\":%.4f,\"adx\":%.2f,\"atr\":%.5f,\"rsi\":%.2f,"
      "\"ai_score\":%.4f,\"timestamp\":\"%s\"}",
      s.symbol,
      s.regime_name, s.regime_broad, s.regime_id,
      s.strategy_used,
      s.regime_conf, s.adx, s.atr, s.rsi,
      (double)s.signal * s.confidence,
      TimeToString(TimeCurrent(), TIME_DATE|TIME_SECONDS)
   );
   DBPost("regime_changes", json);
}

// ── Realtime live patch — called on every tick ──────────────────────
// Tiny PATCH to ea_config only. No INSERT. Keeps dashboard realtime.
void DBPatchLive() {
   if(!Inp_DB_Enable) return;
   double bal = AccountInfoDouble(ACCOUNT_BALANCE);
   double eq  = AccountInfoDouble(ACCOUNT_EQUITY);
   string ts  = TimeToString(TimeCurrent(), TIME_DATE|TIME_SECONDS);
   string patch = StringFormat(
      "{\"live_balance\":%.2f,\"live_equity\":%.2f,"
      "\"live_dd_pct\":%.4f,\"live_pnl\":%.2f,"
      "\"live_trades\":%d,\"live_wins\":%d,\"live_losses\":%d,"
      "\"live_ts\":\"%s\"}",
      bal, eq, g_dd_pct, g_total_pnl,
      g_total_trades, g_total_wins, g_total_losses, ts
   );
   string url     = Inp_DB_URL + "/rest/v1/ea_config?magic=eq." + IntegerToString(Inp_Magic);
   string headers = "Content-Type: application/json\r\n"
                  + "apikey: "         + Inp_DB_Key + "\r\n"
                  + "Authorization: Bearer " + Inp_DB_Key + "\r\n"
                  + "Prefer: return=minimal\r\n";
   char req[], res[];
   StringToCharArray(patch, req, 0, StringLen(patch));
   string res_hdr;
   WebRequest("PATCH", url, headers, 3000, req, res, res_hdr);
}

void DBSyncPerformance(bool final_snap) {
   double bal = AccountInfoDouble(ACCOUNT_BALANCE);
   double eq  = AccountInfoDouble(ACCOUNT_EQUITY);
   double acc = (g_total_trades > 0 ? (double)g_total_wins / g_total_trades : 0.5);
   string ts  = TimeToString(TimeCurrent(), TIME_DATE|TIME_SECONDS);

   // INSERT history row for equity curve chart (15s cadence)
   string json = StringFormat(
      "{\"balance\":%.2f,\"equity\":%.2f,\"drawdown\":%.4f,"
      "\"total_trades\":%d,\"wins\":%d,\"losses\":%d,"
      "\"total_pnl\":%.2f,\"global_accuracy\":%.4f,"
      "\"halted\":%s,\"final\":%s,\"timestamp\":\"%s\"}",
      bal, eq, g_dd_pct / 100.0,
      g_total_trades, g_total_wins, g_total_losses, g_total_pnl, acc,
      (g_cfg.halted ? "true" : "false"),
      (final_snap   ? "true" : "false"),
      ts
   );
   DBPost("performance", json);
   // Also update live patch so values are always fresh
   DBPatchLive();
   g_last_live_eq = eq;
}

string BuildEventJSON(string type, string msg) {
   return StringFormat("{\"type\":\"%s\",\"message\":\"%s\",\"timestamp\":\"%s\"}",
                       type, msg, TimeToString(TimeCurrent(),TIME_DATE|TIME_SECONDS));
}

//═══════════════════════════════════════════════════════════════════
//  LIGHTWEIGHT JSON PARSER
//═══════════════════════════════════════════════════════════════════

double JSONGetDbl(const string &json, const string &key) {
   string search = "\"" + key + "\":";
   int p = StringFind(json, search);
   if(p < 0) return 0;
   p += StringLen(search);
   // Skip whitespace
   while(p < StringLen(json) && StringGetCharacter(json,p)==' ') p++;
   // Skip quotes if present
   bool quoted = (StringGetCharacter(json,p) == '"');
   if(quoted) p++;
   string num = "";
   while(p < StringLen(json)) {
      ushort ch = StringGetCharacter(json,p);
      if(ch=='"' || ch==',' || ch=='}' || ch==']') break;
      num += ShortToString(ch);
      p++;
   }
   return StringToDouble(num);
}

long JSONGetInt(const string &json, const string &key) {
   return (long)JSONGetDbl(json, key);
}

string JSONGetStr(const string &json, const string &key) {
   string search = "\"" + key + "\":\"";
   int p = StringFind(json, search);
   if(p < 0) {
      // Try without quotes (bool values)
      search = "\"" + key + "\":";
      p = StringFind(json, search);
      if(p < 0) return "";
      p += StringLen(search);
      string val = "";
      while(p < StringLen(json)) {
         ushort ch = StringGetCharacter(json,p);
         if(ch==',' || ch=='}' || ch==']') break;
         val += ShortToString(ch);
         p++;
      }
      return val;
   }
   p += StringLen(search);
   string val = "";
   while(p < StringLen(json)) {
      ushort ch = StringGetCharacter(json,p);
      if(ch=='"') break;
      val += ShortToString(ch);
      p++;
   }
   return val;
}

//═══════════════════════════════════════════════════════════════════
//  UTILITIES
//═══════════════════════════════════════════════════════════════════

int CountOpenPos() {
   int c=0;
   for(int i=0;i<PositionsTotal();i++)
      if(g_pos.SelectByIndex(i) && g_pos.Magic()==Inp_Magic) c++;
   return c;
}

int FindSym(string sym) {
   for(int i=0;i<g_sym_cnt;i++) if(g_syms[i].symbol==sym) return i;
   return -1;
}

double GetClosedPnL(ulong ticket) {
   HistorySelect(TimeCurrent()-86400*7, TimeCurrent());
   double total_pnl = 0;
   for(int d=HistoryDealsTotal()-1; d>=0; d--) {
      ulong dt = HistoryDealGetTicket(d);
      if(HistoryDealGetInteger(dt, DEAL_POSITION_ID) == (long)ticket &&
         HistoryDealGetInteger(dt, DEAL_ENTRY) == DEAL_ENTRY_OUT)
         total_pnl += HistoryDealGetDouble(dt, DEAL_PROFIT)
                   +  HistoryDealGetDouble(dt, DEAL_SWAP)
                   +  HistoryDealGetDouble(dt, DEAL_COMMISSION);
   }
   return total_pnl;
}

void Log(string msg) {
   string entry = TimeToString(TimeCurrent(),TIME_SECONDS) + " | " + msg;
   Print("AH4: ", entry);
   if(g_log_cnt < 50) { ArrayResize(g_log_lines, g_log_cnt+1); g_log_lines[g_log_cnt]=entry; g_log_cnt++; }
   else { for(int i=1;i<50;i++) g_log_lines[i-1]=g_log_lines[i]; g_log_lines[49]=entry; }
}

//═══════════════════════════════════════════════════════════════════
//  DASHBOARD (MT5 Overlay)
//═══════════════════════════════════════════════════════════════════

void BuildDashboard() { UpdateDashboard(); }

void UpdateDashboard() {
   if(!Inp_Dash) return;
   int x=Inp_Dash_X+10, y=Inp_Dash_Y+8, lh=18;
   color CYAN=C'0,188,212', GREEN=C'0,230,118', RED=C'255,82,82', GOLD=C'255,193,7', GRAY=C'120,130,150';

   // Background
   if(ObjectFind(0,DASH+"BG")<0) ObjectCreate(0,DASH+"BG",OBJ_RECTANGLE_LABEL,0,0,0);
   ObjectSetInteger(0,DASH+"BG",OBJPROP_XDISTANCE,Inp_Dash_X);
   ObjectSetInteger(0,DASH+"BG",OBJPROP_YDISTANCE,Inp_Dash_Y);
   ObjectSetInteger(0,DASH+"BG",OBJPROP_XSIZE,510);
   ObjectSetInteger(0,DASH+"BG",OBJPROP_YSIZE,80+g_sym_cnt*20+140);
   ObjectSetInteger(0,DASH+"BG",OBJPROP_BGCOLOR,C'13,17,23');
   ObjectSetInteger(0,DASH+"BG",OBJPROP_BACK,false);
   ObjectSetInteger(0,DASH+"BG",OBJPROP_SELECTABLE,false);

   Lbl(DASH+"T1","⚡ APEXHYDRA v4.1  [Modal AI + Supabase + News Filter]",x,y,10,CYAN,true); y+=lh+2;

   double bal=AccountInfoDouble(ACCOUNT_BALANCE), eq=AccountInfoDouble(ACCOUNT_EQUITY);
   double wr=(g_total_trades>0?(double)g_total_wins/g_total_trades*100:0);
   color ddClr = (g_dd_pct>g_cfg.max_dd_pct*0.7)?RED:(g_dd_pct>g_cfg.max_dd_pct*0.4)?GOLD:GREEN;

   Lbl(DASH+"A1",StringFormat("Bal:$%.0f  Eq:$%.0f  Float:%+.0f  DD:%.1f%%  Status:%s",
       bal,eq,eq-bal,g_dd_pct,(g_cfg.halted?"HALTED":(g_cfg.paused?"PAUSED":"ACTIVE"))),
       x,y,9,(g_cfg.halted?RED:(g_cfg.paused?GOLD:GREEN)),false); y+=lh;

   Lbl(DASH+"A2",StringFormat("Trades:%d  W:%d  L:%d  WR:%.1f%%  PnL:%+.2f  Pos:%d/%d",
       g_total_trades,g_total_wins,g_total_losses,wr,g_total_pnl,CountOpenPos(),g_cfg.max_positions),
       x,y,9,C'200,210,220',false); y+=lh;

   Lbl(DASH+"A3",StringFormat("Risk:%.1f%%  MaxDD:%.0f%%  MinConf:%.0f%%  AI Sync:%.0fs ago",
       g_cfg.risk_pct,g_cfg.max_dd_pct,g_cfg.min_confidence*100,
       (double)(TimeCurrent()-g_cfg.last_pull)),
       x,y,9,GRAY,false); y+=lh;

   Lbl(DASH+"SEP1","─────────────────────────────────────────────────────────────────────",x,y,7,C'40,50,70',false); y+=12;
   Lbl(DASH+"HDR",StringFormat("%-10s %-12s %-10s %5s %6s %7s %6s %7s %8s",
       "Symbol","Regime","Strategy","ADX","RSI","Signal","Conf%","Lots","Today"),
       x,y,9,CYAN,false); y+=lh;

   for(int i=0;i<g_sym_cnt;i++) {
      SSymbolData s=g_syms[i];
      string sigStr=(s.signal==2?"STRBUY":(s.signal==1?"BUY":(s.signal==-1?"SELL":(s.signal==-2?"STRSEL":"WAIT"))));
      // Shorten strategy name for display
      string strat = (s.strategy_used=="trend_following"?"TF":(s.strategy_used=="mean_reversion"?"MR":(s.strategy_used=="breakout"?"BO":"--")));
      color rc=(s.signal>0?GREEN:(s.signal<0?RED:(s.has_pos?GOLD:GRAY)));
      Lbl(DASH+"SY"+IntegerToString(i),StringFormat("%-10s %-12s %-10s %5.1f %6.1f %7s %5.1f%% %7.2f %+8.2f%s",
          s.symbol, s.regime_name, strat, s.adx, s.rsi, sigStr,
          s.confidence*100, s.ai_lots, s.pnl_today,
          s.has_pos?"  ●":""),
          x,y,9,rc,false); y+=lh;
   }

   Lbl(DASH+"SEP2","─────────────────────────────────────────────────────────────────────",x,y,7,C'40,50,70',false); y+=12;
   Lbl(DASH+"LH","LOG:",x,y,9,CYAN,false); y+=lh-2;
   int ls=MathMax(0,g_log_cnt-4);
   for(int l=ls;l<g_log_cnt;l++) {
      string ln=g_log_lines[l];
      if(StringLen(ln)>82) ln=StringSubstr(ln,0,79)+"...";
      color lc=(StringFind(ln,"OPEN")>=0||StringFind(ln,"WIN")>=0)?GREEN:(StringFind(ln,"LOSS")>=0||StringFind(ln,"HALT")>=0?RED:GRAY);
      Lbl(DASH+"L"+IntegerToString(l-ls),ln,x,y,8,lc,false); y+=16;
   }
   ChartRedraw(0);
}

void Lbl(string name,string text,int x,int y,int sz,color clr,bool bold) {
   if(ObjectFind(0,name)<0) ObjectCreate(0,name,OBJ_LABEL,0,0,0);
   ObjectSetString(0,name,OBJPROP_TEXT,text);
   ObjectSetInteger(0,name,OBJPROP_XDISTANCE,x);
   ObjectSetInteger(0,name,OBJPROP_YDISTANCE,y);
   ObjectSetInteger(0,name,OBJPROP_FONTSIZE,sz);
   ObjectSetInteger(0,name,OBJPROP_COLOR,clr);
   ObjectSetInteger(0,name,OBJPROP_CORNER,CORNER_LEFT_UPPER);
   ObjectSetString(0,name,OBJPROP_FONT,bold?"Consolas Bold":"Consolas");
   ObjectSetInteger(0,name,OBJPROP_SELECTABLE,false);
   ObjectSetInteger(0,name,OBJPROP_BACK,false);
}
//+------------------------------------------------------------------+