
# forex_full_app_final.py
"""Forex Live Dashboard — Final Version
- Robust download via yfinance.Ticker.history
- NewsAPI integration (default key provided) + RSS fallback
- VADER sentiment mapping to currencies
- Composite signal (News + Tech + Vol) with SL/TP and position sizing
- Creative UI with CSS cards and summary widgets
- Defensive error handling and logs to fx_data/errors.log
"""

import os, time, traceback, logging, math
from datetime import datetime, timedelta
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import feedparser
import pytz
from nltk.sentiment.vader import SentimentIntensityAnalyzer

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download vader_lexicon if missing
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()
# ------------------ Configuration ------------------
DATA_DIR = os.path.join(os.getcwd(), "fx_data")
os.makedirs(DATA_DIR, exist_ok=True)
LOG_FILE = os.path.join(DATA_DIR, "errors.log")
SIGNALS_FILE = os.path.join(DATA_DIR, "signals.csv")

# Put your NewsAPI key here as default (you provided it). You may override via environment variable NEWSAPI_KEY.
DEFAULT_NEWSAPI_KEY = "1502cba32d134f4095aa03d4bd5bfe3c"
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", DEFAULT_NEWSAPI_KEY).strip()

TIMEZONE = os.getenv("TIMEZONE", "Asia/Kolkata")
IST = pytz.timezone(TIMEZONE)

DEFAULT_PAIRS = ["EURUSD=X","GBPUSD=X","USDJPY=X","AUDUSD=X","USDCAD=X","USDCHF=X","USDINR=X"]

# Logging
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

# ------------------ Utilities ------------------
def ist_now():
    return datetime.now(IST)

@st.cache_data(ttl=60)
def download_history(pair, period="60d", interval="1h"):
    """Use yf.Ticker.history for more consistent return types"""
    try:
        t = yf.Ticker(pair)
        df = t.history(period=period, interval=interval, actions=False, auto_adjust=False)
        if df is None or df.empty:
            return None
        # Ensure numeric
        for col in ["Open","High","Low","Close","Volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["Open","High","Low","Close"])
        if df.empty:
            return None
        return df
    except Exception as e:
        logging.exception("download_history failed for %s", pair)
        return None

def compute_indicators(df):
    # returns new df copy with EMA, RSI, MACD, ATR
    df = df.copy()
    try:
        df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
        df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()
        # RSI
        delta = df['Close'].diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        roll_up = up.ewm(alpha=1/14, adjust=False).mean()
        roll_down = down.ewm(alpha=1/14, adjust=False).mean()
        rs = roll_up / roll_down.replace(0,1e-10)
        df['RSI'] = 100 - (100/(1+rs))
        # MACD
        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema12 - ema26
        df['MACD_SIGNAL'] = df['MACD'].ewm(span=9, adjust=False).mean()
        # ATR (safe)
        tr1 = (df['High'] - df['Low']).abs()
        tr2 = (df['High'] - df['Close'].shift()).abs()
        tr3 = (df['Low'] - df['Close'].shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(14, min_periods=1).mean()
    except Exception as e:
        logging.exception("compute_indicators failed")
    return df

# ------------------ News & Sentiment ------------------
sia = SentimentIntensityAnalyzer()
RSS_FEEDS = [
    "https://feeds.reuters.com/reuters/businessNews",
    "https://feeds.reuters.com/reuters/marketsNews",
    "https://www.fxstreet.com/rss",
    "https://www.investing.com/rss/news_25.rss"
]
CURRENCY_KEYWORDS = {
    "USD":["fed","fomc","powell","usd","treasury","cpi","nonfarm","payroll","pce"],
    "EUR":["ecb","euro","lagarde","eurozone","germany","euro"],
    "GBP":["boe","bank of england","uk","gbp","britain"],
    "JPY":["boj","bank of japan","yen","japan","jpy"],
    "AUD":["rba","australia","aussie","aud"],
    "CAD":["boc","canada","cad","oil"],
    "CHF":["snb","swiss","switzerland","chf","safe haven"],
    "INR":["rbi","india","rupee","inr"]
}

def fetch_news_newsapi(q="forex OR currency", page_size=25):
    if not NEWSAPI_KEY:
        return []
    try:
        r = requests.get("https://newsapi.org/v2/everything", params={
            "q": q, "language":"en", "pageSize": page_size, "sortBy":"publishedAt", "apiKey": NEWSAPI_KEY
        }, timeout=10)
        j = r.json()
        items = []
        for a in j.get("articles", []):
            items.append({"title": a.get("title"), "desc": a.get("description") or "", "source": a.get("source",{}).get("name"), "url": a.get("url")})
        return items
    except Exception as e:
        logging.exception("NewsAPI fetch failed")
        return []

def fetch_news_rss(limit=25):
    items = []
    for feed in RSS_FEEDS:
        try:
            f = feedparser.parse(feed)
            for e in f.entries[: max(3, limit//len(RSS_FEEDS))]:
                items.append({"title": e.get("title"), "desc": e.get("summary") or "", "source": feed, "url": e.get("link")})
        except Exception:
            logging.exception("RSS fetch failed for %s", feed)
    # dedupe by title
    seen = set(); out = []
    for it in items:
        t = (it.get("title") or "").strip()
        if not t or t in seen: continue
        seen.add(t); out.append(it)
    return out[:limit]

@st.cache_data(ttl=120)
def fetch_headlines(prefer_newsapi=True, q="forex OR currency", limit=25):
    if prefer_newsapi and NEWSAPI_KEY:
        items = fetch_news_newsapi(q=q, page_size=limit)
        if items:
            return items
    # fallback
    return fetch_news_rss(limit=limit)

def analyze_headlines(headlines):
    per_currency = {k:0.0 for k in CURRENCY_KEYWORDS.keys()}
    detailed = []
    for h in headlines:
        text = ((h.get("title") or "") + " " + (h.get("desc") or "")).strip()
        if not text: continue
        try:
            score = float(sia.polarity_scores(text)["compound"])
        except Exception:
            score = 0.0
        detailed.append({"title": h.get("title"), "score": score, "source": h.get("source"), "url": h.get("url")})
        low = text.lower()
        for cur, kws in CURRENCY_KEYWORDS.items():
            for kw in kws:
                if kw in low:
                    per_currency[cur] += (1 if score>=0 else -1) * abs(score)
    return per_currency, detailed

def pair_news_bias(pair, per_currency_scores):
    p = pair.replace("=X", "").upper()
    if len(p) < 6:
        return 0.0
    base = p[:3]; quote = p[3:6]
    return float(per_currency_scores.get(base,0.0) - per_currency_scores.get(quote,0.0))

# ------------------ Signal logic (scalar-only) ------------------
def compute_composite(action, news_bias, atr):
    tech_score = 1.0 if action=="BUY" else -1.0 if action=="SELL" else 0.0
    vol_score = 0.5 if atr>0 else 0.0
    corr_score = 0.0
    composite = 0.30 * (news_bias/3.0) + 0.25*tech_score + 0.15*vol_score + 0.10*corr_score
    composite = max(-1.0, min(1.0, float(composite)))
    return composite

def generate_for_pair(pair, account_balance=100000):
    """Return result dict and list of headlines used"""
    try:
        df = download_history(pair, period="60d", interval="1h")
        if df is None or df.empty or len(df) < 3:
            return {"error":"no price data"}, []
        df = compute_indicators(df)
        last = df.iloc[-1]; prev = df.iloc[-2]
        last_close = float(last['Close']); prev_close = float(prev['Close'])
        # momentum-based action but with EMA confirmation
        action = "HOLD"
        if last_close > prev_close and last['Close'] >= last.get('EMA20', last_close):
            action = "BUY"
        elif last_close < prev_close and last['Close'] <= last.get('EMA20', last_close):
            action = "SELL"
        # ATR fallback
        atr = last.get('ATR', None)
        try:
            atr = float(atr) if (atr is not None and not pd.isna(atr)) else None
        except Exception:
            atr = None
        if not atr or atr <= 0:
            close_std = float(df['Close'].pct_change().std()) if len(df['Close'])>1 else 0.0
            atr = max(close_std * last_close, last_close * 0.0001)
        sl = tp1 = tp2 = None
        if action == "BUY":
            sl = last_close - 1.2*atr
            tp1 = last_close + 1.5*(last_close - sl)
            tp2 = last_close + 2.5*(last_close - sl)
        elif action == "SELL":
            sl = last_close + 1.2*atr
            tp1 = last_close - 1.5*(sl - last_close)
            tp2 = last_close - 2.5*(sl - last_close)
        # news & bias
        headlines = fetch_headlines(prefer_newsapi=True, q=f"{pair.split('=')[0]} OR forex", limit=25)
        per_cur, detailed = analyze_headlines(headlines)
        news_bias = pair_news_bias(pair, per_cur)
        composite = compute_composite(action, news_bias, atr)
        if composite >= 0.4:
            final = "STRONG BUY"
        elif composite >= 0.1:
            final = "BUY"
        elif composite <= -0.4:
            final = "STRONG SELL"
        elif composite <= -0.1:
            final = "SELL"
        else:
            final = "HOLD"
        conf = "Low"
        if abs(composite) >= 0.6: conf="High"
        elif abs(composite) >= 0.3: conf="Medium"
        # position sizing
        risk_pct = 0.005 if conf=="Medium" else 0.0075 if conf=="High" else 0.0025
        risk_amount = float(account_balance) * risk_pct
        position_units = None
        if sl is not None and abs(last_close - sl) > 0:
            position_units = risk_amount / abs(last_close - sl)
        result = {
            "pair": pair, "time": ist_now().strftime("%Y-%m-%d %H:%M:%S"),
            "price": round(last_close,6), "action": action, "final": final, "composite": round(composite,3),
            "news_bias": round(news_bias,3), "confidence": conf,
            "entry": round(last_close,6), "stop_loss": round(sl,6) if sl else None,
            "tp1": round(tp1,6) if tp1 else None, "tp2": round(tp2,6) if tp2 else None,
            "position_units": round(position_units,2) if position_units else None, "atr": round(atr,6)
        }
        # safe log
        try:
            df_log = pd.DataFrame([result])
            header = not os.path.exists(SIGNALS_FILE)
            df_log.to_csv(SIGNALS_FILE, mode='a', header=header, index=False)
        except Exception:
            logging.exception("failed to log signal")
        return result, detailed
    except Exception as e:
        logging.exception("generate_for_pair failed for %s", pair)
        return {"error": str(e)}, []

# ------------------ Streamlit UI (creative) ------------------
st.set_page_config(page_title="Forex Pro · Live Signals", layout="wide")
st.markdown("""
<style>
body { background: linear-gradient(90deg,#0f1724 0%, #071129 100%); color: #e6eef6; }
h1, h2, h3 { color: #ffffff; }
.card { background: linear-gradient(180deg,#071b2f,#022033); padding:14px; border-radius:12px; box-shadow:0 6px 18px rgba(0,0,0,0.6); }
.small { color:#9fb0c8; }
.pair-badge { display:inline-block; padding:6px 10px; border-radius:8px; background:#112b3c; color:#fff; margin-right:8px; }
</style>
""", unsafe_allow_html=True)

st.title("🔔 Forex Pro · Live Signals & Suggestions")
st.write("Realtime composite signals combining Technical + News + Volatility. Built for reliability.")

# Sidebar inputs
st.sidebar.header("Settings")
pairs = st.sidebar.multiselect("Pairs to analyze", DEFAULT_PAIRS, DEFAULT_PAIRS[:5])
if not pairs:
    st.sidebar.info("Select at least one pair to start.")
account_balance = st.sidebar.number_input("Account balance (base currency)", value=100000.0, step=1000.0)
refresh = st.sidebar.slider("Auto-refresh (seconds)", min_value=30, max_value=1800, value=300)
st.sidebar.markdown("NewsAPI key used: " + ("Yes" if NEWSAPI_KEY else "No (RSS fallback)"))
if st.sidebar.button("Force refresh now"):
    st.cache_data.clear()

# Header summary
col1, col2, col3 = st.columns([1,1,1])
col1.metric("Local Time", ist_now().strftime("%Y-%m-%d %H:%M:%S"))
col2.metric("Pairs", len(pairs))
col3.metric("Signals logged", len(pd.read_csv(SIGNALS_FILE)) if os.path.exists(SIGNALS_FILE) else 0)

st.markdown("---")
# Main panels
left, right = st.columns([2,1])
with left:
    st.header("Live Signals & Charts")
    for pair in pairs:
        try:
            result, detailed = generate_for_pair(pair, account_balance=account_balance)
            if isinstance(result, dict) and result.get("error"):
                st.error(f"Error for {pair}: {result.get('error')}")
                continue
            # display card
            st.markdown(f"<div class='card'><h3 style='margin:0'>{pair} — <span style='color:#a7f3d0'>{result['final']}</span></h3>", unsafe_allow_html=True)
            st.markdown(f"<div class='small'>Price: {result['price']} | ATR: {result['atr']} | Confidence: {result['confidence']}</div>", unsafe_allow_html=True)
            st.markdown(f"<p>Entry: <b>{result['entry']}</b>  &nbsp;  SL: <b>{result['stop_loss']}</b>  &nbsp;  TP1: <b>{result['tp1']}</b>  &nbsp; TP2: <b>{result['tp2']}</b></p>", unsafe_allow_html=True)
            st.markdown(f"<div class='small'>Composite: {result['composite']} | News bias: {result['news_bias']} | Units (≈): {result['position_units']}</div></div>", unsafe_allow_html=True)
            # show headlines
            if detailed:
                with st.expander("Headlines influencing this signal", expanded=False):
                    for h in detailed[:8]:
                        st.write(f"- ({h.get('score'):+.3f}) {h.get('title')}")
            # mini-chart
            df = download_history(pair, period="7d", interval="1h")
            if df is not None and not df.empty:
                import plotly.graph_objects as go
                fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'])])
                fig.update_layout(height=300, margin=dict(l=0,r=0,t=10,b=0), template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)
            st.markdown("---")
        except Exception as e:
            logging.exception("UI loop error for %s", pair)
            st.error(f"Unhandled error for {pair}: {e}")

with right:
    st.header("Market Headlines & Sentiment")
    headlines = fetch_headlines(prefer_newsapi=True, limit=40)
    per_cur, detailed_all = analyze_headlines(headlines)
    st.write("Top currency news scores (positive = favors base currency):")
    for k,v in per_cur.items():
        st.write(f"- {k}: {v:+.2f}")
    if detailed_all:
        st.write("Recent Headlines:")
        for h in detailed_all[:12]:
            st.write(f"- ({h.get('score'):+.3f}) {h.get('title')}")
    st.markdown("---")
    st.write("Signals log file: fx_data/signals.csv")
    if os.path.exists(SIGNALS_FILE):
        st.download_button("Download signals CSV", data=open(SIGNALS_FILE,"rb").read(), file_name="signals.csv", mime="text/csv")

st.markdown("---")
st.caption("Educational only. Not financial advice.")

# Auto-refresh
time.sleep(refresh)
st.experimental_rerun()
