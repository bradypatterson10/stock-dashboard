import os
import time
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from datetime import datetime, timedelta

st.set_page_config(
    page_title="Sector Dashboard",
    page_icon="📈",
    layout="wide",
)

# === CONFIG ===
TICKERS = {
    "SPY": "S&P 500", "XLE": "Energy", "XLI": "Industrials",
    "XLF": "Financials", "XLK": "Technology", "XLC": "Comm Svcs",
    "XLB": "Materials", "XLRE": "Real Estate", "XLU": "Utilities",
    "GLD": "Gold ETF", "QQQ": "Nasdaq 100",
}
DETAIL_TICKERS = ["XLE", "XLI", "XLK", "GLD", "QQQ"]
LOOKBACK_DAYS = 90
RSI_PERIOD = 14
ZSCORE_WINDOW = 20


AV_BASE = "https://www.alphavantage.co/query"
# Free tier: 25 requests/day, 5 requests/minute.
# 11 tickers = 11 requests per full fetch. Stay under 5/min with a small delay.
AV_REQUEST_DELAY = 13  # seconds between ticker calls (safe under 5/min limit)


def _fetch_ticker_av(ticker: str, api_key: str, session: requests.Session) -> pd.Series:
    """Fetch daily closing prices for one ticker from Alpha Vantage."""
    resp = session.get(AV_BASE, params={
        "function": "TIME_SERIES_DAILY",
        "symbol": ticker,
        "outputsize": "compact",   # last ~100 trading days — enough for 90-day lookback
        "datatype": "json",
        "apikey": api_key,
    }, timeout=15)
    resp.raise_for_status()
    payload = resp.json()

    if "Time Series (Daily)" not in payload:
        # Surface rate-limit / bad key messages from AV
        note = payload.get("Note") or payload.get("Information") or payload
        print(f"[fetch_data] {ticker} — unexpected response: {note}")
        return pd.Series(dtype=float)

    series = pd.Series({
        pd.Timestamp(date): float(vals["4. close"])
        for date, vals in payload["Time Series (Daily)"].items()
    }).sort_index()
    return series


@st.cache_data(ttl=3600)
def fetch_data(as_of: str) -> pd.DataFrame:
    """Fetch 90 days of closing prices from Alpha Vantage TIME_SERIES_DAILY.
    `as_of` is a cache-bust key; the real data is always fresh on a cache miss."""
    api_key = os.environ.get("ALPHA_VANTAGE_KEY", "")
    if not api_key:
        return pd.DataFrame()   # caller checks and shows st.error()

    session = requests.Session()
    session.headers["User-Agent"] = "sector-dashboard/1.0"

    frames = {}
    tickers = list(TICKERS.keys())
    for i, ticker in enumerate(tickers):
        try:
            s = _fetch_ticker_av(ticker, api_key, session)
            if not s.empty:
                frames[ticker] = s
        except Exception as exc:
            print(f"[fetch_data] {ticker} failed: {exc}")

        # Rate-limit delay between calls (skip after last ticker)
        if i < len(tickers) - 1:
            time.sleep(AV_REQUEST_DELAY)

    if not frames:
        print("[fetch_data] ERROR: all tickers returned empty")
        return pd.DataFrame()

    cutoff = datetime.today() - timedelta(days=LOOKBACK_DAYS + 10)
    prices = pd.DataFrame(frames)
    prices = prices[prices.index >= pd.Timestamp(cutoff)].dropna(how="all").tail(LOOKBACK_DAYS)
    print(f"[fetch_data] shape={prices.shape}  last_date={prices.index[-1].date() if not prices.empty else 'N/A'}")
    return prices


def compute_rsi(series: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def compute_zscore(series: pd.Series, window: int = ZSCORE_WINDOW) -> pd.Series:
    roll_mean = series.rolling(window).mean()
    roll_std = series.rolling(window).std()
    return (series - roll_mean) / roll_std


def build_rs_df(prices: pd.DataFrame) -> pd.DataFrame:
    returns = prices.pct_change().dropna()
    scores = {}
    for ticker in TICKERS:
        if ticker == "SPY":
            continue
        cum_ticker = (1 + returns[ticker]).cumprod().iloc[-1]
        cum_spy = (1 + returns["SPY"]).cumprod().iloc[-1]
        scores[ticker] = round((cum_ticker / cum_spy - 1) * 100, 2)

    df = pd.DataFrame.from_dict(scores, orient="index", columns=["RS vs SPY (%)"])
    df["Sector"] = df.index.map(TICKERS)
    return df.sort_values("RS vs SPY (%)", ascending=False)


def build_detail_df(prices: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ticker in DETAIL_TICKERS:
        rsi_series = compute_rsi(prices[ticker])
        zscore_series = compute_zscore(prices[ticker])
        rows.append({
            "Ticker": ticker,
            "Name": TICKERS[ticker],
            "RSI (14)": round(rsi_series.iloc[-1], 1),
            "Z-Score (20d)": round(zscore_series.iloc[-1], 2),
        })
    return pd.DataFrame(rows).set_index("Ticker")


def make_bar_chart(rs_df: pd.DataFrame, end_date: datetime) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["#2ecc71" if x > 0 else "#e74c3c" for x in rs_df["RS vs SPY (%)"]]
    ax.barh(rs_df["Sector"], rs_df["RS vs SPY (%)"], color=colors)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_title(
        f"Sector Relative Strength vs SPY — 90-Day  |  {end_date.strftime('%Y-%m-%d')}",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlabel("Excess Return vs SPY (%)")
    plt.tight_layout()
    return fig


# ─── UI ────────────────────────────────────────────────────────────────────────

st.title("📈 Sector Relative Strength Dashboard")
st.caption("Data via Alpha Vantage · Cache refreshes every hour · End-of-day prices")

if not os.environ.get("ALPHA_VANTAGE_KEY"):
    st.error(
        "**ALPHA_VANTAGE_KEY environment variable is not set.**\n\n"
        "1. Get a free key at https://www.alphavantage.co/support/#api-key\n"
        "2. Run: `railway variables set ALPHA_VANTAGE_KEY=<your_key>`\n"
        "3. Redeploy or wait for Railway to restart the service."
    )
    st.stop()

# Cache key: reset when the user clicks the button, otherwise stay on the hour.
if "cache_key" not in st.session_state:
    st.session_state.cache_key = datetime.now().strftime("%Y-%m-%d-%H")

col_btn, col_ts = st.columns([1, 5])
with col_btn:
    if st.button("🔄 Refresh Data"):
        st.session_state.cache_key = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        st.cache_data.clear()

with st.spinner("Fetching market data…"):
    prices = fetch_data(st.session_state.cache_key)

if prices.empty:
    st.error(
        "Could not fetch market data from Alpha Vantage. "
        "This usually means the daily request quota (25/day) has been reached, "
        "or the API key is invalid. Check Railway logs for details. "
        "Data will be available again tomorrow, or upgrade to a paid AV plan."
    )
    st.stop()

end_date = prices.index[-1].to_pydatetime()

with col_ts:
    st.caption(f"Latest trading day in dataset: **{end_date.strftime('%A, %B %d %Y')}**")

st.divider()

# ─── Chart ─────────────────────────────────────────────────────────────────────

rs_df = build_rs_df(prices)
fig = make_bar_chart(rs_df, end_date)
st.pyplot(fig)
plt.close(fig)

st.divider()

# ─── RS Table ──────────────────────────────────────────────────────────────────

st.subheader("Relative Strength Table")
display_df = rs_df[["Sector", "RS vs SPY (%)"]].copy()
st.dataframe(
    display_df.style.bar(
        subset=["RS vs SPY (%)"],
        align="zero",
        color=["#e74c3c", "#2ecc71"],
    ),
    use_container_width=True,
)

st.divider()

# ─── RSI / Z-Score Detail ──────────────────────────────────────────────────────

st.subheader("RSI & Z-Score — Selected Tickers")
detail_df = build_detail_df(prices)

metric_cols = st.columns(len(DETAIL_TICKERS))
for col, ticker in zip(metric_cols, DETAIL_TICKERS):
    rsi = detail_df.loc[ticker, "RSI (14)"]
    zscore = detail_df.loc[ticker, "Z-Score (20d)"]
    rsi_delta = "Overbought" if rsi > 70 else ("Oversold" if rsi < 30 else "Neutral")
    col.metric(
        label=f"{ticker} — {TICKERS[ticker]}",
        value=f"RSI {rsi}",
        delta=f"Z {zscore:+.2f}  ·  {rsi_delta}",
    )

st.table(detail_df)
