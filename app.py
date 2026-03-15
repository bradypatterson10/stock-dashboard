import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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


@st.cache_data(ttl=3600)
def fetch_data(as_of: str) -> pd.DataFrame:
    """Download 90 days of closing prices. `as_of` busts cache when user clicks Refresh.
    Retries with end pushed 2 days forward to handle weekend/holiday boundaries."""
    end = datetime.today()
    start = end - timedelta(days=LOOKBACK_DAYS + 10)  # small buffer for weekends

    prices = yf.download(list(TICKERS.keys()), start=start, end=end, progress=False)["Close"]
    prices = prices.dropna(how="all")

    if prices.empty:
        # Weekend/holiday: yfinance sometimes needs end pushed past today to return data
        end_extended = end + timedelta(days=2)
        prices = yf.download(
            list(TICKERS.keys()), start=start, end=end_extended, progress=False
        )["Close"]
        prices = prices.dropna(how="all")

    return prices.tail(LOOKBACK_DAYS)


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
st.caption("Data via Yahoo Finance · Cache refreshes every hour · Prices 15-min delayed")

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
        "Could not fetch market data from Yahoo Finance. "
        "This can happen during extended outages or if the ticker list changed. "
        "Try clicking **Refresh Data** in a few minutes."
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
