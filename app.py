import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
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

@st.cache_data(ttl=86400)  # 24-hour TTL — 3-year history doesn't change meaningfully day to day
def fetch_mc_history(ticker: str) -> pd.Series:
    """Fetch ~3 years of adjusted daily closes for one ticker (Monte Carlo use only)."""
    hist = yf.Ticker(ticker).history(period="3y", auto_adjust=True)
    series = hist["Close"].copy()
    series.index = series.index.tz_localize(None)
    print(f"[fetch_mc_history] {ticker} — {len(series)} trading days returned")
    return series


@st.cache_data(ttl=3600)
def fetch_data(as_of: str) -> pd.DataFrame:
    """Fetch 90 days of adjusted closing prices via yfinance bulk download.
    `as_of` is a cache-bust key; data is always fresh on a cache miss."""
    end = datetime.today()
    start = end - timedelta(days=LOOKBACK_DAYS + 10)
    raw = yf.download(
        list(TICKERS.keys()), start=start, end=end,
        auto_adjust=True, progress=False,
    )
    prices = raw["Close"].dropna(how="all").tail(LOOKBACK_DAYS)
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
st.caption("Data via Yahoo Finance · Cache refreshes every hour · Adjusted end-of-day prices")

# ─── Sidebar: fetch counter ─────────────────────────────────────────────────────

if "api_calls" not in st.session_state:
    st.session_state.api_calls = 0

st.sidebar.metric("API calls this session", st.session_state.api_calls)
st.sidebar.caption("Counts yfinance fetch triggers (cache hits are free)")

# ─── Main data load ─────────────────────────────────────────────────────────────

# Cache key: reset on Refresh, otherwise keyed to the current hour.
if "cache_key" not in st.session_state:
    st.session_state.cache_key = datetime.now().strftime("%Y-%m-%d-%H")

col_btn, col_ts = st.columns([1, 5])
with col_btn:
    if st.button("🔄 Refresh Data"):
        st.session_state.cache_key = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        st.session_state.api_calls += 11  # one per ticker
        st.cache_data.clear()

with st.spinner("Fetching market data…"):
    prices = fetch_data(st.session_state.cache_key)

if prices.empty:
    st.error(
        "Could not fetch market data from Yahoo Finance. "
        "This can happen during market closures or yfinance outages. "
        "Try clicking **🔄 Refresh Data** in a few minutes."
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

st.divider()

# ─── Monte Carlo Simulation ────────────────────────────────────────────────────

st.subheader("Monte Carlo Simulation")

mc_col1, mc_col2, mc_col3 = st.columns(3)
with mc_col1:
    mc_ticker = st.selectbox(
        "Ticker",
        options=list(TICKERS.keys()),
        format_func=lambda t: f"{t} — {TICKERS[t]}",
        index=0,
    )
with mc_col2:
    mc_investment = st.number_input(
        "Starting investment ($)",
        min_value=100,
        max_value=10_000_000,
        value=10_000,
        step=100,
    )
with mc_col3:
    mc_days = st.slider(
        "Time horizon (trading days)",
        min_value=21,
        max_value=756,
        value=252,
        step=21,
        format="%d days",
    )

N_SIMS = 1000

if st.button("▶ Run Simulation"):
    st.session_state.mc_run_ticker = mc_ticker
    st.session_state.api_calls += 1  # one request for the selected ticker

if st.session_state.get("mc_run_ticker") != mc_ticker:
    # Ticker changed since last run — prompt user to re-run
    st.session_state.pop("mc_run_ticker", None)

if not st.session_state.get("mc_run_ticker"):
    st.info("Configure the parameters above and click **▶ Run Simulation** to fetch 3 years of history and generate paths. No API call is made until you click.")
else:
    with st.spinner(f"Fetching 3-year price history for {mc_ticker}…"):
        mc_price_series = fetch_mc_history(mc_ticker)

    mc_daily_returns = mc_price_series.pct_change().dropna()
    n_history = len(mc_daily_returns)

    if n_history < 500:
        st.warning(
            f"Only {n_history} trading days of history available for {mc_ticker} "
            f"(expected ~756 for 3 years). The simulation will run but estimates "
            f"may be less reliable due to limited historical data."
        )

    st.caption(f"Simulation based on **{n_history} trading days** of history (~3 years) · cached 24 hours")

    # Fit a Student's t-distribution to capture fat tails (crashes/rallies)
    nu, t_loc, t_scale = stats.t.fit(mc_daily_returns)

    current_price = mc_price_series.iloc[-1]
    shares = mc_investment / current_price

    rand_returns = stats.t.rvs(
        df=nu, loc=t_loc, scale=t_scale,
        size=(mc_days, N_SIMS),
        random_state=42,
    )
    price_paths = current_price * np.cumprod(1 + rand_returns, axis=0)
    portfolio_paths = shares * price_paths  # shape: (mc_days, N_SIMS)

    p10 = np.percentile(portfolio_paths, 10, axis=1)
    p50 = np.percentile(portfolio_paths, 50, axis=1)
    p90 = np.percentile(portfolio_paths, 90, axis=1)

    final_p10 = float(p10[-1])
    final_p50 = float(p50[-1])
    final_p90 = float(p90[-1])

    fig_mc, ax_mc = plt.subplots(figsize=(12, 5))
    ax_mc.plot(portfolio_paths, color="gray", alpha=0.04, linewidth=0.5)
    ax_mc.plot(p10, color="#e74c3c", linewidth=2, label="10th percentile (worst 10%)")
    ax_mc.plot(p50, color="#2980b9", linewidth=2, label="50th percentile (median)")
    ax_mc.plot(p90, color="#2ecc71", linewidth=2, label="90th percentile (best 10%)")
    ax_mc.axhline(mc_investment, color="black", linewidth=1, linestyle="--", label="Starting value")
    ax_mc.set_title(
        f"Monte Carlo Simulation — {mc_ticker} — {N_SIMS:,} paths over {mc_days} trading days",
        fontsize=13, fontweight="bold",
    )
    ax_mc.set_xlabel("Trading Days")
    ax_mc.set_ylabel("Portfolio Value ($)")
    ax_mc.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax_mc.legend()
    plt.tight_layout()
    st.pyplot(fig_mc)
    plt.close(fig_mc)

    with st.expander("What does this chart mean?", expanded=True):
        st.info(
            f"""
**What is Monte Carlo simulation?**
It runs {N_SIMS:,} hypothetical futures for {TICKERS[mc_ticker]} by randomly sampling daily price moves from 3 years of historical returns ({n_history} trading days). Each gray line is one possible outcome — not a prediction, just a range of plausible scenarios based on how this ETF has actually behaved.

**Why 3 years of history?**
Using 3 years instead of 90 days gives a much more reliable picture of the asset's typical behavior across different market conditions — bull runs, corrections, and sideways periods. 90 days might capture only one regime and skew the estimate.

**Why a fat-tailed distribution?**
Real markets crash and rally more violently than a standard bell curve predicts. This simulation uses a Student's t-distribution (fitted degrees of freedom: {nu:.1f}) which gives more weight to extreme outcomes, so big drawdowns and big gains are modeled more realistically.

**The three highlighted lines**
- 🟢 **Green (90th percentile):** Only 10% of simulated outcomes end higher than this — a strong bull run.
- 🔵 **Blue (50th percentile / median):** Half of all simulations finish above this, half below. The most likely single outcome.
- 🔴 **Red (10th percentile):** Only 10% of outcomes end lower than this — a significant bear scenario.

**Your numbers over {mc_days} trading days (~{mc_days//21} months)**

| Scenario | Ending value | Change |
|---|---|---|
| Median | **${final_p50:,.0f}** | {(final_p50/mc_investment - 1)*100:+.1f}% |
| Worst 10% | **${final_p10:,.0f}** | {(final_p10/mc_investment - 1)*100:+.1f}% |
| Best 10% | **${final_p90:,.0f}** | {(final_p90/mc_investment - 1)*100:+.1f}% |

⚠️ **Disclaimer:** These projections are based on {TICKERS[mc_ticker]}'s historical return distribution and are not a guarantee of future results. Past volatility does not predict future performance. This is not financial advice.
            """
        )
