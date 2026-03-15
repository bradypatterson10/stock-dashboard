# Sector Relative Strength Dashboard

A Streamlit app showing 90-day sector RS vs SPY, RSI, Z-Score, and Monte Carlo simulation for major US equity ETFs. Data via [Yahoo Finance](https://finance.yahoo.com) — no API key required.

## Local development

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Community Cloud

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app** → select your repo → set **Main file path** to `app.py` → click **Deploy**.

Share the resulting `*.streamlit.app` URL with anyone — no login required to view.

## Tickers covered

| Ticker | Name |
|--------|------|
| SPY | S&P 500 (benchmark) |
| XLE | Energy |
| XLI | Industrials |
| XLF | Financials |
| XLK | Technology |
| XLC | Comm Svcs |
| XLB | Materials |
| XLRE | Real Estate |
| XLU | Utilities |
| GLD | Gold ETF |
| QQQ | Nasdaq 100 |
