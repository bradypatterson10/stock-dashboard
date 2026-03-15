# Sector Relative Strength Dashboard

A Streamlit app showing 90-day sector RS vs SPY, RSI, and Z-Score for major US equity ETFs. Data via [Tiingo](https://tiingo.com).

## Local development

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Add your Tiingo token
mkdir -p .streamlit
cp secrets.toml.example .streamlit/secrets.toml
# edit .streamlit/secrets.toml and replace "your-token-here"

streamlit run app.py
```

Get a free Tiingo token at [tiingo.com](https://tiingo.com) — the free tier allows 50 requests/hour, which comfortably covers 11 tickers per fetch.

## Deploy to Streamlit Community Cloud

1. Push this repo to GitHub (the `.streamlit/secrets.toml` file is gitignored — never commit it).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app** → select your repo → set **Main file path** to `app.py` → click **Deploy**.
4. Once deployed, open **App settings → Secrets** and paste:
   ```
   TIINGO_TOKEN = "your-token-here"
   ```
5. Click **Save** — the app will reboot and load live data.

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
