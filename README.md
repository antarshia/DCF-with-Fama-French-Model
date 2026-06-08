# DCF Valuation Model App

to run this, you can open vs code and run the following code
python -m streamlit run app.py
or python3 
depends on what you have in your computer. 
It will pop up automatically or give you two options (locally and network URL) both could works. 


This Streamlit app calculates a discounted cash-flow valuation for US-listed stocks from a ticker symbol. Enter a ticker, edit your assumptions, and the app estimates intrinsic value per share.

Use this as an education and stock-screening tool. It is not financial advice, and trading decisions should not rely on this model alone.

## Run

```bash
streamlit run app.py
```

## Inputs

- Ticker symbol, such as `AAPL`
- Revenue growth for years 1-5
- Revenue growth for years 6+
- Target free cash flow margin
- WACC
- Terminal growth
- Projection years
- Margin of safety

## Outputs

- Intrinsic value per share
- Current price comparison
- Buy-below price after margin of safety
- Projected revenue and free cash flow
- Single-stock valuation and quality metrics, including FCF yield, P/E, P/S, EV/EBITDA, ROIC, EVA, DuPont ROE, Altman Z-score, DDM, and residual income value
- WACC vs terminal growth sensitivity table
- Portfolio performance and risk analytics, including CAGR, volatility, Sharpe, Sortino, max drawdown, beta, Jensen alpha, Treynor, information ratio, VaR, CVaR, correlation, risk contribution, and efficient frontier
- Fama-French portfolio regression with factor exposures, alpha, and R-squared
- CSV and Excel exports

## Valuation Metrics

Open the `Valuation Metrics` tab after loading a ticker to see extra stock-selection formulas:

- Relative valuation: `P/E`, `Forward P/E`, `P/S`, `EV/EBITDA`, `P/FCF`
- Cash flow and earnings yield: `FCF yield`, `Earnings yield`
- Quality: `Revenue CAGR`, `FCF CAGR`, `FCF margin`, `ROIC`, `ROIC - WACC spread`, `EVA`, `ROE`, `DuPont ROE`
- Balance sheet and risk: `Debt / market cap`, `Net cash / market cap`, `Altman Z-score`, reported beta
- Extra valuation models: dividend discount model and residual income value when the needed inputs are available

## Portfolio Regression

Open the `Portfolio Regression` tab to run a monthly Fama-French regression for a basket of US-listed stocks.

- Enter tickers separated by commas, such as `AAPL, MSFT, NVDA`.
- Leave weights blank for an equal-weight portfolio, or enter matching weights like `40, 40, 20`.
- Choose the Fama-French 3-factor or 5-factor model.
- Choose `Monthly rebalance` to keep weights fixed each month, or `Buy and hold` to use starting weights and let the portfolio drift.
- Click `Run Regression` to see annualized alpha, R-squared, market beta, factor exposures, diagnostics, monthly regression data, and a CSV export.

## Portfolio Analytics

Open the `Portfolio Analytics` tab to evaluate a group of tickers against a benchmark such as `SPY`.

- Enter tickers and optional weights the same way as the regression tab.
- Choose the rebalancing method, benchmark ticker, risk-free rate, and date range.
- Click `Run Portfolio Analytics` to see performance metrics, drawdowns, correlation, risk contribution, efficient frontier, monthly returns, and a CSV export.

## Data

Financial statements are loaded from Yahoo Finance through `yfinance` when available, with SEC EDGAR company facts as a free fallback for US-listed companies. Market price data is loaded through `yfinance` when available, with Yahoo chart data and Stooq fallbacks for price history. Portfolio regression factors are loaded from Kenneth French's data library.

SEC requests use a default user agent. For a shared or production setup, set a more specific contact string:

```bash
export SEC_USER_AGENT="Your Name your.email@example.com"
```

The model is intended for education and stock screening, not as financial advice.
