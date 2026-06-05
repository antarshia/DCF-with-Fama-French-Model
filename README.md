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
- WACC vs terminal growth sensitivity table
- CSV and Excel exports

## Data

Financial statements are loaded from Yahoo Finance through `yfinance` when available, with SEC EDGAR company facts as a free fallback for US-listed companies. Market price data is loaded through `yfinance` when available.

SEC requests use a default user agent. For a shared or production setup, set a more specific contact string:

```bash
export SEC_USER_AGENT="Your Name your.email@example.com"
```

The model is intended for education and stock screening, not as financial advice.
