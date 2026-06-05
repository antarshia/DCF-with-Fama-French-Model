# DCF-valuation-model-app-
A model that can generate intrinsic value per share. Just simply input the ticker and the expectation based on what you believe. It will generate the intrinsic value per share for you. However, based on your idea to trade not this. 


README.md


DCF model 3
README.md



DCF Valuation Streamlit App
This app calculates a discounted cash-flow valuation for US-listed stocks from a ticker symbol.

Run
streamlit run app.py
Inputs
Ticker symbol, such as AAPL
Revenue growth for years 1-5
Revenue growth for years 6+
Target free cash flow margin
WACC
Terminal growth
Projection years
Margin of safety
Outputs
Intrinsic value per share
Current price comparison
Buy-below price after margin of safety
Projected revenue and free cash flow
WACC vs terminal growth sensitivity table
CSV and Excel exports
Data
Financial statements are loaded from Yahoo Finance through yfinance when available, with SEC EDGAR company facts as a free fallback for US-listed companies. Market price data is loaded through yfinance when available.

SEC requests use a default user agent. For a shared or production setup, set a more specific contact string:

export SEC_USER_AGENT="Your Name your.email@example.com"
The model is intended for education and stock screening, not as financial advice.
