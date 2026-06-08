from __future__ import annotations

# Run locally with: python -m streamlit run app.py
# Streamlit will open a local URL automatically, or print local/network URLs in the terminal.
from datetime import date

import pandas as pd
import plotly.express as px
import streamlit as st

from dcf_model import (
    DCFAssumptions,
    DCFError,
    FactorRegressionError,
    PortfolioAnalyticsError,
    analyze_fama_french_portfolio,
    analyze_portfolio,
    calculate_dcf,
    calculate_stock_metrics,
    default_assumptions,
    export_workbook,
    factor_regression_csv,
    fetch_ticker_data,
    is_probably_us_listed,
    portfolio_analytics_csv,
    sensitivity_table,
    summary_csv,
)


st.set_page_config(
    page_title="DCF Valuation",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# Hide Streamlit's default app chrome so the page reads like a focused valuation tool.
st.markdown(
    """
    <style>
    :root {
        --accent: #176b64;
        --ink: #14211f;
        --muted: #65726f;
        --line: #dde6e3;
        --surface: #ffffff;
        --band: #f6f8f7;
    }
    .stApp {
        background: var(--band);
        color: var(--ink);
    }
    [data-testid="stHeader"], .stAppHeader {
        display: none !important;
        height: 0 !important;
    }
    [data-testid="stToolbar"], [data-testid="stDecoration"], #MainMenu, footer {
        display: none !important;
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1280px;
    }
    h1, h2, h3 {
        letter-spacing: 0;
        color: var(--ink);
    }
    h1 {
        font-size: 2.1rem;
        line-height: 1.1;
        margin-bottom: .35rem;
    }
    h2 {
        font-size: 1.25rem;
        margin-top: .5rem;
    }
    div[data-testid="stMetric"] {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 1rem;
    }
    div[data-testid="stMetricValue"] {
        color: var(--ink);
        font-size: 1.55rem;
    }
    div[data-testid="stMetricDelta"] {
        font-size: .9rem;
    }
    .stTextInput input, .stNumberInput input {
        background-color: #ffffff !important;
        color: var(--ink) !important;
        border: 1px solid var(--line) !important;
        border-radius: 6px;
    }
    .stButton > button, .stDownloadButton > button {
        border-radius: 6px;
        border: 1px solid var(--accent) !important;
        background-color: var(--accent) !important;
        color: #ffffff !important;
        font-weight: 600;
        min-height: 2.55rem;
    }
    .stDownloadButton > button {
        background-color: #ffffff !important;
        color: var(--accent) !important;
    }
    section[data-testid="stSidebar"] {
        background: #ffffff;
    }
    div[data-testid="stDataFrame"] {
        border: 1px solid var(--line);
        border-radius: 8px;
        overflow: hidden;
    }
    .small-note {
        color: var(--muted);
        font-size: .88rem;
        line-height: 1.45;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=3600, show_spinner=False)
def load_ticker(ticker: str):
    return fetch_ticker_data(ticker)


# Cache network-heavy model calls; most inputs are stable enough for a one-hour app session.
@st.cache_data(ttl=3600, show_spinner=False)
def load_factor_regression(
    tickers: tuple[str, ...],
    weights: tuple[float, ...] | None,
    start_date: date,
    end_date: date,
    model_name: str,
    rebalancing_method: str,
):
    return analyze_fama_french_portfolio(tickers, weights, start_date, end_date, model_name, rebalancing_method)


@st.cache_data(ttl=3600, show_spinner=False)
def load_portfolio_analytics(
    tickers: tuple[str, ...],
    weights: tuple[float, ...] | None,
    start_date: date,
    end_date: date,
    rebalancing_method: str,
    benchmark_ticker: str,
    risk_free_rate: float,
):
    return analyze_portfolio(
        tickers,
        weights,
        start_date,
        end_date,
        rebalancing_method,
        benchmark_ticker,
        risk_free_rate,
    )


def money(value: float | None, currency: str = "USD") -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{currency} {value:,.2f}"


def compact_money(value: float | None, currency: str = "USD") -> str:
    if value is None or pd.isna(value):
        return "N/A"
    abs_value = abs(value)
    if abs_value >= 1_000_000_000_000:
        return f"{currency} {value / 1_000_000_000_000:,.2f}T"
    if abs_value >= 1_000_000_000:
        return f"{currency} {value / 1_000_000_000:,.2f}B"
    if abs_value >= 1_000_000:
        return f"{currency} {value / 1_000_000:,.2f}M"
    return money(value, currency)


def pct(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:.1%}"


def parse_tickers(value: str) -> tuple[str, ...]:
    return tuple(part.strip().upper() for part in value.replace("\n", ",").split(",") if part.strip())


def parse_weights(value: str) -> tuple[float, ...] | None:
    cleaned = value.strip()
    if not cleaned:
        return None
    return tuple(float(part.strip().replace("%", "")) for part in cleaned.replace("\n", ",").split(",") if part.strip())


def diagnostic_value(diagnostics: pd.DataFrame, metric: str):
    row = diagnostics[diagnostics["Metric"].eq(metric)]
    if row.empty:
        return None
    return row.iloc[0]["Value"]


def metric_value(metrics: pd.DataFrame, metric: str):
    row = metrics[metrics["Metric"].eq(metric)]
    if row.empty:
        return None
    return row.iloc[0]["Value"]


def format_metric_value(value: float | None, value_type: str, currency: str = "USD") -> str:
    if value is None or pd.isna(value):
        return "N/A"
    if value_type == "currency":
        return compact_money(value, currency)
    if value_type == "percent":
        return pct(value)
    if value_type == "multiple":
        return f"{value:.1f}x"
    if value_type == "number":
        return f"{value:.2f}"
    return str(value)


def style_metric_table(df: pd.DataFrame, currency: str = "USD") -> pd.DataFrame:
    # Backend metrics carry raw values and value types; the UI formats them at render time.
    display = df.copy()
    display["Value"] = display.apply(lambda row: format_metric_value(row["Value"], row["Type"], currency), axis=1)
    return display.drop(columns=["Type"])


def style_percent_frame(df: pd.DataFrame) -> pd.DataFrame:
    display = df.copy()
    for column in display.columns:
        display[column] = display[column].map(lambda value: pct(value) if pd.notna(value) else "N/A")
    return display


def style_factor_coefficients(df: pd.DataFrame) -> pd.DataFrame:
    display = df.copy()
    display["Coefficient"] = display.apply(
        lambda row: pct(row["Coefficient"]) if row["Factor"] == "Alpha" else f"{row['Coefficient']:.2f}",
        axis=1,
    )
    display["Annualized alpha"] = display["Annualized alpha"].map(lambda value: pct(value) if pd.notna(value) else "")
    display["t-stat"] = display["t-stat"].map(lambda value: f"{value:.2f}")
    display["p-value"] = display["p-value"].map(lambda value: f"{value:.3f}")
    return display


def style_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    percent_metrics = {"Monthly alpha", "Annualized alpha", "R-squared", "Adjusted R-squared"}
    display = df.copy()
    display["Value"] = display.apply(
        lambda row: pct(row["Value"]) if row["Metric"] in percent_metrics and isinstance(row["Value"], float) else row["Value"],
        axis=1,
    )
    return display


def style_weight_table(starting_weights: pd.Series, ending_weights: pd.Series, include_ending: bool) -> pd.DataFrame:
    display = starting_weights.rename("Starting weight").rename_axis("Ticker").reset_index()
    if include_ending:
        ending = ending_weights.rename("Ending weight").rename_axis("Ticker").reset_index()
        display = display.merge(ending, on="Ticker", how="left")
    for column in display.columns:
        if column.endswith("weight"):
            display[column] = display[column].map(pct)
    return display


def style_projection(df: pd.DataFrame, currency: str):
    display = df.copy()
    for column in ("revenue", "free_cash_flow", "present_value_fcf"):
        display[column] = display[column].map(lambda value: compact_money(value, currency))
    display["growth_rate"] = display["growth_rate"].map(pct)
    display["target_fcf_margin"] = display["target_fcf_margin"].map(pct)
    display["discount_factor"] = display["discount_factor"].map(lambda value: f"{value:.3f}")
    return display.rename(
        columns={
            "year": "Year",
            "growth_rate": "Growth",
            "revenue": "Revenue",
            "target_fcf_margin": "FCF margin",
            "free_cash_flow": "Free cash flow",
            "discount_factor": "Discount factor",
            "present_value_fcf": "PV of FCF",
        }
    )


def style_history(df: pd.DataFrame, currency: str):
    display = df.copy()
    for column in ("revenue", "operating_cash_flow", "capex", "free_cash_flow"):
        if column in display:
            display[column] = display[column].map(lambda value: compact_money(value, currency))
    if "fcf_margin" in display:
        display["fcf_margin"] = display["fcf_margin"].map(pct)
    return display.rename(
        columns={
            "fiscal_year": "Fiscal year",
            "revenue": "Revenue",
            "operating_cash_flow": "Operating cash flow",
            "capex": "CapEx",
            "free_cash_flow": "Free cash flow",
            "fcf_margin": "FCF margin",
        }
    )


def main() -> None:
    st.title("DCF Valuation")

    if "active_ticker" not in st.session_state:
        st.session_state.active_ticker = "AAPL"

    query_ticker = st.query_params.get("ticker")
    if query_ticker:
        query_ticker = str(query_ticker).strip().upper()
        if query_ticker and query_ticker != st.session_state.active_ticker:
            st.session_state.active_ticker = query_ticker

    with st.form("ticker-form", border=False):
        ticker_col, button_col = st.columns([4, 1])
        with ticker_col:
            default_ticker = st.session_state.active_ticker or "AAPL"
            ticker = st.text_input("Ticker", value=default_ticker, max_chars=16).strip().upper()
        with button_col:
            st.write("")
            submitted = st.form_submit_button("Calculate DCF", use_container_width=True)

    if submitted:
        if not ticker:
            st.session_state.active_ticker = None
            st.error("Ticker is required.")
            return
        st.session_state.active_ticker = ticker
        st.query_params["ticker"] = ticker

    active_ticker = st.session_state.active_ticker

    try:
        with st.spinner(f"Loading {active_ticker} financial statements..."):
            data = load_ticker(active_ticker)
    except Exception as exc:
        st.error(str(exc))
        return

    defaults = default_assumptions(data)
    if not is_probably_us_listed(data):
        st.warning(
            f"{data.ticker} does not look like a USD US-listed common stock from the available metadata "
            f"(currency: {data.currency}, exchange: {data.exchange or 'unknown'})."
        )

    st.subheader(f"{data.company_name} ({data.ticker})")

    assumption_col, output_col = st.columns([0.34, 0.66], gap="large")
    with assumption_col:
        st.markdown("### Assumptions")
        growth_rate_stage_1 = st.number_input(
            "Revenue growth years 1-5",
            min_value=-50.0,
            max_value=80.0,
            value=round(defaults.growth_rate_stage_1 * 100, 1),
            step=0.5,
            format="%.1f",
        )
        growth_rate_stage_2 = st.number_input(
            "Revenue growth years 6+",
            min_value=-50.0,
            max_value=50.0,
            value=round(defaults.growth_rate_stage_2 * 100, 1),
            step=0.5,
            format="%.1f",
        )
        target_fcf_margin = st.number_input(
            "Target FCF margin",
            min_value=0.1,
            max_value=80.0,
            value=round(defaults.target_fcf_margin * 100, 1),
            step=0.5,
            format="%.1f",
        )
        wacc = st.number_input(
            "WACC",
            min_value=0.1,
            max_value=40.0,
            value=round(defaults.wacc * 100, 1),
            step=0.25,
            format="%.2f",
        )
        terminal_growth_rate = st.number_input(
            "Terminal growth",
            min_value=-5.0,
            max_value=8.0,
            value=round(defaults.terminal_growth_rate * 100, 1),
            step=0.25,
            format="%.2f",
        )
        projection_years = st.number_input(
            "Projection years",
            min_value=3,
            max_value=15,
            value=defaults.projection_years,
            step=1,
        )
        margin_of_safety = st.number_input(
            "Margin of safety",
            min_value=0.0,
            max_value=80.0,
            value=round(defaults.margin_of_safety * 100, 1),
            step=1.0,
            format="%.1f",
        )

    assumptions = DCFAssumptions(
        growth_rate_stage_1=growth_rate_stage_1 / 100,
        growth_rate_stage_2=growth_rate_stage_2 / 100,
        target_fcf_margin=target_fcf_margin / 100,
        wacc=wacc / 100,
        terminal_growth_rate=terminal_growth_rate / 100,
        projection_years=int(projection_years),
        margin_of_safety=margin_of_safety / 100,
    )

    try:
        result = calculate_dcf(data, assumptions)
        sensitivity = sensitivity_table(data, assumptions)
    except DCFError as exc:
        st.error(str(exc))
        return
    stock_metrics = calculate_stock_metrics(data, result, assumptions)

    with output_col:
        metric_1, metric_2, metric_3 = st.columns(3)
        metric_1.metric("Intrinsic value / share", money(result.intrinsic_value_per_share, data.currency))
        metric_2.metric(
            "Current price",
            money(result.current_price, data.currency),
            delta=pct(result.upside_to_intrinsic) if result.upside_to_intrinsic is not None else None,
        )
        metric_3.metric("Buy-below price", money(result.buy_below_price, data.currency))

        value_col_1, value_col_2, value_col_3 = st.columns(3)
        value_col_1.metric("Enterprise value", compact_money(result.enterprise_value, data.currency))
        value_col_2.metric("Equity value", compact_money(result.equity_value, data.currency))
        value_col_3.metric("PV terminal value", compact_money(result.pv_terminal_value, data.currency))

        chart_data = result.projection[["year", "revenue", "free_cash_flow", "present_value_fcf"]].melt(
            id_vars="year",
            var_name="Metric",
            value_name="Value",
        )
        chart_data["Metric"] = chart_data["Metric"].map(
            {
                "revenue": "Revenue",
                "free_cash_flow": "Free cash flow",
                "present_value_fcf": "PV of FCF",
            }
        )
        fig = px.line(
            chart_data,
            x="year",
            y="Value",
            color="Metric",
            markers=True,
            color_discrete_map={
                "Revenue": "#176b64",
                "Free cash flow": "#9a5b1f",
                "PV of FCF": "#4b5f8f",
            },
        )
        fig.update_layout(
            height=310,
            margin=dict(l=10, r=10, t=25, b=10),
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            legend_title_text="",
            xaxis_title="Projection year",
            yaxis_title=data.currency,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Keep single-stock valuation, portfolio risk, and factor regression in separate workflows.
    (
        tab_projection,
        tab_stock_metrics,
        tab_sensitivity,
        tab_history,
        tab_portfolio_analytics,
        tab_portfolio,
        tab_export,
    ) = st.tabs(
        [
            "Projection",
            "Valuation Metrics",
            "Sensitivity",
            "Historical data",
            "Portfolio Analytics",
            "Portfolio Regression",
            "Export",
        ]
    )

    with tab_projection:
        st.dataframe(style_projection(result.projection, data.currency), use_container_width=True, hide_index=True)

    with tab_stock_metrics:
        st.markdown("### Stock Valuation and Quality Metrics")
        stock_metric_1, stock_metric_2, stock_metric_3, stock_metric_4, stock_metric_5 = st.columns(5)
        stock_metric_1.metric("FCF yield", format_metric_value(metric_value(stock_metrics, "FCF yield"), "percent", data.currency))
        stock_metric_2.metric("P/FCF", format_metric_value(metric_value(stock_metrics, "P/FCF"), "multiple", data.currency))
        stock_metric_3.metric("ROIC", format_metric_value(metric_value(stock_metrics, "ROIC"), "percent", data.currency))
        stock_metric_4.metric("ROIC spread", format_metric_value(metric_value(stock_metrics, "ROIC - WACC spread"), "percent", data.currency))
        stock_metric_5.metric("Altman Z", format_metric_value(metric_value(stock_metrics, "Altman Z-score"), "number", data.currency))
        st.dataframe(style_metric_table(stock_metrics, data.currency), use_container_width=True, hide_index=True)
        st.markdown(
            "<p class='small-note'>Metrics use the latest available annual statements and market price. "
            "Some formulas show N/A when Yahoo Finance does not provide the needed input.</p>",
            unsafe_allow_html=True,
        )

    with tab_sensitivity:
        formatted_sensitivity = sensitivity.map(lambda value: money(value, data.currency) if pd.notna(value) else "N/A")
        st.dataframe(formatted_sensitivity, use_container_width=True)

    with tab_history:
        st.dataframe(style_history(data.financial_history, data.currency), use_container_width=True, hide_index=True)

    with tab_portfolio_analytics:
        st.markdown("### Portfolio Performance and Risk")
        with st.form("portfolio-analytics-form", border=False):
            input_col_1, input_col_2 = st.columns([0.55, 0.45])
            with input_col_1:
                analytics_tickers = st.text_input(
                    "Portfolio tickers",
                    value="AAPL, MSFT, NVDA",
                    key="analytics_portfolio_tickers",
                )
                analytics_weights = st.text_input(
                    "Weights",
                    value="",
                    help="Leave blank for equal weights, or enter values like 40, 40, 20.",
                    key="analytics_portfolio_weights",
                )
                analytics_rebalancing = st.radio(
                    "Rebalancing method",
                    ["Monthly rebalance", "Buy and hold"],
                    horizontal=True,
                    key="analytics_rebalancing_method",
                )
            with input_col_2:
                analytics_benchmark = st.text_input("Benchmark", value="SPY", key="analytics_benchmark")
                analytics_risk_free = st.number_input(
                    "Risk-free rate",
                    min_value=0.0,
                    max_value=20.0,
                    value=4.0,
                    step=0.25,
                    format="%.2f",
                    key="analytics_risk_free",
                )
                analytics_start_date = st.date_input(
                    "Start date",
                    value=date(2020, 1, 1),
                    min_value=date(1980, 1, 1),
                    key="analytics_start_date",
                )
                analytics_end_date = st.date_input(
                    "End date",
                    value=date.today(),
                    min_value=date(1980, 1, 1),
                    key="analytics_end_date",
                )
                run_analytics = st.form_submit_button("Run Portfolio Analytics", use_container_width=True)

        if not run_analytics:
            st.markdown(
                "<p class='small-note'>Run this to calculate CAGR, volatility, Sharpe, Sortino, drawdown, beta, "
                "Jensen alpha, Treynor, information ratio, VaR, CVaR, correlation, risk contribution, and an efficient frontier.</p>",
                unsafe_allow_html=True,
            )
        else:
            try:
                tickers = parse_tickers(analytics_tickers)
                weights = parse_weights(analytics_weights)
                with st.spinner("Loading prices and calculating portfolio analytics..."):
                    analytics = load_portfolio_analytics(
                        tickers,
                        weights,
                        analytics_start_date,
                        analytics_end_date,
                        analytics_rebalancing,
                        analytics_benchmark,
                        analytics_risk_free / 100,
                    )
            except (PortfolioAnalyticsError, FactorRegressionError, ValueError) as exc:
                st.error(str(exc))
            else:
                metric_cagr, metric_vol, metric_sharpe, metric_drawdown, metric_beta = st.columns(5)
                metric_cagr.metric("CAGR", format_metric_value(metric_value(analytics.metrics, "CAGR"), "percent"))
                metric_vol.metric("Volatility", format_metric_value(metric_value(analytics.metrics, "Annualized volatility"), "percent"))
                metric_sharpe.metric("Sharpe", format_metric_value(metric_value(analytics.metrics, "Sharpe ratio"), "number"))
                metric_drawdown.metric("Max drawdown", format_metric_value(metric_value(analytics.metrics, "Maximum drawdown"), "percent"))
                metric_beta.metric("Beta", format_metric_value(metric_value(analytics.metrics, "Beta"), "number"))

                portfolio_chart_col, drawdown_chart_col = st.columns([0.55, 0.45], gap="large")
                with portfolio_chart_col:
                    cumulative = analytics.cumulative_returns.reset_index(names="Month").melt(
                        id_vars="Month",
                        var_name="Series",
                        value_name="Cumulative return",
                    )
                    cumulative_fig = px.line(
                        cumulative,
                        x="Month",
                        y="Cumulative return",
                        color="Series",
                        color_discrete_sequence=["#176b64", "#4b5f8f", "#9a5b1f"],
                    )
                    cumulative_fig.update_layout(
                        height=320,
                        margin=dict(l=10, r=10, t=20, b=10),
                        paper_bgcolor="#ffffff",
                        plot_bgcolor="#ffffff",
                        legend_title_text="",
                        yaxis_tickformat=".0%",
                    )
                    st.plotly_chart(cumulative_fig, use_container_width=True)
                with drawdown_chart_col:
                    drawdowns = analytics.drawdowns.reset_index(names="Month").melt(
                        id_vars="Month",
                        var_name="Series",
                        value_name="Drawdown",
                    )
                    drawdown_fig = px.area(
                        drawdowns,
                        x="Month",
                        y="Drawdown",
                        color="Series",
                        color_discrete_sequence=["#176b64", "#4b5f8f", "#9a5b1f"],
                    )
                    drawdown_fig.update_layout(
                        height=320,
                        margin=dict(l=10, r=10, t=20, b=10),
                        paper_bgcolor="#ffffff",
                        plot_bgcolor="#ffffff",
                        legend_title_text="",
                        yaxis_tickformat=".0%",
                    )
                    st.plotly_chart(drawdown_fig, use_container_width=True)

                analytics_tab_1, analytics_tab_2, analytics_tab_3, analytics_tab_4, analytics_tab_5, analytics_tab_6 = st.tabs(
                    ["Metrics", "Correlation", "Risk contribution", "Efficient frontier", "Monthly returns", "Export analytics"]
                )
                with analytics_tab_1:
                    st.dataframe(style_metric_table(analytics.metrics), use_container_width=True, hide_index=True)
                    st.dataframe(
                        style_weight_table(
                            analytics.weights,
                            analytics.ending_weights,
                            analytics.rebalancing_method == "Buy and hold",
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )
                with analytics_tab_2:
                    corr_fig = px.imshow(
                        analytics.correlation,
                        text_auto=".2f",
                        color_continuous_scale=["#9a5b1f", "#f3f6f4", "#176b64"],
                        zmin=-1,
                        zmax=1,
                    )
                    corr_fig.update_layout(
                        height=420,
                        margin=dict(l=10, r=10, t=20, b=10),
                        paper_bgcolor="#ffffff",
                        plot_bgcolor="#ffffff",
                    )
                    st.plotly_chart(corr_fig, use_container_width=True)
                    st.dataframe(analytics.correlation.round(3), use_container_width=True)
                with analytics_tab_3:
                    risk_fig = px.bar(
                        analytics.risk_contribution,
                        x="Ticker",
                        y="Risk contribution",
                        color="Risk contribution",
                        color_continuous_scale=["#9a5b1f", "#f3f6f4", "#176b64"],
                    )
                    risk_fig.update_layout(
                        height=360,
                        margin=dict(l=10, r=10, t=20, b=10),
                        paper_bgcolor="#ffffff",
                        plot_bgcolor="#ffffff",
                        coloraxis_showscale=False,
                        yaxis_tickformat=".0%",
                    )
                    st.plotly_chart(risk_fig, use_container_width=True)
                    risk_display = analytics.risk_contribution.copy()
                    risk_display["Weight"] = risk_display["Weight"].map(pct)
                    risk_display["Risk contribution"] = risk_display["Risk contribution"].map(pct)
                    st.dataframe(risk_display, use_container_width=True, hide_index=True)
                with analytics_tab_4:
                    frontier = analytics.efficient_frontier.copy()
                    weight_columns = [column for column in frontier.columns if column.endswith(" weight")]
                    frontier_fig = px.scatter(
                        frontier,
                        x="Annualized volatility",
                        y="Annualized return",
                        color="Portfolio",
                        hover_data=["Sharpe ratio"] + weight_columns,
                        color_discrete_map={
                            "Sample": "#aab5b2",
                            "Current": "#176b64",
                            "Max Sharpe": "#4b5f8f",
                            "Min Volatility": "#9a5b1f",
                        },
                    )
                    frontier_fig.update_layout(
                        height=430,
                        margin=dict(l=10, r=10, t=20, b=10),
                        paper_bgcolor="#ffffff",
                        plot_bgcolor="#ffffff",
                        xaxis_tickformat=".0%",
                        yaxis_tickformat=".0%",
                    )
                    st.plotly_chart(frontier_fig, use_container_width=True)
                    frontier_display = frontier[frontier["Portfolio"].ne("Sample")].copy()
                    for column in ["Annualized return", "Annualized volatility"]:
                        frontier_display[column] = frontier_display[column].map(pct)
                    frontier_display["Sharpe ratio"] = frontier_display["Sharpe ratio"].map(lambda value: f"{value:.2f}" if pd.notna(value) else "N/A")
                    for column in weight_columns:
                        frontier_display[column] = frontier_display[column].map(pct)
                    st.dataframe(frontier_display, use_container_width=True, hide_index=True)
                with analytics_tab_5:
                    st.dataframe(style_percent_frame(analytics.monthly_returns), use_container_width=True)
                with analytics_tab_6:
                    st.download_button(
                        "Download Portfolio Analytics CSV",
                        data=portfolio_analytics_csv(analytics),
                        file_name="portfolio_analytics.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
                    notes = " ".join(analytics.price_source_notes)
                    st.markdown(f"<p class='small-note'>{notes}</p>", unsafe_allow_html=True)

    with tab_portfolio:
        st.markdown("### Fama-French Portfolio Regression")
        with st.form("factor-regression-form", border=False):
            input_col_1, input_col_2 = st.columns([0.55, 0.45])
            with input_col_1:
                portfolio_tickers = st.text_input("Portfolio tickers", value="AAPL, MSFT, NVDA")
                portfolio_weights = st.text_input("Weights", value="", help="Leave blank for equal weights, or enter values like 40, 40, 20.")
                model_name = st.selectbox(
                    "Factor model",
                    ["Fama-French 5 Factor", "Fama-French 3 Factor"],
                    index=0,
                )
            with input_col_2:
                rebalancing_method = st.radio(
                    "Rebalancing method",
                    ["Monthly rebalance", "Buy and hold"],
                    horizontal=True,
                )
                start_date = st.date_input("Start date", value=date(2020, 1, 1), min_value=date(1980, 1, 1))
                end_date = st.date_input("End date", value=date.today(), min_value=date(1980, 1, 1))
                run_regression = st.form_submit_button("Run Regression", use_container_width=True)

        if not run_regression:
            st.markdown(
                "<p class='small-note'>Enter a weighted portfolio and run a monthly Fama-French regression. "
                "Blank weights use equal starting weights. Monthly rebalance keeps weights fixed; buy and hold lets weights drift.</p>",
                unsafe_allow_html=True,
            )
        else:
            try:
                tickers = parse_tickers(portfolio_tickers)
                weights = parse_weights(portfolio_weights)
                if start_date >= end_date:
                    raise FactorRegressionError("Start date must be before end date.")
                with st.spinner("Loading prices, factors, and running regression..."):
                    regression = load_factor_regression(
                        tickers,
                        weights,
                        start_date,
                        end_date,
                        model_name,
                        rebalancing_method,
                    )
            except (FactorRegressionError, ValueError) as exc:
                st.error(str(exc))
            else:
                market_beta = regression.coefficients.loc[
                    regression.coefficients["Factor"].eq("Mkt-RF"), "Coefficient"
                ].iloc[0]
                metric_alpha, metric_r2, metric_beta, metric_obs = st.columns(4)
                metric_alpha.metric("Annualized alpha", pct(diagnostic_value(regression.diagnostics, "Annualized alpha")))
                metric_r2.metric("R-squared", pct(diagnostic_value(regression.diagnostics, "R-squared")))
                metric_beta.metric("Market beta", f"{market_beta:.2f}")
                metric_obs.metric("Monthly observations", f"{int(diagnostic_value(regression.diagnostics, 'Observations'))}")

                chart_col_1, chart_col_2 = st.columns([0.58, 0.42], gap="large")
                with chart_col_1:
                    cumulative = regression.cumulative_returns.reset_index(names="Month").melt(
                        id_vars="Month",
                        var_name="Series",
                        value_name="Cumulative return",
                    )
                    cumulative_fig = px.line(
                        cumulative,
                        x="Month",
                        y="Cumulative return",
                        color="Series",
                        color_discrete_map={"Portfolio": "#176b64", "Market proxy": "#4b5f8f"},
                    )
                    cumulative_fig.update_layout(
                        height=330,
                        margin=dict(l=10, r=10, t=20, b=10),
                        paper_bgcolor="#ffffff",
                        plot_bgcolor="#ffffff",
                        legend_title_text="",
                        yaxis_tickformat=".0%",
                    )
                    st.plotly_chart(cumulative_fig, use_container_width=True)

                with chart_col_2:
                    exposures = regression.coefficients[~regression.coefficients["Factor"].eq("Alpha")].copy()
                    exposure_fig = px.bar(
                        exposures,
                        x="Factor",
                        y="Coefficient",
                        color="Coefficient",
                        color_continuous_scale=["#9a5b1f", "#f3f6f4", "#176b64"],
                    )
                    exposure_fig.update_layout(
                        height=330,
                        margin=dict(l=10, r=10, t=20, b=10),
                        paper_bgcolor="#ffffff",
                        plot_bgcolor="#ffffff",
                        coloraxis_showscale=False,
                        yaxis_title="Exposure",
                    )
                    st.plotly_chart(exposure_fig, use_container_width=True)

                result_tab_1, result_tab_2, result_tab_3, result_tab_4 = st.tabs(
                    ["Coefficients", "Diagnostics", "Regression data", "Export regression"]
                )
                with result_tab_1:
                    st.dataframe(style_factor_coefficients(regression.coefficients), use_container_width=True, hide_index=True)
                with result_tab_2:
                    st.dataframe(
                        style_weight_table(
                            regression.weights,
                            regression.ending_weights,
                            regression.rebalancing_method == "Buy and hold",
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )
                    st.dataframe(style_diagnostics(regression.diagnostics), use_container_width=True, hide_index=True)
                    notes = " ".join(regression.price_source_notes + (regression.factor_source_note,))
                    st.markdown(f"<p class='small-note'>{notes}</p>", unsafe_allow_html=True)
                with result_tab_3:
                    regression_display = regression.regression_data.copy()
                    for column in regression_display.columns:
                        regression_display[column] = regression_display[column].map(lambda value: f"{value:.2%}")
                    st.dataframe(regression_display, use_container_width=True)
                with result_tab_4:
                    st.download_button(
                        "Download Regression CSV",
                        data=factor_regression_csv(regression),
                        file_name="fama_french_portfolio_regression.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

    with tab_export:
        download_1, download_2 = st.columns(2)
        download_1.download_button(
            "Download CSV",
            data=summary_csv(result, sensitivity),
            file_name=f"{data.ticker}_dcf.csv",
            mime="text/csv",
            use_container_width=True,
        )
        download_2.download_button(
            "Download Excel",
            data=export_workbook(data, assumptions, result, sensitivity),
            file_name=f"{data.ticker}_dcf.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

        notes = " ".join(data.source_notes)
        st.markdown(f"<p class='small-note'>{notes}</p>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
