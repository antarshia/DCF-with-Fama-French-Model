from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from dcf_model import (
    DCFAssumptions,
    DCFError,
    calculate_dcf,
    default_assumptions,
    export_workbook,
    fetch_ticker_data,
    is_probably_us_listed,
    sensitivity_table,
    summary_csv,
)


st.set_page_config(
    page_title="DCF Valuation",
    layout="wide",
    initial_sidebar_state="collapsed",
)


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
        st.session_state.active_ticker = None

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
    if not active_ticker:
        st.markdown(
            "<p class='small-note'>Enter a US-listed stock ticker and calculate a discounted cash-flow valuation.</p>",
            unsafe_allow_html=True,
        )
        return

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

    tab_projection, tab_sensitivity, tab_history, tab_export = st.tabs(
        ["Projection", "Sensitivity", "Historical data", "Export"]
    )

    with tab_projection:
        st.dataframe(style_projection(result.projection, data.currency), use_container_width=True, hide_index=True)

    with tab_sensitivity:
        formatted_sensitivity = sensitivity.map(lambda value: money(value, data.currency) if pd.notna(value) else "N/A")
        st.dataframe(formatted_sensitivity, use_container_width=True)

    with tab_history:
        st.dataframe(style_history(data.financial_history, data.currency), use_container_width=True, hide_index=True)

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
