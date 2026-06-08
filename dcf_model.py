from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO, StringIO
import os
from zipfile import ZipFile
from typing import Iterable

import numpy as np
import pandas as pd


REVENUE_ROWS = (
    "Total Revenue",
    "Revenue",
    "Operating Revenue",
)

# Yahoo Finance and SEC statements use slightly different row/tag names.
# These aliases keep the app tolerant of common statement naming differences.
FREE_CASH_FLOW_ROWS = (
    "Free Cash Flow",
    "FreeCashFlow",
)

OPERATING_CASH_FLOW_ROWS = (
    "Operating Cash Flow",
    "Total Cash From Operating Activities",
    "Cash Flow From Continuing Operating Activities",
)

CAPEX_ROWS = (
    "Capital Expenditure",
    "Capital Expenditures",
    "CapitalExpenditures",
    "Purchase Of Property Plant Equipment",
)

CASH_ROWS = (
    "Cash And Cash Equivalents",
    "Cash Cash Equivalents And Short Term Investments",
    "Cash And Cash Equivalents And Short Term Investments",
)

DEBT_ROWS = (
    "Total Debt",
    "Long Term Debt",
    "Long Term Debt And Capital Lease Obligation",
    "Short Long Term Debt Total",
)

SHARES_ROWS = (
    "Ordinary Shares Number",
    "Share Issued",
    "Common Stock Shares Outstanding",
)

NET_INCOME_ROWS = (
    "Net Income",
    "Net Income Common Stockholders",
    "Net Income Continuous Operations",
)

PRETAX_INCOME_ROWS = (
    "Pretax Income",
    "Income Before Tax",
    "Earnings Before Tax",
)

TAX_PROVISION_ROWS = (
    "Tax Provision",
    "Income Tax Expense",
)

EBIT_ROWS = (
    "EBIT",
    "Earnings Before Interest And Taxes",
    "Operating Income",
    "Operating Income Loss",
)

EBITDA_ROWS = (
    "EBITDA",
    "Normalized EBITDA",
)

TOTAL_EQUITY_ROWS = (
    "Stockholders Equity",
    "Total Equity Gross Minority Interest",
    "Common Stock Equity",
    "Total Stockholder Equity",
)

TOTAL_ASSETS_ROWS = (
    "Total Assets",
)

TOTAL_LIABILITIES_ROWS = (
    "Total Liabilities Net Minority Interest",
    "Total Liab",
    "Total Liabilities",
)

CURRENT_ASSETS_ROWS = (
    "Current Assets",
    "Total Current Assets",
)

CURRENT_LIABILITIES_ROWS = (
    "Current Liabilities",
    "Total Current Liabilities",
)

RETAINED_EARNINGS_ROWS = (
    "Retained Earnings",
)

SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
SEC_DEFAULT_USER_AGENT = "dcf-valuation-streamlit-app/1.0 contact@example.com"

FAMA_FRENCH_DATASETS = {
    # Official Kenneth French monthly factor datasets, downloaded as CSV ZIP files.
    "Fama-French 3 Factor": {
        "url": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_CSV.zip",
        "factors": ("Mkt-RF", "SMB", "HML"),
    },
    "Fama-French 5 Factor": {
        "url": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip",
        "factors": ("Mkt-RF", "SMB", "HML", "RMW", "CMA"),
    },
}

SEC_REVENUE_TAGS = (
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "Revenues",
    "SalesRevenueNet",
    "SalesRevenueGoodsNet",
    "TotalRevenuesAndOtherIncome",
)

SEC_OPERATING_CASH_FLOW_TAGS = (
    "NetCashProvidedByUsedInOperatingActivities",
    "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
)

SEC_CAPEX_TAGS = (
    "PaymentsToAcquirePropertyPlantAndEquipment",
    "PaymentsToAcquireProductiveAssets",
)

SEC_CASH_TAGS = (
    "CashCashEquivalentsAndShortTermInvestments",
    "CashAndCashEquivalentsAtCarryingValue",
    "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
)

SEC_SHORT_BORROWINGS_TAGS = (
    "CommercialPaper",
    "ShortTermBorrowings",
    "ShortTermBorrowingsAndFinanceLeaseObligationsCurrent",
)

SEC_CURRENT_LONG_TERM_DEBT_TAGS = (
    "LongTermDebtCurrent",
    "CurrentPortionOfLongTermDebt",
    "CurrentPortionOfLongTermDebtAndFinanceLeaseObligations",
)

SEC_NONCURRENT_DEBT_TAGS = (
    "LongTermDebtNoncurrent",
    "LongTermDebtAndFinanceLeaseObligationsNoncurrent",
)

SEC_TOTAL_DEBT_TAGS = (
    "LongTermDebtAndFinanceLeaseObligations",
    "LongTermDebt",
)


@dataclass(frozen=True)
class TickerData:
    ticker: str
    company_name: str
    currency: str
    exchange: str
    current_price: float | None
    shares_outstanding: float | None
    cash_and_equivalents: float
    total_debt: float
    financial_history: pd.DataFrame
    source_notes: tuple[str, ...]
    market_cap: float | None = None
    net_income: float | None = None
    pretax_income: float | None = None
    tax_provision: float | None = None
    ebit: float | None = None
    ebitda: float | None = None
    total_equity: float | None = None
    total_assets: float | None = None
    total_liabilities: float | None = None
    current_assets: float | None = None
    current_liabilities: float | None = None
    retained_earnings: float | None = None
    trailing_eps: float | None = None
    forward_eps: float | None = None
    dividend_rate: float | None = None
    beta: float | None = None


@dataclass(frozen=True)
class DCFAssumptions:
    growth_rate_stage_1: float
    growth_rate_stage_2: float
    target_fcf_margin: float
    wacc: float
    terminal_growth_rate: float
    projection_years: int
    margin_of_safety: float


@dataclass(frozen=True)
class DCFResult:
    ticker: str
    intrinsic_value_per_share: float
    current_price: float | None
    upside_to_intrinsic: float | None
    buy_below_price: float
    enterprise_value: float
    equity_value: float
    pv_projected_fcf: float
    pv_terminal_value: float
    terminal_value: float
    projection: pd.DataFrame
    summary: pd.DataFrame


@dataclass(frozen=True)
class FactorRegressionResult:
    model_name: str
    rebalancing_method: str
    tickers: tuple[str, ...]
    weights: pd.Series
    ending_weights: pd.Series
    coefficients: pd.DataFrame
    diagnostics: pd.DataFrame
    regression_data: pd.DataFrame
    cumulative_returns: pd.DataFrame
    price_source_notes: tuple[str, ...]
    factor_source_note: str


@dataclass(frozen=True)
class PortfolioAnalyticsResult:
    tickers: tuple[str, ...]
    benchmark_ticker: str
    rebalancing_method: str
    risk_free_rate: float
    weights: pd.Series
    ending_weights: pd.Series
    metrics: pd.DataFrame
    cumulative_returns: pd.DataFrame
    drawdowns: pd.DataFrame
    correlation: pd.DataFrame
    risk_contribution: pd.DataFrame
    efficient_frontier: pd.DataFrame
    monthly_returns: pd.DataFrame
    price_source_notes: tuple[str, ...]


class DCFError(ValueError):
    """Raised when the valuation cannot be calculated from the available data."""


class FactorRegressionError(ValueError):
    """Raised when portfolio factor regression cannot be calculated."""


class PortfolioAnalyticsError(ValueError):
    """Raised when portfolio analytics cannot be calculated."""


def normalize_ticker(raw_ticker: str) -> str:
    """Normalize common US ticker input while preserving Yahoo class-share format."""
    ticker = raw_ticker.strip().upper()
    if "." in ticker and not ticker.endswith((".A", ".B")):
        return ticker
    return ticker.replace(".", "-")


def fetch_ticker_data(raw_ticker: str) -> TickerData:
    """Fetch annual financial statement data for a US-listed ticker."""
    import yfinance as yf

    ticker = normalize_ticker(raw_ticker)
    stock = yf.Ticker(ticker)

    financials = _safe_call(lambda: stock.financials)
    cashflow = _safe_call(lambda: stock.cashflow)
    balance_sheet = _safe_call(lambda: stock.balance_sheet)
    info = _safe_call(lambda: stock.info) or {}
    fast_info = _safe_call(lambda: stock.fast_info) or {}

    history = build_financial_history(financials, cashflow)
    sec_data = _safe_call(lambda: fetch_sec_ticker_data(ticker)) if history.empty else None
    if history.empty and sec_data is not None:
        history = sec_data["financial_history"]
    if history.empty:
        raise DCFError(
            f"No annual revenue or cash-flow data was returned for {ticker}. "
            "Try another US-listed common stock ticker, or set SEC_USER_AGENT if SEC EDGAR rejects the request."
        )

    current_price = _current_price(stock, fast_info, ticker)
    shares = _number_from_mapping(fast_info, ("shares", "sharesOutstanding"))
    if shares is None:
        shares = _number_from_mapping(info, ("sharesOutstanding", "impliedSharesOutstanding"))
    if shares is None:
        shares = _latest_statement_value(balance_sheet, SHARES_ROWS)
    if shares is None and sec_data is not None:
        shares = sec_data["shares_outstanding"]

    cash = _latest_statement_value(balance_sheet, CASH_ROWS) or 0.0
    if cash == 0.0 and sec_data is not None:
        cash = sec_data["cash_and_equivalents"] or 0.0
    debt = _latest_statement_value(balance_sheet, DEBT_ROWS) or 0.0
    if debt == 0.0 and sec_data is not None:
        debt = sec_data["total_debt"] or 0.0

    # Extra statement fields are optional. When Yahoo omits them, downstream metrics render as N/A.
    market_cap = _number_from_mapping(fast_info, ("market_cap", "marketCap"))
    if market_cap is None:
        market_cap = _number_from_mapping(info, ("marketCap",))
    if market_cap is None and current_price is not None and shares is not None:
        market_cap = current_price * shares

    net_income = _latest_statement_value(financials, NET_INCOME_ROWS)
    pretax_income = _latest_statement_value(financials, PRETAX_INCOME_ROWS)
    tax_provision = _latest_statement_value(financials, TAX_PROVISION_ROWS)
    ebit = _latest_statement_value(financials, EBIT_ROWS)
    ebitda = _latest_statement_value(financials, EBITDA_ROWS)
    total_equity = _latest_statement_value(balance_sheet, TOTAL_EQUITY_ROWS)
    total_assets = _latest_statement_value(balance_sheet, TOTAL_ASSETS_ROWS)
    total_liabilities = _latest_statement_value(balance_sheet, TOTAL_LIABILITIES_ROWS)
    current_assets = _latest_statement_value(balance_sheet, CURRENT_ASSETS_ROWS)
    current_liabilities = _latest_statement_value(balance_sheet, CURRENT_LIABILITIES_ROWS)
    retained_earnings = _latest_statement_value(balance_sheet, RETAINED_EARNINGS_ROWS)

    notes: list[str] = []
    if sec_data is not None:
        notes.append("Annual financial statement data loaded from SEC EDGAR company facts.")
    else:
        notes.append("Financial statement data loaded from Yahoo Finance via yfinance.")
    if "free_cash_flow" in history and history["free_cash_flow"].notna().any():
        notes.append("Free cash flow uses reported FCF when available, otherwise operating cash flow plus/minus CapEx.")
    if shares is None:
        notes.append("Shares outstanding was not available, so intrinsic value per share cannot be calculated.")
    if current_price is None:
        notes.append("Current price was not available, so upside versus market price is blank.")

    sec_company_name = sec_data["company_name"] if sec_data is not None else None
    return TickerData(
        ticker=ticker,
        company_name=str(info.get("longName") or info.get("shortName") or sec_company_name or ticker),
        currency=str(info.get("currency") or _mapping_get(fast_info, "currency") or "USD"),
        exchange=str(info.get("exchange") or info.get("fullExchangeName") or ""),
        current_price=current_price,
        shares_outstanding=shares,
        cash_and_equivalents=float(cash),
        total_debt=float(debt),
        financial_history=history,
        source_notes=tuple(notes),
        market_cap=market_cap,
        net_income=net_income,
        pretax_income=pretax_income,
        tax_provision=tax_provision,
        ebit=ebit,
        ebitda=ebitda,
        total_equity=total_equity,
        total_assets=total_assets,
        total_liabilities=total_liabilities,
        current_assets=current_assets,
        current_liabilities=current_liabilities,
        retained_earnings=retained_earnings,
        trailing_eps=_clean_number(info.get("trailingEps")),
        forward_eps=_clean_number(info.get("forwardEps")),
        dividend_rate=_clean_number(info.get("dividendRate")),
        beta=_clean_number(info.get("beta")),
    )


def fetch_sec_ticker_data(raw_ticker: str) -> dict[str, object]:
    """Fetch annual facts from SEC EDGAR companyfacts for a US-listed ticker."""
    import requests

    ticker = normalize_ticker(raw_ticker)
    headers = {"User-Agent": os.getenv("SEC_USER_AGENT", SEC_DEFAULT_USER_AGENT)}

    ticker_response = requests.get(SEC_TICKERS_URL, headers=headers, timeout=20)
    ticker_response.raise_for_status()
    ticker_map = ticker_response.json()
    sec_row = _find_sec_ticker_row(ticker_map, ticker)
    if sec_row is None:
        raise DCFError(f"{ticker} was not found in the SEC ticker mapping.")

    cik = int(sec_row["cik_str"])
    facts_response = requests.get(SEC_COMPANY_FACTS_URL.format(cik=cik), headers=headers, timeout=30)
    facts_response.raise_for_status()
    companyfacts = facts_response.json()

    return {
        "ticker": ticker,
        "cik": cik,
        "company_name": companyfacts.get("entityName") or sec_row.get("title") or ticker,
        "financial_history": build_financial_history_from_sec(companyfacts),
        "shares_outstanding": _sec_latest_dei_value(companyfacts, "EntityCommonStockSharesOutstanding", "shares"),
        "cash_and_equivalents": _sec_latest_value(companyfacts, SEC_CASH_TAGS, duration=False),
        "total_debt": _sec_total_debt(companyfacts),
    }


def build_financial_history_from_sec(companyfacts: dict) -> pd.DataFrame:
    revenue = _sec_first_series(companyfacts, SEC_REVENUE_TAGS, duration=True)
    operating_cash_flow = _sec_first_series(companyfacts, SEC_OPERATING_CASH_FLOW_TAGS, duration=True)
    capex = _sec_first_series(companyfacts, SEC_CAPEX_TAGS, duration=True)

    if not operating_cash_flow.empty and not capex.empty:
        operating_cash_flow, capex = operating_cash_flow.align(capex, join="outer")
        free_cash_flow = operating_cash_flow.combine(capex, _combine_cfo_and_capex)
    else:
        free_cash_flow = pd.Series(dtype=float)

    history = pd.DataFrame(
        {
            "revenue": revenue,
            "operating_cash_flow": operating_cash_flow,
            "capex": capex,
            "free_cash_flow": free_cash_flow,
        }
    )
    history = history.replace([np.inf, -np.inf], np.nan).dropna(how="all")
    if history.empty:
        return pd.DataFrame(columns=["fiscal_year", "revenue", "operating_cash_flow", "capex", "free_cash_flow", "fcf_margin"])

    history = history.sort_index()
    history.insert(0, "fiscal_year", pd.to_datetime(history.index).year.astype(int))
    history = history.reset_index(drop=True)
    history["fcf_margin"] = history["free_cash_flow"] / history["revenue"]
    return history


def build_financial_history(financials: pd.DataFrame | None, cashflow: pd.DataFrame | None) -> pd.DataFrame:
    revenue = _statement_series(financials, REVENUE_ROWS)
    reported_fcf = _statement_series(cashflow, FREE_CASH_FLOW_ROWS)
    operating_cash_flow = _statement_series(cashflow, OPERATING_CASH_FLOW_ROWS)
    capex = _statement_series(cashflow, CAPEX_ROWS)

    if reported_fcf.empty and not operating_cash_flow.empty and not capex.empty:
        operating_cash_flow, capex = operating_cash_flow.align(capex, join="outer")
        reported_fcf = operating_cash_flow.combine(capex, _combine_cfo_and_capex)

    history = pd.DataFrame(
        {
            "revenue": revenue,
            "operating_cash_flow": operating_cash_flow,
            "capex": capex,
            "free_cash_flow": reported_fcf,
        }
    )
    history = history.replace([np.inf, -np.inf], np.nan).dropna(how="all")
    if history.empty:
        return pd.DataFrame(columns=["fiscal_year", "revenue", "operating_cash_flow", "capex", "free_cash_flow", "fcf_margin"])

    history.index = pd.to_datetime(history.index, errors="coerce")
    history = history[history.index.notna()]
    history = history.sort_index()
    history.insert(0, "fiscal_year", history.index.year.astype(int))
    history = history.reset_index(drop=True)
    history["fcf_margin"] = history["free_cash_flow"] / history["revenue"]
    return history


def default_assumptions(data: TickerData) -> DCFAssumptions:
    history = data.financial_history
    revenue_growth = _historical_cagr(history["revenue"]) if "revenue" in history else None
    fcf_margin = _recent_median(history.get("fcf_margin", pd.Series(dtype=float)))

    stage_1_growth = _clamp(revenue_growth if revenue_growth is not None else 0.08, -0.05, 0.25)
    stage_2_growth = _clamp(stage_1_growth * 0.55, -0.02, 0.12)
    target_margin = _clamp(fcf_margin if fcf_margin is not None else 0.15, 0.01, 0.45)

    return DCFAssumptions(
        growth_rate_stage_1=stage_1_growth,
        growth_rate_stage_2=stage_2_growth,
        target_fcf_margin=target_margin,
        wacc=0.09,
        terminal_growth_rate=0.025,
        projection_years=10,
        margin_of_safety=0.25,
    )


def calculate_dcf(data: TickerData, assumptions: DCFAssumptions) -> DCFResult:
    validate_assumptions(assumptions)

    if not data.shares_outstanding or data.shares_outstanding <= 0:
        raise DCFError("Shares outstanding is required to calculate intrinsic value per share.")

    base_revenue = _latest_positive(data.financial_history["revenue"])
    if base_revenue is None:
        raise DCFError("A positive latest annual revenue value is required.")

    rows: list[dict[str, float | int]] = []
    revenue = base_revenue
    for year in range(1, assumptions.projection_years + 1):
        growth = assumptions.growth_rate_stage_1 if year <= 5 else assumptions.growth_rate_stage_2
        revenue *= 1 + growth
        free_cash_flow = revenue * assumptions.target_fcf_margin
        discount_factor = 1 / ((1 + assumptions.wacc) ** year)
        present_value_fcf = free_cash_flow * discount_factor
        rows.append(
            {
                "year": year,
                "growth_rate": growth,
                "revenue": revenue,
                "target_fcf_margin": assumptions.target_fcf_margin,
                "free_cash_flow": free_cash_flow,
                "discount_factor": discount_factor,
                "present_value_fcf": present_value_fcf,
            }
        )

    projection = pd.DataFrame(rows)
    final_fcf = float(projection.iloc[-1]["free_cash_flow"])
    terminal_value = final_fcf * (1 + assumptions.terminal_growth_rate) / (
        assumptions.wacc - assumptions.terminal_growth_rate
    )
    pv_terminal_value = terminal_value / ((1 + assumptions.wacc) ** assumptions.projection_years)
    pv_projected_fcf = float(projection["present_value_fcf"].sum())
    enterprise_value = pv_projected_fcf + pv_terminal_value
    equity_value = enterprise_value + data.cash_and_equivalents - data.total_debt
    intrinsic_value_per_share = equity_value / data.shares_outstanding
    buy_below_price = intrinsic_value_per_share * (1 - assumptions.margin_of_safety)

    upside = None
    if data.current_price and data.current_price > 0:
        upside = intrinsic_value_per_share / data.current_price - 1

    summary = pd.DataFrame(
        [
            ("Ticker", data.ticker),
            ("Company", data.company_name),
            ("Currency", data.currency),
            ("Current price", data.current_price),
            ("Intrinsic value / share", intrinsic_value_per_share),
            ("Upside to intrinsic", upside),
            ("Buy-below price", buy_below_price),
            ("Enterprise value", enterprise_value),
            ("Equity value", equity_value),
            ("PV projected FCF", pv_projected_fcf),
            ("PV terminal value", pv_terminal_value),
            ("Cash and equivalents", data.cash_and_equivalents),
            ("Total debt", data.total_debt),
            ("Shares outstanding", data.shares_outstanding),
        ],
        columns=["Metric", "Value"],
    )

    return DCFResult(
        ticker=data.ticker,
        intrinsic_value_per_share=intrinsic_value_per_share,
        current_price=data.current_price,
        upside_to_intrinsic=upside,
        buy_below_price=buy_below_price,
        enterprise_value=enterprise_value,
        equity_value=equity_value,
        pv_projected_fcf=pv_projected_fcf,
        pv_terminal_value=pv_terminal_value,
        terminal_value=terminal_value,
        projection=projection,
        summary=summary,
    )


def sensitivity_table(
    data: TickerData,
    assumptions: DCFAssumptions,
    wacc_points: Iterable[float] | None = None,
    terminal_growth_points: Iterable[float] | None = None,
) -> pd.DataFrame:
    wacc_points = list(wacc_points or _spread_points(assumptions.wacc, step=0.005, count=5, floor=0.01))
    terminal_growth_points = list(
        terminal_growth_points or _spread_points(assumptions.terminal_growth_rate, step=0.005, count=5, floor=-0.02)
    )

    table = pd.DataFrame(index=[_format_percent(w) for w in wacc_points])
    table.index.name = "WACC \\ Terminal growth"

    for terminal_growth in terminal_growth_points:
        values: list[float | None] = []
        for wacc in wacc_points:
            if wacc <= terminal_growth:
                values.append(None)
                continue
            scenario = DCFAssumptions(
                growth_rate_stage_1=assumptions.growth_rate_stage_1,
                growth_rate_stage_2=assumptions.growth_rate_stage_2,
                target_fcf_margin=assumptions.target_fcf_margin,
                wacc=wacc,
                terminal_growth_rate=terminal_growth,
                projection_years=assumptions.projection_years,
                margin_of_safety=assumptions.margin_of_safety,
            )
            values.append(calculate_dcf(data, scenario).intrinsic_value_per_share)
        table[_format_percent(terminal_growth)] = values

    return table


def export_workbook(
    data: TickerData,
    assumptions: DCFAssumptions,
    result: DCFResult,
    sensitivity: pd.DataFrame,
) -> bytes:
    assumptions_df = pd.DataFrame(
        [
            ("Growth rate years 1-5", assumptions.growth_rate_stage_1),
            ("Growth rate years 6+", assumptions.growth_rate_stage_2),
            ("Target FCF margin", assumptions.target_fcf_margin),
            ("WACC", assumptions.wacc),
            ("Terminal growth rate", assumptions.terminal_growth_rate),
            ("Projection years", assumptions.projection_years),
            ("Margin of safety", assumptions.margin_of_safety),
        ],
        columns=["Assumption", "Value"],
    )

    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        result.summary.to_excel(writer, sheet_name="Summary", index=False)
        assumptions_df.to_excel(writer, sheet_name="Assumptions", index=False)
        data.financial_history.to_excel(writer, sheet_name="Historical data", index=False)
        result.projection.to_excel(writer, sheet_name="Projection", index=False)
        sensitivity.to_excel(writer, sheet_name="Sensitivity")
    return buffer.getvalue()


def summary_csv(result: DCFResult, sensitivity: pd.DataFrame) -> bytes:
    parts = [
        "Summary",
        result.summary.to_csv(index=False),
        "\nProjection",
        result.projection.to_csv(index=False),
        "\nSensitivity",
        sensitivity.to_csv(),
    ]
    return "\n".join(parts).encode("utf-8")


def calculate_stock_metrics(
    data: TickerData,
    result: DCFResult,
    assumptions: DCFAssumptions,
) -> pd.DataFrame:
    # Build one normalized metrics table so Streamlit can display formulas and values together.
    market_cap = data.market_cap
    if market_cap is None and data.current_price is not None and data.shares_outstanding is not None:
        market_cap = data.current_price * data.shares_outstanding
    market_enterprise_value = market_cap + data.total_debt - data.cash_and_equivalents if market_cap is not None else None
    latest_revenue = _latest_positive(data.financial_history["revenue"]) if "revenue" in data.financial_history else None
    latest_fcf = _latest_positive(data.financial_history["free_cash_flow"]) if "free_cash_flow" in data.financial_history else None
    revenue_cagr = _historical_cagr(data.financial_history["revenue"]) if "revenue" in data.financial_history else None
    fcf_cagr = _historical_cagr(data.financial_history["free_cash_flow"]) if "free_cash_flow" in data.financial_history else None
    fcf_margin = _recent_median(data.financial_history.get("fcf_margin", pd.Series(dtype=float)))

    tax_rate = _safe_divide(data.tax_provision, data.pretax_income)
    if tax_rate is None or tax_rate < 0 or tax_rate > 0.6:
        tax_rate = 0.21
    nopat = data.ebit * (1 - tax_rate) if data.ebit is not None else None
    invested_capital = None
    if data.total_equity is not None:
        invested_capital = data.total_equity + data.total_debt - data.cash_and_equivalents
        if invested_capital <= 0:
            invested_capital = None

    roic = _safe_divide(nopat, invested_capital)
    roe = _safe_divide(data.net_income, data.total_equity)
    net_margin = _safe_divide(data.net_income, latest_revenue)
    asset_turnover = _safe_divide(latest_revenue, data.total_assets)
    equity_multiplier = _safe_divide(data.total_assets, data.total_equity)
    dupont_roe = None
    if net_margin is not None and asset_turnover is not None and equity_multiplier is not None:
        dupont_roe = net_margin * asset_turnover * equity_multiplier

    residual_income_value = None
    if data.total_equity is not None and data.net_income is not None and assumptions.wacc > assumptions.terminal_growth_rate:
        equity_charge = data.total_equity * assumptions.wacc
        residual_income = data.net_income - equity_charge
        residual_income_value = data.total_equity + residual_income * (1 + assumptions.terminal_growth_rate) / (
            assumptions.wacc - assumptions.terminal_growth_rate
        )

    ddm_value_per_share = None
    if (
        data.dividend_rate is not None
        and data.dividend_rate > 0
        and assumptions.wacc > assumptions.terminal_growth_rate
    ):
        ddm_value_per_share = data.dividend_rate * (1 + assumptions.terminal_growth_rate) / (
            assumptions.wacc - assumptions.terminal_growth_rate
        )

    working_capital = None
    if data.current_assets is not None and data.current_liabilities is not None:
        working_capital = data.current_assets - data.current_liabilities
    altman_z = None
    if (
        working_capital is not None
        and data.retained_earnings is not None
        and data.ebit is not None
        and data.total_assets is not None
        and data.total_assets > 0
        and data.total_liabilities is not None
        and data.total_liabilities > 0
        and market_cap is not None
        and latest_revenue is not None
    ):
        altman_z = (
            1.2 * working_capital / data.total_assets
            + 1.4 * data.retained_earnings / data.total_assets
            + 3.3 * data.ebit / data.total_assets
            + 0.6 * market_cap / data.total_liabilities
            + latest_revenue / data.total_assets
        )

    pe = _safe_divide(data.current_price, data.trailing_eps)
    if pe is None:
        pe = _safe_divide(market_cap, data.net_income)
    rows = [
        _metric_row("Valuation", "Market cap", market_cap, "currency", "Price x shares outstanding", ""),
        _metric_row("Valuation", "Enterprise value", market_enterprise_value, "currency", "Market cap + debt - cash", ""),
        _metric_row("Valuation", "P/E", pe, "multiple", "Price / EPS or market cap / net income", ""),
        _metric_row("Valuation", "Forward P/E", _safe_divide(data.current_price, data.forward_eps), "multiple", "Price / forward EPS", ""),
        _metric_row("Valuation", "P/S", _safe_divide(market_cap, latest_revenue), "multiple", "Market cap / revenue", ""),
        _metric_row("Valuation", "EV/EBITDA", _safe_divide(market_enterprise_value, data.ebitda), "multiple", "Enterprise value / EBITDA", ""),
        _metric_row("Valuation", "P/FCF", _safe_divide(market_cap, latest_fcf), "multiple", "Market cap / free cash flow", ""),
        _metric_row("Valuation", "FCF yield", _safe_divide(latest_fcf, market_cap), "percent", "Free cash flow / market cap", ""),
        _metric_row("Valuation", "Earnings yield", _safe_divide(data.net_income, market_cap), "percent", "Net income / market cap", ""),
        _metric_row("Valuation", "DCF upside", result.upside_to_intrinsic, "percent", "Intrinsic value / current price - 1", ""),
        _metric_row("Dividend", "Dividend yield", _safe_divide(data.dividend_rate, data.current_price), "percent", "Dividend per share / price", ""),
        _metric_row("Dividend", "DDM value / share", ddm_value_per_share, "currency", "Next dividend / (WACC - terminal growth)", "Uses WACC as a practical cost-of-equity proxy."),
        _metric_row("Quality", "Revenue CAGR", revenue_cagr, "percent", "Historical revenue compound annual growth", ""),
        _metric_row("Quality", "FCF CAGR", fcf_cagr, "percent", "Historical free cash flow compound annual growth", ""),
        _metric_row("Quality", "FCF margin", fcf_margin, "percent", "Free cash flow / revenue", "Uses recent median margin."),
        _metric_row("Quality", "ROIC", roic, "percent", "NOPAT / invested capital", "NOPAT uses reported EBIT and an estimated tax rate."),
        _metric_row("Quality", "ROIC - WACC spread", roic - assumptions.wacc if roic is not None else None, "percent", "ROIC - WACC", ""),
        _metric_row("Quality", "EVA", nopat - invested_capital * assumptions.wacc if nopat is not None and invested_capital is not None else None, "currency", "NOPAT - invested capital x WACC", ""),
        _metric_row("Quality", "ROE", roe, "percent", "Net income / book equity", ""),
        _metric_row("Quality", "DuPont ROE", dupont_roe, "percent", "Net margin x asset turnover x equity multiplier", ""),
        _metric_row("Balance sheet", "Debt / market cap", _safe_divide(data.total_debt, market_cap), "percent", "Total debt / market cap", ""),
        _metric_row("Balance sheet", "Net cash / market cap", _safe_divide(data.cash_and_equivalents - data.total_debt, market_cap), "percent", "(Cash - debt) / market cap", ""),
        _metric_row("Balance sheet", "Altman Z-score", altman_z, "number", "1.2 WC/TA + 1.4 RE/TA + 3.3 EBIT/TA + 0.6 MVE/TL + sales/TA", "Classic public-company Z-score; less useful for financial companies."),
        _metric_row("Residual income", "Residual income value / share", _safe_divide(residual_income_value, data.shares_outstanding), "currency", "Book equity + PV residual income, divided by shares", "Uses WACC and terminal growth from the DCF assumptions."),
        _metric_row("Market risk", "Yahoo beta", data.beta, "number", "Reported beta from Yahoo Finance", ""),
    ]
    return pd.DataFrame(rows)


def analyze_portfolio(
    tickers: Iterable[str],
    weights: Iterable[float] | None,
    start,
    end,
    rebalancing_method: str = "Monthly rebalance",
    benchmark_ticker: str = "SPY",
    risk_free_rate: float = 0.04,
) -> PortfolioAnalyticsResult:
    # Portfolio analytics use monthly returns to match the Fama-French regression horizon.
    tickers = tuple(normalize_ticker(ticker) for ticker in tickers if str(ticker).strip())
    if not tickers:
        raise PortfolioAnalyticsError("Enter at least one ticker.")
    if pd.Timestamp(start) >= pd.Timestamp(end):
        raise PortfolioAnalyticsError("Start date must be before end date.")

    weight_series = normalize_portfolio_weights(tickers, weights)
    rebalancing_method = normalize_rebalancing_method(rebalancing_method)
    ticker_returns, source_notes = fetch_portfolio_monthly_returns(tickers, start, end)
    ticker_returns = ticker_returns.dropna(how="any")
    if ticker_returns.empty:
        raise PortfolioAnalyticsError("No overlapping monthly return history was available for the selected tickers.")

    portfolio_returns, ending_weights = build_portfolio_returns(ticker_returns, weight_series, rebalancing_method)
    portfolio_returns = portfolio_returns.dropna()
    benchmark_ticker = normalize_ticker(benchmark_ticker or "SPY")
    benchmark_returns = pd.Series(dtype=float, name=benchmark_ticker)
    try:
        benchmark_prices, benchmark_source = fetch_price_history(benchmark_ticker, start, end)
        benchmark_returns = benchmark_prices.resample("ME").last().dropna().pct_change().dropna().rename(benchmark_ticker)
        source_notes.append(f"{benchmark_ticker}: {benchmark_source}.")
    except FactorRegressionError:
        source_notes.append(f"{benchmark_ticker}: benchmark history was not available.")

    metrics = portfolio_performance_metrics(portfolio_returns, benchmark_returns, risk_free_rate, benchmark_ticker)
    cumulative_returns = build_cumulative_returns(portfolio_returns, benchmark_returns, benchmark_ticker)
    drawdowns = build_drawdowns(cumulative_returns)
    correlation = ticker_returns.corr()
    risk_contribution = portfolio_risk_contribution(ticker_returns, weight_series, ending_weights, rebalancing_method)
    frontier = efficient_frontier(ticker_returns, weight_series, risk_free_rate)

    return PortfolioAnalyticsResult(
        tickers=tickers,
        benchmark_ticker=benchmark_ticker,
        rebalancing_method=rebalancing_method,
        risk_free_rate=float(risk_free_rate),
        weights=weight_series,
        ending_weights=ending_weights,
        metrics=metrics,
        cumulative_returns=cumulative_returns,
        drawdowns=drawdowns,
        correlation=correlation,
        risk_contribution=risk_contribution,
        efficient_frontier=frontier,
        monthly_returns=ticker_returns.assign(portfolio_return=portfolio_returns),
        price_source_notes=tuple(source_notes),
    )


def portfolio_performance_metrics(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series | None,
    risk_free_rate: float,
    benchmark_name: str = "Benchmark",
) -> pd.DataFrame:
    portfolio_returns = pd.to_numeric(portfolio_returns, errors="coerce").dropna()
    if portfolio_returns.empty:
        raise PortfolioAnalyticsError("Portfolio returns were empty.")

    annual_return = _annualized_return(portfolio_returns)
    annual_volatility = _annualized_volatility(portfolio_returns)
    downside_volatility = _downside_volatility(portfolio_returns, risk_free_rate)
    max_drawdown = _max_drawdown(portfolio_returns)
    monthly_var = _historical_var(portfolio_returns, 0.95)
    monthly_cvar = _historical_cvar(portfolio_returns, 0.95)
    positive_months = float((portfolio_returns > 0).mean())

    beta = None
    jensen_alpha = None
    treynor = None
    information_ratio = None
    benchmark_return = None
    if benchmark_returns is not None and not benchmark_returns.empty:
        aligned = pd.concat([portfolio_returns.rename("portfolio"), benchmark_returns.rename("benchmark")], axis=1).dropna()
        if len(aligned) > 1 and aligned["benchmark"].var() > 0:
            beta = float(aligned["portfolio"].cov(aligned["benchmark"]) / aligned["benchmark"].var())
            benchmark_return = _annualized_return(aligned["benchmark"])
            jensen_alpha = annual_return - (risk_free_rate + beta * (benchmark_return - risk_free_rate))
            if beta != 0:
                treynor = (annual_return - risk_free_rate) / beta
            active_returns = aligned["portfolio"] - aligned["benchmark"]
            tracking_error = active_returns.std(ddof=1) * np.sqrt(12)
            if tracking_error > 0:
                information_ratio = active_returns.mean() * 12 / tracking_error

    rows = [
        _metric_row("Return", "CAGR", annual_return, "percent", "(Ending value / beginning value)^(1 / years) - 1", ""),
        _metric_row("Return", "Cumulative return", float((1 + portfolio_returns).prod() - 1), "percent", "Total compounded portfolio return", ""),
        _metric_row("Return", "Best month", float(portfolio_returns.max()), "percent", "Highest monthly return", ""),
        _metric_row("Return", "Worst month", float(portfolio_returns.min()), "percent", "Lowest monthly return", ""),
        _metric_row("Return", "Positive months", positive_months, "percent", "Share of months above 0%", ""),
        _metric_row("Risk", "Annualized volatility", annual_volatility, "percent", "Monthly return standard deviation x sqrt(12)", ""),
        _metric_row("Risk", "Downside volatility", downside_volatility, "percent", "Downside deviation x sqrt(12)", ""),
        _metric_row("Risk", "Maximum drawdown", max_drawdown, "percent", "Worst peak-to-trough decline", ""),
        _metric_row("Risk", "Monthly VaR 95%", monthly_var, "percent", "Historical 5th percentile loss", ""),
        _metric_row("Risk", "Monthly CVaR 95%", monthly_cvar, "percent", "Average loss beyond VaR", ""),
        _metric_row("Risk-adjusted", "Sharpe ratio", _safe_divide(annual_return - risk_free_rate, annual_volatility), "number", "(CAGR - risk-free rate) / volatility", ""),
        _metric_row("Risk-adjusted", "Sortino ratio", _safe_divide(annual_return - risk_free_rate, downside_volatility), "number", "(CAGR - risk-free rate) / downside volatility", ""),
        _metric_row("Benchmark", f"{benchmark_name} CAGR", benchmark_return, "percent", "Benchmark compounded annual return", ""),
        _metric_row("Benchmark", "Beta", beta, "number", "Covariance(portfolio, benchmark) / variance(benchmark)", ""),
        _metric_row("Benchmark", "Jensen alpha", jensen_alpha, "percent", "CAGR - [risk-free + beta x benchmark risk premium]", ""),
        _metric_row("Benchmark", "Treynor ratio", treynor, "number", "(CAGR - risk-free rate) / beta", ""),
        _metric_row("Benchmark", "Information ratio", information_ratio, "number", "Annualized active return / tracking error", ""),
        _metric_row("Data", "Monthly observations", float(len(portfolio_returns)), "number", "Number of monthly return observations", ""),
    ]
    return pd.DataFrame(rows)


def build_cumulative_returns(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series | None,
    benchmark_name: str,
) -> pd.DataFrame:
    curves = {"Portfolio": (1 + portfolio_returns.dropna()).cumprod() - 1}
    if benchmark_returns is not None and not benchmark_returns.empty:
        curves[benchmark_name] = (1 + benchmark_returns.dropna()).cumprod() - 1
    return pd.concat(curves, axis=1).dropna(how="all")


def build_drawdowns(cumulative_returns: pd.DataFrame) -> pd.DataFrame:
    wealth = 1 + cumulative_returns
    peaks = wealth.cummax()
    return wealth / peaks - 1


def portfolio_risk_contribution(
    ticker_returns: pd.DataFrame,
    starting_weights: pd.Series,
    ending_weights: pd.Series,
    rebalancing_method: str,
) -> pd.DataFrame:
    effective_weights = starting_weights if rebalancing_method == "Monthly rebalance" else ending_weights
    effective_weights = effective_weights.reindex(ticker_returns.columns).fillna(0.0)
    covariance = ticker_returns.cov() * 12
    portfolio_variance = float(effective_weights.to_numpy() @ covariance.to_numpy() @ effective_weights.to_numpy())
    if portfolio_variance <= 0:
        contribution = pd.Series(0.0, index=ticker_returns.columns)
    else:
        marginal = covariance.dot(effective_weights)
        contribution = effective_weights * marginal / portfolio_variance
    return pd.DataFrame(
        {
            "Ticker": ticker_returns.columns,
            "Weight": effective_weights.to_numpy(),
            "Risk contribution": contribution.reindex(ticker_returns.columns).to_numpy(),
        }
    )


def efficient_frontier(
    ticker_returns: pd.DataFrame,
    current_weights: pd.Series,
    risk_free_rate: float,
    samples: int = 1500,
) -> pd.DataFrame:
    # Use a fixed seed so the efficient-frontier chart is stable across app reruns.
    tickers = tuple(ticker_returns.columns)
    if not tickers:
        return pd.DataFrame()

    expected_returns = ticker_returns.mean() * 12
    covariance = ticker_returns.cov() * 12
    current = current_weights.reindex(tickers).fillna(0.0).to_numpy(dtype=float)
    current = current / current.sum() if current.sum() > 0 else np.repeat(1 / len(tickers), len(tickers))
    rng = np.random.default_rng(42)
    random_weights = rng.dirichlet(np.ones(len(tickers)), size=samples) if len(tickers) > 1 else current.reshape(1, -1)
    all_weights = np.vstack([current, random_weights])

    rows: list[dict[str, float | str]] = []
    for idx, weights in enumerate(all_weights):
        annual_return = float(weights @ expected_returns.to_numpy())
        annual_volatility = float(np.sqrt(max(weights @ covariance.to_numpy() @ weights, 0)))
        sharpe = _safe_divide(annual_return - risk_free_rate, annual_volatility)
        row: dict[str, float | str] = {
            "Portfolio": "Current" if idx == 0 else "Sample",
            "Annualized return": annual_return,
            "Annualized volatility": annual_volatility,
            "Sharpe ratio": sharpe,
        }
        for ticker, weight in zip(tickers, weights):
            row[f"{ticker} weight"] = float(weight)
        rows.append(row)

    frontier = pd.DataFrame(rows)
    sample_rows = frontier[frontier["Portfolio"].eq("Sample")]
    if not sample_rows.empty:
        max_sharpe_idx = sample_rows["Sharpe ratio"].astype(float).idxmax()
        min_vol_idx = sample_rows["Annualized volatility"].astype(float).idxmin()
        frontier.loc[max_sharpe_idx, "Portfolio"] = "Max Sharpe"
        frontier.loc[min_vol_idx, "Portfolio"] = "Min Volatility"
    return frontier


def portfolio_analytics_csv(result: PortfolioAnalyticsResult) -> bytes:
    parts = [
        "Metrics",
        result.metrics.to_csv(index=False),
        "\nStarting weights",
        result.weights.rename_axis("Ticker").reset_index().to_csv(index=False),
        "\nEnding weights",
        result.ending_weights.rename_axis("Ticker").reset_index().to_csv(index=False),
        "\nRisk contribution",
        result.risk_contribution.to_csv(index=False),
        "\nCorrelation",
        result.correlation.to_csv(),
        "\nEfficient frontier",
        result.efficient_frontier.to_csv(index=False),
        "\nMonthly returns",
        result.monthly_returns.to_csv(),
    ]
    return "\n".join(parts).encode("utf-8")


def analyze_fama_french_portfolio(
    tickers: Iterable[str],
    weights: Iterable[float] | None,
    start,
    end,
    model_name: str,
    rebalancing_method: str = "Monthly rebalance",
) -> FactorRegressionResult:
    tickers = tuple(normalize_ticker(ticker) for ticker in tickers if str(ticker).strip())
    if not tickers:
        raise FactorRegressionError("Enter at least one ticker.")
    if model_name not in FAMA_FRENCH_DATASETS:
        raise FactorRegressionError("Choose a supported Fama-French model.")

    weight_series = normalize_portfolio_weights(tickers, weights)
    rebalancing_method = normalize_rebalancing_method(rebalancing_method)
    ticker_returns, source_notes = fetch_portfolio_monthly_returns(tickers, start, end)
    ticker_returns = ticker_returns.dropna(how="any")
    if ticker_returns.empty:
        raise FactorRegressionError("No overlapping monthly return history was available for the selected tickers.")

    portfolio_returns, ending_weights = build_portfolio_returns(ticker_returns, weight_series, rebalancing_method)
    factors = fetch_fama_french_factors(model_name)
    result = run_factor_regression(portfolio_returns, factors, model_name)
    diagnostics = pd.concat(
        [
            pd.DataFrame([("Rebalancing method", rebalancing_method)], columns=["Metric", "Value"]),
            result.diagnostics,
        ],
        ignore_index=True,
    )

    return FactorRegressionResult(
        model_name=model_name,
        rebalancing_method=rebalancing_method,
        tickers=tickers,
        weights=weight_series,
        ending_weights=ending_weights,
        coefficients=result.coefficients,
        diagnostics=diagnostics,
        regression_data=result.regression_data,
        cumulative_returns=result.cumulative_returns,
        price_source_notes=tuple(source_notes),
        factor_source_note="Factor returns loaded from Kenneth French's data library.",
    )


def normalize_rebalancing_method(value: str) -> str:
    normalized = str(value).strip().lower().replace("_", " ").replace("-", " ")
    if normalized in {"monthly rebalance", "monthly rebalanced", "rebalance monthly"}:
        return "Monthly rebalance"
    if normalized in {"buy and hold", "buy hold", "buy-and-hold"}:
        return "Buy and hold"
    raise FactorRegressionError("Choose either Monthly rebalance or Buy and hold.")


def normalize_portfolio_weights(tickers: Iterable[str], weights: Iterable[float] | None) -> pd.Series:
    tickers = tuple(tickers)
    if weights is None:
        values = np.repeat(1 / len(tickers), len(tickers))
    else:
        values = np.array(list(weights), dtype=float)
        if len(values) == 0:
            values = np.repeat(1 / len(tickers), len(tickers))
        if len(values) != len(tickers):
            raise FactorRegressionError("The number of weights must match the number of tickers.")
        if np.any(values < 0):
            raise FactorRegressionError("Portfolio weights cannot be negative.")
        if values.sum() > 1.5:
            values = values / 100
        total = values.sum()
        if total <= 0:
            raise FactorRegressionError("Portfolio weights must sum to more than zero.")
        values = values / total

    return pd.Series(values, index=tickers, name="Weight")


def build_portfolio_returns(
    ticker_returns: pd.DataFrame,
    weights: pd.Series,
    rebalancing_method: str,
) -> tuple[pd.Series, pd.Series]:
    rebalancing_method = normalize_rebalancing_method(rebalancing_method)
    ticker_returns = ticker_returns.dropna(how="any")
    if ticker_returns.empty:
        raise FactorRegressionError("No overlapping monthly return history was available for the selected tickers.")

    weights = weights.reindex(ticker_returns.columns)
    if weights.isna().any():
        missing = ", ".join(weights[weights.isna()].index)
        raise FactorRegressionError(f"Missing portfolio weights for: {missing}.")

    if rebalancing_method == "Monthly rebalance":
        portfolio_returns = ticker_returns.dot(weights).rename("Portfolio return")
        return portfolio_returns, weights.rename("Ending weight")

    holding_values = (1 + ticker_returns).cumprod().mul(weights, axis=1)
    portfolio_values = holding_values.sum(axis=1)
    portfolio_returns = portfolio_values.pct_change()
    portfolio_returns.iloc[0] = portfolio_values.iloc[0] - 1
    portfolio_returns = portfolio_returns.rename("Portfolio return")
    ending_weights = (holding_values.iloc[-1] / portfolio_values.iloc[-1]).rename("Ending weight")
    return portfolio_returns, ending_weights


def fetch_portfolio_monthly_returns(tickers: Iterable[str], start, end) -> tuple[pd.DataFrame, list[str]]:
    returns: dict[str, pd.Series] = {}
    notes: list[str] = []
    for ticker in tickers:
        prices, source = fetch_price_history(ticker, start, end)
        monthly_prices = prices.resample("ME").last().dropna()
        monthly_returns = monthly_prices.pct_change().dropna()
        if monthly_returns.empty:
            raise FactorRegressionError(f"Not enough monthly price history was available for {ticker}.")
        returns[ticker] = monthly_returns.rename(ticker)
        notes.append(f"{ticker}: {source}.")

    return pd.DataFrame(returns), notes


def fetch_price_history(ticker: str, start, end) -> tuple[pd.Series, str]:
    import yfinance as yf

    ticker = normalize_ticker(ticker)
    yahoo_prices = _yahoo_chart_price_history(ticker, start, end)
    if not yahoo_prices.empty:
        return yahoo_prices, "Yahoo Finance chart API adjusted close"

    history = _safe_call(
        lambda: yf.Ticker(ticker).history(start=str(start), end=str(end), auto_adjust=True)
    )
    if isinstance(history, pd.DataFrame) and not history.empty and "Close" in history:
        prices = _clean_price_series(history["Close"])
        if not prices.empty:
            return prices, "Yahoo Finance adjusted close"

    stooq_prices = _stooq_price_history(ticker, start, end)
    if not stooq_prices.empty:
        return stooq_prices, "Stooq close price fallback"

    raise FactorRegressionError(f"No price history was available for {ticker}.")


def _yahoo_chart_price_history(ticker: str, start, end) -> pd.Series:
    import requests

    start_ts = int(pd.Timestamp(start).timestamp())
    end_ts = int((pd.Timestamp(end) + pd.Timedelta(days=1)).timestamp())
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        f"?period1={start_ts}&period2={end_ts}&interval=1d&events=history&includeAdjustedClose=true"
    )
    try:
        response = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        chart = response.json()["chart"]["result"][0]
        timestamps = chart.get("timestamp", [])
        indicators = chart.get("indicators", {})
        adjusted = indicators.get("adjclose", [{}])[0].get("adjclose")
        closes = adjusted or indicators.get("quote", [{}])[0].get("close")
    except Exception:
        return pd.Series(dtype=float)

    if not timestamps or not closes:
        return pd.Series(dtype=float)
    index = pd.to_datetime(timestamps, unit="s", errors="coerce")
    prices = pd.Series(closes, index=index)
    return _clean_price_series(prices)


def fetch_fama_french_factors(model_name: str) -> pd.DataFrame:
    import requests

    dataset = FAMA_FRENCH_DATASETS[model_name]
    response = requests.get(dataset["url"], timeout=30)
    response.raise_for_status()
    with ZipFile(BytesIO(response.content)) as archive:
        csv_name = archive.namelist()[0]
        text = archive.read(csv_name).decode("latin1")
    factors = parse_fama_french_csv(text)
    expected_columns = list(dataset["factors"]) + ["RF"]
    missing = [column for column in expected_columns if column not in factors]
    if missing:
        raise FactorRegressionError(f"Fama-French data is missing columns: {', '.join(missing)}.")
    return factors[expected_columns]


def parse_fama_french_csv(text: str) -> pd.DataFrame:
    lines = text.splitlines()
    try:
        header_index = next(index for index, line in enumerate(lines) if line.strip().startswith(",Mkt-RF"))
    except StopIteration as exc:
        raise FactorRegressionError("Could not find the Fama-French factor table in the downloaded file.") from exc

    table_lines: list[str] = []
    for line in lines[header_index:]:
        if table_lines and not line.strip():
            break
        first_cell = line.split(",", 1)[0].strip()
        if not table_lines or (first_cell.isdigit() and len(first_cell) == 6):
            table_lines.append(line)

    raw = pd.read_csv(StringIO("\n".join(table_lines)))
    date_column = raw.columns[0]
    raw = raw.rename(columns={date_column: "date"})
    raw = raw[raw["date"].astype(str).str.fullmatch(r"\d{6}")]
    raw["date"] = pd.to_datetime(raw["date"].astype(str) + "01", format="%Y%m%d") + pd.offsets.MonthEnd(0)

    factor_columns = [column for column in raw.columns if column != "date"]
    for column in factor_columns:
        raw[column] = pd.to_numeric(raw[column], errors="coerce") / 100
    return raw.set_index("date").sort_index().dropna(how="all")


def run_factor_regression(
    portfolio_returns: pd.Series,
    factors: pd.DataFrame,
    model_name: str,
) -> FactorRegressionResult:
    import statsmodels.api as sm

    factor_columns = list(FAMA_FRENCH_DATASETS[model_name]["factors"])
    joined = pd.concat([portfolio_returns.rename("portfolio_return"), factors], axis=1, join="inner").dropna()
    if len(joined) <= len(factor_columns) + 2:
        raise FactorRegressionError("Not enough overlapping monthly observations for a reliable regression.")

    y = joined["portfolio_return"] - joined["RF"]
    x = sm.add_constant(joined[factor_columns], has_constant="add")
    fitted = sm.OLS(y, x).fit()

    coefficients = pd.DataFrame(
        {
            "Factor": ["Alpha" if index == "const" else index for index in fitted.params.index],
            "Coefficient": fitted.params.to_numpy(),
            "t-stat": fitted.tvalues.to_numpy(),
            "p-value": fitted.pvalues.to_numpy(),
        }
    )
    coefficients["Annualized alpha"] = np.where(
        coefficients["Factor"].eq("Alpha"),
        (1 + coefficients["Coefficient"]) ** 12 - 1,
        np.nan,
    )

    regression_data = joined.copy()
    regression_data["excess_return"] = y
    regression_data["predicted_excess_return"] = fitted.fittedvalues
    regression_data["residual"] = fitted.resid
    regression_data["market_return"] = regression_data["Mkt-RF"] + regression_data["RF"]

    cumulative_returns = pd.DataFrame(
        {
            "Portfolio": (1 + regression_data["portfolio_return"]).cumprod() - 1,
            "Market proxy": (1 + regression_data["market_return"]).cumprod() - 1,
        }
    )

    monthly_alpha = float(fitted.params["const"])
    diagnostics = pd.DataFrame(
        [
            ("Model", model_name),
            ("Start month", regression_data.index.min().strftime("%Y-%m")),
            ("End month", regression_data.index.max().strftime("%Y-%m")),
            ("Monthly alpha", monthly_alpha),
            ("Annualized alpha", (1 + monthly_alpha) ** 12 - 1),
            ("R-squared", float(fitted.rsquared)),
            ("Adjusted R-squared", float(fitted.rsquared_adj)),
            ("Observations", int(fitted.nobs)),
        ],
        columns=["Metric", "Value"],
    )

    return FactorRegressionResult(
        model_name=model_name,
        rebalancing_method="Monthly rebalance",
        tickers=(),
        weights=pd.Series(dtype=float),
        ending_weights=pd.Series(dtype=float),
        coefficients=coefficients,
        diagnostics=diagnostics,
        regression_data=regression_data,
        cumulative_returns=cumulative_returns,
        price_source_notes=(),
        factor_source_note="Factor returns loaded from Kenneth French's data library.",
    )


def factor_regression_csv(result: FactorRegressionResult) -> bytes:
    parts = [
        "Diagnostics",
        result.diagnostics.to_csv(index=False),
        "\nStarting weights",
        result.weights.rename_axis("Ticker").reset_index().to_csv(index=False),
        "\nEnding weights",
        result.ending_weights.rename_axis("Ticker").reset_index().to_csv(index=False),
        "\nCoefficients",
        result.coefficients.to_csv(index=False),
        "\nRegression data",
        result.regression_data.to_csv(),
    ]
    return "\n".join(parts).encode("utf-8")


def validate_assumptions(assumptions: DCFAssumptions) -> None:
    if assumptions.projection_years < 1:
        raise DCFError("Projection years must be at least 1.")
    if assumptions.wacc <= 0:
        raise DCFError("WACC must be greater than 0%.")
    if assumptions.wacc <= assumptions.terminal_growth_rate:
        raise DCFError("WACC must be greater than terminal growth.")
    if assumptions.target_fcf_margin <= 0:
        raise DCFError("Target FCF margin must be greater than 0%.")


def is_probably_us_listed(data: TickerData) -> bool:
    us_exchanges = {
        "ASE",
        "AMEX",
        "BATS",
        "NCM",
        "NGM",
        "NMS",
        "NASDAQ",
        "NYQ",
        "NYSE",
        "PCX",
        "PNK",
    }
    exchange = data.exchange.upper()
    return data.currency.upper() == "USD" and (not exchange or exchange in us_exchanges)


def _find_sec_ticker_row(ticker_map: dict, ticker: str) -> dict | None:
    for row in ticker_map.values():
        sec_ticker = normalize_ticker(str(row.get("ticker", "")))
        if sec_ticker == ticker:
            return row
    return None


def _sec_first_series(companyfacts: dict, tags: Iterable[str], duration: bool) -> pd.Series:
    for tag in tags:
        series = _sec_fact_series(companyfacts, tag, "USD", duration=duration, annual_only=True)
        if not series.empty:
            return series
    return pd.Series(dtype=float)


def _sec_latest_value(companyfacts: dict, tags: Iterable[str], duration: bool) -> float | None:
    series = _sec_first_series(companyfacts, tags, duration=duration)
    if series.empty:
        return None
    return _clean_number(series.sort_index().iloc[-1])


def _sec_total_debt(companyfacts: dict) -> float | None:
    total_debt = _sec_latest_value(companyfacts, SEC_TOTAL_DEBT_TAGS, duration=False)
    if total_debt is not None:
        return total_debt

    component_series = [
        _sec_first_series(companyfacts, SEC_SHORT_BORROWINGS_TAGS, duration=False),
        _sec_first_series(companyfacts, SEC_CURRENT_LONG_TERM_DEBT_TAGS, duration=False),
        _sec_first_series(companyfacts, SEC_NONCURRENT_DEBT_TAGS, duration=False),
    ]
    component_series = [series for series in component_series if not series.empty]
    if not component_series:
        return None
    combined = pd.concat(component_series, axis=1).sum(axis=1, min_count=1).dropna()
    if combined.empty:
        return None
    return _clean_number(combined.sort_index().iloc[-1])


def _sec_latest_dei_value(companyfacts: dict, tag: str, unit: str) -> float | None:
    facts = companyfacts.get("facts", {}).get("dei", {})
    fact = facts.get(tag, {})
    units = fact.get("units", {})
    values = units.get(unit, [])
    if not values:
        return None

    df = pd.DataFrame(values)
    if df.empty or "val" not in df or "end" not in df:
        return None
    if "form" in df:
        df = df[df["form"].isin(["10-K", "10-K/A", "10-Q", "10-Q/A"])]
    df["end_dt"] = pd.to_datetime(df["end"], errors="coerce")
    df["filed_dt"] = pd.to_datetime(df["filed"], errors="coerce") if "filed" in df else pd.NaT
    df = df.dropna(subset=["end_dt"]).sort_values(["end_dt", "filed_dt"])
    if df.empty:
        return None
    return _clean_number(df.iloc[-1]["val"])


def _sec_fact_series(companyfacts: dict, tag: str, unit: str, duration: bool, annual_only: bool) -> pd.Series:
    facts = companyfacts.get("facts", {}).get("us-gaap", {})
    fact = facts.get(tag, {})
    units = fact.get("units", {})
    values = units.get(unit, [])
    if not values:
        return pd.Series(dtype=float)

    df = pd.DataFrame(values)
    if df.empty or "val" not in df or "end" not in df:
        return pd.Series(dtype=float)

    allowed_forms = ["10-K", "10-K/A"] if annual_only else ["10-K", "10-K/A", "10-Q", "10-Q/A"]
    if "form" in df:
        df = df[df["form"].isin(allowed_forms)]
    if annual_only and "fp" in df:
        df = df[df["fp"].eq("FY")]

    df["end_dt"] = pd.to_datetime(df["end"], errors="coerce")
    df["filed_dt"] = pd.to_datetime(df["filed"], errors="coerce") if "filed" in df else pd.NaT
    df = df.dropna(subset=["end_dt"])

    if duration:
        if "start" not in df:
            return pd.Series(dtype=float)
        df["start_dt"] = pd.to_datetime(df["start"], errors="coerce")
        df = df.dropna(subset=["start_dt"])
        duration_days = (df["end_dt"] - df["start_dt"]).dt.days
        df = df[(duration_days >= 250) & (duration_days <= 450)]

    df["val"] = pd.to_numeric(df["val"], errors="coerce")
    df = df.dropna(subset=["val"]).sort_values(["end_dt", "filed_dt"])
    if df.empty:
        return pd.Series(dtype=float)

    df = df.drop_duplicates(subset=["end_dt"], keep="last")
    series = pd.Series(df["val"].to_numpy(dtype=float), index=df["end_dt"])
    series.name = tag
    return series.sort_index()


def _statement_series(statement: pd.DataFrame | None, names: Iterable[str]) -> pd.Series:
    if statement is None or statement.empty:
        return pd.Series(dtype=float)

    index_lookup = {str(row).strip().lower(): row for row in statement.index}
    for name in names:
        row = index_lookup.get(name.strip().lower())
        if row is not None:
            series = pd.to_numeric(statement.loc[row], errors="coerce").dropna()
            series.name = name
            return series
    return pd.Series(dtype=float)


def _latest_statement_value(statement: pd.DataFrame | None, names: Iterable[str]) -> float | None:
    series = _statement_series(statement, names)
    if series.empty:
        return None
    series.index = pd.to_datetime(series.index, errors="coerce")
    series = series[series.index.notna()].sort_index()
    if series.empty:
        return None
    return _clean_number(series.iloc[-1])


def _combine_cfo_and_capex(cfo: float, capex: float) -> float:
    cfo_value = _clean_number(cfo)
    capex_value = _clean_number(capex)
    if cfo_value is None or capex_value is None:
        return np.nan
    if capex_value <= 0:
        return cfo_value + capex_value
    return cfo_value - capex_value


def _current_price(stock, fast_info, ticker: str) -> float | None:
    # Try lightweight quote fields first, then progressively fall back to recent price history.
    for key in ("last_price", "lastPrice", "regularMarketPrice", "previousClose"):
        value = _mapping_get(fast_info, key)
        cleaned = _clean_number(value)
        if cleaned and cleaned > 0:
            return cleaned

    history = _safe_call(lambda: stock.history(period="5d", interval="1d", auto_adjust=False))
    if isinstance(history, pd.DataFrame) and not history.empty and "Close" in history:
        cleaned = _clean_number(history["Close"].dropna().iloc[-1])
        if cleaned and cleaned > 0:
            return cleaned

    recent_prices = _yahoo_chart_price_history(
        ticker,
        pd.Timestamp.today().normalize() - pd.Timedelta(days=14),
        pd.Timestamp.today().normalize(),
    )
    if not recent_prices.empty:
        cleaned = _clean_number(recent_prices.iloc[-1])
        if cleaned and cleaned > 0:
            return cleaned
    return _stooq_current_price(ticker)


def _stooq_current_price(ticker: str) -> float | None:
    import requests

    stooq_symbol = ticker.replace("-", ".").lower()
    url = f"https://stooq.com/q/l/?s={stooq_symbol}.us&f=sd2t2ohlcv&h&e=csv"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except Exception:
        return None

    try:
        quote = pd.read_csv(StringIO(response.text))
    except Exception:
        return None
    if quote.empty or "Close" not in quote:
        return None
    return _clean_number(quote.iloc[0]["Close"])


def _stooq_price_history(ticker: str, start, end) -> pd.Series:
    import requests

    start_date = pd.to_datetime(start).strftime("%Y%m%d")
    end_date = pd.to_datetime(end).strftime("%Y%m%d")
    stooq_symbol = ticker.replace("-", ".").lower()
    url = f"https://stooq.com/q/d/l/?s={stooq_symbol}.us&d1={start_date}&d2={end_date}&i=d"
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        history = pd.read_csv(StringIO(response.text))
    except Exception:
        return pd.Series(dtype=float)

    if history.empty or "Date" not in history or "Close" not in history:
        return pd.Series(dtype=float)
    history["Date"] = pd.to_datetime(history["Date"], errors="coerce")
    prices = pd.Series(pd.to_numeric(history["Close"], errors="coerce").to_numpy(), index=history["Date"])
    return _clean_price_series(prices)


def _clean_price_series(series: pd.Series) -> pd.Series:
    prices = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    prices.index = pd.to_datetime(prices.index, errors="coerce")
    prices = prices[prices.index.notna()].sort_index()
    if getattr(prices.index, "tz", None) is not None:
        prices.index = prices.index.tz_localize(None)
    return prices[prices > 0]


def _number_from_mapping(mapping, keys: Iterable[str]) -> float | None:
    for key in keys:
        cleaned = _clean_number(_mapping_get(mapping, key))
        if cleaned is not None and cleaned > 0:
            return cleaned
    return None


def _mapping_get(mapping, key: str):
    try:
        return mapping.get(key)
    except Exception:
        try:
            return mapping[key]
        except Exception:
            return None


def _safe_call(func):
    try:
        return func()
    except Exception:
        return None


def _clean_number(value) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


def _safe_divide(numerator, denominator) -> float | None:
    numerator = _clean_number(numerator)
    denominator = _clean_number(denominator)
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def _metric_row(category: str, metric: str, value, value_type: str, formula: str, notes: str) -> dict[str, object]:
    return {
        "Category": category,
        "Metric": metric,
        "Value": _clean_number(value),
        "Type": value_type,
        "Formula": formula,
        "Notes": notes,
    }


def _annualized_return(returns: pd.Series) -> float:
    returns = pd.to_numeric(returns, errors="coerce").dropna()
    compounded = float((1 + returns).prod())
    years = len(returns) / 12
    if compounded <= 0 or years <= 0:
        return -1.0
    return compounded ** (1 / years) - 1


def _annualized_volatility(returns: pd.Series) -> float | None:
    returns = pd.to_numeric(returns, errors="coerce").dropna()
    if len(returns) < 2:
        return None
    return float(returns.std(ddof=1) * np.sqrt(12))


def _downside_volatility(returns: pd.Series, risk_free_rate: float) -> float | None:
    returns = pd.to_numeric(returns, errors="coerce").dropna()
    monthly_risk_free = (1 + risk_free_rate) ** (1 / 12) - 1
    downside = returns[returns < monthly_risk_free] - monthly_risk_free
    if len(downside) < 2:
        return None
    return float(np.sqrt((downside.pow(2).sum() / (len(downside) - 1)) * 12))


def _max_drawdown(returns: pd.Series) -> float:
    wealth = (1 + pd.to_numeric(returns, errors="coerce").dropna()).cumprod()
    return float((wealth / wealth.cummax() - 1).min())


def _historical_var(returns: pd.Series, confidence: float) -> float | None:
    returns = pd.to_numeric(returns, errors="coerce").dropna()
    if returns.empty:
        return None
    return float(-returns.quantile(1 - confidence))


def _historical_cvar(returns: pd.Series, confidence: float) -> float | None:
    returns = pd.to_numeric(returns, errors="coerce").dropna()
    if returns.empty:
        return None
    cutoff = returns.quantile(1 - confidence)
    tail = returns[returns <= cutoff]
    if tail.empty:
        return None
    return float(-tail.mean())


def _latest_positive(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    values = values[values > 0]
    if values.empty:
        return None
    return float(values.iloc[-1])


def _recent_median(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    values = values[(values > -1) & (values < 1)]
    if values.empty:
        return None
    return float(values.tail(3).median())


def _historical_cagr(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    values = values[values > 0]
    if len(values) < 2:
        return None
    first = float(values.iloc[0])
    last = float(values.iloc[-1])
    periods = len(values) - 1
    if first <= 0 or periods <= 0:
        return None
    return (last / first) ** (1 / periods) - 1


def _spread_points(center: float, step: float, count: int, floor: float) -> list[float]:
    midpoint = count // 2
    values = [max(floor, center + (idx - midpoint) * step) for idx in range(count)]
    return sorted(set(round(value, 4) for value in values))


def _format_percent(value: float) -> str:
    return f"{value:.1%}"


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)
