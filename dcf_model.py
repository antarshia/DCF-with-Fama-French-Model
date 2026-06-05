from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO, StringIO
import os
from typing import Iterable

import numpy as np
import pandas as pd


REVENUE_ROWS = (
    "Total Revenue",
    "Revenue",
    "Operating Revenue",
)

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

SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
SEC_DEFAULT_USER_AGENT = "dcf-valuation-streamlit-app/1.0 contact@example.com"

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


class DCFError(ValueError):
    """Raised when the valuation cannot be calculated from the available data."""


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
