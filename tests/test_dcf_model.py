import math
from dataclasses import replace

import pandas as pd

from dcf_model import (
    DCFAssumptions,
    TickerData,
    build_portfolio_returns,
    build_financial_history_from_sec,
    calculate_dcf,
    calculate_stock_metrics,
    portfolio_performance_metrics,
    portfolio_risk_contribution,
    normalize_portfolio_weights,
    parse_fama_french_csv,
    run_factor_regression,
    sensitivity_table,
)


def sample_ticker_data():
    return TickerData(
        ticker="TEST",
        company_name="Test Company",
        currency="USD",
        exchange="NMS",
        current_price=100.0,
        shares_outstanding=100_000_000,
        cash_and_equivalents=500_000_000,
        total_debt=200_000_000,
        financial_history=pd.DataFrame(
            {
                "fiscal_year": [2021, 2022, 2023],
                "revenue": [900_000_000, 1_000_000_000, 1_100_000_000],
                "operating_cash_flow": [150_000_000, 170_000_000, 190_000_000],
                "capex": [-25_000_000, -30_000_000, -35_000_000],
                "free_cash_flow": [125_000_000, 140_000_000, 155_000_000],
                "fcf_margin": [0.1389, 0.14, 0.1409],
            }
        ),
        source_notes=(),
    )


def test_calculate_dcf_returns_intrinsic_value_per_share():
    assumptions = DCFAssumptions(
        growth_rate_stage_1=0.08,
        growth_rate_stage_2=0.04,
        target_fcf_margin=0.15,
        wacc=0.09,
        terminal_growth_rate=0.025,
        projection_years=10,
        margin_of_safety=0.25,
    )

    result = calculate_dcf(sample_ticker_data(), assumptions)

    assert result.intrinsic_value_per_share > 0
    assert math.isclose(result.buy_below_price, result.intrinsic_value_per_share * 0.75)
    assert len(result.projection) == 10


def test_sensitivity_table_builds_wacc_terminal_grid():
    assumptions = DCFAssumptions(
        growth_rate_stage_1=0.08,
        growth_rate_stage_2=0.04,
        target_fcf_margin=0.15,
        wacc=0.09,
        terminal_growth_rate=0.025,
        projection_years=10,
        margin_of_safety=0.25,
    )

    table = sensitivity_table(sample_ticker_data(), assumptions)

    assert table.shape == (5, 5)
    assert table.notna().any().any()


def test_build_financial_history_from_sec_companyfacts():
    companyfacts = {
        "facts": {
            "us-gaap": {
                "Revenues": {
                    "units": {
                        "USD": [
                            {
                                "start": "2022-01-01",
                                "end": "2022-12-31",
                                "val": 1_000_000,
                                "form": "10-K",
                                "fp": "FY",
                                "filed": "2023-02-01",
                            },
                            {
                                "start": "2023-01-01",
                                "end": "2023-12-31",
                                "val": 1_200_000,
                                "form": "10-K",
                                "fp": "FY",
                                "filed": "2024-02-01",
                            },
                        ]
                    }
                },
                "NetCashProvidedByUsedInOperatingActivities": {
                    "units": {
                        "USD": [
                            {
                                "start": "2022-01-01",
                                "end": "2022-12-31",
                                "val": 180_000,
                                "form": "10-K",
                                "fp": "FY",
                                "filed": "2023-02-01",
                            },
                            {
                                "start": "2023-01-01",
                                "end": "2023-12-31",
                                "val": 210_000,
                                "form": "10-K",
                                "fp": "FY",
                                "filed": "2024-02-01",
                            },
                        ]
                    }
                },
                "PaymentsToAcquirePropertyPlantAndEquipment": {
                    "units": {
                        "USD": [
                            {
                                "start": "2022-01-01",
                                "end": "2022-12-31",
                                "val": 30_000,
                                "form": "10-K",
                                "fp": "FY",
                                "filed": "2023-02-01",
                            },
                            {
                                "start": "2023-01-01",
                                "end": "2023-12-31",
                                "val": 40_000,
                                "form": "10-K",
                                "fp": "FY",
                                "filed": "2024-02-01",
                            },
                        ]
                    }
                },
            }
        }
    }

    history = build_financial_history_from_sec(companyfacts)

    assert list(history["fiscal_year"]) == [2022, 2023]
    assert list(history["free_cash_flow"]) == [150_000, 170_000]


def test_normalize_portfolio_weights_accepts_percent_inputs():
    weights = normalize_portfolio_weights(("AAPL", "MSFT", "NVDA"), (40, 40, 20))

    assert math.isclose(weights.sum(), 1.0)
    assert math.isclose(weights["AAPL"], 0.4)
    assert math.isclose(weights["NVDA"], 0.2)


def test_monthly_rebalance_portfolio_returns_keep_weights_fixed():
    index = pd.date_range("2024-01-31", periods=3, freq="ME")
    ticker_returns = pd.DataFrame(
        {
            "AAA": [0.10, 0.00, 0.10],
            "BBB": [0.00, 0.10, 0.00],
        },
        index=index,
    )
    weights = normalize_portfolio_weights(("AAA", "BBB"), (50, 50))

    portfolio_returns, ending_weights = build_portfolio_returns(ticker_returns, weights, "Monthly rebalance")

    assert list(portfolio_returns.round(4)) == [0.05, 0.05, 0.05]
    assert math.isclose(ending_weights["AAA"], 0.5)
    assert math.isclose(ending_weights["BBB"], 0.5)


def test_buy_and_hold_portfolio_returns_allow_weight_drift():
    index = pd.date_range("2024-01-31", periods=3, freq="ME")
    ticker_returns = pd.DataFrame(
        {
            "AAA": [0.10, 0.00, 0.10],
            "BBB": [0.00, 0.10, 0.00],
        },
        index=index,
    )
    weights = normalize_portfolio_weights(("AAA", "BBB"), (50, 50))

    portfolio_returns, ending_weights = build_portfolio_returns(ticker_returns, weights, "Buy and hold")

    assert math.isclose(portfolio_returns.iloc[0], 0.05)
    assert math.isclose(portfolio_returns.iloc[1], 0.05 / 1.05)
    assert math.isclose(portfolio_returns.iloc[2], 0.055 / 1.10)
    assert ending_weights["AAA"] > 0.5
    assert ending_weights["BBB"] < 0.5


def test_calculate_stock_metrics_includes_valuation_quality_and_risk_formulas():
    assumptions = DCFAssumptions(
        growth_rate_stage_1=0.08,
        growth_rate_stage_2=0.04,
        target_fcf_margin=0.15,
        wacc=0.09,
        terminal_growth_rate=0.025,
        projection_years=10,
        margin_of_safety=0.25,
    )
    data = replace(
        sample_ticker_data(),
        market_cap=10_000_000_000,
        net_income=1_000_000_000,
        pretax_income=1_250_000_000,
        tax_provision=250_000_000,
        ebit=1_200_000_000,
        ebitda=1_500_000_000,
        total_equity=5_000_000_000,
        total_assets=8_000_000_000,
        total_liabilities=3_000_000_000,
        current_assets=2_000_000_000,
        current_liabilities=1_000_000_000,
        retained_earnings=2_000_000_000,
        trailing_eps=10.0,
        forward_eps=12.0,
        dividend_rate=2.0,
        beta=1.1,
    )

    result = calculate_dcf(data, assumptions)
    metrics = calculate_stock_metrics(data, result, assumptions).set_index("Metric")

    assert math.isclose(metrics.loc["FCF yield", "Value"], 155_000_000 / 10_000_000_000)
    assert math.isclose(metrics.loc["P/E", "Value"], 10.0)
    assert metrics.loc["ROIC", "Value"] > 0
    assert metrics.loc["Altman Z-score", "Value"] > 0
    assert metrics.loc["DDM value / share", "Value"] > 0


def test_portfolio_performance_metrics_calculate_risk_adjusted_values():
    index = pd.date_range("2024-01-31", periods=8, freq="ME")
    portfolio_returns = pd.Series([0.02, -0.01, 0.03, 0.01, -0.02, 0.04, 0.00, 0.02], index=index)
    benchmark_returns = pd.Series([0.01, -0.02, 0.02, 0.01, -0.01, 0.03, 0.00, 0.01], index=index)

    metrics = portfolio_performance_metrics(portfolio_returns, benchmark_returns, 0.04, "SPY").set_index("Metric")

    assert metrics.loc["CAGR", "Value"] > 0
    assert metrics.loc["Annualized volatility", "Value"] > 0
    assert metrics.loc["Maximum drawdown", "Value"] < 0
    assert metrics.loc["Sharpe ratio", "Value"] is not None
    assert metrics.loc["Beta", "Value"] > 0


def test_portfolio_risk_contribution_sums_to_full_portfolio_risk():
    index = pd.date_range("2024-01-31", periods=6, freq="ME")
    ticker_returns = pd.DataFrame(
        {
            "AAA": [0.02, -0.01, 0.03, 0.01, -0.02, 0.04],
            "BBB": [0.01, 0.00, 0.01, -0.01, 0.02, 0.01],
            "CCC": [-0.01, 0.02, 0.00, 0.03, 0.01, -0.02],
        },
        index=index,
    )
    weights = normalize_portfolio_weights(("AAA", "BBB", "CCC"), (50, 30, 20))

    contribution = portfolio_risk_contribution(ticker_returns, weights, weights, "Monthly rebalance")

    assert math.isclose(contribution["Risk contribution"].sum(), 1.0)


def test_parse_fama_french_csv_converts_percent_to_decimal():
    sample = """This file was created using a test database.

,Mkt-RF,SMB,HML,RMW,CMA,RF
202401,    1.00,   -2.00,    3.00,    0.50,   -0.25,    0.40
202402,   -1.50,    0.25,   -0.75,    0.10,    0.20,    0.38

Annual Factors:
"""

    factors = parse_fama_french_csv(sample)

    assert list(factors.columns) == ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]
    assert math.isclose(factors.iloc[0]["Mkt-RF"], 0.01)
    assert math.isclose(factors.iloc[1]["RF"], 0.0038)


def test_run_factor_regression_estimates_synthetic_betas():
    index = pd.date_range("2020-01-31", periods=24, freq="ME")
    factors = pd.DataFrame(
        {
            "Mkt-RF": [0.01, -0.02, 0.03, 0.00] * 6,
            "SMB": [0.02, 0.01, -0.01, -0.02] * 6,
            "HML": [-0.01, 0.01, 0.02, -0.02] * 6,
            "RF": [0.001] * 24,
        },
        index=index,
    )
    portfolio_returns = factors["RF"] + 0.002 + 1.25 * factors["Mkt-RF"] - 0.4 * factors["SMB"] + 0.3 * factors["HML"]

    result = run_factor_regression(portfolio_returns, factors, "Fama-French 3 Factor")
    coefficients = result.coefficients.set_index("Factor")["Coefficient"]

    assert math.isclose(coefficients["Alpha"], 0.002, abs_tol=1e-12)
    assert math.isclose(coefficients["Mkt-RF"], 1.25, abs_tol=1e-12)
    assert math.isclose(coefficients["SMB"], -0.4, abs_tol=1e-12)
