import math

import pandas as pd

from dcf_model import (
    DCFAssumptions,
    TickerData,
    build_financial_history_from_sec,
    calculate_dcf,
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
