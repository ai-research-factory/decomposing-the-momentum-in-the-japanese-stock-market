"""
Unit tests for the momentum decomposition algorithm.
"""
import numpy as np
import pandas as pd
import pytest
from src.decomposition import decompose_momentum


def _make_test_data(n_dates=60, seed=42):
    """
    Create a small deterministic test dataset.

    3 industries, 2 stocks each, with controlled price dynamics:
    - Industry A: uptrend
    - Industry B: downtrend
    - Industry C: flat/mean-reverting
    """
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_dates)

    stocks = {
        "A1": ("ind_A", 0.002),   # uptrend
        "A2": ("ind_A", 0.0015),  # uptrend (slower)
        "B1": ("ind_B", -0.001),  # downtrend
        "B2": ("ind_B", -0.0015), # downtrend (faster)
        "C1": ("ind_C", 0.0),     # flat
        "C2": ("ind_C", 0.0),     # flat
    }

    rows = []
    for stock_id, (industry_id, drift) in stocks.items():
        price = 100.0
        for d in dates:
            rows.append({
                "date": d,
                "stock_id": stock_id,
                "industry_id": industry_id,
                "price": price,
            })
            price *= 1 + drift + rng.normal(0, 0.01)

    return pd.DataFrame(rows)


class TestDecomposeMomentum:
    """Tests for decompose_momentum function."""

    def test_output_columns(self):
        """Output should contain required momentum columns."""
        df = _make_test_data()
        result = decompose_momentum(df, lookback=12)
        expected_cols = {
            "date", "stock_id", "industry_id",
            "total_momentum", "industry_momentum", "stock_specific_momentum",
        }
        assert expected_cols.issubset(set(result.columns))

    def test_output_has_all_stocks(self):
        """Output should contain all input stocks."""
        df = _make_test_data()
        result = decompose_momentum(df, lookback=12)
        assert set(result["stock_id"].unique()) == set(df["stock_id"].unique())

    def test_output_has_all_dates(self):
        """Output should contain all input dates."""
        df = _make_test_data()
        result = decompose_momentum(df, lookback=12)
        assert set(result["date"].unique()) == set(pd.to_datetime(df["date"]).unique())

    def test_momentum_values_not_all_nan(self):
        """After lookback period, momentum values should be computed."""
        df = _make_test_data(n_dates=60)
        result = decompose_momentum(df, lookback=12)
        # After lookback+1 periods, we should have non-NaN momentum
        for col in ["total_momentum", "industry_momentum", "stock_specific_momentum"]:
            valid = result[col].dropna()
            assert len(valid) > 0, f"{col} has no non-NaN values"

    def test_uptrend_has_positive_total_momentum(self):
        """Stocks in an uptrend industry should have positive total momentum."""
        df = _make_test_data(n_dates=60, seed=42)
        result = decompose_momentum(df, lookback=12)
        # Filter to uptrend stocks at the end of the period
        latest = result.groupby("stock_id").last()
        a1_mom = latest.loc["A1", "total_momentum"]
        assert not np.isnan(a1_mom), "Total momentum for A1 should not be NaN"
        # With drift=0.002 per day over 60 days, expect positive momentum
        assert a1_mom > 0, f"Expected positive total momentum for uptrend stock, got {a1_mom}"

    def test_downtrend_has_negative_total_momentum(self):
        """Stocks in a downtrend industry should have negative total momentum."""
        df = _make_test_data(n_dates=60, seed=42)
        result = decompose_momentum(df, lookback=12)
        latest = result.groupby("stock_id").last()
        b2_mom = latest.loc["B2", "total_momentum"]
        assert not np.isnan(b2_mom), "Total momentum for B2 should not be NaN"
        assert b2_mom < 0, f"Expected negative total momentum for downtrend stock, got {b2_mom}"

    def test_industry_momentum_same_within_industry(self):
        """Stocks in the same industry should have the same industry momentum."""
        df = _make_test_data(n_dates=40)
        result = decompose_momentum(df, lookback=12)
        # For each date, stocks in the same industry should have equal industry_momentum
        for date in result["date"].unique():
            day_data = result[result["date"] == date]
            for ind_id in day_data["industry_id"].unique():
                ind_data = day_data[day_data["industry_id"] == ind_id]
                moms = ind_data["industry_momentum"].dropna()
                if len(moms) > 1:
                    assert moms.nunique() == 1, (
                        f"Industry momentum should be equal within {ind_id} on {date}, "
                        f"got {moms.values}"
                    )

    def test_lookback_parameter(self):
        """Different lookback values should produce different results."""
        df = _make_test_data(n_dates=60)
        r1 = decompose_momentum(df, lookback=10)
        r2 = decompose_momentum(df, lookback=20)
        # More NaN values expected with longer lookback
        nan_count_1 = r1["total_momentum"].isna().sum()
        nan_count_2 = r2["total_momentum"].isna().sum()
        assert nan_count_2 >= nan_count_1

    def test_empty_dataframe(self):
        """Should handle empty input gracefully."""
        df = pd.DataFrame(columns=["date", "stock_id", "industry_id", "price"])
        result = decompose_momentum(df, lookback=12)
        assert len(result) == 0

    def test_single_stock(self):
        """Should work with a single stock."""
        rng = np.random.RandomState(0)
        dates = pd.bdate_range("2020-01-01", periods=30)
        prices = 100 * np.cumprod(1 + rng.normal(0.001, 0.01, len(dates)))
        df = pd.DataFrame({
            "date": dates,
            "stock_id": "SOLO",
            "industry_id": "ind_X",
            "price": prices,
        })
        result = decompose_momentum(df, lookback=10)
        assert len(result) == len(df)
        # For a single stock, industry_return == stock_return
        # so industry_momentum should equal total_momentum
        valid = result.dropna(subset=["total_momentum", "industry_momentum"])
        if len(valid) > 0:
            np.testing.assert_allclose(
                valid["total_momentum"].values,
                valid["industry_momentum"].values,
                atol=1e-10,
            )
