"""
Tests for the data pipeline (src/data_fetcher.py).

Tests cover: date alignment, forward-fill, outlier detection,
panel validation, and end-to-end build_panel on cached data.
"""
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import patch

from src.data_fetcher import (
    _align_dates,
    _forward_fill_prices,
    _detect_outliers,
    _validate_panel,
    build_panel,
    DataQualityReport,
    STOCK_INDUSTRY_MAP,
)


def _make_raw_panel(n_dates=30, n_stocks=4, seed=42):
    """Create a small raw panel for testing preprocessing steps."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2023-01-02", periods=n_dates)
    industries = {"S1": "ind_A", "S2": "ind_A", "S3": "ind_B", "S4": "ind_B"}
    stocks = list(industries.keys())[:n_stocks]

    rows = []
    for stock in stocks:
        price = 100.0
        for d in dates:
            rows.append({
                "date": d,
                "stock_id": stock,
                "industry_id": industries[stock],
                "price": price,
            })
            price *= 1 + rng.normal(0.001, 0.01)
    return pd.DataFrame(rows)


class TestAlignDates:
    """Tests for _align_dates."""

    def test_fills_missing_dates(self):
        """Stocks missing some dates should get NaN rows after alignment."""
        panel = _make_raw_panel(n_dates=10)
        # Remove some dates for S1
        panel = panel[~((panel["stock_id"] == "S1") & (panel["date"] == panel["date"].iloc[2]))]
        assert len(panel) < 40  # one row removed

        aligned = _align_dates(panel)
        # After alignment, every stock should have all dates
        for stock in aligned["stock_id"].unique():
            stock_dates = aligned[aligned["stock_id"] == stock]["date"]
            assert len(stock_dates) == aligned["date"].nunique()

    def test_preserves_existing_data(self):
        """Existing price values should not change."""
        panel = _make_raw_panel(n_dates=5)
        aligned = _align_dates(panel)
        merged = panel.merge(
            aligned, on=["date", "stock_id"], suffixes=("_orig", "_aligned")
        )
        np.testing.assert_array_equal(
            merged["price_orig"].values, merged["price_aligned"].values
        )

    def test_industry_propagated(self):
        """Industry ID should be present for all rows after alignment."""
        panel = _make_raw_panel(n_dates=5)
        aligned = _align_dates(panel)
        assert aligned["industry_id"].isna().sum() == 0


class TestForwardFill:
    """Tests for _forward_fill_prices."""

    def test_fills_gaps(self):
        """NaN prices should be forward-filled within each stock."""
        panel = _make_raw_panel(n_dates=10)
        # Introduce a NaN gap
        panel.loc[panel.index[5], "price"] = np.nan
        filled, count = _forward_fill_prices(panel)
        assert count == 1
        assert filled["price"].isna().sum() < panel.shape[0]

    def test_no_fill_needed(self):
        """If no NaN, filled count should be 0."""
        panel = _make_raw_panel(n_dates=10)
        filled, count = _forward_fill_prices(panel)
        assert count == 0

    def test_leading_nans_remain(self):
        """Forward-fill cannot fill leading NaNs (no prior value)."""
        panel = _make_raw_panel(n_dates=10)
        # Set first row of each stock to NaN
        for stock in panel["stock_id"].unique():
            idx = panel[panel["stock_id"] == stock].index[0]
            panel.loc[idx, "price"] = np.nan
        filled, _ = _forward_fill_prices(panel)
        # Leading NaNs should remain
        for stock in filled["stock_id"].unique():
            first = filled[filled["stock_id"] == stock].iloc[0]
            assert np.isnan(first["price"])


class TestOutlierDetection:
    """Tests for _detect_outliers."""

    def test_adds_outlier_column(self):
        """Should add is_outlier boolean column."""
        panel = _make_raw_panel(n_dates=30)
        result = _detect_outliers(panel)
        assert "is_outlier" in result.columns
        assert result["is_outlier"].dtype == bool

    def test_extreme_return_flagged(self):
        """An extreme price jump should be flagged as outlier."""
        panel = _make_raw_panel(n_dates=30)
        # Inject an extreme price spike
        stock_mask = panel["stock_id"] == "S1"
        stock_idx = panel[stock_mask].index
        panel.loc[stock_idx[15], "price"] = panel.loc[stock_idx[14], "price"] * 3.0
        result = _detect_outliers(panel, z_threshold=3.0)
        assert result["is_outlier"].sum() > 0

    def test_normal_data_no_outliers(self):
        """Normal data with small noise should have no/very few outliers."""
        panel = _make_raw_panel(n_dates=100, seed=0)
        result = _detect_outliers(panel, z_threshold=5.0)
        # With normal noise, outliers should be very rare
        assert result["is_outlier"].sum() < 5


class TestValidatePanel:
    """Tests for _validate_panel."""

    def test_valid_panel_passes(self):
        """A well-formed panel should pass validation."""
        panel = _make_raw_panel(n_dates=10)
        _validate_panel(panel)  # Should not raise

    def test_missing_column_raises(self):
        """Missing required columns should raise ValueError."""
        panel = _make_raw_panel(n_dates=10).drop(columns=["price"])
        with pytest.raises(ValueError, match="missing required columns"):
            _validate_panel(panel)

    def test_empty_panel_raises(self):
        """Empty panel should raise ValueError."""
        panel = pd.DataFrame(columns=["date", "stock_id", "industry_id", "price"])
        with pytest.raises(ValueError, match="empty"):
            _validate_panel(panel)


class TestDataQualityReport:
    """Tests for the DataQualityReport dataclass."""

    def test_to_dict(self):
        """to_dict should produce a serializable dict."""
        report = DataQualityReport(
            total_stocks_requested=10,
            total_stocks_fetched=9,
            failed_tickers=["X.T"],
            n_industries=3,
        )
        d = report.to_dict()
        assert d["total_stocks_requested"] == 10
        assert d["failed_tickers"] == ["X.T"]


class TestBuildPanelIntegration:
    """Integration test: build_panel with a small stock map using cached data."""

    def test_build_panel_small(self, tmp_path):
        """
        Build panel with 4 stocks, using mock fetch that returns synthetic data.
        Verifies the full pipeline end-to-end.
        """
        small_map = {"S1": "ind_A", "S2": "ind_A", "S3": "ind_B", "S4": "ind_B"}
        rng = np.random.RandomState(42)
        dates = pd.bdate_range("2023-01-02", periods=50)

        def mock_fetch(ticker, interval, period, cache_dir=None):
            price = 100.0
            prices = []
            for _ in dates:
                prices.append(price)
                price *= 1 + rng.normal(0.001, 0.01)
            return pd.DataFrame({
                "timestamp": dates,
                "close": prices,
                "open": prices,
                "high": prices,
                "low": prices,
                "volume": [1000] * len(dates),
            })

        with patch("src.data_fetcher.fetch_stock_data", side_effect=mock_fetch):
            panel, report = build_panel(
                stock_map=small_map,
                cache_dir=tmp_path,
                rate_limit_sec=0,
            )

        assert report.total_stocks_fetched == 4
        assert report.n_industries == 2
        assert {"date", "stock_id", "industry_id", "price", "is_outlier"}.issubset(panel.columns)
        assert panel["price"].isna().sum() == 0
        assert len(panel) > 0

    def test_build_panel_drops_thin_industry(self, tmp_path):
        """Industries with fewer than MIN_STOCKS_PER_INDUSTRY stocks are dropped."""
        # Only 1 stock in ind_C → should be dropped
        small_map = {"S1": "ind_A", "S2": "ind_A", "S3": "ind_C"}
        dates = pd.bdate_range("2023-01-02", periods=20)

        def mock_fetch(ticker, interval, period, cache_dir=None):
            return pd.DataFrame({
                "timestamp": dates,
                "close": np.linspace(100, 110, len(dates)),
                "open": [100] * len(dates),
                "high": [110] * len(dates),
                "low": [90] * len(dates),
                "volume": [1000] * len(dates),
            })

        with patch("src.data_fetcher.fetch_stock_data", side_effect=mock_fetch):
            panel, report = build_panel(
                stock_map=small_map,
                cache_dir=tmp_path,
                rate_limit_sec=0,
            )

        assert "ind_C" not in panel["industry_id"].values
        assert report.n_industries == 1

    def test_stock_industry_map_has_enough_stocks(self):
        """The default STOCK_INDUSTRY_MAP should have at least 2 stocks per industry."""
        from collections import Counter
        counts = Counter(STOCK_INDUSTRY_MAP.values())
        for industry, count in counts.items():
            assert count >= 2, f"Industry {industry} has only {count} stock(s)"
