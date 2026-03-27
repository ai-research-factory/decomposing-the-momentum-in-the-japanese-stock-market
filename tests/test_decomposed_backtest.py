"""
Tests for the decomposed momentum backtest (Phase 4).

Tests cover: factor correlations, decomposed backtest execution,
strategy comparison, and end-to-end evaluation with all 3 strategies.
"""
import numpy as np
import pandas as pd
import pytest

from src.backtest import BacktestConfig, BacktestResult
from src.evaluation import (
    STRATEGY_COLUMNS,
    compare_strategies,
    compute_factor_correlations,
    evaluate_decomposed_strategies,
    run_decomposed_backtest,
)


def _make_panel(n_dates=200, seed=42):
    """
    Create a deterministic panel for testing decomposed backtest.

    3 industries, 2 stocks each:
    - ind_A: strong uptrend (positive momentum)
    - ind_B: strong downtrend (negative momentum)
    - ind_C: flat / mean-reverting
    """
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2022-01-03", periods=n_dates)

    stocks = {
        "A1": ("ind_A", 0.003),
        "A2": ("ind_A", 0.002),
        "B1": ("ind_B", -0.002),
        "B2": ("ind_B", -0.003),
        "C1": ("ind_C", 0.0),
        "C2": ("ind_C", 0.0),
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
            price *= 1 + drift + rng.normal(0, 0.005)

    return pd.DataFrame(rows)


class TestComputeFactorCorrelations:
    """Tests for compute_factor_correlations."""

    def test_returns_all_correlation_keys(self):
        """Should return all 3 correlation pairs."""
        panel = _make_panel()
        corr = compute_factor_correlations(panel, lookback=12)
        assert "industry_vs_stock_specific" in corr
        assert "total_vs_industry" in corr
        assert "total_vs_stock_specific" in corr

    def test_correlations_are_finite(self):
        """All correlation values should be finite numbers."""
        panel = _make_panel()
        corr = compute_factor_correlations(panel, lookback=12)
        for key, val in corr.items():
            assert np.isfinite(val), f"{key} is not finite: {val}"

    def test_correlations_in_range(self):
        """Correlations should be between -1 and 1."""
        panel = _make_panel()
        corr = compute_factor_correlations(panel, lookback=12)
        for key, val in corr.items():
            assert -1.0 <= val <= 1.0, f"{key} out of range: {val}"

    def test_total_vs_industry_positive(self):
        """Total and industry momentum should be positively correlated."""
        panel = _make_panel(n_dates=200)
        corr = compute_factor_correlations(panel, lookback=12)
        assert corr["total_vs_industry"] > 0, (
            f"Expected positive total-industry corr, got {corr['total_vs_industry']}"
        )


class TestRunDecomposedBacktest:
    """Tests for run_decomposed_backtest."""

    def test_returns_all_three_strategies(self):
        """Should return results for total, industry, and stock-specific strategies."""
        panel = _make_panel()
        config = BacktestConfig(n_splits=3, min_train_size=50)
        results = run_decomposed_backtest(panel, config=config, lookback=12)
        for col in STRATEGY_COLUMNS:
            assert col in results, f"Missing strategy: {col}"

    def test_each_strategy_has_results(self):
        """Each strategy should produce at least one window."""
        panel = _make_panel()
        config = BacktestConfig(n_splits=3, min_train_size=50)
        results = run_decomposed_backtest(panel, config=config, lookback=12)
        for col in STRATEGY_COLUMNS:
            assert len(results[col]) >= 1, f"{col} produced no results"

    def test_results_are_backtest_results(self):
        """Each result should be a BacktestResult instance."""
        panel = _make_panel()
        config = BacktestConfig(n_splits=3, min_train_size=50)
        results = run_decomposed_backtest(panel, config=config, lookback=12)
        for col in STRATEGY_COLUMNS:
            for r in results[col]:
                assert isinstance(r, BacktestResult)

    def test_strategies_produce_different_sharpes(self):
        """Different strategies should generally produce different Sharpe ratios."""
        panel = _make_panel()
        config = BacktestConfig(n_splits=3, min_train_size=50)
        results = run_decomposed_backtest(panel, config=config, lookback=12)
        sharpes = {}
        for col in STRATEGY_COLUMNS:
            if results[col]:
                sharpes[col] = np.mean([r.gross_sharpe for r in results[col]])
        # At least two strategies should differ
        vals = list(sharpes.values())
        assert len(set(round(v, 4) for v in vals)) > 1, (
            f"All strategies have same Sharpe: {sharpes}"
        )


class TestCompareStrategies:
    """Tests for compare_strategies."""

    def test_returns_metrics_for_all_strategies(self):
        """Should return metrics dict for each strategy."""
        panel = _make_panel()
        config = BacktestConfig(n_splits=3, min_train_size=50)
        all_results = run_decomposed_backtest(panel, config=config, lookback=12)
        comparison = compare_strategies(all_results, config)
        for col in STRATEGY_COLUMNS:
            assert col in comparison

    def test_each_strategy_has_arf_schema(self):
        """Each strategy's metrics should follow the ARF schema."""
        panel = _make_panel()
        config = BacktestConfig(n_splits=3, min_train_size=50)
        all_results = run_decomposed_backtest(panel, config=config, lookback=12)
        comparison = compare_strategies(all_results, config)
        for col in STRATEGY_COLUMNS:
            m = comparison[col]
            assert "sharpeRatio" in m
            assert "annualReturn" in m
            assert "maxDrawdown" in m
            assert "transactionCosts" in m
            assert "walkForward" in m


class TestEvaluateDecomposedStrategies:
    """Tests for evaluate_decomposed_strategies (end-to-end)."""

    def test_returns_valid_arf_schema(self):
        """Output should match the ARF metrics.json schema."""
        panel = _make_panel()
        config = BacktestConfig(n_splits=3, min_train_size=50)
        metrics = evaluate_decomposed_strategies(panel, config=config, lookback=12)

        assert "sharpeRatio" in metrics
        assert "annualReturn" in metrics
        assert "maxDrawdown" in metrics
        assert "hitRate" in metrics
        assert "totalTrades" in metrics
        assert "transactionCosts" in metrics
        assert "walkForward" in metrics
        assert "customMetrics" in metrics

    def test_custom_metrics_has_strategy_comparison(self):
        """customMetrics should contain strategy comparison data."""
        panel = _make_panel()
        config = BacktestConfig(n_splits=3, min_train_size=50)
        metrics = evaluate_decomposed_strategies(panel, config=config, lookback=12)
        cm = metrics["customMetrics"]
        assert "strategyComparison" in cm
        assert "bestStrategy" in cm
        assert "factorCorrelations" in cm

    def test_strategy_comparison_has_all_strategies(self):
        """Strategy comparison should include all 3 strategies."""
        panel = _make_panel()
        config = BacktestConfig(n_splits=3, min_train_size=50)
        metrics = evaluate_decomposed_strategies(panel, config=config, lookback=12)
        sc = metrics["customMetrics"]["strategyComparison"]
        for col in STRATEGY_COLUMNS:
            assert col in sc
            assert "grossSharpe" in sc[col]
            assert "netSharpe" in sc[col]
            assert "annualReturn" in sc[col]
            assert "maxDrawdown" in sc[col]

    def test_best_strategy_is_valid(self):
        """Best strategy should be one of the 3 strategies."""
        panel = _make_panel()
        config = BacktestConfig(n_splits=3, min_train_size=50)
        metrics = evaluate_decomposed_strategies(panel, config=config, lookback=12)
        best = metrics["customMetrics"]["bestStrategy"]
        assert best in STRATEGY_COLUMNS

    def test_factor_correlations_present(self):
        """Factor correlations should be present and valid."""
        panel = _make_panel()
        config = BacktestConfig(n_splits=3, min_train_size=50)
        metrics = evaluate_decomposed_strategies(panel, config=config, lookback=12)
        corr = metrics["customMetrics"]["factorCorrelations"]
        assert "industry_vs_stock_specific" in corr
        assert "total_vs_industry" in corr
        assert "total_vs_stock_specific" in corr

    def test_transaction_costs_structure(self):
        """transactionCosts should have feeBps, slippageBps, netSharpe."""
        panel = _make_panel()
        config = BacktestConfig(n_splits=3, min_train_size=50)
        metrics = evaluate_decomposed_strategies(panel, config=config, lookback=12)
        tc = metrics["transactionCosts"]
        assert "feeBps" in tc
        assert "slippageBps" in tc
        assert "netSharpe" in tc

    def test_walk_forward_has_windows(self):
        """walkForward should report at least 1 window."""
        panel = _make_panel()
        config = BacktestConfig(n_splits=3, min_train_size=50)
        metrics = evaluate_decomposed_strategies(panel, config=config, lookback=12)
        assert metrics["walkForward"]["windows"] >= 1
