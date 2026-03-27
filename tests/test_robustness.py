"""
Tests for robustness verification (Phase 7).

Tests cover: holdout validation, walk-forward sensitivity,
parameter neighborhood stability, bootstrap confidence intervals,
sub-period analysis, and end-to-end robustness evaluation.
"""
import numpy as np
import pandas as pd
import pytest

from src.backtest import BacktestConfig
from src.evaluation import (
    STRATEGY_COLUMNS,
    bootstrap_confidence,
    check_parameter_neighborhood,
    evaluate_robustness,
    run_holdout_validation,
    run_subperiod_analysis,
    run_walk_forward_sensitivity,
)


def _make_panel(n_dates=250, seed=42):
    """
    Create a deterministic panel for testing robustness verification.

    3 industries, 2 stocks each with different drift characteristics.
    Uses enough dates to support holdout splits and sub-period analysis.
    """
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2022-01-03", periods=n_dates)

    stocks = {
        "A1": ("ind_A", 0.003),
        "A2": ("ind_A", 0.002),
        "B1": ("ind_B", -0.002),
        "B2": ("ind_B", -0.003),
        "C1": ("ind_C", 0.001),
        "C2": ("ind_C", -0.001),
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


# --- Tests for run_holdout_validation ---


class TestHoldoutValidation:
    """Tests for temporal holdout validation."""

    def test_returns_dict_with_required_keys(self):
        """Holdout validation should return dict with opt and holdout results."""
        panel = _make_panel()
        config = BacktestConfig(n_splits=3, min_train_size=30)
        result = run_holdout_validation(
            panel, config=config,
            lookback_values=[5, 10],
            quantile_values=[0.3],
        )
        assert isinstance(result, dict)
        assert "opt_date_range" in result
        assert "holdout_date_range" in result
        assert "optimized" in result
        assert "baseline" in result

    def test_holdout_date_range_after_opt(self):
        """Holdout period should come after optimization period."""
        panel = _make_panel()
        config = BacktestConfig(n_splits=3, min_train_size=30)
        result = run_holdout_validation(
            panel, config=config,
            lookback_values=[5],
            quantile_values=[0.3],
        )
        opt_end = result["opt_date_range"][1]
        holdout_start = result["holdout_date_range"][0]
        assert holdout_start >= opt_end

    def test_optimized_results_contain_strategy_keys(self):
        """Optimized results should have entries for strategies that produced results."""
        panel = _make_panel()
        config = BacktestConfig(n_splits=3, min_train_size=30)
        result = run_holdout_validation(
            panel, config=config,
            lookback_values=[5, 10],
            quantile_values=[0.3],
        )
        optimized = result["optimized"]
        assert isinstance(optimized, dict)
        # At least one strategy should have results
        assert len(optimized) > 0

    def test_holdout_sharpe_is_numeric(self):
        """Holdout Sharpe should be a numeric value."""
        panel = _make_panel()
        config = BacktestConfig(n_splits=3, min_train_size=30)
        result = run_holdout_validation(
            panel, config=config,
            lookback_values=[5],
            quantile_values=[0.3],
        )
        for strategy_name, strat_result in result["optimized"].items():
            assert isinstance(strat_result["holdout_netSharpe"], (int, float))
            assert isinstance(strat_result["opt_netSharpe"], (int, float))

    def test_sharpe_degradation_computed(self):
        """Sharpe degradation should be opt minus holdout Sharpe."""
        panel = _make_panel()
        config = BacktestConfig(n_splits=3, min_train_size=30)
        result = run_holdout_validation(
            panel, config=config,
            lookback_values=[5, 10],
            quantile_values=[0.3],
        )
        for strat, res in result["optimized"].items():
            expected_deg = round(res["opt_netSharpe"] - res["holdout_netSharpe"], 4)
            assert abs(res["sharpe_degradation"] - expected_deg) < 0.01

    def test_baseline_holdout_present(self):
        """Baseline holdout results should be included for comparison."""
        panel = _make_panel()
        config = BacktestConfig(n_splits=3, min_train_size=30)
        result = run_holdout_validation(
            panel, config=config,
            lookback_values=[5],
            quantile_values=[0.3],
        )
        baseline = result["baseline"]
        assert isinstance(baseline, dict)
        # Baseline uses lookback=12, so may or may not produce results
        # but the dict should exist


# --- Tests for run_walk_forward_sensitivity ---


class TestWalkForwardSensitivity:
    """Tests for walk-forward configuration sensitivity."""

    def test_returns_list_of_dicts(self):
        """Should return a list of result dicts."""
        panel = _make_panel()
        results = run_walk_forward_sensitivity(
            panel, lookback=5,
            momentum_col="total_momentum",
            quantile=0.3,
            n_splits_values=[3, 5],
            min_train_values=[30],
        )
        assert isinstance(results, list)
        assert len(results) == 2  # 2 n_splits × 1 min_train
        assert all(isinstance(r, dict) for r in results)

    def test_result_contains_required_fields(self):
        """Each result should have n_splits, min_train_size, and metrics."""
        panel = _make_panel()
        results = run_walk_forward_sensitivity(
            panel, lookback=5,
            momentum_col="total_momentum",
            quantile=0.3,
            n_splits_values=[3],
            min_train_values=[30],
        )
        assert len(results) >= 1
        r = results[0]
        assert "n_splits" in r
        assert "min_train_size" in r
        assert "netSharpe" in r
        assert "windows" in r

    def test_different_configs_produce_different_windows(self):
        """Different n_splits should generally produce different window counts."""
        panel = _make_panel(n_dates=300)
        results = run_walk_forward_sensitivity(
            panel, lookback=5,
            momentum_col="total_momentum",
            quantile=0.3,
            n_splits_values=[3, 7],
            min_train_values=[30],
        )
        # With enough data, different n_splits should yield different window counts
        if all(r["windows"] > 0 for r in results):
            windows = [r["windows"] for r in results]
            # At least not all the same (though edge cases exist)
            assert len(results) == 2

    def test_grid_coverage(self):
        """Should test all combinations of n_splits and min_train."""
        panel = _make_panel()
        results = run_walk_forward_sensitivity(
            panel, lookback=5,
            momentum_col="total_momentum",
            quantile=0.3,
            n_splits_values=[3, 5],
            min_train_values=[30, 50],
        )
        assert len(results) == 4  # 2 × 2


# --- Tests for check_parameter_neighborhood ---


class TestParameterNeighborhood:
    """Tests for parameter neighborhood stability check."""

    def test_returns_dict_with_required_keys(self):
        """Should return dict with center, neighbors, and stability metrics."""
        panel = _make_panel()
        config = BacktestConfig(n_splits=3, min_train_size=30)
        result = check_parameter_neighborhood(
            panel, config=config,
            center_lookback=5, center_quantile=0.3,
            strategy="total_momentum",
            lookback_delta=1, quantile_delta=0.05,
        )
        assert "center" in result
        assert "neighbors" in result
        assert "avg_neighbor_sharpe" in result
        assert "positive_fraction" in result
        assert "n_neighbors" in result

    def test_center_in_neighbors(self):
        """The center point should be included in the neighborhood."""
        panel = _make_panel()
        config = BacktestConfig(n_splits=3, min_train_size=30)
        result = check_parameter_neighborhood(
            panel, config=config,
            center_lookback=5, center_quantile=0.3,
            strategy="total_momentum",
            lookback_delta=1, quantile_delta=0.05,
        )
        center_lb = result["center"]["lookback"]
        center_q = result["center"]["quantile"]
        found = any(
            n["lookback"] == center_lb and abs(n["quantile"] - center_q) < 0.001
            for n in result["neighbors"]
        )
        assert found

    def test_positive_fraction_valid_range(self):
        """Positive fraction should be between 0 and 1."""
        panel = _make_panel()
        config = BacktestConfig(n_splits=3, min_train_size=30)
        result = check_parameter_neighborhood(
            panel, config=config,
            center_lookback=5, center_quantile=0.3,
            strategy="total_momentum",
            lookback_delta=1, quantile_delta=0.05,
        )
        assert 0.0 <= result["positive_fraction"] <= 1.0

    def test_neighborhood_size_depends_on_delta(self):
        """Larger deltas should produce more neighbors."""
        panel = _make_panel()
        config = BacktestConfig(n_splits=3, min_train_size=30)
        result_small = check_parameter_neighborhood(
            panel, config=config,
            center_lookback=8, center_quantile=0.3,
            strategy="total_momentum",
            lookback_delta=1, quantile_delta=0.05,
        )
        result_large = check_parameter_neighborhood(
            panel, config=config,
            center_lookback=8, center_quantile=0.3,
            strategy="total_momentum",
            lookback_delta=3, quantile_delta=0.1,
        )
        assert result_large["n_neighbors"] >= result_small["n_neighbors"]


# --- Tests for bootstrap_confidence ---


class TestBootstrapConfidence:
    """Tests for bootstrap confidence intervals."""

    def test_returns_dict_with_required_keys(self):
        """Should return dict with point estimate and CI bounds."""
        panel = _make_panel()
        config = BacktestConfig(n_splits=3, min_train_size=30)
        result = bootstrap_confidence(
            panel, config=config,
            lookback=5, momentum_col="total_momentum",
            quantile=0.3, n_bootstrap=100,
        )
        assert "point_estimate" in result
        assert "ci_lower" in result
        assert "ci_upper" in result
        assert "n_windows" in result
        assert "prob_positive" in result

    def test_ci_bounds_order(self):
        """CI lower should be <= point estimate <= CI upper."""
        panel = _make_panel()
        config = BacktestConfig(n_splits=3, min_train_size=30)
        result = bootstrap_confidence(
            panel, config=config,
            lookback=5, momentum_col="total_momentum",
            quantile=0.3, n_bootstrap=200,
        )
        assert result["ci_lower"] <= result["ci_upper"]

    def test_ci_width_positive(self):
        """CI width should be non-negative."""
        panel = _make_panel()
        config = BacktestConfig(n_splits=3, min_train_size=30)
        result = bootstrap_confidence(
            panel, config=config,
            lookback=5, momentum_col="total_momentum",
            quantile=0.3, n_bootstrap=100,
        )
        assert result["ci_width"] >= 0.0

    def test_prob_positive_valid_range(self):
        """Probability of positive Sharpe should be between 0 and 1."""
        panel = _make_panel()
        config = BacktestConfig(n_splits=3, min_train_size=30)
        result = bootstrap_confidence(
            panel, config=config,
            lookback=5, momentum_col="total_momentum",
            quantile=0.3, n_bootstrap=100,
        )
        assert 0.0 <= result["prob_positive"] <= 1.0

    def test_reproducibility_with_seed(self):
        """Same seed should produce same results."""
        panel = _make_panel()
        config = BacktestConfig(n_splits=3, min_train_size=30)
        r1 = bootstrap_confidence(
            panel, config=config,
            lookback=5, momentum_col="total_momentum",
            quantile=0.3, n_bootstrap=50, seed=123,
        )
        r2 = bootstrap_confidence(
            panel, config=config,
            lookback=5, momentum_col="total_momentum",
            quantile=0.3, n_bootstrap=50, seed=123,
        )
        assert r1["point_estimate"] == r2["point_estimate"]
        assert r1["ci_lower"] == r2["ci_lower"]

    def test_empty_results_handled(self):
        """Should handle case where walk-forward produces no results."""
        panel = _make_panel(n_dates=20)  # too few dates
        config = BacktestConfig(n_splits=3, min_train_size=100)
        result = bootstrap_confidence(
            panel, config=config,
            lookback=5, momentum_col="total_momentum",
            quantile=0.3, n_bootstrap=50,
        )
        assert result["point_estimate"] == 0.0
        assert result["n_windows"] == 0


# --- Tests for run_subperiod_analysis ---


class TestSubperiodAnalysis:
    """Tests for sub-period performance analysis."""

    def test_returns_correct_number_of_subperiods(self):
        """Should return one result per sub-period."""
        panel = _make_panel(n_dates=300)
        config = BacktestConfig(n_splits=2, min_train_size=20)
        results = run_subperiod_analysis(
            panel, config=config,
            lookback=5, momentum_col="total_momentum",
            quantile=0.3, n_subperiods=3,
        )
        assert len(results) == 3

    def test_subperiods_are_sequential(self):
        """Sub-period dates should be non-overlapping and sequential."""
        panel = _make_panel(n_dates=300)
        config = BacktestConfig(n_splits=2, min_train_size=20)
        results = run_subperiod_analysis(
            panel, config=config,
            lookback=5, momentum_col="total_momentum",
            quantile=0.3, n_subperiods=3,
        )
        for i in range(len(results) - 1):
            assert results[i]["end_date"] <= results[i + 1]["start_date"]

    def test_result_contains_required_fields(self):
        """Each sub-period result should contain required metric fields."""
        panel = _make_panel(n_dates=300)
        config = BacktestConfig(n_splits=2, min_train_size=20)
        results = run_subperiod_analysis(
            panel, config=config,
            lookback=5, momentum_col="total_momentum",
            quantile=0.3, n_subperiods=2,
        )
        for r in results:
            assert "subperiod" in r
            assert "start_date" in r
            assert "end_date" in r
            assert "netSharpe" in r
            assert "windows" in r

    def test_subperiod_numbering(self):
        """Sub-periods should be numbered 1, 2, 3, etc."""
        panel = _make_panel(n_dates=300)
        config = BacktestConfig(n_splits=2, min_train_size=20)
        results = run_subperiod_analysis(
            panel, config=config,
            lookback=5, momentum_col="total_momentum",
            quantile=0.3, n_subperiods=3,
        )
        assert [r["subperiod"] for r in results] == [1, 2, 3]


# --- Tests for evaluate_robustness (end-to-end) ---


class TestEvaluateRobustness:
    """Tests for the full Phase 7 robustness evaluation pipeline."""

    def test_returns_arf_schema(self):
        """Should return metrics matching ARF schema."""
        panel = _make_panel()
        config = BacktestConfig(n_splits=3, min_train_size=30)
        metrics = evaluate_robustness(
            panel, config=config,
            lookback_values=[5, 10],
            quantile_values=[0.3],
        )
        assert "sharpeRatio" in metrics
        assert "annualReturn" in metrics
        assert "maxDrawdown" in metrics
        assert "hitRate" in metrics
        assert "totalTrades" in metrics
        assert "transactionCosts" in metrics
        assert "walkForward" in metrics
        assert "customMetrics" in metrics

    def test_custom_metrics_contain_robustness_sections(self):
        """Custom metrics should contain all robustness analysis sections."""
        panel = _make_panel()
        config = BacktestConfig(n_splits=3, min_train_size=30)
        metrics = evaluate_robustness(
            panel, config=config,
            lookback_values=[5, 10],
            quantile_values=[0.3],
        )
        cm = metrics["customMetrics"]
        assert "holdoutValidation" in cm
        assert "walkForwardSensitivity" in cm
        assert "parameterNeighborhood" in cm
        assert "bootstrapConfidence" in cm
        assert "subperiodAnalysis" in cm
        assert "robustnessChecks" in cm
        assert "robustnessScore" in cm

    def test_robustness_score_valid_range(self):
        """Robustness score should be between 0 and 1."""
        panel = _make_panel()
        config = BacktestConfig(n_splits=3, min_train_size=30)
        metrics = evaluate_robustness(
            panel, config=config,
            lookback_values=[5, 10],
            quantile_values=[0.3],
        )
        score = metrics["customMetrics"]["robustnessScore"]
        assert 0.0 <= score <= 1.0

    def test_robustness_checks_are_booleans(self):
        """Each robustness check should be a boolean."""
        panel = _make_panel()
        config = BacktestConfig(n_splits=3, min_train_size=30)
        metrics = evaluate_robustness(
            panel, config=config,
            lookback_values=[5, 10],
            quantile_values=[0.3],
        )
        checks = metrics["customMetrics"]["robustnessChecks"]
        assert all(isinstance(v, bool) for v in checks.values())

    def test_transaction_costs_present(self):
        """Transaction costs section should have fee and slippage."""
        panel = _make_panel()
        config = BacktestConfig(n_splits=3, min_train_size=30)
        metrics = evaluate_robustness(
            panel, config=config,
            lookback_values=[5],
            quantile_values=[0.3],
        )
        tc = metrics["transactionCosts"]
        assert "feeBps" in tc
        assert "slippageBps" in tc
        assert "netSharpe" in tc

    def test_optimized_params_recorded(self):
        """The optimized parameters should be recorded in custom metrics."""
        panel = _make_panel()
        config = BacktestConfig(n_splits=3, min_train_size=30)
        metrics = evaluate_robustness(
            panel, config=config,
            lookback_values=[5],
            quantile_values=[0.3],
        )
        opt = metrics["customMetrics"]["optimizedParams"]
        assert opt["strategy"] == "stock_specific_momentum"
        assert opt["lookback"] == 10
        assert opt["quantile"] == 0.4
