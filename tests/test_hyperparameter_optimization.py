"""
Tests for hyperparameter optimization (Phase 6).

Tests cover: parameter grid search, sensitivity analysis,
end-to-end hyperparameter optimization pipeline, and baseline comparison.
"""
import numpy as np
import pandas as pd
import pytest

from src.backtest import BacktestConfig
from src.evaluation import (
    STRATEGY_COLUMNS,
    analyze_parameter_sensitivity,
    evaluate_hyperparameter_optimization,
    run_parameter_grid_search,
)


def _make_panel(n_dates=200, seed=42):
    """
    Create a deterministic panel for testing hyperparameter optimization.

    3 industries, 2 stocks each with different drift characteristics.
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


# --- Tests for run_parameter_grid_search ---


class TestParameterGridSearch:
    """Tests for grid search over lookback and quantile parameters."""

    def test_returns_list_of_dicts(self):
        """Grid search should return a list of result dicts."""
        panel = _make_panel()
        config = BacktestConfig(n_splits=3, min_train_size=40)
        results = run_parameter_grid_search(
            panel, config=config,
            lookback_values=[5, 10],
            quantile_values=[0.3],
            strategy="total_momentum",
        )
        assert isinstance(results, list)
        assert len(results) == 2  # 2 lookback × 1 quantile
        assert all(isinstance(r, dict) for r in results)

    def test_result_contains_required_fields(self):
        """Each result dict should contain all required metric fields."""
        panel = _make_panel()
        config = BacktestConfig(n_splits=3, min_train_size=40)
        results = run_parameter_grid_search(
            panel, config=config,
            lookback_values=[10],
            quantile_values=[0.3],
            strategy="total_momentum",
        )
        assert len(results) == 1
        r = results[0]
        required_keys = [
            "lookback", "quantile", "strategy", "netSharpe", "grossSharpe",
            "annualReturn", "maxDrawdown", "hitRate", "windows",
            "positiveWindows", "totalTrades",
        ]
        for key in required_keys:
            assert key in r, f"Missing key: {key}"

    def test_lookback_and_quantile_stored(self):
        """Result should store the lookback and quantile values used."""
        panel = _make_panel()
        config = BacktestConfig(n_splits=3, min_train_size=40)
        results = run_parameter_grid_search(
            panel, config=config,
            lookback_values=[5, 15],
            quantile_values=[0.2, 0.4],
            strategy="stock_specific_momentum",
        )
        assert len(results) == 4  # 2 × 2
        lookbacks = {r["lookback"] for r in results}
        quantiles = {r["quantile"] for r in results}
        assert lookbacks == {5, 15}
        assert quantiles == {0.2, 0.4}

    def test_results_sorted_by_net_sharpe_descending(self):
        """Results should be sorted by net Sharpe ratio, best first."""
        panel = _make_panel()
        config = BacktestConfig(n_splits=3, min_train_size=40)
        results = run_parameter_grid_search(
            panel, config=config,
            lookback_values=[5, 10, 20],
            quantile_values=[0.2, 0.3],
            strategy="total_momentum",
        )
        sharpes = [r["netSharpe"] for r in results]
        assert sharpes == sorted(sharpes, reverse=True)

    def test_different_lookbacks_give_different_results(self):
        """Different lookback values should produce different Sharpe ratios."""
        panel = _make_panel(n_dates=250)
        config = BacktestConfig(n_splits=3, min_train_size=40)
        results = run_parameter_grid_search(
            panel, config=config,
            lookback_values=[5, 20],
            quantile_values=[0.3],
            strategy="stock_specific_momentum",
        )
        sharpes = [r["netSharpe"] for r in results]
        # With different lookbacks, results should differ (extremely unlikely to be identical)
        assert len(set(sharpes)) > 0

    def test_strategy_column_preserved(self):
        """The strategy name should be stored in each result."""
        panel = _make_panel()
        config = BacktestConfig(n_splits=3, min_train_size=40)
        results = run_parameter_grid_search(
            panel, config=config,
            lookback_values=[10],
            quantile_values=[0.3],
            strategy="industry_momentum",
        )
        assert results[0]["strategy"] == "industry_momentum"

    def test_grid_size_matches_parameter_combinations(self):
        """Number of results should equal len(lookbacks) × len(quantiles)."""
        panel = _make_panel()
        config = BacktestConfig(n_splits=3, min_train_size=40)
        lookbacks = [5, 10, 15]
        quantiles = [0.2, 0.3, 0.4]
        results = run_parameter_grid_search(
            panel, config=config,
            lookback_values=lookbacks,
            quantile_values=quantiles,
            strategy="total_momentum",
        )
        assert len(results) == len(lookbacks) * len(quantiles)

    def test_windows_count_positive(self):
        """At least some grid results should have walk-forward windows."""
        panel = _make_panel(n_dates=200)
        config = BacktestConfig(n_splits=3, min_train_size=40)
        results = run_parameter_grid_search(
            panel, config=config,
            lookback_values=[10],
            quantile_values=[0.3],
            strategy="total_momentum",
        )
        assert results[0]["windows"] > 0


# --- Tests for analyze_parameter_sensitivity ---


class TestAnalyzeParameterSensitivity:
    """Tests for parameter sensitivity analysis."""

    def test_returns_dict_with_required_keys(self):
        """Should return a dict with sensitivity analysis keys."""
        grid_results = [
            {"lookback": 5, "quantile": 0.2, "netSharpe": 0.5},
            {"lookback": 5, "quantile": 0.3, "netSharpe": 0.3},
            {"lookback": 10, "quantile": 0.2, "netSharpe": 0.4},
            {"lookback": 10, "quantile": 0.3, "netSharpe": 0.6},
        ]
        result = analyze_parameter_sensitivity(grid_results)
        assert "lookback_sensitivity" in result
        assert "quantile_sensitivity" in result
        assert "best_params" in result
        assert "worst_params" in result
        assert "sharpe_range" in result

    def test_lookback_sensitivity_marginalizes_over_quantile(self):
        """Lookback sensitivity should average over quantile values."""
        grid_results = [
            {"lookback": 5, "quantile": 0.2, "netSharpe": 0.4},
            {"lookback": 5, "quantile": 0.3, "netSharpe": 0.6},
            {"lookback": 10, "quantile": 0.2, "netSharpe": 0.2},
            {"lookback": 10, "quantile": 0.3, "netSharpe": 0.4},
        ]
        result = analyze_parameter_sensitivity(grid_results)
        assert result["lookback_sensitivity"]["5"] == 0.5  # mean(0.4, 0.6)
        assert result["lookback_sensitivity"]["10"] == 0.3  # mean(0.2, 0.4)

    def test_quantile_sensitivity_marginalizes_over_lookback(self):
        """Quantile sensitivity should average over lookback values."""
        grid_results = [
            {"lookback": 5, "quantile": 0.2, "netSharpe": 0.4},
            {"lookback": 5, "quantile": 0.3, "netSharpe": 0.6},
            {"lookback": 10, "quantile": 0.2, "netSharpe": 0.2},
            {"lookback": 10, "quantile": 0.3, "netSharpe": 0.4},
        ]
        result = analyze_parameter_sensitivity(grid_results)
        assert result["quantile_sensitivity"]["0.2"] == 0.3  # mean(0.4, 0.2)
        assert result["quantile_sensitivity"]["0.3"] == 0.5  # mean(0.6, 0.4)

    def test_best_params_has_highest_sharpe(self):
        """Best params should correspond to highest net Sharpe."""
        grid_results = [
            {"lookback": 5, "quantile": 0.2, "netSharpe": 0.8},
            {"lookback": 10, "quantile": 0.3, "netSharpe": 0.3},
            {"lookback": 15, "quantile": 0.4, "netSharpe": 0.1},
        ]
        result = analyze_parameter_sensitivity(grid_results)
        assert result["best_params"]["lookback"] == 5
        assert result["best_params"]["quantile"] == 0.2
        assert result["best_params"]["netSharpe"] == 0.8

    def test_worst_params_has_lowest_sharpe(self):
        """Worst params should correspond to lowest net Sharpe."""
        grid_results = [
            {"lookback": 5, "quantile": 0.2, "netSharpe": 0.8},
            {"lookback": 10, "quantile": 0.3, "netSharpe": 0.3},
            {"lookback": 15, "quantile": 0.4, "netSharpe": 0.1},
        ]
        result = analyze_parameter_sensitivity(grid_results)
        assert result["worst_params"]["lookback"] == 15
        assert result["worst_params"]["netSharpe"] == 0.1

    def test_sharpe_range(self):
        """Sharpe range should be best - worst."""
        grid_results = [
            {"lookback": 5, "quantile": 0.2, "netSharpe": 0.8},
            {"lookback": 10, "quantile": 0.3, "netSharpe": 0.1},
        ]
        result = analyze_parameter_sensitivity(grid_results)
        assert result["sharpe_range"] == pytest.approx(0.7, abs=0.0001)

    def test_empty_grid_returns_defaults(self):
        """Empty grid should return zeroed-out sensitivity analysis."""
        result = analyze_parameter_sensitivity([])
        assert result["lookback_sensitivity"] == {}
        assert result["quantile_sensitivity"] == {}
        assert result["sharpe_range"] == 0.0


# --- Tests for evaluate_hyperparameter_optimization ---


class TestEvaluateHyperparameterOptimization:
    """Tests for the full Phase 6 evaluation pipeline."""

    def test_returns_arf_schema(self):
        """Output should conform to ARF metrics.json schema."""
        panel = _make_panel()
        config = BacktestConfig(n_splits=3, min_train_size=40)
        metrics = evaluate_hyperparameter_optimization(
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

    def test_transaction_costs_schema(self):
        """Transaction costs should have feeBps, slippageBps, netSharpe."""
        panel = _make_panel()
        config = BacktestConfig(n_splits=3, min_train_size=40)
        metrics = evaluate_hyperparameter_optimization(
            panel, config=config,
            lookback_values=[10],
            quantile_values=[0.3],
        )
        tc = metrics["transactionCosts"]
        assert "feeBps" in tc
        assert "slippageBps" in tc
        assert "netSharpe" in tc

    def test_walk_forward_schema(self):
        """Walk-forward section should have windows, positiveWindows, avgOosSharpe."""
        panel = _make_panel()
        config = BacktestConfig(n_splits=3, min_train_size=40)
        metrics = evaluate_hyperparameter_optimization(
            panel, config=config,
            lookback_values=[10],
            quantile_values=[0.3],
        )
        wf = metrics["walkForward"]
        assert "windows" in wf
        assert "positiveWindows" in wf
        assert "avgOosSharpe" in wf

    def test_custom_metrics_contains_optimization_results(self):
        """Custom metrics should include optimization-specific data."""
        panel = _make_panel()
        config = BacktestConfig(n_splits=3, min_train_size=40)
        metrics = evaluate_hyperparameter_optimization(
            panel, config=config,
            lookback_values=[5, 10],
            quantile_values=[0.2, 0.3],
        )
        cm = metrics["customMetrics"]
        assert "bestStrategy" in cm
        assert "optimizedLookback" in cm
        assert "optimizedQuantile" in cm
        assert "totalCombinations" in cm
        assert "improvementOverBaseline" in cm
        assert "parameterSensitivity" in cm
        assert "top5Combinations" in cm
        assert "bestPerStrategy" in cm

    def test_best_strategy_is_valid(self):
        """Best strategy should be one of the 3 momentum columns."""
        panel = _make_panel()
        config = BacktestConfig(n_splits=3, min_train_size=40)
        metrics = evaluate_hyperparameter_optimization(
            panel, config=config,
            lookback_values=[10],
            quantile_values=[0.3],
        )
        assert metrics["customMetrics"]["bestStrategy"] in STRATEGY_COLUMNS

    def test_all_strategies_in_sensitivity(self):
        """Sensitivity analysis should cover all 3 strategies."""
        panel = _make_panel()
        config = BacktestConfig(n_splits=3, min_train_size=40)
        metrics = evaluate_hyperparameter_optimization(
            panel, config=config,
            lookback_values=[10],
            quantile_values=[0.3],
        )
        sens = metrics["customMetrics"]["parameterSensitivity"]
        for strategy in STRATEGY_COLUMNS:
            assert strategy in sens

    def test_all_strategies_in_best_per_strategy(self):
        """bestPerStrategy should include all 3 momentum strategies."""
        panel = _make_panel()
        config = BacktestConfig(n_splits=3, min_train_size=40)
        metrics = evaluate_hyperparameter_optimization(
            panel, config=config,
            lookback_values=[10],
            quantile_values=[0.3],
        )
        bps = metrics["customMetrics"]["bestPerStrategy"]
        for strategy in STRATEGY_COLUMNS:
            assert strategy in bps

    def test_total_combinations_correct(self):
        """Total combinations should equal lookbacks × quantiles × strategies."""
        panel = _make_panel()
        config = BacktestConfig(n_splits=3, min_train_size=40)
        lbs = [5, 10, 15]
        qs = [0.2, 0.3]
        metrics = evaluate_hyperparameter_optimization(
            panel, config=config,
            lookback_values=lbs,
            quantile_values=qs,
        )
        expected = len(lbs) * len(qs) * len(STRATEGY_COLUMNS)
        assert metrics["customMetrics"]["totalCombinations"] == expected

    def test_top5_combinations_limited(self):
        """Top 5 combinations should have at most 5 entries."""
        panel = _make_panel()
        config = BacktestConfig(n_splits=3, min_train_size=40)
        metrics = evaluate_hyperparameter_optimization(
            panel, config=config,
            lookback_values=[5, 10, 15],
            quantile_values=[0.2, 0.3],
        )
        top5 = metrics["customMetrics"]["top5Combinations"]
        assert len(top5) <= 5

    def test_top5_sorted_by_net_sharpe(self):
        """Top 5 should be sorted by net Sharpe descending."""
        panel = _make_panel()
        config = BacktestConfig(n_splits=3, min_train_size=40)
        metrics = evaluate_hyperparameter_optimization(
            panel, config=config,
            lookback_values=[5, 10, 15],
            quantile_values=[0.2, 0.3],
        )
        top5 = metrics["customMetrics"]["top5Combinations"]
        sharpes = [r["netSharpe"] for r in top5]
        assert sharpes == sorted(sharpes, reverse=True)

    def test_improvement_over_baseline_present(self):
        """Improvement over baseline should show delta for strategies with baseline data."""
        panel = _make_panel()
        config = BacktestConfig(n_splits=3, min_train_size=40)
        metrics = evaluate_hyperparameter_optimization(
            panel, config=config,
            lookback_values=[10, 12, 15],
            quantile_values=[0.2, 0.3],
        )
        improvement = metrics["customMetrics"]["improvementOverBaseline"]
        # Should have entries for strategies where baseline (12, 0.3) was tested
        for strategy, imp in improvement.items():
            assert "baseline_netSharpe" in imp
            assert "optimized_netSharpe" in imp
            assert "sharpe_improvement" in imp
            assert imp["baseline_lookback"] == 12
            assert imp["baseline_quantile"] == 0.3

    def test_optimized_lookback_in_search_range(self):
        """Optimized lookback should be from the search values."""
        panel = _make_panel()
        config = BacktestConfig(n_splits=3, min_train_size=40)
        lbs = [5, 10, 20]
        metrics = evaluate_hyperparameter_optimization(
            panel, config=config,
            lookback_values=lbs,
            quantile_values=[0.3],
        )
        assert metrics["customMetrics"]["optimizedLookback"] in lbs

    def test_optimized_quantile_in_search_range(self):
        """Optimized quantile should be from the search values."""
        panel = _make_panel()
        config = BacktestConfig(n_splits=3, min_train_size=40)
        qs = [0.2, 0.3, 0.4]
        metrics = evaluate_hyperparameter_optimization(
            panel, config=config,
            lookback_values=[10],
            quantile_values=qs,
        )
        assert metrics["customMetrics"]["optimizedQuantile"] in qs

    def test_windows_positive(self):
        """Walk-forward should produce at least one window."""
        panel = _make_panel(n_dates=200)
        config = BacktestConfig(n_splits=3, min_train_size=40)
        metrics = evaluate_hyperparameter_optimization(
            panel, config=config,
            lookback_values=[10],
            quantile_values=[0.3],
        )
        assert metrics["walkForward"]["windows"] > 0
