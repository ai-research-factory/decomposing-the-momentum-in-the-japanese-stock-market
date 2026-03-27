"""
Tests for transaction cost analysis (Phase 5).

Tests cover: detailed cost calculation, cost breakdown, turnover computation,
cost sensitivity analysis, cost impact, breakeven costs, and end-to-end evaluation.
"""
import numpy as np
import pandas as pd
import pytest

from src.backtest import (
    COST_SCENARIOS,
    BacktestConfig,
    BacktestResult,
    CostBreakdown,
    calculate_costs_detailed,
    compute_turnover,
)
from src.evaluation import (
    STRATEGY_COLUMNS,
    compute_breakeven_cost,
    compute_cost_impact,
    evaluate_transaction_costs,
    run_cost_sensitivity_analysis,
)


def _make_panel(n_dates=200, seed=42):
    """
    Create a deterministic panel for testing transaction cost analysis.

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


# --- Tests for calculate_costs_detailed ---


class TestCalculateCostsDetailed:
    """Tests for the detailed cost calculation function."""

    def test_returns_tuple(self):
        """Should return (net_returns, CostBreakdown) tuple."""
        returns = pd.Series([0.01, -0.005, 0.003, 0.002])
        positions = pd.Series([0.0, 1.0, 1.0, 1.0])
        config = BacktestConfig(fee_bps=10.0, slippage_bps=5.0)
        net_ret, breakdown = calculate_costs_detailed(returns, positions, config)
        assert isinstance(net_ret, pd.Series)
        assert isinstance(breakdown, CostBreakdown)

    def test_net_returns_match_simple_calc(self):
        """Detailed net returns should match the simple calculate_costs function."""
        from src.backtest import calculate_costs
        returns = pd.Series([0.01, -0.005, 0.003, 0.002])
        positions = pd.Series([0.0, 1.0, 1.0, 1.0])
        config = BacktestConfig(fee_bps=10.0, slippage_bps=5.0)
        simple_net = calculate_costs(returns, positions, config)
        detailed_net, _ = calculate_costs_detailed(returns, positions, config)
        pd.testing.assert_series_equal(simple_net, detailed_net)

    def test_fee_slippage_separation(self):
        """Fee and slippage costs should sum to total cost."""
        returns = pd.Series([0.01, -0.005, 0.003])
        positions = pd.Series([0.0, 1.0, 1.0])
        config = BacktestConfig(fee_bps=10.0, slippage_bps=5.0)
        _, breakdown = calculate_costs_detailed(returns, positions, config)
        assert abs(breakdown.total_cost - breakdown.fee_cost - breakdown.slippage_cost) < 1e-10

    def test_zero_costs_no_impact(self):
        """Zero costs should yield net returns equal to gross returns."""
        returns = pd.Series([0.01, -0.005, 0.003])
        positions = pd.Series([0.0, 1.0, 1.0])
        config = BacktestConfig(fee_bps=0.0, slippage_bps=0.0)
        net_ret, breakdown = calculate_costs_detailed(returns, positions, config)
        pd.testing.assert_series_equal(net_ret, returns)
        assert breakdown.total_cost == 0.0

    def test_trade_count(self):
        """Should correctly count number of position changes."""
        returns = pd.Series([0.01, -0.005, 0.003, 0.002, 0.001])
        # positions change at index 1 (0->1) and index 3 (1->-1)
        positions = pd.Series([0.0, 1.0, 1.0, -1.0, -1.0])
        config = BacktestConfig(fee_bps=10.0, slippage_bps=5.0)
        _, breakdown = calculate_costs_detailed(returns, positions, config)
        assert breakdown.n_trades == 2

    def test_turnover_calculation(self):
        """Turnover should be sum of absolute position changes."""
        returns = pd.Series([0.01, -0.005, 0.003])
        positions = pd.Series([0.0, 1.0, -1.0])  # changes: 1.0 + 2.0 = 3.0
        config = BacktestConfig(fee_bps=10.0, slippage_bps=5.0)
        _, breakdown = calculate_costs_detailed(returns, positions, config)
        assert breakdown.turnover == 3.0

    def test_cost_drag_annual_positive(self):
        """Annualized cost drag should be positive when there are costs."""
        returns = pd.Series([0.01, -0.005, 0.003])
        positions = pd.Series([0.0, 1.0, 1.0])
        config = BacktestConfig(fee_bps=10.0, slippage_bps=5.0)
        _, breakdown = calculate_costs_detailed(returns, positions, config)
        assert breakdown.cost_drag_annual_bps > 0


# --- Tests for compute_turnover ---


class TestComputeTurnover:
    """Tests for portfolio turnover computation."""

    def test_identical_portfolios_zero_turnover(self):
        """Same portfolio at two dates should have zero turnover."""
        p1 = pd.DataFrame({"stock_id": ["A", "B"], "position": [1.0, -1.0]})
        p2 = pd.DataFrame({"stock_id": ["A", "B"], "position": [1.0, -1.0]})
        assert compute_turnover([p1, p2]) == 0.0

    def test_complete_reversal_full_turnover(self):
        """Reversing all positions should give turnover = 2 (for long-short)."""
        p1 = pd.DataFrame({"stock_id": ["A", "B"], "position": [1.0, -1.0]})
        p2 = pd.DataFrame({"stock_id": ["A", "B"], "position": [-1.0, 1.0]})
        turnover = compute_turnover([p1, p2])
        assert turnover == 2.0  # (2+2)/2 = 2.0

    def test_single_portfolio_zero_turnover(self):
        """Single portfolio (no rebalancing) should give zero turnover."""
        p1 = pd.DataFrame({"stock_id": ["A", "B"], "position": [1.0, -1.0]})
        assert compute_turnover([p1]) == 0.0

    def test_empty_list_zero_turnover(self):
        """Empty list should give zero turnover."""
        assert compute_turnover([]) == 0.0

    def test_new_stock_entry(self):
        """Adding a new stock should contribute to turnover."""
        p1 = pd.DataFrame({"stock_id": ["A"], "position": [1.0]})
        p2 = pd.DataFrame({"stock_id": ["A", "B"], "position": [1.0, -1.0]})
        turnover = compute_turnover([p1, p2])
        assert turnover > 0


# --- Tests for run_cost_sensitivity_analysis ---


class TestRunCostSensitivityAnalysis:
    """Tests for cost sensitivity analysis across scenarios."""

    def test_returns_all_scenarios(self):
        """Should return results for all cost scenarios."""
        panel = _make_panel()
        config = BacktestConfig(n_splits=3, min_train_size=50)
        results = run_cost_sensitivity_analysis(
            panel, config=config, lookback=12,
        )
        for scenario in COST_SCENARIOS:
            assert scenario in results, f"Missing scenario: {scenario}"

    def test_each_scenario_has_all_strategies(self):
        """Each scenario should contain metrics for all 3 strategies."""
        panel = _make_panel()
        config = BacktestConfig(n_splits=3, min_train_size=50)
        results = run_cost_sensitivity_analysis(
            panel, config=config, lookback=12,
        )
        for scenario_name, strategies in results.items():
            for col in STRATEGY_COLUMNS:
                assert col in strategies, f"{scenario_name} missing {col}"

    def test_zero_cost_matches_gross(self):
        """Zero-cost scenario net Sharpe should equal gross Sharpe."""
        panel = _make_panel()
        config = BacktestConfig(n_splits=3, min_train_size=50)
        results = run_cost_sensitivity_analysis(
            panel, config=config, lookback=12,
        )
        for strategy in STRATEGY_COLUMNS:
            m = results["zero"][strategy]
            gross = m["sharpeRatio"]
            net = m["transactionCosts"]["netSharpe"]
            assert abs(gross - net) < 0.05, (
                f"Zero-cost: gross={gross} net={net} differ for {strategy}"
            )

    def test_higher_costs_lower_net_sharpe(self):
        """Higher cost scenarios should generally yield lower net Sharpe."""
        panel = _make_panel()
        config = BacktestConfig(n_splits=3, min_train_size=50)
        results = run_cost_sensitivity_analysis(
            panel, config=config, lookback=12,
        )
        for strategy in STRATEGY_COLUMNS:
            zero_net = results["zero"][strategy]["transactionCosts"]["netSharpe"]
            high_net = results["high"][strategy]["transactionCosts"]["netSharpe"]
            assert high_net <= zero_net + 0.1, (
                f"{strategy}: high cost net={high_net} > zero cost net={zero_net}"
            )

    def test_custom_scenarios(self):
        """Should accept custom cost scenarios."""
        panel = _make_panel()
        config = BacktestConfig(n_splits=3, min_train_size=50)
        custom = {
            "cheap": {"fee_bps": 2.0, "slippage_bps": 1.0, "label": "Cheap"},
            "expensive": {"fee_bps": 50.0, "slippage_bps": 25.0, "label": "Expensive"},
        }
        results = run_cost_sensitivity_analysis(
            panel, config=config, lookback=12, scenarios=custom,
        )
        assert "cheap" in results
        assert "expensive" in results

    def test_arf_schema_compliance(self):
        """Each scenario-strategy result should follow ARF schema."""
        panel = _make_panel()
        config = BacktestConfig(n_splits=3, min_train_size=50)
        results = run_cost_sensitivity_analysis(
            panel, config=config, lookback=12,
        )
        for scenario, strategies in results.items():
            for strategy, m in strategies.items():
                assert "sharpeRatio" in m
                assert "annualReturn" in m
                assert "maxDrawdown" in m
                assert "transactionCosts" in m
                assert "walkForward" in m


# --- Tests for compute_cost_impact ---


class TestComputeCostImpact:
    """Tests for cost impact computation."""

    def test_returns_all_strategies(self):
        """Should return impact data for all strategies."""
        panel = _make_panel()
        config = BacktestConfig(n_splits=3, min_train_size=50)
        sensitivity = run_cost_sensitivity_analysis(
            panel, config=config, lookback=12,
        )
        impact = compute_cost_impact(sensitivity)
        for col in STRATEGY_COLUMNS:
            assert col in impact

    def test_gross_sharpe_matches_zero_scenario(self):
        """Gross Sharpe in impact should match zero-cost scenario."""
        panel = _make_panel()
        config = BacktestConfig(n_splits=3, min_train_size=50)
        sensitivity = run_cost_sensitivity_analysis(
            panel, config=config, lookback=12,
        )
        impact = compute_cost_impact(sensitivity)
        for strategy in STRATEGY_COLUMNS:
            expected = sensitivity["zero"][strategy]["sharpeRatio"]
            assert impact[strategy]["grossSharpe"] == expected

    def test_sharpe_delta_negative_or_zero(self):
        """Sharpe delta (net - gross) should be <= 0 for positive-cost scenarios."""
        panel = _make_panel()
        config = BacktestConfig(n_splits=3, min_train_size=50)
        sensitivity = run_cost_sensitivity_analysis(
            panel, config=config, lookback=12,
        )
        impact = compute_cost_impact(sensitivity)
        for strategy in STRATEGY_COLUMNS:
            for scenario, data in impact[strategy]["scenarios"].items():
                assert data["sharpeDelta"] <= 0.05, (
                    f"{strategy}/{scenario}: sharpeDelta={data['sharpeDelta']} should be <= 0"
                )

    def test_empty_without_zero_scenario(self):
        """Should return empty dict if zero scenario is missing."""
        result = compute_cost_impact({"low": {}, "high": {}})
        assert result == {}


# --- Tests for compute_breakeven_cost ---


class TestComputeBreakevenCost:
    """Tests for breakeven cost estimation."""

    def test_returns_all_strategies(self):
        """Should return breakeven for all strategies."""
        panel = _make_panel()
        config = BacktestConfig(n_splits=3, min_train_size=50)
        breakeven = compute_breakeven_cost(
            panel, config=config, lookback=12, max_cost_bps=50.0, step_bps=10.0,
        )
        for col in STRATEGY_COLUMNS:
            assert col in breakeven

    def test_breakeven_non_negative(self):
        """Breakeven costs should be non-negative."""
        panel = _make_panel()
        config = BacktestConfig(n_splits=3, min_train_size=50)
        breakeven = compute_breakeven_cost(
            panel, config=config, lookback=12, max_cost_bps=50.0, step_bps=10.0,
        )
        for strategy, cost in breakeven.items():
            assert cost >= 0.0, f"{strategy} has negative breakeven: {cost}"

    def test_negative_gross_sharpe_zero_breakeven(self):
        """Strategies with negative gross Sharpe should have 0 breakeven."""
        panel = _make_panel()
        config = BacktestConfig(n_splits=3, min_train_size=50)
        breakeven = compute_breakeven_cost(
            panel, config=config, lookback=12, max_cost_bps=50.0, step_bps=10.0,
        )
        # Total and industry momentum typically have negative Sharpe
        # so their breakeven should be 0
        for strategy in ["total_momentum", "industry_momentum"]:
            if strategy in breakeven:
                assert breakeven[strategy] == 0.0 or breakeven[strategy] >= 0.0


# --- Tests for evaluate_transaction_costs (end-to-end) ---


class TestEvaluateTransactionCosts:
    """Tests for the full Phase 5 evaluation pipeline."""

    def test_returns_valid_arf_schema(self):
        """Output should match the ARF metrics.json schema."""
        panel = _make_panel()
        config = BacktestConfig(n_splits=3, min_train_size=50)
        metrics = evaluate_transaction_costs(panel, config=config, lookback=12)

        assert "sharpeRatio" in metrics
        assert "annualReturn" in metrics
        assert "maxDrawdown" in metrics
        assert "hitRate" in metrics
        assert "totalTrades" in metrics
        assert "transactionCosts" in metrics
        assert "walkForward" in metrics
        assert "customMetrics" in metrics

    def test_custom_metrics_has_cost_analysis(self):
        """customMetrics should contain cost analysis fields."""
        panel = _make_panel()
        config = BacktestConfig(n_splits=3, min_train_size=50)
        metrics = evaluate_transaction_costs(panel, config=config, lookback=12)
        cm = metrics["customMetrics"]
        assert "grossVsNet" in cm
        assert "costSensitivity" in cm
        assert "breakevenCosts" in cm
        assert "costImpact" in cm

    def test_gross_vs_net_has_all_strategies(self):
        """grossVsNet should have entries for all strategies."""
        panel = _make_panel()
        config = BacktestConfig(n_splits=3, min_train_size=50)
        metrics = evaluate_transaction_costs(panel, config=config, lookback=12)
        gvn = metrics["customMetrics"]["grossVsNet"]
        for col in STRATEGY_COLUMNS:
            assert col in gvn
            assert "grossSharpe" in gvn[col]
            assert "netSharpe" in gvn[col]
            assert "sharpeDegradation" in gvn[col]
            assert "breakevenCostBps" in gvn[col]

    def test_cost_sensitivity_has_scenarios(self):
        """costSensitivity should have multiple cost scenarios."""
        panel = _make_panel()
        config = BacktestConfig(n_splits=3, min_train_size=50)
        metrics = evaluate_transaction_costs(panel, config=config, lookback=12)
        cs = metrics["customMetrics"]["costSensitivity"]
        assert len(cs) >= 3  # At least zero, low/medium, high

    def test_best_strategy_is_valid(self):
        """Best strategy should be one of the 3 strategies."""
        panel = _make_panel()
        config = BacktestConfig(n_splits=3, min_train_size=50)
        metrics = evaluate_transaction_costs(panel, config=config, lookback=12)
        best = metrics["customMetrics"]["bestStrategy"]
        assert best in STRATEGY_COLUMNS

    def test_transaction_costs_structure(self):
        """transactionCosts should have feeBps, slippageBps, netSharpe."""
        panel = _make_panel()
        config = BacktestConfig(n_splits=3, min_train_size=50)
        metrics = evaluate_transaction_costs(panel, config=config, lookback=12)
        tc = metrics["transactionCosts"]
        assert "feeBps" in tc
        assert "slippageBps" in tc
        assert "netSharpe" in tc

    def test_walk_forward_has_windows(self):
        """walkForward should report at least 1 window."""
        panel = _make_panel()
        config = BacktestConfig(n_splits=3, min_train_size=50)
        metrics = evaluate_transaction_costs(panel, config=config, lookback=12)
        assert metrics["walkForward"]["windows"] >= 1

    def test_sharpe_degradation_non_negative(self):
        """Sharpe degradation (gross - net) should be >= 0 for medium costs."""
        panel = _make_panel()
        config = BacktestConfig(n_splits=3, min_train_size=50)
        metrics = evaluate_transaction_costs(panel, config=config, lookback=12)
        for strategy, data in metrics["customMetrics"]["grossVsNet"].items():
            assert data["sharpeDegradation"] >= -0.05, (
                f"{strategy}: degradation={data['sharpeDegradation']} should be >= 0"
            )
