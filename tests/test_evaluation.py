"""
Tests for the walk-forward evaluation framework (src/evaluation.py).

Tests cover: momentum score computation, portfolio formation,
portfolio return calculation, walk-forward evaluation, and metrics generation.
"""
import numpy as np
import pandas as pd
import pytest

from src.backtest import BacktestConfig, BacktestResult
from src.evaluation import (
    compute_momentum_scores,
    compute_portfolio_return,
    evaluate_momentum_strategy,
    form_long_short_portfolio,
    run_walk_forward_evaluation,
)


def _make_panel(n_dates=120, n_stocks=6, seed=42):
    """
    Create a deterministic panel for testing evaluation.

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


class TestComputeMomentumScores:
    """Tests for compute_momentum_scores."""

    def test_returns_expected_columns(self):
        """Output should contain momentum score columns."""
        panel = _make_panel()
        scores = compute_momentum_scores(panel, lookback=12)
        for col in ["date", "stock_id", "total_momentum", "industry_momentum", "stock_specific_momentum"]:
            assert col in scores.columns

    def test_no_nan_in_total_momentum(self):
        """Returned scores should have no NaN in total_momentum (NaNs are dropped)."""
        panel = _make_panel()
        scores = compute_momentum_scores(panel, lookback=12)
        assert scores["total_momentum"].isna().sum() == 0

    def test_uptrend_stocks_have_positive_momentum(self):
        """Stocks with positive drift should have positive total momentum."""
        panel = _make_panel(n_dates=120)
        scores = compute_momentum_scores(panel, lookback=12)
        latest = scores.groupby("stock_id").last()
        assert latest.loc["A1", "total_momentum"] > 0
        assert latest.loc["A2", "total_momentum"] > 0

    def test_downtrend_stocks_have_negative_momentum(self):
        """Stocks with negative drift should have negative total momentum."""
        panel = _make_panel(n_dates=120)
        scores = compute_momentum_scores(panel, lookback=12)
        latest = scores.groupby("stock_id").last()
        assert latest.loc["B1", "total_momentum"] < 0
        assert latest.loc["B2", "total_momentum"] < 0


class TestFormLongShortPortfolio:
    """Tests for form_long_short_portfolio."""

    def test_returns_positions(self):
        """Should return a DataFrame with stock_id and position columns."""
        panel = _make_panel()
        scores = compute_momentum_scores(panel, lookback=12)
        latest_date = scores["date"].max()
        portfolio = form_long_short_portfolio(scores, latest_date)
        assert "stock_id" in portfolio.columns
        assert "position" in portfolio.columns

    def test_long_short_balance(self):
        """Should have both long (+1) and short (-1) positions."""
        panel = _make_panel()
        scores = compute_momentum_scores(panel, lookback=12)
        latest_date = scores["date"].max()
        portfolio = form_long_short_portfolio(scores, latest_date, quantile=0.34)
        assert (portfolio["position"] > 0).any(), "Should have long positions"
        assert (portfolio["position"] < 0).any(), "Should have short positions"

    def test_winners_are_long(self):
        """High momentum stocks should be in the long leg."""
        panel = _make_panel(n_dates=120)
        scores = compute_momentum_scores(panel, lookback=12)
        latest_date = scores["date"].max()
        portfolio = form_long_short_portfolio(scores, latest_date, quantile=0.34)
        longs = set(portfolio[portfolio["position"] > 0]["stock_id"])
        # A1 and A2 have highest momentum, so at least one should be long
        assert longs & {"A1", "A2"}, f"Expected uptrend stocks in longs, got {longs}"

    def test_losers_are_short(self):
        """Low momentum stocks should be in the short leg."""
        panel = _make_panel(n_dates=120)
        scores = compute_momentum_scores(panel, lookback=12)
        latest_date = scores["date"].max()
        portfolio = form_long_short_portfolio(scores, latest_date, quantile=0.34)
        shorts = set(portfolio[portfolio["position"] < 0]["stock_id"])
        # B1 and B2 have lowest momentum, so at least one should be short
        assert shorts & {"B1", "B2"}, f"Expected downtrend stocks in shorts, got {shorts}"

    def test_empty_date_returns_empty(self):
        """Non-existent date should return empty portfolio."""
        panel = _make_panel()
        scores = compute_momentum_scores(panel, lookback=12)
        portfolio = form_long_short_portfolio(scores, pd.Timestamp("1999-01-01"))
        assert len(portfolio) == 0

    def test_n_long_n_short_override(self):
        """n_long and n_short should override quantile."""
        panel = _make_panel()
        scores = compute_momentum_scores(panel, lookback=12)
        latest_date = scores["date"].max()
        portfolio = form_long_short_portfolio(
            scores, latest_date, n_long=1, n_short=1,
        )
        assert len(portfolio) == 2
        assert (portfolio["position"] > 0).sum() == 1
        assert (portfolio["position"] < 0).sum() == 1


class TestComputePortfolioReturn:
    """Tests for compute_portfolio_return."""

    def test_positive_return_from_long_winner(self):
        """Long a rising stock should give positive return."""
        dates = pd.bdate_range("2023-01-02", periods=2)
        panel = pd.DataFrame({
            "date": [dates[0], dates[0], dates[1], dates[1]],
            "stock_id": ["X", "Y", "X", "Y"],
            "price": [100, 100, 110, 90],
        })
        portfolio = pd.DataFrame({
            "stock_id": ["X"],
            "position": [1.0],
        })
        ret = compute_portfolio_return(panel, portfolio, dates[0], dates[1])
        assert ret > 0

    def test_positive_return_from_short_loser(self):
        """Short a falling stock should give positive return."""
        dates = pd.bdate_range("2023-01-02", periods=2)
        panel = pd.DataFrame({
            "date": [dates[0], dates[0], dates[1], dates[1]],
            "stock_id": ["X", "Y", "X", "Y"],
            "price": [100, 100, 90, 110],
        })
        portfolio = pd.DataFrame({
            "stock_id": ["X"],
            "position": [-1.0],
        })
        ret = compute_portfolio_return(panel, portfolio, dates[0], dates[1])
        assert ret > 0

    def test_long_short_hedged(self):
        """Long-short portfolio with equal moves should net to zero."""
        dates = pd.bdate_range("2023-01-02", periods=2)
        panel = pd.DataFrame({
            "date": [dates[0], dates[0], dates[1], dates[1]],
            "stock_id": ["X", "Y", "X", "Y"],
            "price": [100, 100, 110, 110],  # both up equally
        })
        portfolio = pd.DataFrame({
            "stock_id": ["X", "Y"],
            "position": [1.0, -1.0],
        })
        ret = compute_portfolio_return(panel, portfolio, dates[0], dates[1])
        assert abs(ret) < 1e-10

    def test_empty_portfolio_returns_zero(self):
        """Empty portfolio should return 0."""
        dates = pd.bdate_range("2023-01-02", periods=2)
        panel = pd.DataFrame({
            "date": dates, "stock_id": ["X", "X"], "price": [100, 110],
        })
        empty = pd.DataFrame(columns=["stock_id", "position"])
        ret = compute_portfolio_return(panel, empty, dates[0], dates[1])
        assert ret == 0.0


class TestRunWalkForwardEvaluation:
    """Tests for run_walk_forward_evaluation."""

    def test_returns_list_of_backtest_results(self):
        """Should return a list of BacktestResult objects."""
        panel = _make_panel(n_dates=200)
        config = BacktestConfig(n_splits=3, min_train_size=50)
        results = run_walk_forward_evaluation(panel, config=config, lookback=12)
        assert isinstance(results, list)
        assert all(isinstance(r, BacktestResult) for r in results)

    def test_at_least_one_window(self):
        """Should produce at least one walk-forward window."""
        panel = _make_panel(n_dates=200)
        config = BacktestConfig(n_splits=3, min_train_size=50)
        results = run_walk_forward_evaluation(panel, config=config, lookback=12)
        assert len(results) >= 1

    def test_no_future_leakage(self):
        """Training period should end before test period starts."""
        panel = _make_panel(n_dates=200)
        config = BacktestConfig(n_splits=3, min_train_size=50)
        results = run_walk_forward_evaluation(panel, config=config, lookback=12)
        for r in results:
            assert r.train_end < r.test_start, (
                f"Train end {r.train_end} must be before test start {r.test_start}"
            )

    def test_results_have_valid_metrics(self):
        """Each result should have finite metric values."""
        panel = _make_panel(n_dates=200)
        config = BacktestConfig(n_splits=3, min_train_size=50)
        results = run_walk_forward_evaluation(panel, config=config, lookback=12)
        for r in results:
            assert np.isfinite(r.gross_sharpe)
            assert np.isfinite(r.net_sharpe)
            assert np.isfinite(r.annual_return)
            assert np.isfinite(r.max_drawdown)
            assert r.total_trades > 0

    def test_pnl_series_present(self):
        """Each result should have a non-empty PnL series."""
        panel = _make_panel(n_dates=200)
        config = BacktestConfig(n_splits=3, min_train_size=50)
        results = run_walk_forward_evaluation(panel, config=config, lookback=12)
        for r in results:
            assert r.pnl_series is not None
            assert len(r.pnl_series) > 0


class TestEvaluateMomentumStrategy:
    """Tests for evaluate_momentum_strategy (end-to-end)."""

    def test_returns_valid_metrics_json(self):
        """Output should match the ARF metrics.json schema."""
        panel = _make_panel(n_dates=200)
        config = BacktestConfig(n_splits=3, min_train_size=50)
        metrics = evaluate_momentum_strategy(panel, config=config, lookback=12)

        # Required top-level keys
        assert "sharpeRatio" in metrics
        assert "annualReturn" in metrics
        assert "maxDrawdown" in metrics
        assert "hitRate" in metrics
        assert "totalTrades" in metrics
        assert "transactionCosts" in metrics
        assert "walkForward" in metrics
        assert "customMetrics" in metrics

    def test_transaction_costs_structure(self):
        """transactionCosts should have feeBps, slippageBps, netSharpe."""
        panel = _make_panel(n_dates=200)
        config = BacktestConfig(n_splits=3, min_train_size=50)
        metrics = evaluate_momentum_strategy(panel, config=config, lookback=12)
        tc = metrics["transactionCosts"]
        assert "feeBps" in tc
        assert "slippageBps" in tc
        assert "netSharpe" in tc

    def test_walk_forward_structure(self):
        """walkForward should have windows, positiveWindows, avgOosSharpe."""
        panel = _make_panel(n_dates=200)
        config = BacktestConfig(n_splits=3, min_train_size=50)
        metrics = evaluate_momentum_strategy(panel, config=config, lookback=12)
        wf = metrics["walkForward"]
        assert "windows" in wf
        assert "positiveWindows" in wf
        assert "avgOosSharpe" in wf
        assert wf["windows"] >= 1

    def test_custom_metrics_populated(self):
        """customMetrics should contain strategy metadata."""
        panel = _make_panel(n_dates=200)
        config = BacktestConfig(n_splits=3, min_train_size=50)
        metrics = evaluate_momentum_strategy(panel, config=config, lookback=12)
        cm = metrics["customMetrics"]
        assert "strategy" in cm
        assert "lookback_periods" in cm
        assert cm["lookback_periods"] == 12

    def test_net_sharpe_lte_gross(self):
        """Net Sharpe should generally be less than or equal to gross Sharpe."""
        panel = _make_panel(n_dates=200)
        config = BacktestConfig(n_splits=3, min_train_size=50)
        metrics = evaluate_momentum_strategy(panel, config=config, lookback=12)
        # Transaction costs should reduce performance
        # (equality possible if costs are negligible relative to returns)
        assert metrics["transactionCosts"]["netSharpe"] <= metrics["sharpeRatio"] + 0.1
