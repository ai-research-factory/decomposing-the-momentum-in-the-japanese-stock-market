"""
ARF Standard Backtest Framework
Walk-forward validation with transaction cost accounting.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BacktestConfig:
    """Backtest configuration."""
    fee_bps: float = 10.0       # Transaction fee in basis points
    slippage_bps: float = 5.0   # Slippage in basis points
    train_ratio: float = 0.7    # Train window ratio for walk-forward
    n_splits: int = 10          # Number of walk-forward windows
    gap: int = 1                # Gap between train and test (prevent leakage)
    min_train_size: int = 252   # Minimum training samples (~1 year daily)


# Pre-defined cost scenarios for sensitivity analysis
COST_SCENARIOS = {
    "zero": {"fee_bps": 0.0, "slippage_bps": 0.0, "label": "Zero costs"},
    "low": {"fee_bps": 5.0, "slippage_bps": 3.0, "label": "Low costs (8 bps)"},
    "medium": {"fee_bps": 10.0, "slippage_bps": 5.0, "label": "Medium costs (15 bps)"},
    "high": {"fee_bps": 20.0, "slippage_bps": 10.0, "label": "High costs (30 bps)"},
    "very_high": {"fee_bps": 30.0, "slippage_bps": 20.0, "label": "Very high costs (50 bps)"},
}


@dataclass
class CostBreakdown:
    """Detailed breakdown of transaction costs for a backtest window."""
    window: int
    total_cost: float          # Total cost as fraction of portfolio
    fee_cost: float            # Commission/fee portion
    slippage_cost: float       # Slippage/market-impact portion
    n_trades: int              # Number of trades (position changes)
    turnover: float            # Sum of absolute position changes
    cost_per_trade_bps: float  # Average cost per trade in bps
    cost_drag_annual_bps: float  # Annualized cost drag in bps


@dataclass
class BacktestResult:
    """Results from a single walk-forward window."""
    window: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    gross_sharpe: float = 0.0
    net_sharpe: float = 0.0
    annual_return: float = 0.0
    max_drawdown: float = 0.0
    total_trades: int = 0
    hit_rate: float = 0.0
    pnl_series: Optional[pd.Series] = field(default=None, repr=False)


class WalkForwardValidator:
    """
    Walk-forward out-of-sample validation.

    Usage:
        validator = WalkForwardValidator(config)
        for train_idx, test_idx in validator.split(df):
            train_df = df.iloc[train_idx]
            test_df = df.iloc[test_idx]
            # Train model on train_df, evaluate on test_df
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()

    def split(self, data: pd.DataFrame):
        """Generate train/test index pairs for walk-forward validation."""
        n = len(data)
        cfg = self.config
        test_size = max(1, (n - cfg.min_train_size) // cfg.n_splits)

        for i in range(cfg.n_splits):
            test_end = n - (cfg.n_splits - 1 - i) * test_size
            test_start = test_end - test_size
            train_end = test_start - cfg.gap
            train_start = max(0, int(train_end * (1 - cfg.train_ratio))) if cfg.train_ratio < 1.0 else 0

            if train_end - train_start < cfg.min_train_size:
                continue
            if test_start >= test_end:
                continue

            yield (
                list(range(train_start, train_end)),
                list(range(test_start, test_end)),
            )


def calculate_costs(returns: pd.Series, positions: pd.Series, config: BacktestConfig) -> pd.Series:
    """
    Calculate transaction costs from position changes.

    Args:
        returns: Gross returns series
        positions: Position series (-1, 0, 1 or continuous)
        config: Backtest configuration with fee/slippage settings

    Returns:
        Net returns after costs
    """
    trades = positions.diff().abs().fillna(0)
    cost_per_trade = (config.fee_bps + config.slippage_bps) / 10000
    costs = trades * cost_per_trade
    return returns - costs


def calculate_costs_detailed(
    returns: pd.Series,
    positions: pd.Series,
    config: BacktestConfig,
    window: int = 0,
    periods_per_year: int = 252,
) -> tuple[pd.Series, CostBreakdown]:
    """
    Calculate transaction costs with a detailed breakdown.

    Separates fee and slippage components and computes turnover,
    per-trade costs, and annualized cost drag.

    Args:
        returns: Gross returns series.
        positions: Position series (-1, 0, 1 or continuous).
        config: Backtest configuration.
        window: Walk-forward window number for labeling.
        periods_per_year: Trading periods per year.

    Returns:
        Tuple of (net_returns, CostBreakdown).
    """
    trades = positions.diff().abs().fillna(0)
    turnover = float(trades.sum())
    n_trades = int((trades > 0).sum())

    fee_cost_total = turnover * config.fee_bps / 10000
    slippage_cost_total = turnover * config.slippage_bps / 10000
    total_cost = fee_cost_total + slippage_cost_total

    cost_per_trade_bps = 0.0
    if n_trades > 0:
        cost_per_trade_bps = (total_cost / n_trades) * 10000

    n_periods = len(returns)
    if n_periods > 0:
        cost_drag_annual_bps = (total_cost / n_periods) * periods_per_year * 10000
    else:
        cost_drag_annual_bps = 0.0

    costs = trades * (config.fee_bps + config.slippage_bps) / 10000
    net_returns = returns - costs

    breakdown = CostBreakdown(
        window=window,
        total_cost=round(total_cost, 6),
        fee_cost=round(fee_cost_total, 6),
        slippage_cost=round(slippage_cost_total, 6),
        n_trades=n_trades,
        turnover=round(turnover, 4),
        cost_per_trade_bps=round(cost_per_trade_bps, 2),
        cost_drag_annual_bps=round(cost_drag_annual_bps, 2),
    )
    return net_returns, breakdown


def compute_turnover(positions_history: list[pd.DataFrame]) -> float:
    """
    Compute average one-way turnover across rebalancing events.

    Each element of positions_history is a DataFrame with stock_id, position
    columns representing the portfolio at that point.

    Args:
        positions_history: List of portfolio DataFrames in chronological order.

    Returns:
        Average one-way turnover (fraction of portfolio changed per rebalance).
    """
    if len(positions_history) < 2:
        return 0.0

    turnovers = []
    for i in range(1, len(positions_history)):
        prev = positions_history[i - 1].set_index("stock_id")["position"]
        curr = positions_history[i].set_index("stock_id")["position"]
        all_stocks = prev.index.union(curr.index)
        prev_full = prev.reindex(all_stocks, fill_value=0.0)
        curr_full = curr.reindex(all_stocks, fill_value=0.0)
        turnover = float((curr_full - prev_full).abs().sum()) / 2.0
        turnovers.append(turnover)

    return float(np.mean(turnovers))


def compute_metrics(returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> dict:
    """
    Compute standard performance metrics from a returns series.

    Args:
        returns: Daily (or periodic) returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Trading periods per year (252 for daily, 365 for crypto)

    Returns:
        Dict with sharpeRatio, annualReturn, maxDrawdown, hitRate, totalTrades
    """
    if len(returns) == 0:
        return {"sharpeRatio": 0.0, "annualReturn": 0.0, "maxDrawdown": 0.0, "hitRate": 0.0}

    excess = returns - risk_free_rate / periods_per_year
    sharpe = float(np.sqrt(periods_per_year) * excess.mean() / excess.std()) if excess.std() > 0 else 0.0

    cumulative = (1 + returns).cumprod()
    annual_return = float(cumulative.iloc[-1] ** (periods_per_year / len(returns)) - 1)

    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = float(drawdown.min())

    hit_rate = float((returns > 0).sum() / len(returns)) if len(returns) > 0 else 0.0

    return {
        "sharpeRatio": round(sharpe, 4),
        "annualReturn": round(annual_return, 4),
        "maxDrawdown": round(max_drawdown, 4),
        "hitRate": round(hit_rate, 4),
    }


def generate_metrics_json(
    results: list[BacktestResult],
    config: BacktestConfig,
    custom_metrics: Optional[dict] = None,
) -> dict:
    """
    Generate ARF-standard metrics.json from walk-forward results.

    Args:
        results: List of BacktestResult from each window
        config: Backtest configuration
        custom_metrics: Optional paper-specific metrics

    Returns:
        Dict matching ARF metrics.json schema
    """
    if not results:
        return {
            "sharpeRatio": 0.0, "annualReturn": 0.0, "maxDrawdown": 0.0,
            "hitRate": 0.0, "totalTrades": 0,
            "transactionCosts": {"feeBps": config.fee_bps, "slippageBps": config.slippage_bps, "netSharpe": 0.0},
            "walkForward": {"windows": 0, "positiveWindows": 0, "avgOosSharpe": 0.0},
            "customMetrics": custom_metrics or {},
        }

    net_sharpes = [r.net_sharpe for r in results]
    positive_windows = sum(1 for s in net_sharpes if s > 0)

    return {
        "sharpeRatio": round(float(np.mean([r.gross_sharpe for r in results])), 4),
        "annualReturn": round(float(np.mean([r.annual_return for r in results])), 4),
        "maxDrawdown": round(float(min(r.max_drawdown for r in results)), 4),
        "hitRate": round(float(np.mean([r.hit_rate for r in results])), 4),
        "totalTrades": sum(r.total_trades for r in results),
        "transactionCosts": {
            "feeBps": config.fee_bps,
            "slippageBps": config.slippage_bps,
            "netSharpe": round(float(np.mean(net_sharpes)), 4),
        },
        "walkForward": {
            "windows": len(results),
            "positiveWindows": positive_windows,
            "avgOosSharpe": round(float(np.mean(net_sharpes)), 4),
        },
        "customMetrics": custom_metrics or {},
    }
