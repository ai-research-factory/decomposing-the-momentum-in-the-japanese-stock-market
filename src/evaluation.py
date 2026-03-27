"""
Walk-forward evaluation framework for momentum strategies.

Evaluates long-short portfolios formed on total momentum scores
using the WalkForwardValidator and backtest utilities.
"""
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest import (
    COST_SCENARIOS,
    BacktestConfig,
    BacktestResult,
    CostBreakdown,
    WalkForwardValidator,
    calculate_costs,
    calculate_costs_detailed,
    compute_metrics,
    compute_turnover,
    generate_metrics_json,
)
from src.decomposition import decompose_momentum

logger = logging.getLogger(__name__)

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"


def compute_momentum_scores(panel: pd.DataFrame, lookback: int = 12) -> pd.DataFrame:
    """
    Compute momentum scores for each stock at each date.

    Uses the decomposition module to get total, industry, and stock-specific
    momentum. Returns a wide-format DataFrame indexed by date with stock_id
    columns containing total_momentum scores.

    Args:
        panel: Panel with columns [date, stock_id, industry_id, price].
        lookback: Rolling lookback window for momentum calculation.

    Returns:
        DataFrame with columns: date, stock_id, total_momentum,
        industry_momentum, stock_specific_momentum.
    """
    result = decompose_momentum(panel, lookback=lookback)
    return result.dropna(subset=["total_momentum"])


def form_long_short_portfolio(
    scores: pd.DataFrame,
    date: pd.Timestamp,
    momentum_col: str = "total_momentum",
    n_long: int | None = None,
    n_short: int | None = None,
    quantile: float = 0.3,
) -> pd.DataFrame:
    """
    Form a long-short portfolio for a given date based on momentum scores.

    Goes long the top quantile and short the bottom quantile of stocks
    ranked by momentum score.

    Args:
        scores: DataFrame with date, stock_id, and momentum columns.
        date: The date for which to form the portfolio.
        momentum_col: Column name to rank stocks on.
        n_long: Number of stocks to go long (overrides quantile).
        n_short: Number of stocks to go short (overrides quantile).
        quantile: Fraction of stocks for each leg if n_long/n_short not set.

    Returns:
        DataFrame with columns: stock_id, position (-1 or +1).
    """
    day_scores = scores[scores["date"] == date].copy()
    if day_scores.empty:
        return pd.DataFrame(columns=["stock_id", "position"])

    day_scores = day_scores.dropna(subset=[momentum_col])
    n_stocks = len(day_scores)
    if n_stocks < 2:
        return pd.DataFrame(columns=["stock_id", "position"])

    n_l = n_long or max(1, int(n_stocks * quantile))
    n_s = n_short or max(1, int(n_stocks * quantile))

    ranked = day_scores.sort_values(momentum_col, ascending=False)
    longs = ranked.head(n_l)["stock_id"].values
    shorts = ranked.tail(n_s)["stock_id"].values

    positions = []
    for sid in longs:
        positions.append({"stock_id": sid, "position": 1.0})
    for sid in shorts:
        positions.append({"stock_id": sid, "position": -1.0})

    return pd.DataFrame(positions)


def compute_portfolio_return(
    panel: pd.DataFrame,
    portfolio: pd.DataFrame,
    date: pd.Timestamp,
    next_date: pd.Timestamp,
) -> float:
    """
    Compute the equal-weighted return of a long-short portfolio over one period.

    Args:
        panel: Full panel with date, stock_id, price columns.
        portfolio: DataFrame with stock_id, position columns.
        date: Portfolio formation date.
        next_date: Date at which to measure return.

    Returns:
        Portfolio return (float). Long leg contributes positively,
        short leg negatively.
    """
    if portfolio.empty:
        return 0.0

    prices_now = panel[panel["date"] == date][["stock_id", "price"]]
    prices_next = panel[panel["date"] == next_date][["stock_id", "price"]]

    merged = (
        portfolio.merge(prices_now, on="stock_id", how="inner")
        .rename(columns={"price": "price_now"})
        .merge(prices_next, on="stock_id", how="inner")
        .rename(columns={"price": "price_next"})
    )

    if merged.empty:
        return 0.0

    merged["stock_return"] = (merged["price_next"] - merged["price_now"]) / merged["price_now"]
    # Equal-weighted: each leg gets 1/n_total weight, direction from position sign
    n_positions = len(merged)
    merged["weighted_return"] = merged["position"] * merged["stock_return"] / n_positions
    return float(merged["weighted_return"].sum())


def run_walk_forward_evaluation(
    panel: pd.DataFrame,
    config: BacktestConfig | None = None,
    lookback: int = 12,
    momentum_col: str = "total_momentum",
    quantile: float = 0.3,
) -> list[BacktestResult]:
    """
    Run walk-forward evaluation of a momentum strategy.

    For each walk-forward window:
    1. Use the training period to compute momentum scores.
    2. At each date in the test period, form a long-short portfolio
       based on the most recent momentum scores available.
    3. Compute daily portfolio returns and metrics.

    Args:
        panel: Preprocessed panel with [date, stock_id, industry_id, price].
        config: Backtest configuration. Uses defaults if None.
        lookback: Momentum lookback window in periods.
        momentum_col: Which momentum column to use for portfolio formation.
        quantile: Fraction of stocks for long/short legs.

    Returns:
        List of BacktestResult, one per walk-forward window.
    """
    config = config or BacktestConfig()
    validator = WalkForwardValidator(config)

    # Get unique sorted dates
    dates = sorted(panel["date"].unique())
    date_index = pd.DataFrame({"date": dates}).reset_index()

    results = []
    window_num = 0

    for train_idx, test_idx in validator.split(pd.DataFrame(index=range(len(dates)))):
        train_dates = [dates[i] for i in train_idx]
        test_dates = [dates[i] for i in test_idx]

        if len(train_dates) < lookback + 2 or len(test_dates) < 2:
            continue

        # Compute momentum scores on training data
        train_panel = panel[panel["date"].isin(train_dates)].copy()
        scores = compute_momentum_scores(train_panel, lookback=lookback)

        if scores.empty:
            continue

        # Get the latest available momentum scores (end of training period)
        latest_score_date = scores["date"].max()

        # Form portfolio based on latest training scores
        portfolio = form_long_short_portfolio(
            scores, latest_score_date,
            momentum_col=momentum_col, quantile=quantile,
        )

        if portfolio.empty:
            continue

        # Compute daily returns for the test period
        test_panel = panel[panel["date"].isin(test_dates)].copy()
        test_dates_sorted = sorted(test_dates)

        daily_returns = []
        positions_series = []

        for i in range(len(test_dates_sorted) - 1):
            d_now = test_dates_sorted[i]
            d_next = test_dates_sorted[i + 1]
            ret = compute_portfolio_return(panel, portfolio, d_now, d_next)
            daily_returns.append(ret)

            # Track position changes (static portfolio = same positions each day)
            n_pos = len(portfolio)
            positions_series.append(n_pos)

        if not daily_returns:
            continue

        returns_series = pd.Series(daily_returns, dtype=float)

        # Position series for cost calculation: constant positions (no rebalancing within window)
        pos_series = pd.Series(
            [1.0] * len(daily_returns), dtype=float
        )
        # First day has a position change (entering), rest are held
        pos_series.iloc[0] = 0.0  # signals initial entry

        # Compute gross metrics
        gross_metrics = compute_metrics(returns_series)

        # Compute net returns
        net_returns = calculate_costs(returns_series, pos_series, config)
        net_metrics = compute_metrics(net_returns)

        total_trades = len(portfolio)  # initial entry trades

        result = BacktestResult(
            window=window_num,
            train_start=str(pd.Timestamp(train_dates[0]).date()),
            train_end=str(pd.Timestamp(train_dates[-1]).date()),
            test_start=str(pd.Timestamp(test_dates_sorted[0]).date()),
            test_end=str(pd.Timestamp(test_dates_sorted[-1]).date()),
            gross_sharpe=gross_metrics["sharpeRatio"],
            net_sharpe=net_metrics["sharpeRatio"],
            annual_return=gross_metrics["annualReturn"],
            max_drawdown=gross_metrics["maxDrawdown"],
            total_trades=total_trades,
            hit_rate=gross_metrics["hitRate"],
            pnl_series=returns_series,
        )
        results.append(result)
        window_num += 1

        logger.info(
            "Window %d: train=%s..%s test=%s..%s gross_sharpe=%.3f net_sharpe=%.3f",
            result.window, result.train_start, result.train_end,
            result.test_start, result.test_end,
            result.gross_sharpe, result.net_sharpe,
        )

    return results


def evaluate_momentum_strategy(
    panel: pd.DataFrame,
    config: BacktestConfig | None = None,
    lookback: int = 12,
    momentum_col: str = "total_momentum",
    quantile: float = 0.3,
) -> dict:
    """
    Full evaluation pipeline: run walk-forward and generate metrics JSON.

    Args:
        panel: Preprocessed panel data.
        config: Backtest configuration.
        lookback: Momentum lookback window.
        momentum_col: Momentum column for ranking.
        quantile: Long/short quantile.

    Returns:
        Dict matching the ARF metrics.json schema.
    """
    config = config or BacktestConfig()
    results = run_walk_forward_evaluation(
        panel, config=config, lookback=lookback,
        momentum_col=momentum_col, quantile=quantile,
    )

    custom_metrics = {
        "strategy": f"long_short_{momentum_col}",
        "lookback_periods": lookback,
        "quantile": quantile,
        "momentum_column": momentum_col,
    }

    metrics = generate_metrics_json(results, config, custom_metrics=custom_metrics)
    return metrics


STRATEGY_COLUMNS = ["total_momentum", "industry_momentum", "stock_specific_momentum"]


def compute_factor_correlations(
    panel: pd.DataFrame,
    lookback: int = 12,
) -> dict[str, float]:
    """
    Compute time-series correlations between decomposed momentum factors.

    Calculates the cross-sectional average momentum for each factor at each
    date, then computes the time-series correlation between these averages.

    Args:
        panel: Panel with [date, stock_id, industry_id, price].
        lookback: Momentum lookback window.

    Returns:
        Dict with correlation pairs, e.g. {"industry_vs_stock_specific": -0.12, ...}.
    """
    scores = compute_momentum_scores(panel, lookback=lookback)
    if scores.empty:
        return {
            "industry_vs_stock_specific": 0.0,
            "total_vs_industry": 0.0,
            "total_vs_stock_specific": 0.0,
        }

    # Compute cross-sectional mean at each date
    daily_means = scores.groupby("date")[STRATEGY_COLUMNS].mean().dropna()

    if len(daily_means) < 3:
        return {
            "industry_vs_stock_specific": 0.0,
            "total_vs_industry": 0.0,
            "total_vs_stock_specific": 0.0,
        }

    corr = daily_means.corr()
    return {
        "industry_vs_stock_specific": round(
            float(corr.loc["industry_momentum", "stock_specific_momentum"]), 4
        ),
        "total_vs_industry": round(
            float(corr.loc["total_momentum", "industry_momentum"]), 4
        ),
        "total_vs_stock_specific": round(
            float(corr.loc["total_momentum", "stock_specific_momentum"]), 4
        ),
    }


def run_decomposed_backtest(
    panel: pd.DataFrame,
    config: BacktestConfig | None = None,
    lookback: int = 12,
    quantile: float = 0.3,
) -> dict[str, list[BacktestResult]]:
    """
    Run walk-forward backtest for all 3 decomposed momentum strategies.

    Evaluates total, industry, and stock-specific momentum long-short
    portfolios side by side using the same walk-forward windows.

    Args:
        panel: Preprocessed panel with [date, stock_id, industry_id, price].
        config: Backtest configuration.
        lookback: Momentum lookback window.
        quantile: Fraction of stocks for long/short legs.

    Returns:
        Dict mapping strategy name -> list of BacktestResult.
    """
    config = config or BacktestConfig()
    results = {}
    for col in STRATEGY_COLUMNS:
        logger.info("Running walk-forward for strategy: %s", col)
        results[col] = run_walk_forward_evaluation(
            panel, config=config, lookback=lookback,
            momentum_col=col, quantile=quantile,
        )
    return results


def compare_strategies(
    all_results: dict[str, list[BacktestResult]],
    config: BacktestConfig | None = None,
) -> dict[str, dict]:
    """
    Generate metrics for each strategy and a comparison summary.

    Args:
        all_results: Dict mapping strategy name -> list of BacktestResult.
        config: Backtest configuration.

    Returns:
        Dict with per-strategy metrics keyed by strategy name.
    """
    config = config or BacktestConfig()
    comparison = {}
    for strategy_name, results in all_results.items():
        custom = {"strategy": f"long_short_{strategy_name}"}
        metrics = generate_metrics_json(results, config, custom_metrics=custom)
        comparison[strategy_name] = metrics
    return comparison


def evaluate_decomposed_strategies(
    panel: pd.DataFrame,
    config: BacktestConfig | None = None,
    lookback: int = 12,
    quantile: float = 0.3,
) -> dict:
    """
    Full decomposed backtest: run all 3 strategies and generate combined metrics.

    Returns a metrics dict conforming to the ARF schema, with per-strategy
    breakdowns and factor correlations in customMetrics.

    Args:
        panel: Preprocessed panel data.
        config: Backtest configuration.
        lookback: Momentum lookback window.
        quantile: Long/short quantile.

    Returns:
        Dict matching the ARF metrics.json schema with extended customMetrics.
    """
    config = config or BacktestConfig()

    # Run all 3 strategies
    all_results = run_decomposed_backtest(
        panel, config=config, lookback=lookback, quantile=quantile,
    )

    # Compare strategies
    comparison = compare_strategies(all_results, config)

    # Compute factor correlations
    correlations = compute_factor_correlations(panel, lookback=lookback)

    # Find best strategy by net Sharpe
    best_strategy = max(
        comparison.keys(),
        key=lambda k: comparison[k]["transactionCosts"]["netSharpe"],
    )
    best_metrics = comparison[best_strategy]

    # Build per-strategy summary
    strategy_summary = {}
    for name, m in comparison.items():
        strategy_summary[name] = {
            "grossSharpe": m["sharpeRatio"],
            "netSharpe": m["transactionCosts"]["netSharpe"],
            "annualReturn": m["annualReturn"],
            "maxDrawdown": m["maxDrawdown"],
            "hitRate": m["hitRate"],
            "windows": m["walkForward"]["windows"],
            "positiveWindows": m["walkForward"]["positiveWindows"],
        }

    # Use best strategy for top-level metrics
    metrics = {
        "sharpeRatio": best_metrics["sharpeRatio"],
        "annualReturn": best_metrics["annualReturn"],
        "maxDrawdown": best_metrics["maxDrawdown"],
        "hitRate": best_metrics["hitRate"],
        "totalTrades": best_metrics["totalTrades"],
        "transactionCosts": best_metrics["transactionCosts"],
        "walkForward": best_metrics["walkForward"],
        "customMetrics": {
            "bestStrategy": best_strategy,
            "lookback_periods": lookback,
            "quantile": quantile,
            "factorCorrelations": correlations,
            "strategyComparison": strategy_summary,
        },
    }
    return metrics


def run_cost_sensitivity_analysis(
    panel: pd.DataFrame,
    config: BacktestConfig | None = None,
    lookback: int = 12,
    quantile: float = 0.3,
    scenarios: dict[str, dict] | None = None,
) -> dict[str, dict[str, dict]]:
    """
    Run all 3 momentum strategies under multiple cost scenarios.

    For each cost scenario, runs decomposed backtest and collects metrics.
    This enables comparing gross vs net returns and understanding
    how sensitive each strategy is to transaction costs.

    Args:
        panel: Preprocessed panel with [date, stock_id, industry_id, price].
        config: Base backtest configuration (cost params will be overridden).
        lookback: Momentum lookback window.
        quantile: Long/short quantile.
        scenarios: Dict of scenario_name -> {"fee_bps": ..., "slippage_bps": ...}.
                   Defaults to COST_SCENARIOS.

    Returns:
        Dict: scenario_name -> strategy_name -> ARF metrics dict.
    """
    config = config or BacktestConfig()
    scenarios = scenarios or COST_SCENARIOS

    results_by_scenario = {}

    # First run the walk-forward once (gross returns are the same across scenarios)
    all_results = run_decomposed_backtest(
        panel, config=config, lookback=lookback, quantile=quantile,
    )

    for scenario_name, scenario_params in scenarios.items():
        scenario_config = BacktestConfig(
            fee_bps=scenario_params["fee_bps"],
            slippage_bps=scenario_params["slippage_bps"],
            train_ratio=config.train_ratio,
            n_splits=config.n_splits,
            gap=config.gap,
            min_train_size=config.min_train_size,
        )

        strategy_metrics = {}
        for strategy_name, results in all_results.items():
            # Recompute net returns under this cost scenario
            adjusted_results = []
            for r in results:
                if r.pnl_series is not None and len(r.pnl_series) > 0:
                    pos_series = pd.Series([1.0] * len(r.pnl_series), dtype=float)
                    pos_series.iloc[0] = 0.0
                    net_returns = calculate_costs(r.pnl_series, pos_series, scenario_config)
                    net_metrics = compute_metrics(net_returns)
                    adjusted = BacktestResult(
                        window=r.window,
                        train_start=r.train_start,
                        train_end=r.train_end,
                        test_start=r.test_start,
                        test_end=r.test_end,
                        gross_sharpe=r.gross_sharpe,
                        net_sharpe=net_metrics["sharpeRatio"],
                        annual_return=r.annual_return,
                        max_drawdown=r.max_drawdown,
                        total_trades=r.total_trades,
                        hit_rate=r.hit_rate,
                        pnl_series=r.pnl_series,
                    )
                else:
                    adjusted = r
                adjusted_results.append(adjusted)

            custom = {"strategy": f"long_short_{strategy_name}", "scenario": scenario_name}
            metrics = generate_metrics_json(adjusted_results, scenario_config, custom_metrics=custom)
            strategy_metrics[strategy_name] = metrics

        results_by_scenario[scenario_name] = strategy_metrics
        logger.info("Scenario %s complete", scenario_name)

    return results_by_scenario


def compute_cost_impact(
    sensitivity_results: dict[str, dict[str, dict]],
) -> dict[str, dict]:
    """
    Compute the impact of transaction costs on each strategy.

    Compares gross (zero-cost) performance to each cost scenario,
    quantifying the Sharpe degradation and return drag.

    Args:
        sensitivity_results: Output of run_cost_sensitivity_analysis.

    Returns:
        Dict: strategy_name -> {
            "grossSharpe": ..., "grossReturn": ...,
            "scenarios": {scenario_name -> {"netSharpe": ..., "sharpeDelta": ..., ...}}
        }
    """
    if "zero" not in sensitivity_results:
        return {}

    zero_scenario = sensitivity_results["zero"]
    impact = {}

    for strategy_name in STRATEGY_COLUMNS:
        if strategy_name not in zero_scenario:
            continue

        gross_sharpe = zero_scenario[strategy_name]["sharpeRatio"]
        gross_return = zero_scenario[strategy_name]["annualReturn"]

        scenario_impacts = {}
        for scenario_name, strategies in sensitivity_results.items():
            if scenario_name == "zero":
                continue
            if strategy_name not in strategies:
                continue

            net_sharpe = strategies[strategy_name]["transactionCosts"]["netSharpe"]
            net_return = strategies[strategy_name]["annualReturn"]
            total_bps = (
                strategies[strategy_name]["transactionCosts"]["feeBps"]
                + strategies[strategy_name]["transactionCosts"]["slippageBps"]
            )

            scenario_impacts[scenario_name] = {
                "totalCostBps": total_bps,
                "netSharpe": net_sharpe,
                "sharpeDelta": round(net_sharpe - gross_sharpe, 4),
                "returnDragPct": round((gross_return - net_return) * 100, 4) if gross_return != 0 else 0.0,
            }

        impact[strategy_name] = {
            "grossSharpe": gross_sharpe,
            "grossReturn": gross_return,
            "scenarios": scenario_impacts,
        }

    return impact


def compute_breakeven_cost(
    panel: pd.DataFrame,
    config: BacktestConfig | None = None,
    lookback: int = 12,
    quantile: float = 0.3,
    max_cost_bps: float = 100.0,
    step_bps: float = 5.0,
) -> dict[str, float]:
    """
    Find the breakeven transaction cost level for each strategy.

    The breakeven cost is the total cost (fee + slippage) in bps at which
    the strategy's net Sharpe drops to zero.

    Args:
        panel: Preprocessed panel data.
        config: Base backtest configuration.
        lookback: Momentum lookback window.
        quantile: Long/short quantile.
        max_cost_bps: Maximum cost to test.
        step_bps: Step size for cost grid search.

    Returns:
        Dict: strategy_name -> breakeven cost in bps (0.0 if never profitable).
    """
    config = config or BacktestConfig()

    # Run walk-forward once to get gross PnL series
    all_results = run_decomposed_backtest(
        panel, config=config, lookback=lookback, quantile=quantile,
    )

    breakeven = {}
    cost_levels = np.arange(0, max_cost_bps + step_bps, step_bps)

    for strategy_name, results in all_results.items():
        if not results:
            breakeven[strategy_name] = 0.0
            continue

        # Check if gross Sharpe is positive
        gross_sharpes = [r.gross_sharpe for r in results]
        avg_gross = float(np.mean(gross_sharpes))
        if avg_gross <= 0:
            breakeven[strategy_name] = 0.0
            continue

        # Binary-search style: find where net Sharpe crosses zero
        be_cost = 0.0
        for cost_bps in cost_levels:
            test_config = BacktestConfig(
                fee_bps=cost_bps * 0.67,  # 2:1 fee:slippage ratio
                slippage_bps=cost_bps * 0.33,
                train_ratio=config.train_ratio,
                n_splits=config.n_splits,
                gap=config.gap,
                min_train_size=config.min_train_size,
            )
            net_sharpes = []
            for r in results:
                if r.pnl_series is not None and len(r.pnl_series) > 0:
                    pos_series = pd.Series([1.0] * len(r.pnl_series), dtype=float)
                    pos_series.iloc[0] = 0.0
                    net_ret = calculate_costs(r.pnl_series, pos_series, test_config)
                    nm = compute_metrics(net_ret)
                    net_sharpes.append(nm["sharpeRatio"])
                else:
                    net_sharpes.append(0.0)

            avg_net = float(np.mean(net_sharpes))
            if avg_net <= 0:
                # Interpolate between previous and current
                if cost_bps > 0 and be_cost == 0.0:
                    be_cost = cost_bps - step_bps / 2  # approximate
                break
            be_cost = cost_bps

        breakeven[strategy_name] = round(be_cost, 1)

    return breakeven


def evaluate_transaction_costs(
    panel: pd.DataFrame,
    config: BacktestConfig | None = None,
    lookback: int = 12,
    quantile: float = 0.3,
) -> dict:
    """
    Full Phase 5 pipeline: transaction cost analysis across strategies.

    Runs cost sensitivity analysis, computes cost impact, and finds
    breakeven costs. Returns ARF-schema metrics with extended customMetrics.

    Args:
        panel: Preprocessed panel data.
        config: Backtest configuration.
        lookback: Momentum lookback window.
        quantile: Long/short quantile.

    Returns:
        Dict matching ARF metrics.json schema with cost analysis in customMetrics.
    """
    config = config or BacktestConfig()

    logger.info("Running cost sensitivity analysis...")
    sensitivity = run_cost_sensitivity_analysis(
        panel, config=config, lookback=lookback, quantile=quantile,
    )

    logger.info("Computing cost impact...")
    cost_impact = compute_cost_impact(sensitivity)

    logger.info("Computing breakeven costs...")
    breakeven = compute_breakeven_cost(
        panel, config=config, lookback=lookback, quantile=quantile,
    )

    # Use medium cost scenario as the primary result
    medium = sensitivity.get("medium", {})

    # Find best strategy under medium costs
    best_strategy = None
    best_net_sharpe = -999.0
    for sname, metrics in medium.items():
        ns = metrics["transactionCosts"]["netSharpe"]
        if ns > best_net_sharpe:
            best_net_sharpe = ns
            best_strategy = sname

    if not best_strategy or best_strategy not in medium:
        # Fallback to base evaluation
        return evaluate_decomposed_strategies(panel, config=config, lookback=lookback, quantile=quantile)

    best_metrics = medium[best_strategy]

    # Build gross vs net comparison table
    gross_vs_net = {}
    for sname in STRATEGY_COLUMNS:
        if sname in sensitivity.get("zero", {}) and sname in medium:
            zero_m = sensitivity["zero"][sname]
            med_m = medium[sname]
            gross_vs_net[sname] = {
                "grossSharpe": zero_m["sharpeRatio"],
                "grossReturn": zero_m["annualReturn"],
                "netSharpe": med_m["transactionCosts"]["netSharpe"],
                "netReturn": med_m["annualReturn"],
                "sharpeDegradation": round(
                    zero_m["sharpeRatio"] - med_m["transactionCosts"]["netSharpe"], 4
                ),
                "breakevenCostBps": breakeven.get(sname, 0.0),
            }

    # Build per-scenario summary for best strategy
    scenario_summary = {}
    for scenario_name, strategies in sensitivity.items():
        if best_strategy in strategies:
            m = strategies[best_strategy]
            scenario_summary[scenario_name] = {
                "feeBps": m["transactionCosts"]["feeBps"],
                "slippageBps": m["transactionCosts"]["slippageBps"],
                "totalCostBps": m["transactionCosts"]["feeBps"] + m["transactionCosts"]["slippageBps"],
                "netSharpe": m["transactionCosts"]["netSharpe"],
                "netReturn": m["annualReturn"],
            }

    metrics = {
        "sharpeRatio": best_metrics["sharpeRatio"],
        "annualReturn": best_metrics["annualReturn"],
        "maxDrawdown": best_metrics["maxDrawdown"],
        "hitRate": best_metrics["hitRate"],
        "totalTrades": best_metrics["totalTrades"],
        "transactionCosts": best_metrics["transactionCosts"],
        "walkForward": best_metrics["walkForward"],
        "customMetrics": {
            "bestStrategy": best_strategy,
            "lookback_periods": lookback,
            "quantile": quantile,
            "grossVsNet": gross_vs_net,
            "costSensitivity": scenario_summary,
            "breakevenCosts": breakeven,
            "costImpact": cost_impact,
        },
    }
    return metrics


def save_metrics(metrics: dict, cycle: int = 4) -> Path:
    """Save metrics.json to the reports directory."""
    out_dir = REPORTS_DIR / f"cycle_{cycle}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "metrics.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved metrics to %s", out_path)
    return out_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Load panel data
    from src.data_fetcher import build_panel, DATA_DIR

    panel_path = DATA_DIR / "jp_stocks_panel.csv"
    if panel_path.exists():
        print("Loading cached panel data...")
        panel = pd.read_csv(panel_path, parse_dates=["date"])
    else:
        print("Building panel from API...")
        panel, report = build_panel()
        panel.to_csv(panel_path, index=False)

    # Run transaction cost analysis (Phase 5)
    print("\nRunning transaction cost analysis (all 3 strategies x 5 cost scenarios)...")
    config = BacktestConfig(n_splits=5, min_train_size=60)
    metrics = evaluate_transaction_costs(panel, config=config, lookback=12)

    # Save results
    save_metrics(metrics, cycle=5)
    print("\n=== Metrics ===")
    print(json.dumps(metrics, indent=2))
