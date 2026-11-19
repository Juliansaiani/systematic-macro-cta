"""Causal position sizing and self-financing daily ETF portfolio accounting."""
from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class SimulationResult:
    net_return: pd.Series
    gross_return: pd.Series
    turnover: pd.Series
    costs: pd.Series
    positions: pd.DataFrame


def compute_positions(signals: pd.DataFrame, returns: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """Compute close-of-day targets; execution must lag them by one observation."""
    vol = returns.ewm(span=config.get("vol_lookback", 63), adjust=False).std() * np.sqrt(252)
    positions = signals * config.get("target_vol_instrument", 0.15) / (vol.replace(0, np.nan).ffill() + 1e-9)
    positions = positions.clip(-config.get("position_clip", 3.0), config.get("position_clip", 3.0))
    scale = np.minimum(1.0, config.get("max_leverage", 3.0) / (positions.abs().sum(axis=1) + 1e-9))
    return positions.mul(scale, axis=0)


def rolling_positions(returns: pd.DataFrame, portfolio_config: Dict, model_config: Dict) -> pd.DataFrame:
    """Refresh close-of-day signals every 21 observations using a trailing window.

    This is a fixed-rule rolling-history variant, not parameter fitting or selection.
    Targets at close t use observations through t and can earn returns only on t+1.
    """
    from .models import compute_ewma_tsmom_signals

    window_size = portfolio_config.get("train_window", 504)
    step = portfolio_config.get("retrain_every", 21)
    if window_size < 2 or step < 1:
        raise ValueError("Rolling window and refresh interval must be positive.")
    positions = pd.DataFrame(np.nan, index=returns.index, columns=returns.columns)
    for t in range(window_size - 1, len(returns), step):
        window = returns.iloc[t - window_size + 1:t + 1]
        _, composite = compute_ewma_tsmom_signals(window, model_config)
        target = compute_positions(composite, window, portfolio_config).iloc[-1]
        positions.iloc[t:min(t + step, len(returns))] = target.to_numpy()
    return positions


def _rebalance_fee(target: np.ndarray, drifted: np.ndarray, rate: float) -> tuple[float, float]:
    """Solve fee = rate * sum(abs((1-fee)*target - old holdings)).

    Targets are fractions of NAV after transaction costs. Turnover and the fee
    are fractions of NAV before transaction costs, including holding drift.
    """
    fee = 0.0
    for _ in range(100):
        turnover = float(np.abs((1.0 - fee) * target - drifted).sum())
        next_fee = rate * turnover
        if next_fee >= 1.0:
            raise ValueError("Rebalancing costs exhaust portfolio equity.")
        if abs(next_fee - fee) <= 1e-14:
            return next_fee, turnover
        fee = next_fee
    raise RuntimeError("Transaction-cost accounting did not converge.")


def simulate_strategy(
    returns: pd.DataFrame,
    positions: pd.DataFrame,
    cost_bps: float,
    target_vol_portfolio: float = 0.15,
    max_leverage: float = 3.0,
    position_clip: float = 3.0,
    evaluation_start=None,
) -> SimulationResult:
    """Simulate SIMPLE asset returns using lagged, executable portfolio weights.

    Historical baseline P&L estimates the volatility multiplier using only data
    through t-1. Final weights are capped AFTER that multiplier. Portfolios start
    in cash on evaluation_start; entry trades and subsequent drift trades cost.
    Cash earns zero; funding, borrow fees and market impact are excluded.
    """
    if cost_bps < 0 or max_leverage <= 0 or position_clip <= 0 or target_vol_portfolio <= 0:
        raise ValueError("Invalid costs or portfolio risk limits.")
    if cost_bps / 10000.0 * max_leverage >= 1:
        raise ValueError("Costs and leverage do not allow a stable fee calculation.")
    if not returns.index.equals(positions.index) or not returns.columns.equals(positions.columns):
        raise ValueError("Returns and target positions must have identical axes.")
    if not np.isfinite(returns.to_numpy()).all() or (returns <= -1).any().any():
        raise ValueError("Asset returns must be finite SIMPLE returns greater than -1.")
    if not returns.index.is_unique or not returns.index.is_monotonic_increasing:
        raise ValueError("Return dates must be unique and increasing.")
    lagged = positions.shift(1)
    baseline = (lagged * returns).sum(axis=1)
    historical_vol = baseline.ewm(span=63, adjust=False).std() * np.sqrt(252)
    scale = (target_vol_portfolio / (historical_vol + 1e-9)).clip(0.1, 3.0).shift(1).fillna(1.0)
    executed = lagged.mul(scale, axis=0).clip(-position_clip, position_clip)
    gross = executed.abs().sum(axis=1)
    executed = executed.mul(np.minimum(1.0, max_leverage / (gross + 1e-9)), axis=0).fillna(0.0)
    if evaluation_start is not None:
        executed = executed.loc[evaluation_start:]
    if executed.empty:
        raise ValueError("No observations in evaluation period.")
    asset_returns = returns.loc[executed.index]
    drifted = np.zeros(len(executed.columns))
    net, gross_pnl, turnovers, fees = [], [], [], []
    for target, asset_return in zip(executed.to_numpy(), asset_returns.to_numpy()):
        fee, turnover = _rebalance_fee(target, drifted, cost_bps / 10000.0)
        pnl_after_fee = float(target @ asset_return)
        if pnl_after_fee <= -1:
            raise ValueError("Portfolio equity exhausted; leveraged strategy became insolvent.")
        gross_pnl.append((1.0 - fee) * pnl_after_fee)
        net.append((1.0 - fee) * (1.0 + pnl_after_fee) - 1.0)
        turnovers.append(turnover)
        fees.append(fee)
        drifted = target * (1.0 + asset_return) / (1.0 + pnl_after_fee)
    idx = executed.index
    return SimulationResult(
        net_return=pd.Series(net, index=idx),
        gross_return=pd.Series(gross_pnl, index=idx),
        turnover=pd.Series(turnovers, index=idx),
        costs=pd.Series(fees, index=idx),
        positions=executed,
    )


def run_walk_forward(returns: pd.DataFrame, portfolio_config: Dict, model_config: Dict, cost_bps: float) -> pd.Series:
    """Compatibility wrapper: input is log returns; output is net simple return."""
    positions = rolling_positions(returns, portfolio_config, model_config)
    active = positions.shift(1).notna().all(axis=1)
    if not active.any():
        raise ValueError("Not enough history for rolling signals.")
    return simulate_strategy(
        np.expm1(returns), positions, cost_bps,
        target_vol_portfolio=portfolio_config.get("target_vol_portfolio", 0.15),
        max_leverage=portfolio_config.get("max_leverage", 3.0),
        position_clip=portfolio_config.get("position_clip", 3.0),
        evaluation_start=active[active].index[0],
    ).net_return


def score_performance(returns: pd.Series, rf_annual: float = 0.0) -> Dict[str, float]:
    """Compound simple returns; annualize arithmetic daily excess-return Sharpe."""
    values = returns.to_numpy(dtype=float)
    if len(values) < 2 or not np.isfinite(values).all() or (values <= -1).any():
        raise ValueError("At least two finite, solvent daily simple returns are required.")
    wealth = np.r_[1.0, np.cumprod(1.0 + values)]
    vol = float(values.std(ddof=1) * np.sqrt(252))
    rf_daily = np.expm1(np.log1p(rf_annual) / 252)
    sharpe = float((values.mean() - rf_daily) * 252 / vol) if vol > 0 else 0.0
    return {
        "days": len(values),
        "sharpe": sharpe,
        "cagr": float(wealth[-1] ** (252 / len(values)) - 1.0),
        "ann_vol": vol,
        "max_drawdown": float((wealth / np.maximum.accumulate(wealth) - 1.0).min()),
        "skewness": float(stats.skew(values)) if values.std() > 0 else 0.0,
        "total_return": float(wealth[-1] - 1.0),
    }
