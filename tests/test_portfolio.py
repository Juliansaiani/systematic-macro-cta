"""Tests for position sizing, leverage clamping and portfolio accounting."""
import numpy as np
import pandas as pd
import pytest

from macro_cta.portfolio import (
    compute_positions,
    simulate_strategy,
    score_performance,
    rolling_positions,
)


def sample_data(n=300):
    rng = np.random.default_rng(123)
    dates = pd.bdate_range("2019-01-01", periods=n)
    cols = ["A1", "A2", "A3"]
    rets = pd.DataFrame(rng.normal(0.0002, 0.015, size=(n, 3)), index=dates, columns=cols)
    sigs = pd.DataFrame(rng.uniform(-1.5, 1.5, size=(n, 3)), index=dates, columns=cols)
    return rets, sigs


def test_position_sizing_respects_leverage_and_clipping():
    rets, sigs = sample_data()
    config = {
        "vol_lookback": 30,
        "target_vol_instrument": 0.15,
        "position_clip": 2.5,
        "max_leverage": 4.0,
    }
    pos = compute_positions(sigs, rets, config).dropna()
    assert (pos.abs() <= 2.5 + 1e-9).all().all()
    assert (pos.abs().sum(axis=1) <= 4.0 + 1e-9).all()


def test_simulation_accounting_and_friction():
    rets, sigs = sample_data()
    config = {
        "vol_lookback": 30,
        "target_vol_instrument": 0.15,
        "position_clip": 2.5,
        "max_leverage": 4.0,
    }
    pos = compute_positions(sigs, rets, config)
    sim_0 = simulate_strategy(rets, pos, cost_bps=0.0)
    sim_10 = simulate_strategy(rets, pos, cost_bps=10.0)

    # 10 bps costs must strictly reduce cumulative wealth compared to 0 bps
    assert sim_10.net_return.sum() < sim_0.net_return.sum()
    assert (sim_10.costs >= 0).all()

    scores = score_performance(sim_10.net_return)
    assert "sharpe" in scores
    assert "max_drawdown" in scores
    assert scores["max_drawdown"] <= 0


def test_self_financing_ledger_matches_hand_calculation():
    dates = pd.bdate_range("2020-01-01", periods=4)
    returns = pd.DataFrame({"ETF": [0.0, 0.0, 0.10, -0.10]}, index=dates)
    targets = pd.DataFrame({"ETF": [1.0] * 4}, index=dates)
    result = simulate_strategy(returns, targets, cost_bps=100, target_vol_portfolio=10,
                               max_leverage=0.5, position_clip=0.5, evaluation_start=dates[1])
    # Buy a 50% post-fee allocation: fee = .01 * .5 * (1-fee).
    entry_fee = .005 / 1.005
    # After a 10% rise: risky holdings are .5*1.1 / 1.05 of NAV.
    drifted = .5 * 1.1 / 1.05
    sell_fee = .01 * (drifted - .5) / (1 - .005)
    expected_net = [-entry_fee, .05, (1 - sell_fee) * .95 - 1]
    np.testing.assert_allclose(result.net_return, expected_net, atol=1e-13)
    np.testing.assert_allclose(result.costs, [entry_fee, 0, sell_fee], atol=1e-13)
    np.testing.assert_allclose(result.positions, .5, atol=1e-12)
    np.testing.assert_allclose(result.costs, result.turnover * .01, atol=1e-13)
    np.testing.assert_allclose(result.net_return, result.gross_return - result.costs, atol=1e-13)


def test_final_exposure_limits_and_costs_include_volatility_resizing():
    dates = pd.bdate_range("2020-01-01", periods=50)
    returns = pd.DataFrame(np.tile([.0001, -.0002], (50, 1)), index=dates, columns=["A", "B"])
    targets = pd.DataFrame(3.0, index=dates, columns=returns.columns)
    sim = simulate_strategy(returns, targets, 5, target_vol_portfolio=.15, max_leverage=5, position_clip=3)
    assert sim.positions.abs().sum(axis=1).max() <= 5 + 1e-12
    assert sim.positions.abs().max().max() <= 3 + 1e-12
    # Entry is charged on final scaled holdings, not the unscaled source targets.
    fee = .0005 * 5 / (1 + .0005 * 5)
    assert sim.costs.iloc[1] == pytest.approx(fee, abs=1e-12)


def test_simple_return_compounding_and_arithmetic_sharpe():
    r = pd.Series([.1, -.1, .1])
    s = score_performance(r, rf_annual=0)
    assert s["total_return"] == pytest.approx(1.1 * .9 * 1.1 - 1)
    assert s["cagr"] == pytest.approx((1.1 * .9 * 1.1) ** (252 / 3) - 1)
    assert s["sharpe"] == pytest.approx(r.mean() / r.std(ddof=1) * np.sqrt(252))
    assert s["max_drawdown"] == pytest.approx(-.1)
    assert score_performance(pd.Series([-.1, 0.0]), rf_annual=0)["max_drawdown"] == pytest.approx(-.1)


def test_future_returns_cannot_change_past_executions():
    returns, targets = sample_data(n=100)
    altered = returns.copy()
    altered.iloc[70:] *= -3
    before = simulate_strategy(returns, targets, 5)
    after = simulate_strategy(altered, targets, 5)
    pd.testing.assert_frame_equal(before.positions.iloc[:71], after.positions.iloc[:71])
    pd.testing.assert_series_equal(before.net_return.iloc[:70], after.net_return.iloc[:70])


def test_rolling_signals_are_causal_and_execute_next_observation():
    from macro_cta.models import compute_ewma_tsmom_signals
    returns, _ = sample_data(n=320)
    model = {"horizons": [{"label": "Test", "fast_d": 10, "slow_d": 30, "weight": 1}],
             "vol_standardization": 30}
    portfolio = {"train_window": 252, "retrain_every": 21}
    targets = rolling_positions(returns, portfolio, model)
    _, first_signal = compute_ewma_tsmom_signals(returns.iloc[:252], model)
    expected = compute_positions(first_signal, returns.iloc[:252], portfolio).iloc[-1]
    np.testing.assert_allclose(targets.iloc[251], expected)
    assert targets.iloc[:251].isna().all().all()
    pd.testing.assert_frame_equal(targets.iloc[251:272], pd.DataFrame(
        np.tile(expected, (21, 1)), index=targets.index[251:272], columns=targets.columns))
    altered = returns.copy()
    altered.iloc[280:] *= -2
    updated = rolling_positions(altered, portfolio, model)
    pd.testing.assert_frame_equal(targets.iloc[:280], updated.iloc[:280])
    sim = simulate_strategy(np.expm1(returns), targets, 5, evaluation_start=returns.index[252])
    assert sim.positions.index[0] == returns.index[252]
    assert sim.turnover.iloc[0] > 0
