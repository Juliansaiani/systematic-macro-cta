"""Tests for multi-horizon trend signals and TSMOM blending."""
import numpy as np
import pandas as pd
import pytest

from macro_cta.models import single_asset_signal, compute_ewma_tsmom_signals


def sample_returns(n=400):
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2018-01-01", periods=n)
    values = rng.normal(0.0003, 0.01, size=(n, 4))
    return pd.DataFrame(values, index=dates, columns=["EQ1", "EQ2", "BD1", "COM1"])


def test_single_asset_signal_properties():
    ret = sample_returns()["EQ1"]
    sig = single_asset_signal(ret, fast_d=20, slow_d=60)
    assert len(sig) == len(ret)
    valid = sig.dropna()
    assert len(valid) > 0
    assert np.isfinite(valid).all()


def test_multi_horizon_composite_signals():
    df = sample_returns()
    config = {
        "horizons": [
            {"label": "Short", "fast_d": 10, "slow_d": 30, "weight": 0.5},
            {"label": "Medium", "fast_d": 20, "slow_d": 60, "weight": 0.5},
        ],
        "vol_lookback": 30,
        "vol_standardization": 60,
        "blend_tsmom_weight": 0.5,
    }
    horizons, composite = compute_ewma_tsmom_signals(df, config)
    assert len(horizons) == 2
    assert composite.shape == df.shape
    assert composite.index.equals(df.index)
