"""Tests for pipeline execution and verification harness."""
from pathlib import Path
import pandas as pd
import pytest
import json
import numpy as np

from macro_cta.pipeline import verify_reference


def test_pipeline_exports_and_verifies_offline(tmp_path, monkeypatch):
    from macro_cta import pipeline

    config = json.loads((Path(__file__).resolve().parents[1] / "config.json").read_text())
    config["data"]["universe"] = {
        "EQ": {"class": "equity"}, "BD": {"class": "bond"},
    }
    config["model"]["horizons"] = [
        {"label": "Test", "fast_d": 10, "slow_d": 30, "weight": 1.0},
    ]
    config["model"]["vol_standardization"] = 30
    config["portfolio"]["train_window"] = 252
    config["portfolio"]["retrain_every"] = 21
    (tmp_path / "config.json").write_text(json.dumps(config))
    returns = pd.DataFrame(
        np.random.default_rng(42).normal(0.0002, 0.01, (340, 2)),
        index=pd.bdate_range("2019-01-01", periods=340), columns=["EQ", "BD"],
    )
    prices = 100 * np.exp(returns.cumsum())
    monkeypatch.setattr(pipeline, "load_market_data", lambda *a, **k: (prices, returns))
    pipeline.run(tmp_path, write_reference=True)
    pipeline.run(tmp_path, verify=True)
    artifacts = tmp_path / "artifacts"
    metrics = pd.read_csv(artifacts / "metrics.csv")
    assert len(metrics) == 2 * len(config["cost_bps"])
    assert metrics.days.nunique() == 1
    assert metrics.max_gross_leverage.max() <= config["portfolio"]["max_leverage"] + 1e-12
    assert metrics.annual_turnover.gt(0).all()
    assert np.isfinite(metrics[["sharpe", "cagr", "ann_vol", "max_drawdown"]]).all().all()
    report = pd.read_csv(artifacts / "verification.csv")
    assert len(report) == 2 * len(config["cost_bps"])
    assert report.filter(like="_error").max().max() <= 1e-6
    assert (artifacts / "overview.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert not pd.read_csv(artifacts / "crisis_returns.csv").empty
    reference_path = tmp_path / config["verification"]["reference"]
    reference = pd.read_csv(reference_path)
    reference.loc[0, "sharpe"] += 0.1
    reference.to_csv(reference_path, index=False)
    before = {p.name: p.read_bytes() for p in artifacts.iterdir()}
    with pytest.raises(AssertionError, match="Reference mismatch"):
        pipeline.run(tmp_path, verify=True)
    assert before == {p.name: p.read_bytes() for p in artifacts.iterdir()}


def test_verify_missing_reference_never_creates_baseline(tmp_path):
    from macro_cta import pipeline
    config = json.loads((Path(__file__).resolve().parents[1] / "config.json").read_text())
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(FileNotFoundError, match="never creates"):
        pipeline.run(tmp_path, verify=True)
    assert not (tmp_path / config["verification"]["reference"]).exists()
    assert not (tmp_path / "artifacts").exists()


def test_verify_requires_all_metrics():
    incomplete = pd.DataFrame([{"variant": "x", "cost_bps": 0}])
    with pytest.raises(ValueError, match="Missing required"):
        verify_reference(incomplete, incomplete)


def test_verify_reference_detects_mismatch():
    actual = pd.DataFrame([
        {"variant": "In-Sample", "cost_bps": 5, "sharpe": 0.50, "cagr": 0.08, "ann_vol": 0.15, "max_drawdown": -0.20},
    ])
    expected = pd.DataFrame([
        {"variant": "In-Sample", "cost_bps": 5, "sharpe": 0.55, "cagr": 0.08, "ann_vol": 0.15, "max_drawdown": -0.20},
    ])
    with pytest.raises(AssertionError):
        verify_reference(actual, expected, tolerance=1e-4)


def test_verify_reference_passes_within_tolerance():
    actual = pd.DataFrame([
        {"variant": "In-Sample", "cost_bps": 5, "sharpe": 0.5000001, "cagr": 0.08, "ann_vol": 0.15, "max_drawdown": -0.20},
    ])
    expected = pd.DataFrame([
        {"variant": "In-Sample", "cost_bps": 5, "sharpe": 0.5000000, "cagr": 0.08, "ann_vol": 0.15, "max_drawdown": -0.20},
    ])
    report = verify_reference(actual, expected, tolerance=1e-5)
    assert not report.empty
