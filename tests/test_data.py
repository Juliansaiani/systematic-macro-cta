"""Tests for market data ingestion and alignment."""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from macro_cta.data import load_market_data


def test_market_data_caching_and_alignment(tmp_path, monkeypatch):
    root = tmp_path
    dates = pd.bdate_range("2020-01-01", periods=20)
    calls = []

    def download(ticker, **kwargs):
        calls.append(ticker)
        return pd.DataFrame({"Close": np.linspace(100, 110, len(dates))}, index=dates)

    monkeypatch.setattr("macro_cta.data.yf.download", download)
    config = {
        "cache_dir": "data/cache",
        "start_date": "2015-01-01",
        "universe": {
            "SP500": {"ticker": "SPY", "class": "equity"},
            "US10Y": {"ticker": "IEF", "class": "bond"},
            "GOLD": {"ticker": "GLD", "class": "commodity"},
        },
    }
    price_df, ret_df = load_market_data(root, config, use_cache=True)

    assert isinstance(price_df, pd.DataFrame)
    assert isinstance(ret_df, pd.DataFrame)
    assert not price_df.empty
    assert not ret_df.empty
    assert price_df.index.equals(ret_df.index)
    assert np.isfinite(ret_df.values).all()
    assert list(ret_df.columns) == list(config["universe"])
    assert calls == ["SPY", "IEF", "GLD"]
    simple = np.expm1(ret_df)
    pd.testing.assert_frame_equal(simple.iloc[1:], price_df.pct_change().iloc[1:], atol=1e-13, rtol=0)
    cached_prices, cached_returns = load_market_data(root, config, use_cache=True)
    pd.testing.assert_frame_equal(cached_prices, price_df, check_freq=False)
    pd.testing.assert_frame_equal(cached_returns, ret_df, check_freq=False)
    assert calls == ["SPY", "IEF", "GLD"]
    wrong = {**config, "universe": {"SP500": config["universe"]["SP500"]}}
    with pytest.raises(ValueError, match="universe differs"):
        load_market_data(root, wrong)
    ret_df.iloc[1, 0] += .1
    ret_df.to_csv(root / "data/cache/returns.csv")
    with pytest.raises(ValueError, match="inconsistent"):
        load_market_data(root, config)


def test_end_date_is_inclusive_for_download_and_existing_cache(tmp_path, monkeypatch):
    dates = pd.bdate_range("2020-01-01", periods=10)
    full = pd.DataFrame({"Close": np.arange(100., 110.)}, index=dates)
    ends = []

    def download(ticker, **kwargs):
        ends.append(kwargs["end"])
        return full.loc[full.index < pd.Timestamp(kwargs["end"])] if kwargs["end"] else full

    monkeypatch.setattr("macro_cta.data.yf.download", download)
    config = {"start_date": "2019-01-01", "universe": {"ETF": {"ticker": "ETF"}}}
    load_market_data(tmp_path / "cached", config)
    config["end_date"] = str(dates[6].date())
    cached = load_market_data(tmp_path / "cached", config)
    fresh = load_market_data(tmp_path / "fresh", config)
    assert ends == [None, str((dates[6] + pd.Timedelta(days=1)).date())]
    for actual, expected in zip(cached, fresh):
        pd.testing.assert_frame_equal(actual, expected, check_freq=False)
        assert actual.index[-1] == dates[6]


def test_snapshot_priority_and_refresh_flag(tmp_path, monkeypatch):
    dates = pd.bdate_range("2020-01-01", periods=10)
    snap_dir = tmp_path / "data/snapshot"
    snap_dir.mkdir(parents=True)
    snap_prices = pd.DataFrame({"ETF": np.arange(100., 110.)}, index=dates)
    snap_ret = np.log(snap_prices).diff().dropna()
    snap_prices = snap_prices.loc[snap_ret.index]
    snap_prices.to_csv(snap_dir / "prices.csv")
    snap_ret.to_csv(snap_dir / "returns.csv")

    calls = []
    live_prices = pd.DataFrame({"Close": np.arange(200., 210.)}, index=dates)

    def download(ticker, **kwargs):
        calls.append(ticker)
        return live_prices

    monkeypatch.setattr("macro_cta.data.yf.download", download)
    config = {"snapshot_dir": "data/snapshot", "cache_dir": "data/cache", "start_date": "2020-01-01", "universe": {"ETF": {"ticker": "ETF"}}}

    p_snap, r_snap = load_market_data(tmp_path, config, use_cache=True, refresh=False)
    assert calls == []
    pd.testing.assert_frame_equal(p_snap, snap_prices, check_freq=False)
    pd.testing.assert_frame_equal(r_snap, snap_ret, check_freq=False)

    p_live, _ = load_market_data(tmp_path, config, use_cache=True, refresh=True)
    assert calls == ["ETF"]
    assert p_live.iloc[0, 0] == pytest.approx(201.0)
    assert (tmp_path / "data/cache/prices.csv").exists()
