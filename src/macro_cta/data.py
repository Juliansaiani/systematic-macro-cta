"""Market data ingestion, alignment and local caching for the 22-instrument CTA universe."""
from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
import yfinance as yf


def _validate_panels(prices, returns, universe):
    if set(prices.columns) != set(universe) or not prices.columns.equals(returns.columns):
        raise ValueError("Cached/downloaded instrument universe differs from configuration.")
    if prices.empty or not prices.index.equals(returns.index):
        raise ValueError("Prices and log returns must share a nonempty calendar.")
    if not prices.index.is_unique or not prices.index.is_monotonic_increasing:
        raise ValueError("Market dates must be unique and increasing.")
    if not np.isfinite(prices.to_numpy()).all() or (prices <= 0).any().any() or not np.isfinite(returns.to_numpy()).all():
        raise ValueError("Prices must be finite and positive; log returns must be finite.")
    # The first return may refer to a price preceding this aligned price panel.
    if not np.allclose(np.log(prices).diff().iloc[1:], returns.iloc[1:], atol=1e-10, rtol=0):
        raise ValueError("Return cache is inconsistent with adjusted prices.")


def _load_panel_pair(
    directory: Path,
    start_date: str,
    end_date: Optional[str],
    universe: Dict,
) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
    price_path = directory / "prices.csv"
    ret_path = directory / "returns.csv"
    if not (price_path.exists() and ret_path.exists()):
        return None
    price_df = pd.read_csv(price_path, index_col=0, parse_dates=True).loc[start_date:end_date]
    ret_df = pd.read_csv(ret_path, index_col=0, parse_dates=True).loc[start_date:end_date]
    _validate_panels(price_df, ret_df, universe)
    return price_df, ret_df


def load_market_data(
    root: Path,
    config: Dict,
    use_cache: bool = True,
    refresh: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load or fetch aligned price and log-return panels.

    Parameters
    ----------
    root : Path
        Project root directory.
    config : Dict
        Data section of config.json containing snapshot/cache directories and universe.
    use_cache : bool
        Whether to load from/save to local disk cache.
    refresh : bool
        Whether to bypass pinned snapshots and caches and download live data via yfinance.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (price_df, ret_df) aligned on matching business dates.
    """
    snapshot_dir = root / config.get("snapshot_dir", "data/snapshot")
    cache_dir = root / config.get("cache_dir", "data/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    price_cache = cache_dir / "prices.csv"
    ret_cache = cache_dir / "returns.csv"
    # Configuration dates are inclusive; Yahoo's download end is exclusive.
    start_date = config.get("start_date", "2006-01-01")
    end_date = config.get("end_date")
    download_end = str((pd.Timestamp(end_date) + pd.Timedelta(days=1)).date()) if end_date else None

    if not refresh:
        snapshot = _load_panel_pair(snapshot_dir, start_date, end_date, config["universe"])
        if snapshot is not None:
            return snapshot
        if use_cache:
            cached = _load_panel_pair(cache_dir, start_date, end_date, config["universe"])
            if cached is not None:
                return cached

    universe = config["universe"]

    prices: Dict[str, pd.Series] = {}
    for name, meta in universe.items():
        ticker = meta["ticker"]
        raw = yf.download(ticker, start=start_date, end=download_end, auto_adjust=True, progress=False)
        if raw.empty:
            raise RuntimeError(f"Failed to retrieve configured instrument {name} ({ticker}).")

        if isinstance(raw.columns, pd.MultiIndex):
            series = raw[("Close", ticker)] if ("Close", ticker) in raw.columns else raw.iloc[:, 0]
        else:
            series = raw["Close"] if "Close" in raw.columns else raw["Adj Close"] if "Adj Close" in raw.columns else raw.iloc[:, 0]

        prices[name] = series.dropna()

    if not prices:
        raise RuntimeError("Failed to retrieve market data from Yahoo Finance.")

    price_df = pd.DataFrame(prices).ffill()
    min_coverage = int(len(price_df.columns) * 0.60)
    price_df = price_df.dropna(thresh=min_coverage)

    ret_df = np.log(price_df).diff().dropna()
    price_df = price_df.loc[ret_df.index]
    _validate_panels(price_df, ret_df, universe)

    if use_cache:
        price_df.to_csv(price_cache)
        ret_df.to_csv(ret_cache)

    return price_df, ret_df
