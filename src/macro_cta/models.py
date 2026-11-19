"""Multi-horizon EWMA crossover blended with time-series momentum (TSMOM) on log-prices."""
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


def single_asset_signal(
    returns: pd.Series,
    fast_d: int,
    slow_d: int,
    momentum_lookback: int = 126,
    vol_lookback: int = 63,
    vol_standardization: int = 252,
    momentum_weight: float = 0.20,
) -> pd.Series:
    """Calculate volatility-standardized EWMA crossover + TSMOM signal for a single asset on log-prices."""
    cum_prices = returns.cumsum()
    vol = returns.ewm(span=vol_lookback, adjust=False).std().replace(0, np.nan).ffill()

    # 1. EWMA Crossover on cumulative log-prices
    fast_ma = cum_prices.ewm(span=fast_d, adjust=False).mean()
    slow_ma = cum_prices.ewm(span=slow_d, adjust=False).mean()
    ewma_raw = fast_ma - slow_ma
    ewma_norm = ewma_raw / (vol * np.sqrt(fast_d) + 1e-9)

    # 2. Time-Series Momentum (TSMOM)
    tsmom_raw = cum_prices - cum_prices.shift(momentum_lookback)
    tsmom_norm = tsmom_raw / (vol * np.sqrt(momentum_lookback) + 1e-9)

    combined = (1.0 - momentum_weight) * ewma_norm + momentum_weight * tsmom_norm
    return combined.clip(-2.5, 2.5)


def compute_ewma_tsmom_signals(
    ret_df: pd.DataFrame,
    config: Dict,
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """Compute multi-horizon trend signals across all instruments on cumulative log-prices.

    Returns
    -------
    Tuple[Dict[str, pd.DataFrame], pd.DataFrame]
        (horizon_signals, composite_signals)
    """
    horizons = config["horizons"]
    vol_lookback = config.get("vol_lookback", 63)
    mom_lookback = config.get("momentum_lookback", 126)
    mom_weight = config.get("momentum_weight", 0.20)

    cum_prices = ret_df.cumsum()
    vol = ret_df.ewm(span=vol_lookback, adjust=False).std().replace(0, np.nan).ffill()

    horizon_signals: Dict[str, pd.DataFrame] = {}
    for h in horizons:
        fast_ma = cum_prices.ewm(span=h["fast_d"], adjust=False).mean()
        slow_ma = cum_prices.ewm(span=h["slow_d"], adjust=False).mean()
        ewma_raw = fast_ma - slow_ma
        horizon_signals[h["label"]] = (ewma_raw / (vol * np.sqrt(h["fast_d"]) + 1e-9)).clip(-2.5, 2.5)

    mom_raw = cum_prices - cum_prices.shift(mom_lookback)
    mom_sig = (mom_raw / (vol * np.sqrt(mom_lookback) + 1e-9)).clip(-2.5, 2.5)

    weights = [h["weight"] for h in horizons]
    stacked = sum(horizon_signals[h["label"]] * w for h, w in zip(horizons, weights))
    composite_df = (stacked + mom_weight * mom_sig).clip(-2.5, 2.5)

    return horizon_signals, composite_df
