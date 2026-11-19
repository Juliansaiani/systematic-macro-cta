"""Macro CTA systematic trend-following research package."""
from .data import load_market_data
from .models import compute_ewma_tsmom_signals
from .portfolio import compute_positions, simulate_strategy

__all__ = [
    "load_market_data",
    "compute_ewma_tsmom_signals",
    "compute_positions",
    "simulate_strategy",
]
