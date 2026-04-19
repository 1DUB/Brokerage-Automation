"""
Composite Dual Momentum (25% sleeve)

Exact rules from Allocate Smartly + Antonacci’s Dual Momentum paper.
Four independent modules, each selects the best asset or CASH.
"""

import pandas as pd
from dataclasses import dataclass, field
from typing import Dict

from lib.momentum import momentum_13612u

# Four modules as described on Allocate Smartly
MODULES = [
    ("Equities", ["SPY", "IEFA"]),
    ("Credit",   ["LQD", "HYG"]),
    ("RealEstate", ["VNQ", "REM"]),
    ("Stress",   ["GLD", "TLT"]),
]

ALL_TICKERS = ["SPY", "IEFA", "LQD", "HYG", "VNQ", "REM", "GLD", "TLT", "CASH"]


@dataclass
class CDM_Signal:
    signal_date: str
    allocation: Dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [f"  Signal date: {self.signal_date}"]
        lines.append("  Final allocation (25% per module):")
        for t, w in sorted(self.allocation.items(), key=lambda x: -x[1]):
            if w > 0.001:
                lines.append(f"    {t:6s} {w:6.1%}")
        return "\n".join(lines)


def _dual_momentum(prices: pd.DataFrame, pair: list) -> str:
    """For a pair, choose winner by relative momentum, then apply absolute vs BIL."""
    a, b = pair
    mom_a = momentum_13612u(prices[a])
    mom_b = momentum_13612u(prices[b])
    winner = a if mom_a > mom_b else b

    # Absolute momentum vs cash (BIL)
    mom_cash = momentum_13612u(prices["BIL"])
    return winner if mom_winner := momentum_13612u(prices[winner]) > mom_cash else "CASH"


def compute_cdm_signals(monthly_prices: pd.DataFrame) -> CDM_Signal:
    signal_date = monthly_prices.index[-1].strftime("%Y-%m-%d")
    allocation = {}

    for module_name, pair in MODULES:
        choice = _dual_momentum(monthly_prices, pair)
        allocation[choice] = allocation.get(choice, 0) + 0.25   # 25% per module

    # Remove any zero-weight dust
    allocation = {t: w for t, w in allocation.items() if w > 0.0001}

    return CDM_Signal(signal_date=signal_date, allocation=allocation)
