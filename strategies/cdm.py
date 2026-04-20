"""
Composite Dual Momentum (25% sleeve)

Rules per Allocate Smartly and Antonacci's "Risk Premia Harvesting Through
Dual Momentum" (SSRN 2042750).

Key design points:
  - Four independent modules, 25% each: Equities, Credit, Real Estate, Stress
  - Momentum measure: simple 12-month total return (NOT 13612U)
  - Dual momentum: relative (pick best of pair) + absolute (beat cash or go to CASH)
  - Cash proxy: BIL (3-month T-bills) for price data; mapped to CASH in output
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict

from lib.momentum import total_return

# Four modules as described on Allocate Smartly
MODULES = [
    ("Equities",    ["SPY", "IEFA"]),
    ("Credit",      ["LQD", "HYG"]),
    ("Real Estate", ["VNQ", "REM"]),
    ("Stress",      ["GLD", "TLT"]),
]

ALL_TICKERS = ["SPY", "IEFA", "LQD", "HYG", "VNQ", "REM", "GLD", "TLT", "BIL"]


@dataclass
class CDM_Signal:
    signal_date: str
    allocation: Dict[str, float] = field(default_factory=dict)
    module_details: Dict[str, str] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [f"  Signal date: {self.signal_date}"]
        for module_name, detail in self.module_details.items():
            lines.append(f"  {module_name}: {detail}")
        lines.append("  Final allocation (25% per module):")
        for t, w in sorted(self.allocation.items(), key=lambda x: -x[1]):
            if w > 0.001:
                lines.append(f"    {t:6s} {w:6.1%}")
        return "\n".join(lines)


def _dual_momentum_12m(prices: pd.DataFrame, pair: list, module_name: str) -> tuple:
    """
    For a pair, choose winner by 12-month relative momentum,
    then apply absolute momentum vs BIL (cash proxy).
    
    Returns: (chosen_ticker, detail_string)
    """
    a, b = pair
    
    # Simple 12-month total return for relative momentum
    ret_a = total_return(prices[a], 12)
    ret_b = total_return(prices[b], 12)
    
    winner = a if ret_a > ret_b else b
    winner_ret = ret_a if ret_a > ret_b else ret_b
    loser = b if winner == a else a
    loser_ret = ret_b if winner == a else ret_a
    
    # Absolute momentum check: does winner beat cash (BIL) over 12 months?
    ret_cash = total_return(prices["BIL"], 12)
    
    if np.isnan(winner_ret) or np.isnan(ret_cash):
        detail = f"Insufficient data → CASH"
        return "CASH", detail
    
    if winner_ret > ret_cash:
        detail = (
            f"{winner} ({winner_ret:+.1%}) > {loser} ({loser_ret:+.1%}), "
            f"beats BIL ({ret_cash:+.1%}) → {winner}"
        )
        return winner, detail
    else:
        detail = (
            f"{winner} ({winner_ret:+.1%}) > {loser} ({loser_ret:+.1%}), "
            f"but < BIL ({ret_cash:+.1%}) → CASH"
        )
        return "CASH", detail


def compute_cdm_signals(monthly_prices: pd.DataFrame) -> CDM_Signal:
    """
    Compute Composite Dual Momentum signals.
    
    Per Allocate Smartly:
      "At the close on the last trading day of the month, calculate the
       12-month return of each of the eight asset classes shown above,
       plus 3-month US Treasuries (BIL). Divide the portfolio into four
       equally-sized modules (25% allocated to each). For each of the
       modules, determine the asset with the highest 12-month return
       (relative momentum). If that asset's return exceeds the 12-month
       return of BIL (absolute momentum), go long that asset at the close,
       otherwise move to cash."
    """
    signal_date = monthly_prices.index[-1].strftime("%Y-%m-%d")
    allocation = {}
    module_details = {}

    for module_name, pair in MODULES:
        choice, detail = _dual_momentum_12m(monthly_prices, pair, module_name)
        allocation[choice] = allocation.get(choice, 0) + 0.25
        module_details[module_name] = detail

    # Remove zero-weight dust
    allocation = {t: w for t, w in allocation.items() if w > 0.0001}

    return CDM_Signal(
        signal_date=signal_date,
        allocation=allocation,
        module_details=module_details,
    )
