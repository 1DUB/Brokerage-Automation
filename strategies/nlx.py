"""
NLX Finance’s Hybrid Asset Allocation 60/40 (40% sleeve)
"""

import pandas as pd
from dataclasses import dataclass, field
from typing import Dict

from lib.momentum import momentum_13612u

OFFENSIVE_UNIVERSE = ["SPY", "IWM", "IEFA", "IEMG", "VNQ", "PDBC", "IEF", "TLT"]
DEFENSIVE_UNIVERSE = ["BIL", "IEF"]
CANARY_UNIVERSE = ["TIP"]

ALL_TICKERS = list(set(OFFENSIVE_UNIVERSE + DEFENSIVE_UNIVERSE + CANARY_UNIVERSE))


@dataclass
class NLXSignal:
    signal_date: str
    canary_momentum: float
    is_risk_on: bool
    allocation: Dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [f"  Signal date: {self.signal_date}"]
        lines.append(f"  Canary TIP momentum: {self.canary_momentum:+.2%} ({'RISK-ON' if self.is_risk_on else 'RISK-OFF'})")
        if self.is_risk_on:
            lines.append("  Risk-on → 60% SPY / 40% IEF")
        else:
            lines.append("  Risk-off → defensive asset")
        lines.append("  Final allocation:")
        for t, w in sorted(self.allocation.items(), key=lambda x: -x[1]):
            if w > 0.001:
                lines.append(f"    {t:6s} {w:6.1%}")
        return "\n".join(lines)


def compute_nlx_signals(prices: pd.DataFrame) -> NLXSignal:
    signal_date = prices.index[-1].strftime("%Y-%m-%d")
    tip_mom = momentum_13612u(prices["TIP"])
    is_risk_on = tip_mom > 0

    def_moms = {t: momentum_13612u(prices[t]) for t in DEFENSIVE_UNIVERSE}
    best_def = max(def_moms, key=def_moms.get)

    if is_risk_on:
        allocation = {"SPY": 0.60, "IEF": 0.40}
    else:
        allocation = {best_def: 1.0}

    return NLXSignal(
        signal_date=signal_date,
        canary_momentum=tip_mom,
        is_risk_on=is_risk_on,
        allocation=allocation,
    )
