"""
Wouter Keller’s Lethargic Asset Allocation (20% sleeve)
"""

import pandas as pd
from dataclasses import dataclass, field
from typing import Dict

from lib.data import fetch_unemployment_rate
from lib.momentum import momentum_13612u

ALL_TICKERS = ["VTV", "GLD", "IEF", "QQQ"]


@dataclass
class LethargicSignal:
    signal_date: str
    allocation: Dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [f"  Signal date: {self.signal_date}"]
        lines.append("  Final allocation:")
        for t, w in sorted(self.allocation.items(), key=lambda x: -x[1]):
            if w > 0.001:
                lines.append(f"    {t:6s} {w:6.1%}")
        return "\n".join(lines)


def compute_lethargic_signals(monthly_prices: pd.DataFrame) -> LethargicSignal:
    signal_date = monthly_prices.index[-1].strftime("%Y-%m-%d")
    
    # Growth-Trend Timing rule
    ue_rate = fetch_unemployment_rate()
    ue_ma3 = ue_rate.rolling(3).mean()
    ue_rising = ue_ma3.iloc[-1] > ue_ma3.iloc[-2] if len(ue_ma3) >= 2 else False
    
    spy_trend = monthly_prices["SPY"].iloc[-1] > monthly_prices["SPY"].iloc[-13]
    
    tactical = "CASH" if (ue_rising and not spy_trend) else "QQQ"
    
    allocation = {
        "VTV": 0.25,
        "GLD": 0.25,
        "IEF": 0.25,
        tactical: 0.25,
    }
    
    return LethargicSignal(signal_date=signal_date, allocation=allocation)
