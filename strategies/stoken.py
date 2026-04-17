"""
Stoken’s Active Combined Asset – Monthly [Dynamic Bond] (40% sleeve)
"""

import pandas as pd
from dataclasses import dataclass, field
from typing import Dict

from lib.momentum import momentum_13612u

PAIRS = [
    ("SPY", "IEF"),
    ("GLD", "TLT"),
    ("VNQ", "IEF"),
]

ALL_TICKERS = ["SPY", "IEF", "GLD", "TLT", "VNQ"]


@dataclass
class StokenSignal:
    signal_date: str
    allocation: Dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [f"  Signal date: {self.signal_date}"]
        lines.append("  Final allocation:")
        for t, w in sorted(self.allocation.items(), key=lambda x: -x[1]):
            if w > 0.001:
                lines.append(f"    {t:6s} {w:6.1%}")
        return "\n".join(lines)


def _price_channel_breakout(prices: pd.Series, months: int = 12) -> bool:
    if len(prices) < months:
        return False
    current = prices.iloc[-1]
    upper = prices.iloc[-months:].max()
    return current > upper


def compute_stoken_signals(monthly_prices: pd.DataFrame) -> StokenSignal:
    signal_date = monthly_prices.index[-1].strftime("%Y-%m-%d")
    allocation = {}

    for risk, defen in PAIRS:
        risk_price = monthly_prices[risk]
        def_price = monthly_prices[defen]

        if _price_channel_breakout(risk_price, 12) or _price_channel_breakout(risk_price, 6):
            allocation[risk] = allocation.get(risk, 0) + 1/3
        else:
            def_mom = momentum_13612u(def_price)
            if def_mom > 0:
                allocation[defen] = allocation.get(defen, 0) + 1/3
            else:
                allocation["CASH"] = allocation.get("CASH", 0) + 1/3

    allocation = {t: w for t, w in allocation.items() if w > 0.0001}
    return StokenSignal(signal_date=signal_date, allocation=allocation)
