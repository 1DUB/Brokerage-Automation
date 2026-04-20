"""
Wouter Keller's Lethargic Asset Allocation (20% sleeve)

Rules per Keller (2020), "Growth-Trend Timing and 60-40 Variations:
Lethargic Asset Allocation (LAA)", SSRN 3498092.

Key design points:
  - 75% permanent holdings: VTV (25%), GLD (25%), IEF (25%)
  - 25% tactical sleeve: QQQ (risk-on) or CASH (risk-off)
  - Growth-Trend Timing signal requires BOTH conditions for risk-off:
      1. Unemployment rate > its 12-month SMA (economy weakening)
      2. S&P 500 < its 10-month SMA (market trend negative)
  - If either condition is false → risk-on (QQQ)
  - Historically risk-off < 15% of the time
"""

import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Dict

from lib.data import fetch_unemployment_rate

logger = logging.getLogger(__name__)

ALL_TICKERS = ["VTV", "GLD", "IEF", "QQQ", "SPY"]


@dataclass
class LethargicSignal:
    signal_date: str
    allocation: Dict[str, float] = field(default_factory=dict)
    gt_details: str = ""

    def summary(self) -> str:
        lines = [f"  Signal date: {self.signal_date}"]
        lines.append(f"  {self.gt_details}")
        lines.append("  Final allocation:")
        for t, w in sorted(self.allocation.items(), key=lambda x: -x[1]):
            if w > 0.001:
                lines.append(f"    {t:6s} {w:6.1%}")
        return "\n".join(lines)


def compute_lethargic_signals(monthly_prices: pd.DataFrame) -> LethargicSignal:
    """
    Compute Lethargic Asset Allocation signals.
    
    Growth-Trend Timing (per Keller 2020 / CXO Advisory / QuantReturns):
      Risk-off when BOTH:
        1. Most recent unemployment rate > 12-month SMA of unemployment rate
        2. S&P 500 month-end close < 10-month SMA of S&P 500 month-end closes
      Otherwise: risk-on
    """
    signal_date = monthly_prices.index[-1].strftime("%Y-%m-%d")
    
    # ── Condition 1: Unemployment rate vs its 12-month SMA ──
    ue_rate = fetch_unemployment_rate()
    
    if len(ue_rate) < 12:
        logger.warning("Insufficient unemployment data — defaulting to risk-on")
        ue_bearish = False
        ue_current = float("nan")
        ue_sma12 = float("nan")
    else:
        ue_sma12_series = ue_rate.rolling(12).mean()
        ue_current = ue_rate.iloc[-1]
        ue_sma12 = ue_sma12_series.iloc[-1]
        ue_bearish = ue_current > ue_sma12
    
    # ── Condition 2: S&P 500 vs its 10-month SMA ──
    spy_prices = monthly_prices["SPY"]
    
    if len(spy_prices) < 10:
        logger.warning("Insufficient SPY data — defaulting to risk-on")
        spy_bearish = False
        spy_current = float("nan")
        spy_sma10 = float("nan")
    else:
        spy_sma10_series = spy_prices.rolling(10).mean()
        spy_current = spy_prices.iloc[-1]
        spy_sma10 = spy_sma10_series.iloc[-1]
        spy_bearish = spy_current < spy_sma10
    
    # ── Growth-Trend Timing decision ──
    # Risk-off ONLY when BOTH conditions are bearish
    risk_off = ue_bearish and spy_bearish
    tactical = "CASH" if risk_off else "QQQ"
    
    # Build detail string for the report
    ue_status = f"UE {ue_current:.1f}% {'>' if ue_bearish else '<='} SMA12 {ue_sma12:.1f}% ({'BEARISH' if ue_bearish else 'OK'})"
    spy_status = f"SPY ${spy_current:.2f} {'<' if spy_bearish else '>='} SMA10 ${spy_sma10:.2f} ({'BEARISH' if spy_bearish else 'OK'})"
    gt_decision = "RISK-OFF → CASH" if risk_off else "RISK-ON → QQQ"
    gt_details = f"GT Timing: {ue_status} | {spy_status} → {gt_decision}"
    
    allocation = {
        "VTV": 0.25,
        "GLD": 0.25,
        "IEF": 0.25,
        tactical: 0.25,
    }
    
    return LethargicSignal(
        signal_date=signal_date,
        allocation=allocation,
        gt_details=gt_details,
    )
