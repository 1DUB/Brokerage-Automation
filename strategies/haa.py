"""
Hybrid Asset Allocation – Balanced (HAA) — AS-ALIGNED

Per Keller & Keuning (2023), "Dual and Canary Momentum with Rising 
Yields/Inflation: Hybrid Asset Allocation (HAA)", SSRN 4346906.

This version is aligned with Allocate Smartly's live model:
- Uses TICKER_ALIASES in lib/data.py to map EFA→IEFA and EEM→IEMG.
- All other rules (canary, offensive/defensive universes, 13612U momentum)
  are unchanged from the paper and your original implementation.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from lib.momentum import momentum_13612u

# ── Universe definitions (paper exact — aliases handle AS liquidity) ──

OFFENSIVE_UNIVERSE = ["SPY", "IWM", "IEFA", "IEMG", "VNQ", "PDBC", "IEF", "TLT"]
DEFENSIVE_UNIVERSE = ["BIL", "IEF"]  # pick best by momentum
CANARY_UNIVERSE = ["TIP"]

ALL_TICKERS = list(set(OFFENSIVE_UNIVERSE + DEFENSIVE_UNIVERSE + CANARY_UNIVERSE))

TOP_N = 4  # Select top 4 offensive assets


@dataclass
class HAASignal:
    """Output of the HAA signal computation for one month."""
    
    signal_date: str
    canary_momentum: float  # TIP's 13612U momentum
    is_risk_on: bool  # True if TIP momentum > 0
    
    # Offensive asset details (only populated if risk_on)
    offensive_rankings: List[Dict] = field(default_factory=list)
    
    # Final allocation for this sleeve (ticker -> weight, sums to 1.0)
    allocation: Dict[str, float] = field(default_factory=dict)
    
    # Which defensive asset was chosen and why
    defensive_choice: str = ""
    defensive_momentum: Dict[str, float] = field(default_factory=dict)
    
    def summary(self) -> str:
        """Human-readable summary for the email report."""
        lines = []
        lines.append(f"  Signal date: {self.signal_date}")
        lines.append(
            f"  Canary TIP momentum: {self.canary_momentum:+.2%} "
            f"({'RISK-ON' if self.is_risk_on else 'RISK-OFF'})"
        )
        
        if self.is_risk_on:
            lines.append(f"  Top {TOP_N} selected from offensive universe:")
            for asset in self.offensive_rankings[:TOP_N]:
                status = "HOLD" if asset["positive"] else f"-> {self.defensive_choice}"
                lines.append(
                    f"    {asset['ticker']:6s} momentum: {asset['momentum']:+.2%}  {status}"
                )
        else:
            lines.append(
                f"  100% defensive -> {self.defensive_choice} "
                f"(IEF: {self.defensive_momentum.get('IEF', 0):+.2%}, "
                f"BIL: {self.defensive_momentum.get('BIL', 0):+.2%})"
            )
        
        lines.append(f"  Final allocation:")
        for ticker, weight in sorted(
            self.allocation.items(), key=lambda x: -x[1]
        ):
            if weight > 0.001:
                lines.append(f"    {ticker:6s} {weight:6.1%}")
        
        return "\n".join(lines)


def compute_haa_signals(prices: pd.DataFrame) -> HAASignal:
    """
    Compute HAA-Balanced signals from monthly price data.
    
    (Identical logic to the paper; AS liquidity is handled transparently
     via TICKER_ALIASES in lib/data.py.)
    """
    signal_date = prices.index[-1].strftime("%Y-%m-%d")
    
    # ── Step 1: Compute canary momentum (TIP) ────────────────────────
    tip_mom = momentum_13612u(prices["TIP"])
    is_risk_on = tip_mom > 0
    
    # ── Compute defensive momentum ───────────────────────────────────
    def_moms = {}
    for ticker in DEFENSIVE_UNIVERSE:
        def_moms[ticker] = momentum_13612u(prices[ticker])
    
    # Best defensive asset = highest momentum between BIL and IEF
    best_defensive = max(def_moms, key=def_moms.get)
    
    # ── Step 2a: Risk-on path ────────────────────────────────────────
    if is_risk_on:
        # Compute momentum for all offensive assets
        off_moms = {}
        for ticker in OFFENSIVE_UNIVERSE:
            off_moms[ticker] = momentum_13612u(prices[ticker])
        
        # Rank by momentum (descending)
        ranked = sorted(off_moms.items(), key=lambda x: x[1], reverse=True)
        
        # Build rankings detail
        rankings = []
        for i, (ticker, mom) in enumerate(ranked):
            rankings.append({
                "ticker": ticker,
                "momentum": mom,
                "selected": i < TOP_N,
                "positive": mom > 0,
            })
        
        # Allocate
        allocation = {}
        weight_per_slot = 1.0 / TOP_N  # 25% each
        
        for asset in rankings[:TOP_N]:
            if asset["positive"]:
                ticker = asset["ticker"]
                allocation[ticker] = allocation.get(ticker, 0) + weight_per_slot
            else:
                allocation[best_defensive] = (
                    allocation.get(best_defensive, 0) + weight_per_slot
                )
        
        return HAASignal(
            signal_date=signal_date,
            canary_momentum=tip_mom,
            is_risk_on=True,
            offensive_rankings=rankings,
            allocation=allocation,
            defensive_choice=best_defensive,
            defensive_momentum=def_moms,
        )
    
    # ── Step 2b: Risk-off path ───────────────────────────────────────
    else:
        allocation = {best_defensive: 1.0}
        
        return HAASignal(
            signal_date=signal_date,
            canary_momentum=tip_mom,
            is_risk_on=False,
            allocation=allocation,
            defensive_choice=best_defensive,
            defensive_momentum=def_moms,
        )
