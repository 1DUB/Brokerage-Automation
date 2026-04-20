"""
Stoken's Active Combined Asset – Monthly [Dynamic Bond] (40% sleeve)

Rules per Allocate Smartly and Stoken's "Survival of the Fittest for Investors" (2012).

Monthly variant: replaces original 252/126 trading-day lookbacks with
12/6 month-end lookbacks. Signals evaluated on the last trading day of
each month only.

Key design points:
  - Three independent modules (1/3 each): SPY/IEF, GLD/TLT, VNQ/IEF
  - Asymmetric lookbacks per Stoken:
      SPY & VNQ: upper channel = 6-month high (easy to enter),
                 lower channel = 12-month low (hard to exit)
      GLD:       upper channel = 12-month high (hard to enter),
                 lower channel = 6-month low (easy to exit)
  - Hold zone: if price is between upper and lower channels, hold
    current position (no change).
  - Dynamic Bond modification (per Allocate Smartly): when in defensive
    mode, apply the same channel logic to the defensive asset. If the
    defensive asset also fails (below its own lower channel), go to CASH.
"""

import pandas as pd
import json
import os
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Module definitions: (risk_asset, defensive_asset, upper_channel_months, lower_channel_months)
MODULES = [
    ("SPY", "IEF", 6, 12),   # Equities: easy entry (6mo high), hard exit (12mo low)
    ("GLD", "TLT", 12, 6),   # Gold: hard entry (12mo high), easy exit (6mo low)
    ("VNQ", "IEF", 6, 12),   # Real Estate: easy entry (6mo high), hard exit (12mo low)
]

ALL_TICKERS = ["SPY", "IEF", "GLD", "TLT", "VNQ"]

# File to persist module state between months
STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "stoken_state.json")


def _load_state() -> Dict[str, str]:
    """Load previous month's module positions. Returns dict of risk_asset -> 'risk' or 'defensive'."""
    try:
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        # Default: assume risk-on for all modules (per AS backtest convention:
        # "we assumed that SPY, GLD and VNQ were all long on day 1 of the test")
        return {"SPY": "risk", "GLD": "risk", "VNQ": "risk"}


def _save_state(state: Dict[str, str]):
    """Persist current module positions for next month."""
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(state, f)
    except Exception as e:
        logger.warning(f"Could not save Stoken state: {e}")


@dataclass
class StokenSignal:
    signal_date: str
    allocation: Dict[str, float] = field(default_factory=dict)
    module_details: Dict[str, str] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [f"  Signal date: {self.signal_date}"]
        for module_name, detail in self.module_details.items():
            lines.append(f"  {module_name}: {detail}")
        lines.append("  Final allocation:")
        for t, w in sorted(self.allocation.items(), key=lambda x: -x[1]):
            if w > 0.001:
                lines.append(f"    {t:6s} {w:6.1%}")
        return "\n".join(lines)


def compute_stoken_signals(monthly_prices: pd.DataFrame) -> StokenSignal:
    """
    Compute Stoken's ACA Monthly [Dynamic Bond] signals.
    
    For each module:
      1. Check if risk asset price > upper channel (highest close of previous N months)
         → If yes: go to risk asset
      2. Check if risk asset price < lower channel (lowest close of previous M months)
         → If yes: check defensive asset (Dynamic Bond), else go to CASH
      3. If between channels: hold current position (no change)
    """
    signal_date = monthly_prices.index[-1].strftime("%Y-%m-%d")
    allocation = {}
    module_details = {}
    
    # Load previous state
    prev_state = _load_state()
    new_state = {}
    
    for risk_asset, def_asset, upper_months, lower_months in MODULES:
        risk_prices = monthly_prices[risk_asset]
        current_price = risk_prices.iloc[-1]
        
        # Upper channel: highest close of the PREVIOUS n months (excluding current)
        # i.e., look at months [-upper_months-1:-1] to get the previous n month-end closes
        if len(risk_prices) > upper_months:
            upper_channel = risk_prices.iloc[-(upper_months + 1):-1].max()
        else:
            upper_channel = risk_prices.iloc[:-1].max()
        
        # Lower channel: lowest close of the PREVIOUS n months (excluding current)
        if len(risk_prices) > lower_months:
            lower_channel = risk_prices.iloc[-(lower_months + 1):-1].min()
        else:
            lower_channel = risk_prices.iloc[:-1].min()
        
        # Channel logic with hold zone
        if current_price > upper_channel:
            # Breakout above upper channel → go to risk asset
            new_state[risk_asset] = "risk"
            allocation[risk_asset] = allocation.get(risk_asset, 0) + 1/3
            module_details[f"{risk_asset}/{def_asset}"] = (
                f"RISK-ON: {risk_asset} at {current_price:.2f} > upper channel {upper_channel:.2f} "
                f"({upper_months}mo high)"
            )
        elif current_price < lower_channel:
            # Breakdown below lower channel → defensive mode
            # Apply Dynamic Bond check on the defensive asset
            def_prices = monthly_prices[def_asset]
            def_current = def_prices.iloc[-1]
            
            # For the defensive asset, use the same channel structure:
            # Check if defensive asset is above its own lower channel
            # Using 12-month low as the defensive asset's lower channel
            if len(def_prices) > 12:
                def_lower = def_prices.iloc[-13:-1].min()
            else:
                def_lower = def_prices.iloc[:-1].min()
            
            if def_current >= def_lower:
                # Defensive asset is holding up → allocate to defensive
                new_state[risk_asset] = "defensive"
                allocation[def_asset] = allocation.get(def_asset, 0) + 1/3
                module_details[f"{risk_asset}/{def_asset}"] = (
                    f"DEFENSIVE: {risk_asset} at {current_price:.2f} < lower channel {lower_channel:.2f} "
                    f"({lower_months}mo low) → {def_asset}"
                )
            else:
                # Defensive asset also failing → CASH (Dynamic Bond)
                new_state[risk_asset] = "defensive"
                allocation["CASH"] = allocation.get("CASH", 0) + 1/3
                module_details[f"{risk_asset}/{def_asset}"] = (
                    f"CASH (Dynamic Bond): {risk_asset} at {current_price:.2f} < lower {lower_channel:.2f}, "
                    f"{def_asset} at {def_current:.2f} < its lower {def_lower:.2f} → CASH"
                )
        else:
            # Between channels → hold previous position
            prev_position = prev_state.get(risk_asset, "risk")
            new_state[risk_asset] = prev_position
            
            if prev_position == "risk":
                allocation[risk_asset] = allocation.get(risk_asset, 0) + 1/3
                module_details[f"{risk_asset}/{def_asset}"] = (
                    f"HOLD RISK: {risk_asset} at {current_price:.2f} between channels "
                    f"[{lower_channel:.2f}, {upper_channel:.2f}] → holding {risk_asset}"
                )
            else:
                # Holding defensive — apply Dynamic Bond check
                def_prices = monthly_prices[def_asset]
                def_current = def_prices.iloc[-1]
                if len(def_prices) > 12:
                    def_lower = def_prices.iloc[-13:-1].min()
                else:
                    def_lower = def_prices.iloc[:-1].min()
                
                if def_current >= def_lower:
                    allocation[def_asset] = allocation.get(def_asset, 0) + 1/3
                    module_details[f"{risk_asset}/{def_asset}"] = (
                        f"HOLD DEFENSIVE: {risk_asset} between channels → holding {def_asset}"
                    )
                else:
                    allocation["CASH"] = allocation.get("CASH", 0) + 1/3
                    module_details[f"{risk_asset}/{def_asset}"] = (
                        f"HOLD DEFENSIVE → CASH (Dynamic Bond): {def_asset} below its lower channel"
                    )
    
    # Save state for next month
    _save_state(new_state)
    
    # Clean up dust
    allocation = {t: w for t, w in allocation.items() if w > 0.0001}
    
    return StokenSignal(
        signal_date=signal_date,
        allocation=allocation,
        module_details=module_details,
    )
