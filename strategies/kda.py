"""
Kipnis Defensive Adaptive Asset Allocation (KDA) — AS-ALIGNED

Per Kipnis (2019), "Right Now It's KDA…Asset Allocation".
QuantStratTradeR blog, January 24, 2019.

This version is aligned with Allocate Smartly's live model:
- CANARY_UNIVERSE changed to ["EEM", "AGG"] (AS standard; original paper/blog used VWO/BND).
- All other logic (13612W momentum, blended correlation, min-variance optimization,
  defensive routing to IEF or CASH) is unchanged and already matches AS.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from lib.momentum import momentum_13612w
from lib.optimization import min_variance_weights, build_covariance

# ── Universe definitions (AS-aligned) ──────────────────────────────────────────────

RISKY_UNIVERSE = ["SPY", "VGK", "EWJ", "IEMG", "VNQ", "RWX", "IEF", "TLT", "DBC", "GLD"]
CANARY_UNIVERSE = ["EEM", "AGG"]          # ← AS standardized canaries
SAFETY_ASSET = "IEF"  # Used when canaries trigger defensive allocation

ALL_TICKERS = list(set(RISKY_UNIVERSE + CANARY_UNIVERSE))

TOP_N = 5  # Select top 5 risky assets


@dataclass
class KDASignal:
    """Output of KDA signal computation for one month."""
    
    signal_date: str
    
    # Canary status
    canary_momentum: Dict[str, float] = field(default_factory=dict)
    n_positive_canaries: int = 0
    pct_aggressive: float = 0.0  # 0.0, 0.5, or 1.0
    
    # Risky asset details
    risky_momentum: Dict[str, float] = field(default_factory=dict)
    selected_assets: List[str] = field(default_factory=list)
    risky_weights: Dict[str, float] = field(default_factory=dict)  # raw min-var weights
    
    # IEF safety status
    ief_momentum: float = 0.0
    safety_to_ief: bool = False  # True if defensive portion goes to IEF
    
    # Final allocation for this sleeve (ticker -> weight, sums to 1.0)
    allocation: Dict[str, float] = field(default_factory=dict)
    
    def summary(self) -> str:
        """Human-readable summary for the email report."""
        lines = []
        lines.append(f"  Signal date: {self.signal_date}")
        
        # Canary status
        canary_str = ", ".join(
            f"{t}: {m:+.2%}" for t, m in self.canary_momentum.items()
        )
        lines.append(f"  Canaries ({canary_str})")
        lines.append(
            f"  → {self.n_positive_canaries}/2 positive → "
            f"{self.pct_aggressive*100:.0f}% risk-on, "
            f"{(1-self.pct_aggressive)*100:.0f}% defensive"
        )
        
        # Risky selections
        if self.selected_assets:
            lines.append(f"  Top {len(self.selected_assets)} risky assets selected:")
            for ticker in self.selected_assets:
                mom = self.risky_momentum.get(ticker, 0)
                wt = self.risky_weights.get(ticker, 0)
                lines.append(
                    f"    {ticker:6s} momentum: {mom:+.4f}  min-var weight: {wt:6.1%}"
                )
        else:
            lines.append("  No risky assets selected (none had positive momentum + top-5 rank)")
        
        # Defensive routing
        if self.pct_aggressive < 1.0:
            destination = "IEF" if self.safety_to_ief else "CASH"
            lines.append(
                f"  Defensive portion ({(1-self.pct_aggressive)*100:.0f}%) → {destination} "
                f"(IEF momentum: {self.ief_momentum:+.4f})"
            )
        
        # Final allocation
        lines.append(f"  Final allocation:")
        for ticker, weight in sorted(
            self.allocation.items(), key=lambda x: -x[1]
        ):
            if weight > 0.001:
                lines.append(f"    {ticker:6s} {weight:6.1%}")
        
        return "\n".join(lines)


# (All helper functions compute_blended_correlation, compute_volatilities, 
#  and compute_kda_signals remain 100% unchanged from your original file.
#  Only the CANARY_UNIVERSE line above was updated.)

def compute_blended_correlation(
    daily_prices: pd.DataFrame,
    tickers: List[str],
    end_date: pd.Timestamp,
) -> np.ndarray:
    """
    Compute KDA's blended correlation matrix using daily returns.
    
    Per Kipnis: blended weights of 12*1mo + 4*3mo + 2*6mo + 1*12mo,
    divided by 19 (sum of weights).
    
    Args:
        daily_prices: DataFrame of daily prices.
        tickers: List of tickers to include.
        end_date: As-of date for the correlation calculation.
    
    Returns:
        NxN correlation matrix (numpy array) for the given tickers.
    """
    # Subset daily prices up to end_date
    prices = daily_prices.loc[:end_date, tickers]
    
    # Compute daily returns
    returns = prices.pct_change().dropna()
    
    # Trading-day windows (approximate)
    window_1mo = 21
    window_3mo = 63
    window_6mo = 126
    window_12mo = 252
    
    if len(returns) < window_12mo:
        raise ValueError(
            f"Need at least {window_12mo} days of returns, got {len(returns)}"
        )
    
    # Compute correlation over each window
    corr_1mo = returns.iloc[-window_1mo:].corr().values
    corr_3mo = returns.iloc[-window_3mo:].corr().values
    corr_6mo = returns.iloc[-window_6mo:].corr().values
    corr_12mo = returns.iloc[-window_12mo:].corr().values
    
    # Weighted average per Kipnis formula
    blended = (12 * corr_1mo + 4 * corr_3mo + 2 * corr_6mo + 1 * corr_12mo) / 19.0
    
    return blended


def compute_volatilities(
    daily_prices: pd.DataFrame,
    tickers: List[str],
    end_date: pd.Timestamp,
    window_days: int = 21,
) -> np.ndarray:
    """
    Compute per-asset volatilities using last-month daily returns.
    
    Per Kipnis: uses last-month (≈21 trading days) of daily returns.
    Returns standard deviation (not annualized — KDA's covariance 
    construction doesn't need annualization).
    
    Args:
        daily_prices: DataFrame of daily prices.
        tickers: List of tickers.
        end_date: As-of date.
        window_days: Number of trading days for the volatility window.
    
    Returns:
        1D numpy array of volatilities.
    """
    prices = daily_prices.loc[:end_date, tickers]
    returns = prices.pct_change().dropna()
    
    if len(returns) < window_days:
        raise ValueError(f"Need at least {window_days} days, got {len(returns)}")
    
    return returns.iloc[-window_days:].std().values


def compute_kda_signals(
    monthly_prices: pd.DataFrame,
    daily_prices: pd.DataFrame,
) -> KDASignal:
    """
    Compute KDA signals for the most recent month.
    
    (Canary universe now matches AS exactly; everything else identical to paper.)
    """
    signal_date_ts = monthly_prices.index[-1]
    signal_date = signal_date_ts.strftime("%Y-%m-%d")
    
    # ── Step 1: Compute 13612W momentum for risky and canary assets ──
    risky_moms = {}
    for ticker in RISKY_UNIVERSE:
        risky_moms[ticker] = momentum_13612w(monthly_prices[ticker])
    
    canary_moms = {}
    for ticker in CANARY_UNIVERSE:
        canary_moms[ticker] = momentum_13612w(monthly_prices[ticker])
    
    ief_mom = risky_moms[SAFETY_ASSET]
    
    # ── Step 2: Rank risky assets and select ─────────────────────────
    # Rank by momentum (descending). Top 5 by rank, must also have positive momentum.
    ranked = sorted(risky_moms.items(), key=lambda x: x[1], reverse=True)
    top_5_tickers = [t for t, _ in ranked[:TOP_N]]
    
    # Selected assets: must be in top 5 AND have positive momentum
    selected = [t for t in top_5_tickers if risky_moms[t] > 0]
    
    # ── Step 3: Compute min-variance weights for selected assets ─────
    raw_weights = {}
    if len(selected) == 0:
        # No selections: 100% defensive
        pass
    elif len(selected) == 1:
        raw_weights[selected[0]] = 1.0
    else:
        # Compute correlation and volatility for selected assets
        corr = compute_blended_correlation(daily_prices, selected, signal_date_ts)
        vols = compute_volatilities(daily_prices, selected, signal_date_ts)
        
        # Build covariance matrix
        cov = build_covariance(corr, vols)
        
        # Solve min-variance optimization
        weights_array = min_variance_weights(cov)
        
        for i, ticker in enumerate(selected):
            raw_weights[ticker] = float(weights_array[i])
    
    # ── Step 4: Apply canary protection ──────────────────────────────
    n_positive = sum(1 for m in canary_moms.values() if m > 0)
    pct_aggressive = n_positive / 2.0  # 0.0, 0.5, or 1.0
    pct_defensive = 1.0 - pct_aggressive
    
    # Scale risky weights by pct_aggressive
    allocation = {ticker: w * pct_aggressive for ticker, w in raw_weights.items()}
    
    # ── Step 5: Allocate defensive portion ───────────────────────────
    safety_to_ief = ief_mom > 0
    
    if pct_defensive > 0:
        if safety_to_ief:
            # Add defensive portion to IEF (which may already have some risky weight)
            allocation[SAFETY_ASSET] = allocation.get(SAFETY_ASSET, 0) + pct_defensive
        else:
            # IEF momentum is negative → go to cash
            allocation["CASH"] = allocation.get("CASH", 0) + pct_defensive
    
    # Remove zero entries for cleanliness
    allocation = {t: w for t, w in allocation.items() if w > 0.0001}
    
    return KDASignal(
        signal_date=signal_date,
        canary_momentum=canary_moms,
        n_positive_canaries=n_positive,
        pct_aggressive=pct_aggressive,
        risky_momentum=risky_moms,
        selected_assets=selected,
        risky_weights=raw_weights,
        ief_momentum=ief_mom,
        safety_to_ief=safety_to_ief,
        allocation=allocation,
    )
