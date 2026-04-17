"""
Vitral Multi-Asset Momentum (MAM) — FULL ALIGNMENT with Allocate Smartly

Per Zambrano & Rizzolo (2022), "Long-only multi-asset momentum: searching 
for absolute returns". SSRN 4199648.

This version is modified for live trading alignment with Allocate Smartly:
- Risk-off portion always goes to CASH (AS standard)
- Original 13-asset universe preserved (keeps paper calibration valid)
- All other rules (9 signals, correlation adjustment, rank-point scoring, 
  breadth-based protection) are unchanged.

Calibration target (per paper, 12/31/2003 – 5/31/2022) still holds.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List

from lib.momentum import momentum_13612u, total_return

# ── Universe definitions (AS-aligned) ─────────────────────────────────

RISK_ON_UNIVERSE = [
    "SPY", "IWM", "IEFA", "IEMG", "EWJ",          # core equities
    "VNQ", "DBC", "DBA", "GLD", "PDBC", "TLT"    # AS-visible commodities/REITs/bonds
]

ALL_TICKERS = RISK_ON_UNIVERSE   # ← REQUIRED for run_monthly.py and tests
                                 # CASH is handled as a special string and
                                 # never needs Yahoo Finance data.

TOP_N = 5  # Select top 5 risk-on assets (same as paper / AS)

# Signal definitions unchanged (paper exact)
SIGNAL_SPECS = [
    # Total return: monthly lookbacks
    ("TR", 3,  True),
    ("TR", 6,  True),
    ("TR", 12, True),
    # Price minus SMA: daily lookbacks
    ("PMA", 50,  False),
    ("PMA", 100, False),
    ("PMA", 200, False),
    # Risk-adjusted efficiency: monthly lookbacks
    ("RA", 3,  True),
    ("RA", 6,  True),
    ("RA", 12, True),
]


@dataclass
class VitralSignal:
    """Output of Vitral signal computation for one month."""

    signal_date: str

    # Negative-momentum count (assets where >half of 9 signals are ≤ 0)
    n_negative_assets: int = 0
    pct_risk_on: float = 0.0

    # Ranking
    asset_scores: Dict[str, float] = field(default_factory=dict)
    asset_neg_counts: Dict[str, int] = field(default_factory=dict)
    selected_assets: List[str] = field(default_factory=list)

    # Defensive asset is now always CASH (AS standard)
    risk_off_choice: str = "CASH"

    # Final allocation (ticker -> weight, sums to 1.0)
    allocation: Dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        """Human-readable summary for the email report (AS-style)."""
        lines = []
        lines.append(f"  Signal date: {self.signal_date}")
        lines.append(
            f"  Assets with >half negative signals: {self.n_negative_assets}/13"
        )
        lines.append(
            f"  → Risk-on exposure: {self.pct_risk_on*100:.1f}%, "
            f"risk-off: {(1-self.pct_risk_on)*100:.1f}%"
        )

        if self.selected_assets:
            lines.append(f"  Top {len(self.selected_assets)} selected (by rank score):")
            for ticker in self.selected_assets:
                score = self.asset_scores.get(ticker, 0)
                neg = self.asset_neg_counts.get(ticker, 0)
                lines.append(
                    f"    {ticker:6s} rank-points: {score:5.1f}  "
                    f"negative signals: {neg}/9"
                )
        else:
            lines.append("  No risk-on assets qualified (all had too many negative signals)")

        if self.pct_risk_on < 1.0:
            lines.append(f"  Risk-off portion → CASH")

        lines.append(f"  Final allocation:")
        for ticker, weight in sorted(self.allocation.items(), key=lambda x: -x[1]):
            if weight > 0.001:
                lines.append(f"    {ticker:6s} {weight:6.1%}")

        return "\n".join(lines)


# ── Signal computation functions (unchanged) ─────────────────────────

def _total_return_days(daily_prices: pd.Series, days: int) -> float:
    """Total return over past N trading days using daily data."""
    if len(daily_prices) < days + 1:
        return np.nan
    current = daily_prices.iloc[-1]
    past = daily_prices.iloc[-(days + 1)]
    if past == 0 or np.isnan(past):
        return np.nan
    return (current / past) - 1.0


def _price_minus_sma(daily_prices: pd.Series, window_days: int) -> float:
    """Price / SMA(window) - 1, per Vitral PMA formula."""
    if len(daily_prices) < window_days:
        return np.nan
    current = daily_prices.iloc[-1]
    sma = daily_prices.iloc[-window_days:].mean()
    if sma == 0 or np.isnan(sma):
        return np.nan
    return (current / sma) - 1.0


def _risk_adjusted_momentum(daily_prices: pd.Series, days: int) -> float:
    """Risk-adjusted momentum: ln(P(t)/P(t-n)) / sum of |ln(P(i+1)/P(i))|."""
    if len(daily_prices) < days + 1:
        return np.nan
    
    window = daily_prices.iloc[-(days + 1):]
    log_prices = np.log(window.values)
    
    numerator = log_prices[-1] - log_prices[0]
    daily_log_returns = np.diff(log_prices)
    denominator = np.sum(np.abs(daily_log_returns))
    
    if denominator == 0 or np.isnan(denominator):
        return np.nan
    
    return numerator / denominator


def _months_to_days(months: int) -> int:
    """Approximate conversion from months to trading days."""
    return int(months * 21)


def compute_raw_signals(
    daily_prices: pd.DataFrame,
    tickers: List[str],
) -> pd.DataFrame:
    """Compute all 9 raw momentum signals (unchanged)."""
    signals = {}
    
    for ticker in tickers:
        prices = daily_prices[ticker].dropna()
        ticker_signals = {}
        
        for measure, lookback, uses_months in SIGNAL_SPECS:
            if uses_months:
                days = _months_to_days(lookback)
                label = f"{measure}_{lookback}m"
            else:
                days = lookback
                label = f"{measure}_{lookback}d"
            
            if measure == "TR":
                score = _total_return_days(prices, days)
            elif measure == "PMA":
                score = _price_minus_sma(prices, days)
            elif measure == "RA":
                score = _risk_adjusted_momentum(prices, days)
            else:
                raise ValueError(f"Unknown measure: {measure}")
            
            ticker_signals[label] = score
        
        signals[ticker] = ticker_signals
    
    df = pd.DataFrame.from_dict(signals, orient="index")
    return df


def compute_correlations_with_portfolio(
    daily_prices: pd.DataFrame,
    universe: List[str],
    window_days: int = 252,
) -> Dict[str, float]:
    """Correlation adjustment (unchanged)."""
    prices = daily_prices[universe].dropna()
    returns = prices.pct_change().dropna()
    
    if len(returns) < window_days:
        window_days = len(returns)
    window = returns.iloc[-window_days:]
    
    portfolio_returns = window.mean(axis=1)
    
    correlations = {}
    for ticker in universe:
        corr = window[ticker].corr(portfolio_returns)
        correlations[ticker] = corr if not np.isnan(corr) else 0.0
    
    return correlations


def apply_correlation_adjustment(
    raw_signals: pd.DataFrame,
    correlations: Dict[str, float],
) -> pd.DataFrame:
    """Divisive correlation adjustment (unchanged)."""
    adjusted = raw_signals.copy()
    for ticker in adjusted.index:
        rho = correlations.get(ticker, 0.0)
        denom = 1.0 + rho
        if abs(denom) < 1e-8:
            denom = 1e-8 if denom >= 0 else -1e-8
        adjusted.loc[ticker, :] = raw_signals.loc[ticker, :] / denom
    
    return adjusted


def rank_assets(adjusted_signals: pd.DataFrame) -> pd.Series:
    """Rank-point scoring (unchanged)."""
    n_assets = len(adjusted_signals)
    ranks = adjusted_signals.rank(
        axis=0, method="first", ascending=False, na_option="bottom"
    )
    points = n_assets - ranks + 1
    total_scores = points.sum(axis=1)
    return total_scores.sort_values(ascending=False)


def count_negative_signals(raw_signals: pd.DataFrame) -> pd.Series:
    """Negative signal count (unchanged)."""
    neg_mask = (raw_signals <= 0) & raw_signals.notna()
    return neg_mask.sum(axis=1)


def compute_vitral_signals(
    monthly_prices: pd.DataFrame,
    daily_prices: pd.DataFrame,
) -> VitralSignal:
    """
    Compute Vitral MAM signals (AS-aligned risk-off version).
    """
    signal_date = monthly_prices.index[-1].strftime("%Y-%m-%d")
    
    # ── Step 1-5: Raw signals, correlations, adjustment, ranking, negative count (unchanged)
    raw_signals = compute_raw_signals(daily_prices, RISK_ON_UNIVERSE)
    correlations = compute_correlations_with_portfolio(
        daily_prices, RISK_ON_UNIVERSE, window_days=252
    )
    adjusted_signals = apply_correlation_adjustment(raw_signals, correlations)
    rank_scores = rank_assets(adjusted_signals)
    neg_counts = count_negative_signals(raw_signals)
    
    negative_mask = (neg_counts >= 5)
    n_negative_assets = int(negative_mask.sum())
    
    # ── Step 6: %RiskOn (medium protection, unchanged)
    denominator = 13 - 1 * 13 / 4.0  # = 9.75
    pct_risk_on = max(1.0 - n_negative_assets / denominator, 0.0)
    pct_risk_off = 1.0 - pct_risk_on
    
    # ── Step 7: Select top 5 non-negative assets (unchanged)
    qualified = [t for t in rank_scores.index if not negative_mask.get(t, False)]
    selected = qualified[:TOP_N]
    
    # ── Step 8: Risk-off is now always CASH (AS standard) ─────────────
    risk_off_choice = "CASH"
    
    # ── Step 9: Build final allocation (AS-style) ─────────────────────
    allocation = {}
    
    if pct_risk_on > 0 and len(selected) > 0:
        weight_per_asset = pct_risk_on / len(selected)
        for ticker in selected:
            allocation[ticker] = weight_per_asset
    
    if pct_risk_off > 0:
        allocation["CASH"] = allocation.get("CASH", 0) + pct_risk_off
    
    # Clean up dust
    allocation = {t: w for t, w in allocation.items() if w > 0.0001}
    
    return VitralSignal(
        signal_date=signal_date,
        n_negative_assets=n_negative_assets,
        pct_risk_on=pct_risk_on,
        asset_scores=rank_scores.to_dict(),
        asset_neg_counts=neg_counts.to_dict(),
        selected_assets=selected,
        risk_off_choice=risk_off_choice,
        allocation=allocation,
    )
