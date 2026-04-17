"""
Shared momentum computation functions used by all three TAA strategies.

Formulas per Appendix A of the Investment Policy Statement.
"""

import pandas as pd
import numpy as np


def total_return(prices: pd.Series, months: int) -> float:
    """
    Compute total return over the past N months using month-end prices.
    prices: Series of month-end adjusted close prices, oldest first.
    months: lookback in months.
    Returns: decimal return (e.g., 0.05 for 5%).
    """
    if len(prices) < months + 1:
        return np.nan
    current = prices.iloc[-1]
    past = prices.iloc[-(months + 1)]
    if past == 0 or np.isnan(past):
        return np.nan
    return (current / past) - 1.0


def momentum_13612u(prices: pd.Series) -> float:
    """
    HAA / Vitral momentum: unweighted average of 1, 3, 6, 12 month returns.
    
    Formula: (r1 + r3 + r6 + r12) / 4
    
    Per Keller & Keuning (2023), Section 4.
    """
    r1 = total_return(prices, 1)
    r3 = total_return(prices, 3)
    r6 = total_return(prices, 6)
    r12 = total_return(prices, 12)
    
    if any(np.isnan(x) for x in [r1, r3, r6, r12]):
        return np.nan
    
    return (r1 + r3 + r6 + r12) / 4.0


def momentum_13612w(prices: pd.Series) -> float:
    """
    KDA / DAA / VAA momentum: weighted average emphasizing recent months.
    
    Formula: (12 * r1) + (4 * r3) + (2 * r6) + (1 * r12)
    
    Per Keller & Keuning (2018), used in Kipnis KDA (2019).
    Note: this is NOT divided by 19. The raw weighted sum is used for 
    ranking and sign-testing. Division by 19 would not change rankings
    or the sign, so it's omitted per convention.
    """
    r1 = total_return(prices, 1)
    r3 = total_return(prices, 3)
    r6 = total_return(prices, 6)
    r12 = total_return(prices, 12)
    
    if any(np.isnan(x) for x in [r1, r3, r6, r12]):
        return np.nan
    
    return (12 * r1) + (4 * r3) + (2 * r6) + (1 * r12)
