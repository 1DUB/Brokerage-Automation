"""
Data fetching layer. Downloads monthly adjusted-close prices from Yahoo Finance.

Designed to be swappable if Yahoo Finance changes or dies.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

# Ticker aliases: only keep if needed for future delistings/renames
TICKER_ALIASES = {
    # No EFA/EEM aliases needed — strategies now use IEFA/IEMG directly
}


def resolve_ticker(ticker: str) -> str:
    """Resolve a strategy ticker to its current Yahoo Finance ticker."""
    return TICKER_ALIASES.get(ticker, ticker)


def fetch_monthly_prices(
    tickers: List[str],
    months_history: int = 15,
    end_date: Optional[datetime] = None,
) -> pd.DataFrame:
    """
    Fetch month-end adjusted close prices for a list of tickers.
    
    Args:
        tickers: List of ETF ticker symbols.
        months_history: How many months of history to fetch (13 minimum 
                        for 12-month momentum; we fetch extra as buffer).
        end_date: Last date to include. Defaults to today.
    
    Returns:
        DataFrame with month-end dates as index, tickers as columns,
        values are dividend-adjusted close prices.
    """
    if end_date is None:
        end_date = datetime.now()
    
    # Fetch extra buffer for safety
    start_date = end_date - timedelta(days=(months_history + 3) * 31)
    
    resolved = {t: resolve_ticker(t) for t in tickers}
    yahoo_tickers = list(set(resolved.values()))
    
    logger.info(f"Fetching {len(yahoo_tickers)} tickers from Yahoo Finance...")
    
    # Download all at once for efficiency
    raw = yf.download(
        yahoo_tickers,
        start=start_date.strftime("%Y-%m-%d"),
        end=(end_date + timedelta(days=1)).strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
    )
    
    if raw.empty:
        raise RuntimeError("Yahoo Finance returned no data. Check tickers and dates.")
    
    # Extract Close prices
    if len(yahoo_tickers) == 1:
        close = raw[["Close"]].copy()
        close.columns = [yahoo_tickers[0]]
    else:
        close = raw["Close"].copy()
    
    # Resample to month-end
    monthly = close.resample("ME").last()
    
    # Map back to strategy tickers
    result = pd.DataFrame(index=monthly.index)
    for strategy_ticker, yahoo_ticker in resolved.items():
        if yahoo_ticker in monthly.columns:
            result[strategy_ticker] = monthly[yahoo_ticker]
        else:
            logger.warning(f"No data for {strategy_ticker} ({yahoo_ticker})")
            result[strategy_ticker] = float("nan")
    
    # Drop rows where all values are NaN
    result = result.dropna(how="all")
    
    # Verify we have enough history
    if len(result) < 13:
        raise RuntimeError(
            f"Only {len(result)} months of data available, need at least 13 "
            f"for 12-month momentum calculation."
        )
    
    return result


def fetch_daily_prices(
    tickers: List[str],
    months_history: int = 13,
    end_date: Optional[datetime] = None,
) -> pd.DataFrame:
    """
    Fetch daily adjusted close prices for a list of tickers.
    
    Used by KDA for daily-return correlation calculations.
    
    Args:
        tickers: List of ETF ticker symbols.
        months_history: How many months of daily history to fetch.
        end_date: Last date to include. Defaults to today.
    
    Returns:
        DataFrame with daily dates as index, tickers as columns,
        values are dividend-adjusted close prices.
    """
    if end_date is None:
        end_date = datetime.now()
    
    start_date = end_date - timedelta(days=(months_history + 2) * 31)
    
    resolved = {t: resolve_ticker(t) for t in tickers}
    yahoo_tickers = list(set(resolved.values()))
    
    logger.info(f"Fetching daily data for {len(yahoo_tickers)} tickers...")
    
    raw = yf.download(
        yahoo_tickers,
        start=start_date.strftime("%Y-%m-%d"),
        end=(end_date + timedelta(days=1)).strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
    )
    
    if raw.empty:
        raise RuntimeError("Yahoo Finance returned no daily data.")
    
    if len(yahoo_tickers) == 1:
        close = raw[["Close"]].copy()
        close.columns = [yahoo_tickers[0]]
    else:
        close = raw["Close"].copy()
    
    # Map back to strategy tickers
    result = pd.DataFrame(index=close.index)
    for strategy_ticker, yahoo_ticker in resolved.items():
        if yahoo_ticker in close.columns:
            result[strategy_ticker] = close[yahoo_ticker]
        else:
            logger.warning(f"No daily data for {strategy_ticker} ({yahoo_ticker})")
            result[strategy_ticker] = float("nan")
    
    # Forward-fill small gaps (holidays etc), then drop fully-NaN rows
    result = result.ffill().dropna(how="all")
    
    if len(result) < 200:  # Need ~12 months of trading days for correlation
        raise RuntimeError(
            f"Only {len(result)} days of data available, need at least 200 "
            f"for 12-month correlation calculation."
        )
    
    return result


def get_last_trading_day(date: Optional[datetime] = None) -> datetime:
    """
    Determine the last trading day of the month for the given date.
    Uses a simple heuristic: the last business day of the month.
    """
    if date is None:
        date = datetime.now()
    
    # Get last day of the month
    if date.month == 12:
        last_day = datetime(date.year + 1, 1, 1) - timedelta(days=1)
    else:
        last_day = datetime(date.year, date.month + 1, 1) - timedelta(days=1)
    
    # Walk back to the last weekday
    while last_day.weekday() > 4:  # 5=Saturday, 6=Sunday
        last_day -= timedelta(days=1)
    
    return last_day


def is_last_trading_day(date: Optional[datetime] = None) -> bool:
    """Check if today is the last trading day of the month."""
    if date is None:
        date = datetime.now()
    return date.date() == get_last_trading_day(date).date()


def is_first_trading_day(date: Optional[datetime] = None) -> bool:
    """Check if today is the first trading day of the month."""
    if date is None:
        date = datetime.now()
    first_day = datetime(date.year, date.month, 1)
    while first_day.weekday() > 4:
        first_day += timedelta(days=1)
    return date.date() == first_day.date()
