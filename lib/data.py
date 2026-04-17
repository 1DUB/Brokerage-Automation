"""
Data fetching layer. Downloads monthly adjusted-close prices from Yahoo Finance + FRED.
"""

import yfinance as yf
import pandas as pd
import pandas_datareader.data as web
from datetime import datetime, timedelta
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

# Ticker aliases (none needed for this model — all use standard liquid ETFs)
TICKER_ALIASES = {}


def resolve_ticker(ticker: str) -> str:
    return TICKER_ALIASES.get(ticker, ticker)


def fetch_monthly_prices(
    tickers: List[str],
    months_history: int = 15,
    end_date: Optional[datetime] = None,
) -> pd.DataFrame:
    """(Unchanged from your original — fetches Yahoo prices)"""
    if end_date is None:
        end_date = datetime.now()
    start_date = end_date - timedelta(days=(months_history + 3) * 31)
    
    resolved = {t: resolve_ticker(t) for t in tickers}
    yahoo_tickers = list(set(resolved.values()))
    
    logger.info(f"Fetching {len(yahoo_tickers)} tickers from Yahoo Finance...")
    
    raw = yf.download(
        yahoo_tickers,
        start=start_date.strftime("%Y-%m-%d"),
        end=(end_date + timedelta(days=1)).strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
    )
    
    if raw.empty:
        raise RuntimeError("Yahoo Finance returned no data.")
    
    if len(yahoo_tickers) == 1:
        close = raw[["Close"]].copy()
        close.columns = [yahoo_tickers[0]]
    else:
        close = raw["Close"].copy()
    
    monthly = close.resample("ME").last()
    
    result = pd.DataFrame(index=monthly.index)
    for strategy_ticker, yahoo_ticker in resolved.items():
        if yahoo_ticker in monthly.columns:
            result[strategy_ticker] = monthly[yahoo_ticker]
        else:
            result[strategy_ticker] = float("nan")
    
    result = result.dropna(how="all")
    
    if len(result) < 13:
        raise RuntimeError(f"Only {len(result)} months of data available.")
    
    return result


def fetch_daily_prices(
    tickers: List[str],
    months_history: int = 13,
    end_date: Optional[datetime] = None,
) -> pd.DataFrame:
    """(Unchanged — used by Stoken if needed)"""
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
    
    result = pd.DataFrame(index=close.index)
    for strategy_ticker, yahoo_ticker in resolved.items():
        if yahoo_ticker in close.columns:
            result[strategy_ticker] = close[yahoo_ticker]
        else:
            result[strategy_ticker] = float("nan")
    
    result = result.ffill().dropna(how="all")
    return result


def fetch_unemployment_rate(end_date: Optional[datetime] = None) -> pd.Series:
    """
    Fetch monthly US Unemployment Rate (UNRATE) from FRED.
    Used by Lethargic AA for Growth-Trend Timing.
    """
    if end_date is None:
        end_date = datetime.now()
    
    start = end_date - timedelta(days=20*365)  # ~20 years buffer
    
    ue = web.DataReader('UNRATE', 'fred', start, end_date)
    ue = ue.resample('ME').last()
    return ue['UNRATE']


def get_last_trading_day(date: Optional[datetime] = None) -> datetime:
    """(Unchanged from your original)"""
    if date is None:
        date = datetime.now()
    if date.month == 12:
        last_day = datetime(date.year + 1, 1, 1) - timedelta(days=1)
    else:
        last_day = datetime(date.year, date.month + 1, 1) - timedelta(days=1)
    while last_day.weekday() > 4:
        last_day -= timedelta(days=1)
    return last_day


def is_last_trading_day(date: Optional[datetime] = None) -> bool:
    if date is None:
        date = datetime.now()
    return date.date() == get_last_trading_day(date).date()


def is_first_trading_day(date: Optional[datetime] = None) -> bool:
    if date is None:
        date = datetime.now()
    first_day = datetime(date.year, date.month, 1)
    while first_day.weekday() > 4:
        first_day += timedelta(days=1)
    return date.date() == first_day.date()
