"""
Data fetching layer. Downloads monthly adjusted-close prices from Yahoo Finance + FRED API.
"""

import yfinance as yf
import pandas as pd
import json
import os
from datetime import datetime, timedelta
from typing import List, Optional
import logging
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

# Ticker aliases (none needed for this model)
TICKER_ALIASES = {}


def resolve_ticker(ticker: str) -> str:
    return TICKER_ALIASES.get(ticker, ticker)


def fetch_monthly_prices(
    tickers: List[str],
    months_history: int = 15,
    end_date: Optional[datetime] = None,
) -> pd.DataFrame:
    """Fetch month-end adjusted close prices from Yahoo Finance."""
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
        raise RuntimeError("Yahoo Finance returned no data. Check tickers and dates.")
    
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
            logger.warning(f"No data for {strategy_ticker} ({yahoo_ticker})")
            result[strategy_ticker] = float("nan")
    
    result = result.dropna(how="all")
    
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
    """Fetch daily adjusted close prices from Yahoo Finance."""
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
            logger.warning(f"No daily data for {strategy_ticker} ({yahoo_ticker})")
            result[strategy_ticker] = float("nan")
    
    result = result.ffill().dropna(how="all")
    
    if len(result) < 200:
        raise RuntimeError(
            f"Only {len(result)} days of data available, need at least 200 "
            f"for correlation calculation."
        )
    
    return result


def fetch_unemployment_rate(end_date: Optional[datetime] = None) -> pd.Series:
    """
    Fetch US Unemployment Rate (UNRATE) using the official FRED API.
    Uses your FRED_API_KEY stored in GitHub Secrets (exactly like RESEND_API_KEY).
    """
    if end_date is None:
        end_date = datetime.now()
    
    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        logger.warning("FRED_API_KEY not set — using fallback unemployment rate (4.1%)")
        dates = pd.date_range(end=end_date, periods=36, freq="ME")
        return pd.Series([4.1] * 36, index=dates)

    url = (
        f"https://api.stlouisfed.org/fred/series/observations?"
        f"series_id=UNRATE&api_key={api_key}&file_type=json"
        f"&limit=0&sort_order=asc"   # fetch all observations
    )

    for attempt in range(3):
        try:
            req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(req, timeout=15) as response:
                data = json.loads(response.read().decode("utf-8"))
            
            observations = data["observations"]
            df = pd.DataFrame(observations)
            df = df[df["value"] != "."]                     # remove missing values
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
            ue = df["value"].astype(float)
            ue = ue.resample("ME").last()
            
            logger.info(f"Successfully fetched {len(ue)} months of UNRATE via FRED API")
            return ue
            
        except Exception as e:
            logger.warning(f"FRED API attempt {attempt+1} failed: {e}")
            if attempt < 2:
                time.sleep(2 ** attempt)  # backoff

    logger.error("FRED API failed after 3 attempts — using fallback")
    dates = pd.date_range(end=end_date, periods=36, freq="ME")
    return pd.Series([4.1] * 36, index=dates)


def get_last_trading_day(date: Optional[datetime] = None) -> datetime:
    """Determine the last trading day of the month."""
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
