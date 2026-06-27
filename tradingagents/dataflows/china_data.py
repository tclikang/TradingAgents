"""Chinese financial data sources via akshare.

All A-stock data (prices, news, fundamentals, financial reports) sourced
from akshare, which wraps Eastmoney, Sina, and other free Chinese APIs.
Functions match the vendor interface expected by ``interface.py``.
"""

from __future__ import annotations

import contextlib
import logging
from datetime import datetime, timedelta
from typing import Annotated

import akshare as ak
import pandas as pd
from dateutil.relativedelta import relativedelta

from .symbol_utils import _detect_china_suffix

logger = logging.getLogger(__name__)


def _strip_china_suffix(symbol: str) -> str:
    """Remove .SS/.SZ/.BJ suffix to get raw 6-digit code."""
    for s in (".SS", ".SZ", ".BJ"):
        if symbol.upper().endswith(s):
            return symbol[:-len(s)]
    return symbol


# ============================================================================
# Stock Price Data (akshare stock_zh_a_hist)
# ============================================================================


def get_stock_data_china(
    symbol: str,
    start_date: str,
    end_date: str,
) -> str:
    """Retrieve A-stock OHLCV price data via akshare.

    Args:
        symbol: Stock code (e.g. "600519" or "600519.SS")
        start_date: Start date in yyyy-mm-dd format
        end_date: End date in yyyy-mm-dd format

    Returns:
        CSV-formatted price history
    """
    code = _strip_china_suffix(symbol)
    suffix = _detect_china_suffix(code)
    if suffix is None:
        return f"# Stock data for {symbol}: Not a recognized A-stock code\n"

    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")

        # akshare stock_zh_a_hist: daily K-line data from Eastmoney
        # period='daily', adjust='qfq' (前复权) gives clean adjusted prices
        df = ak.stock_zh_a_hist(
            symbol=code,
            period="daily",
            start_date=start_date.replace("-", ""),
            end_date=end_date.replace("-", ""),
            adjust="qfq",
        )

        if df is None or df.empty:
            return (
                f"# Stock data for {symbol}\n"
                f"No data between {start_date} and {end_date}\n"
            )

        # Normalize columns to OHLCV format
        col_map = {
            "日期": "Date",
            "开盘": "Open",
            "收盘": "Close",
            "最高": "High",
            "最低": "Low",
            "成交量": "Volume",
            "成交额": "Amount",
            "振幅": "Amplitude",
            "涨跌幅": "Pct_Change",
            "涨跌额": "Change",
            "换手率": "Turnover",
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

        # Ensure Date column is datetime without timezone
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

        # Round prices to 2 decimal places
        for col in ["Open", "High", "Low", "Close"]:
            if col in df.columns:
                df[col] = df[col].round(2)

        # Add header
        header = f"# Stock data for {symbol} from {start_date} to {end_date}\n"
        header += f"# Total records: {len(df)}\n"
        header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

        return header + df.to_csv(index=False)

    except Exception as e:
        logger.warning("akshare stock data failed for %s: %s", symbol, e)
        return f"# Stock data for {symbol}: Failed to fetch ({e})\n"


# ============================================================================
# News APIs (akshare stock_news_em / stock_info_global_em)
# ============================================================================


def get_news_china(
    ticker: str,
    start_date: str,
    end_date: str,
) -> str:
    """Retrieve news for a Chinese stock using akshare (Eastmoney source).

    Args:
        ticker: Stock code (e.g. "600519" or "600519.SS")
        start_date: Start date in yyyy-mm-dd format
        end_date: End date in yyyy-mm-dd format

    Returns:
        Formatted string containing news articles
    """
    code = _strip_china_suffix(ticker)
    suffix = _detect_china_suffix(code)
    if suffix is None:
        return f"# News for {ticker}: Not a recognized A-stock code\n"

    try:
        # akshare wraps Eastmoney individual stock news API
        df = ak.stock_news_em(symbol=code)

        if df is None or df.empty:
            return f"# News for {ticker}: No recent news available\n"

        lines = [f"# Stock News for {ticker}"]
        lines.append(f"# Retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # News columns: 关键词, 新闻标题, 新闻内容, 发布时间, 文章来源, 新闻链接
        col_map = {
            "新闻标题": "title",
            "新闻内容": "content",
            "发布时间": "pub_time",
            "文章来源": "source",
            "新闻链接": "url",
        }

        shown = 0
        max_articles = 20
        for _, row in df.iterrows():
            if shown >= max_articles:
                break
            title = row.get("新闻标题") or row.get("title", "No title")
            content = row.get("新闻内容") or row.get("content", "")
            pub_time = row.get("发布时间") or row.get("pub_time", "")
            source = row.get("文章来源") or row.get("source", "")
            url = row.get("新闻链接") or row.get("url", "")

            shown += 1
            lines.append(f"{shown}. {title}")
            if content:
                lines.append(f"   {str(content)[:200]}")
            if pub_time:
                lines.append(f"   时间: {pub_time}")
            if source:
                lines.append(f"   来源: {source}")
            lines.append("")

        return "\n".join(lines)

    except Exception as e:
        logger.warning("akshare news failed for %s: %s", ticker, e)
        return f"# News for {ticker}: Failed to fetch ({e})\n"


def get_global_news_china(
    curr_date: str,
    look_back_days: int | None = None,
    limit: int | None = None,
) -> str:
    """Retrieve Chinese market news using akshare (Eastmoney global financial news).

    Args:
        curr_date: Current date in yyyy-mm-dd format
        look_back_days: Days to look back
        limit: Max articles

    Returns:
        Formatted market news headlines
    """
    from .config import get_config

    config = get_config()
    if look_back_days is None:
        look_back_days = config.get("global_news_lookback_days", 7)
    if limit is None:
        limit = config.get("global_news_article_limit", 10)

    try:
        # stock_info_global_em: Eastmoney global financial news
        df = ak.stock_info_global_em()

        lines = ["# China Market News"]
        lines.append(f"# Retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        if df is None or df.empty:
            return "\n".join(lines) + "\nNo market news found.\n"

        # Columns typically: 标题, 内容, 发布时间, 来源
        for i, (_, row) in enumerate(df.iterrows()):
            if i >= limit:
                break
            title = row.get("标题") or row.get("title", "No title")
            content = row.get("内容") or row.get("content", "")
            pub_time = row.get("发布时间") or row.get("pub_time", "")

            lines.append(f"{i + 1}. {title}")
            if content:
                lines.append(f"   {str(content)[:150]}...")
            if pub_time:
                lines.append(f"   ({pub_time})")
            lines.append("")

        return "\n".join(lines)

    except Exception as e:
        logger.warning("akshare global news failed: %s", e)
        return "# China Market News\nFailed to fetch (network error)\n"


# ============================================================================
# Fundamentals (akshare financial report APIs)
# ============================================================================


def _get_fundamentals_abstract(code: str) -> pd.DataFrame | None:
    """Fetch key financial indicators from THS via akshare."""
    try:
        # stock_financial_abstract_ths returns key indicators by report date
        # indicator options: "按报告期" (by report period)
        df = ak.stock_financial_abstract_ths(symbol=code, indicator="按报告期")
        return df
    except Exception as e:
        logger.warning("akshare financial abstract failed for %s: %s", code, e)
        return None


def get_fundamentals_china(
    ticker: str,
    curr_date: str,
) -> str:
    """Retrieve comprehensive fundamental data for an A-stock via akshare.

    Args:
        ticker: Stock code (e.g. "600519")
        curr_date: Current date in yyyy-mm-dd format

    Returns:
        Formatted fundamental data report
    """
    code = _strip_china_suffix(ticker)
    suffix = _detect_china_suffix(code)
    if suffix is None:
        return f"# Fundamentals for {ticker}: Not a recognized A-stock code\n"

    parts = [f"# Fundamental Data for {ticker}", ""]

    try:
        # Basic stock info + real-time quote
        info_df = ak.stock_individual_info_em(symbol=code)
        if info_df is not None and not info_df.empty:
            parts.append("## 基本信息")
            for _, row in info_df.iterrows():
                key = row.get("item") or row.iloc[0]
                val = row.get("value") or row.iloc[1]
                parts.append(f"  {key}: {val}")
            parts.append("")
    except Exception as e:
        logger.warning("stock info failed for %s: %s", code, e)

    # Financial abstract (key indicators across years/reports)
    df = _get_fundamentals_abstract(code)
    if df is not None and not df.empty:
        parts.append("## 核心财务指标")
        parts.append(df.tail(15).to_string(index=False))
        parts.append("")

    if len(parts) <= 2:
        return f"# Fundamentals for {ticker}: No data available\n"

    return "\n".join(parts)


def _get_financial_report_em(code: str) -> pd.DataFrame | None:
    """Fetch the latest financial report from Eastmoney via akshare."""
    try:
        # stock_balance_sheet_by_report_em gets balance sheet by report date
        df = ak.stock_balance_sheet_by_report_em(symbol=code)
        return df
    except Exception:
        return None


def _try_balance_sheet_fallback(code: str) -> str:
    """Try multiple akshare APIs to get balance sheet data."""
    lines = []

    # Try Eastmoney balance sheet
    with contextlib.suppress(Exception):
        df = ak.stock_balance_sheet_by_report_em(symbol=code)
        if df is not None and not df.empty:
            lines.append(df.tail(8).to_string(index=False))
            return "\n".join(lines)

    # Fallback: financial abstract includes some balance-sheet items
    df = _get_fundamentals_abstract(code)
    if df is not None and not df.empty:
        lines.append("(摘要数据)")
        lines.append(df.tail(8).to_string(index=False))
        return "\n".join(lines)

    return "No data available"


def get_balance_sheet_china(
    ticker: str,
    freq: str = "quarterly",
    curr_date: str = None,
) -> str:
    """Retrieve balance sheet data for an A-stock via akshare.

    Args:
        ticker: Stock code
        freq: Reporting frequency (quarterly/annual)
        curr_date: Current date

    Returns:
        Formatted balance sheet report
    """
    code = _strip_china_suffix(ticker)
    suffix = _detect_china_suffix(code)
    if suffix is None:
        return f"# Balance Sheet for {ticker}: Not a recognized A-stock code\n"

    header = f"# Balance Sheet for {ticker}\n"
    body = _try_balance_sheet_fallback(code)
    return header + body + "\n"


def get_cashflow_china(
    ticker: str,
    freq: str = "quarterly",
    curr_date: str = None,
) -> str:
    """Retrieve cash flow data for an A-stock via akshare."""
    code = _strip_china_suffix(ticker)
    suffix = _detect_china_suffix(code)
    if suffix is None:
        return f"# Cash Flow for {ticker}: Not a recognized A-stock code\n"

    try:
        df = ak.stock_cash_flow_sheet_by_report_em(symbol=code)
        if df is not None and not df.empty:
            return f"# Cash Flow Statement for {ticker}\n{df.tail(8).to_string(index=False)}\n"
    except Exception as e:
        logger.warning("cash flow API failed for %s: %s", code, e)

    # Fallback to financial abstract
    df = _get_fundamentals_abstract(code)
    if df is not None and not df.empty:
        return f"# Cash Flow for {ticker} (summary)\n{df.tail(8).to_string(index=False)}\n"
    return f"# Cash Flow for {ticker}: No data available\n"


def get_income_statement_china(
    ticker: str,
    freq: str = "quarterly",
    curr_date: str = None,
) -> str:
    """Retrieve income statement data for an A-stock via akshare."""
    code = _strip_china_suffix(ticker)
    suffix = _detect_china_suffix(code)
    if suffix is None:
        return f"# Income Statement for {ticker}: Not a recognized A-stock code\n"

    try:
        df = ak.stock_profit_sheet_by_report_em(symbol=code)
        if df is not None and not df.empty:
            return f"# Income Statement for {ticker}\n{df.tail(8).to_string(index=False)}\n"
    except Exception as e:
        logger.warning("income statement API failed for %s: %s", code, e)

    # Fallback to financial abstract
    df = _get_fundamentals_abstract(code)
    if df is not None and not df.empty:
        return f"# Income Statement for {ticker} (summary)\n{df.tail(8).to_string(index=False)}\n"
    return f"# Income Statement for {ticker}: No data available\n"


def get_insider_transactions_china(
    ticker: str,
) -> str:
    """Insider transactions not available via free A-stock APIs.
    Returns a sentinel so the agent doesn't hallucinate.
    """
    code = _strip_china_suffix(ticker)
    return (
        f"# Insider Transactions for {code}\n"
        f"A股内幕交易数据暂不可用（免费API限制）。请参考交易所公告。\n"
    )
