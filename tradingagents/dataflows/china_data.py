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


# ============================================================================
# Company Announcements (akshare stock_zh_a_disclosure)
# ============================================================================


def get_company_announcements_china(
    ticker: str,
    curr_date: str = None,
) -> str:
    """Retrieve company announcements/disclosures for an A-stock.

    Uses akshare's ``stock_zh_a_disclosure`` which scrapes the official
    exchange disclosure platform (巨潮资讯网). Returns structured
    announcement metadata (title, date, type).

    Args:
        ticker: Stock code (e.g. "688299")
        curr_date: Analysis date; filters to recent announcements

    Returns:
        Formatted list of company announcements
    """
    code = _strip_china_suffix(ticker)
    suffix = _detect_china_suffix(code)
    if suffix is None:
        return f"# Announcements for {ticker}: Not a recognized A-stock code\n"

    lines = [f"# Company Announcements for {ticker}"]
    lines.append(f"# Retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    try:
        # stock_zh_a_disclosure: 巨潮资讯网-个股披露公告
        # Returns columns: 代码, 简称, 公告标题, 公告类型, 公告日期, 全文链接
        df = ak.stock_zh_a_disclosure(symbol=code)

        if df is None or df.empty:
            lines.append("No recent announcements found for this stock.")
            return "\n".join(lines)

        shown = 0
        max_items = 25
        for _, row in df.iterrows():
            if shown >= max_items:
                break
            title = row.get("公告标题") or row.get("title", "")
            category = row.get("公告类型") or row.get("type", "")
            date_val = row.get("公告日期") or row.get("date", "")
            url = row.get("全文链接") or row.get("url", "")
            code_val = row.get("代码") or row.get("code", "")
            name_val = row.get("简称") or row.get("name", "")

            shown += 1
            parts = [f"{shown}. {title}"]
            if category and str(category).strip():
                parts.append(f"   类型: {category}")
            if date_val and str(date_val).strip():
                parts.append(f"   日期: {date_val}")
            if url and str(url).strip():
                parts.append(f"   链接: {url}")
            lines.append("\n".join(parts))
            lines.append("")

        lines.insert(3, f"# Showing {shown} out of {len(df)} announcements")
        lines.insert(3, "")
        return "\n".join(lines)

    except Exception as e:
        logger.warning("akshare announcement API failed for %s: %s", code, e)

        # Fallback: try stock_info_a_code_name for basic disclosure
        fallback = _get_announcements_fallback(code, lines)
        return fallback


def _get_announcements_fallback(code: str, header_lines: list) -> str:
    """Fallback: try alternative announcement sources."""
    lines = list(header_lines)

    # Try Sina financial report as a proxy for major announcements
    try:
        df = ak.stock_financial_report_sina(stock="sh" + code, symbol="资产负债表")
        if df is not None and not df.empty:
            lines.append("## 新浪财报披露（最近报告期）")
            lines.append(df.head(3).to_string(index=False))
            lines.append("")
    except Exception:
        pass

    if len(lines) <= 3:
        lines.append("未找到公告数据。")
    return "\n".join(lines)


# ============================================================================
# Research Reports (akshare stock_research_report_em)
# ============================================================================


def get_research_reports_china(
    ticker: str,
) -> str:
    """Retrieve analyst research reports for an A-stock.

    Uses akshare's ``stock_research_report_em`` which scrapes Eastmoney's
    research report database. Each entry includes: title, rating, target
    price, research institution, analyst name, and report date.

    Args:
        ticker: Stock code (e.g. "688299")

    Returns:
        Formatted list of research reports
    """
    code = _strip_china_suffix(ticker)
    suffix = _detect_china_suffix(code)
    if suffix is None:
        return f"# Research Reports for {ticker}: Not a recognized A-stock code\n"

    lines = [f"# Analyst Research Reports for {ticker}"]
    lines.append(f"# Retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    try:
        # stock_research_report_em: 东方财富-个股研报
        # Returns: 序号, 股票代码, 股票简称, 报告名称, 评级, 目标价,
        #          机构名称, 作者, 报告日期, 报告摘要, 报告类型
        df = ak.stock_research_report_em(symbol=code)

        if df is None or df.empty:
            lines.append("No recent research reports found.")
            return "\n".join(lines)

        shown = 0
        max_items = 20
        for _, row in df.iterrows():
            if shown >= max_items:
                break
            name = row.get("报告名称") or row.get("title", "No title")
            rating = row.get("评级") or row.get("rating", "")
            target = row.get("目标价") or row.get("target_price", "")
            org = row.get("机构名称") or row.get("org", "")
            author = row.get("作者") or row.get("author", "")
            pub_date = row.get("报告日期") or row.get("pub_date", "")
            summary = row.get("报告摘要") or row.get("summary", "")

            shown += 1
            parts = [f"{shown}. {name}"]
            if org and str(org).strip():
                parts.append(f"   机构: {org}")
            meta = []
            if rating and str(rating).strip():
                meta.append(f"评级: {rating}")
            if target and str(target).strip():
                meta.append(f"目标价: {target}")
            if meta:
                parts.append(f"   {', '.join(meta)}")
            if pub_date and str(pub_date).strip():
                parts.append(f"   日期: {pub_date}")
            if summary and str(summary).strip():
                parts.append(f"   摘要: {str(summary)[:200]}")
            lines.append("\n".join(parts))
            lines.append("")

        lines.insert(3, f"# Showing {shown} out of {len(df)} research reports")
        lines.insert(3, "")
        return "\n".join(lines)

    except Exception as e:
        logger.warning("akshare research report API failed for %s: %s", code, e)
        lines.append(f"研报数据暂时不可用 ({str(e)[:100]})。")
        return "\n".join(lines)


# ============================================================================
# Multi-source Global News (Sina + Eastmoney + optionally CLS)
# ============================================================================


def get_market_news_china(
    curr_date: str,
    look_back_days: int = 7,
    limit: int = 15,
) -> str:
    """Retrieve multi-source Chinese market news.

    Aggregates from Eastmoney global news (stock_info_global_em) and
    Sina finance alerts (stock_info_global_sina) for broader coverage.

    Args:
        curr_date: Current date in yyyy-mm-dd format
        look_back_days: Days to look back
        limit: Max articles per source

    Returns:
        Combined market news from multiple sources
    """
    lines = [
        "# China Market News (Multi-Source)",
        f"# Retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"# Lookback: {look_back_days} days, limit: {limit} per source",
        "",
    ]

    total_shown = 0

    # Source 1: Eastmoney global financial news (already proven working)
    lines.append("## 东方财富全球财经快讯")
    try:
        df_em = ak.stock_info_global_em()
        if df_em is not None and not df_em.empty:
            shown = 0
            for _, row in df_em.iterrows():
                if shown >= limit:
                    break
                title = row.get("标题") or row.get("title", "")
                content = row.get("内容") or row.get("content", "")
                pub_time = row.get("发布时间") or row.get("pub_time", "")
                if title:
                    shown += 1
                    total_shown += 1
                    lines.append(f"  {shown}. {title}")
                    if content:
                        lines.append(f"     {str(content)[:150]}")
                    if pub_time:
                        lines.append(f"     ({pub_time})")
            lines.append(f"  ({shown} articles)")
        else:
            lines.append("  No data from this source.")
    except Exception as e:
        logger.warning("EM global news failed: %s", e)
        lines.append(f"  Source unavailable.")

    lines.append("")

    # Source 2: Sina finance global alerts
    lines.append("## 新浪财经全球快讯")
    try:
        df_sina = ak.stock_info_global_sina()
        if df_sina is not None and not df_sina.empty:
            shown = 0
            for _, row in df_sina.iterrows():
                if shown >= limit:
                    break
                title = row.get("标题") or row.get("title", "")
                content = row.get("内容") or row.get("content", "")
                pub_time = row.get("发布时间") or row.get("pub_time", "")
                if title:
                    shown += 1
                    total_shown += 1
                    lines.append(f"  {shown}. {title}")
                    if content:
                        lines.append(f"     {str(content)[:150]}")
                    if pub_time:
                        lines.append(f"     ({pub_time})")
            lines.append(f"  ({shown} articles)")
        else:
            lines.append("  No data from this source.")
    except Exception as e:
        logger.warning("Sina global news failed: %s", e)
        lines.append(f"  Source unavailable.")

    lines.append("")
    lines.insert(4, f"# Total articles aggregated: {total_shown}")
    return "\n".join(lines)


# ============================================================================
# Sector/Concept Board News (akshare stock_board_concept_name_em)
# ============================================================================


def get_sector_china(
    symbol: str,
) -> str:
    """Get the sector/concept board membership for a stock.

    Uses akshare's ``stock_board_concept_cons_em`` or similar to identify
    which concept boards the stock belongs to, then fetches sector-level
    news/performance data.

    Args:
        symbol: Stock code (e.g. "688299")

    Returns:
        Sector affiliation and related news
    """
    code = _strip_china_suffix(symbol)
    suffix = _detect_china_suffix(code)
    if suffix is None:
        return f"# Sector for {symbol}: Not a recognized A-stock code\n"

    lines = [f"# Sector & Concept Board Analysis for {symbol}"]
    lines.append(f"# Retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    try:
        # Get all concept boards
        df_all = ak.stock_board_concept_name_em()
        if df_all is None or df_all.empty:
            lines.append("No concept board data available.")
            return "\n".join(lines)

        # Try to find the stock's sector by checking industry classification
        try:
            # stock_individual_info_em returns sector info
            info_df = ak.stock_individual_info_em(symbol=code)
            if info_df is not None and not info_df.empty:
                lines.append("## 所属行业")
                for _, row in info_df.iterrows():
                    item = str(row.get("item") or row.iloc[0])
                    val = str(row.get("value") or row.iloc[1])
                    if any(kw in item for kw in ["行业", "板块", "概念", "分类", "地域"]):
                        lines.append(f"  {item}: {val}")
                lines.append("")
        except Exception:
            pass

        # Show top concept boards by change rate (market context)
        lines.append("## 热门概念板块（涨幅最大）")
        try:
            top_boards = df_all.nlargest(10, "涨跌幅") if "涨跌幅" in df_all.columns else df_all.head(10)
            for _, row in top_boards.iterrows():
                name = row.get("板块名称") or str(row.iloc[0])[:20]
                pct = row.get("涨跌幅") if "涨跌幅" in row.index else "N/A"
                lines.append(f"  {name}: {pct}%")
        except Exception:
            pass

        lines.append("")
        return "\n".join(lines)

    except Exception as e:
        logger.warning("sector analysis failed for %s: %s", code, e)
        lines.append(f"板块数据暂时不可用 ({str(e)[:100]})。")
        return "\n".join(lines)
