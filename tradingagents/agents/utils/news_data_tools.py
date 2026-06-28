from typing import Annotated

from langchain_core.tools import tool

from tradingagents.dataflows.interface import route_to_vendor


@tool
def get_news(
    ticker: Annotated[str, "Ticker symbol"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
) -> str:
    """
    Retrieve news data for a given ticker symbol.
    Uses the configured news_data vendor.
    Args:
        ticker (str): Ticker symbol
        start_date (str): Start date in yyyy-mm-dd format
        end_date (str): End date in yyyy-mm-dd format
    Returns:
        str: A formatted string containing news data
    """
    return route_to_vendor("get_news", ticker, start_date, end_date)

@tool
def get_global_news(
    curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
    look_back_days: Annotated[int | None, "Days to look back; omit to use the configured default"] = None,
    limit: Annotated[int | None, "Max articles to return; omit to use the configured default"] = None,
) -> str:
    """
    Retrieve global news data.
    Uses the configured news_data vendor. Defaults for look_back_days and
    limit come from DEFAULT_CONFIG (global_news_lookback_days,
    global_news_article_limit); pass explicit values to override.

    Args:
        curr_date (str): Current date in yyyy-mm-dd format
        look_back_days (int): Number of days to look back; omit to inherit config
        limit (int): Maximum number of articles to return; omit to inherit config

    Returns:
        str: A formatted string containing global news data
    """
    return route_to_vendor("get_global_news", curr_date, look_back_days, limit)

@tool
def get_insider_transactions(
    ticker: Annotated[str, "ticker symbol"],
) -> str:
    """
    Retrieve insider transaction information about a company.
    Uses the configured news_data vendor.
    Args:
        ticker (str): Ticker symbol of the company
    Returns:
        str: A report of insider transaction data
    """
    return route_to_vendor("get_insider_transactions", ticker)


@tool
def get_company_announcements(
    ticker: Annotated[str, "Ticker symbol (e.g., '688299')"],
    curr_date: Annotated[str | None, "Current date in yyyy-mm-dd format"] = None,
) -> str:
    """
    Retrieve company announcements and disclosures for a stock.
    Includes board resolutions, financial reports, material events,
    shareholder meetings, and regulatory filings from the official
    exchange disclosure platform (巨潮资讯网). Use this to check for
    recent events that may impact the stock price but are not covered
    by general news feeds.

    Args:
        ticker: Ticker symbol
        curr_date: Analysis date (for filtering recent announcements)

    Returns:
        Formatted list of recent company announcements with dates and types
    """
    return route_to_vendor("get_company_announcements", ticker, curr_date)


@tool
def get_research_reports(
    ticker: Annotated[str, "Ticker symbol (e.g., '688299')"],
) -> str:
    """
    Retrieve analyst research reports for a stock from Eastmoney's
    research report database. Each entry includes the report title,
    analyst rating (买入/增持/中性/减持/卖出), target price, research
    institution, and a summary. Use this to understand professional
    analyst consensus and valuation expectations.

    Args:
        ticker: Ticker symbol

    Returns:
        Formatted list of research reports with ratings and summaries
    """
    return route_to_vendor("get_research_reports", ticker)


@tool
def get_market_news(
    curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
    look_back_days: Annotated[int, "Days to look back"] = 7,
    limit: Annotated[int, "Max articles per source"] = 15,
) -> str:
    """
    Retrieve comprehensive Chinese market news from multiple sources
    (东方财富 and 新浪财经). Provides broader coverage than get_global_news
    by aggregating financial headlines from different media outlets.
    Use this for market-wide sentiment analysis and macro event tracking.

    Args:
        curr_date: Current date in yyyy-mm-dd format
        look_back_days: Number of days to look back
        limit: Maximum number of articles per source

    Returns:
        Aggregated market news from multiple Chinese financial media
    """
    return route_to_vendor("get_market_news", curr_date, look_back_days, limit)


@tool
def get_sector(
    symbol: Annotated[str, "Stock code (e.g., '688299')"],
) -> str:
    """
    Retrieve sector and concept board information for a stock.
    Identifies which industry sectors and concept boards the stock belongs
    to, and provides current hot sector rankings. Use this to understand
    the stock's sector positioning and whether its sector is in favor.

    Args:
        symbol: Stock code

    Returns:
        Sector affiliation data and hot concept board rankings
    """
    return route_to_vendor("get_sector", symbol)


@tool
def get_hot_rank(
    symbol: Annotated[str, "Stock code (e.g., '000651')"],
) -> str:
    """
    Retrieve stock popularity ranking from 东方财富人气榜 (Eastmoney Hot Rank).
    Shows the stock's current popularity rank among all A-stocks, with
    price and change data. High rank indicates strong retail investor
    attention — useful as a sentiment/momentum signal.

    Args:
        symbol: Stock code

    Returns:
        Popularity ranking and context (top 10 list)
    """
    return route_to_vendor("get_hot_rank", symbol)


@tool
def get_fund_flow(
    symbol: Annotated[str, "Stock code (e.g., '000651')"],
) -> str:
    """
    Retrieve individual stock fund flow data showing daily capital
    inflows/outflows by investor category (超大单/大单/中单/小单).
    Positive major fund ($10M+) net inflow suggests institutional
    accumulation; persistent outflow signals distribution. Covers
    recent 20 trading days.

    Args:
        symbol: Stock code

    Returns:
        Daily fund flow table with net major capital direction
    """
    return route_to_vendor("get_fund_flow", symbol)


@tool
def get_profit_forecast(
    symbol: Annotated[str, "Stock code (e.g., '000651')"],
) -> str:
    """
    Retrieve analyst consensus earnings forecasts from 同花顺 (THS).
    Shows EPS estimates (min/mean/max) for upcoming fiscal years,
    number of analysts covering, and industry average for comparison.
    Use this to understand market expectations for future profitability.

    Args:
        symbol: Stock code

    Returns:
        Consensus EPS forecasts by fiscal year with industry context
    """
    return route_to_vendor("get_profit_forecast", symbol)
