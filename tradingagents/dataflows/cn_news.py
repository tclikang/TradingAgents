"""A股国内新闻聚合模块。

聚合四大数据源，覆盖机构研报、个股新闻、实时快讯：
  1. 东方财富个股新闻   (stock_news_em)               — 该股票直接相关新闻
  2. 东方财富个股研报   (stock_research_report_em)     — 券商研报/评级
  3. 同花顺全球财经快讯 (stock_info_global_ths)       — iFinD/10jqka 实时快讯
  4. 新浪财经新闻       (stock_info_global_sina)       — 新浪财经实时新闻

每个函数返回结构化字符串，友好降级（不抛异常）。
"""

from __future__ import annotations

import logging
import re
from typing import List

logger = logging.getLogger(__name__)


def _get_stock_code_only(ticker: str) -> str:
    """从 '688016.SS' 或 '600519.SS' 中提取纯数字代码。"""
    return ticker.split(".")[0]


def _is_valid_stock_code(code: str) -> bool:
    """校验是否为合法的 A 股代码（6 位纯数字）。"""
    return bool(re.fullmatch(r"\d{6}", code))


# ============================================================
#  1. 东方财富 — 个股新闻
# ============================================================

def fetch_eastmoney_news(ticker: str, limit: int = 15) -> str:
    """东方财富个股新闻。

    通过 AKShare 的 stock_news_em 获取该股票相关的新闻报道。
    每条新闻带偏多/偏空/中性标签。

    Args:
        ticker: 如 "688016.SS"
        limit: 最多返回条数

    Returns:
        格式化的 markdown 字符串；失败时返回 <不可用> 占位符。
    """
    code = _get_stock_code_only(ticker)
    if not _is_valid_stock_code(code):
        return f"<东方财富个股新闻: {ticker} 非有效A股代码>"
    try:
        import akshare as ak
    except ImportError:
        return f"<东方财富个股新闻: AKShare 未安装>"

    try:
        df = ak.stock_news_em(symbol=code)
    except Exception as e:
        logger.warning("东方财富个股新闻 %s 失败: %s", ticker, e)
        return f"<东方财富个股新闻: {ticker} 获取失败>"

    if df is None or df.empty:
        return f"<东方财富个股新闻: {ticker} 暂无相关新闻>"

    return _format_news_with_sentiment(
        df, ticker, "东方财富个股新闻", limit,
        title_col="新闻标题", content_col="新闻内容",
        time_col="发布时间", source_col="文章来源",
    )


# ============================================================
#  2. 东方财富 — 个股研报
# ============================================================

def fetch_eastmoney_reports(ticker: str, limit: int = 15) -> str:
    """东方财富个股研报。

    券商对该股票的研究报告，包含评级、盈利预测等专业信息。

    Args:
        ticker: 如 "688016.SS"
        limit: 最多返回条数

    Returns:
        格式化的 markdown 字符串。
    """
    code = _get_stock_code_only(ticker)
    if not _is_valid_stock_code(code):
        return f"<东方财富研报: {ticker} 非有效A股代码>"
    try:
        import akshare as ak
    except ImportError:
        return f"<东方财富研报: AKShare 未安装>"

    try:
        df = ak.stock_research_report_em(symbol=code)
    except Exception as e:
        logger.warning("东方财富研报 %s 失败: %s", ticker, e)
        return f"<东方财富研报: {ticker} 获取失败>"

    if df is None or df.empty:
        return f"<东方财富研报: {ticker} 暂无>"

    lines = []
    for _, row in df.head(limit).iterrows():
        name = row.get("报告名称", "")
        rating = row.get("东财评级", "")
        org = row.get("机构", "")
        date = row.get("日期", "")
        profit = row.get("2026-盈利预测-收益", "")
        pe = row.get("2026-盈利预测-市盈率", "")
        parts = [f"[{date} · {org}]"]
        if rating:
            parts.append(f"评级: {rating}")
        parts.append(f"  {name}")
        if profit:
            parts.append(f"  | 预测EPS: {profit} | 预测PE: {pe}")
        lines.append("".join(parts))

    summary = (
        f"## {ticker} 东方财富券商研报 (最近 {min(limit, len(df))} 篇):\n\n"
        + "\n".join(lines)
    )
    return summary


# ============================================================
#  3. 同花顺 — 全球财经快讯
# ============================================================

def fetch_ths_news(limit: int = 20) -> str:
    """同花顺 (10jqka.com.cn) 全球财经快讯。

    实时滚动快讯，覆盖 A 股、宏观、行业、国际等。市场通用 news，不针对单只股票。

    Args:
        limit: 最多返回条数

    Returns:
        格式化的 markdown 字符串。
    """
    try:
        import akshare as ak
    except ImportError:
        return f"<同花顺快讯: AKShare 未安装>"

    try:
        df = ak.stock_info_global_ths()
    except Exception as e:
        logger.warning("同花顺快讯失败: %s", e)
        return f"<同花顺快讯: 获取失败>"

    if df is None or df.empty:
        return f"<同花顺快讯: 暂无数据>"

    return _format_simple_news(
        df, "同花顺财经快讯", limit,
        title_col="标题", content_col="内容", time_col="发布时间",
    )


# ============================================================
#  4. 新浪财经 — 实时新闻
# ============================================================

def fetch_sina_news(limit: int = 20) -> str:
    """新浪财经实时新闻。

    滚动新闻流，包含宏观、行业、公司等各类财经资讯。

    Args:
        limit: 最多返回条数

    Returns:
        格式化的 markdown 字符串。
    """
    try:
        import akshare as ak
    except ImportError:
        return f"<新浪财经: AKShare 未安装>"

    try:
        df = ak.stock_info_global_sina()
    except Exception as e:
        logger.warning("新浪财经失败: %s", e)
        return f"<新浪财经: 获取失败>"

    if df is None or df.empty:
        return f"<新浪财经: 暂无数据>"

    return _format_simple_news(
        df, "新浪财经新闻", limit,
        title_col="内容", content_col=None, time_col="时间",
    )


# ============================================================
#  格式化工具
# ============================================================

def _format_news_with_sentiment(
    df,
    ticker: str,
    source_name: str,
    limit: int,
    title_col: str,
    content_col: str,
    time_col: str,
    source_col: str,
) -> str:
    """格式化带情感标注的新闻列表。"""
    bullish_kw = ["买入", "增持", "看涨", "利好", "超预期", "业绩增长", "突破", "增长"]
    bearish_kw = ["卖出", "减持", "看跌", "利空", "亏损", "下滑", "跌停", "下降", "预警"]

    lines = []
    bull_count = bear_count = neutral_count = 0

    for _, row in df.head(limit).iterrows():
        title = str(row.get(title_col, ""))
        title_lower = title.lower()
        if any(kw in title_lower for kw in bullish_kw):
            bull_count += 1
            tag = "[偏多]"
        elif any(kw in title_lower for kw in bearish_kw):
            bear_count += 1
            tag = "[偏空]"
        else:
            neutral_count += 1
            tag = ""

        pub_time = row.get(time_col, "")
        source = row.get(source_col, "") if source_col else ""
        prefix = f"[{pub_time} · {source}] " if source else f"[{pub_time}] "
        line = f"{prefix}{tag} {title}"

        if content_col:
            content = str(row.get(content_col, ""))[:200]
            if content and content not in ("nan", ""):
                line += f"\n    {content}"
        lines.append(line)

    total = bull_count + bear_count + neutral_count
    bull_pct = round(100 * bull_count / total) if total else 0
    bear_pct = round(100 * bear_count / total) if total else 0
    header = (
        f"## {ticker} {source_name} (最近 {min(limit, len(df))} 条)\n"
        f"情感统计: 偏多 {bull_count}({bull_pct}%) · "
        f"偏空 {bear_count}({bear_pct}%) · "
        f"中性 {neutral_count}\n"
    )
    return header + "\n" + "\n\n".join(lines)


def _format_simple_news(
    df,
    source_name: str,
    limit: int,
    title_col: str,
    content_col: str | None,
    time_col: str,
) -> str:
    """格式化通用新闻列表（无情感标注）。"""
    lines = []
    for _, row in df.head(limit).iterrows():
        title = str(row.get(title_col, ""))
        if content_col:
            content = str(row.get(content_col, ""))[:200]
            if content and content not in ("nan", ""):
                title += f": {content}"
        pub_time = row.get(time_col, "")
        line = f"[{pub_time}] {title}"
        lines.append(line)

    header = f"## {source_name} (最近 {min(limit, len(df))} 条):\n"
    return header + "\n" + "\n".join(lines)


# ============================================================
#  聚合函数
# ============================================================

def fetch_all_cn_news(ticker: str, limit_per_source: int = 8) -> dict:
    """一次性获取所有国内新闻源数据。

    Args:
        ticker: 如 "688016.SS"
        limit_per_source: 每个来源最多返回条数

    Returns:
        {
            "eastmoney_news": str,        # 东方财富个股新闻
            "eastmoney_reports": str,     # 东方财富研报
            "ths_news": str,              # 同花顺快讯
            "sina_news": str,             # 新浪财经
        }
    """
    return {
        "eastmoney_news": fetch_eastmoney_news(ticker, limit_per_source),
        "eastmoney_reports": fetch_eastmoney_reports(ticker, limit_per_source),
        "ths_news": fetch_ths_news(limit_per_source * 2),
        "sina_news": fetch_sina_news(limit_per_source * 2),
    }


# ============================================================
#  interface.py 兼容适配器（用于 VENDOR_METHODS 注册）
# ============================================================

def get_news_cn(
    ticker: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> str:
    """个股新闻聚合（AKShare 供应商接口兼容格式）。

    聚合东方财富个股新闻 + 个股研报，格式化为统一输出。
    start_date/end_date 保留兼容性接口，实际由 AKShare API 返回最新数据。

    Args:
        ticker: 如 "688016.SS"
        start_date: 保留（AKShare 返回最新新闻）
        end_date: 保留（AKShare 返回最新新闻）

    Returns:
        格式化的 markdown 字符串
    """
    em_news = fetch_eastmoney_news(ticker, limit=15)
    em_reports = fetch_eastmoney_reports(ticker, limit=10)

    parts = [em_news]
    if em_reports and "暂无" not in em_reports:
        parts.append("\n---\n")
        parts.append(em_reports)

    return "\n".join(parts)


def get_global_news_cn(
    curr_date: str | None = None,
    look_back_days: int | None = None,
    limit: int | None = None,
) -> str:
    """宏观/全球新闻聚合（AKShare 供应商接口兼容格式）。

    聚合同花顺财经快讯 + 新浪财经新闻。
    look_back_days 保留兼容性接口，实际由 AKShare API 返回最新快讯。

    Args:
        curr_date: 当前日期（保留兼容性）
        look_back_days: 保留（AKShare 返回最新快讯）
        limit: 每个来源返回的条数，默认 10

    Returns:
        格式化的 markdown 字符串
    """
    n = limit or 10
    ths_block = fetch_ths_news(limit=n)
    sina_block = fetch_sina_news(limit=n)

    parts = [ths_block]
    if sina_block and "暂无" not in sina_block:
        parts.append("\n---\n")
        parts.append(sina_block)

    return "\n".join(parts)
