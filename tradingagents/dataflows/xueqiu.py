"""A股社交媒体情绪数据获取。

替代原来的 Reddit (reddit.py) 和 StockTwits (stocktwits.py):
  - 东方财富个股新闻: akshare.stock_news_em() → 获取该股票相关新闻
  - 东方财富股吧: 页面为 JS 动态渲染，传统爬取不可用
  - 机构评级: akshare.stock_comment_detail_zlkp_jgcyd_em() → 机构关注度

雪球 API 有严格反爬机制，改用 AKShare 封装的东方财富接口。
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def _get_stock_code_only(ticker: str) -> str:
    """从 ticker 中提取数字代码。"""
    return ticker.split(".")[0]


def fetch_stock_news(ticker: str, limit: int = 20) -> str:
    """获取东方财富个股新闻（通过 AKShare）。

    Args:
        ticker: 股票代码，如 "688016.SS"
        limit: 返回新闻条数上限

    Returns:
        格式化的新闻文本
    """
    code = _get_stock_code_only(ticker)
    try:
        import akshare as ak
    except ImportError:
        return f"<东方财富个股新闻: AKShare 未安装>"

    try:
        df = ak.stock_news_em(symbol=code)
        if df is None or df.empty:
            return f"<东方财富个股新闻: {ticker} 暂无相关新闻>"

        # 简易情感统计
        bullish_kw = ["买入", "增持", "看涨", "利好", "超预期", "业绩增长", "突破"]
        bearish_kw = ["卖出", "减持", "看跌", "利空", "亏损", "下滑", "跌停"]
        lines = []
        bull_count = bear_count = neutral_count = 0

        for _, row in df.head(limit).iterrows():
            title = str(row.get("新闻标题", ""))
            title_lower = title.lower()
            if any(kw in title_lower for kw in bullish_kw):
                bull_count += 1
                sentiment = "[偏多]"
            elif any(kw in title_lower for kw in bearish_kw):
                bear_count += 1
                sentiment = "[偏空]"
            else:
                neutral_count += 1
                sentiment = ""

            content = str(row.get("新闻内容", ""))[:200]
            pub_time = row.get("发布时间", "")
            source = row.get("文章来源", "")
            line = f"[{pub_time} · {source}] {sentiment} {title}"
            if content:
                line += f"\n    {content}"
            lines.append(line)

        total = bull_count + bear_count + neutral_count
        bull_pct = round(100 * bull_count / total) if total else 0
        bear_pct = round(100 * bear_count / total) if total else 0
        summary = (
            f"## {ticker} 东方财富个股新闻 (最近 {min(limit, len(df))} 条)\n"
            f"新闻情感统计: 偏多 {bull_count}({bull_pct}%) · "
            f"偏空 {bear_count}({bear_pct}%) · "
            f"中性 {neutral_count}\n"
        )
        return summary + "\n" + "\n\n".join(lines)

    except Exception as e:
        logger.warning("东方财富个股新闻获取 %s 失败: %s", ticker, e)
        return f"<东方财富个股新闻: {ticker} 获取失败 ({type(e).__name__})>"


def fetch_guba_posts(ticker: str, limit: int = 20, timeout: float = 10.0) -> str:
    """获取该股票的机构关注/评级数据（东方财富）。

    股吧页面为 JS 动态渲染，传统 HTTP 爬取无法获取帖子列表。
    此函数通过 AKShare 获取机构关注度和评级数据作为替代的散户情绪代理。

    Args:
        ticker: 股票代码，如 "688016.SS"
        limit: 保留参数（API 返回固定数量）
        timeout: 保留参数

    Returns:
        格式化的评级/关注度文本
    """
    code = _get_stock_code_only(ticker)
    try:
        import akshare as ak
    except ImportError:
        return f"<东方财富机构评级: AKShare 未安装>"

    try:
        df = ak.stock_comment_detail_zlkp_jgcyd_em(symbol=code)
        if df is None or df.empty:
            return f"<东方财富机构评级: {ticker} 暂无数据>"

        # 列名: 交易日, 机构参与度
        lines = []
        for _, row in df.head(limit).iterrows():
            date = row.get("交易日", "")
            participation = row.get("机构参与度", "")
            line = f"{date}: 机构参与度 {participation}%"
            lines.append(line)

        if lines:
            return (
                f"## {ticker} 东方财富机构关注度 (最近 {len(lines)} 条):\n\n"
                "机构参与度反映机构资金在该股票交易中的占比，越高说明机构关注度越高。\n\n"
                + "\n".join(f"- {l}" for l in lines)
            )
        return f"<东方财富机构评级: {ticker} 暂无数据>"

    except Exception as e:
        logger.warning("东方财富机构评级获取 %s 失败: %s", ticker, e)
        # 尝试用 stock_news_em 作为后备
        try:
            return fetch_stock_news(ticker, limit // 2)
        except Exception:
            return f"<东方财富机构评级: {ticker} 数据获取失败 ({type(e).__name__})>"


def fetch_xueqiu_posts(ticker: str, limit: int = 20, timeout: float = 10.0) -> str:
    """获取雪球帖子（已废弃 — 雪球 API 反爬严格）。

    此函数作为兼容性接口保留，委托给 fetch_stock_news。
    """
    return fetch_stock_news(ticker, limit)
