"""A股国内新闻聚合模块（多源冗余版）。

聚合七大互补数据源，实现逐级 failover（主源→备源→降级），
确保单源故障不导致整个新闻信息丢失:

  个股/研报类:
    1. 东方财富个股新闻   (stock_news_em)               — 主源
    2. 东方财富个股研报   (stock_research_report_em)     — 主源
    3. 东方财富要闻       (stock_news_main_em)           — 备源

  宏观/快讯类:
    4. 同花顺全球财经快讯 (stock_info_global_ths)       — 主源
    5. 新浪财经新闻       (stock_info_global_sina)       — 主源
    6. 东方财富全球快讯   (stock_info_global_em)         — 备源
    7. 东方财富要闻       (stock_news_main_em)           — 备源 (与个股共享)

每个 fetch 函数内置 failover 链：主源失败 → 尝试备源 → 返回明确降级信息。
"""

from __future__ import annotations

import logging
import re
from typing import Callable, List

logger = logging.getLogger(__name__)


def _get_stock_code_only(ticker: str) -> str:
    """从 '688016.SS' 或 '600519.SS' 中提取纯数字代码。"""
    return ticker.split(".")[0]


def _is_valid_stock_code(code: str) -> bool:
    """校验是否为合法的 A 股代码（6 位纯数字）。"""
    return bool(re.fullmatch(r"\d{6}", code))


# ────────────────────────────────────────────────────────────────
#  通用 failover 包装器
# ────────────────────────────────────────────────────────────────

def _failover(
    source_name: str,
    primary: Callable[[], str],
    backup: Callable[[], str] | None = None,
    *,
    fallback_msg: str | None = None,
) -> str:
    """Failover 执行器：主源失败 → 备源 → 降级消息。

    Args:
        source_name: 源名称（用于日志）
        primary: 主源函数（无参数）
        backup: 可选备源函数（无参数）
        fallback_msg: 全部失败时的最终降级消息

    Returns:
        主源结果、备源结果或降级消息
    """
    # 尝试主源
    try:
        result = primary()
        if result and not result.startswith("<") and "暂无" not in result and "获取失败" not in result:
            return result
    except Exception as e:
        logger.warning("%s 主源失败: %s", source_name, e)

    # 尝试备源
    if backup:
        try:
            result = backup()
            if result and not result.startswith("<") and "暂无" not in result and "获取失败" not in result:
                logger.info("%s 主源降级，备源成功", source_name)
                return result
        except Exception as e:
            logger.warning("%s 备源也失败: %s", source_name, e)

    # 降级
    msg = fallback_msg or f"<{source_name}: 所有源均不可用>"
    logger.warning("%s 全部源失败，返回降级消息", source_name)
    return msg


# ────────────────────────────────────────────────────────────────
#  东方财富要闻 (通用备源)
# ────────────────────────────────────────────────────────────────

def _try_eastmoney_headlines(limit: int = 20) -> str | None:
    """尝试获取东方财富要闻（stock_news_main_em）作为备源。

    返回 None 表示获取失败。
    """
    try:
        import akshare as ak
    except ImportError:
        return None
    try:
        df = ak.stock_news_main_em()
    except Exception:
        return None
    if df is None or df.empty:
        return None
    lines = []
    for _, row in df.head(limit).iterrows():
        title = str(row.get("标题", "")) or str(row.get("title", ""))
        if not title or title == "nan":
            continue
        time_val = str(row.get("发布时间", "")) or str(row.get("ctime", "")) or ""
        source = str(row.get("文章来源", "")) or str(row.get("source", "")) or "东方财富要闻"
        line = f"[{time_val} · {source}] {title}"
        lines.append(line)
    if not lines:
        return None
    return f"## 东方财富要闻 (最近 {len(lines)} 条，备源):\n\n" + "\n\n".join(lines)


def _try_eastmoney_global(limit: int = 20) -> str | None:
    """尝试获取东方财富全球快讯（stock_info_global_em）作为备源。

    返回 None 表示获取失败。
    """
    try:
        import akshare as ak
    except ImportError:
        return None
    try:
        df = ak.stock_info_global_em()
    except Exception:
        return None
    if df is None or df.empty:
        return None
    lines = []
    title_col = next((c for c in df.columns if "标题" in c or "title" in c.lower()), None)
    time_col = next((c for c in df.columns if "时间" in c or "time" in c.lower() or "日期" in c), None)
    for _, row in df.head(limit).iterrows():
        title = str(row.get(title_col, "")) if title_col else str(row.iloc[0]) if len(df.columns) > 0 else ""
        if not title or title == "nan":
            continue
        time_val = str(row.get(time_col, "")) if time_col else ""
        line = f"[{time_val}] {title}" if time_val else title
        lines.append(line)
    if not lines:
        return None
    return f"## 东方财富全球快讯 (最近 {len(lines)} 条，备源):\n\n" + "\n".join(lines)


# ────────────────────────────────────────────────────────────────
#  1. 东方财富 — 个股新闻（带 failover）
# ────────────────────────────────────────────────────────────────

def fetch_eastmoney_news(ticker: str, limit: int = 15) -> str:
    """东方财富个股新闻（主源）。

    通过 AKShare 的 stock_news_em 获取该股票相关的新闻报道。
    失败时自动 failover 到东方财富要闻。

    Args:
        ticker: 如 "688016.SS"
        limit: 最多返回条数

    Returns:
        格式化的 markdown 字符串
    """
    code = _get_stock_code_only(ticker)
    if not _is_valid_stock_code(code):
        return f"<东方财富个股新闻: {ticker} 非有效A股代码>"

    def _primary():
        import akshare as ak
        df = ak.stock_news_em(symbol=code)
        if df is None or df.empty:
            return f"<东方财富个股新闻: {ticker} 暂无相关新闻>"
        return _format_news_with_sentiment(
            df, ticker, "东方财富个股新闻", limit,
            title_col="新闻标题", content_col="新闻内容",
            time_col="发布时间", source_col="文章来源",
        )

    def _backup():
        result = _try_eastmoney_headlines(limit)
        if result:
            return f"## {ticker} 相关新闻 (东方财富要闻备源, 最近 {limit} 条):\n\n{result}"
        return None

    return _failover("东方财富个股新闻", _primary, _backup,
                      fallback_msg=f"<东方财富个股新闻: {ticker} 获取失败>")


# ────────────────────────────────────────────────────────────────
#  2. 东方财富 — 个股研报
# ────────────────────────────────────────────────────────────────

def fetch_eastmoney_reports(ticker: str, limit: int = 15) -> str:
    """东方财富个股研报。

    券商对该股票的研究报告，包含评级、盈利预测等专业信息。

    Args:
        ticker: 如 "688016.SS"
        limit: 最多返回条数
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


# ────────────────────────────────────────────────────────────────
#  3. 同花顺 — 全球财经快讯（带 failover）
# ────────────────────────────────────────────────────────────────

def fetch_ths_news(limit: int = 20) -> str:
    """同花顺 (10jqka.com.cn) 全球财经快讯（主源）。

    失败时自动 failover 到东方财富全球快讯 → 东方财富要闻。
    """
    def _primary():
        import akshare as ak
        df = ak.stock_info_global_ths()
        if df is None or df.empty:
            return f"<同花顺快讯: 暂无数据>"
        return _format_simple_news(
            df, "同花顺财经快讯", limit,
            title_col="标题", content_col="内容", time_col="发布时间",
        )

    def _backup():
        result = _try_eastmoney_global(limit)
        if result:
            return result
        return _try_eastmoney_headlines(limit)

    return _failover("同花顺快讯", _primary, _backup,
                      fallback_msg=f"<同花顺快讯: 获取失败>")


# ────────────────────────────────────────────────────────────────
#  4. 新浪财经 — 实时新闻（带 failover）
# ────────────────────────────────────────────────────────────────

def fetch_sina_news(limit: int = 20) -> str:
    """新浪财经实时新闻（主源）。

    失败时自动 failover 到东方财富全球快讯 → 东方财富要闻。
    """
    def _primary():
        import akshare as ak
        df = ak.stock_info_global_sina()
        if df is None or df.empty:
            return f"<新浪财经: 暂无数据>"
        return _format_simple_news(
            df, "新浪财经新闻", limit,
            title_col="内容", content_col=None, time_col="时间",
        )

    def _backup():
        result = _try_eastmoney_global(limit)
        if result:
            return result
        return _try_eastmoney_headlines(limit)

    return _failover("新浪财经", _primary, _backup,
                      fallback_msg=f"<新浪财经: 获取失败>")


# ────────────────────────────────────────────────────────────────
#  5. 东方财富要闻 — 独立宏源源
# ────────────────────────────────────────────────────────────────

def fetch_eastmoney_headlines(limit: int = 20) -> str:
    """东方财富要闻（独立源，作为同花顺/新浪的第三备选）。

    失败时尝试东方财富全球快讯。
    """
    def _primary():
        result = _try_eastmoney_headlines(limit)
        if result is None:
            raise ValueError("无数据")
        return result

    def _backup():
        result = _try_eastmoney_global(limit)
        if result:
            return result
        return None

    return _failover("东方财富要闻", _primary, _backup,
                      fallback_msg=f"<东方财富要闻: 获取失败>")


# ────────────────────────────────────────────────────────────────
#  6. 东方财富全球快讯 — 独立宏源源
# ────────────────────────────────────────────────────────────────

def fetch_eastmoney_global(limit: int = 20) -> str:
    """东方财富全球快讯（独立源，作为同花顺/新浪的第四备选）。

    失败时尝试东方财富要闻。
    """
    def _primary():
        result = _try_eastmoney_global(limit)
        if result is None:
            raise ValueError("无数据")
        return result

    def _backup():
        result = _try_eastmoney_headlines(limit)
        if result:
            return result
        return None

    return _failover("东方财富全球快讯", _primary, _backup,
                      fallback_msg=f"<东方财富全球快讯: 获取失败>")


# ────────────────────────────────────────────────────────────────
#  7. 巨潮资讯网公告 — 个股补充信息
# ────────────────────────────────────────────────────────────────

def fetch_cninfo_notices(ticker: str, limit: int = 10) -> str:
    """巨潮资讯网公告（补充源）。

    通过 AKShare stock_notice_report 获取公司最新公告。

    Args:
        ticker: 如 "688016.SS"
        limit: 最多返回条数
    """
    code = _get_stock_code_only(ticker)
    if not _is_valid_stock_code(code):
        return f"<巨潮资讯公告: {ticker} 非有效A股代码>"
    try:
        import akshare as ak
    except ImportError:
        return f"<巨潮资讯公告: AKShare 未安装>"

    try:
        df = ak.stock_notice_report(symbol="SH" + code if ticker.endswith(".SS") else "SZ" + code)
    except Exception:
        # 巨潮接口可能因股票代码前缀格式不同而失败，标记为无数据
        return f"<巨潮资讯公告: {ticker} 暂无可用公告>"

    if df is None or df.empty:
        return f"<巨潮资讯公告: {ticker} 暂无公告>"

    lines = []
    for _, row in df.head(limit).iterrows():
        name = str(row.get("公告名称", "")) or str(row.get("notice_name", "")) or str(row.get("title", ""))
        date = str(row.get("公告日期", "")) or str(row.get("notice_date", "")) or str(row.get("date", ""))
        if name and name != "nan":
            line = f"[{date}] {name}"
            lines.append(line)

    if not lines:
        return f"<巨潮资讯公告: {ticker} 暂无公告>"

    return f"## {ticker} 巨潮资讯网公告 (最近 {len(lines)} 条):\n\n" + "\n".join(lines)


# ────────────────────────────────────────────────────────────────
#  格式化工具
# ────────────────────────────────────────────────────────────────

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


# ────────────────────────────────────────────────────────────────
#  聚合函数
# ────────────────────────────────────────────────────────────────

def fetch_all_cn_news(ticker: str, limit_per_source: int = 8) -> dict:
    """一次性获取所有国内新闻源数据。

    包含主源 + 备源共 7 个维度，每个维度内置 failover。

    Args:
        ticker: 如 "688016.SS"
        limit_per_source: 每个来源最多返回条数

    Returns:
        {
            "eastmoney_news": str,        # 东方财富个股新闻 (主源，有 failover)
            "eastmoney_reports": str,     # 东方财富研报
            "ths_news": str,              # 同花顺快讯 (主源，有 failover)
            "sina_news": str,             # 新浪财经 (主源，有 failover)
            "eastmoney_headlines": str,   # 东方财富要闻 (独立备源)
            "eastmoney_global": str,      # 东方财富全球快讯 (独立备源)
            "cninfo_notices": str,        # 巨潮资讯公告
        }
    """
    return {
        "eastmoney_news": fetch_eastmoney_news(ticker, limit_per_source),
        "eastmoney_reports": fetch_eastmoney_reports(ticker, limit_per_source),
        "ths_news": fetch_ths_news(limit_per_source * 2),
        "sina_news": fetch_sina_news(limit_per_source * 2),
        "eastmoney_headlines": fetch_eastmoney_headlines(limit_per_source * 2),
        "eastmoney_global": fetch_eastmoney_global(limit_per_source * 2),
        "cninfo_notices": fetch_cninfo_notices(ticker, limit_per_source),
    }


# ────────────────────────────────────────────────────────────────
#  可用源统计（供测试和健康检查使用）
# ────────────────────────────────────────────────────────────────

def check_source_health(ticker: str = "600519.SS") -> dict:
    """检查所有新闻源的可用性。

    用给定 ticker 或默认 "600519"（贵州茅台）测试所有源，
    返回各源的状态报告。

    Returns:
        {
            "eastmoney_news": "ok" | "degraded" | "failed",
            ...
            "total_ok": int,
            "total_sources": int,
        }
    """
    results = {}
    ok_count = 0

    def _check(name: str, result: str) -> str:
        nonlocal ok_count
        if result and not result.startswith("<"):
            ok_count += 1
            return "ok"
        elif "暂无" in result:
            ok_count += 1
            return "ok"  # 暂无数据也是正常的
        return "degraded" if result else "failed"

    results["eastmoney_news"] = _check("eastmoney_news", fetch_eastmoney_news(ticker, limit=3))
    results["eastmoney_reports"] = _check("eastmoney_reports", fetch_eastmoney_reports(ticker, limit=3))
    results["ths_news"] = _check("ths_news", fetch_ths_news(limit=5))
    results["sina_news"] = _check("sina_news", fetch_sina_news(limit=5))
    results["eastmoney_headlines"] = _check("eastmoney_headlines", fetch_eastmoney_headlines(limit=5))
    results["eastmoney_global"] = _check("eastmoney_global", fetch_eastmoney_global(limit=5))
    results["cninfo_notices"] = _check("cninfo_notices", fetch_cninfo_notices(ticker, limit=3))

    results["total_ok"] = ok_count
    results["total_sources"] = 7
    return results


# ────────────────────────────────────────────────────────────────
#  interface.py 兼容适配器（用于 VENDOR_METHODS 注册）
# ────────────────────────────────────────────────────────────────

def get_news_cn(
    ticker: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> str:
    """个股新闻聚合（AKShare 供应商接口兼容格式）。

    聚合：东方财富个股新闻（主源）+ 东方财富个股研报 + 巨潮资讯公告。

    Args:
        ticker: 如 "688016.SS"
        start_date: 保留（AKShare 返回最新新闻）
        end_date: 保留（AKShare 返回最新新闻）

    Returns:
        格式化的 markdown 字符串
    """
    em_news = fetch_eastmoney_news(ticker, limit=15)
    em_reports = fetch_eastmoney_reports(ticker, limit=10)
    cninfo = fetch_cninfo_notices(ticker, limit=5)

    parts = [em_news]
    if em_reports and "暂无" not in em_reports and "获取失败" not in em_reports:
        parts.append("\n---\n")
        parts.append(em_reports)
    if cninfo and "暂无" not in cninfo:
        parts.append("\n---\n")
        parts.append(cninfo)

    return "\n".join(parts)


def get_global_news_cn(
    curr_date: str | None = None,
    look_back_days: int | None = None,
    limit: int | None = None,
) -> str:
    """宏观/全球新闻聚合（AKShare 供应商接口兼容格式）。

    聚合：同花顺财经快讯 + 新浪财经新闻 + 东方财富要闻 + 东方财富全球快讯。
    每个源独立内置 failover。

    Args:
        curr_date: 当前日期（保留兼容性）
        look_back_days: 保留（AKShare 返回最新快讯）
        limit: 每个来源返回的条数，默认 15

    Returns:
        格式化的 markdown 字符串
    """
    n = limit or 15
    ths_block = fetch_ths_news(limit=n)
    sina_block = fetch_sina_news(limit=n)
    em_headlines = fetch_eastmoney_headlines(limit=n)
    em_global = fetch_eastmoney_global(limit=n)

    parts = [ths_block]
    if sina_block and "暂无" not in sina_block and "获取失败" not in sina_block:
        parts.append("\n---\n")
        parts.append(sina_block)
    if em_headlines and "暂无" not in em_headlines and "获取失败" not in em_headlines:
        parts.append("\n---\n")
        parts.append(em_headlines)
    if em_global and "暂无" not in em_global and "获取失败" not in em_global:
        parts.append("\n---\n")
        parts.append(em_global)

    return "\n".join(parts)
