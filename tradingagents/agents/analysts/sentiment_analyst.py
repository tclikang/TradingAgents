"""Sentiment analyst — 多源情绪分析（A股适配版）。

聚合四个互补数据来源，覆盖机构研报、个股新闻、实时快讯：
  1. 东方财富个股新闻 (stock_news_em)           — 该股票直接相关新闻
  2. 东方财富券商研报 (stock_research_report_em) — 机构评级+盈利预测
  3. 同花顺全球快讯 (stock_info_global_ths)      — 实时财经快讯
  4. 新浪财经新闻   (stock_info_global_sina)      — 实时滚动新闻

数据在 LLM 调用前预取并注入 prompt，无需 Tool Call。
"""

from datetime import datetime, timedelta

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.schemas import SentimentReport, render_sentiment_report
from tradingagents.agents.utils.agent_utils import (
    get_instrument_context_from_state,
    get_language_instruction,
)
from tradingagents.agents.utils.structured import (
    NO_EXTERNAL_TOOLS,
    bind_structured,
    invoke_structured_or_freetext,
)
from tradingagents.dataflows.cn_news import (
    fetch_all_cn_news,
    fetch_eastmoney_news,
    fetch_eastmoney_reports,
    fetch_sina_news,
    fetch_ths_news,
)


def _seven_days_back(trade_date: str) -> str:
    return (datetime.strptime(trade_date, "%Y-%m-%d") - timedelta(days=7)).strftime("%Y-%m-%d")


def create_sentiment_analyst(llm):
    """Create a sentiment analyst node for the trading graph.

    Pre-fetches 4 domestic Chinese news sources, injects them into the
    prompt as structured blocks, and produces a deterministic sentiment
    report via structured output (with a free-text fallback).
    """
    structured_llm = bind_structured(llm, SentimentReport, "Sentiment Analyst")

    def sentiment_analyst_node(state):
        ticker = state["company_of_interest"]
        end_date = state["trade_date"]
        start_date = _seven_days_back(end_date)
        instrument_context = get_instrument_context_from_state(state)

        # 预取所有数据源
        em_news = fetch_eastmoney_news(ticker, limit=15)
        em_reports = fetch_eastmoney_reports(ticker, limit=10)
        ths_block = fetch_ths_news(limit=15)
        sina_block = fetch_sina_news(limit=15)

        system_message = _build_system_message(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            em_news=em_news,
            em_reports=em_reports,
            ths_block=ths_block,
            sina_block=sina_block,
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
                    " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
                    # No tool-calling here: the data is pre-fetched into the
                    # prompt, so tool-range wording would only invite a
                    # hallucinated tool call (#1130).
                    " Today's date is {current_date}; treat it as 'now' for all analysis. {instrument_context}"
                    " " + NO_EXTERNAL_TOOLS +
                    "\n{system_message}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(current_date=end_date)
        prompt = prompt.partial(instrument_context=instrument_context)

        formatted_messages = prompt.format_messages(messages=state["messages"])

        report_text = invoke_structured_or_freetext(
            structured_llm,
            llm,
            formatted_messages,
            render_sentiment_report,
            "Sentiment Analyst",
        )

        return {
            "messages": [AIMessage(content=report_text)],
            "sentiment_report": report_text,
        }

    return sentiment_analyst_node


# ---------------------------------------------------------------------------
# Legacy backwards-compatibility shim for xueqiu.py
# ---------------------------------------------------------------------------
def _make_compat_fetch_xueqiu():
    """Return a callable for old code that still imports fetch_xueqiu_posts."""
    from tradingagents.dataflows.cn_news import fetch_eastmoney_news as _em
    return lambda ticker, limit=20, timeout=10.0: _em(ticker, limit)


def _make_compat_fetch_guba():
    """Return a callable for old code that still imports fetch_guba_posts."""
    from tradingagents.dataflows.cn_news import fetch_eastmoney_reports as _er
    return lambda ticker, limit=20, timeout=10.0: _er(ticker, limit)


def _build_system_message(
    *,
    ticker: str,
    start_date: str,
    end_date: str,
    em_news: str,
    em_reports: str,
    ths_block: str,
    sina_block: str,
) -> str:
    """组装情绪分析师的 system prompt（A股版，4 数据源）。"""
    return f"""你是一位 A 股市场情绪分析师。请对 {ticker} 在 {start_date} 至 {end_date} 期间的市场情绪做出综合分析报告。以下四个互补数据源已为你预取完毕：

## 数据源

### 1. 东方财富个股新闻
该股票直接相关的 A 股新闻，每条新闻带 [偏多]/[偏空] 情感标签。

<start_of_em_news>
{em_news}
<end_of_em_news>

### 2. 东方财富券商研报
国内券商对该股票的研究报告，包含评级（买入/增持/减持/卖出）和盈利预测数据（EPS/PE），反映专业机构观点。

<start_of_em_reports>
{em_reports}
<end_of_em_reports>

### 3. 同花顺财经快讯（10jqka.com.cn）
实时滚动财经快讯，覆盖 A 股、宏观、行业、国际等最新消息。

<start_of_ths>
{ths_block}
<end_of_ths>

### 4. 新浪财经新闻
新浪实时财经新闻流，包含政策、公司公告、行业动态等。

<start_of_sina>
{sina_block}
<end_of_sina>

## 分析方法（最佳实践）

1. **券商研报是最专业的信号来源。** 评级为"买入"/"增持"且多家机构一致看多 = 机构共识强；出现"减持"/"卖出"评级需高度警惕。

2. **新闻偏多/偏空比辅助情绪判断。** 偏多新闻占比 >70% 为积极信号；偏空为主需警惕。

3. **同花顺/新浪快讯提供实时催化剂。** 注意是否有行业政策、公司公告、大单交易等突发事件。

4. **盈利预测是关键硬数据。** 研报中的 EPS/PE 预测如果持续上调 = 基本面改善信号；持续下调 = 基本面恶化。

5. **区分观点与事件。** 新闻标题是事件，研报评级是专业观点，两者权重不同。

6. **坦诚数据局限性。** 当某个来源只返回少量内容或"<不可用>"占位符时，情绪判断力度要打折扣。

7. **历史情绪不预测未来。** 将你的结论作为信号供交易员结合基本面和技术面综合参考。

## 输出字段

- **overall_band**: 看涨 / 温和看涨 / 中性 / 分歧 / 温和看跌 / 看跌。当来源指向明显不同方向时用"分歧"。
- **overall_score**: 0（极度看跌）到 10（极度看涨）；5 为中性。
- **confidence**: 低 / 中 / 高，基于数据质量和样本量。
- **narrative**: 逐源详细分析、跨源背离、主导情绪主题、催化剂和风险、关键情绪信号汇总表。

{get_language_instruction()}"""


# ---------------------------------------------------------------------------
# Backwards-compatibility shim for social_media_analyst
# ---------------------------------------------------------------------------
def create_social_media_analyst(llm):
    """Deprecated alias for :func:`create_sentiment_analyst`. """
    import warnings
    warnings.warn(
        "create_social_media_analyst is deprecated. Use create_sentiment_analyst instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return create_sentiment_analyst(llm)
