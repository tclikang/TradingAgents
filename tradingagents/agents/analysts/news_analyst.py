from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.utils.agent_utils import (
    get_company_announcements,
    get_global_news,
    get_instrument_context_from_state,
    get_language_instruction,
    get_macro_indicators,
    get_market_news,
    get_news,
    get_prediction_markets,
    get_research_reports,
    get_sector,
)


def create_news_analyst(llm):
    def news_analyst_node(state):
        current_date = state["trade_date"]
        asset_type = state.get("asset_type", "stock")
        asset_label = "company" if asset_type == "stock" else "asset"
        instrument_context = get_instrument_context_from_state(state)

        tools = [
            get_news,
            get_global_news,
            get_market_news,
            get_macro_indicators,
            get_prediction_markets,
        ]

        # For A-stocks (China market), auto-include China-specific tools
        ticker = state.get("ticker", "")
        if ticker and (ticker.endswith(".SS") or ticker.endswith(".SZ") or ticker.endswith(".BJ")):
            tools.extend([
                get_company_announcements,
                get_research_reports,
                get_sector,
            ])

        system_message = (
            f"You are a news researcher tasked with analyzing recent news and trends over the past week. Please write a comprehensive report of the current state of the world that is relevant for trading and macroeconomics."
            " Use the available tools:\n"
            f"  - get_news(query, start_date, end_date): {asset_label}-specific or targeted news searches\n"
            "  - get_global_news(curr_date, look_back_days, limit): broader macroeconomic news\n"
            "  - get_market_news(curr_date, look_back_days, limit): multi-source China market news from Eastmoney and Sina (REQUIRED for Chinese stocks)\n"
            f"  - get_macro_indicators(indicator, curr_date, look_back_days): FRED macro data (CPI, unemployment, fed funds rate, 10y treasury, yield curve)\n"
            "  - get_prediction_markets(topic, limit): live market-implied probabilities of forward-looking events\n"
        )
        # Conditionally document China-specific tools
        has_china = any(t.name == "get_company_announcements" for t in tools)
        if has_china:
            system_message += (
                "  - get_company_announcements(ticker, curr_date): REQUIRED - fetch official company announcements from the exchange disclosure platform (board resolutions, material events, filings)\n"
                "  - get_research_reports(ticker): REQUIRED - fetch analyst research reports with ratings, target prices and summaries from Eastmoney\n"
                "  - get_sector(symbol): REQUIRED - fetch sector/concept board affiliation and hot sector rankings\n"
            )

        system_message += (
            " Provide specific, actionable insights with supporting evidence to help traders make informed decisions.\n"
            " IMPORTANT: For Chinese A-stock analysis, you MUST call get_market_news, get_company_announcements, get_research_reports, and get_sector in addition to get_news — these provide critical China-specific data not available from general news sources."
            + "\n Make sure to append a Markdown table at the end of the report to organize key points in the report, organized and easy to read."
            + get_language_instruction()
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " If you are unable to fully answer, that's OK; another assistant with different tools"
                    " will help where you left off. Execute what you can to make progress."
                    " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
                    " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
                    " You have access to the following tools: {tool_names}."
                    " Today's date is {current_date}; treat it as 'now' for all analysis and tool-call date ranges. {instrument_context}\n"
                    "{system_message}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(instrument_context=instrument_context)

        chain = prompt | llm.bind_tools(tools)
        result = chain.invoke(state["messages"])

        report = ""

        if len(result.tool_calls) == 0:
            report = result.content

        return {
            "messages": [result],
            "news_report": report,
        }

    return news_analyst_node
