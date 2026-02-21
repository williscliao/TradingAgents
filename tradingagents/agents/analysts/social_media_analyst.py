from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import time
import json
from tradingagents.agents.utils.agent_utils import get_news, get_social_sentiment, get_quant_grades
from tradingagents.dataflows.config import get_config


def create_social_media_analyst(llm):
    def social_media_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]
        company_display = state.get("company_display_name") or ticker

        tools = [
            get_news,
            get_social_sentiment,
            get_quant_grades,
        ]

        system_message = (
            "You are a **Market Sentiment Analyst** specializing in measuring institutional and insider "
            "sentiment for a specific company. Your analysis provides a complementary perspective to the "
            "News Analyst — while they focus on WHAT happened, you focus on HOW THE MARKET IS REACTING.\n\n"

            "**YOUR PRIMARY DATA SOURCE is `get_social_sentiment`** — use this FIRST to pull:\n"
            "1. Wall Street analyst recommendation trends (buy/sell/hold distribution changes)\n"
            "2. Earnings surprise history (beat/miss pattern)\n"
            "3. Insider sentiment (net buying vs selling by executives/directors)\n\n"
            "Then use `get_news` to search for company-specific news that may explain sentiment changes.\n\n"

            "**Your report MUST cover:**\n"
            "1. **Analyst Consensus**: What is the current buy/sell/hold distribution? Is it shifting "
            "more bullish or bearish? How does the consensus compare to 3-6 months ago?\n"
            "2. **Earnings Track Record**: Is the company consistently beating estimates? What is the "
            "beat rate and average surprise? This indicates execution quality and management credibility.\n"
            "3. **Insider Activity**: Are insiders buying or selling? Insider buying is one of the "
            "strongest bullish signals; heavy selling can be a red flag.\n"
            "4. **Sentiment Drivers**: Use `get_news` to identify WHAT is driving changes in "
            "analyst ratings, earnings trends, or insider behavior.\n"
            "5. **Contrarian Signals**: Is sentiment extremely one-sided? Extreme bullishness with "
            "everyone at 'Strong Buy' may mean limited upside; unanimous bearishness may be a bottom signal.\n"
            "6. **Actionable Insights**: What does the combined sentiment data suggest for trading?\n\n"

            "**OUTPUT FORMAT:**\n"
            "- Append a Markdown summary table at the end with columns: "
            "[Metric, Value, Interpretation]\n"
            "- Be quantitative: cite specific numbers (analyst counts, EPS figures, surprise %)\n"
            "- Clearly distinguish between sentiment DATA and your INTERPRETATION"
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
                    " You have access to the following tools: {tool_names}.\n{system_message}"
                    "For your reference, the current date is {current_date}. The current company we want to analyze is {company_display} (ticker: {ticker}). Use the ticker for tool calls.",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date, ticker=ticker, company_display=company_display)

        chain = prompt | llm.bind_tools(tools)

        result = chain.invoke(state["messages"])

        report = ""

        if len(result.tool_calls) == 0:
            report = result.content

        return {
            "messages": [result],
            "sentiment_report": report,
        }

    return social_media_analyst_node
