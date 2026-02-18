from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import time
import json
from tradingagents.agents.utils.agent_utils import get_fundamentals, get_balance_sheet, get_cashflow, get_income_statement, get_insider_transactions
from tradingagents.dataflows.config import get_config


def create_fundamentals_analyst(llm):
    def fundamentals_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]
        company_display = state.get("company_display_name") or ticker

        tools = [
            get_fundamentals,
            get_balance_sheet,
            get_cashflow,
            get_income_statement,
        ]

        system_message = (
            "You are a senior Fundamental Analyst at a top-tier investment bank. "
            "Your task is to conduct a rigorous assessment of the target company and benchmark it against its peers.\n\n"
            "**PART 1: DEEP DIVE TARGET ANALYSIS**\n"
            "Diagnose the company's health beyond just numbers:\n"
            "1. **Growth Quality:** Analyze Revenue and Net Income trends. Is growth organic or artificial?\n"
            "2. **Profitability & Moat:** Check Gross and Operating Margins. Are they expanding (strong moat) or contracting (competition)?\n"
            "3. **Financial Health:** Scrutinize the Balance Sheet (Debt levels, Liquidity).\n"
            "4. **Earnings Quality:** Compare Net Income vs. Operating Cash Flow to spot accounting red flags.\n\n"
            "**PART 2: PEER BENCHMARKING**\n"
            "Identify 2-3 direct competitors. Compare the target against them across these three pillars:\n"
            "- **Valuation:** P/E, P/S, EV/EBITDA.\n"
            "- **Growth:** Revenue Growth (YoY), EPS Growth.\n"
            "- **Profitability:** Gross Margin, Net Margin, ROE.\n\n"
            "**OUTPUT FORMAT REQUIREMENTS:**\n"
            "- **Benchmark Table:** You MUST output a Markdown table. Columns should be: [Metric, Target Company, Peer 1, Peer 2, Peer Average].\n"
            "- **Verdict:** Synthesize the data into a Bullish/Bearish/Neutral conclusion.\n"
            "- **Tone:** Professional and data-driven.\n\n"
            "Use tools: `get_fundamentals`, `get_balance_sheet`, `get_cashflow`, `get_income_statement`. "
            "**Requirement:** Use **quarterly** (freq='quarterly') financials and prioritize the **latest available quarter** for valuation and peer comparison; use annual data only when explicitly comparing multi-year trends. "
            "Call tools multiple times to get data for both the target and its peers."
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
                    "For your reference, the current date is {current_date}. The company we want to look at is {company_display} (ticker: {ticker}). Use the ticker for all tool calls.",
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
            "fundamentals_report": report,
        }

    return fundamentals_analyst_node
