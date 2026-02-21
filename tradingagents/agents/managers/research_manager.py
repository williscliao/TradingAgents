import time
import json
from tradingagents.agents.utils.agent_utils import truncate_content


def create_research_manager(llm, memory):
    def research_manager_node(state) -> dict:
        history = state["investment_debate_state"].get("history", "")
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]
        industry_report = state.get("industry_report", "")


        investment_debate_state = state["investment_debate_state"]

        # Truncate history to keep context window manageable
        history = truncate_content(history, 12000)

        curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}\n\n{industry_report}"
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        for i, rec in enumerate(past_memories, 1):
            past_memory_str += rec["recommendation"] + "\n\n"

        prompt = f"""As the Portfolio Manager and Debate Facilitator, your role is to critically evaluate this round of debate and make a definitive decision: align with the bear analyst, the bull analyst, or choose Hold only if strongly justified.

To ensure the highest quality judgment, your response MUST be structured into the following distinct sections:

### 1. Fact-Check & Logical Flaws
(Critically evaluate the arguments presented by both the Bull and Bear analysts. Check if their claims are supported by the provided data in the Market and Fundamentals reports. explicitly point out any logical flaws, over-optimistic assumptions, or ignored risks in their arguments.)

### 2. Scenario & Weighting Analysis
(Synthesize the current macroeconomic environment and news landscape. Discuss which analyst's argument carries more weight given the current market conditions. For example, if interest rates are rising, justify why balance sheet risks highlighted by the Bear should be weighted more heavily than growth projections by the Bull.)

### 3. Final Decision & Execution Plan
(Based on your rigorous analysis above, provide a clear, actionable recommendation: BUY, SELL, or HOLD. Provide concrete strategic actions for the trader, including position sizing and risk management.)

Take into account your past mistakes on similar situations. Use these insights to refine your decision-making and ensure you are learning and improving.

Here are your past reflections on mistakes:
"{past_memory_str}"

Here is the supporting data:
Market research report:
{market_research_report}

Fundamentals report:
{fundamentals_report}

Recent News:
{news_report}

Sentiment & Industry:
{sentiment_report}
{industry_report}

Here is the debate you need to judge:
Debate History:
{history}"""
        response = llm.invoke(prompt)

        new_investment_debate_state = {
            "judge_decision": response.content,
            "history": investment_debate_state.get("history", ""),
            "bear_history": investment_debate_state.get("bear_history", ""),
            "bull_history": investment_debate_state.get("bull_history", ""),
            "current_response": response.content,
            "count": investment_debate_state["count"],
        }

        return {
            "investment_debate_state": new_investment_debate_state,
            "investment_plan": response.content,
        }

    return research_manager_node
