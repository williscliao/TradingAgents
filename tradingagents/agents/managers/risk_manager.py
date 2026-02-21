import time
import json
from tradingagents.agents.utils.agent_utils import truncate_content


def create_risk_manager(llm, memory):
    def risk_manager_node(state) -> dict:
        risk_debate_state = state["risk_debate_state"]
        history = risk_debate_state.get("history", "")
        company_name = state.get("company_display_name") or state["company_of_interest"]
        trader_plan = state["investment_plan"]
        market_research_report = state["market_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]
        sentiment_report = state["sentiment_report"]
        industry_report = state.get("industry_report", "")
        risk_metrics = state.get("risk_metrics_report", "")

        # Truncate history and reports to keep context window manageable
        history = truncate_content(history, 12000)
        trader_plan = truncate_content(trader_plan, 4000)
        market_research_report = truncate_content(market_research_report, 3000)
        news_report = truncate_content(news_report, 3000)
        fundamentals_report = truncate_content(fundamentals_report, 3000)
        sentiment_report = truncate_content(sentiment_report, 3000)
        industry_report = truncate_content(industry_report, 3000)

        curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}\n\n{industry_report}"
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        for i, rec in enumerate(past_memories, 1):
            past_memory_str += rec["recommendation"] + "\n\n"

        prompt = f"""As the Risk Management Judge and Debate Facilitator, your goal is to evaluate the debate between three risk analysts—Aggressive, Neutral, and Conservative—and determine the best course of action for the trader. Your decision must result in a clear recommendation: Buy, Sell, or Hold. Choose Hold only if strongly justified by specific arguments, not as a fallback when all sides seem valid. Strive for clarity and decisiveness.

## Quantitative Risk Data
You have access to pre-computed quantitative risk metrics. These are OBJECTIVE, DATA-DRIVEN measurements that should ANCHOR your decision:

{risk_metrics}

## Guidelines for Decision-Making

1. **Quantitative Risk Check (NEW — do this FIRST)**: Before evaluating qualitative arguments, assess whether the quantitative risk profile supports the proposed action:
   - If **VaR (99%)** exceeds 5% daily, the position carries extreme tail risk — acknowledge this.
   - If **Sharpe Ratio < 0.5**, risk-adjusted returns are poor — the burden of proof shifts to the bulls.
   - If **Sharpe Ratio > 1.5**, risk-adjusted returns are excellent — the burden shifts to the bears.
   - If **Max Drawdown > 30%**, this stock has demonstrated severe downside — factor this into position sizing.
   - If **Current Drawdown > 10%**, consider whether this is a buying opportunity (mean reversion) or a falling knife.
   - **Overall Risk Score** of 7+ should trigger extra caution; 3 or below is a green light from a risk standpoint.

2. **Summarize Key Arguments**: Extract the strongest points from each analyst, focusing on relevance to the context.

3. **Quantitative vs. Qualitative Integration**: When analysts' qualitative arguments conflict with quantitative data, explain which one you're prioritizing and why. Data should generally override opinion unless there is a clear catalytic reason for a regime change.

4. **Provide Rationale**: Support your recommendation with direct quotes and counterarguments from the debate, and cite specific risk metrics.

5. **Refine the Trader's Plan**: Start with the trader's original plan, **{trader_plan}**, and adjust it based on both the analysts' insights AND the quantitative risk profile. Specifically include:
   - **Position sizing justification** based on VaR and max drawdown
   - **Risk/reward framing** using the Sharpe ratio
   - **Stop-loss or exit criteria** informed by historical drawdown data

6. **Learn from Past Mistakes**: Use lessons from **{past_memory_str}** to address prior misjudgments and improve the decision.

## Deliverables
- A clear and actionable recommendation: **Buy, Sell, or Hold**
- Quantitative risk summary (cite VaR, Sharpe, Max Drawdown explicitly)
- Detailed reasoning anchored in the debate AND the risk metrics
- Position sizing guidance based on the risk profile

---

**Analysts Debate History:**  
{history}

---

Focus on actionable insights and continuous improvement. Build on past lessons, critically evaluate all perspectives, and ensure each decision advances better outcomes."""

        response = llm.invoke(prompt)

        new_risk_debate_state = {
            "judge_decision": response.content,
            "history": risk_debate_state["history"],
            "aggressive_history": risk_debate_state["aggressive_history"],
            "conservative_history": risk_debate_state["conservative_history"],
            "neutral_history": risk_debate_state["neutral_history"],
            "latest_speaker": "Judge",
            "current_aggressive_response": risk_debate_state["current_aggressive_response"],
            "current_conservative_response": risk_debate_state["current_conservative_response"],
            "current_neutral_response": risk_debate_state["current_neutral_response"],
            "count": risk_debate_state["count"],
        }

        return {
            "risk_debate_state": new_risk_debate_state,
            "final_trade_decision": response.content,
        }

    return risk_manager_node
