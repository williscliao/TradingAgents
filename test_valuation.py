import asyncio
import os
import yfinance as yf
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from tradingagents.agents.analysts.valuation_analyst import create_valuation_analyst
from tradingagents.agents.utils.valuation_calc_tools import get_full_valuation_metrics

async def main():
    print("Testing Valuation Analyst on MU (Micron Technology)...")
    
    api_key = os.getenv("DASHSCOPE_API_KEY", "")

    llm = ChatOpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model="qwen-plus",
        temperature=0
    )

    node = create_valuation_analyst(llm)

    state = {
        "messages": [
            HumanMessage(content="You are a Valuation Analyst. Your task is to analyze MU.")
        ],
        "trade_date": "2026-02-21",
        "company_of_interest": "MU",
        "company_display_name": "Micron Technology, Inc."
    }
    
    try:
        # Pass 1: Ask LLM what tools it wants
        new_state = node(state)
        last_msg = new_state["messages"][-1]
        
        # If it returned a tool call, let's manually execute it and feed it back
        if getattr(last_msg, 'tool_calls', None):
            print(f"Executing Tool: {last_msg.tool_calls[0]['name']}")
            # Execute our macro tool manually
            tool_output_str = get_full_valuation_metrics.invoke({"ticker": "MU"})
            
            # Create a ToolMessage
            tool_msg = ToolMessage(
                content=tool_output_str,
                tool_call_id=last_msg.tool_calls[0]["id"]
            )
            
            # Re-run node with the newly augmented history
            state["messages"] = new_state["messages"] + [tool_msg]
            
            print("\nFeeding Tool Output Back to LLM for Synthesis Phase...")
            final_state = node(state)
            
            print("\n---------- FINAL REPORT OUTPUT ----------")
            print(final_state["valuation_report"])
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
