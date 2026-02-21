from langchain_core.tools import tool
from typing import Annotated
from tradingagents.dataflows.interface import route_to_vendor
from tradingagents.agents.utils.resilient import resilient_call


@tool
def get_indicators(
    symbol: Annotated[str, "ticker symbol of the company"],
    indicator: Annotated[str, "technical indicator to get the analysis and report of (can be comma-separated list, e.g. 'rsi, macd, sma_50')"],
    curr_date: Annotated[str, "The current trading date you are trading on, YYYY-mm-dd"],
    look_back_days: Annotated[int, "how many days to look back"] = 30,
) -> str:
    """
    Retrieve technical indicators for a given ticker symbol.
    Supports batch processing by passing multiple indicators separated by commas.
    Uses the configured technical_indicators vendor.
    
    Args:
        symbol (str): Ticker symbol of the company, e.g. AAPL, TSM
        indicator (str): Single indicator or comma-separated list of indicators
        curr_date (str): The current trading date you are trading on, YYYY-mm-dd
        look_back_days (int): How many days to look back, default is 30
    Returns:
        str: A formatted report containing the technical indicators.
    """
    return resilient_call(
        route_to_vendor, "get_indicators", symbol, indicator, curr_date, look_back_days,
        tool_name=f"get_indicators({indicator})",
    )