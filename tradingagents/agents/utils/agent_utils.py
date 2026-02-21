from langchain_core.messages import HumanMessage, RemoveMessage

# Import tools from separate utility files
from tradingagents.agents.utils.core_stock_tools import (
    get_stock_data
)
from tradingagents.agents.utils.technical_indicators_tools import (
    get_indicators
)
from tradingagents.agents.utils.fundamental_data_tools import (
    get_fundamentals,
    get_balance_sheet,
    get_cashflow,
    get_income_statement
)
from tradingagents.agents.utils.news_data_tools import (
    get_news,
    get_insider_transactions,
    get_global_news
)
from tradingagents.agents.utils.social_sentiment_tools import (
    get_social_sentiment,
    get_quant_grades
)


def create_msg_delete():
    def delete_messages(state):
        """Clear messages and add placeholder for Anthropic compatibility"""
        messages = state["messages"]

        # Remove all messages
        removal_operations = [RemoveMessage(id=m.id) for m in messages]

        # Add a minimal placeholder message
        placeholder = HumanMessage(content="Continue")

        return {"messages": removal_operations + [placeholder]}

    return delete_messages


def truncate_content(content: str, max_length: int = 20000) -> str:
    """Truncate content to max_length characters, keeping the beginning and end.
    
    Args:
        content: The string to truncate.
        max_length: Maximum allowed length. Default 20,000 chars (~5000 tokens).
    
    Returns:
        Truncated string.
    """
    if not isinstance(content, str):
        return str(content)
        
    if len(content) <= max_length:
        return content
    
    # Keep more of the end (recent info usually more important) but start is context
    # Try 30% start, 70% end strategy or just 50/50
    # Let's do 50/50 for simplicity
    half = max_length // 2
    return f"{content[:half]}\n\n[...TRUNCATED {len(content) - max_length} chars...]\n\n{content[-half:]}"