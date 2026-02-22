import sys
import os
sys.path.append("/Users/willis.liao/TradingAgents")

from tradingagents.agents.utils.news_data_tools import get_news, get_global_news, get_insider_transactions
from tradingagents.dataflows.config import set_config
from tradingagents.default_config import DEFAULT_CONFIG

set_config(DEFAULT_CONFIG)

print("Testing get_news.invoke for GOOG:")
try:
    news = get_news.invoke({"ticker": "GOOG", "start_date": "2026-02-14", "end_date": "2026-02-21"})
    print("get_news length:", len(str(news)))
    print("get_news abstract:", str(news)[:400])
except Exception as e:
    print("get_news error:", e)

print("\nTesting get_global_news.invoke:")
try:
    global_news = get_global_news.invoke({"curr_date": "2026-02-21", "look_back_days": 7, "limit": 5})
    print("get_global_news length:", len(str(global_news)))
    print("get_global_news abstract:", str(global_news)[:400])
except Exception as e:
    print("get_global_news error:", e)

print("\nTesting get_insider_transactions.invoke for GOOG:")
try:
    insiders = get_insider_transactions.invoke({"ticker": "GOOG"})
    print("get_insider_transactions length:", len(str(insiders)))
    print("get_insider_transactions abstract:", str(insiders)[:400])
except Exception as e:
    print("get_insider_transactions error:", e)
