import os

DEFAULT_CONFIG = {
    "project_dir": os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
    "results_dir": os.getenv("TRADINGAGENTS_RESULTS_DIR", "./results"),
    "data_cache_dir": os.path.join(
        os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
        "dataflows/data_cache",
    ),
    # LLM settings
    # LLM settings
    "llm_provider": "qwen",
    "deep_think_llm": "qwen3.5-plus",  # High-accuracy model
    "quick_think_llm": "qwen-plus",    # Fast model
    "backend_url": None,               # Uses DashScope default
    # Provider-specific thinking configuration
    "google_thinking_level": None,      # "high", "minimal", etc.
    "openai_reasoning_effort": None,    # "medium", "high", "low"
    # Analysis depth: "quick", "standard", or "deep"
    # Controls which analysts, debate rounds, and data sources are used.
    "analysis_depth": "standard",
    # Debate and discussion settings
    "max_debate_rounds": 1,
    "max_risk_discuss_rounds": 1,
    "max_recur_limit": 1000,
    # Data vendor configuration
    # Category-level configuration (default for all tools in category)
    "data_vendors": {
        "core_stock_apis": "yfinance",       # Options: alpha_vantage, yfinance
        "technical_indicators": "yfinance",  # Options: alpha_vantage, yfinance
        "fundamental_data": "yfinance",      # Options: alpha_vantage, yfinance
        "news_data": "yfinance",             # Options: alpha_vantage, yfinance
    },
    # Tool-level configuration (takes precedence over category-level)
    "tool_vendors": {
        # Example: "get_stock_data": "alpha_vantage",  # Override category default
    },
}

# ── Analysis depth presets ──────────────────────────────────────────
# Each preset defines: analysts, debate rounds, risk rounds, and
# whether to use Alpha Vantage sentiment-scored news (expensive).
DEPTH_PRESETS = {
    "quick": {
        "analysts": ["market", "fundamentals"],
        "max_debate_rounds": 1,
        "max_risk_discuss_rounds": 0,
        "use_av_sentiment_news": False,
    },
    "standard": {
        "analysts": ["market", "fundamentals", "news", "social", "valuation"],
        "max_debate_rounds": 2,
        "max_risk_discuss_rounds": 1,
        "use_av_sentiment_news": False,
    },
    "deep": {
        "analysts": ["market", "fundamentals", "news", "social", "industry", "valuation"],
        "max_debate_rounds": 2,
        "max_risk_discuss_rounds": 2,
        "use_av_sentiment_news": True,
    },
}
