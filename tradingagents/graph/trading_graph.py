# TradingAgents/graph/trading_graph.py

import os
from pathlib import Path
import json
from datetime import date
from typing import Dict, Any, Tuple, List, Optional

from langgraph.prebuilt import ToolNode

from tradingagents.llm_clients import create_llm_client

from tradingagents.agents import *
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.agents.utils.memory import FinancialSituationMemory
from tradingagents.agents.utils.agent_states import (
    AgentState,
    InvestDebateState,
    RiskDebateState,
)
from tradingagents.dataflows.config import set_config

# Import the new abstract tool methods from agent_utils
from tradingagents.agents.utils.agent_utils import (
    get_stock_data,
    get_indicators,
    get_fundamentals,
    get_balance_sheet,
    get_cashflow,
    get_income_statement,
    get_news,
    get_insider_transactions,
    get_global_news,
    get_social_sentiment,
    get_quant_grades,
)
from tradingagents.agents.utils.valuation_calc_tools import (
    calculate_wacc,
    calculate_dcf,
    calculate_relative_valuation,
)

from .conditional_logic import ConditionalLogic
from .setup import GraphSetup
from .propagation import Propagator
from .reflection import Reflector
from .signal_processing import SignalProcessor


class TradingAgentsGraph:
    """Main class that orchestrates the trading agents framework."""

    def __init__(
        self,
        selected_analysts=["market", "social", "news", "fundamentals"],
        debug=False,
        config: Dict[str, Any] = None,
        callbacks: Optional[List] = None,
        progress_callback: Optional[Any] = None,
    ):
        """Initialize the trading agents graph and components.
 
        Args:
            selected_analysts: List of analyst types to include
            debug: Whether to run in debug mode
            config: Configuration dictionary. If None, uses default config
            callbacks: Optional list of callback handlers (e.g., for tracking LLM/tool stats)
            progress_callback: Optional function for real-time progress updates (e.g. for CLI)
        """
        self.debug = debug
        self.config = config or DEFAULT_CONFIG
        self.callbacks = callbacks or []
        self.progress_callback = progress_callback

        # Update the interface's config
        set_config(self.config)

        # Create necessary directories
        os.makedirs(
            os.path.join(self.config["project_dir"], "dataflows/data_cache"),
            exist_ok=True,
        )

        # Initialize LLMs with provider-specific thinking configuration
        llm_kwargs = self._get_provider_kwargs()

        # Add callbacks to kwargs if provided (passed to LLM constructor)
        if self.callbacks:
            llm_kwargs["callbacks"] = self.callbacks

        deep_client = create_llm_client(
            provider=self.config["llm_provider"],
            model=self.config["deep_think_llm"],
            base_url=self.config.get("backend_url"),
            **llm_kwargs,
        )
        quick_client = create_llm_client(
            provider=self.config["llm_provider"],
            model=self.config["quick_think_llm"],
            base_url=self.config.get("backend_url"),
            **llm_kwargs,
        )

        self.deep_thinking_llm = deep_client.get_llm()
        self.quick_thinking_llm = quick_client.get_llm()
        
        # Initialize memories
        self.bull_memory = FinancialSituationMemory("bull_memory", self.config)
        self.bear_memory = FinancialSituationMemory("bear_memory", self.config)
        self.trader_memory = FinancialSituationMemory("trader_memory", self.config)
        self.invest_judge_memory = FinancialSituationMemory("invest_judge_memory", self.config)
        self.risk_manager_memory = FinancialSituationMemory("risk_manager_memory", self.config)

        # Create tool nodes
        self.tool_nodes = self._create_tool_nodes()

        # Initialize components
        self.conditional_logic = ConditionalLogic(
            max_debate_rounds=self.config.get("max_debate_rounds", 1),
            max_risk_discuss_rounds=self.config.get("max_risk_discuss_rounds", 1),
        )
        self.graph_setup = GraphSetup(
            self.quick_thinking_llm,
            self.deep_thinking_llm,
            self.tool_nodes,
            self.bull_memory,
            self.bear_memory,
            self.trader_memory,
            self.invest_judge_memory,
            self.risk_manager_memory,
            self.conditional_logic,
            progress_callback=self.progress_callback,
        )

        self.propagator = Propagator(
            max_recur_limit=self.config.get("max_recur_limit", 200)
        )
        self.reflector = Reflector(self.quick_thinking_llm)
        self.signal_processor = SignalProcessor(self.quick_thinking_llm)

        # State tracking
        self.curr_state = None
        self.ticker = None
        self.log_states_dict = {}  # date to full state dict

        # Set up the graph
        is_standalone = self.config.get("analysis_depth") == "standalone"
        self.graph = self.graph_setup.setup_graph(selected_analysts, is_standalone=is_standalone)

    def _get_provider_kwargs(self) -> Dict[str, Any]:
        """Get provider-specific kwargs for LLM client creation."""
        kwargs = {}
        provider = self.config.get("llm_provider", "").lower()

        if provider == "google":
            kwargs["max_retries"] = 6
            thinking_level = self.config.get("google_thinking_level")
            if thinking_level:
                kwargs["thinking_level"] = thinking_level

        elif provider == "openai":
            reasoning_effort = self.config.get("openai_reasoning_effort")
            if reasoning_effort:
                kwargs["reasoning_effort"] = reasoning_effort

        return kwargs

    def _create_tool_nodes(self) -> Dict[str, ToolNode]:
        """Create tool nodes for different data sources using abstract methods."""
        return {
            "market": ToolNode(
                [
                    # Core stock data tools
                    get_stock_data,
                    # Technical indicators
                    get_indicators,
                ]
            ),
            "social": ToolNode(
                [
                    # News + sentiment + quant grades for social media analysis
                    get_news,
                    get_social_sentiment,
                    get_quant_grades,
                ]
            ),
            "news": ToolNode(
                [
                    # News and insider information
                    get_news,
                    get_global_news,
                    get_insider_transactions,
                ]
            ),
            "fundamentals": ToolNode(
                [
                    # Fundamental analysis tools
                    get_fundamentals,
                    get_balance_sheet,
                    get_cashflow,
                    get_income_statement,
                    get_insider_transactions,
                ]
            ),
            "industry": ToolNode(
                [
                    # Industry & strategy analysis tools
                    get_news,
                    get_global_news,
                    get_fundamentals,
                ]
            ),
            "valuation": ToolNode(
                [
                    # Valuation analysis tools (financial data + market data + calc tools)
                    get_fundamentals,
                    get_balance_sheet,
                    get_cashflow,
                    get_income_statement,
                    get_stock_data,
                    get_indicators,
                    calculate_wacc,
                    calculate_dcf,
                    calculate_relative_valuation,
                ]
            ),
        }

    def propagate(self, company_name, trade_date):
        """Run the trading agents graph for a company on a specific date.
        company_name: User input (ticker symbol, e.g. VRT). Resolved via Yahoo Finance to avoid wrong mapping (e.g. VRT -> Vertiv not Vertex).
        """
        from tradingagents.dataflows.y_finance import resolve_ticker_to_symbol_and_name

        ticker, company_display_name = resolve_ticker_to_symbol_and_name(company_name)
        self.ticker = ticker

        # Initialize state with canonical ticker and official company name from Yahoo
        init_agent_state = self.propagator.create_initial_state(
            ticker, trade_date, company_display_name=company_display_name
        )
        args = self.propagator.get_graph_args()

        if self.debug:
            # Debug mode with tracing
            trace = []
            for chunk in self.graph.stream(init_agent_state, **args):
                messages = chunk.get("messages", [])
                if len(messages) > 0:
                    messages[-1].pretty_print()
                    trace.append(chunk)

            final_state = trace[-1] if trace else {}
        else:
            # Standard mode without tracing
            final_state = self.graph.invoke(init_agent_state, **args)

        # Store current state for reflection
        self.curr_state = final_state

        # Log state
        self._log_state(trade_date, final_state)

        # Return decision and processed signal
        decision = final_state.get("final_trade_decision")
        signal = self.process_signal(decision) if decision else None
        return final_state, signal

    def _log_state(self, trade_date, final_state):
        """Log the final state to a JSON file."""
        inv_state = final_state.get("investment_debate_state", {})
        risk_state = final_state.get("risk_debate_state", {})

        self.log_states_dict[str(trade_date)] = {
            "company_of_interest": final_state.get("company_of_interest"),
            "trade_date": final_state.get("trade_date"),
            "market_report": final_state.get("market_report", ""),
            "sentiment_report": final_state.get("sentiment_report", ""),
            "news_report": final_state.get("news_report", ""),
            "fundamentals_report": final_state.get("fundamentals_report", ""),
            "industry_report": final_state.get("industry_report", ""),
            "investment_debate_state": {
                "bull_history": inv_state.get("bull_history", ""),
                "bear_history": inv_state.get("bear_history", ""),
                "history": inv_state.get("history", ""),
                "current_response": inv_state.get("current_response", ""),
                "judge_decision": inv_state.get("judge_decision", ""),
            },
            "trader_investment_decision": final_state.get("trader_investment_plan"),
            "risk_debate_state": {
                "aggressive_history": risk_state.get("aggressive_history", ""),
                "conservative_history": risk_state.get("conservative_history", ""),
                "neutral_history": risk_state.get("neutral_history", ""),
                "history": risk_state.get("history", ""),
                "judge_decision": risk_state.get("judge_decision", ""),
            },
            "investment_plan": final_state.get("investment_plan"),
            "final_trade_decision": final_state.get("final_trade_decision"),
        }

        # Save to file under configured results directory
        results_dir = self.config.get("results_dir", "./results")
        directory = Path(results_dir) / self.ticker / "TradingAgentsStrategy_logs"
        directory.mkdir(parents=True, exist_ok=True)

        log_path = directory / f"full_states_log_{trade_date}.json"
        with open(log_path, "w") as f:
            json.dump(self.log_states_dict, f, indent=4)

    def reflect_and_remember(self, returns_losses):
        """Reflect on decisions and update memory based on returns."""
        self.reflector.reflect_bull_researcher(
            self.curr_state, returns_losses, self.bull_memory
        )
        self.reflector.reflect_bear_researcher(
            self.curr_state, returns_losses, self.bear_memory
        )
        self.reflector.reflect_trader(
            self.curr_state, returns_losses, self.trader_memory
        )
        self.reflector.reflect_invest_judge(
            self.curr_state, returns_losses, self.invest_judge_memory
        )
        self.reflector.reflect_risk_manager(
            self.curr_state, returns_losses, self.risk_manager_memory
        )

    def process_signal(self, full_signal):
        """Process a signal to extract the core decision."""
        return self.signal_processor.process_signal(full_signal)
