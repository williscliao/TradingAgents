# TradingAgents/graph/setup.py

from typing import Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode

from tradingagents.agents import *
from tradingagents.agents.risk_mgmt.risk_metrics_node import create_risk_metrics_node
from tradingagents.agents.utils.agent_states import AgentState

from .conditional_logic import ConditionalLogic
import logging

logger = logging.getLogger(__name__)

class GraphSetup:
    """Configures the TradingAgentsGraph structure and node connections."""

    def __init__(
        self,
        quick_thinking_llm: ChatOpenAI,
        deep_thinking_llm: ChatOpenAI,
        tool_nodes: Dict[str, ToolNode],
        bull_memory=None,
        bear_memory=None,
        trader_memory=None,
        invest_judge_memory=None,
        risk_manager_memory=None,
        conditional_logic: Optional[ConditionalLogic] = None,
        progress_callback: Optional[Any] = None,
    ):
        self.quick_thinking_llm = quick_thinking_llm
        self.deep_thinking_llm = deep_thinking_llm
        self.tool_nodes = tool_nodes
        self.bull_memory = bull_memory
        self.bear_memory = bear_memory
        self.trader_memory = trader_memory
        self.invest_judge_memory = invest_judge_memory
        self.risk_manager_memory = risk_manager_memory
        self.conditional_logic = conditional_logic
        self.progress_callback = progress_callback

    def build_analyst_subgraph(self, node, tools, delete, condition, mapping):
        """Builds a standardized subgraph for an individual analyst."""
        subgraph = StateGraph(AgentState)
        subgraph.add_node("Analyst", node)
        subgraph.add_node("Tools", tools)
        subgraph.add_node("MsgClear", delete)

        subgraph.add_edge(START, "Analyst")
        subgraph.add_conditional_edges("Analyst", condition, mapping)
        subgraph.add_edge("Tools", "Analyst")
        subgraph.add_edge("MsgClear", END)

        return subgraph.compile()

    def setup_graph(self, selected_analysts: list, is_standalone: bool = False):
        """Constructs the high-level graph topology by connecting agent nodes."""
        workflow = StateGraph(AgentState)
        
        # SEQUENTIAL EXECUTION: Chain Analyst Subgraphs
        # We start with the START node as the entry point
        last_node = START

        for analyst_type in selected_analysts:
            # 1. Create specific nodes for this analyst
            if analyst_type == "market":
                node = create_market_analyst(self.quick_thinking_llm)
            elif analyst_type == "social":
                node = create_social_media_analyst(self.quick_thinking_llm)
            elif analyst_type == "news":
                node = create_news_analyst(self.quick_thinking_llm)
            elif analyst_type == "fundamentals":
                node = create_fundamentals_analyst(self.quick_thinking_llm)
            elif analyst_type == "industry":
                node = create_industry_analyst(self.quick_thinking_llm)
            elif analyst_type == "valuation":
                node = create_valuation_analyst(self.quick_thinking_llm)
            else:
                continue

            # 2. Get tools, delete node, and standard condition
            tools = self.tool_nodes[analyst_type]
            delete = create_msg_delete()
            condition = self.conditional_logic.should_continue_analyst

            # 3. Create mapping for conditional edges
            mapping = {
                "continue": "Tools",
                "end": "MsgClear"
            }

            # 4. Build Subgraph
            subgraph = self.build_analyst_subgraph(node, tools, delete, condition, mapping)
            
            # Map analyst type to report key
            report_key_map = {
                "market": "market_report",
                "social": "sentiment_report",
                "news": "news_report",
                "fundamentals": "fundamentals_report",
                "industry": "industry_report",
                "valuation": "valuation_report"
            }
            target_report_key = report_key_map.get(analyst_type)

            def make_safe_analyst_wrapper(compiled_subgraph, report_key, analyst_type):
                def wrapper(state):
                    # Use a generous recursion limit for subgraphs (but we'll have a hard limit too)
                    sub_config = {"recursion_limit": 500}
                    MAX_STEPS = 120 # ~40 tool cycles
                    
                    final_result = {}
                    step_count = 0
                    if self.progress_callback:
                        self.progress_callback("status", f"Starting {analyst_type} analysis...")
                    
                    try:
                        for chunk in compiled_subgraph.stream(state, config=sub_config):
                            step_count += 1
                            if step_count > MAX_STEPS:
                                msg = f"⚠️ [{analyst_type.capitalize()}] Step limit ({MAX_STEPS}) reached. Forcing termination."
                                if self.progress_callback:
                                    self.progress_callback("status", msg)
                                logger.warning(msg)
                                break

                            # Capture any updates from any node in the subgraph
                            for node_name, node_output in chunk.items():
                                if node_output:
                                    # Merge updates into final_result
                                    final_result.update(node_output)
                                
                                # Log activity for better visibility
                                if node_name == "Analyst":
                                    msg = f"[{analyst_type.capitalize()}] Step {step_count}: Thinking..."
                                    if self.progress_callback:
                                        self.progress_callback("status", msg)
                                elif node_name == "Tools":
                                    msg = f"[{analyst_type.capitalize()}] Step {step_count}: Executing tools..."
                                    if self.progress_callback:
                                        self.progress_callback("status", msg)
                                elif node_name == "MsgClear":
                                    msg = f"[{analyst_type.capitalize()}] Completing analysis..."
                                    if self.progress_callback:
                                        self.progress_callback("status", msg)

                    except Exception as e:
                        msg = f"❌ [{analyst_type.capitalize()}] Analysis failed: {e}"
                        if self.progress_callback:
                            self.progress_callback("status", msg)
                        logger.error(msg)
                    
                    report_content = final_result.get(report_key)
                    if not report_content or not str(report_content).strip():
                        # SALVAGE MECHANISM: Search messages for partial report
                        messages = final_result.get("messages", [])
                        salvaged_text = ""
                        # Look for the longest AI message that isn't a tool call
                        for msg in reversed(messages):
                            if hasattr(msg, "content") and msg.content and not getattr(msg, "tool_calls", None):
                                if len(str(msg.content)) > len(salvaged_text):
                                    salvaged_text = str(msg.content)
                        
                        if salvaged_text:
                            report_content = f"⚠️ [{analyst_type.capitalize()}] **Step limit reached.** Recovered partial analysis:\n\n{salvaged_text}"
                        else:
                            report_content = f"⚠️ [{analyst_type.capitalize()}] Analysis incomplete (hit step limit or encountered an error)."
                    
                    updates = {report_key: report_content}
                    return updates
                return wrapper

            safe_node = make_safe_analyst_wrapper(subgraph, target_report_key, analyst_type)
            
            node_name = f"{analyst_type.capitalize()} Analyst"
            workflow.add_node(node_name, safe_node)
            
            # 5. Connect Sequentially: last_node -> current analyst
            workflow.add_edge(last_node, node_name)
            last_node = node_name

        # Create researcher and manager nodes
        bull_researcher_node = create_bull_researcher(
            self.quick_thinking_llm, self.bull_memory
        )
        bear_researcher_node = create_bear_researcher(
            self.quick_thinking_llm, self.bear_memory
        )
        research_manager_node = create_research_manager(
            self.deep_thinking_llm, self.invest_judge_memory
        )
        trader_node = create_trader(self.quick_thinking_llm, self.trader_memory)

        # Create risk analysis nodes
        aggressive_analyst = create_aggressive_debator(self.quick_thinking_llm)
        neutral_analyst = create_neutral_debator(self.quick_thinking_llm)
        conservative_analyst = create_conservative_debator(self.quick_thinking_llm)
        risk_manager_node = create_risk_manager(
            self.deep_thinking_llm, self.risk_manager_memory
        )
        risk_metrics_computer = create_risk_metrics_node()

        # Add other nodes
        workflow.add_node("Bull Researcher", bull_researcher_node)
        workflow.add_node("Bear Researcher", bear_researcher_node)
        workflow.add_node("Research Manager", research_manager_node)
        workflow.add_node("Trader", trader_node)
        workflow.add_node("Risk Metrics Computer", risk_metrics_computer)
        workflow.add_node("Aggressive Analyst", aggressive_analyst)
        workflow.add_node("Neutral Analyst", neutral_analyst)
        workflow.add_node("Conservative Analyst", conservative_analyst)
        workflow.add_node("Risk Judge", risk_manager_node)

        # Connect last analyst to Bull Researcher (or END if standalone)
        if is_standalone:
            workflow.add_edge(last_node, END)
        else:
            workflow.add_edge(last_node, "Bull Researcher")

        # Add remaining edges
        workflow.add_conditional_edges(
            "Bull Researcher",
            self.conditional_logic.should_continue_debate,
            {
                "Bear Researcher": "Bear Researcher",
                "Research Manager": "Research Manager",
            },
        )
        workflow.add_conditional_edges(
            "Bear Researcher",
            self.conditional_logic.should_continue_debate,
            {
                "Bull Researcher": "Bull Researcher",
                "Research Manager": "Research Manager",
            },
        )
        workflow.add_edge("Research Manager", "Trader")
        workflow.add_edge("Trader", "Risk Metrics Computer")
        workflow.add_edge("Risk Metrics Computer", "Aggressive Analyst")
        workflow.add_conditional_edges(
            "Aggressive Analyst",
            self.conditional_logic.should_continue_risk_analysis,
            {
                "Conservative Analyst": "Conservative Analyst",
                "Risk Judge": "Risk Judge",
            },
        )
        workflow.add_conditional_edges(
            "Conservative Analyst",
            self.conditional_logic.should_continue_risk_analysis,
            {
                "Neutral Analyst": "Neutral Analyst",
                "Risk Judge": "Risk Judge",
            },
        )
        workflow.add_conditional_edges(
            "Neutral Analyst",
            self.conditional_logic.should_continue_risk_analysis,
            {
                "Aggressive Analyst": "Aggressive Analyst",
                "Risk Judge": "Risk Judge",
            },
        )

        workflow.add_edge("Risk Judge", END)

        # Compile and return
        return workflow.compile()
