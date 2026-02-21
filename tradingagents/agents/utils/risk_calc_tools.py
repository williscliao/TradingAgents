"""LangChain @tool wrappers for risk calculations.

These tools fetch price data via yfinance and pass it
to the pure-math functions in risk_models.py.
They can be used by agents that need quantitative risk metrics.
"""

from langchain_core.tools import tool
from typing import Annotated
import logging

logger = logging.getLogger(__name__)


@tool
def calculate_risk_metrics(
    ticker: Annotated[str, "Stock ticker symbol, e.g. AAPL"],
    look_back_days: Annotated[int, "Number of trading days to analyze (90, 180, or 252 recommended)"] = 252,
    investment: Annotated[float, "Portfolio investment amount in dollars"] = 100000.0,
) -> str:
    """
    Calculate a comprehensive risk profile for a stock:
    - Value at Risk (VaR) at 95% and 99% confidence
    - Conditional VaR (Expected Shortfall)
    - Sharpe Ratio and Sortino Ratio
    - Maximum Drawdown with duration and recovery stats
    - Overall Risk Score (1-10 scale)

    Args:
        ticker: Stock symbol
        look_back_days: How many trading days of history to use (default 252 = 1 year)
        investment: Dollar amount to calculate VaR against (default $100,000)

    Returns:
        Formatted risk report as markdown string
    """
    try:
        import yfinance as yf
        from datetime import datetime, timedelta

        # Fetch price data
        end = datetime.now()
        # Request extra days to account for weekends/holidays
        start = end - timedelta(days=int(look_back_days * 1.5))

        stock = yf.Ticker(ticker)
        hist = stock.history(start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"))

        if hist.empty or len(hist) < 20:
            return f"ERROR [calculate_risk_metrics]: Insufficient price data for {ticker}. Need at least 20 trading days."

        prices = hist["Close"].tolist()
        # Trim to requested look_back_days
        if len(prices) > look_back_days:
            prices = prices[-look_back_days:]

        from tradingagents.dataflows.risk_models import (
            calc_var,
            calc_sharpe_ratio,
            calc_max_drawdown,
            calc_risk_profile,
        )

        profile = calc_risk_profile(prices)

        if "error" in profile:
            return f"ERROR [calculate_risk_metrics]: {profile['error']}"

        var = profile["var"]
        sharpe = profile["sharpe"]
        dd = profile["drawdown"]
        risk_score = profile["overall_risk_score"]

        # Format report
        lines = [
            f"# Quantitative Risk Profile: {ticker.upper()}",
            f"**Period**: {len(prices)} trading days | **Investment**: ${investment:,.0f}",
            "",
            "## Value at Risk (Historical Simulation)",
            "",
            "| Metric | 95% Confidence | 99% Confidence |",
            "|--------|:--------------:|:--------------:|",
        ]

        v95 = var.get("var_95", {})
        v99 = var.get("var_99", {})
        cvar = var.get("cvar_95", {})

        lines.append(
            f"| Daily VaR | {v95.get('daily_var_pct', 'N/A')}% | {v99.get('daily_var_pct', 'N/A')}% |"
        )

        # Recalculate dollar VaR with user's investment amount
        if v95.get("daily_var_pct"):
            dollar_95 = v95["daily_var_pct"] / 100 * investment
            dollar_99 = v99.get("daily_var_pct", 0) / 100 * investment
            lines.append(
                f"| Daily VaR ($) | ${dollar_95:,.0f} | ${dollar_99:,.0f} |"
            )

        lines.append("")
        if cvar.get("daily_cvar_pct"):
            cvar_dollar = cvar["daily_cvar_pct"] / 100 * investment
            lines.append(
                f"**CVaR (Expected Shortfall) at 95%**: {cvar['daily_cvar_pct']}% (${cvar_dollar:,.0f})"
            )
            lines.append(
                "_If losses exceed VaR, the average loss would be this amount._"
            )
        lines.append("")

        # Sharpe / Sortino
        lines.extend([
            "## Risk-Adjusted Returns",
            "",
            "| Metric | Value |",
            "|--------|------:|",
            f"| Annualized Return | {sharpe.get('annualized_return', 'N/A')}% |",
            f"| Annualized Volatility | {sharpe.get('annualized_volatility', 'N/A')}% |",
            f"| **Sharpe Ratio** | **{sharpe.get('sharpe_ratio', 'N/A')}** |",
            f"| **Sortino Ratio** | **{sharpe.get('sortino_ratio', 'N/A')}** |",
            f"| Downside Volatility | {sharpe.get('downside_volatility', 'N/A')}% |",
            f"| Risk-Free Rate | {sharpe.get('risk_free_rate', 'N/A')}% |",
            "",
            f"_Interpretation: {sharpe.get('interpretation', 'N/A')}_",
            "",
        ])

        # Max Drawdown
        lines.extend([
            "## Maximum Drawdown Analysis",
            "",
            "| Metric | Value |",
            "|--------|------:|",
            f"| **Max Drawdown** | **-{dd.get('max_drawdown_pct', 'N/A')}%** |",
            f"| Peak Price | ${dd.get('max_dd_peak_price', 'N/A')} |",
            f"| Trough Price | ${dd.get('max_dd_trough_price', 'N/A')} |",
            f"| Drawdown Duration | {dd.get('max_dd_duration_days', 'N/A')} trading days |",
            f"| Recovered | {'Yes' if dd.get('recovered') else 'No'} |",
        ])
        if dd.get("recovery_days"):
            lines.append(f"| Recovery Time | {dd['recovery_days']} trading days |")
        lines.extend([
            f"| Current Drawdown | -{dd.get('current_drawdown_pct', 'N/A')}% |",
            f"| Avg Drawdown | -{dd.get('average_drawdown_pct', 'N/A')}% |",
            f"| Significant Drawdowns (>5%) | {dd.get('significant_drawdown_count', 'N/A')} |",
            "",
            f"**Risk Level**: {dd.get('risk_level', 'N/A')}",
            "",
        ])

        # Overall
        risk_bar = "█" * risk_score + "░" * (10 - risk_score)
        lines.extend([
            "## Overall Risk Score",
            "",
            f"**{risk_bar}  {risk_score}/10**",
            "",
            "| Score Range | Meaning |",
            "|:----------:|---------|",
            "| 1-3 | Low risk — stable, low volatility |",
            "| 4-6 | Moderate risk — normal equity characteristics |",
            "| 7-8 | High risk — elevated volatility and/or drawdowns |",
            "| 9-10 | Very high risk — extreme volatility, potential capital loss |",
        ])

        return "\n".join(lines)

    except ImportError as e:
        return f"ERROR [calculate_risk_metrics]: Missing dependency: {e}"
    except Exception as e:
        return f"ERROR [calculate_risk_metrics]: {type(e).__name__}: {e}"
