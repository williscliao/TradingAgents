"""Pre-compute quantitative risk metrics before the risk debate.

This node runs between the Trader and the risk debate analysts,
fetching price data and computing VaR, Sharpe, and Drawdown metrics
that are then injected into state for the debaters and risk judge.
"""

import logging

logger = logging.getLogger(__name__)


def create_risk_metrics_node():
    """Create a node that pre-computes quantitative risk metrics."""

    def risk_metrics_node(state) -> dict:
        ticker = state["company_of_interest"]

        try:
            import yfinance as yf
            from datetime import datetime, timedelta
            from tradingagents.dataflows.risk_models import calc_risk_profile

            # Fetch ~1 year of price data
            end = datetime.now()
            start = end - timedelta(days=400)

            stock = yf.Ticker(ticker)
            hist = stock.history(
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
            )

            if hist.empty or len(hist) < 30:
                return {
                    "risk_metrics_report": f"⚠️ Insufficient price data for {ticker} to compute risk metrics. Proceed with qualitative assessment only."
                }

            prices = hist["Close"].tolist()
            profile = calc_risk_profile(prices)

            if "error" in profile:
                return {
                    "risk_metrics_report": f"⚠️ Risk calculation error: {profile['error']}. Proceed with qualitative assessment only."
                }

            # Format a concise report for the debaters
            var = profile["var"]
            sharpe = profile["sharpe"]
            dd = profile["drawdown"]
            risk_score = profile["overall_risk_score"]
            investment = 100000

            v95 = var.get("var_95", {})
            v99 = var.get("var_99", {})
            cvar = var.get("cvar_95", {})

            dollar_95 = v95.get("daily_var_pct", 0) / 100 * investment
            dollar_99 = v99.get("daily_var_pct", 0) / 100 * investment
            cvar_dollar = cvar.get("daily_cvar_pct", 0) / 100 * investment

            risk_bar = "█" * risk_score + "░" * (10 - risk_score)

            report = f"""# Quantitative Risk Profile: {ticker.upper()}
**Period**: {len(prices)} trading days (~1 year) | **Reference Investment**: $100,000

## Value at Risk (Historical Simulation)

| Metric | 95% Confidence | 99% Confidence |
|--------|:--------------:|:--------------:|
| Daily VaR | {v95.get('daily_var_pct', 'N/A')}% | {v99.get('daily_var_pct', 'N/A')}% |
| Daily VaR ($) | ${dollar_95:,.0f} | ${dollar_99:,.0f} |

**CVaR (Expected Shortfall) at 95%**: {cvar.get('daily_cvar_pct', 'N/A')}% (${cvar_dollar:,.0f})
_If losses exceed VaR, the expected average loss is this amount._

## Risk-Adjusted Returns

| Metric | Value |
|--------|------:|
| Annualized Return | {sharpe.get('annualized_return', 'N/A')}% |
| Annualized Volatility | {sharpe.get('annualized_volatility', 'N/A')}% |
| **Sharpe Ratio** | **{sharpe.get('sharpe_ratio', 'N/A')}** |
| **Sortino Ratio** | **{sharpe.get('sortino_ratio', 'N/A')}** |
| Downside Volatility | {sharpe.get('downside_volatility', 'N/A')}% |

_Interpretation: {sharpe.get('interpretation', 'N/A')}_

## Maximum Drawdown

| Metric | Value |
|--------|------:|
| **Max Drawdown** | **-{dd.get('max_drawdown_pct', 'N/A')}%** |
| Peak → Trough | ${dd.get('max_dd_peak_price', 'N/A')} → ${dd.get('max_dd_trough_price', 'N/A')} |
| Duration | {dd.get('max_dd_duration_days', 'N/A')} trading days |
| Recovery | {'Yes (' + str(dd.get('recovery_days', '')) + ' days)' if dd.get('recovered') else 'Not yet recovered'} |
| Current Drawdown | -{dd.get('current_drawdown_pct', 'N/A')}% |
| Significant DD Events (>5%) | {dd.get('significant_drawdown_count', 'N/A')} |

**Risk Level**: {dd.get('risk_level', 'N/A')}

## Overall Risk Score: {risk_bar} {risk_score}/10
"""

            return {"risk_metrics_report": report}

        except Exception as e:
            logger.error(f"Risk metrics computation failed: {e}")
            return {
                "risk_metrics_report": f"⚠️ Risk metrics unavailable ({type(e).__name__}: {e}). Proceed with qualitative assessment only."
            }

    return risk_metrics_node
