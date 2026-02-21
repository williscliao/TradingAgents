"""LangChain @tool wrappers for valuation calculations.

These tools parse LLM-provided arguments, call the pure-math engine
in valuation_models.py, and return formatted markdown results.
"""

import json
from typing import Annotated
from langchain_core.tools import tool
from tradingagents.dataflows.valuation_models import (
    calc_capm,
    calc_wacc,
    calc_dcf,
    calc_relative_valuation,
)


@tool
def calculate_wacc(
    market_cap: Annotated[float, "Market capitalization in dollars"],
    total_debt: Annotated[float, "Total debt in dollars"],
    beta: Annotated[float, "Company beta"],
    risk_free_rate: Annotated[float, "Risk-free rate as decimal, e.g. 0.042 for 4.2%"],
    equity_risk_premium: Annotated[float, "Equity risk premium as decimal, e.g. 0.055 for 5.5%"],
    cost_of_debt: Annotated[float, "Pre-tax cost of debt as decimal (interest_expense / total_debt)"],
    tax_rate: Annotated[float, "Effective tax rate as decimal, e.g. 0.21 for 21%"],
) -> str:
    """Calculate WACC (Weighted Average Cost of Capital) using CAPM for cost of equity.
    Returns full breakdown with all components. Use this instead of doing the math yourself.
    """
    capm = calc_capm(risk_free_rate, beta, equity_risk_premium)
    wacc = calc_wacc(market_cap, total_debt, capm["cost_of_equity"], cost_of_debt, tax_rate)

    if wacc.get("error"):
        return f"ERROR: {wacc['error']}"

    lines = [
        "## WACC Calculation Results",
        "",
        "### Cost of Equity (CAPM)",
        f"- Risk-Free Rate: {risk_free_rate:.2%}",
        f"- Beta: {beta:.2f}",
        f"- Equity Risk Premium: {equity_risk_premium:.2%}",
        f"- **Cost of Equity = {capm['cost_of_equity']:.2%}**",
        "",
        "### Capital Structure",
        f"- Market Cap: ${market_cap:,.0f}",
        f"- Total Debt: ${total_debt:,.0f}",
        f"- Equity Weight: {wacc['equity_weight']:.1%}",
        f"- Debt Weight: {wacc['debt_weight']:.1%}",
        "",
        "### Cost of Debt",
        f"- Pre-tax: {cost_of_debt:.2%}",
        f"- Tax Rate: {tax_rate:.1%}",
        f"- After-tax: {wacc['cost_of_debt_aftertax']:.2%}",
        "",
        f"### **WACC = {wacc['wacc']:.2%}**",
        f"= ({wacc['equity_weight']:.1%} × {capm['cost_of_equity']:.2%}) + ({wacc['debt_weight']:.1%} × {wacc['cost_of_debt_aftertax']:.2%})",
    ]
    return "\n".join(lines)


@tool
def calculate_dcf(
    fcf_projections: Annotated[str, "Comma-separated FCF projections for each year, e.g. '1.2e9,1.4e9,1.5e9,1.6e9,1.7e9'"],
    terminal_growth: Annotated[float, "Long-term growth rate as decimal, e.g. 0.025 for 2.5%"],
    wacc: Annotated[float, "WACC as decimal, e.g. 0.09 for 9%. Get this from calculate_wacc first."],
    net_debt: Annotated[float, "Net debt = total debt - cash & equivalents, in dollars"],
    shares_outstanding: Annotated[float, "Diluted shares outstanding"],
) -> str:
    """Run a full DCF (Discounted Cash Flow) model with sensitivity analysis.
    Returns enterprise value, equity value, per-share intrinsic value, and a WACC × terminal growth sensitivity matrix.
    Always call calculate_wacc first to get the WACC input.
    """
    try:
        fcfs = [float(x.strip()) for x in fcf_projections.split(",")]
    except (ValueError, AttributeError):
        return f"ERROR: Cannot parse FCF projections. Provide comma-separated numbers, e.g. '1.2e9,1.4e9,1.5e9'"

    if len(fcfs) < 2:
        return "ERROR: Need at least 2 years of FCF projections"

    result = calc_dcf(fcfs, terminal_growth, wacc, net_debt, shares_outstanding)

    if result.get("error"):
        return f"ERROR: {result['error']}"

    # Format yearly detail table
    lines = [
        "## DCF Valuation Results",
        "",
        "### Projected Free Cash Flows",
        "| Year | FCF | Discount Factor | Present Value |",
        "|------|-----|----------------|---------------|",
    ]
    for yr in result["yearly_detail"]:
        lines.append(f"| {yr['year']} | ${yr['fcf']:,.0f} | {yr['discount_factor']:.4f} | ${yr['present_value']:,.0f} |")

    lines.extend([
        "",
        "### Valuation Summary",
        f"- PV of Projected FCFs: ${result['pv_of_fcfs']:,.0f}",
        f"- Terminal Value (undiscounted): ${result['terminal_value_undiscounted']:,.0f}",
        f"- PV of Terminal Value: ${result['pv_of_terminal_value']:,.0f}",
        f"- Terminal Value as % of EV: {result['terminal_value_pct_of_ev']:.1f}%",
        f"- **Enterprise Value: ${result['enterprise_value']:,.0f}**",
        f"- Net Debt: ${result['net_debt']:,.0f}",
        f"- **Equity Value: ${result['equity_value']:,.0f}**",
        f"- Shares Outstanding: {result['shares_outstanding']:,.0f}",
        f"- **Intrinsic Value Per Share: ${result['per_share_value']:.2f}**",
        "",
        "### Sensitivity Analysis (Per-Share Value)",
        f"WACC ↓ \\ Terminal Growth →",
    ])

    # Build sensitivity table
    if result["sensitivity_matrix"]:
        growth_keys = [k for k in result["sensitivity_matrix"][0] if k != "wacc"]
        header = "| WACC | " + " | ".join(growth_keys) + " |"
        sep = "|------|" + "|".join(["------"] * len(growth_keys)) + "|"
        lines.append(header)
        lines.append(sep)
        for row in result["sensitivity_matrix"]:
            wacc_str = f"{row['wacc']:.2%}"
            vals = []
            for gk in growth_keys:
                v = row[gk]
                vals.append(f"${v:,.2f}" if isinstance(v, (int, float)) else str(v))
            lines.append(f"| {wacc_str} | " + " | ".join(vals) + " |")

    return "\n".join(lines)


@tool
def calculate_relative_valuation(
    target_name: Annotated[str, "Target company name"],
    target_eps: Annotated[float, "Target company's adjusted EPS (0 if not profitable)"],
    target_ebitda: Annotated[float, "Target company's adjusted EBITDA in dollars (0 if N/A)"],
    target_revenue: Annotated[float, "Target company's annual revenue in dollars"],
    shares_outstanding: Annotated[float, "Target diluted shares outstanding"],
    net_debt: Annotated[float, "Target net debt (total debt - cash)"],
    peers_json: Annotated[str, "JSON array of peer companies. Each object must have 'name' and any of: 'pe', 'ev_ebitda', 'ps', 'pb'. Example: '[{\"name\":\"PEER1\",\"pe\":25,\"ev_ebitda\":15,\"ps\":5},{\"name\":\"PEER2\",\"pe\":30,\"ev_ebitda\":18,\"ps\":6}]'"],
) -> str:
    """Calculate implied share price using peer multiple comparison (Relative Valuation).
    Computes implied price from P/E, EV/EBITDA, EV/Revenue, and P/B methods using peer medians and averages.
    """
    try:
        peers = json.loads(peers_json)
    except (json.JSONDecodeError, TypeError):
        return "ERROR: Cannot parse peers_json. Provide a valid JSON array of peer objects."

    target_metrics = {}
    if target_eps and target_eps > 0:
        target_metrics["eps"] = target_eps
    if target_ebitda and target_ebitda > 0:
        target_metrics["ebitda"] = target_ebitda
    if target_revenue and target_revenue > 0:
        target_metrics["revenue"] = target_revenue

    if not target_metrics:
        return "ERROR: At least one of target_eps, target_ebitda, or target_revenue must be positive."

    result = calc_relative_valuation(
        target_name, target_metrics, peers, shares_outstanding, net_debt
    )

    if result.get("error"):
        return f"ERROR: {result['error']}"

    lines = [
        f"## Relative Valuation: {target_name}",
        "",
    ]

    if result["methods"]:
        lines.extend([
            "### Implied Share Price by Method",
            "| Method | Target Metric | Peer Median | Peer Avg | Implied (Median) | Implied (Avg) |",
            "|--------|--------------|-------------|----------|------------------|---------------|",
        ])
        for m in result["methods"]:
            lines.append(
                f"| {m['method']} | {m['target_metric']} | {m['peer_median']:.1f}x "
                f"| {m['peer_average']:.1f}x | ${m['implied_price_median']:,.2f} "
                f"| ${m['implied_price_average']:,.2f} |"
            )

    if result["blended_implied_price"]:
        lines.extend([
            "",
            f"### **Blended Implied Price (Median-based): ${result['blended_implied_price']:,.2f}**",
        ])

    # Peer table
    lines.extend(["", "### Peer Comparison", "| Company | P/E | EV/EBITDA | P/S | P/B |", "|---------|-----|-----------|-----|-----|"])
    for p in peers:
        pe = f"{p.get('pe', '-'):.1f}x" if p.get('pe') else "-"
        ev = f"{p.get('ev_ebitda', '-'):.1f}x" if p.get('ev_ebitda') else "-"
        ps = f"{p.get('ps', '-'):.1f}x" if p.get('ps') else "-"
        pb = f"{p.get('pb', '-'):.1f}x" if p.get('pb') else "-"
        lines.append(f"| {p.get('name', '?')} | {pe} | {ev} | {ps} | {pb} |")

    return "\n".join(lines)
