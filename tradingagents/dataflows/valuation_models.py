"""Pure Python valuation calculation engine.

All functions are deterministic math — no LLM dependency.
Can be independently unit-tested with known inputs/outputs.
"""

from typing import Dict, List, Optional


def calc_capm(
    risk_free_rate: float,
    beta: float,
    equity_risk_premium: float = 0.055,
) -> Dict[str, float]:
    """Calculate Cost of Equity using CAPM.

    Args:
        risk_free_rate: Risk-free rate (e.g., 0.042 for 4.2%)
        beta: Company beta
        equity_risk_premium: Market risk premium (default 5.5%)

    Returns:
        Dict with cost_of_equity and component breakdown
    """
    cost_of_equity = risk_free_rate + beta * equity_risk_premium
    return {
        "cost_of_equity": round(cost_of_equity, 6),
        "risk_free_rate": risk_free_rate,
        "beta": beta,
        "equity_risk_premium": equity_risk_premium,
    }


def calc_wacc(
    market_cap: float,
    total_debt: float,
    cost_of_equity: float,
    cost_of_debt: float,
    tax_rate: float,
) -> Dict[str, float]:
    """Calculate Weighted Average Cost of Capital.

    Args:
        market_cap: Market capitalization (equity value)
        total_debt: Total debt
        cost_of_equity: From CAPM
        cost_of_debt: Interest expense / total debt
        tax_rate: Effective corporate tax rate (e.g., 0.21)

    Returns:
        Dict with WACC and all components
    """
    total_capital = market_cap + total_debt
    if total_capital <= 0:
        return {"error": "Total capital must be positive", "wacc": 0.0}

    equity_weight = market_cap / total_capital
    debt_weight = total_debt / total_capital
    after_tax_cost_of_debt = cost_of_debt * (1 - tax_rate)
    wacc = equity_weight * cost_of_equity + debt_weight * after_tax_cost_of_debt

    return {
        "wacc": round(wacc, 6),
        "equity_weight": round(equity_weight, 4),
        "debt_weight": round(debt_weight, 4),
        "cost_of_equity": round(cost_of_equity, 6),
        "cost_of_debt_pretax": round(cost_of_debt, 6),
        "cost_of_debt_aftertax": round(after_tax_cost_of_debt, 6),
        "tax_rate": tax_rate,
        "market_cap": market_cap,
        "total_debt": total_debt,
    }


def calc_dcf(
    fcf_projections: List[float],
    terminal_growth: float,
    wacc: float,
    net_debt: float,
    shares_outstanding: float,
    projection_labels: Optional[List[str]] = None,
) -> Dict:
    """Run a full DCF valuation model.

    Args:
        fcf_projections: List of projected Free Cash Flows (e.g., 5 years)
        terminal_growth: Long-term growth rate (e.g., 0.025 for 2.5%)
        wacc: Weighted Average Cost of Capital
        net_debt: Total debt minus cash & equivalents
        shares_outstanding: Diluted shares outstanding

    Returns:
        Dict with enterprise_value, equity_value, per_share_value,
        yearly breakdown, and sensitivity matrix.
    """
    if wacc <= terminal_growth:
        return {"error": f"WACC ({wacc:.2%}) must be greater than terminal growth ({terminal_growth:.2%})"}

    if shares_outstanding <= 0:
        return {"error": "Shares outstanding must be positive"}

    n = len(fcf_projections)
    labels = projection_labels or [f"Year {i+1}" for i in range(n)]

    # Discount projected FCFs
    pv_fcfs = []
    yearly_detail = []
    for i, fcf in enumerate(fcf_projections):
        discount_factor = (1 + wacc) ** (i + 1)
        pv = fcf / discount_factor
        pv_fcfs.append(pv)
        yearly_detail.append({
            "year": labels[i],
            "fcf": round(fcf, 2),
            "discount_factor": round(discount_factor, 4),
            "present_value": round(pv, 2),
        })

    sum_pv_fcf = sum(pv_fcfs)

    # Terminal value (Gordon Growth Model)
    terminal_fcf = fcf_projections[-1] * (1 + terminal_growth)
    terminal_value = terminal_fcf / (wacc - terminal_growth)
    pv_terminal = terminal_value / ((1 + wacc) ** n)

    # Enterprise and equity value
    enterprise_value = sum_pv_fcf + pv_terminal
    equity_value = enterprise_value - net_debt
    per_share_value = equity_value / shares_outstanding

    # Sensitivity analysis: WACC ± 1% × Terminal Growth ± 0.5%
    wacc_range = [wacc - 0.01, wacc - 0.005, wacc, wacc + 0.005, wacc + 0.01]
    growth_range = [terminal_growth - 0.005, terminal_growth, terminal_growth + 0.005]

    sensitivity = []
    for w in wacc_range:
        row = {"wacc": round(w, 4)}
        for g in growth_range:
            if w <= g:
                row[f"g={g:.1%}"] = "N/A"
                continue
            tv = fcf_projections[-1] * (1 + g) / (w - g)
            pv_tv = tv / ((1 + w) ** n)
            # Recalculate PV of FCFs with this WACC
            spv = sum(fcf / ((1 + w) ** (i + 1)) for i, fcf in enumerate(fcf_projections))
            ev = spv + pv_tv
            eqv = ev - net_debt
            ps = eqv / shares_outstanding
            row[f"g={g:.1%}"] = round(ps, 2)
        sensitivity.append(row)

    return {
        "enterprise_value": round(enterprise_value, 2),
        "pv_of_fcfs": round(sum_pv_fcf, 2),
        "terminal_value_undiscounted": round(terminal_value, 2),
        "pv_of_terminal_value": round(pv_terminal, 2),
        "terminal_value_pct_of_ev": round(pv_terminal / enterprise_value * 100, 1) if enterprise_value > 0 else 0,
        "net_debt": round(net_debt, 2),
        "equity_value": round(equity_value, 2),
        "shares_outstanding": shares_outstanding,
        "per_share_value": round(per_share_value, 2),
        "wacc_used": round(wacc, 4),
        "terminal_growth_used": round(terminal_growth, 4),
        "yearly_detail": yearly_detail,
        "sensitivity_matrix": sensitivity,
    }


def calc_relative_valuation(
    target_name: str,
    target_metrics: Dict[str, float],
    peers: List[Dict[str, float]],
    shares_outstanding: float,
    net_debt: float = 0.0,
) -> Dict:
    """Calculate implied share price from peer multiples.

    Args:
        target_name: Target company name
        target_metrics: Dict with keys like 'eps', 'ebitda', 'revenue', 'book_value'
        peers: List of dicts, each with 'name' and multiple keys like 'pe', 'ev_ebitda', 'ps'
        shares_outstanding: Diluted shares outstanding
        net_debt: Total debt minus cash (for EV-based multiples)

    Returns:
        Dict with implied prices from each method and peer comparison table
    """
    if shares_outstanding <= 0:
        return {"error": "Shares outstanding must be positive"}

    results = []

    # P/E method
    if target_metrics.get("eps") and target_metrics["eps"] > 0:
        pe_multiples = [p["pe"] for p in peers if p.get("pe") and p["pe"] > 0]
        if pe_multiples:
            median_pe = sorted(pe_multiples)[len(pe_multiples) // 2]
            avg_pe = sum(pe_multiples) / len(pe_multiples)
            results.append({
                "method": "P/E",
                "target_metric": f"EPS = ${target_metrics['eps']:.2f}",
                "peer_median": round(median_pe, 1),
                "peer_average": round(avg_pe, 1),
                "implied_price_median": round(target_metrics["eps"] * median_pe, 2),
                "implied_price_average": round(target_metrics["eps"] * avg_pe, 2),
            })

    # EV/EBITDA method
    if target_metrics.get("ebitda") and target_metrics["ebitda"] > 0:
        ev_multiples = [p["ev_ebitda"] for p in peers if p.get("ev_ebitda") and p["ev_ebitda"] > 0]
        if ev_multiples:
            median_ev = sorted(ev_multiples)[len(ev_multiples) // 2]
            avg_ev = sum(ev_multiples) / len(ev_multiples)
            implied_ev_med = target_metrics["ebitda"] * median_ev
            implied_ev_avg = target_metrics["ebitda"] * avg_ev
            results.append({
                "method": "EV/EBITDA",
                "target_metric": f"EBITDA = ${target_metrics['ebitda']:,.0f}",
                "peer_median": round(median_ev, 1),
                "peer_average": round(avg_ev, 1),
                "implied_price_median": round((implied_ev_med - net_debt) / shares_outstanding, 2),
                "implied_price_average": round((implied_ev_avg - net_debt) / shares_outstanding, 2),
            })

    # P/S (EV/Revenue) method
    if target_metrics.get("revenue") and target_metrics["revenue"] > 0:
        ps_multiples = [p["ps"] for p in peers if p.get("ps") and p["ps"] > 0]
        if ps_multiples:
            median_ps = sorted(ps_multiples)[len(ps_multiples) // 2]
            avg_ps = sum(ps_multiples) / len(ps_multiples)
            implied_ev_med = target_metrics["revenue"] * median_ps
            implied_ev_avg = target_metrics["revenue"] * avg_ps
            results.append({
                "method": "EV/Revenue",
                "target_metric": f"Revenue = ${target_metrics['revenue']:,.0f}",
                "peer_median": round(median_ps, 1),
                "peer_average": round(avg_ps, 1),
                "implied_price_median": round((implied_ev_med - net_debt) / shares_outstanding, 2),
                "implied_price_average": round((implied_ev_avg - net_debt) / shares_outstanding, 2),
            })

    # P/B method
    if target_metrics.get("book_value") and target_metrics["book_value"] > 0:
        pb_multiples = [p["pb"] for p in peers if p.get("pb") and p["pb"] > 0]
        if pb_multiples:
            median_pb = sorted(pb_multiples)[len(pb_multiples) // 2]
            avg_pb = sum(pb_multiples) / len(pb_multiples)
            bvps = target_metrics["book_value"] / shares_outstanding
            results.append({
                "method": "P/B",
                "target_metric": f"BVPS = ${bvps:.2f}",
                "peer_median": round(median_pb, 1),
                "peer_average": round(avg_pb, 1),
                "implied_price_median": round(bvps * median_pb, 2),
                "implied_price_average": round(bvps * avg_pb, 2),
            })

    # Blended average (median-based)
    median_prices = [r["implied_price_median"] for r in results if r.get("implied_price_median")]
    blended = round(sum(median_prices) / len(median_prices), 2) if median_prices else None

    return {
        "target": target_name,
        "methods": results,
        "blended_implied_price": blended,
        "peer_comparison": peers,
    }
