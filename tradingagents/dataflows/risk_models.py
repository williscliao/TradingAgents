"""Pure Python risk calculation engine.

All functions are deterministic math — no LLM dependency.
Calculates VaR, Sharpe Ratio, Max Drawdown, and other risk metrics
from historical price data.
"""

import math
from typing import Dict, List, Optional


def calc_daily_returns(prices: List[float]) -> List[float]:
    """Calculate daily returns from a price series.

    Args:
        prices: List of closing prices (oldest first)

    Returns:
        List of daily returns (length = len(prices) - 1)
    """
    if len(prices) < 2:
        return []
    return [(prices[i] / prices[i - 1]) - 1 for i in range(1, len(prices))]


def calc_var(
    prices: List[float],
    confidence: float = 0.95,
    holding_period_days: int = 1,
    investment: float = 100000.0,
) -> Dict:
    """Calculate Value at Risk using historical simulation.

    Args:
        prices: List of closing prices (oldest first, at least 30 data points)
        confidence: Confidence level (0.95 = 95%, 0.99 = 99%)
        holding_period_days: Holding period in trading days
        investment: Portfolio value in dollars

    Returns:
        Dict with VaR metrics at both confidence levels
    """
    returns = calc_daily_returns(prices)
    if len(returns) < 20:
        return {"error": "Need at least 20 daily returns for VaR calculation"}

    sorted_returns = sorted(returns)
    n = len(sorted_returns)

    results = {}
    for conf in [0.95, 0.99]:
        idx = int((1 - conf) * n)
        idx = max(0, min(idx, n - 1))
        daily_var_pct = abs(sorted_returns[idx])

        # Scale to holding period (square root of time)
        period_var_pct = daily_var_pct * math.sqrt(holding_period_days)
        var_dollar = period_var_pct * investment

        results[f"var_{int(conf*100)}"] = {
            "confidence": conf,
            "daily_var_pct": round(daily_var_pct * 100, 2),
            "period_var_pct": round(period_var_pct * 100, 2),
            "var_dollar": round(var_dollar, 2),
            "holding_period_days": holding_period_days,
        }

    # Conditional VaR (Expected Shortfall) at 95%
    cutoff_idx = int(0.05 * n)
    if cutoff_idx > 0:
        tail_losses = sorted_returns[:cutoff_idx]
        cvar_pct = abs(sum(tail_losses) / len(tail_losses))
        cvar_dollar = cvar_pct * math.sqrt(holding_period_days) * investment
    else:
        cvar_pct = abs(sorted_returns[0])
        cvar_dollar = cvar_pct * math.sqrt(holding_period_days) * investment

    results["cvar_95"] = {
        "daily_cvar_pct": round(cvar_pct * 100, 2),
        "cvar_dollar": round(cvar_dollar, 2),
    }

    results["data_points"] = n
    results["investment"] = investment

    return results


def calc_sharpe_ratio(
    prices: List[float],
    risk_free_annual: float = 0.042,
) -> Dict:
    """Calculate annualized Sharpe Ratio.

    Args:
        prices: List of closing prices (oldest first)
        risk_free_annual: Annual risk-free rate (default 4.2%)

    Returns:
        Dict with Sharpe ratio and components
    """
    returns = calc_daily_returns(prices)
    if len(returns) < 20:
        return {"error": "Need at least 20 daily returns for Sharpe calculation"}

    n = len(returns)
    mean_daily = sum(returns) / n
    variance = sum((r - mean_daily) ** 2 for r in returns) / (n - 1)
    std_daily = math.sqrt(variance)

    # Annualize
    trading_days = 252
    annual_return = mean_daily * trading_days
    annual_vol = std_daily * math.sqrt(trading_days)
    risk_free_daily = risk_free_annual / trading_days

    # Sharpe
    sharpe = (annual_return - risk_free_annual) / annual_vol if annual_vol > 0 else 0

    # Sortino (downside deviation only)
    downside_returns = [r for r in returns if r < risk_free_daily]
    if downside_returns:
        downside_var = sum((r - risk_free_daily) ** 2 for r in downside_returns) / len(downside_returns)
        downside_dev = math.sqrt(downside_var) * math.sqrt(trading_days)
        sortino = (annual_return - risk_free_annual) / downside_dev if downside_dev > 0 else 0
    else:
        sortino = float("inf")
        downside_dev = 0

    # Interpretation
    if sharpe >= 2.0:
        interpretation = "Excellent risk-adjusted returns"
    elif sharpe >= 1.0:
        interpretation = "Good risk-adjusted returns"
    elif sharpe >= 0.5:
        interpretation = "Acceptable risk-adjusted returns"
    elif sharpe >= 0:
        interpretation = "Poor risk-adjusted returns (barely beating risk-free)"
    else:
        interpretation = "Negative risk-adjusted returns (destroying value)"

    return {
        "sharpe_ratio": round(sharpe, 3),
        "sortino_ratio": round(sortino, 3) if sortino != float("inf") else "∞ (no downside)",
        "annualized_return": round(annual_return * 100, 2),
        "annualized_volatility": round(annual_vol * 100, 2),
        "downside_volatility": round(downside_dev * 100, 2),
        "risk_free_rate": round(risk_free_annual * 100, 2),
        "daily_mean_return": round(mean_daily * 100, 4),
        "daily_std": round(std_daily * 100, 4),
        "data_points": n,
        "interpretation": interpretation,
    }


def calc_max_drawdown(prices: List[float]) -> Dict:
    """Calculate Maximum Drawdown and drawdown statistics.

    Args:
        prices: List of closing prices (oldest first)

    Returns:
        Dict with max drawdown, drawdown duration, and current drawdown
    """
    if len(prices) < 2:
        return {"error": "Need at least 2 prices"}

    # Track running maximum and drawdowns
    peak = prices[0]
    max_dd = 0.0
    max_dd_peak_idx = 0
    max_dd_trough_idx = 0
    current_peak_idx = 0

    drawdowns = []

    for i, price in enumerate(prices):
        if price > peak:
            peak = price
            current_peak_idx = i

        dd = (peak - price) / peak if peak > 0 else 0
        drawdowns.append(dd)

        if dd > max_dd:
            max_dd = dd
            max_dd_peak_idx = current_peak_idx
            max_dd_trough_idx = i

    # Current drawdown (from most recent peak)
    current_dd = drawdowns[-1] if drawdowns else 0

    # Drawdown duration (days from peak to trough for max DD)
    dd_duration = max_dd_trough_idx - max_dd_peak_idx

    # Recovery: did it recover from max DD?
    recovered = False
    recovery_days = None
    if max_dd_trough_idx < len(prices) - 1:
        peak_price = prices[max_dd_peak_idx]
        for i in range(max_dd_trough_idx + 1, len(prices)):
            if prices[i] >= peak_price:
                recovered = True
                recovery_days = i - max_dd_trough_idx
                break

    # Average drawdown
    avg_dd = sum(drawdowns) / len(drawdowns) if drawdowns else 0

    # Count significant drawdowns (> 5%)
    in_dd = False
    dd_count = 0
    for dd in drawdowns:
        if dd > 0.05 and not in_dd:
            dd_count += 1
            in_dd = True
        elif dd < 0.02:
            in_dd = False

    # Risk interpretation
    if max_dd < 0.10:
        risk_level = "LOW — drawdowns contained under 10%"
    elif max_dd < 0.20:
        risk_level = "MODERATE — drawdowns up to 20%"
    elif max_dd < 0.35:
        risk_level = "HIGH — significant drawdown experienced"
    else:
        risk_level = "VERY HIGH — severe drawdown, potential capital destruction"

    return {
        "max_drawdown_pct": round(max_dd * 100, 2),
        "max_dd_peak_price": round(prices[max_dd_peak_idx], 2),
        "max_dd_trough_price": round(prices[max_dd_trough_idx], 2),
        "max_dd_duration_days": dd_duration,
        "recovered": recovered,
        "recovery_days": recovery_days,
        "current_drawdown_pct": round(current_dd * 100, 2),
        "average_drawdown_pct": round(avg_dd * 100, 2),
        "significant_drawdown_count": dd_count,
        "risk_level": risk_level,
        "data_points": len(prices),
    }


def calc_risk_profile(prices: List[float], risk_free_annual: float = 0.042) -> Dict:
    """Calculate complete risk profile combining VaR, Sharpe, and Drawdown.

    This is the main entry point — produces a full risk dashboard.

    Args:
        prices: List of closing prices (oldest first, at least 30 points)
        risk_free_annual: Annual risk-free rate

    Returns:
        Dict with all risk metrics combined
    """
    var = calc_var(prices)
    sharpe = calc_sharpe_ratio(prices, risk_free_annual)
    drawdown = calc_max_drawdown(prices)

    # Overall risk score (1-10, 10 = highest risk)
    risk_score = 5  # baseline

    if not isinstance(var, dict) or var.get("error"):
        return {"error": var.get("error", "VaR calculation failed")}

    # Adjust by volatility
    vol = sharpe.get("annualized_volatility", 20)
    if vol > 50:
        risk_score += 2
    elif vol > 35:
        risk_score += 1
    elif vol < 15:
        risk_score -= 1

    # Adjust by max drawdown
    mdd = drawdown.get("max_drawdown_pct", 20)
    if mdd > 40:
        risk_score += 2
    elif mdd > 25:
        risk_score += 1
    elif mdd < 10:
        risk_score -= 1

    # Adjust by current drawdown
    curr_dd = drawdown.get("current_drawdown_pct", 0)
    if curr_dd > 15:
        risk_score += 1

    risk_score = max(1, min(10, risk_score))

    return {
        "var": var,
        "sharpe": sharpe,
        "drawdown": drawdown,
        "overall_risk_score": risk_score,
    }
