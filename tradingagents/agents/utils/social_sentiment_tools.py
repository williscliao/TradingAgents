import os
import requests
from datetime import datetime, timedelta
from langchain_core.tools import tool
from typing import Annotated


@tool
def get_social_sentiment(
    ticker: Annotated[str, "Ticker symbol (e.g., AAPL, MU)"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
) -> str:
    """
    Retrieve market sentiment indicators for a given ticker from Finnhub.
    Includes: analyst recommendation trends, earnings surprise history,
    and insider sentiment data. This gives a comprehensive view of how
    Wall Street, corporate insiders, and the market perceive the stock.

    Args:
        ticker: Stock ticker symbol
        end_date: Current/end date in yyyy-mm-dd format

    Returns:
        str: A formatted report of sentiment indicators
    """
    api_key = os.environ.get("FINNHUB_API_KEY", "")
    if not api_key:
        return "Error: FINNHUB_API_KEY environment variable not set."

    base = "https://finnhub.io/api/v1"
    symbol = ticker.upper()
    report_lines = [
        f"# Market Sentiment Indicators for {symbol}",
        f"**As of**: {end_date}",
        "",
    ]

    # 1. Analyst Recommendation Trends
    try:
        resp = requests.get(
            f"{base}/stock/recommendation",
            params={"symbol": symbol, "token": api_key},
            timeout=15,
        )
        resp.raise_for_status()
        recs = resp.json()

        if recs:
            report_lines.extend([
                "## Analyst Recommendation Trends",
                "Wall Street analyst consensus over recent months:",
                "",
                "| Period | Strong Buy | Buy | Hold | Sell | Strong Sell | Total | Bull % |",
                "|--------|-----------|-----|------|------|------------|-------|--------|",
            ])

            for r in recs[:6]:  # Last 6 months
                sb = r.get("strongBuy", 0)
                b = r.get("buy", 0)
                h = r.get("hold", 0)
                s = r.get("sell", 0)
                ss = r.get("strongSell", 0)
                total = sb + b + h + s + ss
                bull_pct = (sb + b) / total * 100 if total > 0 else 0
                report_lines.append(
                    f"| {r.get('period', '')[:7]} | {sb} | {b} | {h} | {s} | {ss} | {total} | {bull_pct:.0f}% |"
                )

            # Trend analysis
            if len(recs) >= 2:
                latest = recs[0]
                previous = recs[1]
                latest_bull = latest.get("strongBuy", 0) + latest.get("buy", 0)
                prev_bull = previous.get("strongBuy", 0) + previous.get("buy", 0)
                latest_bear = latest.get("sell", 0) + latest.get("strongSell", 0)
                prev_bear = previous.get("sell", 0) + previous.get("strongSell", 0)

                if latest_bull > prev_bull:
                    trend = "**Bullish momentum increasing** — more analysts upgrading"
                elif latest_bull < prev_bull:
                    trend = "**Bullish momentum decreasing** — analysts downgrading"
                else:
                    trend = "**Consensus stable** — no significant changes"

                report_lines.extend(["", f"**Trend**: {trend}", ""])
        else:
            report_lines.append("## Analyst Recommendations\nNo data available.\n")

    except requests.RequestException as e:
        report_lines.append(f"## Analyst Recommendations\nError fetching data: {e}\n")

    # 2. Earnings Surprises
    try:
        resp = requests.get(
            f"{base}/stock/earnings",
            params={"symbol": symbol, "token": api_key},
            timeout=15,
        )
        resp.raise_for_status()
        earnings = resp.json()

        if earnings:
            report_lines.extend([
                "## Earnings Surprise History",
                "How the company has performed vs. analyst expectations:",
                "",
                "| Quarter | Actual EPS | Estimate | Surprise | Surprise % |",
                "|---------|-----------|----------|----------|------------|",
            ])

            beat_count = 0
            for e in earnings[:8]:  # Last 8 quarters
                actual = e.get("actual", 0) or 0
                est = e.get("estimate", 0) or 0
                surprise = e.get("surprise", 0) or 0
                surprise_pct = e.get("surprisePercent", 0) or 0
                period = e.get("period", "")

                if surprise > 0:
                    beat_count += 1

                emoji = "✅" if surprise > 0 else "❌" if surprise < 0 else "➖"
                report_lines.append(
                    f"| {period} | ${actual:.2f} | ${est:.2f} | {emoji} ${surprise:+.3f} | {surprise_pct:+.1f}% |"
                )

            total_quarters = min(len(earnings), 8)
            report_lines.extend([
                "",
                f"**Beat Rate**: {beat_count}/{total_quarters} quarters ({beat_count/total_quarters*100:.0f}%)",
                "",
            ])

            # Average surprise
            avg_surprise = sum((e.get("surprisePercent", 0) or 0) for e in earnings[:8]) / total_quarters
            if avg_surprise > 5:
                report_lines.append(f"**Pattern**: Consistently beating estimates by avg {avg_surprise:.1f}% — strong execution")
            elif avg_surprise > 0:
                report_lines.append(f"**Pattern**: Slight beat tendency (avg +{avg_surprise:.1f}%) — meeting expectations")
            else:
                report_lines.append(f"**Pattern**: Missing estimates (avg {avg_surprise:.1f}%) — execution concerns")
            report_lines.append("")
        else:
            report_lines.append("## Earnings Surprise History\nNo data available.\n")

    except requests.RequestException as e:
        report_lines.append(f"## Earnings Surprise History\nError fetching data: {e}\n")

    # 3. Insider Sentiment
    try:
        start_dt = datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=365)
        start_str = start_dt.strftime("%Y-%m-%d")

        resp = requests.get(
            f"{base}/stock/insider-sentiment",
            params={"symbol": symbol, "from": start_str, "to": end_date, "token": api_key},
            timeout=15,
        )
        resp.raise_for_status()
        insider = resp.json()
        insider_data = insider.get("data", [])

        if insider_data:
            report_lines.extend([
                "## Insider Sentiment (MSPR)",
                "Monthly Share Purchase Ratio — positive = net buying, negative = net selling:",
                "",
                "| Month | Year | MSPR Change | # Transactions |",
                "|-------|------|-------------|----------------|",
            ])

            for d in insider_data[-6:]:  # Last 6 months
                mspr = d.get("mspr", 0)
                change = d.get("change", 0)
                month = d.get("month", "")
                year = d.get("year", "")
                emoji = "🟢" if mspr > 0 else "🔴" if mspr < 0 else "⚪"
                report_lines.append(
                    f"| {month} | {year} | {emoji} {mspr:+.2f} | {change} |"
                )

            # Insider trend
            recent_mspr = [d.get("mspr", 0) for d in insider_data[-3:]]
            avg_mspr = sum(recent_mspr) / len(recent_mspr) if recent_mspr else 0
            if avg_mspr > 0:
                report_lines.append("\n**Insider Signal**: Net BUYING — insiders are bullish")
            elif avg_mspr < 0:
                report_lines.append("\n**Insider Signal**: Net SELLING — insiders reducing exposure")
            else:
                report_lines.append("\n**Insider Signal**: Neutral activity")
            report_lines.append("")
        else:
            report_lines.append("## Insider Sentiment\nNo insider sentiment data available for this period.\n")

    except requests.RequestException as e:
        report_lines.append(f"## Insider Sentiment\nError fetching data: {e}\n")

    return "\n".join(report_lines)


GRADE_MAP = {1: "A+", 2: "A", 3: "B", 4: "C", 5: "D", 6: "F"}


@tool
def get_quant_grades(
    ticker: Annotated[str, "Ticker symbol (e.g., AAPL, MU, NVDA)"],
) -> str:
    """
    Retrieve Seeking Alpha Quant Factor Grades for a given ticker.
    Returns 5-dimension quantitative ratings: Value, Growth, Profitability,
    Momentum, and EPS Revisions. Each is graded A+ to F based on 100+ metrics.
    This is a powerful multi-factor stock scoring system.

    Args:
        ticker: Stock ticker symbol

    Returns:
        str: A formatted report of quant factor grades
    """
    api_key = os.environ.get("RAPIDAPI_KEY", "")
    if not api_key:
        return "Error: RAPIDAPI_KEY environment variable not set."

    try:
        resp = requests.get(
            "https://seeking-alpha-api.p.rapidapi.com/metrics-grades",
            params={"slugs": ticker.upper()},
            headers={
                "x-rapidapi-host": "seeking-alpha-api.p.rapidapi.com",
                "x-rapidapi-key": api_key,
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        return f"Error fetching Seeking Alpha quant grades: {e}"

    grades_list = data.get("metrics_grades", [])
    if not grades_list:
        return f"No Seeking Alpha quant grades available for {ticker.upper()}."

    g = grades_list[0]
    value = g.get("value_category", 0)
    growth = g.get("growth_category", 0)
    profit = g.get("profitability_category", 0)
    momentum = g.get("momentum_category", 0)
    eps_rev = g.get("eps_revisions_category", 0)

    factors = {
        "Value": value,
        "Growth": growth,
        "Profitability": profit,
        "Momentum": momentum,
        "EPS Revisions": eps_rev,
    }

    report_lines = [
        f"# Seeking Alpha Quant Factor Grades for {ticker.upper()}",
        "",
        "| Factor | Grade | Score | Meaning |",
        "|--------|-------|-------|---------|",
    ]

    meanings = {
        "Value": {
            1: "Extremely undervalued", 2: "Undervalued", 3: "Fair value",
            4: "Slightly overvalued", 5: "Overvalued", 6: "Very expensive",
        },
        "Growth": {
            1: "Exceptional growth", 2: "Strong growth", 3: "Moderate growth",
            4: "Below-avg growth", 5: "Weak growth", 6: "Declining",
        },
        "Profitability": {
            1: "Highly profitable", 2: "Strong margins", 3: "Decent margins",
            4: "Below-avg margins", 5: "Weak profitability", 6: "Unprofitable",
        },
        "Momentum": {
            1: "Very strong momentum", 2: "Strong momentum", 3: "Moderate momentum",
            4: "Weak momentum", 5: "Negative momentum", 6: "Sharp decline",
        },
        "EPS Revisions": {
            1: "Strong upward revisions", 2: "Upward revisions", 3: "Stable estimates",
            4: "Slight downgrades", 5: "Downward revisions", 6: "Sharp downgrades",
        },
    }

    strong_count = 0
    weak_count = 0

    for factor, score in factors.items():
        grade = GRADE_MAP.get(score, "N/A")
        meaning = meanings.get(factor, {}).get(score, "")
        emoji = "🟢" if score <= 2 else "🟡" if score <= 4 else "🔴"
        report_lines.append(f"| {factor} | {emoji} {grade} | {score}/6 | {meaning} |")
        if score <= 2:
            strong_count += 1
        if score >= 5:
            weak_count += 1

    report_lines.append("")

    # Overall assessment
    if strong_count >= 4:
        overall = "**Overall: Strong Buy signal** — excelling across most factors"
    elif strong_count >= 3:
        overall = "**Overall: Buy signal** — strong in majority of factors"
    elif weak_count >= 3:
        overall = "**Overall: Sell signal** — weak across most factors"
    elif weak_count >= 2:
        overall = "**Overall: Cautious** — significant weaknesses present"
    else:
        overall = "**Overall: Mixed** — balanced strengths and weaknesses"

    report_lines.append(overall)
    report_lines.append("")

    return "\n".join(report_lines)
