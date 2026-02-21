from typing import Annotated
from datetime import datetime
from dateutil.relativedelta import relativedelta
import yfinance as yf
import os
import threading
from .stockstats_utils import StockstatsUtils



def resolve_ticker_to_symbol_and_name(ticker: str) -> tuple[str, str]:
    """用 Yahoo Finance 将用户输入的 ticker 解析为规范 symbol 与官方公司名。
    严格以用户输入为准：若 Yahoo 返回的 symbol 与输入不一致（如 VRT 被误解析为 VRTX），
    则报错拒绝使用，避免分析错公司。

    Returns:
        (symbol, company_long_name)：与用户输入一致的交易代码 + Yahoo 上的正式公司名称。
    Raises:
        ValueError: 无法找到数据、数据无效、或 Yahoo 返回的代码与输入不一致时。
    """
    raw = (ticker or "").strip().upper()
    if not raw:
        raise ValueError("请输入股票代码（ticker）。")
    t = yf.Ticker(raw)
    info = t.info
    if not info or not isinstance(info, dict):
        raise ValueError(
            f"在 Yahoo Finance 未找到代码 '{raw}' 的数据，请核对是否为有效交易代码（如 VRT、AAPL）。"
        )
    yahoo_symbol = (info.get("symbol") or "").upper().strip()
    # 关键：Yahoo 有时会把 VRT 解析成 VRTX，若不一致则绝不采纳 Yahoo 的 symbol/公司名，只用用户输入
    if yahoo_symbol and yahoo_symbol != raw:
        # 仍用用户输入的 raw 作为 symbol，避免下游拿到 VRTX；公司名不信任 Yahoo，只用 ticker 展示
        return (raw, raw)
    long_name = info.get("longName") or info.get("shortName") or raw
    if not long_name or not str(long_name).strip():
        raise ValueError(
            f"代码 '{raw}' 在 Yahoo Finance 无有效公司名称，请核对代码。"
        )
    # 始终以用户输入为 symbol，仅在 Yahoo 返回一致时才用其公司名
    return (raw, str(long_name).strip())


def get_YFin_data_online(
    symbol: Annotated[str, "ticker symbol of the company"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
):

    datetime.strptime(start_date, "%Y-%m-%d")
    datetime.strptime(end_date, "%Y-%m-%d")

    # Create ticker object
    ticker = yf.Ticker(symbol.upper())

    # Fetch historical data for the specified date range
    data = ticker.history(start=start_date, end=end_date)

    # Check if data is empty
    if data.empty:
        return (
            f"No data found for symbol '{symbol}' between {start_date} and {end_date}"
        )

    # Remove timezone info from index for cleaner output
    if data.index.tz is not None:
        data.index = data.index.tz_localize(None)

    # Round numerical values to 2 decimal places for cleaner display
    numeric_columns = ["Open", "High", "Low", "Close", "Adj Close"]
    for col in numeric_columns:
        if col in data.columns:
            data[col] = data[col].round(2)

    # Convert DataFrame to CSV string
    csv_string = data.to_csv()

    # Add header information
    header = f"# Stock data for {symbol.upper()} from {start_date} to {end_date}\n"
    header += f"# Total records: {len(data)}\n"
    header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    return header + csv_string


def prefetch_stock_data(symbol: str):
    """Pre-fetch stock data once to ensure all parallel analysts hit the cache.
    Called by the graph pre-fetch node.
    """
    from .config import get_config
    import pandas as pd
    
    config = get_config()
    today_date = pd.Timestamp.today()
    start_date = today_date - pd.DateOffset(years=2)
    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = today_date.strftime("%Y-%m-%d")
    
    os.makedirs(config["data_cache_dir"], exist_ok=True)
    data_file = os.path.join(
        config["data_cache_dir"],
        f"{symbol}-YFin-data-{start_date_str}-{end_date_str}.csv",
    )
    
    if not os.path.exists(data_file):
        print(f"Pre-fetching data for {symbol}...")
        data = yf.download(
            symbol,
            start=start_date_str,
            end=end_date_str,
            multi_level_index=False,
            progress=False,
            auto_adjust=True,
        )
        if not data.empty:
            data = data.reset_index()
            # Atomic write via temp file
            temp_file = data_file + ".tmp"
            data.to_csv(temp_file, index=False)
            os.rename(temp_file, data_file)
            print(f"Data for {symbol} cached successfully.")
        else:
            print(f"Warning: No data found for {symbol} during pre-fetch.")


def get_stock_stats_indicators_window(
    symbol: Annotated[str, "ticker symbol of the company"],
    indicator: Annotated[str, "technical indicator to get the analysis and report of (can be comma-separated list)"],
    curr_date: Annotated[
        str, "The current trading date you are trading on, YYYY-mm-dd"
    ],
    look_back_days: Annotated[int, "how many days to look back"],
) -> str:
    # Normalize and handle multiple indicators
    indicators_request = [i.strip().lower() for i in indicator.split(',')]
    
    best_ind_params = {
        # Moving Averages
        "close_50_sma": (
            "50 SMA: A medium-term trend indicator. "
            "Usage: Identify trend direction and serve as dynamic support/resistance. "
            "Tips: It lags price; combine with faster indicators for timely signals."
        ),
        "close_200_sma": (
            "200 SMA: A long-term trend benchmark. "
            "Usage: Confirm overall market trend and identify golden/death cross setups. "
            "Tips: It reacts slowly; best for strategic trend confirmation rather than frequent trading entries."
        ),
        "close_10_ema": (
            "10 EMA: A responsive short-term average. "
            "Usage: Capture quick shifts in momentum and potential entry points. "
            "Tips: Prone to noise in choppy markets; use alongside longer averages for filtering false signals."
        ),
        # MACD Related
        "macd": (
            "MACD: Computes momentum via differences of EMAs. "
            "Usage: Look for crossovers and divergence as signals of trend changes. "
            "Tips: Confirm with other indicators in low-volatility or sideways markets."
        ),
        "macds": (
            "MACD Signal: An EMA smoothing of the MACD line. "
            "Usage: Use crossovers with the MACD line to trigger trades. "
            "Tips: Should be part of a broader strategy to avoid false positives."
        ),
        "macdh": (
            "MACD Histogram: Shows the gap between the MACD line and its signal. "
            "Usage: Visualize momentum strength and spot divergence early. "
            "Tips: Can be volatile; complement with additional filters in fast-moving markets."
        ),
        # Momentum Indicators
        "rsi": (
            "RSI: Measures momentum to flag overbought/oversold conditions. "
            "Usage: Apply 70/30 thresholds and watch for divergence to signal reversals. "
            "Tips: In strong trends, RSI may remain extreme; always cross-check with trend analysis."
        ),
        # Volatility Indicators
        "boll": (
            "Bollinger Middle: A 20 SMA serving as the basis for Bollinger Bands. "
            "Usage: Acts as a dynamic benchmark for price movement. "
            "Tips: Combine with the upper and lower bands to effectively spot breakouts or reversals."
        ),
        "boll_ub": (
            "Bollinger Upper Band: Typically 2 standard deviations above the middle line. "
            "Usage: Signals potential overbought conditions and breakout zones. "
            "Tips: Confirm signals with other tools; prices may ride the band in strong trends."
        ),
        "boll_lb": (
            "Bollinger Lower Band: Typically 2 standard deviations below the middle line. "
            "Usage: Indicates potential oversold conditions. "
            "Tips: Use additional analysis to avoid false reversal signals."
        ),
        "atr": (
            "ATR: Averages true range to measure volatility. "
            "Usage: Set stop-loss levels and adjust position sizes based on current market volatility. "
            "Tips: It's a reactive measure, so use it as part of a broader risk management strategy."
        ),
        # Volume-Based Indicators
        "vwma": (
            "VWMA: A moving average weighted by volume. "
            "Usage: Confirm trends by integrating price action with volume data. "
            "Tips: Watch for skewed results from volume spikes; use in combination with other volume analyses."
        ),
        "mfi": (
            "MFI: The Money Flow Index is a momentum indicator that uses both price and volume to measure buying and selling pressure. "
            "Usage: Identify overbought (>80) or oversold (<20) conditions and confirm the strength of trends or reversals. "
            "Tips: Use alongside RSI or MACD to confirm signals; divergence between price and MFI can indicate potential reversals."
        ),
        # Trend Strength Indicators
        "adx": (
            "ADX: Average Directional Index measures trend strength on a 0-100 scale. "
            "Usage: ADX > 25 = strong trend (use trend-following strategies), ADX < 20 = weak/ranging market (use mean-reversion). "
            "Tips: ADX only measures strength, NOT direction. Combine with +DI/-DI or price action for direction. "
            "Rising ADX = strengthening trend; falling ADX = weakening trend, even if price continues moving."
        ),
        "cci": (
            "CCI: Commodity Channel Index measures deviation from statistical mean. "
            "Usage: CCI > +100 = overbought / bullish momentum; CCI < -100 = oversold / bearish momentum. "
            "Tips: Strong trends can keep CCI extreme for long periods. Best used for identifying cyclical turning points and divergence."
        ),
        "wr": (
            "WR: Williams %R is a fast momentum oscillator ranging from 0 to -100. "
            "Usage: WR > -20 = overbought; WR < -80 = oversold. Faster than RSI for detecting reversals. "
            "Tips: In strong trends, WR can stay overbought/oversold for extended periods. Best for timing entries within a confirmed trend."
        ),
        # Trend Following
        "supertrend": (
            "Supertrend: A trend-following overlay based on ATR. Gives clear buy/sell signals when price crosses the Supertrend line. "
            "Usage: Price above Supertrend = bullish; price below = bearish. Use for trend direction and trailing stop placement. "
            "Tips: Works best in trending markets (ADX > 25). Generates false signals in ranging/choppy markets."
        ),
        "aroon": (
            "Aroon Oscillator: Measures time elapsed since the highest high and lowest low over a period. Range: -100 to +100. "
            "Usage: Aroon > 0 = bullish (recent highs); Aroon < 0 = bearish (recent lows). Strong signals near ±100. "
            "Tips: Early trend detector — Aroon often signals new trends before moving averages. Good for identifying trend initiation."
        ),
        # Asian-market popular
        "kdjk": (
            "KDJ-K: The K line of the KDJ indicator (stochastic-based). "
            "Usage: K crossing above D = bullish signal; K crossing below D = bearish. K > 80 = overbought; K < 20 = oversold. "
            "Tips: More sensitive than standard Stochastic. The J line (kdjj) can exceed 0-100 range, providing early extreme signals."
        ),
        "kdjd": (
            "KDJ-D: The D line (smoothed K) of the KDJ indicator. "
            "Usage: Acts as the signal line for KDJ. K-D crossovers generate trading signals. "
            "Tips: When both K and D are above 80 and K crosses below D, it is a strong sell signal, and vice versa below 20."
        ),
        "trix": (
            "TRIX: Triple Exponential Moving Average rate of change. Filters out insignificant price moves. "
            "Usage: TRIX > 0 = bullish momentum; TRIX < 0 = bearish. Cross above/below zero line = trend change signal. "
            "Tips: Very smooth indicator — excellent for spotting divergence and filtering noise. Slow to react but low false signals."
        ),
    }

    # Common aliases mapping
    _ALIASES = {
        "sma": "close_50_sma", "sma_50": "close_50_sma", "sma_200": "close_200_sma",
        "50_sma": "close_50_sma", "200_sma": "close_200_sma",
        "ema": "close_10_ema", "ema_10": "close_10_ema", "10_ema": "close_10_ema",
        "macd_signal": "macds", "macd_histogram": "macdh",
        "bollinger": "boll", "bollinger_upper": "boll_ub", "bollinger_lower": "boll_lb",
        "money_flow": "mfi", "williams": "wr", "commodity_channel_index": "cci",
        "kdj": "kdjk", "stochastic": "kdjk",
    }

    combined_reports = []
    
    for ind in indicators_request:
        actual_ind = _ALIASES.get(ind, ind)
        
        if actual_ind not in best_ind_params:
            combined_reports.append(f"ERROR: Indicator '{ind}' is not supported.")
            continue

        end_date = curr_date
        curr_date_dt = datetime.strptime(curr_date, "%Y-%m-%d")
        before = curr_date_dt - relativedelta(days=look_back_days)

        try:
            indicator_data = _get_stock_stats_bulk(symbol, actual_ind, curr_date)
            
            # Generate the date range we need
            current_dt = curr_date_dt
            date_values = []
            while current_dt >= before:
                date_str = current_dt.strftime('%Y-%m-%d')
                val = indicator_data.get(date_str, "N/A: Not a trading day")
                date_values.append((date_str, val))
                current_dt = current_dt - relativedelta(days=1)
            
            ind_string = "\n".join([f"{d}: {v}" for d, v in date_values])
            
            report = (
                f"### {actual_ind.upper()} ({before.strftime('%Y-%m-%d')} to {end_date})\n"
                f"{ind_string}\n\n"
                f"Description: {best_ind_params[actual_ind]}\n"
            )
            combined_reports.append(report)
            
        except Exception as e:
            combined_reports.append(f"ERROR processing '{ind}': {e}")

    return f"# Technical Indicators Report for {symbol.upper()}\n\n" + "\n---\n".join(combined_reports)


def _get_stock_stats_bulk(
    symbol: Annotated[str, "ticker symbol of the company"],
    indicator: Annotated[str, "technical indicator to calculate"],
    curr_date: Annotated[str, "current date for reference"]
) -> dict:
    """
    Optimized bulk calculation of stock stats indicators.
    Fetches data once and calculates indicator for all available dates.
    Returns dict mapping date strings to indicator values.
    """
    from .config import get_config
    import pandas as pd
    from stockstats import wrap
    import os
    
    config = get_config()
    online = config["data_vendors"]["technical_indicators"] != "local"
    
    if not online:
        # Local data path
        try:
            data = pd.read_csv(
                os.path.join(
                    config.get("data_cache_dir", "data"),
                    f"{symbol}-YFin-data-2015-01-01-2025-03-25.csv",
                )
            )
            df = wrap(data)
        except FileNotFoundError:
            raise Exception("Stockstats fail: Yahoo Finance data not fetched yet!")
    else:
        # Online data fetching with caching
        today_date = pd.Timestamp.today()
        curr_date_dt = pd.to_datetime(curr_date)
        
        end_date = today_date
        start_date = today_date - pd.DateOffset(years=2)
        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date_str = end_date.strftime("%Y-%m-%d")
        
        os.makedirs(config["data_cache_dir"], exist_ok=True)
        
        data_file = os.path.join(
            config["data_cache_dir"],
            f"{symbol}-YFin-data-{start_date_str}-{end_date_str}.csv",
        )
        
        if os.path.exists(data_file):
            data = pd.read_csv(data_file)
            data["Date"] = pd.to_datetime(data["Date"])
        else:
            print(f"Downloading data for {symbol}...")
            data = yf.download(
                symbol,
                start=start_date_str,
                end=end_date_str,
                multi_level_index=False,
                progress=False,
                auto_adjust=True,
            )
            if data.empty:
                raise Exception(f"No data found for {symbol}")
            data = data.reset_index()
            # Atomic write via temp file
            temp_file = data_file + ".tmp"
            data.to_csv(temp_file, index=False)
            os.rename(temp_file, data_file)
        
        df = wrap(data)
        df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
    
    # Calculate the indicator for all rows at once
    df[indicator]  # This triggers stockstats to calculate the indicator
    
    # Create a dictionary mapping date strings to indicator values
    result_dict = {}
    for _, row in df.iterrows():
        date_str = row["Date"]
        indicator_value = row[indicator]
        
        # Handle NaN/None values
        if pd.isna(indicator_value):
            result_dict[date_str] = "N/A"
        else:
            result_dict[date_str] = str(indicator_value)
    
    return result_dict


def get_stockstats_indicator(
    symbol: Annotated[str, "ticker symbol of the company"],
    indicator: Annotated[str, "technical indicator to get the analysis and report of"],
    curr_date: Annotated[
        str, "The current trading date you are trading on, YYYY-mm-dd"
    ],
) -> str:

    curr_date_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    curr_date = curr_date_dt.strftime("%Y-%m-%d")

    try:
        indicator_value = StockstatsUtils.get_stock_stats(
            symbol,
            indicator,
            curr_date,
        )
    except Exception as e:
        print(
            f"Error getting stockstats indicator data for indicator {indicator} on {curr_date}: {e}"
        )
        return ""

    return str(indicator_value)


def get_fundamentals(
    ticker: Annotated[str, "ticker symbol of the company"],
    curr_date: Annotated[str, "current date (not used for yfinance)"] = None
):
    """Get company fundamentals overview from yfinance."""
    try:
        ticker_obj = yf.Ticker(ticker.upper())
        info = ticker_obj.info

        if not info:
            return f"No fundamentals data found for symbol '{ticker}'"

        fields = [
            ("Name", info.get("longName")),
            ("Sector", info.get("sector")),
            ("Industry", info.get("industry")),
            ("Market Cap", info.get("marketCap")),
            ("PE Ratio (TTM)", info.get("trailingPE")),
            ("Forward PE", info.get("forwardPE")),
            ("PEG Ratio", info.get("pegRatio")),
            ("Price to Book", info.get("priceToBook")),
            ("EPS (TTM)", info.get("trailingEps")),
            ("Forward EPS", info.get("forwardEps")),
            ("Dividend Yield", info.get("dividendYield")),
            ("Beta", info.get("beta")),
            ("52 Week High", info.get("fiftyTwoWeekHigh")),
            ("52 Week Low", info.get("fiftyTwoWeekLow")),
            ("50 Day Average", info.get("fiftyDayAverage")),
            ("200 Day Average", info.get("twoHundredDayAverage")),
            ("Revenue (TTM)", info.get("totalRevenue")),
            ("Gross Profit", info.get("grossProfits")),
            ("EBITDA", info.get("ebitda")),
            ("Net Income", info.get("netIncomeToCommon")),
            ("Profit Margin", info.get("profitMargins")),
            ("Operating Margin", info.get("operatingMargins")),
            ("Return on Equity", info.get("returnOnEquity")),
            ("Return on Assets", info.get("returnOnAssets")),
            ("Debt to Equity", info.get("debtToEquity")),
            ("Current Ratio", info.get("currentRatio")),
            ("Book Value", info.get("bookValue")),
            ("Free Cash Flow", info.get("freeCashflow")),
        ]

        lines = []
        for label, value in fields:
            if value is not None:
                lines.append(f"{label}: {value}")

        header = f"# Company Fundamentals for {ticker.upper()}\n"
        header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

        return header + "\n".join(lines)

    except Exception as e:
        return f"Error retrieving fundamentals for {ticker}: {str(e)}"


def get_balance_sheet(
    ticker: Annotated[str, "ticker symbol of the company"],
    freq: Annotated[str, "frequency of data: 'annual' or 'quarterly'"] = "quarterly",
    curr_date: Annotated[str, "current date (not used for yfinance)"] = None
):
    """Get balance sheet data from yfinance."""
    try:
        ticker_obj = yf.Ticker(ticker.upper())
        
        if freq.lower() == "quarterly":
            data = ticker_obj.quarterly_balance_sheet
        else:
            data = ticker_obj.balance_sheet
            
        if data.empty:
            return f"No balance sheet data found for symbol '{ticker}'"
            
        # Convert to CSV string for consistency with other functions
        csv_string = data.to_csv()
        
        # Add header information
        header = f"# Balance Sheet data for {ticker.upper()} ({freq})\n"
        header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        return header + csv_string
        
    except Exception as e:
        return f"Error retrieving balance sheet for {ticker}: {str(e)}"


def get_cashflow(
    ticker: Annotated[str, "ticker symbol of the company"],
    freq: Annotated[str, "frequency of data: 'annual' or 'quarterly'"] = "quarterly",
    curr_date: Annotated[str, "current date (not used for yfinance)"] = None
):
    """Get cash flow data from yfinance."""
    try:
        ticker_obj = yf.Ticker(ticker.upper())
        
        if freq.lower() == "quarterly":
            data = ticker_obj.quarterly_cashflow
        else:
            data = ticker_obj.cashflow
            
        if data.empty:
            return f"No cash flow data found for symbol '{ticker}'"
            
        # Convert to CSV string for consistency with other functions
        csv_string = data.to_csv()
        
        # Add header information
        header = f"# Cash Flow data for {ticker.upper()} ({freq})\n"
        header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        return header + csv_string
        
    except Exception as e:
        return f"Error retrieving cash flow for {ticker}: {str(e)}"


def get_income_statement(
    ticker: Annotated[str, "ticker symbol of the company"],
    freq: Annotated[str, "frequency of data: 'annual' or 'quarterly'"] = "quarterly",
    curr_date: Annotated[str, "current date (not used for yfinance)"] = None
):
    """Get income statement data from yfinance."""
    try:
        ticker_obj = yf.Ticker(ticker.upper())
        
        if freq.lower() == "quarterly":
            data = ticker_obj.quarterly_income_stmt
        else:
            data = ticker_obj.income_stmt
            
        if data.empty:
            return f"No income statement data found for symbol '{ticker}'"
            
        # Convert to CSV string for consistency with other functions
        csv_string = data.to_csv()
        
        # Add header information
        header = f"# Income Statement data for {ticker.upper()} ({freq})\n"
        header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        return header + csv_string
        
    except Exception as e:
        return f"Error retrieving income statement for {ticker}: {str(e)}"


def get_insider_transactions(
    ticker: Annotated[str, "ticker symbol of the company"]
):
    """Get insider transactions data from yfinance."""
    try:
        ticker_obj = yf.Ticker(ticker.upper())
        data = ticker_obj.insider_transactions
        
        if data is None or data.empty:
            return f"No insider transactions data found for symbol '{ticker}'"
            
        # Convert to CSV string for consistency with other functions
        csv_string = data.to_csv()
        
        # Add header information
        header = f"# Insider Transactions data for {ticker.upper()}\n"
        header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        return header + csv_string
        
    except Exception as e:
        return f"Error retrieving insider transactions for {ticker}: {str(e)}"