"""AKShare A股核心数据供应商。

替代 Yahoo Finance (y_finance.py) 的 stock data / technical / fundamentals
数据源，直接通过 AKShare 获取 A 股数据。

关键特性：
  - 覆盖所有 A 股（含科创板 688xxx、创业板 300xxx）
  - OHLCV 使用腾讯数据源 (stock_zh_a_hist_tx)，通过代理可达
  - 财务报表/基本面使用东方财富数据源，绕过代理直连
  - 接口签名与 y_finance 保持一致，可无缝注册到 interface.py

AKShare 列名映射 (腾讯数据源)：
  stock_zh_a_hist_tx: date→Date, open→Open, close→Close, high→High, low→Low,
                       volume→Volume
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from datetime import datetime
from typing import Annotated

import pandas as pd
from dateutil.relativedelta import relativedelta

from .errors import NoMarketDataError, VendorNotConfiguredError

logger = logging.getLogger(__name__)

# AKShare 连接国内站点需要绕过系统代理
_ORIGINAL_SESSION_REQUEST = None


def _ensure_patched():
    """确保 requests.Session.request 已打补丁（绕过代理）。"""
    global _ORIGINAL_SESSION_REQUEST
    import requests

    if _ORIGINAL_SESSION_REQUEST is not None:
        return

    _ORIGINAL_SESSION_REQUEST = requests.Session.request

    def _no_proxy_request(self, method, url, **kwargs):
        kwargs.setdefault("proxies", {"http": None, "https": None})
        return _ORIGINAL_SESSION_REQUEST(self, method, url, **kwargs)

    requests.Session.request = _no_proxy_request
    logger.debug("requests.Session.request patched for akshare (no-proxy)")


def _restore_patched():
    """恢复 requests.Session.request 原始实现。"""
    global _ORIGINAL_SESSION_REQUEST
    import requests

    if _ORIGINAL_SESSION_REQUEST is not None:
        requests.Session.request = _ORIGINAL_SESSION_REQUEST
        _ORIGINAL_SESSION_REQUEST = None


@contextmanager
def _akshare_bypass():
    """上下文管理器：AKShare 调用期间禁用代理（东方财富 API 直连国内站点）。"""
    import os as _os
    import requests

    # 保存代理环境变量
    _saved = {
        k: _os.environ.pop(k, None)
        for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
                   "ALL_PROXY", "all_proxy", "NO_PROXY", "no_proxy")
    }

    original = requests.Session.request

    def _no_proxy(self, method, url, **kwargs):
        # 强制不使用代理
        kwargs["proxies"] = {"http": None, "https": None}
        return original(self, method, url, **kwargs)

    requests.Session.request = _no_proxy
    try:
        yield
    finally:
        requests.Session.request = original
        # 恢复代理环境变量
        for k, v in _saved.items():
            if v is not None:
                _os.environ[k] = v


# ---- 符号处理 ----

def _as_akshare_code(ticker: str) -> str:
    """将 '688299.SS' / '000001.SZ' 转换为 akshare 用的纯数字代码。"""
    return ticker.split(".")[0]


# 腾讯数据源 symbol 格式映射：.SS/.SH → sh, .SZ → sz, .BJ → bj
_TX_EXCHANGE_PREFIX = {
    "SH": "sh",
    "SS": "sh",
    "SZ": "sz",
    "BJ": "bj",
}


def _as_tx_symbol(ticker: str) -> str:
    """将 '688016.SS' / '000001.SZ' 转换为腾讯数据源的 'sh688016' / 'sz000001'。"""
    parts = ticker.upper().split(".")
    code = parts[0]
    suffix = parts[1] if len(parts) > 1 else ""
    prefix = _TX_EXCHANGE_PREFIX.get(suffix, suffix.lower())
    return f"{prefix}{code}"


def _as_akshare_period(ticker: str) -> str:
    """返回 akshare 需要的 period 参数。"""
    return "daily"


def _date_to_akshare(date_str: str) -> str:
    """YYYY-MM-DD → YYYYMMDD。"""
    return date_str.replace("-", "")


# ---- OHLCV 数据 ----

def get_stock_data(
    symbol: Annotated[str, "ticker symbol, e.g. 688299.SS"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
):
    """获取 A 股 OHLCV 数据，返回与 y_finance.get_YFin_data_online 相同格式的 CSV。

    数据来源：腾讯 (AKShare: stock_zh_a_hist_tx)，通过代理可达。
    """
    tx_symbol = _as_tx_symbol(symbol)
    start = _date_to_akshare(start_date)
    end = _date_to_akshare(end_date)

    try:
        import akshare as ak
    except ImportError:
        raise VendorNotConfiguredError("AKShare 未安装: pip install akshare")

    # 腾讯数据源通过代理可达，不绕过代理
    try:
        df = ak.stock_zh_a_hist_tx(
            symbol=tx_symbol,
            start_date=start,
            end_date=end,
            adjust="qfq",
        )
    except Exception as e:
        raise NoMarketDataError(
            symbol, tx_symbol, f"AKShare(腾讯) OHLCV 获取失败: {e}"
        ) from e

    if df is None or df.empty:
        raise NoMarketDataError(
            symbol, tx_symbol, "AKShare(腾讯) 返回空 OHLCV 数据（股票可能已退市或代码无效）"
        )

    # 映射列名到 yfinance 标准格式（腾讯数据源返回小写英文列名）
    col_map = {
        "date": "Date",
        "open": "Open",
        "close": "Close",
        "high": "High",
        "low": "Low",
        "volume": "Volume",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    # 四舍五入
    for col in ("Open", "High", "Low", "Close"):
        if col in df.columns:
            df[col] = df[col].round(2)

    csv_string = df.to_csv(index=False)
    header = (
        f"# Stock data for {symbol} (via AKShare/腾讯) from {start_date} to {end_date}\n"
        f"# Total records: {len(df)}\n"
        f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    )
    return header + csv_string


# ---- 技术指标 ----

def get_stock_stats_indicators_window(
    symbol: Annotated[str, "ticker symbol"],
    indicator: Annotated[str, "technical indicator name"],
    curr_date: Annotated[str, "current trading date, YYYY-mm-dd"],
    look_back_days: Annotated[int, "how many days to look back"],
) -> str:
    """通过 stockstats 从 AKShare OHLCV 数据计算技术指标。

    使用 load_ohlcv 拉取 5 年数据确保长期指标（如 200 SMA）计算准确，
    look_back_days 仅控制输出显示范围。
    """
    from stockstats import wrap

    best_ind_params = {
        "close_50_sma": "50 SMA: 中期趋势指标。用作动态支撑/阻力位。",
        "close_200_sma": "200 SMA: 长期趋势基准。用于确认整体市场趋势。",
        "close_10_ema": "10 EMA: 短期响应平均线。捕捉快速动量变化。",
        "macd": "MACD: 通过 EMA 差异计算动量。关注交叉和背离信号。",
        "macds": "MACD Signal: MACD 线的平滑 EMA。",
        "macdh": "MACD Histogram: MACD 线与信号线的差距。",
        "rsi": "RSI: 动量指标，超买超卖的 70/30 阈值。",
        "boll": "Bollinger Middle: 20 SMA 布林带中间线。",
        "boll_ub": "Bollinger Upper Band: 布林带上轨。",
        "boll_lb": "Bollinger Lower Band: 布林带下轨。",
        "atr": "ATR: 平均真实波幅，用于设置止损。",
        "vwma": "VWMA: 成交量加权移动平均线。",
        "mfi": "MFI: 资金流量指标，结合价格与成交量。",
    }

    if indicator not in best_ind_params:
        raise ValueError(
            f"Indicator {indicator} not supported. Choose from: {list(best_ind_params.keys())}"
        )

    end_date = curr_date
    curr_date_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    before = curr_date_dt - relativedelta(days=look_back_days)

    # 使用 load_ohlcv 拉取 5 年数据（缓存），确保长期指标计算准确
    # look_back_days 仅控制输出范围，不影响数据拉取窗口
    from .stockstats_utils import load_ohlcv

    try:
        data = load_ohlcv(symbol, curr_date)
    except NoMarketDataError:
        raise
    except Exception as e:
        raise NoMarketDataError(symbol, symbol, f"OHLCV 数据加载失败: {e}") from e

    if data is None or data.empty:
        raise NoMarketDataError(symbol, symbol, "OHLCV 数据为空")

    df = wrap(data)
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")

    try:
        indicator_data = _get_stock_stats_bulk_akshare(df, indicator)
    except Exception as e:
        logger.warning("Bulk indicator calc failed for %s: %s", indicator, e)
        indicator_data = {}

    current_dt = curr_date_dt
    ind_string = ""
    while current_dt >= before:
        ds = current_dt.strftime("%Y-%m-%d")
        val = indicator_data.get(ds, "N/A: Not a trading day (weekend or holiday)")
        ind_string += f"{ds}: {val}\n"
        current_dt -= relativedelta(days=1)

    return (
        f"## {indicator} values from {before.strftime('%Y-%m-%d')} to {end_date}:\n\n"
        + ind_string + "\n\n" + best_ind_params.get(indicator, "")
    )


def _get_stock_stats_bulk_akshare(df, indicator: str) -> dict:
    """从 stockstats 包装后的 DataFrame 批量计算指标。"""
    df[indicator]  # 触发计算
    result = {}
    for _, row in df.iterrows():
        ds = row["Date"]
        val = row[indicator]
        result[ds] = str(val) if not pd.isna(val) else "N/A"
    return result


# ---- 基本面 ----

def get_fundamentals(
    ticker: Annotated[str, "ticker symbol, e.g. 688299.SS"],
    curr_date: Annotated[str, "current date"] = None,
):
    """获取 A 股基本面数据。

    数据来源：东方财富 (AKShare: stock_financial_abstract)。
    """
    code = _as_akshare_code(ticker)

    try:
        import akshare as ak
    except ImportError:
        raise VendorNotConfiguredError("AKShare 未安装")

    with _akshare_bypass():
        try:
            df = ak.stock_financial_abstract(symbol=code)
        except Exception as e:
            # 回退到 Yahoo Finance
            logger.warning("AKShare 基本面获取 %s 失败，回退 yfinance: %s", ticker, e)
            from .y_finance import get_fundamentals as _yf_fundamentals
            return _yf_fundamentals(ticker, curr_date)

    if df is None or df.empty:
        raise NoMarketDataError(ticker, code, "AKShare 基本面数据为空")

    # financial_abstract 格式: 选项, 指标, 日期1, 日期2, ...
    # 提取关键指标和最新一期数据
    latest_col = [c for c in df.columns if c not in ("选项", "指标")][0]

    key_metrics = [
        "归母净利润", "营业总收入", "营业利润", "利润总额",
        "基本每股收益", "每股净资产", "净资产收益率",
        "总资产", "总负债", "资产负债率",
        "经营活动产生的现金流量净额",
    ]

    lines = [f"# Company Fundamentals for {ticker} (via AKShare/东方财富)"]
    lines.append(f"# Latest period: {latest_col}")
    lines.append(f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    found = 0
    for _, row in df.iterrows():
        indicator = str(row.get("指标", ""))
        for km in key_metrics:
            if km in indicator:
                val = row.get(latest_col)
                if val is not None and str(val) not in ("nan", ""):
                    lines.append(f"{indicator}: {val}")
                    found += 1
                break

    if found == 0:
        raise NoMarketDataError(ticker, code, "AKShare 基本面无有效指标")

    return "\n".join(lines)


# ---- 财务报表 ----

# AKShare 财务报表的交易所前缀映射
_ASTOCK_EXCHANGE_PREFIX = {
    "SH": "SH",
    "SS": "SH",
    "SZ": "SZ",
    "BJ": "BJ",
}


def _as_akshare_finance_code(ticker: str) -> str:
    """将 '688016.SS' / '000001.SZ' 转换为 AKShare 财报用的 'SH688016' / 'SZ000001'。"""
    parts = ticker.upper().split(".")
    code = parts[0]
    suffix = parts[1] if len(parts) > 1 else ""
    prefix = _ASTOCK_EXCHANGE_PREFIX.get(suffix, suffix)
    if prefix:
        return f"{prefix}{code}"
    return code


# 核心财报指标 — 中英文友好别名映射
# Balance Sheet
_BS_KEY_ITEMS = {
    "总资产": "TOTAL_ASSETS",
    "总负债": "TOTAL_LIABILITIES",
    "股东权益": "TOTAL_EQUITY",
    "流动资产": "TOTAL_CURRENT_ASSETS",
    "流动负债": "TOTAL_CURRENT_LIABILITIES",
    "货币资金": "MONETARY_CAPITAL",
    "应收账款": "ACCOUNTS_RECE",
    "存货": "INVENTORY",
    "固定资产": "FIXED_ASSET",
    "无形资产": "INTANGIBLE_ASSET",
    "短期借款": "SHORT_BORROW",
    "长期借款": "LONG_BORROW",
    "应付账款": "ACCOUNTS_PAYABLE",
}

# Cash Flow
_CF_KEY_ITEMS = {
    "经营活动现金流": "NETCASH_OPERATE",
    "投资活动现金流": "NETCASH_INVEST",
    "筹资活动现金流": "NETCASH_FINANCE",
    "现金净增加额": "NETCASH_INCREASE",
    "期初现金": "CASH_BEGINNING",
    "期末现金": "CASH_ENDING",
}

# Income Statement
_IS_KEY_ITEMS = {
    "营业总收入": "OPERATE_INCOME",
    "营业收入": "OPERATE_INCOME",
    "营业成本": "OPERATE_COST",
    "营业利润": "OPERATE_PROFIT",
    "利润总额": "TOTAL_PROFIT",
    "净利润": "NETPROFIT",
    "归母净利润": "PARENT_NETPROFIT",
    "基本每股收益": "BASIC_EPS",
    "稀释每股收益": "DILUTED_EPS",
    "营业总收入同比增长": "OPERATE_INCOME_YOY",
    "归母净利润同比增长": "PARENT_NETPROFIT_YOY",
}

# 列名到中文标签的映射（AKShare 列名 -> 人类可读标签）
_COLUMN_LABEL_MAP = {
    # Balance Sheet
    "TOTAL_ASSETS": "总资产",
    "TOTAL_LIABILITIES": "总负债",
    "TOTAL_EQUITY": "股东权益(不含少数股东权益)",
    "TOTAL_CURRENT_ASSETS": "流动资产合计",
    "TOTAL_CURRENT_LIABILITIES": "流动负债合计",
    "MONETARY_CAPITAL": "货币资金",
    "ACCOUNTS_RECE": "应收账款",
    "INVENTORY": "存货",
    "FIXED_ASSET": "固定资产",
    "INTANGIBLE_ASSET": "无形资产",
    "SHORT_BORROW": "短期借款",
    "LONG_BORROW": "长期借款",
    "ACCOUNTS_PAYABLE": "应付账款",
    # Cash Flow
    "NETCASH_OPERATE": "经营活动产生的现金流量净额",
    "NETCASH_INVEST": "投资活动产生的现金流量净额",
    "NETCASH_FINANCE": "筹资活动产生的现金流量净额",
    "NETCASH_INCREASE": "现金及现金等价物净增加额",
    "CASH_BEGINNING": "期初现金及现金等价物余额",
    "CASH_ENDING": "期末现金及现金等价物余额",
    # Income Statement
    "OPERATE_INCOME": "营业总收入",
    "OPERATE_COST": "营业成本",
    "OPERATE_PROFIT": "营业利润",
    "TOTAL_PROFIT": "利润总额",
    "NETPROFIT": "净利润",
    "PARENT_NETPROFIT": "归母净利润",
    "BASIC_EPS": "基本每股收益",
    "DILUTED_EPS": "稀释每股收益",
    "OPERATE_INCOME_YOY": "营业总收入同比增长(%)",
    "PARENT_NETPROFIT_YOY": "归母净利润同比增长(%)",
}


def _format_financial_table(
    df: pd.DataFrame,
    report_name: str,
    ticker: str,
    freq: str,
    key_items: dict[str, str],
) -> str:
    """将 AKShare 宽表格式财报数据格式化为可读的文本报告。

    Args:
        df: AKShare 财报 DataFrame（每行为一期报告，每列为财务科目）
        report_name: 报表中文名（如 "资产负债表"）
        ticker: 原始 ticker
        freq: "annual" 或 "quarterly"
        key_items: 需要提取的关键科目映射 {中文名: AKShare列名}

    Returns:
        格式化后的 CSV-like 文本
    """
    if df is None or df.empty:
        raise NoMarketDataError(ticker, ticker, f"AKShare 返回空{report_name}数据")

    # 按 REPORT_DATE 排序（最新在前），取最近 8 期
    if "REPORT_DATE" in df.columns:
        df = df.copy()
        df["REPORT_DATE"] = pd.to_datetime(df["REPORT_DATE"], errors="coerce")
        df = df.dropna(subset=["REPORT_DATE"])
        df = df.sort_values("REPORT_DATE", ascending=False)

    # 如果传了 freq，过滤（年报 vs 季报）
    if freq == "annual" and "REPORT_TYPE" in df.columns:
        df = df[df["REPORT_TYPE"].str.contains("年报", na=False)]

    df = df.head(8)

    if df.empty:
        raise NoMarketDataError(ticker, ticker, f"AKShare {report_name}过滤后为空")

    # 提取 REPORT_DATE 和关键科目
    date_col = df["REPORT_DATE"].dt.strftime("%Y-%m-%d").tolist()

    lines = [f"# {report_name} for {ticker} ({freq}) via AKShare/东方财富"]
    lines.append(f"# 数据日期: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # 表头
    header = "科目, " + ", ".join(date_col)
    lines.append(header)

    # 每一行：一个科目
    for cn_name, ak_col in key_items.items():
        if ak_col not in df.columns:
            continue
        values = df[ak_col].tolist()
        val_strs = []
        for v in values:
            if pd.isna(v):
                val_strs.append("N/A")
            elif isinstance(v, (int, float)):
                # 格式化大数字
                if abs(v) >= 1e8:
                    val_strs.append(f"{v/1e8:.2f}亿")
                elif abs(v) >= 1e4:
                    val_strs.append(f"{v/1e4:.2f}万")
                else:
                    val_strs.append(f"{v:.2f}")
            else:
                val_strs.append(str(v))
        label = _COLUMN_LABEL_MAP.get(ak_col, cn_name)
        lines.append(f"{label}, " + ", ".join(val_strs))

    return "\n".join(lines)


def get_balance_sheet(
    ticker: Annotated[str, "ticker symbol"],
    freq: Annotated[str, "frequency: 'annual' or 'quarterly'"] = "quarterly",
    curr_date: Annotated[str, "current date"] = None,
):
    """获取资产负债表（AKShare/东方财富 原生数据）。

    通过 AKShare 的 stock_balance_sheet_by_report_em 获取 A 股资产负债表，
    提取核心科目并以结构化文本返回。
    """
    code = _as_akshare_finance_code(ticker)

    try:
        import akshare as ak
    except ImportError:
        raise VendorNotConfiguredError("AKShare 未安装: pip install akshare")

    with _akshare_bypass():
        try:
            df = ak.stock_balance_sheet_by_report_em(symbol=code)
        except Exception as e:
            raise NoMarketDataError(
                ticker, code, f"AKShare 资产负债表获取失败: {e}"
            ) from e

    return _format_financial_table(df, "资产负债表", ticker, freq, _BS_KEY_ITEMS)


def get_cashflow(
    ticker: Annotated[str, "ticker symbol"],
    freq: Annotated[str, "frequency: 'annual' or 'quarterly'"] = "quarterly",
    curr_date: Annotated[str, "current date"] = None,
):
    """获取现金流量表（AKShare/东方财富 原生数据）。

    通过 AKShare 的 stock_cash_flow_sheet_by_report_em 获取数据。
    """
    code = _as_akshare_finance_code(ticker)

    try:
        import akshare as ak
    except ImportError:
        raise VendorNotConfiguredError("AKShare 未安装: pip install akshare")

    with _akshare_bypass():
        try:
            df = ak.stock_cash_flow_sheet_by_report_em(symbol=code)
        except Exception as e:
            raise NoMarketDataError(
                ticker, code, f"AKShare 现金流量表获取失败: {e}"
            ) from e

    return _format_financial_table(df, "现金流量表", ticker, freq, _CF_KEY_ITEMS)


def get_income_statement(
    ticker: Annotated[str, "ticker symbol"],
    freq: Annotated[str, "frequency: 'annual' or 'quarterly'"] = "quarterly",
    curr_date: Annotated[str, "current date"] = None,
):
    """获取利润表（AKShare/东方财富 原生数据）。

    通过 AKShare 的 stock_profit_sheet_by_report_em 获取数据。
    """
    code = _as_akshare_finance_code(ticker)

    try:
        import akshare as ak
    except ImportError:
        raise VendorNotConfiguredError("AKShare 未安装: pip install akshare")

    with _akshare_bypass():
        try:
            df = ak.stock_profit_sheet_by_report_em(symbol=code)
        except Exception as e:
            raise NoMarketDataError(
                ticker, code, f"AKShare 利润表获取失败: {e}"
            ) from e

    return _format_financial_table(df, "利润表", ticker, freq, _IS_KEY_ITEMS)


def get_insider_transactions(
    ticker: Annotated[str, "ticker symbol"],
):
    """A 股无高管交易公开数据，返回占位符。"""
    return f"高管交易数据不可用: A 股市场无公开的 insider transactions 数据源（仅限美股）。"
