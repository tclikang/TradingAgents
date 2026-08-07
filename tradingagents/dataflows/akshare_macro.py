"""AKShare 中国宏观经济数据供应商。

通过 AKShare 免费开源库获取中国宏观数据：LPR利率、社融、PMI、CPI、
PPI、M2、工业增加值等。替代原来的 FRED（美联储数据）。

列名和函数名已验证与 akshare 1.18.64 版本一致。
"""

import logging
from datetime import datetime, timedelta

from .errors import VendorNotConfiguredError

logger = logging.getLogger(__name__)

DEFAULT_LOOKBACK_DAYS = 365
MAX_ROWS = 40

# 中文友好别名 -> AKShare 内部 key
CHINA_MACRO_SERIES = {
    # 利率
    "lpr_1y": "lpr_1y",
    "lpr_5y": "lpr_5y",
    "一年期lpr": "lpr_1y",
    "五年期lpr": "lpr_5y",
    "lpr利率": "lpr_1y",
    "shibor": "shibor",
    # 货币供应
    "m2": "m2",
    "m2货币供应量": "m2",
    "货币供应量": "m2",
    "m1": "m1",
    "社融": "social_financing",
    "社会融资规模": "social_financing",
    # 价格指数
    "cpi": "cpi",
    "居民消费价格指数": "cpi",
    "ppi": "ppi",
    "工业生产者出厂价格指数": "ppi",
    # 景气指数
    "pmi": "pmi",
    "制造业pmi": "pmi",
    "制造业采购经理指数": "pmi",
    "非制造业pmi": "non_manufacturing_pmi",
    # GDP / 工业
    "gdp": "gdp",
    "国内生产总值": "gdp",
    "工业增加值": "industrial_production",
    # 外汇 / 贸易
    "外汇储备": "forex_reserves",
    "贸易差额": "trade_balance",
    # 就业 / 消费
    "失业率": "unemployment",
    "社会消费品零售总额": "retail_sales",
    # 房地产
    "房地产投资": "real_estate_investment",
}


class AkshareNotConfiguredError(VendorNotConfiguredError):
    """AKShare 未安装时抛出。"""
    pass


def _ensure_akshare():
    try:
        import akshare as ak  # noqa: F401
    except ImportError:
        raise AkshareNotConfiguredError("AKShare 未安装。请运行: pip install akshare") from None


def _resolve_key(name: str) -> str:
    key = name.strip().lower().replace(" ", "_").replace("-", "_")
    return CHINA_MACRO_SERIES.get(key, key)


def _select_cols(df, date_col, val_cols=None):
    """从 DataFrame 中选出日期列和数值列，统一命名为 日期/数值。"""
    import pandas as pd

    if df is None or df.empty:
        return None
    if val_cols is None:
        # 自动选第一列非日期列
        val_cols = [c for c in df.columns if c != date_col][:1]
    keep = [date_col] + list(val_cols)
    missing = [c for c in keep if c not in df.columns]
    if missing:
        return None
    result = df[keep].copy()
    result.columns = ["日期"] + [f"数值_{i}" for i in range(1, len(val_cols) + 1)]
    return result


def get_macro_data(
    indicator: str,
    curr_date: str,
    look_back_days: int | None = None,
) -> str:
    if look_back_days is None:
        look_back_days = DEFAULT_LOOKBACK_DAYS

    end_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    start_dt = end_dt - timedelta(days=look_back_days)

    try:
        _ensure_akshare()
        import akshare as ak
    except AkshareNotConfiguredError as e:
        return f"中国宏观数据不可用: {e}"

    key = _resolve_key(indicator)

    try:
        data = _fetch_macro_series(ak, key)
    except Exception as e:
        logger.warning("AKShare 获取 %s 失败: %s", key, e)
        return (
            f"中国宏观数据 '{indicator}' 获取失败: {e}。"
            f"可用指标: {', '.join(CHINA_MACRO_SERIES.keys())}"
        )

    if data is None or data.empty:
        return f"中国宏观指标 '{indicator}' 在当前时间段无数据（可能接口暂时不可用）。"

    # 按日期过滤
    data["日期_dt"] = pd_to_datetime(data["日期"])
    data = data[(data["日期_dt"] >= start_dt) & (data["日期_dt"] <= end_dt)]
    data = data.drop(columns=["日期_dt"])
    if data.empty:
        return f"中国宏观指标 '{indicator}' 在 {start_dt.strftime('%Y-%m-%d')} 至 {curr_date} 期间无数据。"

    # 最近值
    last_row = data.iloc[-1]
    first_row = data.iloc[0]
    last_val = last_row.iloc[1] if len(data.columns) > 1 else last_row.iloc[0]
    first_val = first_row.iloc[1] if len(data.columns) > 1 else first_row.iloc[0]
    last_date_str = str(last_row.iloc[0])

    # 反向查找中文名
    indicator_cn = indicator
    for cn, k in CHINA_MACRO_SERIES.items():
        if k == key:
            indicator_cn = cn
            break

    header = (
        f"## 中国宏观数据: {indicator_cn}\n"
        f"- 数据来源: AKShare (东方财富/国家统计局)\n"
        f"- 时间范围: {start_dt.strftime('%Y-%m-%d')} 至 {curr_date}\n"
    )

    try:
        delta = float(last_val) - float(first_val)
        base = float(first_val)
        pct = f" ({delta / base * 100:+.2f}%)" if base != 0 else ""
        summary = (
            f"\n**最新值:** {last_val} ({last_date_str}) | "
            f"**期间变化:** {delta:+.2f}{pct}（从 {first_val}）\n"
        )
    except (ValueError, TypeError):
        summary = f"\n**最新值:** {last_val} ({last_date_str})\n"

    shown = data
    note = ""
    if len(data) > MAX_ROWS:
        shown = data.iloc[-MAX_ROWS:]
        note = f"\n_(仅显示最近 {MAX_ROWS} 条，共 {len(data)} 条)_\n"

    table = (
        "\n| 日期 | 数值 |\n| --- | --- |\n"
        + "\n".join(
            f"| {str(row.iloc[0])} | {str(row.iloc[1])} |"
            for _, row in shown.iterrows()
        )
        + "\n"
    )

    return header + summary + note + table


def pd_to_datetime(series):
    """鲁棒地将字符串列转为 datetime。"""
    import pandas as pd
    import re

    s = series.astype(str)
    # 处理 "2026年05月份" 格式
    if s.str.contains(r"年.*月份").any():
        cleaned = s.str.replace(r"年(\d+)月份", r"-\1-01", regex=True)
        return pd.to_datetime(cleaned, errors="coerce")
    # 处理 "2026年第1季度" 格式
    if s.str.contains("季度").any():
        def _parse_quarter(v):
            m = re.match(r"(\d{4})年第(\d)季度", str(v))
            if m:
                return pd.Timestamp(f"{m.group(1)}-{int(m.group(2))*3-2:02d}-01")
            return pd.NaT
        return s.apply(_parse_quarter)
    return pd.to_datetime(s, errors="coerce")


def _fetch_macro_series(ak, key: str):
    import pandas as pd

    # ---- 利率 ----
    if key in ("lpr_1y", "lpr_5y"):
        df = ak.macro_china_lpr()
        # 实际列名: TRADE_DATE, LPR1Y, LPR5Y
        target_col = "LPR1Y" if key == "lpr_1y" else "LPR5Y"
        result = df[["TRADE_DATE", target_col]].copy()
        result.columns = ["日期", "数值"]
        return result

    if key == "shibor":
        df = ak.macro_china_shibor_all()
        # 列名: 日期, O/N-定价
        return _select_cols(df, "日期", ["O/N-定价"])

    # ---- 货币供应 ----
    if key == "m2":
        df = ak.macro_china_money_supply()
        # 列名: 月份, 货币和准货币(M2)-数量(亿元), 货币和准货币(M2)-同比增长
        return _select_cols(df, "月份", ["货币和准货币(M2)-数量(亿元)"])

    if key == "m1":
        df = ak.macro_china_money_supply()
        return _select_cols(df, "月份", ["货币(M1)-数量(亿元)"])

    if key == "social_financing":
        try:
            df = ak.macro_china_shrzgm()
            return _select_cols(df, df.columns[0])
        except Exception as e:
            # 政府网站 SSL 可能失败，回退到东方财富版社融
            logger.warning("社融 (mofcom) 获取失败，尝试备用源: %s", e)
            try:
                df = ak.macro_china_new_social_financing() if hasattr(ak, "macro_china_new_social_financing") else None
                if df is not None and not df.empty:
                    return _select_cols(df, df.columns[0])
            except Exception:
                pass
            raise Exception(f"社融数据获取失败（政府网站 SSL 问题）: {e}")

    # ---- 价格指数 ----
    if key == "cpi":
        df = ak.macro_china_cpi_monthly()
        # 列名: 商品, 日期, 今值, 预测值, 前值
        return _select_cols(df, "日期", ["今值"])

    if key == "ppi":
        df = ak.macro_china_ppi()
        # 列名: 月份, 当月, 当月同比增长, 累计
        return _select_cols(df, "月份", ["当月"])

    # ---- 景气指数 ----
    if key == "pmi":
        df = ak.macro_china_pmi()
        # 列名: 月份, 制造业-指数, 制造业-同比增长, 非制造业-指数, 非制造业-同比增长
        return _select_cols(df, "月份", ["制造业-指数"])

    if key == "non_manufacturing_pmi":
        df = ak.macro_china_non_man_pmi()
        # 列名: 商品, 日期, 今值, 预测值, 前值
        return _select_cols(df, "日期", ["今值"])

    # ---- GDP / 工业 ----
    if key == "gdp":
        df = ak.macro_china_gdp()
        # 列名: 季度, 国内生产总值-绝对值, 国内生产总值-同比增长
        return _select_cols(df, "季度", ["国内生产总值-绝对值"])

    if key == "industrial_production":
        # 函数名是 macro_china_gyzjz
        df = ak.macro_china_gyzjz()
        # 列名: 月份, 同比增长, 累计增长, 发布时间
        return _select_cols(df, "月份", ["同比增长"])

    # ---- 外汇 / 贸易 ----
    if key == "forex_reserves":
        df = ak.macro_china_fx_gold()
        # 列名: 月份, 黄金储备-数值, 国家外汇储备-数值
        return _select_cols(df, "月份", ["国家外汇储备-数值"])

    if key == "trade_balance":
        df = ak.macro_china_trade_balance()
        # 列名: 商品, 日期, 今值, 预测值, 前值
        return _select_cols(df, "日期", ["今值"])

    # ---- 就业 / 消费 ----
    if key == "unemployment":
        try:
            df = ak.macro_china_urban_unemployment()
            if df is not None and not df.empty:
                return _select_cols(df, df.columns[0])
        except Exception as e:
            logger.warning("失业率数据获取失败: %s", e)
        raise Exception("城镇调查失业率数据暂不可用（数据源接口变动）")

    if key == "retail_sales":
        df = ak.macro_china_consumer_goods_retail()
        # 列名: 月份, 当月, 同比增长, 环比增长, 累计, 累计-同比增长
        return _select_cols(df, "月份", ["当月"])

    # ---- 房地产 ----
    if key == "real_estate_investment":
        df = ak.macro_china_real_estate()
        # 列名: 日期, 最新值, 涨跌幅...
        return _select_cols(df, "日期", ["最新值"])

    raise ValueError(
        f"不支持的宏观指标: {key}。可用指标: {', '.join(CHINA_MACRO_SERIES.keys())}"
    )
