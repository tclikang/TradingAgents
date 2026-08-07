"""全面测试 AKShare 数据源集成。

覆盖：
  - OHLCV 股票价格数据 (akshare_stock.get_stock_data)
  - 技术指标 (akshare_stock.get_stock_stats_indicators_window)
  - 基本面数据 (akshare_stock.get_fundamentals)
  - 资产负债表 (akshare_stock.get_balance_sheet)
  - 现金流量表 (akshare_stock.get_cashflow)
  - 利润表 (akshare_stock.get_income_statement)
  - 高管交易 (akshare_stock.get_insider_transactions)
  - 个股新闻 (cn_news.get_news_cn)
  - 宏观新闻 (cn_news.get_global_news_cn)
  - 东方财富个股新闻 (cn_news.fetch_eastmoney_news)
  - 东方财富研报 (cn_news.fetch_eastmoney_reports)
  - 同花顺快讯 (cn_news.fetch_ths_news)
  - 新浪财经 (cn_news.fetch_sina_news)
  - 宏观指标 (akshare_macro.get_macro_data)
  - 接口路由 (interface.route_to_vendor for all methods)
  - 符号转换 (_as_akshare_code, _as_akshare_finance_code)
  - 默认配置 (data_vendors 全为 akshare)

测试用 A 股标的：
  - 688016.SS: 心脉医疗 (科创板)
  - 000001.SZ: 平安银行 (深圳主板)
  - 600519.SS: 贵州茅台 (上海主板)
"""

import copy
import os
import time
import unittest
from datetime import datetime

import pytest

from tradingagents.dataflows import akshare_macro, akshare_stock, cn_news, interface
from tradingagents.dataflows.config import get_config, set_config
from tradingagents.dataflows.errors import NoMarketDataError, VendorNotConfiguredError

# AKShare Network retry
_NET_RETRIES = 3
_NET_RETRY_DELAY = 3.0

# ---------------------------------------------------------------------------
# 测试配置隔离
# ---------------------------------------------------------------------------


def _ak_config():
    """返回纯 akshare 配置，无 yfinance/alpha_vantage 兜底。"""
    return {
        "data_vendors": {
            "core_stock_apis": "akshare",
            "technical_indicators": "akshare",
            "fundamental_data": "akshare",
            "news_data": "akshare",
            "macro_data": "akshare",
            "prediction_markets": "",
        },
    }


def _save_restore_config(f):
    """装饰器：测试前后保存/恢复配置。"""

    def wrapper(*args, **kwargs):
        saved = copy.deepcopy(get_config())
        try:
            return f(*args, **kwargs)
        finally:
            set_config(saved)

    return wrapper


# ---------------------------------------------------------------------------
# 符号转换
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSymbolConversion(unittest.TestCase):
    def test_as_akshare_code_ss(self):
        self.assertEqual(akshare_stock._as_akshare_code("688016.SS"), "688016")

    def test_as_akshare_code_sz(self):
        self.assertEqual(akshare_stock._as_akshare_code("000001.SZ"), "000001")

    def test_as_akshare_code_sh(self):
        self.assertEqual(akshare_stock._as_akshare_code("600519.SH"), "600519")

    def test_as_akshare_finance_code_ss(self):
        self.assertEqual(
            akshare_stock._as_akshare_finance_code("688016.SS"), "SH688016"
        )

    def test_as_akshare_finance_code_sz(self):
        self.assertEqual(
            akshare_stock._as_akshare_finance_code("000001.SZ"), "SZ000001"
        )

    def test_as_akshare_finance_code_sh(self):
        self.assertEqual(
            akshare_stock._as_akshare_finance_code("600519.SH"), "SH600519"
        )

    def test_date_to_akshare(self):
        self.assertEqual(akshare_stock._date_to_akshare("2026-01-15"), "20260115")


# ---------------------------------------------------------------------------
# OHLCV 股票数据
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestAkshareOHLCV(unittest.TestCase):
    def setUp(self):
        set_config(_ak_config())

    def _call_with_retry(self, func, *args):
        """Retry AKShare calls on transient network errors."""
        last_err = None
        for attempt in range(_NET_RETRIES):
            try:
                return func(*args)
            except NoMarketDataError as e:
                if "Connection" in str(e.__cause__) or "RemoteDisconnected" in str(e.__cause__):
                    last_err = e
                    if attempt < _NET_RETRIES - 1:
                        time.sleep(_NET_RETRY_DELAY)
                        continue
                raise
        raise last_err  # type: ignore[misc]

    def test_get_stock_data_returns_csv_string(self):
        result = self._call_with_retry(
            akshare_stock.get_stock_data, "688016.SS", "2026-01-01", "2026-06-30"
        )
        self.assertIsInstance(result, str)
        self.assertIn("Stock data for", result)
        self.assertIn("688016.SS", result)
        self.assertIn("Date", result)

    def test_get_stock_data_has_ohlcv(self):
        result = self._call_with_retry(
            akshare_stock.get_stock_data, "000001.SZ", "2026-01-01", "2026-06-30"
        )
        for col in ("Date", "Open", "High", "Low", "Close", "Volume"):
            self.assertIn(col, result, f"Missing column {col}")

    def test_get_stock_data_invalid_ticker_raises(self):
        with self.assertRaises(NoMarketDataError):
            akshare_stock.get_stock_data("999999.SS", "2026-01-01", "2026-01-10")

    def test_get_stock_data_600519(self):
        """贵州茅台应正常返回数据。"""
        result = self._call_with_retry(
            akshare_stock.get_stock_data, "600519.SS", "2026-01-01", "2026-06-30"
        )
        self.assertIn("600519.SS", result)
        self.assertIn("Close", result)


# ---------------------------------------------------------------------------
# 技术指标
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestAkshareTechnicalIndicators(unittest.TestCase):
    def setUp(self):
        set_config(_ak_config())

    def test_get_rsi(self):
        result = akshare_stock.get_stock_stats_indicators_window(
            "688016.SS", "rsi", "2026-06-15", 30
        )
        self.assertIsInstance(result, str)
        self.assertIn("rsi", result.lower())

    def test_get_macd(self):
        result = akshare_stock.get_stock_stats_indicators_window(
            "000001.SZ", "macd", "2026-06-15", 30
        )
        self.assertIn("macd", result.lower())

    def test_get_boll(self):
        result = akshare_stock.get_stock_stats_indicators_window(
            "600519.SS", "boll", "2026-06-15", 30
        )
        self.assertIn("boll", result.lower())

    def test_invalid_indicator_raises(self):
        with self.assertRaises(ValueError):
            akshare_stock.get_stock_stats_indicators_window(
                "688016.SS", "nonexistent_indicator", "2026-06-15", 30
            )


# ---------------------------------------------------------------------------
# 基本面数据
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestAkshareFundamentals(unittest.TestCase):
    def setUp(self):
        set_config(_ak_config())

    def test_get_fundamentals_returns_data(self):
        result = akshare_stock.get_fundamentals("688016.SS", "2026-06-30")
        self.assertIsInstance(result, str)
        self.assertIn("Fundamentals", result)
        self.assertIn("688016.SS", result)

    def test_get_fundamentals_has_key_metrics(self):
        result = akshare_stock.get_fundamentals("000001.SZ", "2026-06-30")
        # 应包含关键指标
        self.assertTrue(
            any(
                kw in result
                for kw in ["归母净利润", "营业总收入", "每股收益", "净资产收益率"]
            ),
            f"No key metric found in: {result[:500]}",
        )

    def test_get_fundamentals_600519(self):
        result = akshare_stock.get_fundamentals("600519.SS", "2026-06-30")
        self.assertIn("600519.SS", result)
        self.assertGreater(len(result), 50)


# ---------------------------------------------------------------------------
# 财务报表
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestAkshareFinancialStatements(unittest.TestCase):
    def setUp(self):
        set_config(_ak_config())

    def test_balance_sheet_quarterly(self):
        result = akshare_stock.get_balance_sheet("688016.SS", "quarterly")
        self.assertIsInstance(result, str)
        self.assertIn("资产负债表", result)
        self.assertIn("688016.SS", result)
        # 应包含核心科目
        self.assertIn("总资产", result)
        self.assertIn("总负债", result)

    def test_balance_sheet_annual(self):
        result = akshare_stock.get_balance_sheet("600519.SS", "annual")
        self.assertIn("资产负债表", result)
        self.assertIn("(annual)", result)  # annual mode tag in header

    def test_cashflow_quarterly(self):
        result = akshare_stock.get_cashflow("000001.SZ", "quarterly")
        self.assertIsInstance(result, str)
        self.assertIn("现金流量表", result)
        self.assertIn("经营活动", result)

    def test_income_statement_quarterly(self):
        result = akshare_stock.get_income_statement("688016.SS", "quarterly")
        self.assertIsInstance(result, str)
        self.assertIn("利润表", result)
        self.assertIn("营业总收入", result)

    def test_income_statement_annual(self):
        result = akshare_stock.get_income_statement("600519.SS", "annual")
        self.assertIn("利润表", result)
        self.assertIn("归母净利润", result)

    def test_balance_sheet_invalid_ticker(self):
        with self.assertRaises(NoMarketDataError):
            akshare_stock.get_balance_sheet("999999.SS", "quarterly")

    def test_income_statement_value_formatting(self):
        """数值应格式化为可读格式（亿/万）。"""
        result = akshare_stock.get_income_statement("600519.SS", "annual")
        # 茅台的营收应该在数百亿级别
        self.assertTrue("亿" in result or "万" in result, f"No unit formatting: {result[:300]}")


# ---------------------------------------------------------------------------
# 高管交易
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAkshareInsiderTransactions(unittest.TestCase):
    def test_returns_unavailable_message(self):
        result = akshare_stock.get_insider_transactions("688016.SS")
        self.assertIsInstance(result, str)
        self.assertIn("不可用", result)


# ---------------------------------------------------------------------------
# 新闻数据
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestAkshareNews(unittest.TestCase):
    def setUp(self):
        set_config(_ak_config())

    def test_get_news_cn(self):
        result = cn_news.get_news_cn("688016.SS")
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 50)
        self.assertIn("688016.SS", result)

    def test_get_news_cn_000001(self):
        result = cn_news.get_news_cn("000001.SZ")
        self.assertIn("000001.SZ", result)

    def test_get_global_news_cn(self):
        result = cn_news.get_global_news_cn(limit=10)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 50)
        # 应包含同花顺或新浪新闻
        self.assertTrue(
            "同花顺" in result or "新浪" in result or "快讯" in result,
            f"Unexpected content: {result[:300]}",
        )

    def test_get_global_news_cn_with_limit(self):
        result = cn_news.get_global_news_cn(limit=3)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 20)


# ---------------------------------------------------------------------------
# 宏观数据
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestAkshareMacro(unittest.TestCase):
    def setUp(self):
        set_config(_ak_config())

    def test_get_lpr_1y(self):
        result = akshare_macro.get_macro_data("lpr_1y", "2026-06-30")
        self.assertIsInstance(result, str)
        self.assertIn("lpr", result.lower())

    def test_get_cpi(self):
        result = akshare_macro.get_macro_data("cpi", "2026-06-30")
        self.assertIsInstance(result, str)
        self.assertIn("cpi", result.lower())

    def test_get_pmi(self):
        result = akshare_macro.get_macro_data("pmi", "2026-06-30")
        self.assertIsInstance(result, str)
        self.assertIn("pmi", result.lower())

    def test_get_m2(self):
        result = akshare_macro.get_macro_data("m2", "2026-06-30")
        self.assertIsInstance(result, str)

    def test_get_gdp(self):
        result = akshare_macro.get_macro_data("gdp", "2026-06-30")
        self.assertIsInstance(result, str)

    def test_chinese_alias_lpr(self):
        result = akshare_macro.get_macro_data("一年期lpr", "2026-06-30")
        self.assertIn("lpr", result.lower())

    def test_chinese_alias_cpi(self):
        result = akshare_macro.get_macro_data("居民消费价格指数", "2026-06-30")
        self.assertIn("cpi", result.lower())

    def test_shibor(self):
        result = akshare_macro.get_macro_data("shibor", "2026-06-30")
        self.assertIsInstance(result, str)

    def test_social_financing(self):
        result = akshare_macro.get_macro_data("社融", "2026-06-30")
        self.assertIsInstance(result, str)

    def test_unemployment(self):
        result = akshare_macro.get_macro_data("失业率", "2026-06-30")
        self.assertIsInstance(result, str)


# ---------------------------------------------------------------------------
# 接口路由 (interface.py)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestInterfaceRouting(unittest.TestCase):
    def setUp(self):
        set_config(_ak_config())

    def _call_with_retry(self, func, *args):
        """Retry on transient network errors."""
        last_err = None
        for attempt in range(_NET_RETRIES):
            try:
                return func(*args)
            except Exception as e:
                result = str(e)
                if "NO_DATA_AVAILABLE" in result or "Connection" in result or "RemoteDisconnected" in result:
                    # Check if NO_DATA_AVAILABLE was due to network
                    if "Connection" in result or "RemoteDisconnected" in result or "proxyerror" in result.lower():
                        last_err = e
                        if attempt < _NET_RETRIES - 1:
                            time.sleep(_NET_RETRY_DELAY)
                            continue
                    else:
                        raise
                else:
                    raise
        raise last_err  # type: ignore[misc]

    def _route(self, method, *args):
        return interface.route_to_vendor(method, *args)

    def test_route_get_stock_data(self):
        result = self._route("get_stock_data", "688016.SS", "2026-01-01", "2026-06-30")
        self.assertIsInstance(result, str)
        self.assertIn("688016.SS", result)

    def test_route_get_indicators(self):
        result = self._call_with_retry(
            self._route, "get_indicators", "688016.SS", "rsi", "2026-06-15", 30
        )
        self.assertIn("rsi", result.lower())

    def test_route_get_fundamentals(self):
        result = self._route("get_fundamentals", "600519.SS", "2026-06-30")
        self.assertIn("600519.SS", result)

    def test_route_get_balance_sheet(self):
        result = self._route("get_balance_sheet", "688016.SS", "quarterly")
        self.assertIn("资产负债表", result)

    def test_route_get_cashflow(self):
        result = self._route("get_cashflow", "000001.SZ", "quarterly")
        self.assertIn("现金流量表", result)

    def test_route_get_income_statement(self):
        result = self._route("get_income_statement", "600519.SS", "annual")
        self.assertIn("利润表", result)

    def test_route_get_news(self):
        result = self._route("get_news", "688016.SS", "2026-06-01", "2026-06-30")
        self.assertIsInstance(result, str)
        self.assertIn("688016.SS", result)

    def test_route_get_global_news(self):
        result = self._route("get_global_news", "2026-06-30", 7, 10)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 50)

    def test_route_get_insider_transactions(self):
        result = self._route("get_insider_transactions", "688016.SS")
        self.assertIn("不可用", result)

    def test_route_get_macro_indicators(self):
        result = self._route("get_macro_indicators", "lpr_1y", "2026-06-30", 365)
        self.assertIn("lpr", result.lower())

    def test_route_get_macro_cpi(self):
        result = self._route("get_macro_indicators", "cpi", "2026-06-30", 365)
        self.assertIn("cpi", result.lower())

    def test_route_get_macro_pmi(self):
        result = self._route("get_macro_indicators", "pmi", "2026-06-30", 365)
        self.assertIn("pmi", result.lower())

    def test_route_no_data_sentinel(self):
        """无效股票应返回 NO_DATA_AVAILABLE 哨兵字符串，而非异常。"""
        result = self._route("get_stock_data", "999999.SS", "2026-01-01", "2026-01-10")
        self.assertIn("NO_DATA_AVAILABLE", result)


# ---------------------------------------------------------------------------
# 默认配置验证
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDefaultConfig(unittest.TestCase):
    def setUp(self):
        from tradingagents import default_config as dc

        self.dc = dc
        set_config(copy.deepcopy(dc.DEFAULT_CONFIG))

    def test_all_vendors_are_akshare(self):
        cfg = get_config()
        vendors = cfg["data_vendors"]
        for category in ["core_stock_apis", "technical_indicators", "fundamental_data"]:
            self.assertEqual(
                vendors[category],
                "akshare",
                f"{category} should default to akshare",
            )
        self.assertEqual(vendors["news_data"], "akshare")
        self.assertEqual(vendors["macro_data"], "akshare")
        self.assertEqual(vendors["prediction_markets"], "")


# ---------------------------------------------------------------------------
# cn_news 详细测试
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestCnNewsComponents(unittest.TestCase):
    def setUp(self):
        set_config(_ak_config())

    def test_fetch_eastmoney_news_returns_content(self):
        result = cn_news.fetch_eastmoney_news("688016.SS", limit=5)
        self.assertIsInstance(result, str)
        self.assertIn("688016.SS", result)
        self.assertIn("情感统计", result)

    def test_fetch_eastmoney_news_contains_sentiment(self):
        result = cn_news.fetch_eastmoney_news("000001.SZ", limit=15)
        self.assertTrue("偏多" in result or "偏空" in result or "中性" in result)

    def test_fetch_eastmoney_reports(self):
        result = cn_news.fetch_eastmoney_reports("600519.SS", limit=5)
        self.assertIsInstance(result, str)
        self.assertIn("600519.SS", result)

    def test_fetch_ths_news(self):
        result = cn_news.fetch_ths_news(limit=5)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 50)

    def test_fetch_sina_news(self):
        result = cn_news.fetch_sina_news(limit=5)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 50)

    def test_fetch_all_cn_news(self):
        result = cn_news.fetch_all_cn_news("688016.SS", limit_per_source=5)
        self.assertIsInstance(result, dict)
        self.assertIn("eastmoney_news", result)
        self.assertIn("eastmoney_reports", result)
        self.assertIn("ths_news", result)
        self.assertIn("sina_news", result)
        for k, v in result.items():
            self.assertIsInstance(v, str, f"{k} should be str")
            self.assertGreater(len(v), 10, f"{k} too short: {v[:100]}")

    def test_graceful_degradation_unknown_ticker(self):
        """未知股票应友好降级，不抛异常。"""
        result = cn_news.fetch_eastmoney_news("999999.SS", limit=5)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 10)


# ---------------------------------------------------------------------------
# 代理绕过验证
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestProxyBypass(unittest.TestCase):
    def test_bypass_context_manager_works(self):
        """_akshare_bypass 应正常进入和退出，恢复环境变量。"""
        os.environ["HTTP_PROXY"] = "http://test.proxy:8080"
        try:
            with akshare_stock._akshare_bypass():
                # bypass 内部会弹出代理环境变量（强制直连），所以不应存在
                self.assertNotIn("HTTP_PROXY", os.environ)
            # 退出后应恢复
            self.assertEqual(os.environ.get("HTTP_PROXY"), "http://test.proxy:8080")
        finally:
            os.environ.pop("HTTP_PROXY", None)

    def test_bypass_does_not_raise_on_no_env(self):
        """即使没有代理环境变量也不报错。"""
        try:
            with akshare_stock._akshare_bypass():
                pass
        except Exception as e:
            self.fail(f"_akshare_bypass raised unexpectedly: {e}")


# ---------------------------------------------------------------------------
# 境外数据源彻底移除验证
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestForeignVendorsRemoved(unittest.TestCase):
    """验证所有境外数据源（yfinance, alpha_vantage, fred, polymarket, reddit, stocktwits）已彻底移除。"""

    def test_no_foreign_vendor_files_exist(self):
        """验证境外 vendor 模块文件已被删除。"""
        foreign_modules = [
            "y_finance.py",
            "yfinance_news.py",
            "fred.py",
            "polymarket.py",
            "alpha_vantage.py",
            "alpha_vantage_common.py",
            "alpha_vantage_fundamentals.py",
            "alpha_vantage_indicator.py",
            "alpha_vantage_news.py",
            "alpha_vantage_stock.py",
            "reddit.py",
            "stocktwits.py",
            "symbol_utils.py",
        ]
        from tradingagents.dataflows import __file__ as pkg_init
        import os
        dataflows_dir = os.path.dirname(os.path.abspath(pkg_init))
        for mod in foreign_modules:
            mod_path = os.path.join(dataflows_dir, mod)
            self.assertFalse(
                os.path.exists(mod_path),
                f"Foreign vendor module should not exist: {mod_path}",
            )

    def test_vendor_list_only_akshare(self):
        """VENDOR_LIST 应只包含 'akshare'。"""
        self.assertEqual(interface.VENDOR_LIST, ["akshare"])

    def test_vendor_methods_only_akshare(self):
        """VENDOR_METHODS 中所有方法应只用 'akshare' vendor。"""
        foreign_vendors = {"yfinance", "fred", "polymarket", "alpha_vantage"}
        for method, vendors in interface.VENDOR_METHODS.items():
            vendor_set = set(vendors.keys())
            overlap = vendor_set & foreign_vendors
            self.assertEqual(
                overlap, set(),
                f"Method '{method}' still maps to foreign vendors: {overlap}",
            )

    def test_get_prediction_markets_is_empty(self):
        """prediction_markets 应被禁用（空 dict）。"""
        vendors = interface.VENDOR_METHODS.get("get_prediction_markets", {})
        self.assertEqual(vendors, {}, "prediction_markets should be empty (disabled)")

    def test_no_yfinance_import_in_interface(self):
        """interface.py 不应 import yfinance / alpha_vantage / fred / polymarket。"""
        import inspect
        import tradingagents.dataflows.interface as iface_mod
        source = inspect.getsource(iface_mod)
        forbidden = ["yfinance", "y_finance", "alpha_vantage", "fred", "polymarket", "yfinance_news"]
        for keyword in forbidden:
            self.assertNotIn(
                f"from .{keyword}", source,
                f"interface.py should not import .{keyword}",
            )

    def test_no_yfinance_import_in_stockstats_utils(self):
        """stockstats_utils.py 不应 import yfinance。"""
        import inspect
        import tradingagents.dataflows.stockstats_utils as su_mod
        source = inspect.getsource(su_mod)
        self.assertNotIn("yfinance", source)
        self.assertNotIn("YFRateLimitError", source)
        self.assertNotIn("yf_retry", source)

    def test_no_yfinance_import_in_trading_graph(self):
        """trading_graph.py 不应 import yfinance。"""
        import inspect
        import tradingagents.graph.trading_graph as tg_mod
        source = inspect.getsource(tg_mod)
        self.assertNotIn("import yfinance", source)
        self.assertNotIn("from yfinance", source)

    def test_sentiment_analyst_no_yahoo(self):
        """sentiment_analyst.py 不应引用 Yahoo Finance。"""
        import inspect
        import tradingagents.agents.analysts.sentiment_analyst as sa_mod
        source = inspect.getsource(sa_mod)
        self.assertNotIn("yahoo_block", source)
        self.assertNotIn("Yahoo Finance", source)
        self.assertNotIn("start_of_yahoo", source)
        self.assertNotIn("end_of_yahoo", source)

    def test_load_ohlcv_rejects_non_astock(self):
        """load_ohlcv 应拒绝非 A 股代码。"""
        from tradingagents.dataflows.stockstats_utils import load_ohlcv
        with self.assertRaises(ValueError):
            load_ohlcv("AAPL", "2026-01-01")
        with self.assertRaises(ValueError):
            load_ohlcv("TSLA", "2026-01-01")
        with self.assertRaises(ValueError):
            load_ohlcv("SPY", "2026-01-01")
        with self.assertRaises(ValueError):
            load_ohlcv("GC=F", "2026-01-01")

    def test_cache_filename_uses_ohlcv_not_yfin(self):
        """缓存文件名应使用 'ohlcv' 而非 'YFin'。"""
        import inspect
        import tradingagents.dataflows.stockstats_utils as su_mod
        source = inspect.getsource(su_mod)
        self.assertNotIn("YFin-data", source, "Cache filename should use 'ohlcv' not 'YFin'")

    def test_benchmark_map_only_astock(self):
        """benchmark_map 应只包含 A 股指数。"""
        from tradingagents.default_config import DEFAULT_CONFIG
        bm = DEFAULT_CONFIG["benchmark_map"]
        allowed_suffixes = {".SS", ".SH", ".SZ", ".BJ"}
        for suffix in bm:
            self.assertIn(suffix, allowed_suffixes,
                          f"benchmark_map contains non-A-stock suffix: {suffix}")
        # 确认不包含境外指数
        foreign = {"^NSEI", "^BSESN", "^N225", "^HSI", "^FTSE", "^GSPTSE", "^AXJO", "SPY"}
        for val in bm.values():
            self.assertNotIn(val, foreign,
                             f"benchmark_map contains foreign index: {val}")

    def test_no_foreign_vendor_import_anywhere(self):
        """tradingagents 包中不应有任何地方 import yfinance 模块。"""
        import subprocess
        import sys
        import os
        tradingagents_dir = os.path.join(
            os.path.dirname(__file__), "..", "tradingagents"
        )
        result = subprocess.run(
            [
                sys.executable, "-c",
                f"import subprocess, sys; "
                f"r = subprocess.run(['grep', '-rl', 'import yfinance\\|from yfinance', "
                f"'{os.path.abspath(tradingagents_dir)}'], capture_output=True, text=True); "
                f"print(r.stdout); sys.exit(1 if r.stdout.strip() else 0)"
            ],
            capture_output=True, text=True,
        )
        self.assertEqual(
            result.returncode, 0,
            f"Found files still importing yfinance:\n{result.stdout}"
        )

    def test_prediction_markets_returns_unavailable(self):
        """get_prediction_markets 应返回不可用消息，不抛异常。"""
        from tradingagents.dataflows.interface import route_to_vendor
        result = route_to_vendor("get_prediction_markets", "test topic", 5)
        self.assertIsInstance(result, str)
        self.assertIn("DATA_UNAVAILABLE", result)

    def test_news_analyst_does_not_import_prediction_markets(self):
        """news_analyst.py 不应 import get_prediction_markets。"""
        import inspect
        import tradingagents.agents.analysts.news_analyst as na_mod
        source = inspect.getsource(na_mod)
        self.assertNotIn("get_prediction_markets", source)

    def test_trading_graph_does_not_import_prediction_markets(self):
        """trading_graph.py 不应 import get_prediction_markets。"""
        import inspect
        import tradingagents.graph.trading_graph as tg_mod
        source = inspect.getsource(tg_mod)
        self.assertNotIn("get_prediction_markets", source)


if __name__ == "__main__":
    unittest.main()
