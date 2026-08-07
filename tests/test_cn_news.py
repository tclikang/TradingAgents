"""Unit tests for cn_news.py — 国内新闻数据源模块。

测试覆盖:
  - 东方财富个股新闻 (stock_news_em)
  - 东方财富个股研报 (stock_research_report_em)
  - 同花顺全球快讯 (stock_info_global_ths)
  - 新浪财经新闻   (stock_info_global_sina)
  - 聚合函数 fetch_all_cn_news
  - 降级行为（不抛异常、返回占位符）

Run:
    cd TradingAgents && python -m pytest tests/test_cn_news.py -v
"""

import pytest

from tradingagents.dataflows.cn_news import (
    fetch_all_cn_news,
    fetch_eastmoney_news,
    fetch_eastmoney_reports,
    fetch_sina_news,
    fetch_ths_news,
    _get_stock_code_only,
)

# 测试用的真实A股代码
TICKER_SS = "688016.SS"   # 心脉医疗 (科创板)
TICKER_SZ = "000001.SZ"   # 平安银行

# ============================================================
#  辅助函数
# ============================================================

class TestGetStockCodeOnly:
    def test_ss_suffix(self):
        assert _get_stock_code_only("688016.SS") == "688016"

    def test_sz_suffix(self):
        assert _get_stock_code_only("000001.SZ") == "000001"

    def test_hk_suffix(self):
        assert _get_stock_code_only("0700.HK") == "0700"

    def test_no_suffix(self):
        assert _get_stock_code_only("600519") == "600519"


# ============================================================
#  东方财富个股新闻
# ============================================================

class TestEastmoneyNews:
    def test_returns_non_empty_string(self):
        result = fetch_eastmoney_news(TICKER_SS, limit=5)
        assert isinstance(result, str)
        assert len(result) > 50, f"Result too short: {result[:100]}"

    def test_contains_ticker_info(self):
        result = fetch_eastmoney_news(TICKER_SS, limit=5)
        assert TICKER_SS in result

    def test_contains_sentiment_stats(self):
        result = fetch_eastmoney_news(TICKER_SS, limit=20)
        assert "情感统计" in result
        assert "偏多" in result
        assert "偏空" in result

    def test_limit_respects_count(self):
        """返回的条目数应该 <= limit。"""
        result = fetch_eastmoney_news(TICKER_SZ, limit=3)
        # 统计 [偏多] / [偏空] / 中性 标签
        tag_count = result.count("[偏多]") + result.count("[偏空]") + result.count("] ") 
        assert tag_count <= 4  # 允许 1 行 header

    def test_graceful_for_unknown_ticker(self):
        """对于无新闻的代码应友好降级。"""
        result = fetch_eastmoney_news("999999.SS", limit=5)
        assert isinstance(result, str)
        assert len(result) > 10

    def test_limit_zero_does_not_crash(self):
        result = fetch_eastmoney_news(TICKER_SS, limit=0)
        assert isinstance(result, str)


# ============================================================
#  东方财富研报
# ============================================================

class TestEastmoneyReports:
    def test_returns_non_empty_string(self):
        result = fetch_eastmoney_reports(TICKER_SS, limit=5)
        assert isinstance(result, str)
        assert len(result) > 50, f"Result too short: {result[:100]}"

    def test_contains_report_keywords(self):
        result = fetch_eastmoney_reports(TICKER_SS, limit=5)
        assert "评级" in result or "研报" in result or "暂无" in result

    def test_no_exception_for_empty_data(self):
        """某些冷门股可能无研报，不应抛异常。"""
        result = fetch_eastmoney_reports("999999.SS", limit=5)
        assert isinstance(result, str)


# ============================================================
#  同花顺快讯
# ============================================================

class TestThsNews:
    def test_returns_non_empty_string(self):
        result = fetch_ths_news(limit=10)
        assert isinstance(result, str)
        assert len(result) > 100, f"Result too short: {result[:100]}"

    def test_contains_chinese_content(self):
        """同花顺快讯应该是中文内容。"""
        result = fetch_ths_news(limit=5)
        # 检查是否包含中文字符
        has_chinese = any('\u4e00' <= c <= '\u9fff' for c in result)
        assert has_chinese, f"No Chinese characters found: {result[:200]}"

    def test_limit_respects_count(self):
        result = fetch_ths_news(limit=3)
        lines = [l for l in result.split('\n') if l.strip().startswith('[')]
        assert len(lines) <= 3

    def test_limit_zero_does_not_crash(self):
        result = fetch_ths_news(limit=0)
        assert isinstance(result, str)


# ============================================================
#  新浪财经
# ============================================================

class TestSinaNews:
    def test_returns_non_empty_string(self):
        result = fetch_sina_news(limit=10)
        assert isinstance(result, str)
        assert len(result) > 100, f"Result too short: {result[:100]}"

    def test_contains_time_stamps(self):
        """每条新闻应有时间戳。"""
        result = fetch_sina_news(limit=5)
        assert "2026" in result, f"No timestamp found: {result[:200]}"

    def test_limit_respects_count(self):
        result = fetch_sina_news(limit=3)
        lines = [l for l in result.split('\n') if l.strip().startswith('[')]
        assert len(lines) <= 3


# ============================================================
#  聚合函数
# ============================================================

class TestFetchAllCnNews:
    def test_returns_all_four_keys(self):
        news = fetch_all_cn_news(TICKER_SS, limit_per_source=5)
        assert isinstance(news, dict)
        expected_keys = {"eastmoney_news", "eastmoney_reports", "ths_news", "sina_news"}
        assert set(news.keys()) == expected_keys

    def test_all_values_are_non_empty_strings(self):
        news = fetch_all_cn_news(TICKER_SS, limit_per_source=5)
        for key, val in news.items():
            assert isinstance(val, str), f"{key} is not a string: {type(val)}"
            assert len(val) > 0, f"{key} is empty"


# ============================================================
#  集成: 降级行为
# ============================================================

class TestGracefulDegradation:
    """所有函数都不应抛异常，失败时返回描述性占位符。"""

    def test_eastmoney_news_no_exception(self):
        try:
            result = fetch_eastmoney_news("XXXXXX.XX", limit=5)
            assert isinstance(result, str)
        except Exception as e:
            pytest.fail(f"fetch_eastmoney_news raised {type(e).__name__}: {e}")

    def test_eastmoney_reports_no_exception(self):
        try:
            result = fetch_eastmoney_reports("XXXXXX.XX", limit=5)
            assert isinstance(result, str)
        except Exception as e:
            pytest.fail(f"fetch_eastmoney_reports raised {type(e).__name__}: {e}")

    def test_ths_news_no_exception(self):
        try:
            result = fetch_ths_news(limit=5)
            assert isinstance(result, str)
        except Exception as e:
            pytest.fail(f"fetch_ths_news raised {type(e).__name__}: {e}")

    def test_sina_news_no_exception(self):
        try:
            result = fetch_sina_news(limit=5)
            assert isinstance(result, str)
        except Exception as e:
            pytest.fail(f"fetch_sina_news raised {type(e).__name__}: {e}")

    def test_fetch_all_no_exception(self):
        try:
            result = fetch_all_cn_news(TICKER_SS, limit_per_source=3)
            assert isinstance(result, dict)
        except Exception as e:
            pytest.fail(f"fetch_all_cn_news raised {type(e).__name__}: {e}")
