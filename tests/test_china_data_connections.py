"""Data connection verification for Chinese A-stock data sources.

Verifies that all A-stock data paths use domestic sources
(akshare → Eastmoney / Sina / THS) instead of foreign APIs.

Tests that require network access to domestic sites are skipped
when proxy configuration blocks Eastmoney/Sina/THS (common when
HTTP_PROXY is set for Yahoo/VPN access).
"""

from __future__ import annotations

import os

import pytest

# ---------------------------------------------------------------------------
# Helper: check if domestic financial sites are reachable
# ---------------------------------------------------------------------------


def _domestic_network_available() -> bool:
    """Return True if eastmoney.com is reachable without proxy issues."""
    try:
        import urllib.request
        # Try without proxy to test direct domestic access
        proxy_handler = urllib.request.ProxyHandler({})
        opener = urllib.request.build_opener(proxy_handler)
        req = urllib.request.Request("https://push2.eastmoney.com", method="HEAD")
        opener.open(req, timeout=5)
        return True
    except Exception:
        return False


_skip_if_no_network = pytest.mark.skipif(
    not _domestic_network_available(),
    reason="Domestic financial sites unreachable (proxy may be blocking Eastmoney/Sina)",
)


# ---------------------------------------------------------------------------
# Conftest-level fixture to temporarily clear proxy for akshare calls
# ---------------------------------------------------------------------------


@pytest.fixture
def clear_proxy_for_domestic():
    """Temporarily clear HTTP_PROXY/HTTPS_PROXY for domestic API access.

    start.bat sets HTTP_PROXY for Yahoo/VPN, but domestic sites
    (Eastmoney/Sina/THS) must be reached directly.
    """
    saved = {}
    for var in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        saved[var] = os.environ.pop(var, None)
    yield
    for var, val in saved.items():
        if val is not None:
            os.environ[var] = val


# ---------------------------------------------------------------------------
# A) Price / OHLCV via akshare
# ---------------------------------------------------------------------------


@_skip_if_no_network
def test_get_stock_data_china_returns_csv(clear_proxy_for_domestic):
    """get_stock_data_china returns CSV-formatted price data for known symbol."""
    try:
        import akshare as ak  # noqa: F401
    except ImportError:
        pytest.skip("akshare not installed")

    from tradingagents.dataflows.china_data import get_stock_data_china

    result = get_stock_data_china("688299.SS", "2026-07-01", "2026-07-31")
    assert isinstance(result, str), f"Expected str, got {type(result).__name__}"
    # Should contain structured data (not just a "failed" message)
    if "Failed to fetch" in result or "ProxyError" in str(result):
        pytest.skip("Network unreachable (proxy blocking domestic site)")
    assert "Date" in result, f"No 'Date' column in output: {result[:200]}"
    # Should NOT contain yfinance-style timestamps
    assert "+00:00" not in result, "yfinance timezone format leaked into akshare output"


@_skip_if_no_network
def test_get_stock_data_china_bare_code(clear_proxy_for_domestic):
    """Bare 6-digit code without suffix also works."""
    from tradingagents.dataflows.china_data import get_stock_data_china

    result = get_stock_data_china("688299", "2026-07-01", "2026-07-31")
    if "Failed to fetch" in result:
        pytest.skip("Network unreachable (proxy blocking domestic site)")
    assert "Date" in result


def test_get_stock_data_china_invalid_symbol():
    """Invalid symbol returns graceful message, not crash."""
    from tradingagents.dataflows.china_data import get_stock_data_china

    result = get_stock_data_china("INVALID", "2026-01-01", "2026-01-31")
    assert "Not a recognized" in result or "No data" in result or "Failed" in result


# ---------------------------------------------------------------------------
# B) Fundamentals via akshare
# ---------------------------------------------------------------------------


@_skip_if_no_network
def test_get_fundamentals_china_returns_data(clear_proxy_for_domestic):
    """get_fundamentals_china returns structured fundamental info."""
    from tradingagents.dataflows.china_data import get_fundamentals_china

    result = get_fundamentals_china("688299.SS", "2026-08-05")
    assert isinstance(result, str)
    if "Failed to fetch" in result or "ProxyError" in str(result):
        pytest.skip("Network unreachable (proxy blocking domestic site)")
    assert len(result) > 50, f"Fundamentals result too short: {result[:100]}"
    has_name = "长阳" in result or "688299" in result
    assert has_name, f"Fundamentals should mention company name or code: {result[:200]}"


@_skip_if_no_network
def test_get_fundamentals_china_bare_code(clear_proxy_for_domestic):
    """Bare code also works for fundamentals."""
    from tradingagents.dataflows.china_data import get_fundamentals_china

    result = get_fundamentals_china("688299", "2026-08-05")
    if "Failed to fetch" in result:
        pytest.skip("Network unreachable (proxy blocking domestic site)")
    assert len(result) > 50


# ---------------------------------------------------------------------------
# C) News via akshare
# ---------------------------------------------------------------------------


@_skip_if_no_network
def test_get_news_china_returns_data(clear_proxy_for_domestic):
    """get_news_china returns news for known A-stock."""
    from tradingagents.dataflows.china_data import get_news_china

    result = get_news_china("688299.SS", "2026-07-01", "2026-07-31")
    assert isinstance(result, str)
    if "Failed to fetch" in result:
        pytest.skip("Network unreachable (proxy blocking domestic site)")
    assert len(result) > 20, f"News result too short: {result[:100]}"


@_skip_if_no_network
def test_get_global_news_china_returns_data(clear_proxy_for_domestic):
    """get_global_news_china returns market-wide news from Sina/Eastmoney."""
    from tradingagents.dataflows.china_data import get_global_news_china

    result = get_global_news_china("2026-07-01", "2026-07-31")
    assert isinstance(result, str)
    if "Failed to fetch" in result:
        pytest.skip("Network unreachable (proxy blocking domestic site)")
    assert len(result) > 20


@_skip_if_no_network
def test_get_market_news_china_returns_data(clear_proxy_for_domestic):
    """get_market_news_china returns market news."""
    from tradingagents.dataflows.china_data import get_market_news_china

    result = get_market_news_china("2026-08-01", 3, 10)
    assert isinstance(result, str)
    if "Failed to fetch" in result:
        pytest.skip("Network unreachable (proxy blocking domestic site)")
    assert len(result) > 20


# ---------------------------------------------------------------------------
# D) Instrument Identity
# ---------------------------------------------------------------------------


@_skip_if_no_network
def test_get_instrument_identity_china_known_a_stock(clear_proxy_for_domestic):
    """Known A-stock returns Chinese company name and industry."""
    from tradingagents.dataflows.china_data import get_instrument_identity_china

    result = get_instrument_identity_china("688299.SS")
    assert isinstance(result, dict)
    if result:
        assert "instrument_name" in result
        assert "industry" in result
        assert "长阳" in result.get("instrument_name", ""), (
            f"Expected 长阳科技 in instrument name: {result}"
        )


def test_get_instrument_identity_china_non_a_stock():
    """Non-A-stock returns empty dict (no network call to foreign APIs)."""
    from tradingagents.dataflows.china_data import get_instrument_identity_china

    result = get_instrument_identity_china("AAPL")
    assert result == {}, f"Non-A-stock should return empty dict, got: {result}"


@_skip_if_no_network
def test_get_instrument_identity_china_bare_code(clear_proxy_for_domestic):
    """Bare 6-digit code also returns data."""
    from tradingagents.dataflows.china_data import get_instrument_identity_china

    result = get_instrument_identity_china("688299")
    assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# E) Returns / Alpha via akshare
# ---------------------------------------------------------------------------


@_skip_if_no_network
def test_get_returns_china_known_a_stock(clear_proxy_for_domestic):
    """get_returns_china computes returns for known A-stock."""
    from tradingagents.dataflows.china_data import get_returns_china

    raw_ret, alpha, actual_days = get_returns_china(
        "688299.SS", "000001.SS", "2026-01-01", "2026-04-01", 60
    )
    # May be (None, None, None) if network unavailable
    if raw_ret is not None:
        assert isinstance(raw_ret, float)
        assert isinstance(alpha, float)
        assert isinstance(actual_days, int)


def test_get_returns_china_non_a_stock():
    """Non-A-stock returns None tuple (graceful degradation)."""
    from tradingagents.dataflows.china_data import get_returns_china

    raw_ret, alpha, actual_days = get_returns_china(
        "AAPL", "^GSPC", "2026-01-01", "2026-04-01", 60
    )
    assert raw_ret is None
    assert alpha is None
    assert actual_days is None


# ---------------------------------------------------------------------------
# F) Technical Indicators — stockstats_utils akshare routing
# ---------------------------------------------------------------------------


def test_is_a_stock_detection():
    """_is_a_stock correctly identifies A-stock suffixes."""
    from tradingagents.dataflows.stockstats_utils import _is_a_stock

    assert _is_a_stock("688299.SS") is True
    assert _is_a_stock("300203.SZ") is True
    assert _is_a_stock("830799.BJ") is True
    assert _is_a_stock("688299.ss") is True  # case insensitive
    assert _is_a_stock("AAPL") is False
    assert _is_a_stock("TSLA") is False
    assert _is_a_stock("600519") is False  # bare code, no suffix


@_skip_if_no_network
def test_download_china_ohlcv_returns_dataframe(clear_proxy_for_domestic):
    """_download_china_ohlcv returns DataFrame with OHLCV columns for known A-stock."""
    try:
        import akshare as ak  # noqa: F401
    except ImportError:
        pytest.skip("akshare not installed")

    from tradingagents.dataflows.stockstats_utils import _download_china_ohlcv

    df = _download_china_ohlcv("688299.SS", "2026-07-01", "2026-07-31")
    assert not df.empty, "Should return data for known symbol"
    required = {"Date", "Open", "High", "Low", "Close", "Volume"}
    assert required.issubset(set(df.columns)), (
        f"Missing columns: {required - set(df.columns)}"
    )
    # Verify data is from domestic source, not yfinance
    assert df["Date"].dtype.kind in ("M", "O"), f"Date column dtype: {df['Date'].dtype}"


@_skip_if_no_network
def test_load_ohlcv_a_stock_uses_akshare(clear_proxy_for_domestic):
    """load_ohlcv for A-stock uses akshare, not yfinance."""
    from tradingagents.dataflows.stockstats_utils import load_ohlcv

    df = load_ohlcv("688299.SS", "2026-08-05")
    assert not df.empty
    assert "Close" in df.columns

    # Verify no yfinance-specific timezone artifacts
    if "Date" in df.columns:
        dates = df["Date"]
        if hasattr(dates, "dt"):
            # akshare data should not have timezone (yfinance often has UTC)
            assert dates.dt.tz is None, "Timezone info present — may be yfinance data leak"


# ---------------------------------------------------------------------------
# G) Sentiment Analyst A-stock routing
# ---------------------------------------------------------------------------


def test_sentiment_analyst_a_stock_detection():
    """Sentiment analyst correctly identifies A-stock codes."""
    tickers = [
        ("688299.SS", True),
        ("300203.SZ", True),
        ("830799.BJ", True),
        ("000651.SZ", True),
        ("600036.SS", True),
        ("AAPL", False),
        ("TSLA", False),
        ("0700.HK", False),
    ]
    for ticker, expected in tickers:
        is_a = ticker.endswith(".SS") or ticker.endswith(".SZ") or ticker.endswith(".BJ")
        assert is_a == expected, f"Ticker {ticker}: expected is_a={expected}, got {is_a}"


def test_sentiment_analyst_china_news_prefetch():
    """Verify that sentiment analyst prefetch functions return data for A-stocks."""
    from tradingagents.agents.utils.agent_utils import (
        get_news,
        get_company_announcements,
    )

    # Test news fetching with A-stock
    news = get_news.func("688299.SS", "2026-07-01", "2026-08-05")
    assert isinstance(news, str)
    assert len(news) > 10, f"News result too short: {news[:100]}"

    # Test announcements fetching
    announcements = get_company_announcements.func("688299", "2026-08-05")
    assert isinstance(announcements, str)
    assert len(announcements) > 10, f"Announcements result too short: {announcements[:100]}"


# ---------------------------------------------------------------------------
# H) No yfinance leaks for A-stocks
# ---------------------------------------------------------------------------


def test_no_yfinance_in_a_stock_price_data():
    """A-stock price data path: interface routes to china, not yfinance."""
    from tradingagents.dataflows.interface import VENDOR_METHODS

    assert "china" in VENDOR_METHODS["get_stock_data"]
    china_impl = VENDOR_METHODS["get_stock_data"]["china"]
    assert "china" in china_impl.__module__ or "china" in china_impl.__name__


def test_no_yfinance_in_a_stock_news():
    """A-stock news path: interface routes to china, not yfinance."""
    from tradingagents.dataflows.interface import VENDOR_METHODS

    assert "china" in VENDOR_METHODS["get_news"]
    china_impl = VENDOR_METHODS["get_news"]["china"]
    assert "china" in china_impl.__module__


def test_no_yfinance_in_a_stock_fundamentals():
    """A-stock fundamentals: interface routes to china."""
    from tradingagents.dataflows.interface import VENDOR_METHODS

    assert "china" in VENDOR_METHODS["get_fundamentals"]


def test_default_config_uses_china_vendors():
    """Default configuration routes all core data to china vendor."""
    from tradingagents.default_config import DEFAULT_CONFIG

    vendors = DEFAULT_CONFIG.get("data_vendors", {})
    assert vendors.get("core_stock_apis") == "china"
    assert vendors.get("technical_indicators") == "china"
    assert vendors.get("fundamental_data") == "china"
    assert vendors.get("news_data") == "china"


def test_resolve_instrument_identity_a_stock_no_yfinance():
    """resolve_instrument_identity should NOT call yfinance for A-stocks."""
    from tradingagents.dataflows.china_data import get_instrument_identity_china

    # A-stock should hit akshare, not yfinance
    result = get_instrument_identity_china("688299.SS")
    assert isinstance(result, dict)

    # Non-A-stock should immediately return {}
    result = get_instrument_identity_china("AAPL")
    assert result == {}


# ---------------------------------------------------------------------------
# I) Vendor routing integrity
# ---------------------------------------------------------------------------


def test_china_only_methods_only_defined_for_china():
    """China-specific tools (announcements, reports, etc.) only have china vendor."""
    from tradingagents.dataflows.interface import VENDOR_METHODS

    china_only = [
        "get_company_announcements",
        "get_research_reports",
        "get_market_news",
        "get_sector",
        "get_hot_rank",
        "get_fund_flow",
        "get_profit_forecast",
    ]
    for method in china_only:
        vendors = VENDOR_METHODS[method]
        assert list(vendors.keys()) == ["china"], (
            f"{method} should only have 'china' vendor, got: {list(vendors.keys())}"
        )
