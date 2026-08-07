from typing import Annotated

from langchain_core.tools import tool

from tradingagents.dataflows.interface import route_to_vendor


@tool
def get_macro_indicators(
    indicator: Annotated[
        str,
        "Macro indicator: a friendly alias such as 'cpi', 'ppi', 'pmi', "
        "'gdp', 'lpr利率', 'm2', '社融', '工业增加值', '外汇储备', "
        "or a raw AKShare series ID.",
    ],
    curr_date: Annotated[str, "Current date in yyyy-mm-dd format; the end of the window"],
    look_back_days: Annotated[
        int | None, "Trailing window length in days; omit for a 1-year window"
    ] = None,
) -> str:
    """
    Retrieve a macroeconomic indicator time series from AKShare (China macro):
    policy rates, inflation, labor, and growth. Returns the series title,
    units, frequency, the latest value, the change over the window, and a
    recent observation table. Uses the configured macro_data vendor
    (default: AKShare for Chinese indicators).

    Args:
        indicator (str): Friendly alias or raw series ID
        curr_date (str): Current date in yyyy-mm-dd format
        look_back_days (int): Trailing window length; omit for a 1-year window

    Returns:
        str: A formatted markdown report of the macro series
    """
    return route_to_vendor("get_macro_indicators", indicator, curr_date, look_back_days)
