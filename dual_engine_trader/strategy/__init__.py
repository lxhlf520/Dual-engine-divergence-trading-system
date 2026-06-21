"""
策略模块 统一入口
"""
from .detector import (
    DivergenceDetector,
    DivergenceParams,
    TrailingStopUpdater,
    Signal,
    SignalType,
    DivergenceType,
)
from .engine import (
    MultiEngineStrategy,
    PositionState,
    Direction,
    LONG_PARAMS,
    SHORT_PARAMS,
)
from .indicators import (
    compute_rsi,
    compute_atr,
    pivotlow,
    pivothigh,
    valuewhen,
    barssince,
)

__all__ = [
    "DivergenceDetector",
    "DivergenceParams",
    "TrailingStopUpdater",
    "Signal",
    "SignalType",
    "DivergenceType",
    "Direction",
    "MultiEngineStrategy",
    "PositionState",
    "LONG_PARAMS",
    "SHORT_PARAMS",
    "compute_rsi",
    "compute_atr",
    "pivotlow",
    "pivothigh",
    "valuewhen",
    "barssince",
]
