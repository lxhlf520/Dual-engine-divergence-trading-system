"""
多空分离策略引擎

核心设计（PRD 要求）：
  - 2H 级别：只允许 BUY（开多）和 CLOSE_LONG（平多）信号
  - 15M 级别：只允许 SELL（开空）和 CLOSE_SHORT（平空）信号

策略引擎通过 DataModule 的 bar_close 回调接入实时数据流。
同时支持回测模式：传入历史 DataFrame，逐 bar 模拟。
"""
from typing import Optional, List, Callable
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd

from ..config import TIMEFRAMES
from ..logger import get_logger
from .detector import (
    DivergenceDetector, DivergenceParams,
    TrailingStopUpdater,
    Signal, SignalType, DivergenceType,
)

logger = get_logger(__name__)


class Direction(Enum):
    LONG = "LONG"
    SHORT = "SHORT"


# ============================================================
# 做多引擎 (2H) 参数

LONG_PARAMS = DivergenceParams(
    rsi_period=14,
    atr_period=14,
    lb_l=1,
    lb_r=2,
    take_profit_rsi=80,         # 多单 RSI > 80 主动止盈
    stop_loss_mult=3.0,         # 多单 ATR 倍数 (Low - 3.0 * ATR)
    range_upper=60,
    range_lower=5,
    plot_bear=True,             # 常规熊背离 → 用于平多
    plot_hidden_bear=False,     # 2H 隐藏熊背离不参与做多引擎
    plot_bull=True,             # 常规牛背离 → 做多入场
)

# ============================================================
# 做空引擎 (15M) 参数

SHORT_PARAMS = DivergenceParams(
    rsi_period=14,
    atr_period=14,
    lb_l=1,
    lb_r=2,
    take_profit_rsi=25,         # 空单 RSI < 25 主动止盈
    stop_loss_mult=2.8,         # 空单 ATR 倍数 (High + 2.8 * ATR) — 稍收紧
    range_upper=60,
    range_lower=5,
    plot_bear=True,             # 常规熊背离 → 做空入场
    plot_hidden_bear=True,      # 隐藏熊背离 → 顺势反弹做空
    plot_bull=True,             # 常规牛背离 → 用于平空
)


# ============================================================
# 策略引擎
# ============================================================
@dataclass
class PositionState:
    """当前持仓状态"""
    direction: Optional[Direction] = None       # 持仓方向
    entry_price: float = 0.0
    trailing_sl: float = 0.0
    entry_bar_ts: int = 0
    bars_held: int = 0


class MultiEngineStrategy:
    """多空分离双引擎策略

    使用方式：
        strategy = MultiEngineStrategy()

        # 方式 1: 实盘通过回调接入 DataModule
        data_module.add_bar_close_callback(strategy.on_bar_close)

        # 方式 2: 回测逐根 bar 调用
        for bar in history:
            signals = strategy.process_bar(df, "15m")
    """

    def __init__(self):
        # 双引擎检测器
        self.long_detector = DivergenceDetector(LONG_PARAMS)
        self.short_detector = DivergenceDetector(SHORT_PARAMS)

        # 追踪止损更新器
        self.long_sl_updater = TrailingStopUpdater(stop_loss_mult=LONG_PARAMS.stop_loss_mult)
        self.short_sl_updater = TrailingStopUpdater(stop_loss_mult=SHORT_PARAMS.stop_loss_mult)

        # 持仓状态（实盘模式下维护）
        self.long_position: Optional[PositionState] = None
        self.short_position: Optional[PositionState] = None

        # 信号输出回调（供交易模块或回测模块注册）
        self._signal_callbacks: List[Callable[[Signal], None]] = []

    def on_signal(self, cb: Callable[[Signal], None]) -> None:
        """注册信号消费回调"""
        self._signal_callbacks.append(cb)

    def _emit_signal(self, signal: Signal) -> None:
        """向所有注册的回调广播信号"""
        for cb in self._signal_callbacks:
            try:
                cb(signal)
            except Exception as e:
                logger.error(f"Signal callback error: {e}", exc_info=True)

    # ============================================================
    # PRD 多空分离规则 — 核心信号过滤器
    # ============================================================
    def _filter_signals(self, signals: List[Signal], timeframe: str) -> List[Signal]:
        """多空分离过滤

        PRD 规定：
          2H → 只允许 BUY / CLOSE_LONG
          15M → 只允许 SELL / CLOSE_SHORT
        """
        if timeframe == "2h":
            allowed = {SignalType.BUY, SignalType.CLOSE_LONG}
        elif timeframe == "15m":
            allowed = {SignalType.SELL, SignalType.CLOSE_SHORT}
        else:
            logger.warning(f"Unknown timeframe '{timeframe}' — allowing all signals")
            return signals

        filtered = [s for s in signals if s.signal_type in allowed]
        dropped = len(signals) - len(filtered)

        if dropped > 0:
            dropped_types = [s.signal_type.value for s in signals if s.signal_type not in allowed]
            logger.debug(
                f"[{timeframe}] Filtered out {dropped} disallowed signals: {dropped_types}"
            )

        return filtered

    # ============================================================
    # 每 bar 收盘时调用（主入口）
    # ============================================================
    async def on_bar_close(
        self,
        timeframe: str,
        closed_bar: pd.Series,
        df: pd.DataFrame,
    ) -> None:
        """数据模块回调：当新 bar 收盘闭合时触发

        Args:
            timeframe: "15m" 或 "2h"
            closed_bar: 刚闭合的 K 线 (K线[1])
            df: 完整的在线 K 线矩阵
        """
        try:
            logger.debug(
                f"Strategy.on_bar_close [{timeframe}] "
                f"ts={int(closed_bar['timestamp'])} c={float(closed_bar['close']):.2f}"
            )

            signals = self.process_bar(df, timeframe)

            if signals:
                logger.info(f"[{timeframe}] {len(signals)} signal(s) generated")
                for sig in signals:
                    self._emit_signal(sig)

            # 更新持仓追踪止损
            self._update_trailing_stops(timeframe, df)

        except Exception as e:
            logger.error(f"Strategy.on_bar_close [{timeframe}] error: {e}", exc_info=True)

    def process_bar(self, df: pd.DataFrame, timeframe: str) -> List[Signal]:
        """处理单根 bar（供回测 / 实盘共用）

        1. 选择对应引擎检测背离
        2. 执行多空分离过滤
        3. 返回有效信号
        """
        # 选择检测器
        if timeframe == "2h":
            detector = self.long_detector
        elif timeframe == "15m":
            detector = self.short_detector
        else:
            raise ValueError(f"Unknown timeframe: {timeframe}")

        # 执行背离检测
        raw_signals = detector.detect(df, timeframe)

        if not raw_signals:
            return []

        # 多空分离过滤
        valid_signals = self._filter_signals(raw_signals, timeframe)

        return valid_signals

    # ============================================================
    # 追踪止损更新（实盘模式）
    # ============================================================
    def _update_trailing_stops(self, timeframe: str, df: pd.DataFrame) -> None:
        """用最新 bar 的 OHLC（K线[1]）更新持仓的追踪止损"""
        if df.empty:
            return

        latest = df.iloc[-2] if len(df) >= 2 else df.iloc[-1]  # K线[1]
        high = float(latest["high"])
        low = float(latest["low"])
        close = float(latest["close"])

        # 计算当前 ATR（从 indicators 模块）
        from .indicators import compute_atr
        atr_series = compute_atr(
            df["high"].astype(float),
            df["low"].astype(float),
            df["close"].astype(float),
            period=14
        )
        atr = float(atr_series.iloc[-1]) if not atr_series.empty else 0.0

        # --- 更新多头追踪止损 ---
        if self.long_position and self.long_position.direction == Direction.LONG:
            new_sl = self.long_sl_updater.update_long_sl(
                self.long_position.trailing_sl, low, atr
            )
            old_sl = self.long_position.trailing_sl
            self.long_position.trailing_sl = new_sl
            self.long_position.bars_held += 1

            if new_sl != old_sl:
                logger.debug(
                    f"Long trailing SL updated: {old_sl:.2f} -> {new_sl:.2f} "
                    f"(bars_held={self.long_position.bars_held})"
                )

            # 止损触发检查
            if self.long_sl_updater.check_long_stop(close, new_sl):
                logger.warning(
                    f"Long STOP triggered! close={close:.2f} sl={new_sl:.2f}"
                )
                self._emit_signal(Signal(
                    timestamp=int(latest["timestamp"]),
                    timeframe=timeframe,
                    signal_type=SignalType.CLOSE_LONG,
                    divergence_type=None,
                    price=close,
                    trailing_sl=new_sl,
                    metadata={"close_reason": "TSL_STOP"}
                ))
                self.long_position = None

        # --- 更新空头追踪止损 ---
        if self.short_position and self.short_position.direction == Direction.SHORT:
            new_sl = self.short_sl_updater.update_short_sl(
                self.short_position.trailing_sl, high, atr
            )
            old_sl = self.short_position.trailing_sl
            self.short_position.trailing_sl = new_sl
            self.short_position.bars_held += 1

            if new_sl != old_sl:
                logger.debug(
                    f"Short trailing SL updated: {old_sl:.2f} -> {new_sl:.2f} "
                    f"(bars_held={self.short_position.bars_held})"
                )

            # 止损触发检查
            if self.short_sl_updater.check_short_stop(close, new_sl):
                logger.warning(
                    f"Short STOP triggered! close={close:.2f} sl={new_sl:.2f}"
                )
                self._emit_signal(Signal(
                    timestamp=int(latest["timestamp"]),
                    timeframe=timeframe,
                    signal_type=SignalType.CLOSE_SHORT,
                    divergence_type=None,
                    price=close,
                    trailing_sl=new_sl,
                    metadata={"close_reason": "TSL_STOP"}
                ))
                self.short_position = None

    # ============================================================
    # 持仓管理（由交易模块在成交后回调）
    # ============================================================
    def on_position_opened(
        self,
        direction: Direction,
        entry_price: float,
        trailing_sl: float,
        bar_ts: int,
    ) -> None:
        """交易模块成交后通知策略"""
        pos = PositionState(
            direction=direction,
            entry_price=entry_price,
            trailing_sl=trailing_sl,
            entry_bar_ts=bar_ts,
            bars_held=0,
        )
        if direction == Direction.LONG:
            self.long_position = pos
        else:
            self.short_position = pos
        logger.info(f"Position opened: {direction.value} @ {entry_price:.2f} sl={trailing_sl:.2f}")

    def on_position_closed(self, direction: Direction) -> None:
        """交易模块平仓后通知策略"""
        if direction == Direction.LONG:
            self.long_position = None
        else:
            self.short_position = None
        logger.info(f"Position closed: {direction.value}")

    def get_long_sl(self) -> Optional[float]:
        """获取当前多头追踪止损价（供交易模块同步改单用）"""
        if self.long_position:
            return self.long_position.trailing_sl
        return None

    def get_short_sl(self) -> Optional[float]:
        """获取当前空头追踪止损价（供交易模块同步改单用）"""
        if self.short_position:
            return self.short_position.trailing_sl
        return None
