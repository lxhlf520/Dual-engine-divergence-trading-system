"""
Divergence detection engine — pivot-pair comparison (revised).
Fixes the valuewhen(..., occurrence=1) self-comparison bug.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List
import numpy as np
import pandas as pd
from ..config import RSI_PERIOD, ATR_PERIOD, LB_L, LB_R
from ..logger import get_logger
from .indicators import compute_rsi, compute_atr, pivotlow, pivothigh

logger = get_logger(__name__)


class SignalType(Enum):
    BUY = "BUY"
    SELL = "SELL"
    CLOSE_LONG = "CLOSE_LONG"
    CLOSE_SHORT = "CLOSE_SHORT"


class DivergenceType(Enum):
    REGULAR_BULLISH = "regular_bullish"
    HIDDEN_BULLISH = "hidden_bullish"
    REGULAR_BEARISH = "regular_bearish"
    HIDDEN_BEARISH = "hidden_bearish"


@dataclass
class Signal:
    timestamp: int
    timeframe: str
    signal_type: SignalType
    divergence_type: Optional[DivergenceType] = None
    price: float = 0.0
    trailing_sl: Optional[float] = None
    rsi_value: Optional[float] = None
    atr_value: Optional[float] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class DivergenceParams:
    rsi_period: int = RSI_PERIOD
    atr_period: int = ATR_PERIOD
    lb_l: int = LB_L
    lb_r: int = LB_R
    take_profit_rsi: int = 25
    stop_loss_mult: float = 2.8
    range_lower: int = 5
    range_upper: int = 60
    plot_bear: bool = True
    plot_hidden_bear: bool = True
    plot_bull: bool = True


class DivergenceDetector:
    """Pivot-pair divergence detector."""

    def __init__(self, params: DivergenceParams):
        self.params = params

    def detect(self, df: pd.DataFrame, timeframe: str) -> List[Signal]:
        p = self.params
        if len(df) < max(p.rsi_period, p.atr_period) + 20:
            return []

        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)

        rsi = compute_rsi(close, period=p.rsi_period)
        atr = compute_atr(high, low, close, period=p.atr_period)

        ph = pivothigh(rsi, lbL=p.lb_l, lbR=p.lb_r)
        pl = pivotlow(rsi, lbL=p.lb_l, lbR=p.lb_r)

        ph_bars = ph[ph].index.tolist()
        pl_bars = pl[pl].index.tolist()

        signals = []
        timestamps = df["timestamp"].astype(int).values
        scan_end = len(df) - 2  # exclude unclosed bar

        # ---- Bearish divergences (uses pivot highs) ----
        for idx_a, idx_b in zip(ph_bars, ph_bars[1:]):
            if idx_b > scan_end:
                continue
            dist = idx_b - idx_a
            if not (p.range_lower <= dist <= p.range_upper):
                continue

            rsi_a = rsi.iloc[idx_a]
            rsi_b = rsi.iloc[idx_b]
            high_shift_a = high.shift(p.lb_r).iloc[idx_a] if not pd.isna(high.shift(p.lb_r).iloc[idx_a]) else high.iloc[idx_a]
            high_shift_b = high.shift(p.lb_r).iloc[idx_b] if not pd.isna(high.shift(p.lb_r).iloc[idx_b]) else high.iloc[idx_b]

            if np.isnan(rsi_a) or np.isnan(rsi_b):
                continue

            # Regular bearish: price higher high, RSI lower high
            if p.plot_bear and rsi_b < rsi_a and high_shift_b > high_shift_a:
                sl_price = float(high.iloc[idx_b]) + p.stop_loss_mult * float(atr.iloc[idx_b])
                signals.append(Signal(
                    timestamp=int(timestamps[idx_b]), timeframe=timeframe,
                    signal_type=SignalType.SELL, divergence_type=DivergenceType.REGULAR_BEARISH,
                    price=float(close.iloc[idx_b]), trailing_sl=sl_price,
                    rsi_value=float(rsi_b), atr_value=float(atr.iloc[idx_b]),
                    metadata={"pivot_a": idx_a, "pivot_b": idx_b, "rsi_a": float(rsi_a), "rsi_b": float(rsi_b)}
                ))

            # Hidden bearish: price lower high, RSI higher high
            if p.plot_hidden_bear and rsi_b > rsi_a and high_shift_b < high_shift_a:
                sl_price = float(high.iloc[idx_b]) + p.stop_loss_mult * float(atr.iloc[idx_b])
                signals.append(Signal(
                    timestamp=int(timestamps[idx_b]), timeframe=timeframe,
                    signal_type=SignalType.SELL, divergence_type=DivergenceType.HIDDEN_BEARISH,
                    price=float(close.iloc[idx_b]), trailing_sl=sl_price,
                    rsi_value=float(rsi_b), atr_value=float(atr.iloc[idx_b]),
                    metadata={"pivot_a": idx_a, "pivot_b": idx_b, "rsi_a": float(rsi_a), "rsi_b": float(rsi_b)}
                ))

        # ---- Bullish divergences (uses pivot lows) ----
        for idx_a, idx_b in zip(pl_bars, pl_bars[1:]):
            if idx_b > scan_end:
                continue
            dist = idx_b - idx_a
            if not (p.range_lower <= dist <= p.range_upper):
                continue

            rsi_a = rsi.iloc[idx_a]
            rsi_b = rsi.iloc[idx_b]
            low_shift_a = low.shift(p.lb_r).iloc[idx_a] if not pd.isna(low.shift(p.lb_r).iloc[idx_a]) else low.iloc[idx_a]
            low_shift_b = low.shift(p.lb_r).iloc[idx_b] if not pd.isna(low.shift(p.lb_r).iloc[idx_b]) else low.iloc[idx_b]

            if np.isnan(rsi_a) or np.isnan(rsi_b):
                continue

            # Regular bullish: price lower low, RSI higher low
            if p.plot_bull and rsi_b > rsi_a and low_shift_b < low_shift_a:
                sl_price = float(low.iloc[idx_b]) - p.stop_loss_mult * float(atr.iloc[idx_b])
                signals.append(Signal(
                    timestamp=int(timestamps[idx_b]), timeframe=timeframe,
                    signal_type=SignalType.BUY, divergence_type=DivergenceType.REGULAR_BULLISH,
                    price=float(close.iloc[idx_b]), trailing_sl=sl_price,
                    rsi_value=float(rsi_b), atr_value=float(atr.iloc[idx_b]),
                    metadata={"pivot_a": idx_a, "pivot_b": idx_b, "rsi_a": float(rsi_a), "rsi_b": float(rsi_b)}
                ))

            # Hidden bullish: price higher low, RSI lower low
            if p.plot_bull and rsi_b < rsi_a and low_shift_b > low_shift_a:
                sl_price = float(low.iloc[idx_b]) - p.stop_loss_mult * float(atr.iloc[idx_b])
                signals.append(Signal(
                    timestamp=int(timestamps[idx_b]), timeframe=timeframe,
                    signal_type=SignalType.BUY, divergence_type=DivergenceType.HIDDEN_BULLISH,
                    price=float(close.iloc[idx_b]), trailing_sl=sl_price,
                    rsi_value=float(rsi_b), atr_value=float(atr.iloc[idx_b]),
                    metadata={"pivot_a": idx_a, "pivot_b": idx_b, "rsi_a": float(rsi_a), "rsi_b": float(rsi_b)}
                ))

        # ---- Take-profit signals (RSI crossovers) ----
        for i in range(max(p.rsi_period + 5, 30), scan_end + 1):
            if i < 1:
                continue
            cb_close = float(close.iloc[i])
            cb_rsi = float(rsi.iloc[i]) if not pd.isna(rsi.iloc[i]) else 50.0
            cb_atr = float(atr.iloc[i]) if not pd.isna(atr.iloc[i]) else 0.0
            cb_ts = int(timestamps[i])

            # Short take-profit: RSI crosses below 25
            if i >= 2 and not pd.isna(rsi.iloc[i-1]):
                if rsi.iloc[i-1] > p.take_profit_rsi and rsi.iloc[i] <= p.take_profit_rsi:
                    signals.append(Signal(timestamp=cb_ts, timeframe=timeframe, signal_type=SignalType.CLOSE_SHORT, price=cb_close, rsi_value=cb_rsi, atr_value=cb_atr, metadata={"close_reason": "RSI_OVERSOLD"}))

            # Long take-profit: RSI crosses above 80
            if i >= 2 and not pd.isna(rsi.iloc[i-1]):
                if rsi.iloc[i-1] < 80 and rsi.iloc[i] >= 80:
                    signals.append(Signal(timestamp=cb_ts, timeframe=timeframe, signal_type=SignalType.CLOSE_LONG, price=cb_close, rsi_value=cb_rsi, atr_value=cb_atr, metadata={"close_reason": "RSI_OVERBOUGHT"}))

            # Bearish divergence can also trigger long close
            bear_bars = [s.metadata.get("pivot_b") for s in signals if s.signal_type == SignalType.SELL and s.divergence_type == DivergenceType.REGULAR_BEARISH]
            if i in bear_bars:
                signals.append(Signal(timestamp=cb_ts, timeframe=timeframe, signal_type=SignalType.CLOSE_LONG, divergence_type=DivergenceType.REGULAR_BEARISH, price=cb_close, rsi_value=cb_rsi, atr_value=cb_atr, metadata={"close_reason": "BEAR_DIV"}))

            # Bullish divergence can also trigger short close
            bull_bars = [s.metadata.get("pivot_b") for s in signals if s.signal_type == SignalType.BUY and s.divergence_type == DivergenceType.REGULAR_BULLISH]
            if i in bull_bars:
                signals.append(Signal(timestamp=cb_ts, timeframe=timeframe, signal_type=SignalType.CLOSE_SHORT, divergence_type=DivergenceType.REGULAR_BULLISH, price=cb_close, rsi_value=cb_rsi, atr_value=cb_atr, metadata={"close_reason": "BULL_DIV"}))

        return signals


class TrailingStopUpdater:
    def __init__(self, stop_loss_mult: float = 2.8):
        self.stop_loss_mult = stop_loss_mult

    def update_long_sl(self, prev_sl: Optional[float], low: float, atr: float) -> float:
        new_sl = low - self.stop_loss_mult * atr
        if prev_sl is None or np.isnan(prev_sl):
            return new_sl
        return max(prev_sl, new_sl)

    def update_short_sl(self, prev_sl: Optional[float], high: float, atr: float) -> float:
        new_sl = high + self.stop_loss_mult * atr
        if prev_sl is None or np.isnan(prev_sl):
            return new_sl
        return min(prev_sl, new_sl)

    def check_long_stop(self, close: float, trailing_sl: float) -> bool:
        return close <= trailing_sl

    def check_short_stop(self, close: float, trailing_sl: float) -> bool:
        return close >= trailing_sl
