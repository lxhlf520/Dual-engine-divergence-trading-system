"""
回测模块 — 虚拟账户账本
维护资金、持仓、手续费、滑点磨损等完整状态
"""
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from enum import Enum
import pandas as pd

from ..config import (
    BACKTEST_INITIAL_CAPITAL,
    BACKTEST_FEE_RATE,
    BACKTEST_SLIPPAGE_RATE,
    BACKTEST_PYRAMIDING,
)
from ..strategy.detector import SignalType
from ..strategy.engine import Direction


@dataclass
class Position:
    """单笔持仓记录"""
    direction: Direction           # LONG or SHORT
    entry_price: float             # 入场价
    entry_ts: int                  # 入场时间戳 (ms)
    quantity: float                # 持仓数量（合约张数等价）
    trailing_sl: float             # 当前追踪止损价
    entry_bar_index: int           # 入场 K 线索引
    sl_log: List[float] = field(default_factory=list)  # 止损线历史


@dataclass
class Trade:
    """单笔完整交易记录（开仓→平仓）"""
    direction: Direction
    entry_time: int                # 入场时间戳 ms
    exit_time: int                 # 出场时间戳 ms
    entry_price: float
    exit_price: float
    quantity: float
    entry_fee: float               # 入场手续费
    exit_fee: float                # 出场手续费
    slippage: float                # 总滑点磨损
    net_pnl: float                 # 净利润（扣除费用后）
    net_pnl_pct: float             # 收益率
    exit_reason: str               # 平仓原因
    divergence_type: str = ""      # 入场背离类型
    bars_held: int = 0             # 持仓 bar 数


@dataclass
class EquityPoint:
    """权益曲线上的一个点"""
    timestamp: int
    equity: float
    drawdown: float                # 当前回撤（百分比）


class VirtualAccount:
    """虚拟账户——完全模拟实盘行为

    核心规则（PRD 3.3）：
    - 初始资金 $10,000
    - 单边手续费 0.05% (taker)
    - 1 bp 滑点磨损
    - pyramiding=2（同向最多 2 次加仓）
    - 做多止损只能上移，做空止损只能下移
    """

    def __init__(
        self,
        initial_capital: float = BACKTEST_INITIAL_CAPITAL,
        fee_rate: float = BACKTEST_FEE_RATE,
        slippage_rate: float = BACKTEST_SLIPPAGE_RATE,
        max_pyramiding: int = BACKTEST_PYRAMIDING,
    ):
        self.initial_capital = initial_capital
        self.capital = initial_capital          # 当前可用资金
        self.fee_rate = fee_rate
        self.slippage_rate = slippage_rate
        self.max_pyramiding = max_pyramiding

        # 持仓
        self.long_positions: List[Position] = []
        self.short_positions: List[Position] = []

        # 交易历史
        self.trades: List[Trade] = []
        self.closed_trades: List[Trade] = []    # 已平仓

        # 权益曲线
        self.equity_curve: List[EquityPoint] = []

        # 统计
        self.peak_equity = initial_capital
        self.max_drawdown_pct = 0.0
        self.total_trades = 0

    # ============================================================
    # 资金与权益
    # ============================================================
    @property
    def total_equity(self) -> float:
        """总权益 = 现金 + 持仓浮动盈亏"""
        equity = self.capital
        # 简化处理：回测中持仓按入场价计算占用保证金，浮动盈亏加到权益
        # 实际 OKX 永续合约以 USDT 结算，这里用名义价值简化
        for pos in self.long_positions + self.short_positions:
            pass  # 浮动盈亏在平仓时结算
        return equity

    def record_equity(self, timestamp: int) -> None:
        """记录当前权益快照"""
        equity = self.capital
        drawdown = (self.peak_equity - equity) / self.peak_equity * 100 if self.peak_equity > 0 else 0
        if equity > self.peak_equity:
            self.peak_equity = equity
        self.max_drawdown_pct = max(self.max_drawdown_pct, drawdown)
        self.equity_curve.append(EquityPoint(
            timestamp=timestamp,
            equity=equity,
            drawdown=drawdown,
        ))

    # ============================================================
    # 开仓
    # ============================================================
    def open_position(
        self,
        direction: Direction,
        price: float,
        timestamp: int,
        bar_index: int,
        trailing_sl: float,
        divergence_type: str = "",
        quantity: float = 1.0,     # 每笔固定 1 单位
    ) -> Optional[Position]:
        """尝试开仓，返回 Position 或 None（余额不足/加仓上限）"""
        # 检查 pyramiding 上限
        positions = self.long_positions if direction == Direction.LONG else self.short_positions
        if len(positions) >= self.max_pyramiding:
            return None

        # 计算费用（市价单 taker fee + 滑点）
        notional = price * quantity
        fee = notional * self.fee_rate
        slippage = notional * self.slippage_rate

        # 资金检查（考虑杠杆，回测中暂不模拟杠杆可用余额细节）
        total_cost = fee + slippage
        if self.capital < total_cost:
            return None

        # 扣费
        self.capital -= total_cost

        # 创建持仓
        pos = Position(
            direction=direction,
            entry_price=price,
            entry_ts=timestamp,
            quantity=quantity,
            trailing_sl=trailing_sl,
            entry_bar_index=bar_index,
            sl_log=[trailing_sl],
        )
        positions.append(pos)
        self.total_trades += 1

        return pos

    # ============================================================
    # 平仓
    # ============================================================
    def close_position(
        self,
        direction: Direction,
        price: float,
        timestamp: int,
        bar_index: int,
        reason: str,
    ) -> Optional[Trade]:
        """平掉最近一笔同向持仓，返回 Trade 记录"""
        positions = self.long_positions if direction == Direction.LONG else self.short_positions
        if not positions:
            return None

        pos = positions.pop(0)  # FIFO 平仓
        bars_held = bar_index - pos.entry_bar_index

        # 计算盈亏
        notional = pos.entry_price * pos.quantity
        exit_notional = price * pos.quantity

        if direction == Direction.LONG:
            gross_pnl = (price - pos.entry_price) * pos.quantity
        else:
            gross_pnl = (pos.entry_price - price) * pos.quantity

        # 费用
        entry_fee = notional * self.fee_rate
        exit_fee = exit_notional * self.fee_rate
        slippage = notional * self.slippage_rate + exit_notional * self.slippage_rate

        net_pnl = gross_pnl - entry_fee - exit_fee - slippage
        net_pnl_pct = (net_pnl / notional) * 100 if notional > 0 else 0

        # 结算
        self.capital += exit_notional - exit_fee - slippage

        trade = Trade(
            direction=direction,
            entry_time=pos.entry_ts,
            exit_time=timestamp,
            entry_price=pos.entry_price,
            exit_price=price,
            quantity=pos.quantity,
            entry_fee=entry_fee,
            exit_fee=exit_fee,
            slippage=slippage,
            net_pnl=net_pnl,
            net_pnl_pct=net_pnl_pct,
            exit_reason=reason,
            divergence_type=getattr(pos, 'divergence_type', ''),
            bars_held=bars_held,
        )
        self.closed_trades.append(trade)
        self.trades.append(trade)

        return trade

    # ============================================================
    # 追踪止损更新（每 bar 调用）
    # ============================================================
    def update_trailing_sl_long(self, new_sl: float, position_index: int = 0) -> None:
        """更新多头止损线（只能上移）"""
        if position_index < len(self.long_positions):
            pos = self.long_positions[position_index]
            if new_sl > pos.trailing_sl:
                pos.trailing_sl = new_sl
                pos.sl_log.append(new_sl)

    def update_trailing_sl_short(self, new_sl: float, position_index: int = 0) -> None:
        """更新空头止损线（只能下移）"""
        if position_index < len(self.short_positions):
            pos = self.short_positions[position_index]
            if new_sl < pos.trailing_sl:
                pos.trailing_sl = new_sl
                pos.sl_log.append(new_sl)

    # ============================================================
    # 止损触发检查（每 bar 调用）
    # ============================================================
    def check_long_stops(self, bar_low: float, bar_high: float,
                         timestamp: int, bar_index: int) -> List[Trade]:
        """检查所有多头持仓的止损（价格跌破止损线）"""
        closed = []
        remaining = []
        for pos in self.long_positions:
            if bar_low <= pos.trailing_sl:
                # 触发止损：以止损价或更低价格平仓
                exit_price = min(pos.trailing_sl, bar_low)
                trade = self._force_close(pos, exit_price, timestamp, bar_index, "TSL_STOP")
                closed.append(trade)
            else:
                remaining.append(pos)
        self.long_positions = remaining
        return closed

    def check_short_stops(self, bar_low: float, bar_high: float,
                          timestamp: int, bar_index: int) -> List[Trade]:
        """检查所有空头持仓的止损（价格突破止损线）"""
        closed = []
        remaining = []
        for pos in self.short_positions:
            if bar_high >= pos.trailing_sl:
                exit_price = max(pos.trailing_sl, bar_high)
                trade = self._force_close(pos, exit_price, timestamp, bar_index, "TSL_STOP")
                closed.append(trade)
            else:
                remaining.append(pos)
        self.short_positions = remaining
        return closed

    def _force_close(self, pos: Position, exit_price: float,
                     timestamp: int, bar_index: int, reason: str) -> Trade:
        """内部强制平仓"""
        notional = pos.entry_price * pos.quantity
        exit_notional = exit_price * pos.quantity

        if pos.direction == Direction.LONG:
            gross_pnl = (exit_price - pos.entry_price) * pos.quantity
        else:
            gross_pnl = (pos.entry_price - exit_price) * pos.quantity

        entry_fee = notional * self.fee_rate
        exit_fee = exit_notional * self.fee_rate
        slippage = (notional + exit_notional) * self.slippage_rate

        net_pnl = gross_pnl - entry_fee - exit_fee - slippage
        net_pnl_pct = (net_pnl / notional) * 100 if notional > 0 else 0

        self.capital += exit_notional - exit_fee - slippage

        trade = Trade(
            direction=pos.direction,
            entry_time=pos.entry_ts,
            exit_time=timestamp,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            quantity=pos.quantity,
            entry_fee=entry_fee,
            exit_fee=exit_fee,
            slippage=slippage,
            net_pnl=net_pnl,
            net_pnl_pct=net_pnl_pct,
            exit_reason=reason,
            bars_held=bar_index - pos.entry_bar_index,
        )
        self.closed_trades.append(trade)
        self.trades.append(trade)
        return trade

    def get_position_count_long(self) -> int:
        return len(self.long_positions)

    def get_position_count_short(self) -> int:
        return len(self.short_positions)
