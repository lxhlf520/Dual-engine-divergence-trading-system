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
    DEFAULT_LEVERAGE,
    BACKTEST_MAX_RISK_PCT,
    BACKTEST_MAX_CAPITAL_PCT,
    BACKTEST_MAX_CONTRACTS,
)
from ..strategy.engine import Direction


@dataclass
class Position:
    """单笔持仓记录"""
    direction: Direction           # LONG or SHORT
    entry_price: float             # 入场价
    entry_ts: int                  # 入场时间戳 (ms)
    quantity: float                # 持仓数量（合约张数等价）
    margin: float                  # 已锁定的保证金 = entry_price * quantity / leverage
    entry_fee: float               # 开仓时已扣除的手续费
    entry_slippage: float          # 开仓时已扣除的滑点
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
    """虚拟账户——永续合约仿真

    核心规则：
    - 初始资金 $10,000
    - 开仓锁定保证金 = 名义价值 / 杠杆 (DEFAULT_LEVERAGE)
    - 单边手续费 0.05% (taker)
    - 1 bp 滑点磨损
    - pyramiding=2（同向最多 2 次加仓）
    - 做多止损只能上移，做空止损只能下移
    - 总权益 = 可用资金 + 全部持仓保证金 + 浮动盈亏
    """

    def __init__(
        self,
        initial_capital: float = BACKTEST_INITIAL_CAPITAL,
        fee_rate: float = BACKTEST_FEE_RATE,
        slippage_rate: float = BACKTEST_SLIPPAGE_RATE,
        max_pyramiding: int = BACKTEST_PYRAMIDING,
        leverage: int = DEFAULT_LEVERAGE,
        contract_size: float = 0.01,   # BTC/USDT 永续合约每张 0.01 BTC
        max_risk_pct: float = BACKTEST_MAX_RISK_PCT,
        max_capital_pct: float = BACKTEST_MAX_CAPITAL_PCT,
        max_contracts: float = BACKTEST_MAX_CONTRACTS,
    ):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.fee_rate = fee_rate
        self.slippage_rate = slippage_rate
        self.max_pyramiding = max_pyramiding
        self.leverage = leverage
        self.contract_size = contract_size
        self.max_risk_pct = max_risk_pct
        self.max_capital_pct = max_capital_pct
        self.max_contracts = max_contracts

        # 持仓
        self.long_positions: List[Position] = []
        self.short_positions: List[Position] = []

        # 交易历史
        self.trades: List[Trade] = []
        self.closed_trades: List[Trade] = []

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
        """总权益 = 可用资金 + 全部持仓的锁定保证金"""
        locked_margin = sum(p.margin for p in self.long_positions + self.short_positions)
        return self.capital + locked_margin

    def record_equity(self, timestamp: int) -> None:
        """记录当前权益快照"""
        equity = self.total_equity
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
    # 风险定仓
    # ============================================================
    def compute_contracts(self, price: float, stop_loss_price: float, equity: float = None) -> float:
        """基于风控参数计算开仓张数（复制 OKX executor._compute_safe_contracts 逻辑）

        约束：
        - ct_by_risk: 单笔亏损不超过 equity × max_risk_pct%
        - ct_by_margin: 占用保证金不超过 equity × max_capital_pct%
        - ct_by_free: 不超过可用资金能覆盖的保证金
        - max_contracts: 绝对上限

        Returns: 合约张数，若 < 0.01 表示无法开仓
        """
        if stop_loss_price <= 0 or price <= 0:
            return 0.0

        eq = equity if equity is not None else self.total_equity
        notional_per_ct = price * self.contract_size
        margin_per_ct = notional_per_ct / self.leverage
        sl_dist = abs(stop_loss_price - price) / price
        risk_per_ct = notional_per_ct * sl_dist

        ct_by_margin = (eq * self.max_capital_pct / 100) / margin_per_ct if margin_per_ct > 0 else 0
        ct_by_risk = (eq * self.max_risk_pct / 100) / risk_per_ct if risk_per_ct > 0 else float('inf')
        ct_by_free = self.capital / margin_per_ct if margin_per_ct > 0 else 0

        raw_ct = min(ct_by_margin, ct_by_risk, ct_by_free, self.max_contracts)
        sz = int(raw_ct * 100) / 100.0
        return max(sz, 0.0)

    # ============================================================
    # 开仓（支持自动风险定仓）
    # ============================================================
    def open_position(
        self,
        direction: Direction,
        price: float,
        timestamp: int,
        bar_index: int,
        trailing_sl: float,
        divergence_type: str = "",
        quantity: Optional[float] = None,
    ) -> Optional[Position]:
        """尝试开仓，返回 Position 或 None（余额不足/加仓上限）

        若 quantity 为 None，通过 stop_loss_price=trailing_sl 自动定仓。
        """
        # 检查 pyramiding 上限
        positions = self.long_positions if direction == Direction.LONG else self.short_positions
        if len(positions) >= self.max_pyramiding:
            return None

        # 自动定仓：基于止损距离计算合约数
        if quantity is None:
            quantity = self.compute_contracts(price, trailing_sl)
            if quantity < 0.01:
                return None

        notional = price * quantity * self.contract_size
        margin = notional / self.leverage
        fee = notional * self.fee_rate
        slippage = notional * self.slippage_rate

        # 资金检查：可用资金要能覆盖保证金 + 费用
        if self.capital < margin + fee + slippage:
            return None

        # 冻结保证金 + 扣费
        self.capital -= (margin + fee + slippage)

        pos = Position(
            direction=direction,
            entry_price=price,
            entry_ts=timestamp,
            quantity=quantity,
            margin=margin,
            entry_fee=fee,
            entry_slippage=slippage,
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
        """平掉最近一笔同向持仓，返回 Trade 记录

        真实永续合约逻辑：
        - 释放保证金
        - 计算盈亏：(exit_price - entry_price) * quantity * (±1)
        - 扣除平仓费用和滑点
        - 最终资金 = 原有可用 + 释放保证金 + 盈亏 - 平仓费用 - 滑点
        """
        positions = self.long_positions if direction == Direction.LONG else self.short_positions
        if not positions:
            return None

        pos = positions.pop(0)  # FIFO 平仓
        bars_held = bar_index - pos.entry_bar_index

        if direction == Direction.LONG:
            gross_pnl = (price - pos.entry_price) * pos.quantity * self.contract_size
        else:
            gross_pnl = (pos.entry_price - price) * pos.quantity * self.contract_size

        # 费用（平仓按 exit notional 算）
        exit_notional = price * pos.quantity * self.contract_size
        exit_fee = exit_notional * self.fee_rate
        slippage = exit_notional * self.slippage_rate

        net_pnl = gross_pnl - exit_fee - slippage
        net_pnl_pct = (net_pnl / pos.margin) * 100 if pos.margin > 0 else 0

        # 资金结算：释放保证金 + 净盈亏 - 平仓费用
        self.capital += pos.margin + net_pnl

        trade = Trade(
            direction=direction,
            entry_time=pos.entry_ts,
            exit_time=timestamp,
            entry_price=pos.entry_price,
            exit_price=price,
            quantity=pos.quantity,
            entry_fee=pos.entry_fee,
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
        exit_notional = exit_price * pos.quantity * self.contract_size

        if pos.direction == Direction.LONG:
            gross_pnl = (exit_price - pos.entry_price) * pos.quantity * self.contract_size
        else:
            gross_pnl = (pos.entry_price - exit_price) * pos.quantity * self.contract_size

        exit_fee = exit_notional * self.fee_rate
        slippage = exit_notional * self.slippage_rate

        net_pnl = gross_pnl - exit_fee - slippage
        net_pnl_pct = (net_pnl / pos.margin) * 100 if pos.margin > 0 else 0

        self.capital += pos.margin + net_pnl

        trade = Trade(
            direction=pos.direction,
            entry_time=pos.entry_ts,
            exit_time=timestamp,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            quantity=pos.quantity,
            entry_fee=pos.entry_fee,
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
