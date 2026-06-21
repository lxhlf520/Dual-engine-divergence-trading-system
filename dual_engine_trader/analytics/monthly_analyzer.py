"""
月度复盘系统 — MonthlyAnalyzer 核心引擎

核心账目比对逻辑:
  实际收益 = 实盘成交盈亏 - 手续费 - 滑点磨损 + 资金费率净收支
  理论收益 = 回测引擎在同一信号下模拟的盈亏（忽略延迟、滑点）
  偏差 = 实际收益 - 理论收益 → 量化执行质量
"""
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd

from ..config import ROOT_DIR, LOG_DIR, DEFAULT_LEVERAGE
from ..logger import get_logger
from ..strategy.indicators import compute_rsi, compute_atr, pivotlow, pivothigh
from ..strategy.detector import DivergenceDetector, DivergenceParams, TrailingStopUpdater, SignalType
from ..strategy.engine import Direction, SHORT_PARAMS
from ..backtest.account import VirtualAccount
from .monthly_types import (
    LiveTradeEntry, PnLDiff, FundingRecord, MonthlySnapshot,
)

logger = get_logger(__name__)


class MonthlyAnalyzer:
    """月度复盘分析引擎

    使用方式:
        analyzer = MonthlyAnalyzer(2026, 6)
        analyzer.load_live_trades("path/to/trade_log.csv")
        analyzer.load_kline_data("path/to/btc_15m.csv")
        analyzer.load_funding_rates("path/to/funding.csv")   # optional
        snapshot = analyzer.run()
        analyzer.generate_report("output/monthly_june.md")
        analyzer.generate_chart("output/monthly_june.png")
    """

    def __init__(self, year: int, month: int, initial_capital: float = 10000.0):
        self.year = year
        self.month = month
        self.initial_capital = initial_capital

        # 数据
        self.live_trades: List[LiveTradeEntry] = []
        self.kline_df: Optional[pd.DataFrame] = None
        self.funding_records: List[FundingRecord] = []

        # 中间状态
        self._theoretical_trades: List = []     # 回测引擎输出的 Trade 对象
        self._daily_nav: List[Dict] = []

        # 时间范围
        self.month_start = pd.Timestamp(year, month, 1)
        if month == 12:
            self.month_end = pd.Timestamp(year + 1, 1, 1)
        else:
            self.month_end = pd.Timestamp(year, month + 1, 1)

    # ============================================================
    # 数据加载
    # ============================================================
    def load_live_trades(self, filepath: str) -> List[LiveTradeEntry]:
        """从交易日志 CSV 加载实盘交易。
        兼容回测导出的 CSV 格式（entry_time, exit_time, entry_price, exit_price, net_pnl 等）
        """
        df = pd.read_csv(filepath)
        trades = []

        # 列名映射（兼容回测导出格式 → LiveTradeEntry 格式）
        col_entry_time = "entry_time" if "entry_time" in df.columns else "signal_time"
        col_exit_time = "exit_time" if "exit_time" in df.columns else None
        col_entry_price = "entry_price" if "entry_price" in df.columns else "fill_price"
        col_exit_price = "exit_price" if "exit_price" in df.columns else None
        col_exit_reason = "exit_reason" if "exit_reason" in df.columns else None
        col_direction = "direction" if "direction" in df.columns else None
        col_net_pnl = "net_pnl" if "net_pnl" in df.columns else None
        col_divergence = "divergence_type" if "divergence_type" in df.columns else None
        col_entry_fee = "entry_fee" if "entry_fee" in df.columns else None
        col_exit_fee = "exit_fee" if "exit_fee" in df.columns else None
        col_slippage = "slippage" if "slippage" in df.columns else None

        for _, row in df.iterrows():
            entry_time = pd.Timestamp(row[col_entry_time])
            exit_time = pd.Timestamp(row[col_exit_time]) if col_exit_time and pd.notna(row.get(col_exit_time)) else None
            entry_price = float(row.get(col_entry_price, 0))
            exit_price_val = float(row.get(col_exit_price, 0)) if col_exit_price and pd.notna(row.get(col_exit_price)) else None

            t = LiveTradeEntry(
                trade_id=str(row.get("trade_id", "")),
                direction=str(row.get(col_direction, "SHORT")) if col_direction else "SHORT",
                signal_time=entry_time,
                order_time=entry_time,
                fill_time=entry_time,
                signal_price=entry_price,
                fill_price=entry_price,
                quantity=1,
                notional_usd=entry_price,
                fee_paid=float(row.get(col_entry_fee, 0)) if col_entry_fee else 0,
                exit_time=exit_time,
                exit_price=exit_price_val,
                exit_fee=float(row.get(col_exit_fee, 0)) if col_exit_fee else 0,
                net_pnl=float(row.get(col_net_pnl, 0)) if col_net_pnl and pd.notna(row.get(col_net_pnl)) else None,
                exit_reason=str(row.get(col_exit_reason, "")) if col_exit_reason else "",
            )
            t.latency_ms = 0
            trades.append(t)

        self.live_trades = trades
        logger.info("Loaded {} trades for {:04d}-{:02d}".format(len(trades), self.year, self.month))
        return trades

    def load_kline_data(self, filepath: str) -> pd.DataFrame:
        """加载 K 线数据并切片到本月"""
        df = pd.read_csv(filepath)
        df["timestamp"] = df["timestamp"].astype(int)
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
        df["datetime"] = df["datetime"].dt.tz_localize(None)  # strip tz for comparison
        # Add warmup: include 3 days before month start for indicator calculations
        warmup_start = self.month_start.tz_localize(None) - pd.Timedelta(days=3)
        month_end = self.month_end.tz_localize(None)
        df = df[(df["datetime"] >= warmup_start) & (df["datetime"] < month_end)]
        df.set_index("datetime", inplace=True)
        df.sort_index(inplace=True)
        self.kline_df = df
        logger.info("Loaded {} K-lines for {:04d}-{:02d}".format(len(df), self.year, self.month))
        return df

    def load_funding_rates(self, filepath: str) -> List[FundingRecord]:
        """从资金费率 CSV 加载"""
        df = pd.read_csv(filepath)
        records = []
        running_sum = 0.0
        for _, row in df.iterrows():
            pos_size = float(row.get("position_size", 0))
            rate = float(row.get("rate", 0))
            payment = -pos_size * rate  # SHORT: payment > 0 when rate < 0 (空付多)
            running_sum += payment
            records.append(FundingRecord(
                time=pd.Timestamp(row["time"]),
                rate=rate,
                position_side=str(row.get("position_side", "")),
                position_size=pos_size,
                payment=payment,
                cumulative=running_sum,
            ))
            # Filter to month
            if self.month_start <= pd.Timestamp(row["time"]) < self.month_end:
                records.append(FundingRecord(
                    time=pd.Timestamp(row["time"]),
                    rate=rate,
                    position_side=str(row.get("position_side", "")),
                    position_size=pos_size,
                    payment=payment,
                    cumulative=running_sum,
                ))
        self.funding_records = records
        logger.info("Loaded {} funding records".format(len(records)))
        return records

    # ============================================================
    # 核心：理论 vs 实际 PnL 比对
    # ============================================================
    def compute_pnl_divergence(self) -> List[PnLDiff]:
        """逐笔对比实盘盈亏与理论回测盈亏

        理论回测: 用相同的策略参数在历史 K 线上回放，提取对应时间的信号，
                  但不加滑点/延迟磨损，得到纯理论收益。
        """
        diffs = []

        if self.kline_df is None or len(self.kline_df) < 100:
            logger.warning("Insufficient K-line data for theoretical PnL")
            return diffs

        # --- Run theoretical backtest on monthly data ---
        theoretical_trades = self._run_theoretical_backtest()
        self._theoretical_trades = theoretical_trades

        if not theoretical_trades or not self.live_trades:
            return diffs

        # --- Match live trades to theoretical by approximate time ---
        # Build lookup: theoretical trade by entry timestamp
        theo_by_time = {}
        for t in theoretical_trades:
            entry_ms = t.entry_time if hasattr(t, 'entry_time') else getattr(t, 'entry_bar_ts', 0)
            ts = pd.Timestamp(entry_ms, unit='ms')
            minute_key = ts.floor("15min")
            theo_by_time[minute_key] = t

        for live in self.live_trades:
            # Find closest theoretical trade
            signal_key = live.signal_time.floor("15min")
            theo = theo_by_time.get(signal_key)

            if theo is None:
                # Try adjacent buckets
                for offset in [-1, 1]:
                    alt_key = signal_key + pd.Timedelta(minutes=15 * offset)
                    if alt_key in theo_by_time:
                        theo = theo_by_time[alt_key]
                        break

            if theo is None:
                continue

            actual_pnl = live.net_pnl or 0.0
            theoretical_pnl = theo.net_pnl if hasattr(theo, 'net_pnl') else 0.0

            diff = PnLDiff(
                actual_pnl=actual_pnl,
                theoretical_pnl=theoretical_pnl,
                diff_abs=actual_pnl - theoretical_pnl,
                diff_pct=(actual_pnl - theoretical_pnl) / abs(theoretical_pnl) * 100 if theoretical_pnl != 0 else 0,
                slippage_entry_bp=abs(live.fill_price - live.signal_price) / live.signal_price * 10000 if live.signal_price > 0 else 0,
                slippage_exit_bp=abs((live.exit_price or live.fill_price) - live.signal_price) / live.signal_price * 10000 if live.signal_price > 0 else 0,
                latency_ms=live.latency_ms,
            )
            diffs.append(diff)

        logger.info("PnL divergence computed: {} matched pairs".format(len(diffs)))
        return diffs

    def _run_theoretical_backtest(self):
        """在月度 K 线数据上运行一次理论回测，返回 Trade 列表"""
        if self.kline_df is None:
            return []

        # Build a clean OHLCV dataframe for the backtest
        df = self.kline_df.reset_index()
        df = df.rename(columns={"datetime": "timestamp_dt"})
        df["timestamp"] = df["timestamp"].astype(int)

        # Clone params
        p = SHORT_PARAMS

        # We use the backtest account + simplified bar loop
        account = VirtualAccount(
            initial_capital=self.initial_capital,
            fee_rate=0.0005,
            slippage_rate=0.0,    # THEORETICAL: no slippage
            max_pyramiding=2,
        )

        detector = DivergenceDetector(p)
        sl_updater = TrailingStopUpdater(stop_loss_mult=p.stop_loss_mult)

        # Pre-compute signals
        all_signals = detector.detect(df, "15m")
        sigs_by_bar = {}
        for sig in all_signals:
            bar_idx = sig.metadata.get("pivot_b")
            if bar_idx is not None:
                sigs_by_bar.setdefault(bar_idx, []).append(sig)

        n = len(df)
        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        atr_series = compute_atr(high, low, close, period=p.atr_period)

        position_entry_bars = []
        warmup = 60

        for i in range(warmup + 20, n):
            bar_ts = int(df["timestamp"].iloc[i])
            bar_high = float(high.iloc[i])
            bar_low = float(low.iloc[i])
            bar_close = float(close.iloc[i])
            current_atr = float(atr_series.iloc[i]) if not pd.isna(atr_series.iloc[i]) else 0.0

            # Update stops
            for pi in range(len(account.short_positions)):
                pos = account.short_positions[pi]
                new_sl = sl_updater.update_short_sl(pos.trailing_sl, bar_high, current_atr)
                account.update_trailing_sl_short(new_sl, pi)

            # Check stops (skip entry bar)
            remaining_pos = []
            remaining_bars = []
            for pi in range(len(account.short_positions)):
                pos = account.short_positions[pi]
                eb = position_entry_bars[pi] if pi < len(position_entry_bars) else i
                if i > eb and bar_high >= pos.trailing_sl:
                    exit_price = max(pos.trailing_sl, bar_high)
                    account._force_close(pos, exit_price, bar_ts, i, "TSL_STOP")
                else:
                    remaining_pos.append(pos)
                    remaining_bars.append(eb)
            account.short_positions = remaining_pos
            position_entry_bars = remaining_bars

            # Process signals
            if i in sigs_by_bar:
                for sig in sigs_by_bar[i]:
                    if sig.signal_type == SignalType.SELL:
                        pos = account.open_position(
                            direction=Direction.SHORT, price=bar_close,
                            timestamp=bar_ts, bar_index=i,
                            trailing_sl=sig.trailing_sl or 0.0,
                            divergence_type=sig.divergence_type.value if sig.divergence_type else "",
                        )
                        if pos:
                            position_entry_bars.append(i)
                    elif sig.signal_type == SignalType.CLOSE_SHORT:
                        while account.short_positions:
                            account.close_position(
                                direction=Direction.SHORT, price=bar_close,
                                timestamp=bar_ts, bar_index=i, reason="SIGNAL_CLOSE",
                            )
                        position_entry_bars = []

        return account.closed_trades

    # ============================================================
    # 月度指标计算
    # ============================================================
    def compute_monthly_snapshot(self) -> MonthlySnapshot:
        """汇总本月所有指标到一个 MonthlySnapshot"""
        # PnL divergence
        diffs = self.compute_pnl_divergence()

        # Actual metrics from live trades
        live_pnl = sum((t.net_pnl or 0.0) for t in self.live_trades)
        live_wins = [t for t in self.live_trades if (t.net_pnl or 0) > 0]
        live_losses = [t for t in self.live_trades if (t.net_pnl or 0) <= 0]
        n = len(self.live_trades)

        actual_return_pct = live_pnl / self.initial_capital * 100
        theoretical_pnl = sum(d.theoretical_pnl for d in diffs)
        theoretical_return_pct = theoretical_pnl / self.initial_capital * 100
        divergence_pct = abs(actual_return_pct - theoretical_return_pct)

        # Profit factor
        total_profit = sum((t.net_pnl or 0) for t in live_wins)
        total_loss = abs(sum((t.net_pnl or 0) for t in live_losses))
        pf = total_profit / total_loss if total_loss > 0 else float('inf')

        # Max win / max loss
        max_win = max(self.live_trades, key=lambda t: t.net_pnl or -999999, default=None)
        max_loss = min(self.live_trades, key=lambda t: t.net_pnl or 999999, default=None)

        # Fees
        total_fees = sum(t.fee_paid + (t.exit_fee or 0) for t in self.live_trades)
        total_slippage = sum(
            abs(t.fill_price - t.signal_price) * t.quantity * 0.01 for t in self.live_trades
        )

        # Funding
        net_funding = sum(f.payment for f in self.funding_records)

        # BTC benchmark
        btc_return = self._compute_btc_benchmark()

        # Daily NAV
        daily_nav = self._compute_daily_nav()

        # Sharpe
        returns = []
        for t in self.live_trades:
            if t.net_pnl is not None:
                returns.append(t.net_pnl / self.initial_capital)
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if len(returns) > 1 and np.std(returns) > 0 else 0.0

        # Max drawdown
        equity_curve = [self.initial_capital]
        for t in self.live_trades:
            equity_curve.append(equity_curve[-1] + (t.net_pnl or 0))
        peak = np.maximum.accumulate(equity_curve)
        dd = (np.array(equity_curve) - peak) / peak * 100
        max_dd = abs(min(dd))

        snapshot = MonthlySnapshot(
            year=self.year, month=self.month,
            actual_return_pct=actual_return_pct,
            theoretical_return_pct=theoretical_return_pct,
            return_divergence_pct=divergence_pct,
            total_trades=n,
            wins=len(live_wins),
            losses=len(live_losses),
            win_rate_pct=len(live_wins) / n * 100 if n > 0 else 0,
            actual_profit_factor=pf,
            max_win_trade_id=max_win.trade_id if max_win else "",
            max_win_pnl=max_win.net_pnl or 0 if max_win else 0,
            max_win_latency_ms=max_win.latency_ms if max_win else 0,
            max_loss_trade_id=max_loss.trade_id if max_loss else "",
            max_loss_pnl=max_loss.net_pnl or 0 if max_loss else 0,
            max_loss_latency_ms=max_loss.latency_ms if max_loss else 0,
            total_fees=total_fees,
            total_slippage_cost=total_slippage,
            net_funding_pnl=net_funding,
            btc_buyhold_return_pct=btc_return,
            strategy_vs_benchmark_pct=actual_return_pct - btc_return,
            max_drawdown_pct=max_dd,
            sharpe_ratio=sharpe,
            daily_nav=daily_nav,
            trade_details=[{
                "id": t.trade_id, "direction": t.direction,
                "entry_time": str(t.signal_time), "entry_price": t.fill_price,
                "exit_time": str(t.exit_time) if t.exit_time else "",
                "exit_price": t.exit_price or 0, "net_pnl": t.net_pnl or 0,
                "exit_reason": t.exit_reason or "", "latency_ms": t.latency_ms,
            } for t in self.live_trades],
        )
        return snapshot

    def _compute_btc_benchmark(self) -> float:
        """计算本月 BTC 买入持有收益率"""
        if self.kline_df is None or len(self.kline_df) < 2:
            return 0.0
        ms = self.month_start.tz_localize(None)
        me = self.month_end.tz_localize(None)
        df_month = self.kline_df[(self.kline_df.index >= ms) & (self.kline_df.index < me)]
        if len(df_month) < 2:
            return 0.0
        first_price = float(df_month["close"].iloc[0])
        last_price = float(df_month["close"].iloc[-1])
        return (last_price / first_price - 1) * 100

    def _compute_daily_nav(self) -> List[Dict]:
        """计算日度净值序列"""
        if self.kline_df is None:
            return []

        nav = []
        equity = self.initial_capital
        btc_initial = self.kline_df["close"].iloc[0]

        df_month = self.kline_df[
            (self.kline_df.index >= self.month_start) & (self.kline_df.index < self.month_end)
        ]
        if df_month.empty:
            return []

        # Group by date
        for date, day_df in df_month.groupby(df_month.index.date):
            # PnL from trades exiting this day
            for t in self.live_trades:
                if t.exit_time and t.exit_time.date() == date:
                    equity += (t.net_pnl or 0)

            nav.append({
                "date": str(date),
                "equity": equity,
                "nav": equity / self.initial_capital,
                "btc_price": float(day_df["close"].iloc[-1]),
                "btc_nav": float(day_df["close"].iloc[-1]) / btc_initial,
            })

        self._daily_nav = nav
        return nav

    # ============================================================
    # Markdown 报告
    # ============================================================
    def generate_report(self, output_path: Optional[str] = None) -> str:
        """生成月度复盘 Markdown 报告"""
        snap = self.compute_monthly_snapshot()
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        lines = []
        lines.append("# 📊 量化交易月度复盘报告")
        lines.append("")
        lines.append("**报告周期**: {:04d}年{:02d}月".format(snap.year, snap.month))
        lines.append("**生成时间**: {}".format(now))
        lines.append("**初始资金**: ${:,.2f}".format(self.initial_capital))
        lines.append("")

        # 1. 核心收益
        lines.append("## 1. 收益总览")
        lines.append("")
        lines.append("| 指标 | 数值 |")
        lines.append("|------|------|")
        lines.append("| 实盘实际收益率 | {:+.2f}% |".format(snap.actual_return_pct))
        lines.append("| 回测理论收益率 | {:+.2f}% |".format(snap.theoretical_return_pct))
        lines.append("| **收益偏差 (滑点+延迟磨损)** | **{:.2f}%** |".format(snap.return_divergence_pct))
        lines.append("")

        # 2. 交易统计
        lines.append("## 2. 交易统计")
        lines.append("")
        lines.append("| 指标 | 数值 |")
        lines.append("|------|------|")
        lines.append("| 总交易次数 | {} |".format(snap.total_trades))
        lines.append("| 胜率 | {:.1f}% |".format(snap.win_rate_pct))
        lines.append("| 实盘盈亏比 | {:.2f} |".format(snap.actual_profit_factor))
        lines.append("| 夏普比率 | {:.2f} |".format(snap.sharpe_ratio))
        lines.append("| 最大回撤 | {:.1f}% |".format(snap.max_drawdown_pct))
        lines.append("")

        # 3. 极值单
        lines.append("## 3. 极值交易单")
        lines.append("")
        lines.append("| 类型 | 单号 | 盈亏 | 延迟(ms) |")
        lines.append("|------|------|------|----------|")
        lines.append("| 🟢 最大盈利 | {} | ${:,.2f} | {}ms |".format(
            snap.max_win_trade_id, snap.max_win_pnl, snap.max_win_latency_ms))
        lines.append("| 🔴 最大亏损 | {} | ${:,.2f} | {}ms |".format(
            snap.max_loss_trade_id, snap.max_loss_pnl, snap.max_loss_latency_ms))
        lines.append("")

        # 4. 费用分析
        lines.append("## 4. 费用分析")
        lines.append("")
        lines.append("| 项目 | 金额 |")
        lines.append("|------|------|")
        lines.append("| 手续费合计 | ${:,.2f} |".format(snap.total_fees))
        lines.append("| 滑点磨损 | ${:,.2f} |".format(snap.total_slippage_cost))
        lines.append("| **净资金费率** | **${:+,.2f}** |".format(snap.net_funding_pnl))
        lines.append("| 总成本 | ${:,.2f} |".format(snap.total_fees + snap.total_slippage_cost - snap.net_funding_pnl))
        lines.append("")

        # 5. 基准对比
        lines.append("## 5. 基准对比")
        lines.append("")
        lines.append("| 指标 | 收益率 |")
        lines.append("|------|--------|")
        lines.append("| 策略实盘收益 | {:+.2f}% |".format(snap.actual_return_pct))
        lines.append("| BTC 买入持有 | {:+.2f}% |".format(snap.btc_buyhold_return_pct))
        lines.append("| **策略超额收益** | **{:+.2f}%** |".format(snap.strategy_vs_benchmark_pct))
        lines.append("")

        # 6. 风险提示
        lines.append("## 6. 风险提示")
        lines.append("")
        lines.append("- 最大回撤 {:.1f}% — 请确认账户保证金充足".format(snap.max_drawdown_pct))
        lines.append("- 单笔最大亏损 ${:,.2f} — 请审视该笔交易的止损设置".format(snap.max_loss_pnl))
        lines.append("- 理论 vs 实际偏差 {:.2f}% — {}".format(
            snap.return_divergence_pct,
            "正常范围" if snap.return_divergence_pct < 2 else "⚠️ 偏高，请检查网络延迟和滑点设置"))
        lines.append("")

        report = "\n".join(lines)

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(report)
            logger.info("Monthly report saved to {}".format(output_path))

        return report

    # ============================================================
    # 可视化
    # ============================================================
    def generate_chart(self, output_path: str) -> str:
        """生成30天 K 线 + 实盘进出场点 + 追踪止损线"""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        from matplotlib.patches import FancyBboxPatch

        if self.kline_df is None:
            logger.warning("No K-line data for chart")
            return ""

        # Slice to this month
        ms = self.month_start.tz_localize(None)
        me = self.month_end.tz_localize(None)
        df = self.kline_df[(self.kline_df.index >= ms) & (self.kline_df.index < me)].copy()
        if df.empty:
            return ""

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 11), sharex=True,
                                        gridspec_kw={"height_ratios": [3, 1]})

        # --- Top: Candlestick + trades + trailing stops ---
        colors = ["#26a69a" if df["close"].iloc[i] >= df["open"].iloc[i] else "#ef5350"
                  for i in range(len(df))]
        width = 0.0004 * len(df)  # scale with days

        # Body
        ax1.bar(df.index, abs(df["close"] - df["open"]),
                bottom=df[["open", "close"]].min(axis=1),
                width=width, color=colors, zorder=2)
        # Wicks
        ax1.bar(df.index, df["high"] - df["low"], bottom=df["low"],
                width=width * 0.2, color=colors, alpha=0.6, zorder=1)

        # Plot trailing stops as dashed lines
        # For each live trade that has SL history, draw the line
        for t in self.live_trades:
            if not t.sl_history:
                continue
            entry_dt = t.signal_time
            exit_dt = t.exit_time if t.exit_time else df.index[-1]
            times = pd.date_range(entry_dt, exit_dt, freq="15min")
            if len(times) == 0:
                continue
            sl_vals = t.sl_history[:len(times)]
            if not sl_vals:
                continue
            sl_line = np.full(len(times), sl_vals[-1])
            ax1.plot(times[:len(sl_line)], sl_line, linestyle="--", color="orange",
                     linewidth=0.8, alpha=0.5)

        # Mark entry/exit points
        for t in self.live_trades:
            if t.direction.upper() == "SHORT":
                # Entry: red down-triangle
                ax1.scatter(t.signal_time, t.fill_price, marker="v", color="red",
                            s=100, zorder=5, edgecolors="white", linewidth=0.5)
                if t.exit_time:
                    # Exit: blue up-triangle
                    ax1.scatter(t.exit_time, t.exit_price, marker="^", color="blue",
                                s=70, zorder=5, edgecolors="white", linewidth=0.5)
            else:
                ax1.scatter(t.signal_time, t.fill_price, marker="^", color="green",
                            s=100, zorder=5, edgecolors="white", linewidth=0.5)
                if t.exit_time:
                    ax1.scatter(t.exit_time, t.exit_price, marker="v", color="blue",
                                s=70, zorder=5, edgecolors="white", linewidth=0.5)

        # Annotate max win and max loss
        if self.live_trades:
            max_win = max(self.live_trades, key=lambda x: x.net_pnl or -999999)
            max_loss = min(self.live_trades, key=lambda x: x.net_pnl or 999999)
            if max_win.exit_time and max_win.net_pnl:
                ax1.annotate("MaxWin\n${:,.0f}".format(max_win.net_pnl),
                            (max_win.exit_time, max_win.exit_price),
                            xytext=(10, 20), textcoords="offset points",
                            fontsize=8, color="green", fontweight="bold",
                            arrowprops=dict(arrowstyle="->", color="green", lw=0.8))
            if max_loss.exit_time and max_loss.net_pnl:
                ax1.annotate("MaxLoss\n${:,.0f}".format(max_loss.net_pnl),
                            (max_loss.exit_time, max_loss.exit_price),
                            xytext=(10, -20), textcoords="offset points",
                            fontsize=8, color="red", fontweight="bold",
                            arrowprops=dict(arrowstyle="->", color="red", lw=0.8))

        ax1.set_ylabel("Price (USDT)", fontsize=11)
        ax1.set_title("BTC/USDT 15m — Monthly Review {:04d}-{:02d}".format(self.year, self.month),
                      fontsize=14, fontweight="bold")
        ax1.grid(True, alpha=0.2)
        ax1.legend(["SL Line", "Short Entry", "Short Exit", "Long Entry", "Long Exit"],
                   loc="upper left", fontsize=7, ncol=5)

        # --- Bottom: RSI ---
        rsi = compute_rsi(df["close"].astype(float), period=14)
        ax2.fill_between(df.index, 30, 70, color="gray", alpha=0.08)
        ax2.axhline(70, color="red", linewidth=0.7, linestyle="--", alpha=0.6)
        ax2.axhline(30, color="green", linewidth=0.7, linestyle="--", alpha=0.6)
        ax2.plot(df.index, rsi, color="#7b1fa2", linewidth=1.2, label="RSI(14)")
        ax2.set_ylabel("RSI", fontsize=11)
        ax2.set_xlabel("Date", fontsize=11)
        ax2.legend(loc="upper left", fontsize=8)
        ax2.grid(True, alpha=0.2)

        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
        ax1.xaxis.set_major_locator(mdates.DayLocator(interval=3))
        plt.xticks(rotation=30)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info("Monthly chart saved to {}".format(output_path))
        return output_path
