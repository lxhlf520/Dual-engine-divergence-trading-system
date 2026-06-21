"""
Post-Trade Analyzer — 独立的复盘分析模块

功能:
1. 读取交易记录 + K线数据
2. 执行质量分析 (滑点、时延)
3. 绩效统计 (胜率、盈亏比、回撤、夏普)
4. 日度 NAV 曲线 + BTC 买入持有基准对比
5. 生成 Markdown 日报 + matplotlib 可视化图
6. 通过通知机器人推送
"""
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import ROOT_DIR, LOG_DIR
from ..logger import get_logger
from ..strategy.indicators import compute_rsi, compute_atr

logger = get_logger(__name__)


@dataclass
class TradeRecord:
    """单笔交易记录"""
    trade_id: int
    direction: str                    # SHORT / LONG
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    entry_signal_price: float = 0.0   # 策略信号触发时的价格
    exit_signal_price: float = 0.0    # 平仓信号触发时的价格
    net_pnl: float = 0.0
    net_pnl_pct: float = 0.0
    exit_reason: str = ""
    divergence_type: str = ""
    bars_held: int = 0
    entry_fee: float = 0.0
    exit_fee: float = 0.0
    slippage_cost: float = 0.0
    entry_slippage: float = 0.0       # 入场滑点 (bp)
    exit_slippage: float = 0.0        # 出场滑点 (bp)
    latency_seconds: int = 0          # 信号到成交延迟
    trailing_sl_history: List[float] = field(default_factory=list)


@dataclass
class DailyNAV:
    """日度净值"""
    date: pd.Timestamp
    nav: float
    equity: float
    btc_price: float
    btc_nav: float                    # 同期买入持有净值
    positions: int = 0
    daily_return: float = 0.0


class PostTradeAnalyzer:
    """复盘分析引擎"""

    def __init__(self, initial_capital: float = 10000.0):
        self.initial_capital = initial_capital
        self.trades: List[TradeRecord] = []
        self.daily_nav: List[DailyNAV] = []
        self.kline_df: Optional[pd.DataFrame] = None

    # ============================================================
    # 数据加载
    # ============================================================
    def load_trades_from_csv(self, filepath: str) -> List[TradeRecord]:
        """从回测导出的 CSV 加载交易记录"""
        df = pd.read_csv(filepath)
        trades = []
        for i, row in df.iterrows():
            t = TradeRecord(
                trade_id=i + 1,
                direction=str(row.get("direction", "SHORT")),
                entry_time=pd.Timestamp(row["entry_time"]),
                exit_time=pd.Timestamp(row["exit_time"]),
                entry_price=float(row["entry_price"]),
                exit_price=float(row["exit_price"]),
                entry_signal_price=float(row.get("entry_signal_price", row["entry_price"])),
                exit_signal_price=float(row.get("exit_signal_price", row["exit_price"])),
                net_pnl=float(row.get("net_pnl", 0)),
                net_pnl_pct=float(row.get("net_pnl_pct", 0)),
                exit_reason=str(row.get("exit_reason", "")),
                divergence_type=str(row.get("divergence_type", "")),
                bars_held=int(row.get("bars_held", 0)),
                entry_fee=float(row.get("entry_fee", 0)),
                exit_fee=float(row.get("exit_fee", 0)),
                slippage_cost=float(row.get("slippage", 0)),
            )
            # Calculate slippage
            t.entry_slippage = abs(t.entry_price - t.entry_signal_price) / t.entry_signal_price * 10000
            t.exit_slippage = abs(t.exit_price - t.exit_signal_price) / t.exit_signal_price * 10000
            trades.append(t)
        self.trades = trades
        logger.info("Loaded {} trades from {}".format(len(trades), filepath))
        return trades

    def load_trades_from_list(self, trade_list: list) -> None:
        """从 Trade 对象列表加载"""
        self.trades = []
        for i, t in enumerate(trade_list):
            self.trades.append(TradeRecord(
                trade_id=i + 1,
                direction=t.direction.value if hasattr(t.direction, 'value') else str(t.direction),
                entry_time=pd.Timestamp(t.entry_time, unit='ms'),
                exit_time=pd.Timestamp(t.exit_time, unit='ms'),
                entry_price=t.entry_price,
                exit_price=t.exit_price,
                net_pnl=t.net_pnl,
                net_pnl_pct=t.net_pnl_pct,
                exit_reason=t.exit_reason,
                divergence_type=getattr(t, 'divergence_type', ''),
                bars_held=t.bars_held,
                entry_fee=t.entry_fee,
                exit_fee=t.exit_fee,
                slippage_cost=t.slippage,
            ))

    def load_kline_data(self, filepath: str) -> pd.DataFrame:
        """加载历史 K 线"""
        df = pd.read_csv(filepath)
        df["timestamp"] = df["timestamp"].astype(int)
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("datetime", inplace=True)
        self.kline_df = df
        logger.info("Loaded {} K-lines".format(len(df)))
        return df

    # ============================================================
    # 执行质量分析
    # ============================================================
    def analyze_execution_quality(self) -> Dict:
        """计算执行滑点与时延"""
        if not self.trades:
            return {}

        entry_slippages = [t.entry_slippage for t in self.trades]
        exit_slippages = [t.exit_slippage for t in self.trades]
        latencies = [t.latency_seconds for t in self.trades]

        return {
            "avg_entry_slippage_bp": np.mean(entry_slippages),
            "max_entry_slippage_bp": np.max(entry_slippages),
            "avg_exit_slippage_bp": np.mean(exit_slippages),
            "max_exit_slippage_bp": np.max(exit_slippages),
            "avg_total_slippage_bp": np.mean([a + b for a, b in zip(entry_slippages, exit_slippages)]),
            "avg_latency_seconds": np.mean(latencies) if latencies else 0,
            "total_slippage_cost_usd": sum(t.slippage_cost for t in self.trades),
        }

    # ============================================================
    # 绩效统计
    # ============================================================
    def compute_performance(self) -> Dict:
        """核心绩效指标"""
        if not self.trades:
            return {"total_trades": 0}

        wins = [t for t in self.trades if t.net_pnl > 0]
        losses = [t for t in self.trades if t.net_pnl <= 0]
        n = len(self.trades)

        net_pnl = sum(t.net_pnl for t in self.trades)
        fees_total = sum(t.entry_fee + t.exit_fee for t in self.trades)
        slippage_total = sum(t.slippage_cost for t in self.trades)

        # 收益序列（逐笔）
        returns = [t.net_pnl_pct / 100 for t in self.trades]  # 转小数

        # 夏普比率（年化，假设每笔交易独立）
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252 * 96)  # 15m bars
        else:
            sharpe = 0.0

        # 最大回撤
        equity_curve = np.cumsum([t.net_pnl for t in self.trades]) + self.initial_capital
        peak = np.maximum.accumulate(equity_curve)
        dd = (equity_curve - peak) / peak * 100
        max_dd = abs(dd.min())

        # 盈亏比
        avg_win = np.mean([t.net_pnl for t in wins]) if wins else 0
        avg_loss = abs(np.mean([t.net_pnl for t in losses])) if losses else 0
        profit_factor = sum(t.net_pnl for t in wins) / abs(sum(t.net_pnl for t in losses)) if losses else float('inf')

        # 按方向统计
        short_trades = [t for t in self.trades if t.direction.upper() == "SHORT"]
        long_trades = [t for t in self.trades if t.direction.upper() == "LONG"]

        # 按平仓原因统计
        by_reason = {}
        for t in self.trades:
            r = t.exit_reason or "UNKNOWN"
            if r not in by_reason:
                by_reason[r] = {"count": 0, "pnl": 0.0}
            by_reason[r]["count"] += 1
            by_reason[r]["pnl"] += t.net_pnl

        # 最大连胜/连败
        max_streak_win = 0
        max_streak_loss = 0
        current_win = 0
        current_loss = 0
        for t in self.trades:
            if t.net_pnl > 0:
                current_win += 1
                max_streak_win = max(max_streak_win, current_win)
                current_loss = 0
            else:
                current_loss += 1
                max_streak_loss = max(max_streak_loss, current_loss)
                current_win = 0

        return {
            "total_trades": n,
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": len(wins) / n * 100 if n > 0 else 0,
            "net_pnl": net_pnl,
            "net_pnl_pct": net_pnl / self.initial_capital * 100,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "max_win": max(t.net_pnl for t in self.trades),
            "max_loss": min(t.net_pnl for t in self.trades),
            "max_drawdown_pct": max_dd,
            "sharpe_ratio": sharpe,
            "total_fees": fees_total,
            "total_slippage": slippage_total,
            "short_trades": len(short_trades),
            "short_pnl": sum(t.net_pnl for t in short_trades),
            "long_trades": len(long_trades),
            "long_pnl": sum(t.net_pnl for t in long_trades),
            "max_streak_win": max_streak_win,
            "max_streak_loss": max_streak_loss,
            "avg_bars_held": np.mean([t.bars_held for t in self.trades]),
            "exit_reasons": by_reason,
        }

    # ============================================================
    # NAV 曲线 + 基准对比
    # ============================================================
    def compute_nav_curve(self) -> List[DailyNAV]:
        """基于 K 线数据构建日度 NAV 曲线"""
        if self.kline_df is None or not self.trades:
            return []

        nav_list = []
        equity = self.initial_capital

        # 获取 BTC 初始价格
        btc_initial = self.kline_df["close"].iloc[0]
        btc_shares = self.initial_capital / btc_initial

        # 按日期分组 K 线
        self.kline_df["date"] = self.kline_df.index.date

        for date, day_df in self.kline_df.groupby("date"):
            close_price = day_df["close"].iloc[-1]
            btc_nav = close_price * btc_shares

            # 当天交易影响
            day_trades = [
                t for t in self.trades
                if t.exit_time.date() == date
            ]
            for t in day_trades:
                equity += t.net_pnl

            # 日期序列
            nav_list.append(DailyNAV(
                date=pd.Timestamp(date),
                nav=equity / self.initial_capital,
                equity=equity,
                btc_price=close_price,
                btc_nav=btc_nav,
                positions=len([t for t in self.trades if t.entry_time.date() <= date < t.exit_time.date()]),
                daily_return=(nav_list[-1].equity - equity) / nav_list[-1].equity * 100 if nav_list else 0,
            ))

        self.daily_nav = nav_list
        return nav_list

    # ============================================================
    # 报告生成
    # ============================================================
    def generate_markdown_report(self, output_path: Optional[str] = None) -> str:
        """生成 Markdown 格式复盘日报"""
        perf = self.compute_performance()
        exec_q = self.analyze_execution_quality()
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        lines = []
        lines.append("# 📊 量化交易复盘日报")
        lines.append("")
        lines.append("**生成时间**: {}".format(now))
        lines.append("**初始资金**: ${:,.2f}".format(self.initial_capital))
        lines.append("")

        # 核心指标
        lines.append("## 1. 核心指标")
        lines.append("")
        lines.append("| 指标 | 数值 |")
        lines.append("|------|------|")
        lines.append("| 总交易次数 | {} |".format(perf.get("total_trades", 0)))
        lines.append("| 胜率 | {:.1f}% |".format(perf.get("win_rate", 0)))
        lines.append("| 净盈亏 | ${:,.2f} ({:+.1f}%) |".format(perf.get("net_pnl", 0), perf.get("net_pnl_pct", 0)))
        lines.append("| 盈亏比 | {:.2f} |".format(perf.get("profit_factor", 0)))
        lines.append("| 夏普比率 | {:.2f} |".format(perf.get("sharpe_ratio", 0)))
        lines.append("| 最大回撤 | {:.1f}% |".format(perf.get("max_drawdown_pct", 0)))
        lines.append("| 最大连胜 | {} 笔 |".format(perf.get("max_streak_win", 0)))
        lines.append("| 最大连败 | {} 笔 |".format(perf.get("max_streak_loss", 0)))
        lines.append("")

        # 交易详情
        lines.append("## 2. 交易详情")
        lines.append("")
        lines.append("|  | 平均 | 最大 | 最小 |")
        lines.append("|---|---|---|---|")
        lines.append("| 盈利 | ${:,.2f} | ${:,.2f} | — |".format(
            perf.get("avg_win", 0), perf.get("max_win", 0)))
        lines.append("| 亏损 | ${:,.2f} | — | ${:,.2f} |".format(
            perf.get("avg_loss", 0), perf.get("max_loss", 0)))
        lines.append("| 平均持仓 | {:.0f} 根K线 | — | — |".format(perf.get("avg_bars_held", 0)))
        lines.append("")

        # 成本分析
        lines.append("## 3. 成本分析")
        lines.append("")
        lines.append("| 项目 | 金额 |")
        lines.append("|------|------|")
        lines.append("| 手续费合计 | ${:,.2f} |".format(perf.get("total_fees", 0)))
        lines.append("| 滑点合计 | ${:,.2f} |".format(perf.get("total_slippage", 0)))
        lines.append("| 总成本 | ${:,.2f} |".format(perf.get("total_fees", 0) + perf.get("total_slippage", 0)))
        lines.append("")

        # 执行质量
        if exec_q:
            lines.append("## 4. 执行质量")
            lines.append("")
            lines.append("| 指标 | 数值 |")
            lines.append("|------|------|")
            lines.append("| 平均入场滑点 | {:.1f} bp |".format(exec_q.get("avg_entry_slippage_bp", 0)))
            lines.append("| 平均出场滑点 | {:.1f} bp |".format(exec_q.get("avg_exit_slippage_bp", 0)))
            lines.append("| 平均总滑点 | {:.1f} bp |".format(exec_q.get("avg_total_slippage_bp", 0)))
            lines.append("| 平均延迟 | {} 秒 |".format(exec_q.get("avg_latency_seconds", 0)))
            lines.append("")

        # 分方向
        lines.append("## 5. 多空分布")
        lines.append("")
        lines.append("| 方向 | 笔数 | 净盈亏 |")
        lines.append("|------|------|--------|")
        lines.append("| 做空 (Short) | {} | ${:,.2f} |".format(perf.get("short_trades", 0), perf.get("short_pnl", 0)))
        lines.append("| 做多 (Long)  | {} | ${:,.2f} |".format(perf.get("long_trades", 0), perf.get("long_pnl", 0)))
        lines.append("")

        # 平仓原因
        by_reason = perf.get("exit_reasons", {})
        if by_reason:
            lines.append("## 6. 平仓原因分布")
            lines.append("")
            lines.append("| 原因 | 笔数 | 净盈亏 |")
            lines.append("|------|------|--------|")
            for reason, stats in sorted(by_reason.items()):
                lines.append("| {} | {} | ${:,.2f} |".format(reason, stats["count"], stats["pnl"]))

        report = "\n".join(lines)

        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(report)
            logger.info("Report saved to {}".format(output_path))

        return report

    # ============================================================
    # 可视化
    # ============================================================
    def generate_chart(self, output_path: str, days_back: int = 30) -> str:
        """生成复盘可视化图：K线 + 买卖点 + 追踪止损线"""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        if self.kline_df is None:
            logger.warning("No K-line data for chart")
            return ""

        # 取最近 N 天数据
        df = self.kline_df.copy()
        cutoff = df.index.max() - pd.Timedelta(days=days_back)
        df = df[df.index >= cutoff]

        if df.empty:
            return ""

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True,
                                        gridspec_kw={"height_ratios": [3, 1]})

        # --- 上轴: K 线 + 买卖点 ---
        colors = ["green" if df["close"].iloc[i] >= df["open"].iloc[i] else "red"
                  for i in range(len(df))]
        ax1.bar(df.index, df["high"] - df["low"], bottom=df["low"],
                width=0.0005, color=colors, alpha=0.3, linewidth=0)
        ax1.bar(df.index, abs(df["close"] - df["open"]),
                bottom=df[["open", "close"]].min(axis=1),
                width=0.0005, color=colors, alpha=0.8, linewidth=0)

        # 标记买卖点
        for t in self.trades:
            if cutoff <= t.entry_time <= df.index.max():
                # 入场点
                marker = "v" if t.direction.upper() == "SHORT" else "^"
                color = "red" if t.direction.upper() == "SHORT" else "green"
                ax1.scatter(t.entry_time, t.entry_price, marker=marker, c=color,
                            s=80, zorder=5, edgecolors="black", linewidth=0.5,
                            label="{} Entry".format(t.direction) if t.trade_id == 1 else "")
            if cutoff <= t.exit_time <= df.index.max():
                # 出场点
                marker = "^" if t.direction.upper() == "SHORT" else "v"
                ax1.scatter(t.exit_time, t.exit_price, marker=marker, c="blue",
                            s=50, zorder=5, edgecolors="black", linewidth=0.5,
                            label="Exit" if t.trade_id == 1 else "")

        # 价格均线
        df["ma20"] = df["close"].rolling(20).mean()
        ax1.plot(df.index, df["ma20"], color="orange", linewidth=1, alpha=0.7, label="MA20")

        ax1.set_ylabel("Price (USDT)")
        ax1.set_title("BTC/USDT 15m — Trade Review ({} days)".format(days_back))
        ax1.legend(loc="upper left", fontsize=8)
        ax1.grid(True, alpha=0.3)

        # --- 下轴: RSI ---
        rsi = compute_rsi(df["close"].astype(float), period=14)
        ax2.fill_between(df.index, 30, 70, color="gray", alpha=0.1)
        ax2.axhline(70, color="red", linewidth=0.5, linestyle="--")
        ax2.axhline(30, color="green", linewidth=0.5, linestyle="--")
        ax2.plot(df.index, rsi, color="purple", linewidth=1, label="RSI(14)")
        ax2.set_ylabel("RSI")
        ax2.set_xlabel("Time")
        ax2.legend(loc="upper left", fontsize=8)
        ax2.grid(True, alpha=0.3)

        # 格式化
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
        plt.xticks(rotation=45)
        plt.tight_layout()

        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Chart saved to {}".format(output_path))
        return output_path

    # ============================================================
    # 一体化输出
    # ============================================================
    def run_full_report(
        self,
        output_dir: str,
        report_name: str = "daily_review",
        send_alert: bool = False,
    ) -> Dict:
        """运行完整复盘流程，返回报告数据"""

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # 1. 绩效
        perf = self.compute_performance()

        # 2. NAV
        nav = self.compute_nav_curve()

        # 3. 执行质量
        exec_q = self.analyze_execution_quality()

        # 4. Markdown 报告
        md_path = out / "{}.md".format(report_name)
        self.generate_markdown_report(str(md_path))

        # 5. 图表
        chart_path = out / "{}_chart.png".format(report_name)
        self.generate_chart(str(chart_path))

        # 6. 与 BTC 基准对比
        btc_return = 0.0
        if nav:
            btc_return = (nav[-1].btc_nav / self.initial_capital - 1) * 100
            strategy_return = (nav[-1].equity / self.initial_capital - 1) * 100
        else:
            strategy_return = perf.get("net_pnl_pct", 0)

        # 7. 通知
        if send_alert:
            try:
                from ..execution.alerts import AlertManager
                alert = AlertManager()
                msg = "Daily Review\nNet PnL: ${:,.2f} ({:+.1f}%)\nWin Rate: {:.1f}%\nTrades: {}\nSharpe: {:.2f}\nStrategy: {:+.1f}% vs BTC HODL: {:+.1f}%".format(
                    perf.get("net_pnl", 0), perf.get("net_pnl_pct", 0),
                    perf.get("win_rate", 0), perf.get("total_trades", 0),
                    perf.get("sharpe_ratio", 0), strategy_return, btc_return)
                alert.send("Post-Trade Review", msg, "INFO")
            except Exception as e:
                logger.warning("Alert send failed: {}".format(e))

        return {
            "performance": perf,
            "execution_quality": exec_q,
            "nav_points": len(nav),
            "strategy_return_pct": strategy_return,
            "btc_benchmark_return_pct": btc_return,
            "report_md": str(md_path),
            "report_chart": str(chart_path),
        }
