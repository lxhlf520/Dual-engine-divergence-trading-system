"""
Backtest engine - bar-by-bar simulation with virtual account.
"""
from datetime import datetime, timezone
from typing import Optional, List, Dict
from pathlib import Path
import numpy as np
import pandas as pd
from ..config import BACKTEST_INITIAL_CAPITAL, BACKTEST_FEE_RATE, BACKTEST_SLIPPAGE_RATE, BACKTEST_PYRAMIDING
from ..strategy.detector import DivergenceDetector, TrailingStopUpdater, Signal, SignalType
from ..strategy.engine import Direction, LONG_PARAMS, SHORT_PARAMS
from ..strategy.indicators import compute_atr
from ..logger import get_logger
from .account import VirtualAccount, Trade, Position

logger = get_logger(__name__)


class BacktestEngine:
    def __init__(self, initial_capital=BACKTEST_INITIAL_CAPITAL, fee_rate=BACKTEST_FEE_RATE, slippage_rate=BACKTEST_SLIPPAGE_RATE, max_pyramiding=BACKTEST_PYRAMIDING):
        self.initial_capital = initial_capital
        self.account = VirtualAccount(initial_capital=initial_capital, fee_rate=fee_rate, slippage_rate=slippage_rate, max_pyramiding=max_pyramiding)
        self.long_detector = DivergenceDetector(LONG_PARAMS)
        self.short_detector = DivergenceDetector(SHORT_PARAMS)
        self.long_sl_updater = TrailingStopUpdater(stop_loss_mult=LONG_PARAMS.stop_loss_mult)
        self.short_sl_updater = TrailingStopUpdater(stop_loss_mult=SHORT_PARAMS.stop_loss_mult)
        self.df_15m = None
        self.df_2h = None
        self.signals_log = []
        self.events_log = []

    def load_data(self, timeframe, df):
        df = df.copy()
        df["timestamp"] = df["timestamp"].astype(int)
        df.sort_values("timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)
        if timeframe == "15m":
            self.df_15m = df
        elif timeframe == "2h":
            self.df_2h = df
        else:
            raise ValueError(f"Unknown timeframe: {timeframe}")
        logger.info(f"Backtest data loaded: [{timeframe}] {len(df)} bars")

    def load_csv(self, timeframe, filepath):
        df = pd.read_csv(filepath)
        self.load_data(timeframe, df)

    def run(self, warmup_bars=50):
        if self.df_15m is None and self.df_2h is None:
            raise RuntimeError("No data loaded")
        logger.info("=" * 60)
        logger.info("Backtest starting...")
        logger.info("  Initial capital: ${:,.2f}".format(self.initial_capital))
        logger.info("=" * 60)

        if self.df_15m is None:
            # Only 2H data — run standalone
            self._run_on_timeframe(self.df_2h, "2h", self.long_detector, self.long_sl_updater, warmup_bars)
            logger.info("=" * 60)
            logger.info("Backtest complete")
            logger.info("=" * 60)
            return self.generate_report()

        # Use 15M as the unified timeline
        df_15m = self.df_15m
        n = len(df_15m)
        warmup_15m = max(warmup_bars, 80)

        # Pre-compute signals for both engines
        short_signals = self.short_detector.detect(df_15m, "15m")
        sigs_by_bar = self._index_signals(short_signals, df_15m)

        long_signals = []
        if self.df_2h is not None:
            long_signals = self.long_detector.detect(self.df_2h, "2h")
            # Map 2H signals to the last 15M bar in each 2H period
            sigs_by_bar = self._merge_2h_signals(long_signals, self.df_2h, df_15m, sigs_by_bar)

        account = self.account

        # Pre-compute 15M ATR for stop updates
        atr_series_15m = compute_atr(
            df_15m["high"].astype(float), df_15m["low"].astype(float),
            df_15m["close"].astype(float), period=14,
        )

        for i in range(warmup_15m, n):
            bar = df_15m.iloc[i]
            bar_ts = int(bar["timestamp"])
            bar_high = float(bar["high"])
            bar_low = float(bar["low"])
            bar_close = float(bar["close"])
            current_atr_val = float(atr_series_15m.iloc[i]) if not pd.isna(atr_series_15m.iloc[i]) else 0.0

            if i % 10 == 0:
                account.record_equity(bar_ts)

            # ---- 1. Update trailing stops for both directions ----
            for pi in range(len(account.long_positions)):
                pos = account.long_positions[pi]
                new_sl = self.long_sl_updater.update_long_sl(pos.trailing_sl, bar_low, current_atr_val)
                account.update_trailing_sl_long(new_sl, pi)

            for pi in range(len(account.short_positions)):
                pos = account.short_positions[pi]
                new_sl = self.short_sl_updater.update_short_sl(pos.trailing_sl, bar_high, current_atr_val)
                account.update_trailing_sl_short(new_sl, pi)

            # ---- 2. Check stop-losses ----
            closed_long = account.check_long_stops(bar_low, bar_high, bar_ts, i)
            for trade in closed_long:
                self.events_log.append({"bar": i, "ts": bar_ts, "event": "stop", "direction": "LONG", "price": trade.exit_price, "pnl": trade.net_pnl, "reason": "TSL_STOP"})

            closed_short = account.check_short_stops(bar_low, bar_high, bar_ts, i)
            for trade in closed_short:
                self.events_log.append({"bar": i, "ts": bar_ts, "event": "stop", "direction": "SHORT", "price": trade.exit_price, "pnl": trade.net_pnl, "reason": "TSL_STOP"})

            # ---- 3. Process signals ----
            if i in sigs_by_bar:
                for sig in sigs_by_bar[i]:
                    self.signals_log.append({
                        "timestamp": sig.timestamp, "timeframe": sig.timeframe,
                        "signal_type": sig.signal_type.value,
                        "divergence_type": sig.divergence_type.value if sig.divergence_type else "",
                        "price": sig.price,
                    })
                    if sig.signal_type == SignalType.BUY:
                        pos = account.open_position(
                            direction=Direction.LONG, price=bar_close, timestamp=bar_ts,
                            bar_index=i, trailing_sl=sig.trailing_sl or 0.0,
                            divergence_type=sig.divergence_type.value if sig.divergence_type else "",
                        )
                        if pos:
                            self.events_log.append({"bar": i, "ts": bar_ts, "event": "open", "direction": "LONG", "price": bar_close, "sl": sig.trailing_sl})
                    elif sig.signal_type == SignalType.SELL:
                        pos = account.open_position(
                            direction=Direction.SHORT, price=bar_close, timestamp=bar_ts,
                            bar_index=i, trailing_sl=sig.trailing_sl or 0.0,
                            divergence_type=sig.divergence_type.value if sig.divergence_type else "",
                        )
                        if pos:
                            self.events_log.append({"bar": i, "ts": bar_ts, "event": "open", "direction": "SHORT", "price": bar_close, "sl": sig.trailing_sl})
                    elif sig.signal_type == SignalType.CLOSE_LONG:
                        while account.long_positions:
                            trade = account.close_position(direction=Direction.LONG, price=bar_close, timestamp=bar_ts, bar_index=i, reason="SIGNAL_CLOSE_LONG")
                            if trade:
                                self.events_log.append({"bar": i, "ts": bar_ts, "event": "close", "direction": "LONG", "price": bar_close, "pnl": trade.net_pnl, "reason": trade.exit_reason})
                    elif sig.signal_type == SignalType.CLOSE_SHORT:
                        while account.short_positions:
                            trade = account.close_position(direction=Direction.SHORT, price=bar_close, timestamp=bar_ts, bar_index=i, reason="SIGNAL_CLOSE_SHORT")
                            if trade:
                                self.events_log.append({"bar": i, "ts": bar_ts, "event": "close", "direction": "SHORT", "price": bar_close, "pnl": trade.net_pnl, "reason": trade.exit_reason})

        # Force close remaining positions
        last_close = float(df_15m["close"].iloc[-1])
        while account.long_positions:
            account.close_position(direction=Direction.LONG, price=last_close, timestamp=int(df_15m["timestamp"].iloc[-1]), bar_index=n-1, reason="EOD_FORCE_CLOSE")
        while account.short_positions:
            account.close_position(direction=Direction.SHORT, price=last_close, timestamp=int(df_15m["timestamp"].iloc[-1]), bar_index=n-1, reason="EOD_FORCE_CLOSE")

        logger.info("=" * 60)
        logger.info("Backtest complete (combined timeline)")
        logger.info("=" * 60)
        return self.generate_report()

    def _index_signals(self, signals, df):
        """Index signals by bar index for O(1) lookup."""
        sigs_by_bar = {}
        for sig in signals:
            meta = sig.metadata
            bar_idx = meta.get("pivot_b") if sig.signal_type in (SignalType.BUY, SignalType.SELL) else None
            if bar_idx is not None:
                sigs_by_bar.setdefault(bar_idx, []).append(sig)
        for sig in signals:
            if sig.signal_type in (SignalType.CLOSE_LONG, SignalType.CLOSE_SHORT):
                ts_diffs = (df["timestamp"].astype(int) - sig.timestamp).abs()
                if ts_diffs.min() == 0:
                    bar_idx = ts_diffs.idxmin()
                    sigs_by_bar.setdefault(bar_idx, []).append(sig)
        return sigs_by_bar

    def _merge_2h_signals(self, long_signals, df_2h, df_15m, sigs_by_bar):
        """Map 2H signals to the corresponding 15M bar timestamp."""
        # Build a map: 2H bar timestamp -> list of signals
        for sig in long_signals:
            sig_ts = sig.timestamp
            # Find the nearest 15M bar (should be exact or the last in that 2H period)
            ts_diffs = (df_15m["timestamp"].astype(int) - sig_ts).abs()
            nearest_idx = ts_diffs.idxmin()
            if ts_diffs.min() <= 7200000:  # within 2 hours = one 2H bar
                sigs_by_bar.setdefault(nearest_idx, []).append(sig)
        return sigs_by_bar

    def generate_report(self):
        trades = self.account.closed_trades
        n = len(trades)
        if n == 0:
            return {"initial_capital": self.initial_capital, "final_capital": self.initial_capital, "total_trades": 0, "net_profit": 0.0, "net_profit_pct": 0.0, "win_rate": 0.0, "profit_factor": 0.0, "avg_win": 0.0, "avg_loss": 0.0, "avg_trade": 0.0, "max_win": 0.0, "max_loss": 0.0, "max_drawdown": 0.0, "avg_win_to_loss": 0.0, "total_fees": 0.0, "total_slippage": 0.0, "total_costs": 0.0, "long_trades": 0, "long_pnl": 0.0, "short_trades": 0, "short_pnl": 0.0, "avg_bars_held": 0.0, "exit_reasons": {}}
        wins = [t for t in trades if t.net_pnl > 0]
        losses = [t for t in trades if t.net_pnl <= 0]
        win_rate = len(wins) / n * 100
        total_profit = sum(t.net_pnl for t in wins)
        total_loss = abs(sum(t.net_pnl for t in losses))
        profit_factor = total_profit / total_loss if total_loss > 0 else float("inf")
        avg_win = np.mean([t.net_pnl for t in wins]) if wins else 0.0
        avg_loss = np.mean([t.net_pnl for t in losses]) if losses else 0.0
        net_profit = sum(t.net_pnl for t in trades)
        final_capital = self.initial_capital + net_profit
        max_drawdown = self.account.max_drawdown_pct
        total_fees = sum(t.entry_fee + t.exit_fee for t in trades)
        total_slippage = sum(t.slippage for t in trades)
        long_trades = [t for t in trades if t.direction == Direction.LONG]
        short_trades = [t for t in trades if t.direction == Direction.SHORT]
        long_pnl = sum(t.net_pnl for t in long_trades)
        short_pnl = sum(t.net_pnl for t in short_trades)
        by_reason = {}
        for t in trades:
            r = t.exit_reason
            if r not in by_reason:
                by_reason[r] = {"count": 0, "pnl": 0.0}
            by_reason[r]["count"] += 1
            by_reason[r]["pnl"] += t.net_pnl
        avg_bars = np.mean([t.bars_held for t in trades]) if trades else 0
        return {"initial_capital": self.initial_capital, "final_capital": final_capital, "total_trades": n, "net_profit": net_profit, "net_profit_pct": (net_profit/self.initial_capital)*100, "win_rate": win_rate, "profit_factor": profit_factor, "avg_win": avg_win, "avg_loss": avg_loss, "avg_trade": np.mean([t.net_pnl for t in trades]), "max_win": max(t.net_pnl for t in trades), "max_loss": min(t.net_pnl for t in trades), "max_drawdown": max_drawdown, "avg_win_to_loss": abs(avg_win/avg_loss) if avg_loss != 0 else float("inf"), "total_fees": total_fees, "total_slippage": total_slippage, "total_costs": total_fees + total_slippage, "long_trades": len(long_trades), "long_pnl": long_pnl, "short_trades": len(short_trades), "short_pnl": short_pnl, "avg_bars_held": avg_bars, "exit_reasons": by_reason, "trades": trades}

    def print_report(self, report=None):
        if report is None:
            report = self.generate_report()
        print("\n" + "=" * 70)
        print("  BACKTEST REPORT - Dual-Engine Divergence Strategy")
        print("=" * 70)
        print("\n  Capital")
        print("     Initial:            ${:>12,.2f}".format(report['initial_capital']))
        print("     Final:              ${:>12,.2f}".format(report['final_capital']))
        print("     Net Profit:         ${:>12,.2f}  ({:+.2f}%)".format(report['net_profit'], report['net_profit_pct']))
        print("\n  Performance")
        print("     Total Trades:        {:>12}".format(report['total_trades']))
        print("     Win Rate:            {:>11.1f}%".format(report['win_rate']))
        print("     Profit Factor:       {:>12.2f}".format(report['profit_factor']))
        print("     Avg Win/Loss Ratio:  {:>12.2f}".format(report['avg_win_to_loss']))
        print("     Max Drawdown:        {:>11.2f}%".format(report['max_drawdown']))
        print("\n  Trade PnL")
        print("     Avg Win:             ${:>12,.2f}".format(report['avg_win']))
        print("     Avg Loss:            ${:>12,.2f}".format(report['avg_loss']))
        print("     Avg Trade:           ${:>12,.2f}".format(report['avg_trade']))
        print("     Max Win:             ${:>12,.2f}".format(report['max_win']))
        print("     Max Loss:            ${:>12,.2f}".format(report['max_loss']))
        print("\n  Costs")
        print("     Total Fees:          ${:>12,.2f}".format(report['total_fees']))
        print("     Total Slippage:      ${:>12,.2f}".format(report['total_slippage']))
        print("     Total Costs:         ${:>12,.2f}".format(report['total_costs']))
        print("\n  By Direction")
        print("     Long  ({:>3} trades): ${:>12,.2f}".format(report['long_trades'], report['long_pnl']))
        print("     Short ({:>3} trades): ${:>12,.2f}".format(report['short_trades'], report['short_pnl']))
        print("\n  Avg Bars Held:     {:>12.1f}".format(report['avg_bars_held']))
        if report.get("exit_reasons"):
            print("\n  Exit Reasons")
            for reason, stats in sorted(report["exit_reasons"].items()):
                print("     {:<20s}  {:>4} trades  PnL: ${:>10,.2f}".format(reason, stats['count'], stats['pnl']))
        print("\n" + "=" * 70)

    def export_report(self, filepath, report=None):
        if report is None:
            report = self.generate_report()
        trades = report.get("trades", [])
        if trades:
            rows = []
            for t in trades:
                rows.append({"direction": t.direction.value, "entry_time": datetime.fromtimestamp(t.entry_time/1000, tz=timezone.utc).isoformat(), "exit_time": datetime.fromtimestamp(t.exit_time/1000, tz=timezone.utc).isoformat(), "entry_price": t.entry_price, "exit_price": t.exit_price, "net_pnl": t.net_pnl, "net_pnl_pct": t.net_pnl_pct, "exit_reason": t.exit_reason, "divergence_type": t.divergence_type, "bars_held": t.bars_held, "entry_fee": t.entry_fee, "exit_fee": t.exit_fee, "slippage": t.slippage})
            df_trades = pd.DataFrame(rows)
            df_trades.to_csv(filepath, index=False)
            logger.info("Trades exported to {} ({} rows)".format(filepath, len(df_trades)))
        summary_path = str(Path(filepath).with_suffix(".summary.csv"))
        summary_rows = [{k: v for k, v in report.items() if k != "trades"}]
        pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
        logger.info("Summary exported to {}".format(summary_path))
