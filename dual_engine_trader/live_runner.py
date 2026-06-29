import sys, warnings, os, json, time, threading
from pathlib import Path
from datetime import datetime, timezone, date
from collections import defaultdict
from typing import Optional, Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import requests
import pandas as pd
import numpy as np

from dual_engine_trader.strategy.detector import DivergenceDetector, DivergenceParams, TrailingStopUpdater, SignalType
from dual_engine_trader.strategy.engine import Direction, LONG_PARAMS, SHORT_PARAMS
from dual_engine_trader.strategy.indicators import compute_rsi, compute_atr, pivotlow, pivothigh
from dual_engine_trader.execution.executor import OKXExecution, INST_ID
from dual_engine_trader.config import (
    HTTP_PROXY, HTTPS_PROXY, TIMEFRAMES,
    LIVE_CAPITAL, DEFAULT_LEVERAGE,
    LIVE_MAX_RISK_PCT, LIVE_MAX_CAPITAL_PCT, LIVE_MAX_CONTRACTS,
    DAILY_DD_LIMIT, CUMULATIVE_DD_REDUCE, CUMULATIVE_DD_STOP,
)
from dual_engine_trader.logger import setup_logger

logger = setup_logger("live_runner")

TRADE_LOG_DIR = Path(__file__).resolve().parent.parent.parent / "live_output"
TRADE_LOG_DIR.mkdir(parents=True, exist_ok=True)

PROXIES = None
if HTTP_PROXY or HTTPS_PROXY:
    PROXIES = {}
    if HTTP_PROXY:
        PROXIES["http"] = HTTP_PROXY
    if HTTPS_PROXY:
        PROXIES["https"] = HTTPS_PROXY

CANDLE_LIMITS = {"15m": 500, "2h": 200}


class LiveRunner:
    """实盘运行器 — 500U 5x 激进方案

    核心参数：
       首笔 2 张 → 加仓 2 张 = 最多 4 张
       ATR(14) 止损距离，盈亏比 1.67

    安全熔断：
       - 当日回撤 >10%    → 暂停当天交易
       - 累计回撤 >25%    → 降为 1 张模式
       - 累计回撤 >40%    → 全部平仓停止
    """

    def __init__(self, capital=None, leverage=None, max_risk_pct=None,
                 max_capital_pct=None, max_contracts=None):
        self.capital = capital if capital is not None else LIVE_CAPITAL
        self.leverage = leverage if leverage is not None else DEFAULT_LEVERAGE
        self.max_risk_pct = max_risk_pct if max_risk_pct is not None else LIVE_MAX_RISK_PCT
        self.max_capital_pct = max_capital_pct if max_capital_pct is not None else LIVE_MAX_CAPITAL_PCT
        self.max_contracts = max_contracts if max_contracts is not None else LIVE_MAX_CONTRACTS

        self.short_detector = DivergenceDetector(SHORT_PARAMS)
        self.long_detector = DivergenceDetector(LONG_PARAMS)
        self.short_sl_updater = TrailingStopUpdater(stop_loss_mult=SHORT_PARAMS.stop_loss_mult)
        self.long_sl_updater = TrailingStopUpdater(stop_loss_mult=LONG_PARAMS.stop_loss_mult)

        self.executor = OKXExecution(
            max_risk_pct=self.max_risk_pct,
            max_capital_pct=self.max_capital_pct,
            max_contracts=self.max_contracts,
            leverage=self.leverage,
        )

        # 在线数据
        self._data: Dict[str, pd.DataFrame] = {}
        self._last_bar_ts: Dict[str, int] = {}
        self._position_entry_bar: Dict[str, int] = {}

        self._running = False
        self._trade_log = []

        # ---- 风险监控 ----
        self._day_start_equity = None          # 今日初始权益
        self._peak_equity = 0.0                # 周期内峰值权益
        self._circuit_breaker = {
            "daily_paused": False,             # 当日熔断暂停
            "reduced_mode": False,             # 降为1张模式
            "stopped": False,                  # 完全停止
            "reason": "",                      # 熔断原因
        }

    # ------------------------------------------------------------
    # 行情获取
    # ------------------------------------------------------------
    def fetch_recent_candles(self, timeframe: str, limit=500) -> Optional[pd.DataFrame]:
        url = "https://www.okx.com/api/v5/market/history-candles"
        bar_map = {"15m": "15m", "2h": "2H"}
        params = {"instId": INST_ID, "bar": bar_map.get(timeframe, timeframe), "limit": str(limit)}
        r = requests.get(url, params=params, proxies=PROXIES, timeout=15)
        data = r.json()
        if data.get("code") != "0" or not data.get("data"):
            return None
        rows = []
        for c in data["data"]:
            ts = int(c[0])
            rows.append({"timestamp": ts, "open": float(c[1]), "high": float(c[2]),
                         "low": float(c[3]), "close": float(c[4]), "volume": float(c[5])})
        df = pd.DataFrame(rows)
        df.sort_values("timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    # ------------------------------------------------------------
    # 风险监控
    # ------------------------------------------------------------
    def _check_circuit_breakers(self, equity: float) -> bool:
        """检查熔断条件，返回 True = 允许继续交易"""
        now = datetime.now(timezone.utc)

        # 每日起始权益
        if self._day_start_equity is None:
            self._day_start_equity = equity
        elif now.hour == 0 and now.minute < 5:
            self._day_start_equity = equity
            self._circuit_breaker["daily_paused"] = False

        # 更新峰值权益
        if equity > self._peak_equity:
            self._peak_equity = equity

        # 当日亏损
        day_pnl_pct = (equity - self._day_start_equity) / self._day_start_equity * 100
        if day_pnl_pct <= -DAILY_DD_LIMIT and not self._circuit_breaker["daily_paused"]:
            self._circuit_breaker["daily_paused"] = True
            self._circuit_breaker["reason"] = "daily_dd"
            msg = "DAILY DD > {}% — paused until UTC 00:00. equity=${:.0f} day_start=${:.0f}".format(
                DAILY_DD_LIMIT, equity, self._day_start_equity)
            logger.warning(msg)
            self._safe_close_position("Daily DD limit hit")

        # 累计亏损
        total_dd_pct = (equity - self.capital) / self.capital * 100
        if total_dd_pct <= -CUMULATIVE_DD_STOP and not self._circuit_breaker["stopped"]:
            self._circuit_breaker["stopped"] = True
            self._circuit_breaker["reason"] = "total_dd"
            msg = "TOTAL DD > {}% — STOPPED. equity=${:.0f} initial=${:.0f}".format(
                CUMULATIVE_DD_STOP, equity, self.capital)
            logger.error(msg)
            self._safe_close_position("Total DD stop")
            return False

        if total_dd_pct <= -CUMULATIVE_DD_REDUCE and not self._circuit_breaker["reduced_mode"]:
            self._circuit_breaker["reduced_mode"] = True
            self._circuit_breaker["reason"] = "total_dd_reduce"
            msg = "TOTAL DD > {}% — reduced to 1 contract mode. equity=${:.0f}".format(
                CUMULATIVE_DD_REDUCE, equity)
            logger.warning(msg)
            self._safe_close_position("Total DD reduction")

        return not self._circuit_breaker["stopped"]

    def _safe_close_position(self, reason: str):
        """安全平仓——不因网络异常中断熔断流程"""
        try:
            pos = self.executor.get_position()
            if pos:
                self.executor.close_position()
                logger.info("Circuit breaker: position closed ({})".format(reason))
        except Exception as e:
            logger.error("Circuit breaker: close_position failed ({}) — will retry: {}".format(reason, e))

    def _effective_max_contracts(self) -> float:
        """返回当前生效的单笔最大合约数"""
        if self._circuit_breaker["reduced_mode"]:
            return min(self.max_contracts, 2.0)  # 缩仓模式保留一半
        return self.max_contracts

    # ------------------------------------------------------------
    # K 线处理
    # ------------------------------------------------------------
    def process_new_bar(self, timeframe: str, bar_index: int):
        df = self._data.get(timeframe)
        warmup = 80 if timeframe == "15m" else 40
        if df is None or len(df) < warmup:
            return

        # 先检查熔断
        bal = self.executor.get_balance()
        equity = bal["total_equity"]
        if not self._check_circuit_breakers(equity):
            return

        # 每次开仓前重设 max_contracts（熔断模式下可能为1）
        effective_mc = self._effective_max_contracts()
        self.executor.max_contracts = effective_mc

        detector = self.short_detector if timeframe == "15m" else self.long_detector
        direction = Direction.SHORT if timeframe == "15m" else Direction.LONG

        # 实盘只使用 15M 做空信号（2H 引擎暂不参与实盘）
        if timeframe == "2h":
            return

        signals = detector.detect(df.iloc[:bar_index + 1], timeframe)
        # 只处理当前 bar 产生的信号，避免重复执行历史信号
        current_signals = []
        for sig in signals:
            meta = sig.metadata
            pivot_idx = meta.get("pivot_b") if sig.signal_type in (SignalType.BUY, SignalType.SELL) else None
            if pivot_idx is not None:
                if pivot_idx == bar_index:
                    current_signals.append(sig)
            elif sig.signal_type == SignalType.CLOSE_SHORT:
                ts_diff = abs(int(sig.timestamp) - int(bar_ts))
                if ts_diff <= 900000:  # 15分钟窗口
                    current_signals.append(sig)
        signals = current_signals

        now = datetime.now(timezone.utc)
        bar = df.iloc[bar_index]
        bar_ts = int(bar["timestamp"])
        bar_close = float(bar["close"])
        bar_high = float(bar["high"])

        for sig in signals:
            log_entry = {
                "time": str(now), "tf": timeframe, "bar_ts": bar_ts,
                "bar_close": bar_close, "signal_type": sig.signal_type.value,
                "divergence_type": sig.divergence_type.value if sig.divergence_type else "",
                "price": sig.price, "rsi": sig.rsi_value, "atr": sig.atr_value,
                "circuit_breaker": self._circuit_breaker["reason"] if self._circuit_breaker.get("daily_paused") or self._circuit_breaker.get("reduced_mode") else "",
            }

            if direction == Direction.SHORT and sig.signal_type == SignalType.SELL:
                if self._circuit_breaker["daily_paused"] or self._circuit_breaker["stopped"]:
                    logger.info("SELL blocked by circuit breaker: {}".format(self._circuit_breaker["reason"]))
                    log_entry["action"] = "OPEN_SHORT"
                    log_entry["result"] = "BLOCKED"
                    log_entry["message"] = "circuit_breaker: {}".format(self._circuit_breaker["reason"])
                else:
                    sl_price = sig.trailing_sl or (bar_high + SHORT_PARAMS.stop_loss_mult * (sig.atr_value or 500))
                    ok, msg, srep = self.executor.open_short(sl_price)
                    log_entry["action"] = "OPEN_SHORT"
                    log_entry["sl_price"] = sl_price
                    log_entry["result"] = "OK" if ok else "FAIL"
                    log_entry["message"] = msg
                    log_entry["sizing"] = srep
                    if ok:
                        self._position_entry_bar["15m"] = bar_index
                    logger.info("SELL: {} | {} (mc={})".format(msg, "OPENED" if ok else "BLOCKED", effective_mc))

            elif sig.signal_type == SignalType.CLOSE_SHORT:
                ok, msg = self.executor.close_position()
                log_entry["action"] = "CLOSE_SHORT"
                log_entry["result"] = "OK" if ok else "FAIL"
                log_entry["message"] = msg
                if ok:
                    self._position_entry_bar["15m"] = None
                logger.info("CLOSE SHORT: {}".format(msg))

            elif sig.signal_type == SignalType.CLOSE_LONG:
                ok, msg = self.executor.close_position()
                log_entry["action"] = "CLOSE_LONG"
                log_entry["result"] = "OK" if ok else "FAIL"
                log_entry["message"] = msg
                if ok:
                    self._position_entry_bar["2h"] = None
                logger.info("CLOSE LONG: {}".format(msg))

            self._trade_log.append(log_entry)

    # ------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------
    def run(self, max_duration_hours=None):
        self._running = True
        start_time = time.time()
        logger.info("=" * 50)
        logger.info("LIVE RUNNER — 500U / 5x / Aggressive")
        logger.info("  capital={} leverage={}x".format(self.capital, self.leverage))
        logger.info("  max_contracts={} risk_pct={}% cap_pct={}%".format(
            self.max_contracts, self.max_risk_pct, self.max_capital_pct))
        logger.info("  Circuit breakers: daily_dd>{}%  reduce>{}%  stop>{}%".format(
            DAILY_DD_LIMIT, CUMULATIVE_DD_REDUCE, CUMULATIVE_DD_STOP))
        logger.info("=" * 50)

        bal = self.executor.get_balance()
        self._peak_equity = bal["total_equity"]
        logger.info("Account: equity=${:.2f} free=${:.2f}".format(bal["total_equity"], bal["free"]))

        # 初始化 K 线
        for tf in TIMEFRAMES:
            limit = CANDLE_LIMITS.get(tf, 500)
            df = self.fetch_recent_candles(tf, limit=limit)
            if df is None or df.empty:
                logger.error("No initial data for {}".format(tf))
                return
            self._data[tf] = df
            self._last_bar_ts[tf] = int(df["timestamp"].iloc[-1])
            logger.info("Data [{}]: {} bars, latest: {}".format(tf, len(df),
                         pd.Timestamp(self._last_bar_ts[tf], unit="ms")))

        iteration = 0
        while self._running:
            iteration += 1
            if max_duration_hours and (time.time() - start_time) > max_duration_hours * 3600:
                break
            if self._circuit_breaker["stopped"]:
                logger.error("STOPPED by circuit breaker — exiting")
                break

            try:
                for tf in TIMEFRAMES:
                    limit = CANDLE_LIMITS.get(tf, 500)
                    new_df = self.fetch_recent_candles(tf, limit=limit)
                    if new_df is not None:
                        combined = pd.concat([self._data[tf], new_df], ignore_index=True)
                        combined.drop_duplicates(subset=["timestamp"], keep="last", inplace=True)
                        combined.sort_values("timestamp", inplace=True)
                        combined.reset_index(drop=True, inplace=True)
                        self._data[tf] = combined

                        latest_ts = int(combined["timestamp"].iloc[-1])
                        if latest_ts > self._last_bar_ts[tf]:
                            warmup = 80 if tf == "15m" else 40
                            for idx in range(len(combined)):
                                ts = int(combined["timestamp"].iloc[idx])
                                if ts > self._last_bar_ts[tf] and idx >= warmup - 1:
                                    logger.info("Bar [{}]: {} C={:.0f}".format(tf,
                                        pd.Timestamp(ts, unit='ms'), combined["close"].iloc[idx]))
                                    self.process_new_bar(tf, idx)
                                    self._last_bar_ts[tf] = ts

                if iteration % 10 == 0:
                    pos = self.executor.get_position()
                    bal = self.executor.get_balance()
                    eq = bal["total_equity"]
                    day_pnl = (eq - (self._day_start_equity or eq)) / (self._day_start_equity or eq) * 100
                    total_pnl = (eq - self.capital) / self.capital * 100
                    cb_status = ""
                    if self._circuit_breaker["daily_paused"]:
                        cb_status = " [DAILY_PAUSED]"
                    elif self._circuit_breaker["reduced_mode"]:
                        cb_status = " [REDUCED-1CT]"
                    logger.info("Status: POS={} EQUITY=${:.0f} day={:+.1f}% total={:+.1f}%{}".format(
                        "{} {}ct".format(pos["side"], pos["contracts"]) if pos else "NONE",
                        eq, day_pnl, total_pnl, cb_status))

            except Exception as e:
                logger.error("Iter {} error: {}".format(iteration, e))
            time.sleep(60)

        self.shutdown()

    def shutdown(self):
        self._running = False
        logger.info("Shutting down...")
        pos = self.executor.get_position()
        if pos:
            logger.warning("Closing position: {} {}ct".format(pos["side"], pos["contracts"]))
            self.executor.close_position()
        log_path = TRADE_LOG_DIR / "live_trades_{}.json".format(datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))
        with open(log_path, "w") as f:
            json.dump(self._trade_log, f, indent=2, default=str)
        logger.info("Log: {}".format(log_path))
        self.executor.close()


def main():
    runner = LiveRunner()
    try:
        runner.run()
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt")
        runner.shutdown()


if __name__ == "__main__":
    main()
