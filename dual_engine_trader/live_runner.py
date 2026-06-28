import sys, warnings, os, json, time, threading
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import requests
import pandas as pd
import numpy as np

from dual_engine_trader.strategy.detector import DivergenceDetector, DivergenceParams, TrailingStopUpdater, SignalType
from dual_engine_trader.strategy.engine import Direction, LONG_PARAMS, SHORT_PARAMS
from dual_engine_trader.strategy.indicators import compute_rsi, compute_atr, pivotlow, pivothigh
from dual_engine_trader.execution.executor import OKXExecution, INST_ID
from dual_engine_trader.config import HTTP_PROXY, HTTPS_PROXY, TIMEFRAMES
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
    def __init__(self, capital=50.0, leverage=2, max_risk_pct=15.0, max_capital_pct=70.0, max_contracts=5):
        self.capital = capital
        self.leverage = leverage

        # 双引擎检测器
        self.short_detector = DivergenceDetector(SHORT_PARAMS)
        self.long_detector = DivergenceDetector(LONG_PARAMS)
        self.short_sl_updater = TrailingStopUpdater(stop_loss_mult=SHORT_PARAMS.stop_loss_mult)
        self.long_sl_updater = TrailingStopUpdater(stop_loss_mult=LONG_PARAMS.stop_loss_mult)

        self.executor = OKXExecution(max_risk_pct=max_risk_pct, max_capital_pct=max_capital_pct, max_contracts=max_contracts)
        self.executor._leverage = leverage

        # 在线 K 线矩阵
        self._data: dict[str, pd.DataFrame] = {}
        self._last_bar_ts: dict[str, int] = {}
        self._position_entry_bar: dict[str, int] = {}

        self._running = False
        self._trade_log = []

    def fetch_recent_candles(self, timeframe: str, limit=500) -> pd.DataFrame | None:
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
            rows.append({"timestamp": ts, "open": float(c[1]), "high": float(c[2]), "low": float(c[3]), "close": float(c[4]), "volume": float(c[5])})
        df = pd.DataFrame(rows)
        df.sort_values("timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    def process_new_bar(self, timeframe: str, bar_index: int):
        """处理新闭合 K 线：检测背离信号并执行交易"""
        df = self._data.get(timeframe)
        warmup = 80 if timeframe == "15m" else 40
        if df is None or len(df) < warmup:
            return

        detector = self.short_detector if timeframe == "15m" else self.long_detector
        sl_updater = self.short_sl_updater if timeframe == "15m" else self.long_sl_updater
        direction = Direction.SHORT if timeframe == "15m" else Direction.LONG

        signals = detector.detect(df.iloc[:bar_index + 1], timeframe)
        now = datetime.now(timezone.utc)
        bar = df.iloc[bar_index]
        bar_ts = int(bar["timestamp"])
        bar_close = float(bar["close"])
        bar_high = float(bar["high"])
        bar_low = float(bar["low"])

        for sig in signals:
            log_entry = {
                "time": str(now), "tf": timeframe, "bar_ts": bar_ts,
                "bar_close": bar_close, "signal_type": sig.signal_type.value,
                "divergence_type": sig.divergence_type.value if sig.divergence_type else "",
                "price": sig.price, "rsi": sig.rsi_value, "atr": sig.atr_value,
            }

            if direction == Direction.SHORT and sig.signal_type == SignalType.SELL:
                sl_price = sig.trailing_sl or (bar_high + SHORT_PARAMS.stop_loss_mult * (sig.atr_value or 500))
                ok, msg, srep = self.executor.open_short(sl_price)
                log_entry["action"] = "OPEN_SHORT"
                log_entry["sl_price"] = sl_price
                log_entry["result"] = "OK" if ok else "FAIL"
                log_entry["message"] = msg
                log_entry["sizing"] = srep
                if ok:
                    self._position_entry_bar["15m"] = bar_index
                logger.info("SELL: {} | {}".format(msg, "OPENED" if ok else "BLOCKED"))

            elif direction == Direction.LONG and sig.signal_type == SignalType.BUY:
                # 多单开仓（暂用 close_position 取反做 demo，实盘应调用 open_long）
                logger.info("LONG signal — manual check required: BUY @ {:.0f} SL={:.0f}".format(bar_close, sig.trailing_sl or 0))
                log_entry["action"] = "OPEN_LONG"
                log_entry["result"] = "SIGNAL_ONLY"
                log_entry["message"] = "LONG signal detected"

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

    def run(self, max_duration_hours=None):
        self._running = True
        start_time = time.time()
        logger.info("LIVE RUNNER STARTED capital={} leverage={}x".format(self.capital, self.leverage))
        bal = self.executor.get_balance()
        logger.info("Account: equity=${:.2f} free=${:.2f}".format(bal["total_equity"], bal["free"]))

        # 初始化两个 timeframes 的 K 线数据
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
                    logger.info("Status: POS={} BAL=${:.2f}".format(
                        "{} {}ct".format(pos["side"], pos["contracts"]) if pos else "NONE",
                        bal["total_equity"]))

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
    runner = LiveRunner(capital=50.0, leverage=2, max_risk_pct=15.0, max_capital_pct=70.0, max_contracts=5)
    try:
        runner.run()
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt")
        runner.shutdown()

if __name__ == "__main__":
    main()
