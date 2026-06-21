"""
WebSocket 实时数据流模块
基于 ccxt.pro 订阅 BTC 永续合约 K 线，本地维护 15M / 2H K 线矩阵

架构说明：
- 使用 asyncio 异步事件循环
- 每个 timeframe 独立维护一份在线 K 线 DataFrame
- 新 bar 闭合时触发 on_bar_close 回调，供策略模块消费
"""
import asyncio
import time
from typing import Callable, Optional

import ccxt.pro as ccxt_pro
import pandas as pd

from ..config import (
    EXCHANGE_ID, SYMBOL, TIMEFRAMES, API_KEY, API_SECRET, API_PASSWORD,
    WS_RECONNECT_DELAY, WS_MAX_RETRIES,
)
from ..logger import get_logger

logger = get_logger(__name__)

# 回调签名: async def callback(timeframe: str, closed_bar: pd.Series, df: pd.DataFrame) -> None
OnBarCloseCallback = Callable[[str, pd.Series, pd.DataFrame], None]


class WebSocketStreamer:
    """WebSocket K 线流 —— 核心实时数据入口。

    使用方式:
        streamer = WebSocketStreamer(existing_df_15m, existing_df_2h)
        streamer.on_bar_close = my_strategy.on_new_bar
        await streamer.start()
    """

    def __init__(
        self,
        df_15m: Optional[pd.DataFrame] = None,
        df_2h: Optional[pd.DataFrame] = None,
    ):
        # 在线 K 线矩阵: 以 timestamp 为索引
        columns = ["timestamp", "open", "high", "low", "close", "volume"]
        self.df_15m = df_15m.copy() if df_15m is not None else pd.DataFrame(columns=columns)
        self.df_2h = df_2h.copy() if df_2h is not None else pd.DataFrame(columns=columns)

        for df in (self.df_15m, self.df_2h):
            if not df.empty and "timestamp" in df.columns:
                df.set_index("timestamp", inplace=True, drop=False)

        self._exchange: Optional[ccxt_pro.Exchange] = None
        self._running = False
        self._tasks: list[asyncio.Task] = []

        # 回调列表：所有订阅者都会收到新 bar 通知
        self._on_bar_close_callbacks: list[OnBarCloseCallback] = []

    @property
    def on_bar_close(self):
        """装饰器风格设置回调：@streamer.on_bar_close"""
        raise AttributeError("Use streamer.add_bar_close_callback(cb) instead")

    def add_bar_close_callback(self, cb: OnBarCloseCallback) -> None:
        """注册 K 线收盘回调"""
        self._on_bar_close_callbacks.append(cb)

    async def _notify_bar_close(self, timeframe: str, closed_bar: pd.Series, df: pd.DataFrame):
        for cb in self._on_bar_close_callbacks:
            try:
                await cb(timeframe, closed_bar, df)
            except Exception as e:
                logger.error(f"Callback error (tf={timeframe}): {e}", exc_info=True)

    # ----- 交易所实例管理 -----
    async def _get_exchange(self) -> ccxt_pro.Exchange:
        if self._exchange is None:
            exchange_class = getattr(ccxt_pro, EXCHANGE_ID)
            config = {
                "enableRateLimit": True,
                "options": {"defaultType": "swap"},
            }
            if API_KEY:
                config["apiKey"] = API_KEY
            if API_SECRET:
                config["secret"] = API_SECRET
            if API_PASSWORD:
                config["password"] = API_PASSWORD
            self._exchange = exchange_class(config)
            logger.info(f"WebSocket exchange [{EXCHANGE_ID}] created")
        return self._exchange

    # ----- 单 timeframe 订阅循环 -----
    async def _watch_timeframe(self, timeframe: str, df_attr: str) -> None:
        """订阅单个 timeframe 的 OHLCV WebSocket 流，持续更新本地矩阵"""
        exchange = await self._get_exchange()
        retries = 0

        while self._running:
            try:
                since_ms = None
                # 如果本地有空矩阵，先拉取最近 500 根 K 线用于缓存
                local_df: pd.DataFrame = getattr(self, df_attr)
                if local_df is not None and not local_df.empty:
                    # 用本地最后 timestamp + 1ms 作为 since
                    max_ts = local_df["timestamp"].max()
                    since_ms = int(max_ts) + 1

                candles = await exchange.watch_ohlcv(
                    SYMBOL,
                    timeframe=timeframe,
                    since=since_ms,
                )
                retries = 0  # 成功则重置重试计数

                # 更新本地矩阵
                self._update_klines(timeframe, df_attr, candles)

            except asyncio.CancelledError:
                logger.info(f"WebSocket [{timeframe}] cancelled")
                break
            except ccxt_pro.NetworkError as e:
                retries += 1
                logger.warning(f"WS [{timeframe}] network error (#{retries}): {e}")
                if retries > WS_MAX_RETRIES:
                    logger.error(f"WS [{timeframe}] max retries exceeded — stopping")
                    break
                await asyncio.sleep(WS_RECONNECT_DELAY)
            except Exception as e:
                retries += 1
                logger.error(f"WS [{timeframe}] unexpected error: {e}", exc_info=True)
                if retries > WS_MAX_RETRIES:
                    break
                await asyncio.sleep(WS_RECONNECT_DELAY)

    def _update_klines(self, timeframe: str, df_attr: str, candles: list) -> None:
        """将 WebSocket 推送的 K 线合并入本地矩阵，并检测新 bar 闭合"""
        local_df: pd.DataFrame = getattr(self, df_attr)

        if not candles:
            return

        # 新 candles 转 DataFrame
        new_df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
        new_df["timestamp"] = new_df["timestamp"].astype(int)

        # 合并：如果本地为空，直接赋值；否则合并去重
        if local_df is not None and not local_df.empty:
            # 确保列一致
            combined = pd.concat([local_df.reset_index(drop=True), new_df], ignore_index=True)
            combined.drop_duplicates(subset=["timestamp"], keep="last", inplace=True)
            combined.sort_values("timestamp", inplace=True)
            combined.reset_index(drop=True, inplace=True)
        else:
            combined = new_df.copy()

        # 检测刚闭合的 K 线
        # WebSocket 推送的最后一个 bar 可能是未闭合的（K 线[0]）
        # 倒数第 2 个 bar 是已闭合的（K 线[1]）
        if len(combined) >= 2:
            old_len = len(local_df) if local_df is not None and not local_df.empty else 0
            if len(combined) > old_len:
                # 有新 bar 闭合
                closed_bar = combined.iloc[-2]  # K 线[1]
                # 按时间戳定位：倒数第2根是否是新的闭合 bar
                if old_len == 0 or closed_bar["timestamp"] > local_df.iloc[-1]["timestamp"]:
                    logger.debug(
                        f"[{timeframe}] Bar closed | ts={closed_bar['timestamp']} | "
                        f"o={closed_bar['open']:.2f} h={closed_bar['high']:.2f} "
                        f"l={closed_bar['low']:.2f} c={closed_bar['close']:.2f}"
                    )
                    # 异步触发回调（不等待，避免阻塞数据流）
                    asyncio.create_task(self._notify_bar_close(timeframe, closed_bar, combined))

        # 更新本地矩阵
        setattr(self, df_attr, combined)

    # ----- 启动 / 停止 -----
    async def start(self) -> None:
        """启动 WebSocket 数据流"""
        self._running = True
        logger.info("WebSocketStreamer starting...")

        self._tasks = [
            asyncio.create_task(self._watch_timeframe("15m", "df_15m")),
            asyncio.create_task(self._watch_timeframe("2h", "df_2h")),
        ]

        # 等待所有任务完成（通常被 stop 中断）
        try:
            await asyncio.gather(*self._tasks)
        except asyncio.CancelledError:
            logger.info("WebSocketStreamer gather cancelled")

    async def stop(self) -> None:
        """优雅关闭 WebSocket 流"""
        logger.info("WebSocketStreamer stopping...")
        self._running = False

        for t in self._tasks:
            if not t.done():
                t.cancel()
        # 等待任务完成取消
        await asyncio.gather(*self._tasks, return_exceptions=True)

        if self._exchange:
            await self._exchange.close()
            logger.info("WebSocket exchange closed")

    def get_df(self, timeframe: str) -> pd.DataFrame:
        """获取当前在线 K 线矩阵的副本"""
        if timeframe == "15m":
            return self.df_15m.copy() if self.df_15m is not None else pd.DataFrame()
        elif timeframe == "2h":
            return self.df_2h.copy() if self.df_2h is not None else pd.DataFrame()
        else:
            raise ValueError(f"Unknown timeframe: {timeframe}")
