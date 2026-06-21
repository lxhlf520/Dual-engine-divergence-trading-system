"""
数据模块 统一入口
对外暴露 DataModule 类，屏蔽底层 Store / Downloader / Streamer / Reconciler 细节
"""
import asyncio
from typing import Optional

import pandas as pd

from ..config import TIMEFRAMES
from ..logger import get_logger
from .store import DataStore
from .downloader import HistoricalDownloader
from .streamer import WebSocketStreamer, OnBarCloseCallback
from .reconciler import ReconciliationEngine

logger = get_logger(__name__)


class DataModule:
    """数据模块 外观类 (Facade)

    提供统一的 API：
        # 历史模式
        dm = DataModule(storage_mode="sqlite")
        dm.download_history("15m", since="2025-01-01")
        dm.load_klines("2h")

        # 实盘模式
        dm = DataModule(storage_mode="sqlite")
        dm.on_bar_close = strategy.on_new_bar
        await dm.run_live()
    """

    def __init__(self, storage_mode: str = "sqlite"):
        """
        Args:
            storage_mode: "sqlite" 或 "csv"
        """
        self.storage_mode = storage_mode
        self.store = DataStore(mode=storage_mode)
        self.downloader = HistoricalDownloader(self.store)
        self.streamer: Optional[WebSocketStreamer] = None
        self.reconciler: Optional[ReconciliationEngine] = None

        self._running = False
        self._callbacks: list[OnBarCloseCallback] = []

    # ----- 历史数据 -----
    def download_history(
        self,
        timeframe: str,
        since: Optional[str] = None,
        until: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """下载历史 K 线到本地"""
        return self.downloader.download(timeframe, since=since, until=until, limit=limit)

    def download_all_history(
        self,
        since: Optional[str] = None,
        until: Optional[str] = None,
    ) -> dict[str, pd.DataFrame]:
        """下载所有 timeframe 的历史 K 线"""
        results = {}
        for tf in TIMEFRAMES:
            results[tf] = self.downloader.download(tf, since=since, until=until)
        return results

    def download_incremental(self) -> dict[str, pd.DataFrame]:
        """增量更新"""
        return self.downloader.download_all_incremental()

    def load_klines(
        self,
        timeframe: str,
        since: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """从本地存储加载 K 线"""
        return self.store.load_klines(timeframe, since=since, limit=limit)

    # ----- 回调注册 -----
    def add_bar_close_callback(self, cb: OnBarCloseCallback) -> None:
        """注册 K 线收盘回调（供策略模块接入）"""
        self._callbacks.append(cb)
        if self.streamer:
            self.streamer.add_bar_close_callback(cb)

    # ----- 实盘模式 -----
    async def run_live(self, start_ws: bool = True, start_reconciler: bool = True) -> None:
        """启动实盘数据流：
        1. 加载本地历史 K 线矩阵
        2. 启动 WebSocket 订阅
        3. 启动 REST 对账引擎
        """
        logger.info("=" * 50)
        logger.info("DataModule — Live mode starting")
        logger.info("=" * 50)

        # 加载本地历史数据作为在线矩阵的初始值
        df_15m = self.store.load_klines("15m")
        df_2h = self.store.load_klines("2h")
        logger.info(f"Loaded historical data: 15m={len(df_15m)} rows, 2h={len(df_2h)} rows")

        # 创建 WebSocket Streamer
        self.streamer = WebSocketStreamer(df_15m=df_15m, df_2h=df_2h)

        # 注册已积累的回调
        for cb in self._callbacks:
            self.streamer.add_bar_close_callback(cb)

        # 创建对账引擎
        self.reconciler = ReconciliationEngine(self.store, self.streamer)

        tasks = []
        if start_ws:
            tasks.append(asyncio.create_task(self.streamer.start()))
        if start_reconciler:
            tasks.append(asyncio.create_task(self.reconciler.start()))

        self._running = True

        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            logger.info("DataModule live mode cancelled")
        finally:
            await self.shutdown()

    async def shutdown(self) -> None:
        """优雅关闭所有数据组件"""
        logger.info("DataModule shutting down...")
        self._running = False

        if self.reconciler:
            await self.reconciler.stop()
        if self.streamer:
            await self.streamer.stop()
        self.downloader.close()
        self.store.close()

        logger.info("DataModule shutdown complete")

    # ----- 便捷获取在线数据 -----
    def get_online_df(self, timeframe: str) -> pd.DataFrame:
        """获取实盘在线 K 线矩阵"""
        if not self.streamer:
            raise RuntimeError("WebSocket streamer not started")
        return self.streamer.get_df(timeframe)
