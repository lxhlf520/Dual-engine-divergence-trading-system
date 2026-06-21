"""
REST 补录 / 对账引擎 （双轨制）
每 5 分钟通过 REST API 向交易所拉取最新 K 线，
与 WebSocket 维护的数据进行比对，自动补齐缺失数据
"""
import asyncio
import time
from typing import Optional

import ccxt
import pandas as pd

from ..config import EXCHANGE_ID, SYMBOL, TIMEFRAMES, RECONCILE_INTERVAL, REST_RATE_LIMIT_DELAY, API_KEY, API_SECRET, API_PASSWORD
from ..logger import get_logger
from .store import DataStore, KLINE_COLUMNS

logger = get_logger(__name__)


class ReconciliationEngine:
    """双轨制对账引擎。

    使用方式（实盘模式）:
        reconciler = ReconciliationEngine(data_store, streamer)
        reconciler.start()       # 在后台线程运行
        # ... 系统运行中 ...
        reconciler.stop()        # 优雅关闭
    """

    def __init__(self, store: DataStore, streamer=None):
        """
        Args:
            store: DataStore 实例，用于持久化比对
            streamer: WebSocketStreamer 实例（可选）；传入后可直接对比在线矩阵
        """
        self.store = store
        self.streamer = streamer
        self._running = False
        self._tasks: list[asyncio.Task] = []
        self._exchange: Optional[ccxt.Exchange] = None

    def _get_exchange(self) -> ccxt.Exchange:
        if self._exchange is None:
            exchange_class = getattr(ccxt, EXCHANGE_ID)
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
        return self._exchange

    # ----- 核心对账逻辑 -----
    async def reconcile_timeframe(self, timeframe: str) -> dict:
        """
        对单个 timeframe 执行 REST 拉取与本地比对，返回补齐统计

        Returns:
            {
                "timeframe": "15m",
                "fetched": 3,        # REST 拉取的条数
                "missing": 1,        # 本地缺失条数
                "inserted": 1,       # 实际补齐条数
            }
        """
        result = {"timeframe": timeframe, "fetched": 0, "missing": 0, "inserted": 0}

        try:
            exchange = self._get_exchange()

            # 1. 获取本地最新时间戳
            local_latest = self.store.get_latest_timestamp(timeframe)

            # 2. 从 REST 拉取最近若干 K 线（覆盖 WebSocket 可能丢失的窗口）
            #    取最近 10 根以确保覆盖
            since_ms = None
            if local_latest:
                # 往回拉一点，确保重叠区域可以比对
                since_ms = local_latest - 10 * self._timeframe_to_ms(timeframe)

            raw = exchange.fetch_ohlcv(
                SYMBOL,
                timeframe=timeframe,
                since=since_ms,
                limit=20,
            )
            time.sleep(REST_RATE_LIMIT_DELAY)

            if not raw:
                logger.warning(f"REST reconciliation [{timeframe}] returned empty")
                return result

            df_rest = pd.DataFrame(raw, columns=KLINE_COLUMNS)
            df_rest["timestamp"] = df_rest["timestamp"].astype(int)
            result["fetched"] = len(df_rest)

            # 3. 加载本地数据，找出缺失
            local_df = self.store.load_klines(timeframe)
            if not local_df.empty:
                local_ts = set(local_df["timestamp"].astype(int))
                rest_ts = set(df_rest["timestamp"])
                missing_ts = rest_ts - local_ts
            else:
                missing_ts = set(df_rest["timestamp"])

            result["missing"] = len(missing_ts)

            # 4. 补齐缺失数据
            if missing_ts:
                missing_df = df_rest[df_rest["timestamp"].isin(missing_ts)]
                inserted = self.store.insert_klines(missing_df, timeframe)
                result["inserted"] = inserted
                logger.info(
                    f"Reconciliation [{timeframe}]: "
                    f"fetched={result['fetched']}, missing={result['missing']}, "
                    f"inserted={result['inserted']}"
                )

                # 5. 如果传入了 streamer，同步更新在线矩阵
                if self.streamer:
                    self._sync_to_streamer(timeframe, missing_df)

            # 6. 数据完整性校验：REST 与本地时间连续性
            self._check_continuity(timeframe, local_df)

        except ccxt.NetworkError as e:
            logger.error(f"Reconciliation [{timeframe}] network error: {e}")
        except ccxt.BaseError as e:
            logger.error(f"Reconciliation [{timeframe}] CCXT error: {e}")
        except Exception as e:
            logger.error(f"Reconciliation [{timeframe}] unexpected error: {e}", exc_info=True)

        return result

    def _sync_to_streamer(self, timeframe: str, missing_df: pd.DataFrame) -> None:
        """将补齐的数据同步到 WebSocket 在线矩阵"""
        try:
            df_attr = "df_15m" if timeframe == "15m" else "df_2h"
            current = getattr(self.streamer, df_attr, None)
            if current is None or current.empty:
                return

            combined = pd.concat([current, missing_df], ignore_index=True)
            combined.drop_duplicates(subset=["timestamp"], keep="last", inplace=True)
            combined.sort_values("timestamp", inplace=True)
            combined.reset_index(drop=True, inplace=True)
            setattr(self.streamer, df_attr, combined)
            logger.debug(f"Streamer df [{timeframe}] synced with {len(missing_df)} rows from REST")
        except Exception as e:
            logger.error(f"Sync to streamer [{timeframe}] failed: {e}")

    def _check_continuity(self, timeframe: str, local_df: pd.DataFrame) -> None:
        """检查本地 K 线时间戳连续性，记录缺口"""
        if local_df.empty or len(local_df) < 2:
            return

        ts = local_df["timestamp"].astype(int).sort_values().values
        expected_interval = self._timeframe_to_ms(timeframe)
        gaps = []
        for i in range(1, len(ts)):
            diff = ts[i] - ts[i - 1]
            if diff > expected_interval * 1.5:  # 1.5 倍容差
                gaps.append((ts[i - 1], ts[i], diff))

        if gaps:
            logger.warning(
                f"Data gaps detected [{timeframe}]: {len(gaps)} gaps found"
            )
            for start, end, diff in gaps[:3]:  # 只记录前3个
                logger.debug(
                    f"  Gap: {start} -> {end} (diff={diff}ms, "
                    f"missing ~{diff // expected_interval - 1} bars)"
                )

    @staticmethod
    def _timeframe_to_ms(timeframe: str) -> int:
        """将 timeframe 字符串转为毫秒"""
        tf_map = {
            "1m": 60_000, "5m": 300_000, "15m": 900_000,
            "30m": 1_800_000, "1h": 3_600_000, "2h": 7_200_000,
            "4h": 14_400_000, "1d": 86_400_000,
        }
        return tf_map.get(timeframe, 60_000)

    # ----- 主循环 -----
    async def _reconcile_loop(self) -> None:
        """后台对账主循环，按配置间隔执行"""
        logger.info(f"Reconciliation loop started (interval={RECONCILE_INTERVAL}s)")

        while self._running:
            try:
                for tf in TIMEFRAMES:
                    await self.reconcile_timeframe(tf)
                    await asyncio.sleep(1)  # timeframe 间小间隔

                await asyncio.sleep(RECONCILE_INTERVAL)

            except asyncio.CancelledError:
                logger.info("Reconciliation loop cancelled")
                break
            except Exception as e:
                logger.error(f"Reconciliation loop error: {e}", exc_info=True)
                await asyncio.sleep(30)  # 出错后等待 30s 再试

    async def start(self) -> None:
        """启动对账引擎"""
        self._running = True
        self._tasks.append(asyncio.create_task(self._reconcile_loop()))
        logger.info("ReconciliationEngine started")

    async def stop(self) -> None:
        """停止对账引擎"""
        self._running = False
        for t in self._tasks:
            if not t.done():
                t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        logger.info("ReconciliationEngine stopped")

    def start_sync(self) -> None:
        """同步启动（在已有事件循环中调用）"""
        loop = asyncio.get_event_loop()
        loop.create_task(self.start())
