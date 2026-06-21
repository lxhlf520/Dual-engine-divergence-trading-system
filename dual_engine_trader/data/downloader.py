"""
CCXT 历史数据下载器
负责从交易所 REST API 稳定下载指定时间段内的 BTC K 线
特性：
- 自动分页获取大批量数据
- 速率限制保护
- 断点续传（支持增量下载）
"""
import time
from datetime import datetime, timezone
from typing import Optional

import ccxt
import pandas as pd

from ..config import EXCHANGE_ID, SYMBOL, TIMEFRAMES, REST_RATE_LIMIT_DELAY, API_KEY, API_SECRET, API_PASSWORD
from ..logger import get_logger
from .store import DataStore, KLINE_COLUMNS

logger = get_logger(__name__)

# CCXT 单次请求最大 K 线条数（不同交易所不同，Binance 为 1500）
MAX_CANDLES_PER_REQUEST = 1500


class HistoricalDownloader:
    """历史 K 线下载器。

    使用方式：
        downloader = HistoricalDownloader(data_store)
        downloader.download("15m", since="2024-01-01", until="2024-06-01")
        downloader.download_all_incremental()  # 增量更新所有 timeframe
    """

    def __init__(self, store: DataStore):
        self.store = store
        self._exchange: Optional[ccxt.Exchange] = None

    def _get_exchange(self) -> ccxt.Exchange:
        """懒加载交易所实例"""
        if self._exchange is None:
            exchange_class = getattr(ccxt, EXCHANGE_ID)
            config = {
                "enableRateLimit": True,
                "options": {"defaultType": "swap"},  # 永续合约
            }
            if API_KEY:
                config["apiKey"] = API_KEY
            if API_SECRET:
                config["secret"] = API_SECRET
            if API_PASSWORD:
                config["password"] = API_PASSWORD
            self._exchange = exchange_class(config)
            logger.info(f"CCXT exchange [{EXCHANGE_ID}] initialized")
        return self._exchange

    def download(
        self,
        timeframe: str,
        since: Optional[str] = None,
        until: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """下载历史 K 线并存入 DataStore。

        Args:
            timeframe: "15m" 或 "2h"
            since: 起始时间字符串 "YYYY-MM-DD" 或 "YYYY-MM-DD HH:MM:SS"
            until: 结束时间字符串，默认为当前时间
            limit: 最大下载条数限制

        Returns:
            下载的 K 线 DataFrame
        """
        exchange = self._get_exchange()

        # 解析时间
        since_ts = self._parse_time(since) if since else None
        until_ts = self._parse_time(until) if until else int(datetime.now(timezone.utc).timestamp() * 1000)

        # 如果传了 limit，限制条数
        max_rows = limit if limit else float("inf")

        all_candles: list = []
        current_since = since_ts

        logger.info(
            f"Downloading [{timeframe}] | "
            f"since={since_ts} ({self._ts_to_str(since_ts) if since_ts else 'N/A'}) | "
            f"until={self._ts_to_str(until_ts)}"
        )

        while len(all_candles) < max_rows:
            try:
                # CCXT fetch_ohlcv 参数：symbol, timeframe, since, limit
                params = {}
                if current_since:
                    params["since"] = current_since

                candles = exchange.fetch_ohlcv(
                    SYMBOL,
                    timeframe=timeframe,
                    since=current_since,
                    limit=MAX_CANDLES_PER_REQUEST,
                )

                if not candles:
                    logger.debug(f"No more candles returned at since={current_since}")
                    break

                # 过滤掉超出 until 的数据
                if until_ts:
                    candles = [c for c in candles if c[0] <= until_ts]

                all_candles.extend(candles)

                # 更新分页游标：最后一条的时间戳 + 1ms
                last_ts = candles[-1][0]
                if last_ts >= until_ts:
                    break
                if current_since and current_since == last_ts:
                    # 死循环保护
                    logger.warning("Pagination cursor stuck — breaking loop")
                    break
                current_since = last_ts + 1

                logger.debug(
                    f"Page fetched: {len(candles)} candles, "
                    f"total={len(all_candles)}, last_ts={last_ts}"
                )

                # 速率限制保护
                time.sleep(REST_RATE_LIMIT_DELAY)

            except ccxt.RateLimitExceeded as e:
                wait = 30
                logger.warning(f"Rate limit hit — waiting {wait}s: {e}")
                time.sleep(wait)
            except ccxt.NetworkError as e:
                wait = 10
                logger.warning(f"Network error — retry in {wait}s: {e}")
                time.sleep(wait)
            except ccxt.BaseError as e:
                logger.error(f"CCXT error during download: {e}")
                break

        if not all_candles:
            logger.warning(f"No data downloaded for [{timeframe}]")
            return pd.DataFrame(columns=KLINE_COLUMNS)

        # 转为 DataFrame 并去重排序
        df = pd.DataFrame(all_candles, columns=KLINE_COLUMNS)
        df["timestamp"] = df["timestamp"].astype(int)
        df.drop_duplicates(subset=["timestamp"], keep="last", inplace=True)
        df.sort_values("timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)

        # 存入 DataStore
        count = self.store.insert_klines(df, timeframe)
        logger.info(f"Download complete [{timeframe}]: {count} rows saved")

        return df

    def download_all_incremental(self) -> dict[str, pd.DataFrame]:
        """增量更新：从本地最新时间戳下载至当前时刻，所有 timeframe"""
        results = {}
        for tf in TIMEFRAMES:
            latest = self.store.get_latest_timestamp(tf)
            if latest:
                since_ms = latest + 1  # 比最新多 1ms，避免重复拉取边界
                logger.info(f"[{tf}] Incremental update from {self._ts_to_str(since_ms)}")
            else:
                since_ms = None  # 无本地数据，全量拉取
            df = self.download(tf, since=None if since_ms is None else None)
            # 如果用 since_ms 分页，手动设置
            if since_ms and latest:
                results[tf] = self.download(tf, since=self._ts_to_str(since_ms))
            else:
                results[tf] = df
        return results

    @staticmethod
    def _parse_time(s: str) -> int:
        """将时间字符串转为毫秒级 Unix 时间戳"""
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                dt = datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
                return int(dt.timestamp() * 1000)
            except ValueError:
                continue
        raise ValueError(f"Unparseable time string: {s}")

    @staticmethod
    def _ts_to_str(ts: int) -> str:
        """毫秒时间戳 -> 可读字符串"""
        return datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    def close(self):
        if self._exchange and hasattr(self._exchange, "close"):
            self._exchange.close()
