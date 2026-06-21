"""Data storage backend - SQLite + CSV dual mode."""
import csv
import sqlite3
from pathlib import Path
from typing import Optional
import pandas as pd
from ..config import DB_PATH, CSV_DIR, TIMEFRAMES
from ..logger import get_logger

logger = get_logger(__name__)
KLINE_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]

class DataStore:
    def __init__(self, mode: str = "sqlite"):
        self.mode = mode
        self._conn: Optional[sqlite3.Connection] = None
        if mode == "sqlite":
            self._init_sqlite()

    def _init_sqlite(self) -> None:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        for tf in TIMEFRAMES:
            tname = self._get_table_name(tf)
            self._conn.execute(f"""CREATE TABLE IF NOT EXISTS {tname} (
                timestamp INTEGER PRIMARY KEY, open REAL NOT NULL,
                high REAL NOT NULL, low REAL NOT NULL, close REAL NOT NULL, volume REAL NOT NULL)""")
            self._conn.execute(f"CREATE UNIQUE INDEX IF NOT EXISTS idx_ts_{tf} ON {tname} (timestamp)")
        self._conn.commit()
        logger.info(f"SQLite initialized at {DB_PATH}")

    def _get_table_name(self, timeframe: str) -> str:
        return f"klines_{timeframe.replace('m', 'min').replace('h', 'hour')}"

    def _get_csv_path(self, timeframe: str) -> Path:
        return CSV_DIR / f"BTC_USDT_{timeframe}.csv"

    def insert_klines(self, df: pd.DataFrame, timeframe: str) -> int:
        if df.empty:
            return 0
        df = df[KLINE_COLUMNS].copy()
        df["timestamp"] = df["timestamp"].astype(int)
        if self.mode == "sqlite":
            return self._insert_sqlite(df, timeframe)
        else:
            return self._insert_csv(df, timeframe)

    def _insert_sqlite(self, df: pd.DataFrame, timeframe: str) -> int:
        table = self._get_table_name(timeframe)
        rows = [tuple(r) for r in df[KLINE_COLUMNS].values]
        cursor = self._conn.cursor()
        count = 0
        for row in rows:
            try:
                cursor.execute(f"INSERT OR REPLACE INTO {table} (timestamp, open, high, low, close, volume) VALUES (?, ?, ?, ?, ?, ?)", row)
                count += 1
            except sqlite3.Error as e:
                logger.error(f"SQLite insert error at ts={row[0]}: {e}")
        self._conn.commit()
        return count

    def _insert_csv(self, df: pd.DataFrame, timeframe: str) -> int:
        path = self._get_csv_path(timeframe)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            existing = pd.read_csv(path)
            existing["timestamp"] = existing["timestamp"].astype(int)
            combined = pd.concat([existing, df], ignore_index=True)
            combined.drop_duplicates(subset=["timestamp"], keep="last", inplace=True)
            combined.sort_values("timestamp", inplace=True)
            combined.to_csv(path, index=False)
        else:
            df.to_csv(path, index=False)
        return len(df)

    def load_klines(self, timeframe: str, since: Optional[int] = None, limit: Optional[int] = None) -> pd.DataFrame:
        if self.mode == "sqlite":
            return self._load_sqlite(timeframe, since, limit)
        else:
            return self._load_csv(timeframe, since, limit)

    def _load_sqlite(self, timeframe: str, since=None, limit=None) -> pd.DataFrame:
        table = self._get_table_name(timeframe)
        query = f"SELECT * FROM {table}"
        params = []
        if since is not None:
            query += " WHERE timestamp >= ?"
            params.append(since)
        query += " ORDER BY timestamp ASC"
        if limit is not None:
            query += f" LIMIT {int(limit)}"
        df = pd.read_sql_query(query, self._conn, params=params)
        if df.empty:
            return pd.DataFrame(columns=KLINE_COLUMNS)
        return df

    def _load_csv(self, timeframe: str, since=None, limit=None) -> pd.DataFrame:
        path = self._get_csv_path(timeframe)
        if not path.exists():
            return pd.DataFrame(columns=KLINE_COLUMNS)
        df = pd.read_csv(path)
        if since is not None:
            df = df[df["timestamp"] >= since]
        if limit is not None:
            df = df.tail(limit)
        return df.reset_index(drop=True)

    def get_latest_timestamp(self, timeframe: str) -> Optional[int]:
        if self.mode == "sqlite":
            table = self._get_table_name(timeframe)
            cursor = self._conn.execute(f"SELECT MAX(timestamp) FROM {table}")
            row = cursor.fetchone()
            return row[0] if row[0] else None
        else:
            path = self._get_csv_path(timeframe)
            if not path.exists():
                return None
            df = pd.read_csv(path)
            return int(df["timestamp"].max()) if not df.empty else None

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            logger.info("SQLite connection closed")
