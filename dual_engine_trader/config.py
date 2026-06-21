import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    env_path = Path(__file__).resolve().parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass

ROOT_DIR = Path(__file__).resolve().parent.parent
EXCHANGE_ID = "okx"
SYMBOL = "BTC/USDT:USDT"
TIMEFRAMES = ["15m", "2h"]
API_KEY = os.getenv("OKX_API_KEY", "")
API_SECRET = os.getenv("OKX_API_SECRET", "")
API_PASSWORD = os.getenv("OKX_API_PASSWORD", "")
HTTP_PROXY = os.getenv("HTTP_PROXY", "")
HTTPS_PROXY = os.getenv("HTTPS_PROXY", "")
DATA_DIR = ROOT_DIR / "historical_data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "btc_klines.db"
CSV_DIR = DATA_DIR / "csv"
CSV_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR = ROOT_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "trading_system.log"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
WS_RECONNECT_DELAY = 5
WS_MAX_RETRIES = 10
RECONCILE_INTERVAL = 300
REST_RATE_LIMIT_DELAY = 1.0
RSI_PERIOD = 14
ATR_PERIOD = 14
LB_L = 1
LB_R = 2
DEFAULT_LEVERAGE = 2
MARGIN_MODE = "isolated"
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
DINGTALK_WEBHOOK = os.getenv("DINGTALK_WEBHOOK", "")
BACKTEST_INITIAL_CAPITAL = 10_000.0
BACKTEST_FEE_RATE = 0.0005
BACKTEST_SLIPPAGE_RATE = 0.0001
BACKTEST_PYRAMIDING = 2
BACKTEST_POSITION_SIZE = 2.0
