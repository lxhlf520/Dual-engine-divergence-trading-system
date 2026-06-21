"""
通知/报警模块 —— Telegram + 钉钉
"""
import requests
from ..config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, DINGTALK_WEBHOOK
from ..logger import get_logger

logger = get_logger(__name__)


class AlertManager:
    """统一报警推送管理"""

    def __init__(self):
        self.tg_token = TELEGRAM_BOT_TOKEN
        self.tg_chat_id = TELEGRAM_CHAT_ID
        self.dingtalk_url = DINGTALK_WEBHOOK

    def send(self, title: str, body: str, level: str = "ERROR") -> None:
        """同时向 Telegram 和钉钉推送报警"""
        msg = f"[{level}] {title}\n{body}"

        if self.tg_token and self.tg_chat_id:
            try:
                url = f"https://api.telegram.org/bot{self.tg_token}/sendMessage"
                payload = {"chat_id": self.tg_chat_id, "text": msg, "parse_mode": "Markdown"}
                r = requests.post(url, json=payload, timeout=10)
                if r.status_code != 200:
                    logger.warning(f"Telegram send failed: {r.text}")
            except Exception as e:
                logger.error(f"Telegram exception: {e}")

        if self.dingtalk_url:
            try:
                payload = {"msgtype": "text", "text": {"content": msg}}
                r = requests.post(self.dingtalk_url, json=payload, timeout=10)
                if r.status_code != 200:
                    logger.warning(f"DingTalk send failed: {r.text}")
            except Exception as e:
                logger.error(f"DingTalk exception: {e}")

    def trade_alert(self, action: str, symbol: str, price: float, details: str = "") -> None:
        """交易操作报警"""
        self.send(f"Trade: {action} {symbol} @ {price:.2f}", details, "INFO")

    def error_alert(self, context: str, error: str) -> None:
        """严重错误报警"""
        self.send(f"ERROR: {context}", error, "ERROR")

    def position_alert(self, side: str, pnl: float, details: str = "") -> None:
        """仓位变动报警"""
        self.send(f"Position Closed: {side} PnL=${pnl:.2f}", details, "INFO")
