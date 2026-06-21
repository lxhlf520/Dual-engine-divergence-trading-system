"""
日志系统初始化模块
所有行为（行情接收、策略计算、下单细节、报错信息）必须实时分类写入 .log 文件
"""
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from .config import LOG_FILE, LOG_LEVEL, LOG_FORMAT, LOG_DIR


def setup_logger(name: str = "trading_system") -> logging.Logger:
    """创建并配置 logger 实例，同时输出到文件和控制台。

    日志文件采用 RotatingFileHandler，单个文件最大 20 MB，保留 10 个历史文件。
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))
    logger.propagate = False

    # 避免重复添加 handler（多次调用 setup_logger 时）
    if logger.handlers:
        return logger

    formatter = logging.Formatter(LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")

    # ---- 文件 Handler（轮转日志）----
    fh = RotatingFileHandler(
        str(LOG_FILE),
        maxBytes=20 * 1024 * 1024,  # 20 MB
        backupCount=10,
        encoding="utf-8",
    )
    fh.setFormatter(formatter)
    fh.setLevel(logging.DEBUG)  # 文件记录 DEBUG 及以上
    logger.addHandler(fh)

    # ---- 控制台 Handler ----
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    ch.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))
    logger.addHandler(ch)

    return logger


def get_logger(name: str) -> logging.Logger:
    """获取模块级 logger 快捷方法"""
    return logging.getLogger(f"trading_system.{name}")
