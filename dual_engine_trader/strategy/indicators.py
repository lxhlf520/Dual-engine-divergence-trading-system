"""
Technical indicators: RSI, ATR, pivot detection, valuewhen, barssince.
Full Pine Script v4 compatibility.
"""
from typing import Tuple
import numpy as np
import pandas as pd
from ..config import RSI_PERIOD, ATR_PERIOD, LB_L, LB_R
from ..logger import get_logger

logger = get_logger(__name__)


def compute_rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """Wilder's RSI (Pine Script compatible)."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    alpha = 1.0 / period
    gv = gain.values
    lv = loss.values
    ag = avg_gain.values.copy()
    al = avg_loss.values.copy()
    for i in range(period + 1, len(close)):
        ag[i] = alpha * gv[i] + (1 - alpha) * ag[i - 1]
        al[i] = alpha * lv[i] + (1 - alpha) * al[i - 1]
    ag_s = pd.Series(ag, index=close.index)
    al_s = pd.Series(al, index=close.index)
    safe_al = al_s.copy()
    safe_ag = ag_s.copy()
    safe_al = safe_al.mask((safe_al == 0) & (safe_ag > 0), 1e-10)
    safe_ag = safe_ag.mask((safe_ag == 0) & (safe_al > 0), 1e-10)
    rs = safe_ag / safe_al
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.mask((ag_s > 0) & (al_s == 0), 100.0)
    rsi = rsi.mask((ag_s == 0) & (al_s > 0), 0.0)
    return rsi.clip(0, 100)


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = ATR_PERIOD) -> pd.Series:
    """Wilder's ATR (Pine Script compatible)."""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=period, min_periods=period).mean()
    trv = true_range.values
    av = atr.values.copy()
    alpha = 1.0 / period
    for i in range(period + 1, len(close)):
        av[i] = alpha * trv[i] + (1 - alpha) * av[i - 1]
    return pd.Series(av, index=close.index)


def pivotlow(series: pd.Series, lbL: int = LB_L, lbR: int = LB_R) -> pd.Series:
    """Detect pivot lows. Returns boolean Series."""
    n = len(series)
    result = pd.Series(False, index=series.index)
    window = lbL + lbR + 1
    if n < window:
        return result
    vals = series.values
    for i in range(lbR, n - lbR):
        left = i - lbL
        right = i + lbR
        if left < 0 or right >= n:
            continue
        w = vals[left:right + 1]
        if vals[i] == np.min(w):
            result.iloc[i] = True
    return result


def pivothigh(series: pd.Series, lbL: int = LB_L, lbR: int = LB_R) -> pd.Series:
    """Detect pivot highs. Returns boolean Series."""
    n = len(series)
    result = pd.Series(False, index=series.index)
    window = lbL + lbR + 1
    if n < window:
        return result
    vals = series.values
    for i in range(lbR, n - lbR):
        left = i - lbL
        right = i + lbR
        if left < 0 or right >= n:
            continue
        w = vals[left:right + 1]
        if vals[i] == np.max(w):
            result.iloc[i] = True
    return result


def valuewhen(condition: pd.Series, source: pd.Series, occurrence: int = 1) -> pd.Series:
    """Replicate Pine Script valuewhen(cond, src, occ)."""
    result = pd.Series(np.nan, index=condition.index)
    true_indices = condition[condition].index.tolist()
    if len(true_indices) < occurrence:
        return result
    last_val = np.nan
    for idx in condition.index:
        past = [i for i in true_indices if i <= idx]
        if len(past) >= occurrence:
            last_val = source.loc[past[-occurrence]]
        result.loc[idx] = last_val
    return result


def barssince(condition: pd.Series) -> pd.Series:
    """Replicate Pine Script barssince(cond)."""
    result = pd.Series(np.nan, index=condition.index)
    last_true = None
    for idx in condition.index:
        if condition.loc[idx]:
            last_true = idx
            result.loc[idx] = 0
        elif last_true is not None:
            result.loc[idx] = condition.index.get_loc(idx) - condition.index.get_loc(last_true)
    return result
