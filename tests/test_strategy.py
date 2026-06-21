"""
Strategy module unit tests.
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent

import numpy as np
import pandas as pd

from dual_engine_trader.strategy.indicators import (
    compute_rsi, compute_atr, pivotlow, pivothigh, valuewhen, barssince,
)
from dual_engine_trader.strategy.detector import (
    DivergenceDetector, DivergenceParams, SignalType,
)
from dual_engine_trader.strategy.engine import (
    MultiEngineStrategy, LONG_PARAMS, SHORT_PARAMS,
)

def create_synthetic_klines(n=200, seed=42):
    rng = np.random.default_rng(seed)
    ts = np.arange(n) * 60000
    ch = rng.normal(0, 200, n).cumsum()
    close = 50000 + ch
    high = close + np.abs(rng.normal(0, 150, n))
    low = close - np.abs(rng.normal(0, 150, n))
    o = close - rng.normal(0, 100, n)
    v = rng.lognormal(3, 1, n)
    return pd.DataFrame({"timestamp": ts, "open": o, "high": high, "low": low, "close": close, "volume": v})

def test_rsi_calculation():
    prices = pd.Series([40000 + i * 10 for i in range(100)])
    rsi = compute_rsi(prices, period=14)
    valid = rsi.dropna()
    assert len(valid) > 0, "no RSI values after warm-up"
    assert valid.max() <= 100
    assert valid.min() >= 0
    assert valid.iloc[-1] > 80, f"RSI too low: {valid.iloc[-1]:.1f}"
    print(f"  RSI: PASS (last={valid.iloc[-1]:.1f})")

def test_atr_calculation():
    n = 100
    h = pd.Series(np.random.default_rng(0).normal(50000, 500, n))
    l = h - pd.Series(np.random.default_rng(1).exponential(300, n))
    c = (h + l) / 2
    atr = compute_atr(h, l, c, period=14)
    valid = atr.dropna()
    assert len(valid) > 0
    assert valid.min() > 0
    print(f"  ATR: PASS (mean={valid.mean():.2f})")

def test_pivot_detection():
    vals = np.array([5, 4, 3, 4, 5, 6, 5, 4, 3, 4, 5, 4, 3, 4, 5])
    s = pd.Series(vals)
    pl = pivotlow(s, lbL=1, lbR=2)
    ph = pivothigh(s, lbL=1, lbR=2)
    assert pl.iloc[2], "expected pivotlow at index 2"
    assert ph.iloc[5], "expected pivothigh at index 5"
    print(f"  Pivot: PASS (lows={pl.sum()} highs={ph.sum()})")

def test_valuewhen():
    idx = pd.RangeIndex(0, 10)
    cond = pd.Series([False, False, True, False, True, False, False, False, False, False], index=idx)
    src = pd.Series([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], index=idx)
    vw = valuewhen(cond, src, 1)
    assert vw.iloc[2] == 2
    assert vw.iloc[3] == 2
    assert vw.iloc[4] == 4
    assert vw.iloc[9] == 4
    bs = barssince(cond)
    assert bs.iloc[2] == 0
    assert bs.iloc[4] == 0
    assert bs.iloc[5] == 1
    print("  valuewhen/barssince: PASS")

def test_divergence_signals():
    df = create_synthetic_klines(n=500, seed=42)
    sd = DivergenceDetector(SHORT_PARAMS)
    sigs = sd.detect(df, "15m")
    print(f"  Divergence 15M: {len(sigs)} signals")
    ld = DivergenceDetector(LONG_PARAMS)
    sigs2 = ld.detect(df, "2h")
    print(f"  Divergence 2H: {len(sigs2)} signals")
    print("  Divergence detection: PASS")

def test_multiframe_separation():
    strategy = MultiEngineStrategy()
    df = create_synthetic_klines(n=500, seed=99)
    for s in strategy.process_bar(df, "2h"):
        assert s.signal_type in (SignalType.BUY, SignalType.CLOSE_LONG)
    for s in strategy.process_bar(df, "15m"):
        assert s.signal_type in (SignalType.SELL, SignalType.CLOSE_SHORT)
    print("  Multi-timeframe separation: PASS")

def test_trailing_stop():
    from dual_engine_trader.strategy.detector import TrailingStopUpdater
    u = TrailingStopUpdater(stop_loss_mult=3.0)
    sl = u.update_long_sl(None, low=100, atr=5)
    assert sl == 85.0
    sl = u.update_long_sl(sl, low=105, atr=5)
    assert sl == 90.0
    sl = u.update_long_sl(sl, low=95, atr=5)
    assert sl == 90.0
    u2 = TrailingStopUpdater(stop_loss_mult=2.8)
    sl = u2.update_short_sl(None, high=100, atr=5)
    assert sl == 114.0
    sl = u2.update_short_sl(sl, high=95, atr=5)
    assert sl == 109.0
    sl = u2.update_short_sl(sl, high=100, atr=5)
    assert sl == 109.0
    print("  Trailing stop: PASS")

def test_stop_trigger():
    from dual_engine_trader.strategy.detector import TrailingStopUpdater
    u = TrailingStopUpdater()
    assert u.check_long_stop(close=84, trailing_sl=85)
    assert not u.check_long_stop(close=86, trailing_sl=85)
    assert u.check_short_stop(close=116, trailing_sl=115)
    assert not u.check_short_stop(close=114, trailing_sl=115)
    print("  Stop trigger: PASS")

if __name__ == "__main__":
    print("=" * 60)
    print("Strategy Module Unit Tests")
    print("=" * 60)
    test_rsi_calculation()
    test_atr_calculation()
    test_pivot_detection()
    test_valuewhen()
    test_trailing_stop()
    test_stop_trigger()
    test_divergence_signals()
    test_multiframe_separation()
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
