"""
Fixed parameter grid search for 15m short-only divergence strategy.

BUG FIXES:
1. Stop loss check was running on the SAME bar as entry - bar_high naturally exceeds
   the trailing_sl set on that bar, causing instant stop-out. Fixed by deferring first
   stop check to the bar AFTER entry.
2. current_atr_val was hardcoded to 0 in the grid search, making stop updates a no-op.
   Fixed by pre-computing ATR and passing actual values.
3. run_single uses the proper bar loop: first update stops on open positions, then check
   stops using LOW/HIGH of current bar, but skip positions opened THIS bar.

Also uses compute_atr once and stores it.
"""
import sys, warnings, os, time
warnings.filterwarnings('ignore')


import pandas as pd
import numpy as np
from itertools import product
from dataclasses import dataclass
from typing import Optional

from dual_engine_trader.strategy.detector import (
    DivergenceDetector, DivergenceParams, TrailingStopUpdater,
    Signal, SignalType,
)
from dual_engine_trader.strategy.engine import Direction
from dual_engine_trader.strategy.indicators import compute_atr
from dual_engine_trader.backtest.account import VirtualAccount


@dataclass
class GridResult:
    stop_loss_mult: float
    range_lower: int
    range_upper: int
    rsi_period: int
    atr_period: int
    take_profit_rsi: int
    total_trades: int
    wins: int
    net_pnl: float
    net_pnl_pct: float
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    max_drawdown: float
    avg_bars: float
    total_costs: float


def run_single(params_dict: dict, df: pd.DataFrame) -> Optional[GridResult]:
    p = DivergenceParams(
        rsi_period=params_dict['rsi_period'],
        atr_period=params_dict['atr_period'],
        lb_l=1, lb_r=2,
        take_profit_rsi=params_dict.get('take_profit_rsi', 25),
        stop_loss_mult=params_dict['stop_loss_mult'],
        range_lower=params_dict['range_lower'],
        range_upper=params_dict['range_upper'],
        plot_bear=True, plot_hidden_bear=True, plot_bull=False,
    )

    detector = DivergenceDetector(p)
    sl_updater = TrailingStopUpdater(stop_loss_mult=p.stop_loss_mult)
    account = VirtualAccount(initial_capital=10000, fee_rate=0.0005,
                             slippage_rate=0.0001, max_pyramiding=2)

    all_signals = detector.detect(df, "15m")
    sigs_by_bar = {}
    for sig in all_signals:
        bar_idx = sig.metadata.get("pivot_b")
        if bar_idx is not None:
            sigs_by_bar.setdefault(bar_idx, []).append(sig)

    n = len(df)
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)

    # Pre-compute ATR once
    atr_series = compute_atr(high, low, close, period=p.atr_period)

    warmup = 60
    # Track entry bar index for each position to skip same-bar stop check
    position_entry_bars: list = []  # [bar_index, ...]

    for i in range(warmup + 20, n):
        bar_ts = int(df["timestamp"].iloc[i])
        bar_high = float(high.iloc[i])
        bar_low = float(low.iloc[i])
        bar_close = float(close.iloc[i])
        current_atr = float(atr_series.iloc[i]) if not pd.isna(atr_series.iloc[i]) else 0.0

        if i % 10 == 0:
            account.record_equity(bar_ts)

        # --- Update trailing stops for existing positions ---
        new_entry_bars = []
        for pi in range(len(account.short_positions)):
            pos = account.short_positions[pi]
            entry_bar = position_entry_bars[pi] if pi < len(position_entry_bars) else i

            new_sl = sl_updater.update_short_sl(pos.trailing_sl, bar_high, current_atr)
            account.update_trailing_sl_short(new_sl, pi)

            # CRITICAL FIX: Don't check stop on the entry bar.
            # The entry bar's high is used to SET the initial SL (High + mult*ATR).
            # On that same bar, bar_high is naturally >= the SL (equality case).
            # We defer the first stop check to the NEXT bar.
            if i > entry_bar:
                # Now check if this bar's high triggers the stop
                pass  # check inline below

            new_entry_bars.append(entry_bar)

        # --- Check stops (only for positions NOT entered this bar) ---
        # We need to check stops manually, skipping same-bar entries
        remaining_positions = []
        remaining_bars = []
        for pi in range(len(account.short_positions)):
            pos = account.short_positions[pi]
            entry_bar = position_entry_bars[pi] if pi < len(position_entry_bars) else i
            if i > entry_bar and bar_high >= pos.trailing_sl:
                # Stop triggered
                exit_price = max(pos.trailing_sl, bar_high)
                account._force_close(pos, exit_price, bar_ts, i, "TSL_STOP")
            else:
                remaining_positions.append(pos)
                remaining_bars.append(entry_bar)

        account.short_positions = remaining_positions
        position_entry_bars = remaining_bars

        # --- Process signals ---
        if i in sigs_by_bar:
            for sig in sigs_by_bar[i]:
                if sig.signal_type == SignalType.SELL:
                    pos = account.open_position(
                        direction=Direction.SHORT, price=bar_close,
                        timestamp=bar_ts, bar_index=i,
                        trailing_sl=sig.trailing_sl or 0.0,
                        divergence_type=sig.divergence_type.value if sig.divergence_type else "",
                    )
                    if pos:
                        position_entry_bars.append(i)

                elif sig.signal_type == SignalType.CLOSE_SHORT:
                    while account.short_positions:
                        account.close_position(
                            direction=Direction.SHORT, price=bar_close,
                            timestamp=bar_ts, bar_index=i, reason="SIGNAL_CLOSE",
                        )
                    position_entry_bars = []

    trades = account.closed_trades
    if len(trades) < 5:
        return None

    wins = [t for t in trades if t.net_pnl > 0]
    losses = [t for t in trades if t.net_pnl <= 0]
    net_pnl = sum(t.net_pnl for t in trades)
    total_profit = sum(t.net_pnl for t in wins)
    total_loss = abs(sum(t.net_pnl for t in losses))
    pf = total_profit / total_loss if total_loss > 0 else float('inf')

    return GridResult(
        stop_loss_mult=p.stop_loss_mult, range_lower=p.range_lower,
        range_upper=p.range_upper, rsi_period=p.rsi_period,
        atr_period=p.atr_period, take_profit_rsi=p.take_profit_rsi,
        total_trades=len(trades), wins=len(wins), net_pnl=net_pnl,
        net_pnl_pct=net_pnl / 10000 * 100,
        win_rate=len(wins) / len(trades) * 100, profit_factor=pf,
        avg_win=np.mean([t.net_pnl for t in wins]) if wins else 0,
        avg_loss=np.mean([t.net_pnl for t in losses]) if losses else 0,
        max_drawdown=account.max_drawdown_pct,
        avg_bars=np.mean([t.bars_held for t in trades]),
        total_costs=sum(t.entry_fee + t.exit_fee + t.slippage for t in trades),
    )


def main():
    t_start = time.time()
    df = pd.read_csv('/sessions/ecstatic-awesome-tesla/mnt/outputs/btc_data/btc_15m_2026.csv')
    df['timestamp'] = df['timestamp'].astype(int)
    print(f'Data: {len(df)} bars', flush=True)

    stop_loss_mults = [3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
    range_lowers = [5, 10, 15]
    range_uppers = [30, 45, 60, 90]
    rsi_periods = [14]
    atr_periods = [14]
    take_profits = [25, 20, 30]

    combos = list(product(stop_loss_mults, range_lowers, range_uppers,
                          rsi_periods, atr_periods, take_profits))
    combos = [c for c in combos if c[2] > c[1] + 5]
    print(f'Combos: {len(combos)}', flush=True)

    results = []
    for idx, (sm, rl, ru, rp, ap, tp) in enumerate(combos, 1):
        params = {
            'stop_loss_mult': sm, 'range_lower': rl, 'range_upper': ru,
            'rsi_period': rp, 'atr_period': ap, 'take_profit_rsi': tp,
        }
        r = run_single(params, df)
        if r:
            results.append(r)
        if idx % 20 == 0:
            print(f'  [{idx}/{len(combos)}] {time.time()-t_start:.0f}s, {len(results)} valid', flush=True)

    if not results:
        print('NO VALID RESULTS', flush=True)
        return

    results.sort(key=lambda x: x.net_pnl, reverse=True)

    print(f'\n{"="*145}', flush=True)
    print(f'TOP 25 PARAMETER SETS (by Net PnL)', flush=True)
    print(f'{"="*145}', flush=True)
    hdr = (f'{"SL":>5s} {"RL":>5s} {"RU":>5s} {"RSI":>5s} {"ATR":>5s} '
           f'{"TP":>5s} {"Trades":>7s} {"Win%":>6s} {"NetPnL":>10s} '
           f'{"PnL%":>7s} {"PF":>6s} {"AvgW":>8s} {"AvgL":>8s} '
           f'{"DD%":>7s} {"Bars":>5s}')
    print(hdr, flush=True)
    print('-' * 145, flush=True)
    for r in results[:25]:
        print(f'{r.stop_loss_mult:5.1f} {r.range_lower:5d} {r.range_upper:5d} '
              f'{r.rsi_period:5d} {r.atr_period:5d} {r.take_profit_rsi:5d} '
              f'{r.total_trades:7d} {r.win_rate:5.1f}% ${r.net_pnl:>9,.0f} '
              f'{r.net_pnl_pct:6.1f}% {r.profit_factor:5.1f} '
              f'${r.avg_win:>7,.0f} ${r.avg_loss:>7,.0f} '
              f'{r.max_drawdown:6.1f}% {r.avg_bars:5.1f}', flush=True)

    results_pf = sorted(results, key=lambda x: x.profit_factor, reverse=True)
    print(f'\nTOP 10 BY PROFIT FACTOR', flush=True)
    for r in results_pf[:10]:
        print(f'{r.stop_loss_mult:5.1f} {r.range_lower:5d} {r.range_upper:5d} '
              f'{r.total_trades:7d} {r.win_rate:5.1f}% PF={r.profit_factor:.1f} '
              f'PnL=${r.net_pnl:,.0f} AvgW=${r.avg_win:,.0f} AvgL=${r.avg_loss:,.0f} '
              f'Bars={r.avg_bars:.1f}', flush=True)

    best = results[0]
    print(f'\nBEST: SL={best.stop_loss_mult} RL={best.range_lower} RU={best.range_upper} '
          f'PnL=${best.net_pnl:,.0f} ({best.net_pnl_pct:.1f}%) Win%={best.win_rate:.1f}% '
          f'PF={best.profit_factor:.1f} Trades={best.total_trades} Bars={best.avg_bars:.1f}',
          flush=True)

    df_out = pd.DataFrame([{
        'stop_loss': r.stop_loss_mult, 'range_lower': r.range_lower,
        'range_upper': r.range_upper, 'rsi_period': r.rsi_period,
        'atr_period': r.atr_period, 'take_profit_rsi': r.take_profit_rsi,
        'trades': r.total_trades, 'wins': r.wins,
        'net_pnl': r.net_pnl, 'net_pnl_pct': r.net_pnl_pct,
        'win_rate': r.win_rate, 'profit_factor': r.profit_factor, 'avg_win': r.avg_win,
        'avg_loss': r.avg_loss, 'max_drawdown': r.max_drawdown,
        'avg_bars': r.avg_bars, 'total_costs': r.total_costs,
    } for r in results[:50]])

    out_path = 'grid_results_top50.csv'
    df_out.to_csv(out_path, index=False)
    elapsed = time.time() - t_start
    print(f'\nSaved top 50 to {out_path}', flush=True)
    print(f'Total time: {elapsed:.0f}s', flush=True)


if __name__ == '__main__':
    main()
