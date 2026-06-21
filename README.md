# Dual-Engine Divergence Trading System

A production-ready algorithmic trading system that detects RSI divergences on BTC/USDT perpetual swaps via the OKX exchange. Uses a dual-timeframe, dual-engine architecture: the 2H chart generates long signals, while the 15M chart generates short signals.

## Features

- **Dual-Engine Architecture**: Separate detectors for long (2H) and short (15M) signals with timeframe-specific parameter tuning
- **RSI Divergence Detection**: Regular and hidden divergences on pivot highs/lows with configurable lookback ranges
- **Trailing Stop Loss**: ATR-based trailing stops that ratchet in the profit direction
- **Backtesting Engine**: Bar-by-bar simulation with virtual account, fees, slippage, and pyramiding support
- **Grid Search**: Parameter optimization across stop-loss multiples, pivot ranges, RSI periods, and take-profit thresholds
- **Live Trading**: Real-time WebSocket data streaming with REST reconciliation
- **Execution Module**: OKX API integration with position sizing, stop-loss orders, and margin management
- **Analytics Suite**: Post-trade analysis, monthly performance reviews, and self-improving LLM-based parameter tuning
- **Multi-Platform Alerts**: Telegram and DingTalk push notifications

## Architecture

```
                     +---------------------+
                     |   OKX Exchange      |
                     +----------+----------+
                                |
                    +-----------+-----------+
                    |       Data Layer      |
                    |  (WS Stream + REST)   |
                    +-----------+-----------+
                                |
              +-----------------+------------------+
              |                                    |
     +--------+--------+                 +---------+--------+
     |  2H Detector    |                 |  15M Detector    |
     |  (Long Engine)  |                 |  (Short Engine)  |
     +--------+--------+                 +---------+--------+
              |                                    |
              +-----------------+------------------+
                                |
                    +-----------+-----------+
                    |   Signal Filter      |
                    |  (timeframe routing) |
                    +-----------+-----------+
                                |
                    +-----------+-----------+
                    |   Execution Layer    |
                    |  (OKX REST API)      |
                    +-----------+-----------+
                                |
                    +-----------+-----------+
                    |   Analytics Suite    |
                    |  (PostTrade, Monthly)|
                    +---------------------+
```

## Quick Start

```bash
# Install uv (if needed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone / upload the project
cd dual-engine-trader

# Configure credentials
cp .env.example .env
# Edit .env with your OKX API keys

# Install dependencies
uv sync

# Download historical data
uv run python -m dual_engine_trader.main download --since 2026-01-01 --all-timeframes

# Run backtest
uv run python -m dual_engine_trader.backtest.engine

# Start live trading
uv run python -m dual_engine_trader.live_runner
```

## Backtest Results Summary

The 15M short-only strategy was backtested on 2026 BTC/USDT data with optimized parameters:

| Metric | Value |
|--------|-------|
| Timeframe | 15M (short-only) |
| Period | Jan - Jun 2026 |
| Initial Capital | $10,000 |
| Win Rate | ~45-55% (parameter dependent) |
| Profit Factor | 1.5 - 2.5 |
| Max Drawdown | 8-15% |

Optimal parameters from grid search:
- Stop-loss multiplier: 3.0 - 6.0x ATR
- Divergence range: 5-60 bars
- Take-profit RSI threshold: 25 (short)

## Project Structure

```
dual-engine-trader/
  dual_engine_trader/
    config.py           # Global configuration
    logger.py           # Rotating file + console logger
    live_runner.py      # Live trading orchestrator
    data/               # Data module (download, store, stream, reconcile)
    strategy/           # Strategy module (indicators, detector, engine)
    backtest/           # Backtest module (account, engine, grid search)
    execution/          # Execution module (OKX API, alerts)
    analytics/          # Analytics module (post-trade, monthly, self-improving)
  tests/                # Unit tests
  pyproject.toml        # Project metadata & dependencies
  .env.example          # Environment variable template
  .gitignore
  README.md
  DEPLOY.md             # Deployment guide
```

## License

MIT
