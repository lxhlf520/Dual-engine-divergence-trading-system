# Dual-Engine Divergence Trading System -- Deployment Guide (uv)

## 1. Prerequisites

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
# Or: pip install uv

# Upload project to server
# scp -r dual-engine-trader/ user@server:/opt/dual-engine-trader
```

## 2. Configuration

```bash
cd /opt/dual-engine-trader
cp .env.example .env

# Edit .env with real API credentials
vim .env
```

`.env` contents:
```bash
OKX_API_KEY=your_api_key_here
OKX_API_SECRET=your_secret_here
OKX_API_PASSWORD=your_passphrase_here

# Domestic server: proxy is required
HTTP_PROXY=http://127.0.0.1:7890
HTTPS_PROXY=http://127.0.0.1:7890

# Telegram notifications (optional)
# TELEGRAM_BOT_TOKEN=
# TELEGRAM_CHAT_ID=
# DINGTALK_WEBHOOK=
```

## 3. Install Dependencies

```bash
cd /opt/dual-engine-trader
uv sync
# Or: uv pip install -e .
```

## 4. Verify Installation

```bash
# Test API connectivity
uv run python -c "
import os
os.environ['LOG_LEVEL'] = 'ERROR'
from dual_engine_trader.execution import OKXExecution
exe = OKXExecution()
bal = exe.get_balance()
print(f'OK - Balance: \${bal[\"total_equity\"]:.2f}')
exe.close()
"
```

## 5. Run Live Trading

```bash
# Foreground (Ctrl+C to stop)
uv run python -m dual_engine_trader.live_runner

# Background
nohup uv run python -m dual_engine_trader.live_runner > /var/log/trader.log 2>&1 &

# systemd service (recommended -- see below)
```

## 6. systemd Service (Recommended)

```ini
# /etc/systemd/system/trader.service
[Unit]
Description=Dual-Engine Divergence Trading System
After=network.target

[Service]
Type=simple
User=your_user
WorkingDirectory=/opt/dual-engine-trader
EnvironmentFile=/opt/dual-engine-trader/.env
ExecStart=/home/your_user/.cargo/bin/uv run python -m dual_engine_trader.live_runner
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable trader
sudo systemctl start trader
sudo systemctl status trader
```

## 7. Viewing Logs

```bash
# Live log tail
tail -f /opt/dual-engine-trader/logs/dual_engine_trader.log

# systemd logs
journalctl -u trader -f
```

## 8. Manual Commands

```bash
# Download historical data
uv run python -m dual_engine_trader.main download --since 2026-01-01 --all-timeframes

# Run backtest
uv run python -m dual_engine_trader.backtest.engine --bars 8000

# Run grid search
uv run python -m dual_engine_trader.backtest.grid_search
```
