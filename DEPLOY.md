# Dual-Engine Divergence Trading System — 部署方案

## 机器要求

```
CPU:     1 核 (ARM/AMD64 均可)     ← 策略极轻量，每60秒一次计算
内存:    512 MB 足够               ← 实际占用约 100-150MB
磁盘:    2 GB                     ← 代码+日志+数据不超 1GB
网络:    稳定连 OKX API            ← ~60次请求/小时，~50KB/h
系统:    Linux (Ubuntu 22.04+)    ← 推荐，Windows/Mac 也可
```

**实际负载说明：**
- 每 60 秒抓一次 K 线（2个timeframe）
- 每次抓取 ~40KB 数据
- 检测计算耗时 <10ms
- 只有开仓平仓才调用 OKX 交易API
- **1核512M的轻量云服务器完全足够，月流量不超 50MB**

---

## 1. 服务器准备

```bash
# Ubuntu 22.04+
apt update && apt install -y python3 python3-pip git curl

# 安装 uv (Python 包管理器, 比 pip 快 10 倍)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 重载 shell
source ~/.bashrc
```

---

## 2. 部署代码

```bash
# 从 GitHub 拉取
git clone https://github.com/lxhlf520/Dual-engine-divergence-trading-system.git /opt/trader
cd /opt/trader

# 安装依赖
uv sync

# 或手动安装
uv pip install pandas numpy requests python-dotenv matplotlib ccxt==4.4.100
```

---

## 3. 配置 API 密钥

```bash
cp .env.example .env
chmod 600 .env   # 密钥文件仅本人可读
vim .env
```

`.env` 文件内容：

```ini
# OKX API (必填)
OKX_API_KEY=your_api_key_here
OKX_API_SECRET=your_secret_here
OKX_API_PASSWORD=your_passphrase_here

# 国内服务器必须配代理（否则连不上 OKX）
HTTP_PROXY=http://127.0.0.1:7890
HTTPS_PROXY=http://127.0.0.1:7890

# 警报推送 (推荐 Telegram)
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
# DINGTALK_WEBHOOK=可选

# 日志级别 (默认 INFO, 调试用 DEBUG)
LOG_LEVEL=INFO
```

---

## 4. 验证连接

```bash
uv run python -c "
import os
os.environ['LOG_LEVEL'] = 'ERROR'
from dual_engine_trader.execution.executor import OKXExecution
exe = OKXExecution()
bal = exe.get_balance()
print(f'OK - Balance: \${bal[\"total_equity\"]:.2f}')
exe.close()
"
```

正常返回：`OK - Balance: $500.00`

---

## 5. 启动方式

### 方式 A：systemd 服务（推荐——开机自启+自动重启）

```ini
# /etc/systemd/system/trader.service
[Unit]
Description=Dual-Engine Divergence Trading System
After=network.target
Wants=network-online.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/trader
EnvironmentFile=/opt/trader/.env
ExecStart=/root/.local/bin/uv run python -m dual_engine_trader.live_runner
Restart=always
RestartSec=15

# 资源限制
CPUQuota=20%
MemoryMax=300M
IOReadBandwidthMax=0
IOWriteBandwidthMax=10M

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable trader
sudo systemctl start trader
sudo systemctl status trader
```

### 方式 B：nohup 后台

```bash
cd /opt/trader
nohup uv run python -m dual_engine_trader.live_runner > /var/log/trader.log 2>&1 &
```

### 方式 C：tmux 会话

```bash
tmux new -s trader
cd /opt/trader && uv run python -m dual_engine_trader.live_runner
# Ctrl+B D 分离，tmux attach -t trader 重新接入
```

---

## 6. 查看运行状态

```bash
# 实时日志
tail -f /opt/trader/logs/trading_system.log

# systemd 日志
journalctl -u trader -f

# 每10轮输出一次状态（约10分钟）:
# Status: POS=NONE EQUITY=$500 day=+0.0% total=+0.0%
# Status: POS=short 4ct EQUITY=$512 day=+2.4% total=+2.4%
```

---

## 7. 风控熔断（代码内置，无需额外配置）

| 熔断条件 | 触发 | 行为 |
|---------|------|------|
| 当日亏损 >5%（$25）| $475 | 暂停当天交易，UTC 0点自动恢复 |
| 累计亏损 >15%（$75）| $425 | 降为 2 张模式 |
| 累计亏损 >25%（$125）| $375 | 全部平仓，停止运行 |

熔断触发后会在日志和 Telegram 上收到警报。

---

## 8. 文件结构

```
/opt/trader/
├── dual_engine_trader/     # 主程序 (2.5MB)
├── logs/
│   └── trading_system.log  # 运行日志（轮转, 20MB×10份）
├── historical_data/csv/
│   └── btc_15m_2026.csv    # 历史 K 线 (909KB)
├── live_output/
│   └── live_trades_*.json  # 交易记录
├── .env                    # API 密钥
├── pyproject.toml
└── DEPLOY.md
```

---

## 9. 国内服务器特别说明

**必须配置代理**才能连接 OKX API（香港/海外服务器不需要）。

推荐方式——在 `.env` 中：
```ini
HTTPS_PROXY=http://127.0.0.1:7890
```

如果你的代理需要认证：
```ini
HTTPS_PROXY=http://user:pass@proxy_ip:port
```

常见代理工具：
- **Clash**: 本地监听 `127.0.0.1:7890`
- **v2ray/xray**: 本地监听 `127.0.0.1:10809`
- **SSH tunnel**: `ssh -D 1080 user@your-proxy`

---

## 10. 日常运维

```bash
# 查看收益
journalctl -u trader --since today | grep "EQUITY"

# 查看信号
tail -f /opt/trader/logs/trading_system.log | grep "SIGNAL\|SELL\|CLOSE"

# 重启
sudo systemctl restart trader

# 更新代码
cd /opt/trader
git pull
sudo systemctl restart trader

# 停止交易
sudo systemctl stop trader
```

---

## 11. 注意事项

1. **不要手动调整持仓**——策略全自动管理，手动干预会破坏内部状态计数
2. **500U 够用**——5x / 4张只需 $480 保证金，留 $20 防穿仓
3. **网络断连会自动重试**——`WS_MAX_RETRIES=10`，重试间隔 5 秒
4. **日志会自动轮转**——单文件最大 20MB，保留 10 份历史
5. **机器人不对**——提币地址不要放在交易账户里
