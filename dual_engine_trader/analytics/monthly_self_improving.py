"""
Self-Improving Monthly Analyzer — Core Engine

Tasks:
1. Data aggregation & JSON packaging
2. Trade classification (wick-stop, rollercoaster, funding)
3. LLM API client with prompt engineering
4. Parameter parser from LLM Markdown output
5. Shadow backtest validator
6. Human-in-the-loop config override
"""
import json
import re
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any
from copy import deepcopy

import numpy as np
import pandas as pd
import requests

from ..config import (
    ROOT_DIR, RSI_PERIOD, ATR_PERIOD, LB_L, LB_R,
    DEFAULT_LEVERAGE, MARGIN_MODE, BACKTEST_INITIAL_CAPITAL,
    BACKTEST_FEE_RATE, BACKTEST_SLIPPAGE_RATE, BACKTEST_PYRAMIDING,
    EXCHANGE_ID, SYMBOL, TIMEFRAMES,
    TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, DINGTALK_WEBHOOK,
)
from ..logger import get_logger
from ..strategy.detector import DivergenceDetector, DivergenceParams, TrailingStopUpdater, SignalType
from ..strategy.engine import Direction, SHORT_PARAMS, LONG_PARAMS
from ..strategy.indicators import compute_rsi, compute_atr, pivotlow, pivothigh
from ..backtest.account import VirtualAccount, Trade
from ..execution.alerts import AlertManager

from .self_improving_types import (
    ExitCategory, WickDiagnosis, RollercoasterDiagnosis,
    MonthlyPerformance, ExecutionQuality, TradeClassification,
    MonthlyReviewPayload, ParamChange, LLMResponse, ShadowBacktestResult,
)

logger = get_logger(__name__)


# Default Anthropic API endpoint (configurable)
ANTHROPIC_API_URL = os.getenv("ANTHROPIC_API_URL", "https://api.anthropic.com/v1/messages")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "claude-sonnet-4-20250514")
LLM_TIMEOUT_SECONDS = 60


class MonthlySelfImprovingAnalyzer:
    """Self-Improving Monthly Analyzer with LLM feedback loop."""

    def __init__(self, year: int, month: int, initial_capital: float = 10000.0):
        self.year = year
        self.month = month
        self.initial_capital = initial_capital
        self.alert = AlertManager()

        # Data containers
        self.kline_df: Optional[pd.DataFrame] = None
        self.live_trades: List[Trade] = []

        # Active parameters (current config snapshot)
        self.active_params: Dict[str, Any] = {}

        # Timestamp bounds
        self.month_start = pd.Timestamp(year, month, 1)
        if month == 12:
            self.month_end = pd.Timestamp(year + 1, 1, 1)
        else:
            self.month_end = pd.Timestamp(year, month + 1, 1)

        # Results
        self.performance: Optional[MonthlyPerformance] = None
        self.execution_quality: Optional[ExecutionQuality] = None
        self.trade_classification: Optional[TradeClassification] = None
        self.llm_response: Optional[LLMResponse] = None
        self.shadow_result: Optional[ShadowBacktestResult] = None

    # ============================================================
    # TASK 1: Data Aggregation & JSON Packaging
    # ============================================================
    def load_data(self, trades: List[Trade], kline_df: pd.DataFrame,
                  current_config: Dict[str, Any] = None) -> None:
        """Load monthly trade and K-line data."""
        # Filter trades to this month
        ms = int(self.month_start.timestamp() * 1000)
        me = int(self.month_end.timestamp() * 1000)

        self.live_trades = []
        for t in trades:
            entry_ts = getattr(t, 'entry_time', 0)
            if hasattr(t, 'entry_bar_ts'):
                entry_ts = t.entry_bar_ts
            if ms <= entry_ts < me:
                self.live_trades.append(t)

        # Slice K-line data (with warmup)
        warmup_start = self.month_start - pd.Timedelta(days=3)
        kline_df["datetime"] = pd.to_datetime(kline_df["timestamp"], unit="ms")
        kline_df["datetime"] = kline_df["datetime"].dt.tz_localize(None)
        self.kline_df = kline_df[
            (kline_df["datetime"] >= warmup_start) & (kline_df["datetime"] < self.month_end)
        ].copy()
        self.kline_df.set_index("datetime", inplace=True)
        self.kline_df.sort_index(inplace=True)

        # Snapshot active parameters
        self.active_params = current_config or {}
        if not self.active_params:
            self.active_params = {
                "long_2h.rsi_period": LONG_PARAMS.rsi_period,
                "long_2h.atr_period": LONG_PARAMS.atr_period,
                "long_2h.tp_rsi": LONG_PARAMS.take_profit_rsi,
                "long_2h.atr_stop": LONG_PARAMS.stop_loss_mult,
                "short_15m.rsi_period": SHORT_PARAMS.rsi_period,
                "short_15m.atr_period": SHORT_PARAMS.atr_period,
                "short_15m.tp_rsi": SHORT_PARAMS.take_profit_rsi,
                "short_15m.atr_stop": SHORT_PARAMS.stop_loss_mult,
            }

        logger.info("Loaded {} trades and {} K-lines for {:04d}-{:02d}".format(
            len(self.live_trades), len(self.kline_df), self.year, self.month))

    def compute_performance(self) -> MonthlyPerformance:
        """Aggregate monthly performance metrics."""
        trades = self.live_trades
        if not trades:
            return MonthlyPerformance(
                year=self.year, month=self.month, total_trades=0,
                wins=0, losses=0, win_rate_pct=0.0, profit_factor=0.0,
                net_pnl_usd=0.0, net_pnl_pct=0.0, max_drawdown_pct=0.0,
                sharpe_ratio=0.0, avg_bars_held=0.0,
                max_streak_win=0, max_streak_loss=0,
                total_fees_usd=0.0, total_slippage_usd=0.0,
                net_funding_fee_usd=0.0,
                btc_buyhold_return_pct=0.0, strategy_vs_benchmark_pct=0.0,
            )

        wins = [t for t in trades if t.net_pnl > 0]
        losses = [t for t in trades if t.net_pnl <= 0]
        n = len(trades)

        net_pnl = sum(t.net_pnl for t in trades)
        net_pnl_pct = net_pnl / self.initial_capital * 100

        total_profit = sum(t.net_pnl for t in wins)
        total_loss = abs(sum(t.net_pnl for t in losses))
        pf = total_profit / total_loss if total_loss > 0 else float('inf')

        # Drawdown
        equity = [self.initial_capital]
        for t in trades:
            equity.append(equity[-1] + t.net_pnl)
        peak = np.maximum.accumulate(equity)
        dd = (np.array(equity) - peak) / peak * 100
        max_dd = abs(min(dd))

        # Sharpe
        returns = [t.net_pnl / self.initial_capital for t in trades]
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(12) if len(returns) > 1 and np.std(returns) > 0 else 0.0

        # Fees
        total_fees = sum(getattr(t, 'entry_fee', 0) + getattr(t, 'exit_fee', 0) for t in trades)
        total_slippage = sum(getattr(t, 'slippage', 0) for t in trades)

        # Funding (placeholder - real impl would fetch from exchange API)
        net_funding = 0.0

        # BTC benchmark
        btc_return = self._compute_btc_benchmark()

        # Streaks
        max_sw, max_sl = 0, 0
        cw, cl = 0, 0
        for t in trades:
            if t.net_pnl > 0:
                cw += 1; cl = 0; max_sw = max(max_sw, cw)
            else:
                cl += 1; cw = 0; max_sl = max(max_sl, cl)

        self.performance = MonthlyPerformance(
            year=self.year, month=self.month,
            total_trades=n, wins=len(wins), losses=len(losses),
            win_rate_pct=len(wins) / n * 100 if n > 0 else 0,
            profit_factor=pf, net_pnl_usd=net_pnl, net_pnl_pct=net_pnl_pct,
            max_drawdown_pct=max_dd, sharpe_ratio=sharpe,
            avg_bars_held=np.mean([getattr(t, 'bars_held', 0) for t in trades]),
            max_streak_win=max_sw, max_streak_loss=max_sl,
            total_fees_usd=total_fees, total_slippage_usd=total_slippage,
            net_funding_fee_usd=net_funding,
            btc_buyhold_return_pct=btc_return,
            strategy_vs_benchmark_pct=net_pnl_pct - btc_return,
        )
        return self.performance

    def _compute_btc_benchmark(self) -> float:
        if self.kline_df is None or len(self.kline_df) < 2:
            return 0.0
        ms = self.month_start.tz_localize(None)
        me = self.month_end.tz_localize(None)
        df = self.kline_df[(self.kline_df.index >= ms) & (self.kline_df.index < me)]
        if len(df) < 2:
            return 0.0
        return (float(df["close"].iloc[-1]) / float(df["close"].iloc[0]) - 1) * 100

    def compute_execution_quality(self) -> ExecutionQuality:
        """Slippage & latency analysis from trade log."""
        if not self.live_trades:
            return ExecutionQuality(0, 0, 0, 0, 0, 0, 0)

        entry_slips = []
        exit_slips = []
        latencies = []
        slip_over_5bp = 0.0
        n = len(self.live_trades)

        for t in self.live_trades:
            # Entry slippage: actual fill vs signal price
            entry_sl = getattr(t, 'entry_slippage', 0)
            exit_sl = getattr(t, 'exit_slippage', 0)
            lat = getattr(t, 'latency_seconds', 0)
            if not entry_sl:
                entry_sl = 0
            if not exit_sl:
                exit_sl = 0
            entry_slips.append(entry_sl)
            exit_slips.append(exit_sl)
            latencies.append(lat)

        # Count orders with >5bp total slippage
        for i in range(n):
            if i < len(entry_slips) and i < len(exit_slips):
                if entry_slips[i] + exit_slips[i] > 5:
                    slip_over_5bp += 1

        self.execution_quality = ExecutionQuality(
            avg_entry_slippage_bp=float(np.mean(entry_slips)),
            max_entry_slippage_bp=float(np.max(entry_slips)),
            avg_exit_slippage_bp=float(np.mean(exit_slips)),
            max_exit_slippage_bp=float(np.max(exit_slips)),
            avg_latency_ms=float(np.mean(latencies)),
            max_latency_ms=float(np.max(latencies)),
            pct_orders_with_slippage_over_5bp=slip_over_5bp / n * 100 if n > 0 else 0,
        )
        return self.execution_quality

    def classify_trades(self) -> TradeClassification:
        """Classify trades: exit reasons, wick diagnosis, rollercoaster."""
        trades = self.live_trades
        n = len(trades)
        if n == 0:
            return TradeClassification(
                total_trades=0, tsl_stop_count=0, signal_close_count=0,
                rsi_tp_count=0, force_close_count=0,
                wick_stop_reversal_count=0, wick_stop_reversal_pct=0,
                wick_stop_avg_reversal_bars=0,
                near_tp_reversed_count=0, near_tp_reversed_pct=0,
                near_tp_avg_miss_margin_pct=0,
                avg_win_per_trade_usd=0, avg_loss_per_trade_usd=0,
                max_single_win_usd=0, max_single_loss_usd=0,
            )

        # Exit reason counts
        tsl_count = sum(1 for t in trades if "TSL" in getattr(t, 'exit_reason', ''))
        sig_count = sum(1 for t in trades if "SIGNAL" in getattr(t, 'exit_reason', ''))
        rsi_count = sum(1 for t in trades if "RSI" in getattr(t, 'exit_reason', ''))
        force_count = sum(1 for t in trades if "FORCE" in getattr(t, 'exit_reason', ''))

        # Wick diagnosis: for TSL_STOP trades, check if price reversed within 3 bars
        wick_reversal = 0
        reversal_bars = []
        for t in trades:
            if "TSL" not in getattr(t, 'exit_reason', ''):
                continue
            # Simple heuristic: if the trade was SHORT and price went DOWN after stop-out,
            # or LONG and price went UP, it's a wick stop
            # Since we don't have bar-level granularity here, use exit-to-close direction
            is_short = getattr(t, 'direction', None)
            if hasattr(is_short, 'value'):
                is_short = is_short.value == "SHORT"
            else:
                is_short = str(is_short).upper() == "SHORT"

            exit_price = t.exit_price
            # Check if price moved favorably after exit (reversal)
            # Placeholder: assume reversal if net_pnl was small negative
            if t.net_pnl > -200:  # arbitrary threshold
                wick_reversal += 1
                reversal_bars.append(3)  # placeholder

        # Rollercoaster: trades that approached RSI take-profit but missed
        # For SHORT: RSI approached 25 but bounced. For LONG: RSI approached 80 but pulled back.
        near_tp = sum(1 for t in trades if "SIGNAL_CLOSE" in getattr(t, 'exit_reason', ''))

        wins = [t for t in trades if t.net_pnl > 0]
        losses = [t for t in trades if t.net_pnl <= 0]

        self.trade_classification = TradeClassification(
            total_trades=n,
            tsl_stop_count=tsl_count,
            signal_close_count=sig_count,
            rsi_tp_count=rsi_count,
            force_close_count=force_count,
            wick_stop_reversal_count=wick_reversal,
            wick_stop_reversal_pct=wick_reversal / max(tsl_count, 1) * 100,
            wick_stop_avg_reversal_bars=float(np.mean(reversal_bars)) if reversal_bars else 0,
            near_tp_reversed_count=near_tp,
            near_tp_reversed_pct=near_tp / n * 100 if n > 0 else 0,
            near_tp_avg_miss_margin_pct=2.5,  # placeholder
            avg_win_per_trade_usd=float(np.mean([t.net_pnl for t in wins])) if wins else 0,
            avg_loss_per_trade_usd=float(np.mean([t.net_pnl for t in losses])) if losses else 0,
            max_single_win_usd=float(max(t.net_pnl for t in trades)),
            max_single_loss_usd=float(min(t.net_pnl for t in trades)),
        )
        return self.trade_classification

    def build_llm_payload(self) -> MonthlyReviewPayload:
        """Build complete JSON payload for LLM consumption."""
        if not all([self.performance, self.execution_quality, self.trade_classification]):
            self.compute_performance()
            self.compute_execution_quality()
            self.classify_trades()

        # Top 5 worst and best trades
        sorted_trades = sorted(self.live_trades, key=lambda t: t.net_pnl, reverse=True)
        trade_summary = []
        for t in sorted_trades[:5] + sorted_trades[-5:]:
            trade_summary.append({
                "entry_price": getattr(t, 'entry_price', 0),
                "exit_price": getattr(t, 'exit_price', 0),
                "net_pnl": getattr(t, 'net_pnl', 0),
                "net_pnl_pct": getattr(t, 'net_pnl_pct', 0),
                "exit_reason": getattr(t, 'exit_reason', ''),
                "bars_held": getattr(t, 'bars_held', 0),
                "direction": str(getattr(t, 'direction', '')),
            })

        # NAV summary
        nav_summary = self._build_nav_summary()

        return MonthlyReviewPayload(
            generated_at=datetime.now(timezone.utc).isoformat(),
            period="{}-{:02d}".format(self.year, self.month),
            initial_capital_usd=self.initial_capital,
            current_equity_usd=self.initial_capital + self.performance.net_pnl_usd,
            active_parameters=self.active_params,
            performance=self.performance.to_dict(),
            execution_quality=self.execution_quality.to_dict(),
            trade_classification=self.trade_classification.to_dict(),
            trade_summary=trade_summary,
            nav_summary=nav_summary,
        )

    def _build_nav_summary(self) -> List[Dict[str, Any]]:
        """Daily NAV points for the month."""
        if self.kline_df is None:
            return []
        ms = self.month_start.tz_localize(None)
        me = self.month_end.tz_localize(None)
        df = self.kline_df[(self.kline_df.index >= ms) & (self.kline_df.index < me)]
        if df.empty:
            return []
        nav = []
        equity = self.initial_capital
        btc_initial = float(df["close"].iloc[0])
        for date, day_df in df.groupby(df.index.date):
            for t in self.live_trades:
                if getattr(t, 'exit_time', 0):
                    exit_dt = pd.Timestamp(t.exit_time, unit='ms' if isinstance(t.exit_time, (int, float)) else None)
                    if exit_dt.date() == date:
                        equity += t.net_pnl
            nav.append({
                "date": str(date),
                "equity": round(equity, 2),
                "btc_price": round(float(day_df["close"].iloc[-1]), 2),
                "btc_nav": round(float(day_df["close"].iloc[-1]) / btc_initial, 4),
            })
        return nav

    # ============================================================
    # TASK 2: LLM Diagnostic Prompt
    # ============================================================
    SYSTEM_PROMPT = """You are a Lead Risk Control Officer for a Crypto Quantitative Hedge Fund. Analyze the provided monthly trading data and propose parameter optimization for the "Dual RSI Divergence Strategy".

Follow these strict quantitative finance rules:
1. [Anti-Overfitting]: Do not radically change baseline parameters due to isolated losses. Only suggest adjustments if a loss pattern is statistically dominant.
2. [Pin-Bar/Wick Diagnosis]: If data shows frequent stop-outs by market wicks followed by immediate reversals in the predicted direction, the market noise is high. Advise INCREASING the ATR stop multiplier (e.g., from 2.5 to 2.8).
3. [Profit Rollercoaster Diagnosis]: If positions frequently approach the RSI take-profit line but miss it by a margin, leading to massive drawdowns, the target is too aggressive. Advise DECREASING the TP RSI level (e.g., Long TP from 80 to 75).
4. [Slippage Control]: If live performance lags far behind the backtest due to slippage, suggest increasing the backtest baseline slippage parameter or implementing limit-order retries.

Output Format (Strictly Markdown):
### Health Diagnosis: [Excellent / Good / Fair / Poor]
### Parameter Adjustments:
- long_2h.tp_rsi: [New Value] (Brief Reason)
- short_15m.atr_stop: [New Value] (Brief Reason)
### Next Month Risk Warning:
[Provide macro or behavioral risk warnings based on current market ATR velocity]"""

    def call_llm(self, payload: MonthlyReviewPayload) -> LLMResponse:
        """Send monthly data to LLM and parse response."""
        json_data = payload.to_json(indent=2)

        user_prompt = "Monthly Trading Review Data:\n```json\n{}\n```".format(json_data)

        # Try Anthropic API first, fallback to local heuristic
        raw_text = ""
        try:
            raw_text = self._call_anthropic(user_prompt)
        except Exception as e:
            logger.error("LLM API call failed: {}. Using heuristic fallback.".format(e))
            self.alert.error_alert("LLM API Failed", str(e))
            raw_text = self._generate_heuristic_response(payload)

        # Parse LLM output
        return self._parse_llm_response(raw_text)

    def _call_anthropic(self, user_prompt: str) -> str:
        """Call Anthropic API."""
        if not ANTHROPIC_API_KEY:
            raise RuntimeError("ANTHROPIC_API_KEY not set")

        headers = {
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        body = {
            "model": LLM_MODEL,
            "max_tokens": 1024,
            "messages": [
                {"role": "user", "content": user_prompt}
            ],
            "system": self.SYSTEM_PROMPT,
        }

        resp = requests.post(ANTHROPIC_API_URL, headers=headers, json=body, timeout=LLM_TIMEOUT_SECONDS)
        resp.raise_for_status()
        data = resp.json()
        return data["content"][0]["text"]

    def _generate_heuristic_response(self, payload: MonthlyReviewPayload) -> str:
        """Fallback: rule-based heuristic when LLM unavailable."""
        perf = payload.performance
        tc = payload.trade_classification
        params = payload.active_parameters

        lines = []
        # Health
        if perf["win_rate_pct"] >= 50 and perf["profit_factor"] >= 1.5:
            lines.append("### Health Diagnosis: Good")
        elif perf["win_rate_pct"] >= 35:
            lines.append("### Health Diagnosis: Fair")
        else:
            lines.append("### Health Diagnosis: Poor")

        lines.append("### Parameter Adjustments:")

        # Wick diagnosis
        wick_pct = tc.get("wick_stop_reversal_pct", 0)
        if wick_pct > 30:
            cur_sl = params.get("short_15m.atr_stop", 2.8)
            new_sl = min(cur_sl * 1.15, 15.0)
            lines.append("- short_15m.atr_stop: {:.1f} (High wick reversal rate {:.0f}% - increase ATR stop)".format(new_sl, wick_pct))

        # Rollercoaster
        near_tp = tc.get("near_tp_reversed_pct", 0)
        if near_tp > 20:
            cur_tp = params.get("short_15m.tp_rsi", 25)
            new_tp = max(cur_tp - 5, 15)
            lines.append("- short_15m.tp_rsi: {:.0f} (High rollercoaster rate {:.0f}% - lower TP RSI)".format(new_tp, near_tp))

        # Default: no changes
        if len(lines) == 2:
            lines.append("- No changes recommended. Strategy performing within expected bounds.")

        lines.append("### Next Month Risk Warning:")
        lines.append("- Monitor ATR velocity for regime change detection.")
        lines.append("- Current max drawdown {:.1f}% — ensure adequate margin buffer.".format(perf["max_drawdown_pct"]))

        return "\n".join(lines)

    def _parse_llm_response(self, raw_text: str) -> LLMResponse:
        """Parse LLM Markdown output into structured changes."""
        changes = []
        warnings = []
        errors = []
        health = "Fair"

        try:
            # Extract health diagnosis
            health_match = re.search(r'(?:#+\s*)?(?:Strategy\s*)?Health\s*Diagnosis[:\s]*(\w+)', raw_text, re.IGNORECASE)
            if health_match:
                health = health_match.group(1).strip()

            # Extract parameter changes
            param_pattern = r'[-*]\s*(long_2h\.\w+|short_15m\.\w+)\s*:\s*([\d.]+)\s*\(?(.+?)\)?'
            for match in re.finditer(param_pattern, raw_text):
                param_name = match.group(1)
                try:
                    new_val = float(match.group(2))
                except ValueError:
                    errors.append("Could not parse value for {}".format(param_name))
                    continue
                reason = match.group(3).strip()
                cur_val = self.active_params.get(param_name, 0)

                changes.append(ParamChange(
                    param_path=param_name,
                    current_value=float(cur_val) if cur_val else 0.0,
                    proposed_value=new_val,
                    reason=reason,
                ))

            # Extract risk warnings
            risk_section = False
            for line in raw_text.split("\n"):
                line = line.strip()
                if "risk warning" in line.lower():
                    risk_section = True
                    continue
                if risk_section and line.startswith("-"):
                    warnings.append(line.lstrip("- ").strip())

        except Exception as e:
            errors.append("Parse error: {}".format(str(e)))

        self.llm_response = LLMResponse(
            raw_text=raw_text,
            health_diagnosis=health,
            parameter_changes=changes,
            risk_warnings=warnings,
            parse_success=len(errors) == 0,
            parse_errors=errors,
        )
        return self.llm_response

    # ============================================================
    # TASK 3: Semi-Automated Parameter Override
    # ============================================================
    def generate_proposed_config(self, output_path: str) -> Dict[str, Any]:
        """Generate proposed config JSON from LLM changes."""
        if not self.llm_response:
            raise RuntimeError("No LLM response available. Run call_llm() first.")

        proposed = deepcopy(self.active_params)
        for change in self.llm_response.parameter_changes:
            key = change.param_path
            proposed[key] = change.proposed_value

        # Write proposed config
        with open(output_path, "w") as f:
            json.dump(proposed, f, indent=2)

        logger.info("Proposed config written to {}".format(output_path))
        return proposed

    def run_shadow_backtest(self, proposed_config: Dict[str, Any],
                            kline_csv_6months: str) -> ShadowBacktestResult:
        """Run 6-month backtest with proposed parameters."""
        # Load 6 months of data
        df = pd.read_csv(kline_csv_6months)
        df["timestamp"] = df["timestamp"].astype(int)

        # Build DivergenceParams from proposed config
        sp_sl = proposed_config.get("short_15m.atr_stop", SHORT_PARAMS.stop_loss_mult)
        sp_tp = proposed_config.get("short_15m.tp_rsi", SHORT_PARAMS.take_profit_rsi)
        sp_rsi = proposed_config.get("short_15m.rsi_period", SHORT_PARAMS.rsi_period)

        proposed_params = DivergenceParams(
            rsi_period=int(sp_rsi), atr_period=SHORT_PARAMS.atr_period,
            lb_l=LB_L, lb_r=LB_R,
            take_profit_rsi=int(sp_tp), stop_loss_mult=sp_sl,
            range_lower=SHORT_PARAMS.range_lower, range_upper=SHORT_PARAMS.range_upper,
            plot_bear=True, plot_hidden_bear=True, plot_bull=False,
        )

        # Current config params
        cur_sl = SHORT_PARAMS.stop_loss_mult
        cur_tp = SHORT_PARAMS.take_profit_rsi
        cur_params = DivergenceParams(
            rsi_period=SHORT_PARAMS.rsi_period, atr_period=SHORT_PARAMS.atr_period,
            lb_l=LB_L, lb_r=LB_R,
            take_profit_rsi=cur_tp, stop_loss_mult=cur_sl,
            range_lower=SHORT_PARAMS.range_lower, range_upper=SHORT_PARAMS.range_upper,
            plot_bear=True, plot_hidden_bear=True, plot_bull=False,
        )

        # Run both backtests
        cur_result = self._backtest_with_params(df, cur_params)
        prop_result = self._backtest_with_params(df, proposed_params)

        # Compare
        pnl_change = prop_result["net_pnl_pct"] - cur_result["net_pnl_pct"]
        dd_change = prop_result["max_dd_pct"] - cur_result["max_dd_pct"]

        # Verdict
        verdict = "APPROVE"
        if prop_result["max_dd_pct"] > 80 and dd_change > 20:
            verdict = "REJECT"
        elif pnl_change < -10:
            verdict = "REJECT"
        elif dd_change > 10:
            verdict = "NEEDS_REVIEW"

        self.shadow_result = ShadowBacktestResult(
            period_months=6,
            current_net_pnl_pct=cur_result["net_pnl_pct"],
            proposed_net_pnl_pct=prop_result["net_pnl_pct"],
            pnl_change_pct=pnl_change,
            current_max_dd_pct=cur_result["max_dd_pct"],
            proposed_max_dd_pct=prop_result["max_dd_pct"],
            dd_change_pct=dd_change,
            current_win_rate=cur_result["win_rate"],
            proposed_win_rate=prop_result["win_rate"],
            current_sharpe=cur_result.get("sharpe", 0),
            proposed_sharpe=prop_result.get("sharpe", 0),
            current_total_trades=cur_result["total_trades"],
            proposed_total_trades=prop_result["total_trades"],
            verdict=verdict,
        )
        return self.shadow_result

    def _backtest_with_params(self, df: pd.DataFrame, params: DivergenceParams) -> Dict[str, float]:
        """Run a single backtest with given parameters."""
        account = VirtualAccount(
            initial_capital=self.initial_capital,
            fee_rate=BACKTEST_FEE_RATE, slippage_rate=BACKTEST_SLIPPAGE_RATE,
            max_pyramiding=BACKTEST_PYRAMIDING,
        )
        detector = DivergenceDetector(params)
        sl_updater = TrailingStopUpdater(stop_loss_mult=params.stop_loss_mult)

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
        atr_series = compute_atr(high, low, close, period=params.atr_period)

        position_entry_bars = []
        warmup = 60

        for i in range(warmup + 20, n):
            bar_high = float(high.iloc[i])
            bar_low = float(low.iloc[i])
            bar_close = float(close.iloc[i])
            bar_ts = int(df["timestamp"].iloc[i])
            current_atr = float(atr_series.iloc[i]) if not pd.isna(atr_series.iloc[i]) else 0.0

            # Update stops
            for pi in range(len(account.short_positions)):
                pos = account.short_positions[pi]
                new_sl = sl_updater.update_short_sl(pos.trailing_sl, bar_high, current_atr)
                account.update_trailing_sl_short(new_sl, pi)

            # Check stops (skip entry bar)
            remaining_pos = []
            remaining_bars = []
            for pi in range(len(account.short_positions)):
                pos = account.short_positions[pi]
                eb = position_entry_bars[pi] if pi < len(position_entry_bars) else i
                if i > eb and bar_high >= pos.trailing_sl:
                    exit_price = max(pos.trailing_sl, bar_high)
                    account._force_close(pos, exit_price, bar_ts, i, "TSL_STOP")
                else:
                    remaining_pos.append(pos)
                    remaining_bars.append(eb)
            account.short_positions = remaining_pos
            position_entry_bars = remaining_bars

            # Process signals
            if i in sigs_by_bar:
                for sig in sigs_by_bar[i]:
                    if sig.signal_type == SignalType.SELL:
                        pos = account.open_position(
                            direction=Direction.SHORT, price=bar_close,
                            timestamp=bar_ts, bar_index=i,
                            trailing_sl=sig.trailing_sl or 0.0,
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
        if not trades:
            return {"net_pnl_pct": 0, "max_dd_pct": 0, "win_rate": 0, "total_trades": 0, "sharpe": 0}

        wins = [t for t in trades if t.net_pnl > 0]
        net_pnl = sum(t.net_pnl for t in trades)
        net_pnl_pct = net_pnl / self.initial_capital * 100

        equity = [self.initial_capital] + [self.initial_capital + sum(t2.net_pnl for t2 in trades[:i+1]) for i in range(len(trades))]
        peak = np.maximum.accumulate(equity)
        max_dd = abs(min((np.array(equity) - peak) / peak * 100))

        win_rate = len(wins) / len(trades) * 100
        returns = [t.net_pnl / self.initial_capital for t in trades]
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(12) if len(returns) > 1 and np.std(returns) > 0 else 0.0

        return {
            "net_pnl_pct": net_pnl_pct, "max_dd_pct": max_dd,
            "win_rate": win_rate, "total_trades": len(trades), "sharpe": sharpe,
        }

    def request_human_confirmation(self) -> bool:
        """Print confirmation prompt and wait for user input."""
        if not self.shadow_result:
            print("No shadow backtest result available.")
            return False

        sr = self.shadow_result
        print("\n" + "=" * 70)
        print("  AI Proposed Parameter Changes")
        print("=" * 70)
        for change in (self.llm_response.parameter_changes if self.llm_response else []):
            print("  {}: {} -> {} ({})".format(change.param_path, change.current_value, change.proposed_value, change.reason))

        print("\n  Shadow Backtest (6-month):")
        print("    Current  PnL: {:+.1f}%   Max DD: {:.1f}%   Win Rate: {:.1f}%   Trades: {}".format(
            sr.current_net_pnl_pct, sr.current_max_dd_pct, sr.current_win_rate, sr.current_total_trades))
        print("    Proposed PnL: {:+.1f}%   Max DD: {:.1f}%   Win Rate: {:.1f}%   Trades: {}".format(
            sr.proposed_net_pnl_pct, sr.proposed_max_dd_pct, sr.proposed_win_rate, sr.proposed_total_trades))
        print("    PnL Change: {:+.1f}%   DD Change: {:+.1f}%".format(sr.pnl_change_pct, sr.dd_change_pct))
        print("    AI Verdict: {}".format(sr.verdict))
        print("=" * 70)

        # In production, wait for real input. For now, return verdict-based auto-approval.
        if sr.verdict == "REJECT":
            print("  Auto-REJECTED: proposed config dangerous.")
            logger.warning("Human confirmation auto-rejected: high DD risk")
            return False

        print("  APPLY new parameters to live config? (Y/N): ", end="")
        user_input = input().strip().upper()
        return user_input == "Y"

    def apply_approved_config(self, proposed_config: Dict[str, Any], config_path: str) -> bool:
        """Write approved config to the main config file."""
        try:
            with open(config_path, "r") as f:
                existing = json.load(f)
        except Exception:
            existing = {}

        existing.update(proposed_config)

        backup_path = config_path + ".backup_{}".format(datetime.now().strftime("%Y%m%d_%H%M%S"))
        try:
            with open(config_path, "r") as f:
                with open(backup_path, "w") as bf:
                    bf.write(f.read())
        except Exception:
            pass

        with open(config_path, "w") as f:
            json.dump(existing, f, indent=2)

        logger.info("Config updated and backed up to {}".format(backup_path))
        self.alert.send("Config Applied", "New params applied. Backup: {}".format(backup_path), "INFO")
        return True

    # ============================================================
    # Main Pipeline
    # ============================================================
    def run_full_pipeline(self, trades: List[Trade], kline_df: pd.DataFrame,
                          kline_csv_6m: str, config_path: str = "config.json",
                          skip_llm: bool = False, auto_apply: bool = False) -> Dict[str, Any]:
        """Execute full monthly self-improvement pipeline."""
        # Task 1: Aggregate
        self.load_data(trades, kline_df)
        perf = self.compute_performance()
        exec_q = self.compute_execution_quality()
        tc = self.classify_trades()

        payload = self.build_llm_payload()
        logger.info("Monthly payload built. Trades: {}, PnL: {:+.2f}%".format(
            perf.total_trades, perf.net_pnl_pct))

        if skip_llm:
            return {"performance": perf.to_dict(), "execution_quality": exec_q.to_dict(),
                    "trade_classification": tc.to_dict(), "payload": payload.to_json()}

        # Task 2: LLM Diagnosis
        llm_resp = self.call_llm(payload)
        logger.info("LLM diagnosis: {}, {} changes proposed".format(
            llm_resp.health_diagnosis, len(llm_resp.parameter_changes)))

        if not llm_resp.parameter_changes:
            logger.info("No parameter changes proposed. Skipping backtest.")
            return {"performance": perf.to_dict(), "execution_quality": exec_q.to_dict(),
                    "trade_classification": tc.to_dict(), "llm_response": llm_resp.to_dict()}

        # Task 3: Shadow backtest + confirmation
        proposed = self.generate_proposed_config(
            str(Path(config_path).parent / "config_proposed.json"))

        shadow = self.run_shadow_backtest(proposed, kline_csv_6m)

        approved = False
        if auto_apply and shadow.verdict == "APPROVE":
            approved = True
        elif not auto_apply:
            approved = self.request_human_confirmation()

        if approved:
            self.apply_approved_config(proposed, config_path)

        return {
            "performance": perf.to_dict(),
            "execution_quality": exec_q.to_dict(),
            "trade_classification": tc.to_dict(),
            "llm_response": llm_resp.to_dict(),
            "shadow_backtest": shadow.to_dict(),
            "config_applied": approved,
        }
