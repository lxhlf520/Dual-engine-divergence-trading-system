"""
Monthly Self-Improving Analyzer — Data Structures & Encoders

JSON schema for the monthly review payload sent to LLM.
"""
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import datetime
import json


# ============================================================
# Trade Classification Enums
# ============================================================
class ExitCategory(Enum):
    TSL_STOP = "tsl_stop"                  # ATR trailing stop
    SIGNAL_CLOSE = "signal_close"          # divergence close signal
    RSI_TP = "rsi_take_profit"             # RSI take-profit hit
    EOD_FORCE = "eod_force_close"          # forced close at period end
    MANUAL = "manual"                      # manual close


class WickDiagnosis(Enum):
    """Pin-bar / wick stop-out classification"""
    WICK_STOP_REVERSAL = "wick_stop_reversal"      # stopped out by wick, price reversed in predicted direction
    WICK_STOP_CONTINUATION = "wick_stop_continuation"  # stopped out by wick, price continued against
    NO_WICK = "no_wick"                             # normal stop-out


class RollercoasterDiagnosis(Enum):
    """Profit rollercoaster classification"""
    NEAR_TP_REVERSED = "near_tp_reversed"    # approached TP but missed, then reversed hard
    HIT_TP = "hit_tp"                        # hit take-profit cleanly
    FAR_FROM_TP = "far_from_tp"              # never approached TP zone
    NO_TP_SET = "no_tp_set"                  # no take-profit configured


# ============================================================
# Core Data Structures for LLM JSON Payload
# ============================================================
@dataclass
class MonthlyPerformance:
    """Aggregated monthly performance metrics"""
    year: int
    month: int
    total_trades: int
    wins: int
    losses: int
    win_rate_pct: float
    profit_factor: float
    net_pnl_usd: float
    net_pnl_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    avg_bars_held: float
    max_streak_win: int
    max_streak_loss: int
    # Fees
    total_fees_usd: float
    total_slippage_usd: float
    net_funding_fee_usd: float
    # Benchmark
    btc_buyhold_return_pct: float
    strategy_vs_benchmark_pct: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExecutionQuality:
    """Slippage and latency analysis"""
    avg_entry_slippage_bp: float
    max_entry_slippage_bp: float
    avg_exit_slippage_bp: float
    max_exit_slippage_bp: float
    avg_latency_ms: float
    max_latency_ms: float
    pct_orders_with_slippage_over_5bp: float   # % of orders > 5bp slippage

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TradeClassification:
    """Categorized trade statistics for pattern diagnosis"""
    total_trades: int
    # Exit reasons
    tsl_stop_count: int
    signal_close_count: int
    rsi_tp_count: int
    force_close_count: int
    # Wick diagnosis
    wick_stop_reversal_count: int       # stopped by wick, reversed → noise too high
    wick_stop_reversal_pct: float
    wick_stop_avg_reversal_bars: float   # how many bars after stop did price reverse?
    # Rollercoaster diagnosis
    near_tp_reversed_count: int          # approached TP but missed → TP too aggressive
    near_tp_reversed_pct: float
    near_tp_avg_miss_margin_pct: float   # avg % away from TP when reversed
    # Profit/loss distribution
    avg_win_per_trade_usd: float
    avg_loss_per_trade_usd: float
    max_single_win_usd: float
    max_single_loss_usd: float
    profit_distribution_buckets: Dict[str, int] = field(default_factory=dict)
    # Time-of-day heatmap (UTC hour -> trade count)
    hour_distribution: Dict[int, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # convert Enum keys to strings for JSON
        return d


@dataclass
class MonthlyReviewPayload:
    """Complete JSON payload sent to LLM"""
    generated_at: str                           # ISO timestamp
    period: str                                 # "YYYY-MM"
    initial_capital_usd: float
    current_equity_usd: float
    active_parameters: Dict[str, Any]           # current strategy params
    performance: Dict[str, Any]
    execution_quality: Dict[str, Any]
    trade_classification: Dict[str, Any]
    trade_summary: List[Dict[str, Any]]         # top 5 worst/best trades
    nav_summary: List[Dict[str, Any]]           # daily NAV points (reduced)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(asdict(self), indent=indent, default=str)


# ============================================================
# Parameter Change Request (from LLM output)
# ============================================================
@dataclass
class ParamChange:
    """Single parameter change proposed by LLM"""
    param_path: str                     # e.g. "short_15m.atr_stop" or "long_2h.tp_rsi"
    current_value: float
    proposed_value: float
    reason: str
    confidence: str = "medium"          # high / medium / low

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LLMResponse:
    """Parsed LLM output"""
    raw_text: str
    health_diagnosis: str               # Excellent / Good / Fair / Poor
    parameter_changes: List[ParamChange]
    risk_warnings: List[str]
    parse_success: bool
    parse_errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "health_diagnosis": self.health_diagnosis,
            "parameter_changes": [pc.to_dict() for pc in self.parameter_changes],
            "risk_warnings": self.risk_warnings,
            "parse_success": self.parse_success,
        }


# ============================================================
# Shadow Backtest Result
# ============================================================
@dataclass
class ShadowBacktestResult:
    """6-month backtest comparison: current vs proposed config"""
    period_months: int
    current_net_pnl_pct: float
    proposed_net_pnl_pct: float
    pnl_change_pct: float               # relative change
    current_max_dd_pct: float
    proposed_max_dd_pct: float
    dd_change_pct: float
    current_win_rate: float
    proposed_win_rate: float
    current_sharpe: float
    proposed_sharpe: float
    current_total_trades: int
    proposed_total_trades: int
    verdict: str                        # "APPROVE" / "REJECT" / "NEEDS_REVIEW"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
