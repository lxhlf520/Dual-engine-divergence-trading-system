import json, time, base64, hmac, hashlib, urllib.parse
from datetime import datetime, timezone
from typing import Optional, Dict, Tuple
import requests
from ..config import SYMBOL, DEFAULT_LEVERAGE, MARGIN_MODE, API_KEY, API_SECRET, API_PASSWORD, HTTP_PROXY, HTTPS_PROXY, LIVE_PYRAMIDING
from ..logger import get_logger
from .alerts import AlertManager

logger = get_logger(__name__)
BASE_URL = "https://www.okx.com"
INST_ID = "BTC-USDT-SWAP"

PROXIES = None
if HTTP_PROXY or HTTPS_PROXY:
    PROXIES = {}
    if HTTP_PROXY:
        PROXIES["http"] = HTTP_PROXY
    if HTTPS_PROXY:
        PROXIES["https"] = HTTPS_PROXY

def _sign(ts, method, path, body=""):
    return base64.b64encode(hmac.new(API_SECRET.encode(), (ts+method+path+body).encode(), hashlib.sha256).digest()).decode()

def _headers(method, path, body=""):
    now = datetime.now(timezone.utc)
    ts = now.strftime('%Y-%m-%dT%H:%M:%S.') + str(int(now.microsecond / 1000)).zfill(3) + 'Z'
    return {"OK-ACCESS-KEY": API_KEY, "OK-ACCESS-SIGN": _sign(ts, method, path, body), "OK-ACCESS-TIMESTAMP": ts, "OK-ACCESS-PASSPHRASE": API_PASSWORD, "Content-Type": "application/json"}

def okx_get(path, params=None):
    sign_path = path
    if params:
        qs = urllib.parse.urlencode(sorted(params.items()))
        sign_path = path + "?" + qs
    r = requests.get(BASE_URL+path, headers=_headers("GET", sign_path, ""), params=params, proxies=PROXIES, timeout=15)
    return r.json()

def okx_post(path, body):
    bs = json.dumps(body)
    r = requests.post(BASE_URL+path, headers=_headers("POST", path, bs), data=bs, proxies=PROXIES, timeout=15)
    return r.json()

class OKXExecution:
    def __init__(self, alert=None, max_risk_pct=15.0, max_capital_pct=70.0, max_contracts=5,
                 leverage=None):
        self.alert = alert or AlertManager()
        self._last_sl_order_id = None
        self._leverage = leverage if leverage is not None else DEFAULT_LEVERAGE
        self.max_risk_pct = max_risk_pct
        self.max_capital_pct = max_capital_pct
        self.max_contracts = max_contracts

    def get_balance(self):
        data = okx_get("/api/v5/account/balance")
        if data.get("code") != "0":
            return {"total_equity": 0, "free": 0, "used": 0}
        usdt = {"availBal": "0", "frozenBal": "0", "eq": "0"}
        for d in data["data"][0].get("details", []):
            if d.get("ccy") == "USDT":
                usdt = d
                break
        return {"total_equity": float(usdt.get("eq", 0)), "free": float(usdt.get("availBal", 0)), "used": float(usdt.get("frozenBal", 0))}

    def get_position(self):
        data = okx_get("/api/v5/account/positions", {"instId": INST_ID})
        if data.get("code") != "0":
            return None
        for p in data.get("data", []):
            if float(p.get("pos", 0)) != 0:
                return {"side": p.get("posSide", ""), "contracts": p.get("pos", "0"), "entry_price": float(p.get("avgPx", 0)), "unrealized_pnl": float(p.get("upl", 0)), "leverage": p.get("lever", "0"), "margin": p.get("margin", "0"), "liq_price": p.get("liqPx", "0")}
        return None

    def get_open_orders(self):
        data = okx_get("/api/v5/trade/orders-pending", {"instId": INST_ID, "instType": "SWAP"})
        return [] if data.get("code") != "0" else data.get("data", [])

    def get_sizing_report(self):
        bal = self.get_balance()
        ticker = okx_get("/api/v5/market/ticker", {"instId": INST_ID})
        price = float(ticker["data"][0]["last"]) if ticker.get("code") == "0" else 0
        ct_val = 0.01
        notional = price * ct_val
        margin_per_ct = notional / self._leverage
        equity = bal["total_equity"]
        return {"equity": round(equity, 2), "free": round(bal["free"], 2), "leverage": self._leverage, "price": round(price, 1), "notional_per_ct": round(notional, 2), "margin_per_ct": round(margin_per_ct, 2), "max_risk_pct": self.max_risk_pct, "max_capital_pct": self.max_capital_pct, "max_contracts": self.max_contracts}

    def _compute_safe_contracts(self, entry_price, stop_loss_price):
        bal = self.get_balance()
        equity, free = bal["total_equity"], bal["free"]
        ct_val, lev = 0.01, self._leverage
        notional_per_ct = entry_price * ct_val
        margin_per_ct = notional_per_ct / lev
        max_margin = equity * (self.max_capital_pct / 100)
        ct_by_margin = max_margin / margin_per_ct if margin_per_ct > 0 else 0
        sl_dist = abs(stop_loss_price - entry_price) / entry_price
        risk_per_ct = notional_per_ct * sl_dist
        max_risk_usd = equity * (self.max_risk_pct / 100)
        ct_by_risk = max_risk_usd / risk_per_ct if risk_per_ct > 0 else 0
        ct_by_free = free / margin_per_ct if margin_per_ct > 0 else 0
        raw_ct = min(ct_by_margin, ct_by_risk, ct_by_free, float(self.max_contracts))
        sz = int(raw_ct * 100) / 100.0
        actual_notional = sz * notional_per_ct
        actual_margin = actual_notional / lev
        actual_risk = sz * risk_per_ct
        report = {"equity": round(equity, 2), "free": round(free, 2), "leverage": lev, "entry_price": round(entry_price, 1), "stop_loss": round(stop_loss_price, 1), "sl_distance_pct": round(sl_dist * 100, 2), "margin_per_ct": round(margin_per_ct, 2), "risk_per_ct": round(risk_per_ct, 2), "max_risk_usd": round(max_risk_usd, 2), "ct_by_margin": round(ct_by_margin, 3), "ct_by_risk": round(ct_by_risk, 3), "ct_by_free": round(ct_by_free, 3), "final_ct": round(sz, 2), "actual_notional": round(actual_notional, 2), "actual_margin": round(actual_margin, 2), "actual_risk_usd": round(actual_risk, 2), "blocked": sz < 0.01, "block_reason": ""}
        if sz < 0.01:
            report["block_reason"] = "margin short"
        return sz, report

    def set_leverage(self, leverage=DEFAULT_LEVERAGE):
        self._leverage = leverage
        data = okx_post("/api/v5/account/set-leverage", {"instId": INST_ID, "lever": str(leverage), "mgnMode": "cross"})
        ok = data.get("code") == "0"
        logger.info("Leverage: {}x cross (OK={})".format(leverage, ok))
        return ok

    def _ct_to_sz(self, contracts):
        return "{:.2f}".format(contracts)

    def open_short(self, stop_loss_price):
        try:
            ticker = okx_get("/api/v5/market/ticker", {"instId": INST_ID})
            if ticker.get("code") != "0":
                return False, "Ticker failed", {}
            price = float(ticker["data"][0]["last"])

            # 检查现有持仓，如果是加仓场景则增量开仓
            existing_pos = self.get_position()
            existing_ct = int(float(existing_pos["contracts"])) if existing_pos and existing_pos["side"] == "short" else 0
            total_max = int(self.max_contracts * LIVE_PYRAMIDING) if hasattr(self, 'max_contracts') else self.max_contracts

            # 计算单笔开仓量（不超过 pyramiding 上限）
            margin_budget = self.get_balance()["total_equity"] * (self.max_capital_pct / 100 if hasattr(self, 'max_capital_pct') else 0.5)
            # 这一笔最大能开多少（考虑已持仓）
            allowed_new = max(0, int(total_max) - existing_ct)
            if allowed_new == 0:
                return False, "BLOCKED: pyramiding limit", {}

            # 按风险计算可开张数（单笔）
            contracts, srep = self._compute_safe_contracts(price, stop_loss_price)
            contracts = min(contracts, float(allowed_new))
            if contracts == 0:
                msg = "BLOCKED: {}".format(srep.get("block_reason", "margin"))
                logger.warning(msg)
                self.alert.error_alert("Position Sizing", msg)
                return False, msg, srep
            self.set_leverage(self._leverage)
            sz = self._ct_to_sz(contracts)
            logger.info("SHORT {}ct sz={} @ ~{:.0f} SL={:.0f} (existing={} total={})".format(
                contracts, sz, price, stop_loss_price, existing_ct, existing_ct + contracts))
            order = okx_post("/api/v5/trade/order", {"instId": INST_ID, "tdMode": "cross", "posSide": "short", "side": "sell", "ordType": "market", "sz": sz})
            if order.get("code") != "0":
                return False, "Order failed: {}".format(order.get("msg")), srep
            sl_id = self._place_stop_loss("buy", existing_ct + contracts, stop_loss_price)
            if sl_id:
                self._last_sl_order_id = sl_id
            return True, "Short {}ct (+{}pyra) SL={:.0f}".format(existing_ct + contracts, contracts, stop_loss_price), srep
        except Exception as e:
            logger.error("open_short: {}".format(e))
            return False, str(e), {}

    def _place_stop_loss(self, side, contracts, sl_price):
        try:
            tp = round(sl_price, 1)
            ps = "short" if side == "buy" else "long"
            sz = self._ct_to_sz(contracts)
            data = okx_post("/api/v5/trade/order-algo", {"instId": INST_ID, "tdMode": "cross", "posSide": ps, "side": side, "ordType": "conditional", "sz": sz, "slTriggerPx": str(tp), "slOrdPx": "-1"})
            if data.get("code") == "0":
                aid = data["data"][0].get("algoId", "")
                logger.info("SL placed: {}".format(aid))
                return aid
            logger.error("SL failed: {}".format(data))
            return None
        except Exception as e:
            logger.error("_place_stop_loss: {}".format(e))
            return None

    def update_stop_loss(self, new_sl_price, contracts=0):
        try:
            if self._last_sl_order_id:
                self._cancel_algo(self._last_sl_order_id)
            pos = self.get_position()
            if not pos:
                return True
            ps = "buy" if pos["side"] == "short" else "sell"
            ct = int(float(pos["contracts"])) if not contracts else contracts
            sl_id = self._place_stop_loss(ps, ct, new_sl_price)
            if sl_id:
                self._last_sl_order_id = sl_id
                return True
            return False
        except Exception as e:
            logger.error("update_stop_loss: {}".format(e))
            return False

    def _cancel_algo(self, algo_id):
        data = okx_post("/api/v5/trade/cancel-algos", [{"instId": INST_ID, "algoId": algo_id}])
        return data.get("code") == "0"

    def cancel_all_algos(self):
        data = okx_post("/api/v5/trade/cancel-algos", [{"instId": INST_ID}])
        if data.get("code") == "0":
            self._last_sl_order_id = None
        return data.get("code") == "0"

    def close_position(self):
        try:
            pos = self.get_position()
            if not pos:
                return True, "No position"
            ct, ps = int(float(pos["contracts"])), pos["side"]
            cs = "buy" if ps == "short" else "sell"
            sz = self._ct_to_sz(ct)
            order = okx_post("/api/v5/trade/order", {"instId": INST_ID, "tdMode": "cross", "posSide": ps, "side": cs, "ordType": "market", "sz": sz})
            if order.get("code") != "0":
                return False, "Close failed: {}".format(order.get("msg"))
            self.cancel_all_algos()
            return True, "Closed"
        except Exception as e:
            return False, str(e)

    def get_status(self):
        bal = self.get_balance()
        pos = self.get_position()
        orders = self.get_open_orders()
        sizing = self.get_sizing_report()
        return {"balance_total": bal["total_equity"], "balance_free": bal["free"], "has_position": pos is not None, "position_side": pos["side"] if pos else "none", "position_size": pos["contracts"] if pos else 0, "unrealized_pnl": pos["unrealized_pnl"] if pos else 0, "open_orders": len(orders), "last_sl_algo_id": self._last_sl_order_id, "sizing": sizing}

    def close(self):
        pass
