"""
real_trading.py — Real money trading via Binance API
Features:
- Manual approval mode (safe for starting)
- Emergency stop
- Real order execution
- BNB fee check
- Full P&L tracking
"""

import json
import time
import hmac
import hashlib
import asyncio
import httpx
import os
from typing import Optional

REAL_FILE        = "real_trades.json"
BINANCE_REST     = "https://api.binance.com"
FEE_PCT          = 0.00075   # 0.075% with BNB discount
MIN_CONFIDENCE   = 92
MAX_PER_TRADE    = 0.10      # 10% per trade
MAX_OPEN_TRADES  = 3
STOP_LOSS_PCT    = 0.03
TAKE_PROFIT_PCT  = 0.06
TRAILING_PCT     = 0.05
CHECK_INTERVAL   = 15        # 15 second price checks

# Get keys from Railway environment variables
API_KEY    = os.environ.get("BINANCE_API_KEY", "")
SECRET_KEY = os.environ.get("BINANCE_SECRET_KEY", "")


def _sign(params: str) -> str:
    return hmac.new(
        SECRET_KEY.encode("utf-8"),
        params.encode("utf-8"),
        hashlib.sha256
    ).hexdigest()


def _headers() -> dict:
    return {
        "X-MBX-APIKEY": API_KEY,
        "Content-Type": "application/x-www-form-urlencoded",
    }


def _load() -> dict:
    if os.path.exists(REAL_FILE):
        try: return json.load(open(REAL_FILE))
        except: pass
    return _default_state()


def _default_state() -> dict:
    return {
        "enabled":        False,    # OFF by default — manual start
        "approval_mode":  True,     # manual approval required
        "emergency_stop": False,
        "open_trades":    [],
        "closed_trades":  [],
        "pending_signals":[],       # waiting for manual approval
        "total_profit":   0.0,
        "total_fees":     0.0,
        "created_at":     int(time.time() * 1000),
        "last_updated":   int(time.time() * 1000),
    }


def _save(state: dict):
    state["last_updated"] = int(time.time() * 1000)
    try: json.dump(state, open(REAL_FILE, "w"), indent=2)
    except Exception as e: print(f"Save error: {e}")


def get_state() -> dict:
    return _load()


def toggle_real_trading(enabled: bool) -> dict:
    state = _load()
    state["enabled"] = bool(enabled)
    state["emergency_stop"] = False
    _save(state)
    return state


def emergency_stop() -> dict:
    """Stop all real trading immediately."""
    state = _load()
    state["emergency_stop"] = True
    state["enabled"]        = False
    state["pending_signals"]= []
    _save(state)
    print("🚨 EMERGENCY STOP ACTIVATED")
    return state


def toggle_approval_mode(manual: bool) -> dict:
    state = _load()
    state["approval_mode"] = bool(manual)
    _save(state)
    return state


async def get_account_balance() -> dict:
    """Get real Binance account balance."""
    if not API_KEY or not SECRET_KEY:
        return {"error": "API keys not configured"}
    try:
        ts     = int(time.time() * 1000)
        params = f"timestamp={ts}"
        sig    = _sign(params)
        url    = f"{BINANCE_REST}/api/v3/account?{params}&signature={sig}"
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(url, headers=_headers())
            if r.status_code == 200:
                data     = r.json()
                balances = {
                    b["asset"]: float(b["free"])
                    for b in data.get("balances", [])
                    if float(b["free"]) > 0
                }
                usdt = balances.get("USDT", 0)
                bnb  = balances.get("BNB", 0)
                return {
                    "usdt":     round(usdt, 2),
                    "bnb":      round(bnb, 4),
                    "bnb_ok":   bool(bnb > 0.01),
                    "balances": {k:v for k,v in balances.items() if v > 0.001},
                }
            return {"error": f"Binance error {r.status_code}: {r.text}"}
    except Exception as e:
        return {"error": str(e)}


async def get_symbol_price(symbol: str) -> Optional[float]:
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{BINANCE_REST}/api/v3/ticker/price?symbol={symbol}")
            if r.status_code == 200:
                return float(r.json()["price"])
    except: pass
    return None


async def get_lot_size(symbol: str) -> dict:
    """Get symbol trading rules — min qty, step size etc."""
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            r = await client.get(f"{BINANCE_REST}/api/v3/exchangeInfo?symbol={symbol}")
            if r.status_code == 200:
                data    = r.json()
                filters = data["symbols"][0]["filters"]
                lot     = next((f for f in filters if f["filterType"]=="LOT_SIZE"), {})
                notional= next((f for f in filters if f["filterType"]=="MIN_NOTIONAL"), {})
                return {
                    "min_qty":    float(lot.get("minQty", 0)),
                    "step_size":  float(lot.get("stepSize", 0)),
                    "min_notional": float(notional.get("minNotional", 10)),
                }
    except: pass
    return {"min_qty":0, "step_size":0.001, "min_notional":10}


def _round_qty(qty: float, step_size: float) -> float:
    """Round quantity to valid step size."""
    if step_size == 0: return qty
    precision = len(str(step_size).rstrip('0').split('.')[-1])
    return round(round(qty / step_size) * step_size, precision)


async def place_real_order(symbol: str, side: str, qty: float) -> dict:
    """Place a real market order on Binance."""
    if not API_KEY or not SECRET_KEY:
        return {"error": "API keys not configured"}
    try:
        ts     = int(time.time() * 1000)
        params = f"symbol={symbol}&side={side}&type=MARKET&quantity={qty}&timestamp={ts}"
        sig    = _sign(params)
        url    = f"{BINANCE_REST}/api/v3/order"
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.post(
                url,
                content=f"{params}&signature={sig}",
                headers=_headers()
            )
            data = r.json()
            if r.status_code == 200:
                filled_qty   = float(data.get("executedQty", qty))
                filled_price = float(data.get("cummulativeQuoteQty", 0)) / filled_qty if filled_qty > 0 else 0
                return {
                    "success":     True,
                    "order_id":    data.get("orderId"),
                    "qty":         filled_qty,
                    "price":       filled_price,
                    "status":      data.get("status"),
                }
            return {"error": data.get("msg", f"Order failed {r.status_code}")}
    except Exception as e:
        return {"error": str(e)}


async def add_pending_signal(signal: dict) -> dict:
    """Add signal to pending approval queue."""
    state = _load()
    if not state.get("enabled") or state.get("emergency_stop"):
        return {"skipped": "Trading disabled or emergency stop"}

    symbol = signal["symbol"]

    # Skip if already open or pending
    open_syms    = [t["symbol"] for t in state["open_trades"]]
    pending_syms = [s["symbol"] for s in state.get("pending_signals", [])]
    if symbol in open_syms or symbol in pending_syms:
        return {"skipped": "Already open or pending"}

    if len(state["open_trades"]) >= MAX_OPEN_TRADES:
        return {"skipped": "Max trades reached"}

    pending = {
        "id":         f"REAL_{symbol}_{int(time.time())}",
        "symbol":     symbol,
        "name":       signal.get("name", symbol.replace("USDT","")),
        "price":      float(signal.get("price", 0)),
        "confidence": float(signal.get("confidence", 0)),
        "signal_type":signal.get("signals",[{}])[0].get("type","") if signal.get("signals") else "",
        "reason":     str(signal.get("reason","")),
        "added_at":   int(time.time() * 1000),
        "expires_at": int(time.time() * 1000) + 3600000,  # 1 hour to approve
    }

    if "pending_signals" not in state:
        state["pending_signals"] = []
    state["pending_signals"].append(pending)
    _save(state)
    print(f"📋 Pending approval: {symbol} @ {signal.get('price')}")
    return {"pending": True, "signal": pending}


async def approve_signal(signal_id: str, invest_amount: float) -> dict:
    """Manually approve a pending signal and place real order."""
    state = _load()
    if state.get("emergency_stop"):
        return {"error": "Emergency stop is active"}

    signal = next((s for s in state.get("pending_signals",[]) if s["id"] == signal_id), None)
    if not signal:
        return {"error": "Signal not found or expired"}

    # Check balance
    balance = await get_account_balance()
    if "error" in balance:
        return {"error": balance["error"]}

    usdt = float(balance.get("usdt", 0))
    if usdt < invest_amount:
        return {"error": f"Insufficient USDT. Have {usdt:.2f}, need {invest_amount:.2f}"}

    if not balance.get("bnb_ok"):
        print("⚠ Low BNB balance — fees will be deducted from USDT")

    # Get current price and lot size
    symbol   = signal["symbol"]
    price    = await get_symbol_price(symbol)
    if not price:
        return {"error": "Could not get current price"}

    lot_info = await get_lot_size(symbol)
    raw_qty  = invest_amount / price
    qty      = _round_qty(raw_qty, lot_info["step_size"])

    if qty * price < lot_info["min_notional"]:
        return {"error": f"Order too small. Min: ${lot_info['min_notional']}"}

    # Place real BUY order
    order = await place_real_order(symbol, "BUY", qty)
    if "error" in order:
        return {"error": order["error"]}

    filled_price = order.get("price") or price
    fee          = round(invest_amount * FEE_PCT, 4)

    trade = {
        "id":            signal_id,
        "order_id":      order.get("order_id"),
        "symbol":        symbol,
        "name":          signal["name"],
        "entry_price":   filled_price,
        "current_price": filled_price,
        "peak_price":    filled_price,
        "target_price":  round(filled_price * (1 + TAKE_PROFIT_PCT), 8),
        "stop_price":    round(filled_price * (1 - STOP_LOSS_PCT), 8),
        "initial_stop":  round(filled_price * (1 - STOP_LOSS_PCT), 8),
        "partial_price": round(filled_price * 1.04, 8),
        "partial_taken": False,
        "qty":           float(order.get("qty", qty)),
        "invested":      invest_amount,
        "buy_fee":       fee,
        "current_value": invest_amount,
        "pnl":           0.0,
        "pnl_pct":       0.0,
        "confidence":    signal["confidence"],
        "signal_type":   signal["signal_type"],
        "reason":        signal["reason"],
        "status":        "OPEN",
        "opened_at":     int(time.time() * 1000),
        "max_hold_until":int(time.time()*1000) + int(1.5 * 24 * 3600 * 1000),
        "closed_at":     None,
        "close_reason":  None,
    }

    # Remove from pending, add to open
    state["pending_signals"] = [s for s in state.get("pending_signals",[]) if s["id"] != signal_id]
    state["open_trades"].append(trade)
    _save(state)
    print(f"✅ REAL BUY: {symbol} @ {filled_price} | qty:{qty} | ₹{invest_amount}")
    return {"success": True, "trade": trade}


def reject_signal(signal_id: str) -> dict:
    """Reject a pending signal."""
    state = _load()
    state["pending_signals"] = [s for s in state.get("pending_signals",[]) if s["id"] != signal_id]
    _save(state)
    return {"rejected": True}


async def close_real_trade(trade_id: str, reason: str) -> dict:
    """Sell a real trade."""
    state = _load()
    trade = next((t for t in state["open_trades"] if t["id"] == trade_id), None)
    if not trade: return {"error": "Trade not found"}

    symbol = trade["symbol"]
    qty    = float(trade["qty"])

    # Place real SELL order
    order = await place_real_order(symbol, "SELL", qty)
    if "error" in order:
        print(f"Sell error: {order['error']}")
        sell_price = await get_symbol_price(symbol) or float(trade["current_price"])
    else:
        sell_price = order.get("price") or float(trade["current_price"])

    sell_fee   = round(qty * sell_price * FEE_PCT, 4)
    sell_value = round(qty * sell_price - sell_fee, 2)
    pnl        = round(sell_value - float(trade["invested"]), 2)
    pnl_pct    = round(pnl / float(trade["invested"]) * 100, 2)

    trade.update({
        "current_price": sell_price,
        "current_value": sell_value,
        "sell_fee":      sell_fee,
        "total_fees":    round(float(trade.get("buy_fee",0)) + sell_fee, 4),
        "pnl":           pnl,
        "pnl_pct":       pnl_pct,
        "status":        "WIN" if pnl > 0 else "LOSS",
        "closed_at":     int(time.time() * 1000),
        "close_reason":  reason,
    })

    state["open_trades"]   = [t for t in state["open_trades"] if t["id"] != trade_id]
    state["closed_trades"].insert(0, trade)
    state["total_profit"]  = round(float(state.get("total_profit",0)) + pnl, 2)
    state["total_fees"]    = round(float(state.get("total_fees",0)) + trade["total_fees"], 4)
    _save(state)
    print(f"{'✅' if pnl>0 else '❌'} REAL SELL: {symbol} | {reason} | {'+' if pnl_pct>0 else ''}{pnl_pct}%")
    return {"success": True, "trade": trade}


async def monitor_real_trades():
    """Monitor real open trades — trailing stop, targets."""
    while True:
        try:
            state = _load()
            if state.get("emergency_stop") or not state.get("enabled"):
                await asyncio.sleep(CHECK_INTERVAL)
                continue

            for trade in list(state["open_trades"]):
                price = await get_symbol_price(trade["symbol"])
                if not price: continue

                price    = float(price)
                invested = float(trade["invested"])
                qty      = float(trade["qty"])

                # Trailing stop calculation
                peak = float(trade.get("peak_price", price))
                if price > peak: peak = price
                trailing_stop = round(peak * (1 - TRAILING_PCT), 8)
                initial_stop  = float(trade.get("initial_stop", price * (1-STOP_LOSS_PCT)))
                new_stop      = max(trailing_stop, initial_stop)

                # Update values
                fee_est = round(qty * price * FEE_PCT, 4)
                trade["current_price"] = price
                trade["current_value"] = round(qty * price - fee_est, 2)
                trade["pnl"]           = round(trade["current_value"] - invested, 2)
                trade["pnl_pct"]       = round(trade["pnl"] / invested * 100, 2) if invested > 0 else 0
                trade["peak_price"]    = peak
                trade["stop_price"]    = new_stop

                close_reason = None

                # Time limit
                if int(time.time()*1000) > trade.get("max_hold_until", 0):
                    close_reason = "TIME_LIMIT"
                # Trailing stop
                elif price <= new_stop:
                    if peak > float(trade["entry_price"]) * 1.01:
                        close_reason = "TRAILING_STOP"
                    elif price <= initial_stop:
                        close_reason = "STOP_LOSS"

                if close_reason:
                    await close_real_trade(trade["id"], close_reason)
                else:
                    _save(state)

                await asyncio.sleep(0.5)

            # Clean expired pending signals
            state = _load()
            now = int(time.time() * 1000)
            state["pending_signals"] = [
                s for s in state.get("pending_signals",[])
                if s.get("expires_at", now+1) > now
            ]
            _save(state)

        except Exception as e:
            print(f"Real monitor error: {e}")

        await asyncio.sleep(CHECK_INTERVAL)


def get_real_stats() -> dict:
    state  = _load()
    closed = state.get("closed_trades", [])
    open_t = state.get("open_trades", [])
    wins   = [t for t in closed if t.get("status") == "WIN"]
    losses = [t for t in closed if t.get("status") == "LOSS"]

    return {
        "enabled":        bool(state.get("enabled", False)),
        "approval_mode":  bool(state.get("approval_mode", True)),
        "emergency_stop": bool(state.get("emergency_stop", False)),
        "open_count":     len(open_t),
        "closed_count":   len(closed),
        "win_count":      len(wins),
        "loss_count":     len(losses),
        "win_rate":       round(len(wins)/len(closed)*100, 1) if closed else 0,
        "total_profit":   round(float(state.get("total_profit",0)), 2),
        "total_fees":     round(float(state.get("total_fees",0)), 4),
        "avg_win":        round(sum(float(t.get("pnl_pct",0)) for t in wins)/len(wins), 2) if wins else 0,
        "avg_loss":       round(sum(float(t.get("pnl_pct",0)) for t in losses)/len(losses), 2) if losses else 0,
        "pending_signals":state.get("pending_signals", []),
        "open_trades":    open_t,
        "closed_trades":  closed[:20],
    }