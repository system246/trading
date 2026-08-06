"""
paper_trading.py — Auto paper trading engine v3
NEW: Trailing stop loss + Profit lock levels
"""

import json
import time
import asyncio
import httpx
import os
from typing import Optional

PAPER_FILE       = "paper_trades.json"
STARTING_BALANCE = 20000.0
MAX_PER_TRADE    = 0.10
MAX_OPEN_TRADES  = 5
MIN_CONFIDENCE   = 85        # raised from 80 to 85
TAKE_PROFIT_PCT  = 0.06
STOP_LOSS_PCT    = 0.03
MAX_HOLD_DAYS    = 2.5
REENTRY_HOURS    = 24
CHECK_INTERVAL   = 60
TRAILING_PCT     = 0.05      # 5% trailing stop

# Profit lock levels
# When profit hits X% → move stop to Y%
PROFIT_LOCKS = [
    (0.30, 0.00),   # at +30% → stop moves to breakeven
    (0.50, 0.20),   # at +50% → stop moves to +20%
    (0.80, 0.50),   # at +80% → stop moves to +50%
]

BINANCE_REST = "https://api.binance.com/api/v3"


def _load() -> dict:
    if os.path.exists(PAPER_FILE):
        try:
            return json.load(open(PAPER_FILE))
        except:
            pass
    return _default_state()


def _default_state() -> dict:
    return {
        "balance":        STARTING_BALANCE,
        "starting":       STARTING_BALANCE,
        "open_trades":    [],
        "closed_trades":  [],
        "total_profit":   0.0,
        "auto_enabled":   True,
        "recent_symbols": {},
        "created_at":     int(time.time() * 1000),
        "last_updated":   int(time.time() * 1000),
    }


def _save(state: dict):
    state["last_updated"] = int(time.time() * 1000)
    try:
        json.dump(state, open(PAPER_FILE, "w"), indent=2)
    except Exception as e:
        print(f"Save error: {e}")


def _portfolio_value(state: dict) -> float:
    val = float(state["balance"])
    for t in state["open_trades"]:
        val += float(t.get("current_value", t["invested"]))
    return round(val, 2)


def _total_return_pct(state: dict) -> float:
    val      = _portfolio_value(state)
    starting = float(state["starting"])
    return round((val - starting) / starting * 100, 2) if starting > 0 else 0.0


def get_state() -> dict:
    return _load()


def toggle_auto(enabled: bool) -> dict:
    state = _load()
    state["auto_enabled"] = bool(enabled)
    _save(state)
    return state


def reset_portfolio() -> dict:
    state = _default_state()
    _save(state)
    return state


async def get_btc_change() -> float:
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{BINANCE_REST}/ticker/24hr?symbol=BTCUSDT")
            if r.status_code == 200:
                return float(r.json().get("priceChangePercent", 0))
    except:
        pass
    return 0.0


async def get_live_price(symbol: str) -> Optional[float]:
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{BINANCE_REST}/ticker/price?symbol={symbol}")
            if r.status_code == 200:
                return float(r.json()["price"])
    except:
        pass
    return None


def _calc_trailing_stop(trade: dict, current_price: float) -> float:
    """
    Calculate trailing stop price.
    Trails 5% below the highest price seen.
    Also applies profit lock levels.
    """
    entry      = float(trade["entry_price"])
    peak       = float(trade.get("peak_price", current_price))
    invested   = float(trade["invested"])

    # Update peak if current price is higher
    if current_price > peak:
        peak = current_price

    # Trailing stop = 5% below peak
    trailing_stop = round(peak * (1 - TRAILING_PCT), 8)

    # Check profit lock levels
    current_profit_pct = (current_price - entry) / entry

    locked_stop = float(trade.get("stop_price", entry * (1 - STOP_LOSS_PCT)))

    for profit_threshold, stop_level in sorted(PROFIT_LOCKS, reverse=True):
        if current_profit_pct >= profit_threshold:
            # Stop must be at least at this level
            lock_price = round(entry * (1 + stop_level), 8)
            locked_stop = max(locked_stop, lock_price)
            break

    # Final stop = highest of trailing and locked
    final_stop = max(trailing_stop, locked_stop)

    return round(final_stop, 8), round(peak, 8)


def auto_buy(signal: dict) -> Optional[dict]:
    state = _load()

    if not state.get("auto_enabled"):           return None
    if str(signal.get("verdict","")) != "STRONG_BUY": return None
    if float(signal.get("confidence", 0)) < MIN_CONFIDENCE: return None

    symbol = signal["symbol"]

    symbols_open = [t["symbol"] for t in state["open_trades"]]
    if symbol in symbols_open: return None
    if len(state["open_trades"]) >= MAX_OPEN_TRADES: return None

    recent      = state.get("recent_symbols", {})
    last_closed = recent.get(symbol, 0)
    hours_since = (time.time() * 1000 - last_closed) / 3600000
    if last_closed > 0 and hours_since < REENTRY_HOURS: return None

    portfolio_val = _portfolio_value(state)
    invest = round(min(float(state["balance"]), portfolio_val * MAX_PER_TRADE), 2)
    if invest < 100: return None

    price        = float(signal["price"])
    target_price = round(price * (1 + TAKE_PROFIT_PCT), 8)
    stop_price   = round(price * (1 - STOP_LOSS_PCT), 8)
    qty          = round(invest / price, 8)
    max_hold_ms  = int(time.time() * 1000) + int(MAX_HOLD_DAYS * 24 * 3600 * 1000)

    signals_list = signal.get("signals", []) or []
    signal_type  = signals_list[0].get("type", "") if signals_list else ""

    trade = {
        "id":            f"PT_{symbol}_{int(time.time())}",
        "symbol":        symbol,
        "name":          str(signal.get("name", symbol.replace("USDT",""))),
        "entry_price":   price,
        "current_price": price,
        "peak_price":    price,       # NEW: track highest price seen
        "target_price":  target_price,
        "stop_price":    stop_price,
        "initial_stop":  stop_price,  # original stop for reference
        "qty":           qty,
        "invested":      invest,
        "current_value": invest,
        "pnl":           0.0,
        "pnl_pct":       0.0,
        "confidence":    float(signal.get("confidence", 0)),
        "signal_type":   signal_type,
        "reason":        str(signal.get("reason", "")),
        "status":        "OPEN",
        "opened_at":     int(time.time() * 1000),
        "max_hold_until":max_hold_ms,
        "closed_at":     None,
        "close_reason":  None,
        "trailing":      True,        # trailing stop enabled
    }

    state["balance"] = round(float(state["balance"]) - invest, 2)
    state["open_trades"].append(trade)
    _save(state)
    return trade


def auto_sell(trade_id: str, current_price: float, reason: str) -> Optional[dict]:
    state = _load()
    trade = next((t for t in state["open_trades"] if t["id"] == trade_id), None)
    if not trade: return None

    current_price = float(current_price)
    sell_value    = round(float(trade["qty"]) * current_price, 2)
    pnl           = round(sell_value - float(trade["invested"]), 2)
    pnl_pct       = round(pnl / float(trade["invested"]) * 100, 2)

    trade.update({
        "current_price": current_price,
        "current_value": sell_value,
        "pnl":           pnl,
        "pnl_pct":       pnl_pct,
        "status":        "WIN" if pnl > 0 else "LOSS",
        "closed_at":     int(time.time() * 1000),
        "close_reason":  reason,
    })

    state["open_trades"]  = [t for t in state["open_trades"] if t["id"] != trade_id]
    state["closed_trades"].insert(0, trade)
    state["balance"]      = round(float(state["balance"]) + sell_value, 2)
    state["total_profit"] = round(float(state["total_profit"]) + pnl, 2)

    if "recent_symbols" not in state:
        state["recent_symbols"] = {}
    state["recent_symbols"][trade["symbol"]] = int(time.time() * 1000)

    _save(state)
    return trade


async def monitor_loop():
    """
    Background loop — runs every 60s.
    Trailing stop follows price up.
    Profit locks protect gains.
    """
    while True:
        try:
            state = _load()

            if state["open_trades"]:
                btc_change = await get_btc_change()
                market_bad = btc_change < -5.0

                for trade in list(state["open_trades"]):
                    price = await get_live_price(trade["symbol"])
                    if not price:
                        continue

                    price    = float(price)
                    invested = float(trade["invested"])
                    qty      = float(trade["qty"])

                    # Calculate trailing stop + profit locks
                    new_stop, new_peak = _calc_trailing_stop(trade, price)

                    # Update trade values
                    trade["current_price"] = price
                    trade["current_value"] = round(qty * price, 2)
                    trade["pnl"]           = round(trade["current_value"] - invested, 2)
                    trade["pnl_pct"]       = round(trade["pnl"] / invested * 100, 2)
                    trade["peak_price"]    = new_peak
                    trade["stop_price"]    = new_stop  # trailing stop updates here

                    close_reason = None

                    # Time limit
                    max_hold = trade.get("max_hold_until", 0)
                    if max_hold and int(time.time() * 1000) > max_hold:
                        close_reason = "TIME_LIMIT"

                    # Trailing stop hit
                    elif price <= new_stop:
                        # Only sell if we've moved up at least 1% first
                        # (avoids selling on initial dip)
                        peak = float(trade.get("peak_price", price))
                        if peak > float(trade["entry_price"]) * 1.01:
                            close_reason = "TRAILING_STOP"
                        elif price <= float(trade["initial_stop"]):
                            close_reason = "STOP_LOSS"

                    # BTC crash
                    elif market_bad:
                        close_reason = "MARKET_CRASH"

                    if close_reason:
                        auto_sell(trade["id"], price, close_reason)
                    else:
                        _save(state)

                    await asyncio.sleep(0.5)

        except Exception as e:
            print(f"Monitor error: {e}")

        await asyncio.sleep(CHECK_INTERVAL)


def get_stats() -> dict:
    state  = _load()
    closed = state["closed_trades"]
    wins   = [t for t in closed if t["status"] == "WIN"]
    losses = [t for t in closed if t["status"] == "LOSS"]
    open_t = state["open_trades"]

    win_rate  = round(len(wins)/len(closed)*100, 1) if closed else 0.0
    avg_win   = round(sum(float(t["pnl_pct"]) for t in wins)/len(wins), 2) if wins else 0.0
    avg_loss  = round(sum(float(t["pnl_pct"]) for t in losses)/len(losses), 2) if losses else 0.0
    portfolio = _portfolio_value(state)
    total_ret = _total_return_pct(state)
    total_pnl = sum(float(t["pnl"]) for t in closed)

    open_display = []
    for t in open_t:
        td = dict(t)
        max_hold = t.get("max_hold_until", 0)
        if max_hold:
            ms_left = max_hold - int(time.time() * 1000)
            td["days_remaining"] = round(max(0, ms_left / 86400000), 1)
        # Show how much profit is locked
        entry = float(t.get("entry_price", 0))
        stop  = float(t.get("stop_price", 0))
        if entry > 0:
            td["locked_profit_pct"] = round((stop - entry) / entry * 100, 2)
        open_display.append(td)

    return {
        "balance":          round(float(state["balance"]), 2),
        "starting":         float(state["starting"]),
        "portfolio_value":  portfolio,
        "total_return_pct": total_ret,
        "total_profit":     round(total_pnl, 2),
        "open_count":       len(open_t),
        "closed_count":     len(closed),
        "win_count":        len(wins),
        "loss_count":       len(losses),
        "win_rate":         win_rate,
        "avg_win_pct":      avg_win,
        "avg_loss_pct":     avg_loss,
        "auto_enabled":     bool(state.get("auto_enabled", True)),
        "open_trades":      open_display,
        "closed_trades":    closed[:20],
    }