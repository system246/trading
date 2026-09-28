"""
paper_trading.py — Auto paper trading engine v4
Changes:
  1. MIN_CONFIDENCE → 88%
  2. MAX_HOLD_DAYS  → 1.5 days
  3. Fee calculation 0.1% per side
  4. Smart position sizing by confidence
  5. Signal performance tracker
  6. Momentum check before buying
"""

import json
import time
import asyncio
import httpx
import os
from typing import Optional

PAPER_FILE       = "paper_trades.json"
STARTING_BALANCE = 20000.0
MAX_OPEN_TRADES  = 3
MIN_CONFIDENCE   = 92          # raised from 85
MAX_HOLD_DAYS    = 1.5         # reduced from 2.5
REENTRY_HOURS    = 24
CHECK_INTERVAL   = 60
TRAILING_PCT     = 0.05
TAKE_PROFIT_PCT  = 0.06
STOP_LOSS_PCT    = 0.03
FEE_PCT          = 0.001
TDS_PCT          = 0.01     # 1% TDS on sell (Indian law)
INCOME_TAX_PCT   = 0.30     # 30% flat tax on crypto profit (Indian law)       # 0.1% per side (Binance fee)

# Smart position sizing by confidence
# confidence >= X → invest Y% of portfolio
POSITION_SIZES = [
    (95, 0.15),   # 95%+ confidence → 15%
    (90, 0.12),   # 90%+ confidence → 12%
    (88, 0.10),   # 88%+ confidence → 10%
]

# Profit lock levels
PROFIT_LOCKS = [
    (0.30, 0.00),
    (0.50, 0.20),
    (0.80, 0.50),
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
        "balance":          STARTING_BALANCE,
        "starting":         STARTING_BALANCE,
        "open_trades":      [],
        "closed_trades":    [],
        "total_profit":     0.0,
        "auto_enabled":     True,
        "recent_symbols":   {},
        "signal_stats":     {},    # signal performance tracker
        "created_at":       int(time.time() * 1000),
        "last_updated":     int(time.time() * 1000),
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


def _get_position_size(confidence: float) -> float:
    """Smart position sizing based on confidence."""
    for threshold, size in POSITION_SIZES:
        if confidence >= threshold:
            return size
    return 0.10


def _update_signal_stats(state: dict, signal_type: str, won: bool, pnl_pct: float):
    """Track performance per signal type."""
    if "signal_stats" not in state:
        state["signal_stats"] = {}
    if signal_type not in state["signal_stats"]:
        state["signal_stats"][signal_type] = {
            "trades": 0, "wins": 0, "losses": 0,
            "total_pnl": 0.0, "win_rate": 0.0, "avg_pnl": 0.0
        }
    s = state["signal_stats"][signal_type]
    s["trades"]    += 1
    s["wins"]      += 1 if won else 0
    s["losses"]    += 0 if won else 1
    s["total_pnl"] += pnl_pct
    s["win_rate"]   = round(s["wins"] / s["trades"] * 100, 1)
    s["avg_pnl"]    = round(s["total_pnl"] / s["trades"], 2)


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


async def check_momentum(symbol: str) -> bool:
    """
    Momentum check — coin must already be moving up.
    Checks last 3 hourly candles.
    Returns True if upward momentum confirmed.
    """
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            r = await client.get(
                f"{BINANCE_REST}/klines?symbol={symbol}&interval=1h&limit=4"
            )
            if r.status_code != 200:
                return True  # if check fails, don't block
            candles = r.json()
            if len(candles) < 3:
                return True
            # Last 3 closes
            c1 = float(candles[-4][4])  # 3 hours ago
            c2 = float(candles[-3][4])  # 2 hours ago
            c3 = float(candles[-2][4])  # 1 hour ago
            c4 = float(candles[-1][4])  # current
            # At least 2 of last 3 candles must be green
            green = sum([c2 > c1, c3 > c2, c4 > c3])
            return green >= 2
    except:
        return True  # if check fails, don't block


def _calc_trailing_stop(trade: dict, current_price: float):
    entry = float(trade["entry_price"])
    peak  = float(trade.get("peak_price", current_price))

    if current_price > peak:
        peak = current_price

    trailing_stop = round(peak * (1 - TRAILING_PCT), 8)
    current_profit_pct = (current_price - entry) / entry
    locked_stop = float(trade.get("initial_stop", entry * (1 - STOP_LOSS_PCT)))

    for profit_threshold, stop_level in sorted(PROFIT_LOCKS, reverse=True):
        if current_profit_pct >= profit_threshold:
            lock_price  = round(entry * (1 + stop_level), 8)
            locked_stop = max(locked_stop, lock_price)
            break

    final_stop = max(trailing_stop, locked_stop)
    return round(final_stop, 8), round(peak, 8)


async def auto_buy(signal: dict) -> Optional[dict]:
    """
    Auto-buy with:
    - 88% confidence threshold
    - Smart position sizing
    - Fee calculation
    - Momentum check
    """
    state = _load()

    if not state.get("auto_enabled"):                          return None
    if str(signal.get("verdict", "")) != "STRONG_BUY":        return None

    confidence = float(signal.get("confidence", 0))
    if confidence < MIN_CONFIDENCE:                            return None

    symbol = signal["symbol"]

    symbols_open = [t["symbol"] for t in state["open_trades"]]
    if symbol in symbols_open:                                 return None
    if len(state["open_trades"]) >= MAX_OPEN_TRADES:          return None

    recent      = state.get("recent_symbols", {})
    last_closed = recent.get(symbol, 0)
    hours_since = (time.time() * 1000 - last_closed) / 3600000
    if last_closed > 0 and hours_since < REENTRY_HOURS:       return None

    # Momentum check
    has_momentum = await check_momentum(symbol)
    if not has_momentum:
        print(f"Skipping {symbol} — no upward momentum")
        return None

    # Smart position sizing
    position_pct  = _get_position_size(confidence)
    portfolio_val = _portfolio_value(state)
    invest_gross  = round(min(float(state["balance"]), portfolio_val * position_pct), 2)
    if invest_gross < 100:                                     return None

    # Fee deduction on buy
    buy_fee   = round(invest_gross * FEE_PCT, 2)
    invest_net= round(invest_gross - buy_fee, 2)

    price        = float(signal["price"])
    target_price = round(price * (1 + TAKE_PROFIT_PCT), 8)
    stop_price   = round(price * (1 - STOP_LOSS_PCT), 8)
    qty          = round(invest_net / price, 8)
    max_hold_ms  = int(time.time() * 1000) + int(MAX_HOLD_DAYS * 24 * 3600 * 1000)

    signals_list = signal.get("signals", []) or []
    signal_type  = signals_list[0].get("type", "") if signals_list else ""

    trade = {
        "id":             f"PT_{symbol}_{int(time.time())}",
        "symbol":         symbol,
        "name":           str(signal.get("name", symbol.replace("USDT", ""))),
        "entry_price":    price,
        "current_price":  price,
        "peak_price":     price,
        "target_price":   target_price,
        "stop_price":     stop_price,
        "initial_stop":   stop_price,
        "qty":            qty,
        "invested":       invest_gross,
        "invested_net":   invest_net,
        "buy_fee":        buy_fee,
        "current_value":  invest_net,
        "pnl":            0.0,
        "pnl_pct":        0.0,
        "confidence":     confidence,
        "position_pct":   position_pct * 100,
        "signal_type":    signal_type,
        "reason":         str(signal.get("reason", "")),
        "status":         "OPEN",
        "opened_at":      int(time.time() * 1000),
        "max_hold_until": max_hold_ms,
        "closed_at":      None,
        "close_reason":   None,
        "trailing":       True,
    }

    state["balance"] = round(float(state["balance"]) - invest_gross, 2)
    state["open_trades"].append(trade)
    _save(state)
    print(f"Bought {symbol} @ {price} | conf:{confidence}% | size:{position_pct*100}% | fee:₹{buy_fee}")
    return trade


def auto_sell(trade_id: str, current_price: float, reason: str) -> Optional[dict]:
    state = _load()
    trade = next((t for t in state["open_trades"] if t["id"] == trade_id), None)
    if not trade:
        return None

    current_price = float(current_price)

    # Fee deduction on sell
    sell_fee      = round(float(trade["qty"]) * current_price * FEE_PCT, 2)
    sell_value    = round(float(trade["qty"]) * current_price - sell_fee, 2)
    pnl           = round(sell_value - float(trade["invested"]), 2)
    pnl_pct       = round(pnl / float(trade["invested"]) * 100, 2)

    trade.update({
        "current_price": current_price,
        "current_value": sell_value,
        "sell_fee":      sell_fee,
        "total_fees":    round(float(trade.get("buy_fee", 0)) + sell_fee, 2),
        "pnl":           pnl,
        "pnl_pct":       pnl_pct,
        "status":        "WIN" if pnl > 0 else "LOSS",
        "closed_at":     int(time.time() * 1000),
        "close_reason":  reason,
    })

    # Update signal stats
    _update_signal_stats(
        state,
        trade.get("signal_type", "UNKNOWN"),
        pnl > 0,
        pnl_pct
    )

    state["open_trades"]   = [t for t in state["open_trades"] if t["id"] != trade_id]
    state["closed_trades"].insert(0, trade)
    state["balance"]       = round(float(state["balance"]) + sell_value, 2)
    state["total_profit"]  = round(float(state["total_profit"]) + pnl, 2)

    if "recent_symbols" not in state:
        state["recent_symbols"] = {}
    state["recent_symbols"][trade["symbol"]] = int(time.time() * 1000)

    _save(state)
    print(f"Sold {trade['symbol']} | {reason} | {'+' if pnl_pct>0 else ''}{pnl_pct}% | fee:₹{sell_fee}")
    return trade


async def monitor_loop():
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

                    new_stop, new_peak = _calc_trailing_stop(trade, price)

                    sell_fee_est     = round(qty * price * FEE_PCT, 2)
                    current_value    = round(qty * price - sell_fee_est, 2)

                    trade["current_price"] = price
                    trade["current_value"] = current_value
                    trade["pnl"]           = round(current_value - invested, 2)
                    trade["pnl_pct"]       = round(trade["pnl"] / invested * 100, 2)
                    trade["peak_price"]    = new_peak
                    trade["stop_price"]    = new_stop

                    close_reason = None

                    max_hold = trade.get("max_hold_until", 0)
                    if max_hold and int(time.time() * 1000) > max_hold:
                        close_reason = "TIME_LIMIT"
                    elif price <= new_stop:
                        peak = float(trade.get("peak_price", price))
                        if peak > float(trade["entry_price"]) * 1.01:
                            close_reason = "TRAILING_STOP"
                        elif price <= float(trade["initial_stop"]):
                            close_reason = "STOP_LOSS"
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

    win_rate  = round(len(wins) / len(closed) * 100, 1) if closed else 0.0
    avg_win   = round(sum(float(t["pnl_pct"]) for t in wins) / len(wins), 2) if wins else 0.0
    avg_loss  = round(sum(float(t["pnl_pct"]) for t in losses) / len(losses), 2) if losses else 0.0
    portfolio = _portfolio_value(state)
    total_ret = _total_return_pct(state)
    total_pnl = sum(float(t["pnl"]) for t in closed)
    # total_fees= sum(float(t.get("total_fees", 0)) for t in closed)
    total_fees = sum(float(t.get("total_fees",0)) for t in closed)
    total_tds  = sum(float(t.get("tds",0)) for t in closed)
    total_tax  = sum(float(t.get("tax_estimate",0)) for t in closed)
    total_net  = sum(float(t.get("net_profit",0)) for t in closed)

    open_display = []
    for t in open_t:
        td = dict(t)
        max_hold = t.get("max_hold_until", 0)
        if max_hold:
            ms_left = max_hold - int(time.time() * 1000)
            td["days_remaining"] = round(max(0, ms_left / 86400000), 1)
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
        "total_fees_paid":  round(total_fees, 2),
        "total_tds_paid":   round(total_tds, 2),
        "total_tax_est":    round(total_tax, 2),
        "total_net_profit": round(total_net, 2),
        "open_count":       len(open_t),
        "closed_count":     len(closed),
        "win_count":        len(wins),
        "loss_count":       len(losses),
        "win_rate":         win_rate,
        "avg_win_pct":      avg_win,
        "avg_loss_pct":     avg_loss,
        "auto_enabled":     bool(state.get("auto_enabled", True)),
        "signal_stats":     state.get("signal_stats", {}),
        "open_trades":      open_display,
        "closed_trades":    closed[:20],
    }