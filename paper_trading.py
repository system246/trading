"""
paper_trading.py — FINAL VERSION
Fixes:
  1. Stop loss checks every 15s (was 60s) — prevents DOLO-like losses
  2. Stablecoin/EUR filter
  3. Loss streak pause (3 losses → pause 6h)
  4. Partial profit at +4% (sell 50%)
  5. Fee calculation fixed
  6. Smart position sizing
  7. Trailing stop
  8. Profit locks
"""

import json
import time
import asyncio
import httpx
import os
from typing import Optional

PAPER_FILE       = "paper_trades.json"
STARTING_BALANCE = 20000.0
MAX_OPEN_TRADES  = 5
MIN_CONFIDENCE   = 88
MAX_HOLD_DAYS    = 1.5
REENTRY_HOURS    = 24
PRICE_CHECK_SEC  = 15        # FIX 1: check every 15s instead of 60s
TRAILING_PCT     = 0.05
TAKE_PROFIT_PCT  = 0.06
STOP_LOSS_PCT    = 0.03
FEE_PCT          = 0.001
PARTIAL_PROFIT   = 0.04      # FIX 4: take 50% profit at +4%
MAX_LOSS_STREAK  = 3         # FIX 3: pause after 3 losses
PAUSE_HOURS      = 6         # FIX 3: pause 6 hours

# FIX 2: Skip these — stablecoins/pegged assets
SKIP_COINS = {
    "EUR","USDC","USDT","BUSD","DAI","TUSD","USDP","FDUSD",
    "PAXG","XAUT","WBTC","STETH","WSTETH","FRAX","LUSD",
}

POSITION_SIZES = [
    (95, 0.15),
    (90, 0.12),
    (88, 0.10),
]

PROFIT_LOCKS = [
    (0.30, 0.00),
    (0.50, 0.20),
    (0.80, 0.50),
]

BINANCE_REST = "https://api.binance.com/api/v3"


def _load() -> dict:
    if os.path.exists(PAPER_FILE):
        try: return json.load(open(PAPER_FILE))
        except: pass
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
        "signal_stats":   {},
        "loss_streak":    0,           # FIX 3
        "pause_until":    0,           # FIX 3
        "created_at":     int(time.time() * 1000),
        "last_updated":   int(time.time() * 1000),
    }


def _save(state: dict):
    state["last_updated"] = int(time.time() * 1000)
    try: json.dump(state, open(PAPER_FILE, "w"), indent=2)
    except Exception as e: print(f"Save error: {e}")


def _portfolio_value(state: dict) -> float:
    val = float(state["balance"])
    for t in state["open_trades"]:
        val += float(t.get("current_value", t["invested"]))
    return round(val, 2)


def _total_return_pct(state: dict) -> float:
    val = _portfolio_value(state)
    starting = float(state["starting"])
    return round((val - starting) / starting * 100, 2) if starting > 0 else 0.0


def _get_position_size(confidence: float) -> float:
    for threshold, size in POSITION_SIZES:
        if confidence >= threshold: return size
    return 0.10


def _update_signal_stats(state, signal_type, won, pnl_pct):
    if "signal_stats" not in state: state["signal_stats"] = {}
    if signal_type not in state["signal_stats"]:
        state["signal_stats"][signal_type] = {"trades":0,"wins":0,"losses":0,"total_pnl":0.0,"win_rate":0.0,"avg_pnl":0.0}
    s = state["signal_stats"][signal_type]
    s["trades"]    += 1
    s["wins"]      += 1 if won else 0
    s["losses"]    += 0 if won else 1
    s["total_pnl"] += pnl_pct
    s["win_rate"]   = round(s["wins"]/s["trades"]*100, 1)
    s["avg_pnl"]    = round(s["total_pnl"]/s["trades"], 2)


def get_state() -> dict: return _load()
def toggle_auto(enabled: bool) -> dict:
    state = _load(); state["auto_enabled"] = bool(enabled); _save(state); return state
def reset_portfolio() -> dict:
    state = _default_state(); _save(state); return state


async def get_btc_change() -> float:
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{BINANCE_REST}/ticker/24hr?symbol=BTCUSDT")
            if r.status_code == 200: return float(r.json().get("priceChangePercent", 0))
    except: pass
    return 0.0


async def get_live_price(symbol: str) -> Optional[float]:
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{BINANCE_REST}/ticker/price?symbol={symbol}")
            if r.status_code == 200: return float(r.json()["price"])
    except: pass
    return None


async def check_momentum(symbol: str) -> bool:
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            r = await client.get(f"{BINANCE_REST}/klines?symbol={symbol}&interval=1h&limit=4")
            if r.status_code != 200: return True
            candles = r.json()
            if len(candles) < 3: return True
            c1=float(candles[-4][4]); c2=float(candles[-3][4])
            c3=float(candles[-2][4]); c4=float(candles[-1][4])
            green = sum([c2>c1, c3>c2, c4>c3])
            return green >= 2
    except: return True


def _calc_trailing_stop(trade, current_price):
    entry = float(trade["entry_price"])
    peak  = float(trade.get("peak_price", current_price))
    if current_price > peak: peak = current_price
    trailing_stop = round(peak * (1 - TRAILING_PCT), 8)
    current_profit_pct = (current_price - entry) / entry
    locked_stop = float(trade.get("initial_stop", entry * (1 - STOP_LOSS_PCT)))
    for profit_threshold, stop_level in sorted(PROFIT_LOCKS, reverse=True):
        if current_profit_pct >= profit_threshold:
            lock_price  = round(entry * (1 + stop_level), 8)
            locked_stop = max(locked_stop, lock_price)
            break
    return round(max(trailing_stop, locked_stop), 8), round(peak, 8)


async def auto_buy(signal: dict) -> Optional[dict]:
    state = _load()

    if not state.get("auto_enabled"): return None
    if str(signal.get("verdict","")) != "STRONG_BUY": return None

    confidence = float(signal.get("confidence", 0))
    if confidence < MIN_CONFIDENCE: return None

    symbol = signal["symbol"]
    name   = str(signal.get("name", symbol.replace("USDT","")))

    # FIX 2: Skip stablecoins
    base_name = name.upper().replace("USDT","")
    if base_name in SKIP_COINS: return None
    if any(s in symbol for s in ["EUR","USDC","PAXG","XAUT"]): return None

    # Already open
    if symbol in [t["symbol"] for t in state["open_trades"]]: return None
    if len(state["open_trades"]) >= MAX_OPEN_TRADES: return None

    # 24h re-entry block
    recent = state.get("recent_symbols", {})
    last_closed = recent.get(symbol, 0)
    if last_closed > 0 and (time.time()*1000 - last_closed)/3600000 < REENTRY_HOURS: return None

    # FIX 3: Loss streak pause
    pause_until = state.get("pause_until", 0)
    if time.time() * 1000 < pause_until:
        hours_left = round((pause_until - time.time()*1000) / 3600000, 1)
        print(f"Paused after {MAX_LOSS_STREAK} losses. {hours_left}h remaining.")
        return None

    # Momentum check
    if not await check_momentum(symbol): return None

    # Position sizing
    position_pct  = _get_position_size(confidence)
    portfolio_val = _portfolio_value(state)
    invest_gross  = round(min(float(state["balance"]), portfolio_val * position_pct), 2)
    if invest_gross < 100: return None

    buy_fee    = round(invest_gross * FEE_PCT, 2)
    invest_net = round(invest_gross - buy_fee, 2)

    price        = float(signal["price"])
    target_price = round(price * (1 + TAKE_PROFIT_PCT), 8)
    partial_price= round(price * (1 + PARTIAL_PROFIT), 8)  # FIX 4
    stop_price   = round(price * (1 - STOP_LOSS_PCT), 8)
    qty          = round(invest_net / price, 8)
    max_hold_ms  = int(time.time()*1000) + int(MAX_HOLD_DAYS * 24 * 3600 * 1000)

    signals_list = signal.get("signals", []) or []
    signal_type  = signals_list[0].get("type","") if signals_list else ""

    trade = {
        "id":             f"PT_{symbol}_{int(time.time())}",
        "symbol":         symbol,
        "name":           name,
        "entry_price":    price,
        "current_price":  price,
        "peak_price":     price,
        "target_price":   target_price,
        "partial_price":  partial_price,   # FIX 4
        "partial_taken":  False,           # FIX 4
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
        "reason":         str(signal.get("reason","")),
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
    print(f"✅ Bought {symbol} @ {price} | conf:{confidence}% | size:{position_pct*100}%")
    return trade


def auto_sell(trade_id: str, current_price: float, reason: str) -> Optional[dict]:
    state = _load()
    trade = next((t for t in state["open_trades"] if t["id"] == trade_id), None)
    if not trade: return None

    current_price = float(current_price)
    sell_fee      = round(float(trade["qty"]) * current_price * FEE_PCT, 2)
    sell_value    = round(float(trade["qty"]) * current_price - sell_fee, 2)
    pnl           = round(sell_value - float(trade["invested"]), 2)
    pnl_pct       = round(pnl / float(trade["invested"]) * 100, 2)
    won            = pnl > 0

    trade.update({
        "current_price": current_price,
        "current_value": sell_value,
        "sell_fee":      sell_fee,
        "total_fees":    round(float(trade.get("buy_fee",0)) + sell_fee, 2),
        "pnl":           pnl,
        "pnl_pct":       pnl_pct,
        "status":        "WIN" if won else "LOSS",
        "closed_at":     int(time.time() * 1000),
        "close_reason":  reason,
    })

    _update_signal_stats(state, trade.get("signal_type","UNKNOWN"), won, pnl_pct)

    # FIX 3: Track loss streak
    if not won:
        state["loss_streak"] = int(state.get("loss_streak", 0)) + 1
        if state["loss_streak"] >= MAX_LOSS_STREAK:
            state["pause_until"] = int(time.time()*1000) + int(PAUSE_HOURS * 3600 * 1000)
            print(f"⚠ {MAX_LOSS_STREAK} losses in a row — pausing {PAUSE_HOURS}h")
    else:
        state["loss_streak"] = 0  # reset on win

    state["open_trades"]   = [t for t in state["open_trades"] if t["id"] != trade_id]
    state["closed_trades"].insert(0, trade)
    state["balance"]       = round(float(state["balance"]) + sell_value, 2)
    state["total_profit"]  = round(float(state["total_profit"]) + pnl, 2)

    if "recent_symbols" not in state: state["recent_symbols"] = {}
    state["recent_symbols"][trade["symbol"]] = int(time.time() * 1000)

    _save(state)
    print(f"{'✅' if won else '❌'} Sold {trade['symbol']} | {reason} | {'+' if pnl_pct>0 else ''}{pnl_pct}%")
    return trade


def take_partial_profit(trade_id: str, current_price: float) -> Optional[dict]:
    """FIX 4: Sell 50% at +4% profit, let rest run free."""
    state = _load()
    trade = next((t for t in state["open_trades"] if t["id"] == trade_id), None)
    if not trade or trade.get("partial_taken"): return None

    current_price = float(current_price)
    half_qty      = float(trade["qty"]) / 2
    sell_fee      = round(half_qty * current_price * FEE_PCT, 2)
    sell_value    = round(half_qty * current_price - sell_fee, 2)

    # Update trade — remove half qty, move stop to breakeven
    trade["qty"]           = round(half_qty, 8)
    trade["invested"]      = round(float(trade["invested"]) / 2, 2)
    trade["partial_taken"] = True
    trade["stop_price"]    = float(trade["entry_price"])  # stop to breakeven
    trade["initial_stop"]  = float(trade["entry_price"])  # can't lose now

    # Return partial profit to balance
    state["balance"]      = round(float(state["balance"]) + sell_value, 2)
    state["total_profit"] = round(float(state["total_profit"]) + (sell_value - float(trade["invested"])), 2)
    _save(state)
    print(f"📈 Partial profit taken on {trade['symbol']} @ {current_price} | +₹{sell_value:.0f}")
    return trade


async def monitor_loop():
    """
    FIX 1: Check prices every 15 seconds (was 60s).
    This prevents DOLO-like losses where stop is missed.
    """
    while True:
        try:
            state = _load()

            if state["open_trades"]:
                btc_change = await get_btc_change()
                market_bad = btc_change < -5.0

                for trade in list(state["open_trades"]):
                    price = await get_live_price(trade["symbol"])
                    if not price: continue

                    price    = float(price)
                    invested = float(trade["invested"])
                    qty      = float(trade["qty"])

                    new_stop, new_peak = _calc_trailing_stop(trade, price)
                    sell_fee_est = round(qty * price * FEE_PCT, 2)
                    current_value= round(qty * price - sell_fee_est, 2)

                    trade["current_price"] = price
                    trade["current_value"] = current_value
                    trade["pnl"]           = round(current_value - invested, 2)
                    trade["pnl_pct"]       = round(trade["pnl"] / invested * 100, 2) if invested > 0 else 0
                    trade["peak_price"]    = new_peak
                    trade["stop_price"]    = new_stop

                    close_reason = None

                    # FIX 4: Partial profit at +4%
                    partial_price = float(trade.get("partial_price", price * 1.04))
                    if price >= partial_price and not trade.get("partial_taken"):
                        take_partial_profit(trade["id"], price)
                        continue  # reload state next iteration

                    # Time limit
                    max_hold = trade.get("max_hold_until", 0)
                    if max_hold and int(time.time()*1000) > max_hold:
                        close_reason = "TIME_LIMIT"

                    # Trailing stop
                    elif price <= new_stop:
                        peak = float(trade.get("peak_price", price))
                        if peak > float(trade["entry_price"]) * 1.01:
                            close_reason = "TRAILING_STOP"
                        elif price <= float(trade["initial_stop"]):
                            close_reason = "STOP_LOSS"

                    # Market crash
                    elif market_bad:
                        close_reason = "MARKET_CRASH"

                    if close_reason:
                        auto_sell(trade["id"], price, close_reason)
                    else:
                        _save(state)

                    await asyncio.sleep(0.3)

        except Exception as e:
            print(f"Monitor error: {e}")

        await asyncio.sleep(PRICE_CHECK_SEC)  # FIX 1: 15 seconds


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
    total_fees= sum(float(t.get("total_fees",0)) for t in closed)

    # Pause status
    pause_until = state.get("pause_until", 0)
    paused      = int(time.time()*1000) < pause_until
    pause_left  = round(max(0, pause_until - time.time()*1000) / 3600000, 1) if paused else 0

    open_display = []
    for t in open_t:
        td = dict(t)
        max_hold = t.get("max_hold_until", 0)
        if max_hold:
            ms_left = max_hold - int(time.time()*1000)
            td["days_remaining"] = round(max(0, ms_left/86400000), 1)
        entry = float(t.get("entry_price",0))
        stop  = float(t.get("stop_price",0))
        if entry > 0:
            td["locked_profit_pct"] = round((stop-entry)/entry*100, 2)
        open_display.append(td)

    return {
        "balance":          round(float(state["balance"]), 2),
        "starting":         float(state["starting"]),
        "portfolio_value":  portfolio,
        "total_return_pct": total_ret,
        "total_profit":     round(total_pnl, 2),
        "total_fees_paid":  round(total_fees, 2),
        "open_count":       len(open_t),
        "closed_count":     len(closed),
        "win_count":        len(wins),
        "loss_count":       len(losses),
        "win_rate":         win_rate,
        "avg_win_pct":      avg_win,
        "avg_loss_pct":     avg_loss,
        "auto_enabled":     bool(state.get("auto_enabled", True)),
        "loss_streak":      int(state.get("loss_streak", 0)),
        "paused":           paused,
        "pause_hours_left": pause_left,
        "signal_stats":     state.get("signal_stats", {}),
        "open_trades":      open_display,
        "closed_trades":    closed[:20],
    }
