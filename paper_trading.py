"""
paper_trading.py — Auto paper trading engine v5
FIXES: breakeven at +4%, min hold 2h, BTC check -2%, TDS reality check
"""

import json
import time
import asyncio
import httpx
import os
from typing import Optional

PAPER_FILE        = "paper_trades.json"
STARTING_BALANCE  = 20000.0
MAX_PER_TRADE     = 0.10
MAX_OPEN_TRADES   = 3
MIN_CONFIDENCE    = 92
TAKE_PROFIT_PCT   = 0.05
STOP_LOSS_PCT     = 0.03
MAX_HOLD_DAYS     = 1.5
REENTRY_HOURS     = 24
CHECK_INTERVAL    = 15        # 15-second checks
TRAILING_PCT      = 0.02      # 2% trailing stop
BREAKEVEN_PCT     = 0.04      # FIX: was 0.02 — trailing only after +4%
MIN_HOLD_HOURS    = 2         # FIX: don't fire stop within first 2h (unless drop > 5%)
TDS_PCT           = 0.01      # 1% TDS on sell (reality check)
INCOME_TAX_PCT    = 0.30      # 30% income tax on profit
DAILY_PROFIT_LOCK = 500.0     # Stop trading for the day after ₹500 profit
MAX_LOSS_STREAK   = 3         # Pause after 3 consecutive losses
PAUSE_HOURS       = 6         # Pause duration after loss streak
SIGNAL_MAX_AGE    = 30        # Signal must be < 30 min old
MIN_SIGNALS       = 2         # Need at least 2 signal types

BINANCE_REST = "https://api.binance.com/api/v3"

# Profit lock levels: (profit_threshold, stop_moves_to)
PROFIT_LOCKS = [
    (0.10, 0.05),   # at +10% → stop to +5%
    (0.07, 0.02),   # at +7%  → stop to +2%
    (0.05, 0.00),   # at +5%  → stop to breakeven
]


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
        "loss_streak":    0,
        "win_streak":     0,
        "paused_until":   0,
        "daily_profit":   0.0,
        "daily_reset":    int(time.time() * 1000),
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


def _reset_daily_if_needed(state: dict):
    now = int(time.time() * 1000)
    last_reset = state.get("daily_reset", 0)
    if now - last_reset > 86400000:  # 24h
        state["daily_profit"] = 0.0
        state["daily_reset"]  = now


def _similar_symbol(sym1: str, sym2: str) -> bool:
    """Check if two symbols are correlated (same base asset family)."""
    bases = {
        "BTC": ["BTC"], "ETH": ["ETH"],
        "SOL": ["SOL"], "BNB": ["BNB"],
        "AVAX": ["AVAX"], "MATIC": ["MATIC", "POL"],
    }
    for group in bases.values():
        s1 = sym1.replace("USDT", "")
        s2 = sym2.replace("USDT", "")
        if s1 in group and s2 in group:
            return True
    return False


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


def _calc_trailing_stop(trade: dict, current_price: float):
    """
    Trailing stop — only activates after price rises BREAKEVEN_PCT (4%).
    Until then, uses the initial hard stop.
    Also applies profit lock levels.
    """
    entry    = float(trade["entry_price"])
    peak     = float(trade.get("peak_price", current_price))
    init_stop = float(trade.get("initial_stop", entry * (1 - STOP_LOSS_PCT)))

    if current_price > peak:
        peak = current_price

    current_profit_pct = (peak - entry) / entry

    # Trailing only kicks in after BREAKEVEN_PCT gain (FIX: was 2%, now 4%)
    if current_profit_pct >= BREAKEVEN_PCT:
        trailing_stop = round(peak * (1 - TRAILING_PCT), 8)
    else:
        trailing_stop = init_stop  # hold original stop until +4%

    # Profit lock levels
    locked_stop = init_stop
    for profit_threshold, stop_level in sorted(PROFIT_LOCKS, reverse=True):
        if current_profit_pct >= profit_threshold:
            lock_price = round(entry * (1 + stop_level), 8)
            locked_stop = max(locked_stop, lock_price)
            break

    final_stop = max(trailing_stop, locked_stop)
    return round(final_stop, 8), round(peak, 8)


async def auto_buy(signal: dict) -> Optional[dict]:
    state = _load()
    _reset_daily_if_needed(state)

    if not state.get("auto_enabled"):
        return None
    if str(signal.get("verdict", "")) != "STRONG_BUY":
        return None
    if float(signal.get("confidence", 0)) < MIN_CONFIDENCE:
        return None

    # Paused after loss streak?
    paused_until = state.get("paused_until", 0)
    if int(time.time() * 1000) < paused_until:
        return None

    # Daily profit lock
    if state.get("daily_profit", 0) >= DAILY_PROFIT_LOCK:
        return None

    symbol = signal["symbol"]

    # Correlation filter — no 2 similar coins open
    for open_trade in state["open_trades"]:
        if _similar_symbol(open_trade["symbol"], symbol):
            return None

    symbols_open = [t["symbol"] for t in state["open_trades"]]
    if symbol in symbols_open:
        return None
    if len(state["open_trades"]) >= MAX_OPEN_TRADES:
        return None

    # Re-entry cooldown
    recent      = state.get("recent_symbols", {})
    last_closed = recent.get(symbol, 0)
    hours_since = (time.time() * 1000 - last_closed) / 3600000
    if last_closed > 0 and hours_since < REENTRY_HOURS:
        return None

    # Signal freshness check
    signal_time = signal.get("signal_time") or signal.get("timestamp", 0)
    if signal_time:
        age_min = (time.time() * 1000 - int(signal_time)) / 60000
        if age_min > SIGNAL_MAX_AGE:
            return None

    # Need at least MIN_SIGNALS types
    signals_list = signal.get("signals", []) or []
    if len(signals_list) < MIN_SIGNALS:
        return None

    # Position sizing based on streak
    loss_streak = state.get("loss_streak", 0)
    win_streak  = state.get("win_streak", 0)
    base_size   = MAX_PER_TRADE
    if loss_streak >= 2:
        base_size = MAX_PER_TRADE * 0.5    # halve size on loss streak
    elif win_streak >= 3:
        base_size = min(MAX_PER_TRADE * 1.25, 0.15)  # boost on win streak

    portfolio_val = _portfolio_value(state)
    invest = round(min(float(state["balance"]), portfolio_val * base_size), 2)
    if invest < 100:
        return None

    price        = float(signal["price"])
    target_price = round(price * (1 + TAKE_PROFIT_PCT), 8)
    stop_price   = round(price * (1 - STOP_LOSS_PCT), 8)
    qty          = round(invest / price, 8)
    max_hold_ms  = int(time.time() * 1000) + int(MAX_HOLD_DAYS * 24 * 3600 * 1000)

    signal_type = signals_list[0].get("type", "") if signals_list else ""

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
        "invested":       invest,
        "current_value":  invest,
        "pnl":            0.0,
        "pnl_pct":        0.0,
        "confidence":     float(signal.get("confidence", 0)),
        "signal_type":    signal_type,
        "reason":         str(signal.get("reason", "")),
        "status":         "OPEN",
        "opened_at":      int(time.time() * 1000),
        "max_hold_until": max_hold_ms,
        "closed_at":      None,
        "close_reason":   None,
        "trailing":       True,
    }

    state["balance"] = round(float(state["balance"]) - invest, 2)
    state["open_trades"].append(trade)
    _save(state)
    return trade


def auto_sell(trade_id: str, current_price: float, reason: str) -> Optional[dict]:
    state = _load()
    _reset_daily_if_needed(state)

    trade = next((t for t in state["open_trades"] if t["id"] == trade_id), None)
    if not trade:
        return None

    current_price = float(current_price)
    sell_value    = round(float(trade["qty"]) * current_price, 2)

    # TDS: 1% deducted on sell value (reality check for real trading)
    tds_amount    = round(sell_value * TDS_PCT, 2)
    net_sell      = round(sell_value - tds_amount, 2)

    pnl           = round(net_sell - float(trade["invested"]), 2)
    pnl_pct       = round(pnl / float(trade["invested"]) * 100, 2)

    # Income tax on profit (30%)
    tax_amount = round(max(0, pnl) * INCOME_TAX_PCT, 2)

    trade.update({
        "current_price": current_price,
        "current_value": sell_value,
        "net_sell":      net_sell,
        "tds":           tds_amount,
        "tax":           tax_amount,
        "pnl":           pnl,
        "pnl_pct":       pnl_pct,
        "status":        "WIN" if pnl > 0 else "LOSS",
        "closed_at":     int(time.time() * 1000),
        "close_reason":  reason,
    })

    # Update streaks
    if pnl > 0:
        state["loss_streak"] = 0
        state["win_streak"]  = state.get("win_streak", 0) + 1
    else:
        state["win_streak"]  = 0
        state["loss_streak"] = state.get("loss_streak", 0) + 1
        # Pause after MAX_LOSS_STREAK consecutive losses
        if state["loss_streak"] >= MAX_LOSS_STREAK:
            state["paused_until"] = int(time.time() * 1000) + int(PAUSE_HOURS * 3600000)
            print(f"⚠️  {MAX_LOSS_STREAK} losses in a row — pausing {PAUSE_HOURS}h")

    # Daily profit tracking
    state["daily_profit"] = round(state.get("daily_profit", 0) + pnl, 2)

    state["open_trades"]  = [t for t in state["open_trades"] if t["id"] != trade_id]
    state["closed_trades"].insert(0, trade)
    state["balance"]      = round(float(state["balance"]) + net_sell, 2)
    state["total_profit"] = round(float(state.get("total_profit", 0)) + pnl, 2)

    if "recent_symbols" not in state:
        state["recent_symbols"] = {}
    state["recent_symbols"][trade["symbol"]] = int(time.time() * 1000)

    _save(state)
    return trade


async def monitor_loop():
    """
    Background loop — runs every 15s.
    FIX: BTC check at -2% (was -5%)
    FIX: trailing only after +4% (was +2%)
    FIX: stop loss only fires after 2h hold (unless drop > 5%)
    """
    while True:
        try:
            state = _load()

            if state["open_trades"]:
                btc_change = await get_btc_change()
                market_bad = btc_change < -2.0    # FIX: was -5.0

                for trade in list(state["open_trades"]):
                    price = await get_live_price(trade["symbol"])
                    if not price:
                        continue

                    price    = float(price)
                    invested = float(trade["invested"])
                    qty      = float(trade["qty"])
                    entry    = float(trade["entry_price"])

                    # How long have we held this trade?
                    hold_h = (int(time.time() * 1000) - trade["opened_at"]) / 3600000

                    new_stop, new_peak = _calc_trailing_stop(trade, price)

                    trade["current_price"] = price
                    trade["current_value"] = round(qty * price, 2)
                    trade["pnl"]           = round(trade["current_value"] - invested, 2)
                    trade["pnl_pct"]       = round(trade["pnl"] / invested * 100, 2)
                    trade["peak_price"]    = new_peak
                    trade["stop_price"]    = new_stop

                    close_reason = None
                    drop_pct     = (entry - price) / entry  # positive = drop

                    # Time limit
                    max_hold = trade.get("max_hold_until", 0)
                    if max_hold and int(time.time() * 1000) > max_hold:
                        close_reason = "TIME_LIMIT"

                    # Take profit
                    elif price >= float(trade["target_price"]):
                        close_reason = "TAKE_PROFIT"

                    # Stop loss / trailing — FIX: respect min hold time
                    elif price <= new_stop:
                        # Fire immediately if severe drop (> 5%) regardless of hold time
                        # Otherwise wait MIN_HOLD_HOURS before firing
                        if drop_pct >= 0.05 or hold_h >= MIN_HOLD_HOURS:
                            peak = float(trade.get("peak_price", price))
                            if peak > entry * (1 + BREAKEVEN_PCT):
                                close_reason = "TRAILING_STOP"
                            else:
                                close_reason = "STOP_LOSS"

                    # BTC market crash — FIX: now -2% triggers (was -5%)
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
    _reset_daily_if_needed(state)
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
    total_tds = sum(float(t.get("tds", 0)) for t in closed)
    total_tax = sum(float(t.get("tax", 0)) for t in closed)

    paused_until = state.get("paused_until", 0)
    is_paused    = int(time.time() * 1000) < paused_until
    pause_mins   = max(0, (paused_until - int(time.time() * 1000)) / 60000) if is_paused else 0

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
        # Show hold time
        hold_h = (int(time.time() * 1000) - t["opened_at"]) / 3600000
        td["hold_hours"] = round(hold_h, 1)
        open_display.append(td)

    return {
        "balance":          round(float(state["balance"]), 2),
        "starting":         float(state["starting"]),
        "portfolio_value":  portfolio,
        "total_return_pct": total_ret,
        "total_profit":     round(total_pnl, 2),
        "total_tds":        round(total_tds, 2),
        "total_tax":        round(total_tax, 2),
        "net_after_tax":    round(total_pnl - total_tax, 2),
        "open_count":       len(open_t),
        "closed_count":     len(closed),
        "win_count":        len(wins),
        "loss_count":       len(losses),
        "win_rate":         win_rate,
        "avg_win_pct":      avg_win,
        "avg_loss_pct":     avg_loss,
        "loss_streak":      state.get("loss_streak", 0),
        "win_streak":       state.get("win_streak", 0),
        "is_paused":        is_paused,
        "pause_mins_left":  round(pause_mins, 0),
        "daily_profit":     round(state.get("daily_profit", 0), 2),
        "auto_enabled":     bool(state.get("auto_enabled", True)),
        "open_trades":      open_display,
        "closed_trades":    closed[:20],
    }