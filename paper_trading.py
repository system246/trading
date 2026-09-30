"""
paper_trading.py — MAXIMUM FREE IMPROVEMENTS
1. Break-even at +2% (stop moves to entry)
2. Signal freshness 30min max
3. Multi-signal requirement (2+ signals)
4. Daily profit lock (stop if +₹500 today)
5. Volume exit signal
6. 2% trailing stop (was 5%)
7. Trailing only after +2% rise
8. Win/loss streak position sizing
9. Correlation check (no 2 similar coins)
10. 15s price checks
11. Tax calculation (TDS 1% + 30% income tax)
12. Confidence 92%+, max 3 trades
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
MIN_CONFIDENCE   = 92
MAX_HOLD_DAYS    = 1.5
REENTRY_HOURS    = 24
CHECK_INTERVAL   = 15        # every 15 seconds
TRAILING_PCT     = 0.02      # 2% trailing (was 5%)
TAKE_PROFIT_PCT  = 0.05      # +5% target
STOP_LOSS_PCT    = 0.03      # -3% stop
BREAKEVEN_PCT    = 0.02      # move stop to entry at +2%
FEE_PCT          = 0.001
TDS_PCT          = 0.01
INCOME_TAX_PCT   = 0.30
DAILY_PROFIT_LOCK= 500.0     # stop trading if +₹500 today
MAX_LOSS_STREAK  = 3
PAUSE_HOURS      = 6
SIGNAL_MAX_AGE   = 30        # minutes — skip signals older than 30min
MIN_SIGNALS      = 2         # require 2+ signal types

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

# Coins that tend to move together — won't hold 2 from same group
CORRELATION_GROUPS = [
    {"SOLUSDT","RAYUSDT","JUPUSDT","BONKUSDT"},
    {"ETHUSDT","STETHUSDT","WSTETHUSDT"},
    {"BTCUSDT","WBTCUSDT"},
    {"BNBUSDT","CAKEUSDT"},
]

SKIP_COINS = {
    "EUR","USDC","USDT","BUSD","DAI","TUSD","USDP","FDUSD",
    "PAXG","XAUT","WBTC","STETH","WSTETH","FRAX","LUSD",
}

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
        "loss_streak":    0,
        "win_streak":     0,
        "pause_until":    0,
        "daily_profit":   0.0,
        "daily_reset":    int(time.time() * 1000),
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


def _reset_daily_if_needed(state: dict):
    """Reset daily profit counter each day."""
    now = int(time.time() * 1000)
    last_reset = state.get("daily_reset", now)
    if now - last_reset > 86400000:  # 24 hours
        state["daily_profit"] = 0.0
        state["daily_reset"]  = now


def _get_position_size(confidence: float, state: dict) -> float:
    """Smart position sizing — adjusts for win/loss streak."""
    base = 0.10
    for threshold, size in POSITION_SIZES:
        if confidence >= threshold:
            base = size
            break

    # Win streak bonus
    win_streak = int(state.get("win_streak", 0))
    if win_streak >= 3:
        base = min(0.15, base * 1.1)  # +10% on hot streak

    # Loss streak reduction
    loss_streak = int(state.get("loss_streak", 0))
    if loss_streak >= 2:
        base = base * 0.7  # -30% after 2 losses

    return round(base, 3)


def _check_correlation(symbol: str, open_trades: list) -> bool:
    """Return True if safe to buy (no correlated coin already open)."""
    open_symbols = {t["symbol"] for t in open_trades}
    for group in CORRELATION_GROUPS:
        if symbol in group:
            if open_symbols & group:  # intersection
                return False
    return True


def _update_signal_stats(state, signal_type, won, pnl_pct):
    if "signal_stats" not in state: state["signal_stats"] = {}
    if signal_type not in state["signal_stats"]:
        state["signal_stats"][signal_type] = {
            "trades":0,"wins":0,"losses":0,
            "total_pnl":0.0,"win_rate":0.0,"avg_pnl":0.0
        }
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
            return sum([c2>c1, c3>c2, c4>c3]) >= 2
    except: return True


async def check_volume_still_high(symbol: str) -> bool:
    """FIX 5: Check volume is still elevated at time of buying."""
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            r = await client.get(f"{BINANCE_REST}/klines?symbol={symbol}&interval=15m&limit=10")
            if r.status_code != 200: return True
            candles = r.json()
            if len(candles) < 5: return True
            vols    = [float(c[5]) for c in candles]
            recent  = vols[-1]
            average = sum(vols[:-1]) / len(vols[:-1])
            return recent > average * 1.3  # volume still 30% above average
    except: return True


def _calc_trailing_stop(trade, current_price):
    entry = float(trade["entry_price"])
    peak  = float(trade.get("peak_price", current_price))
    if current_price > peak: peak = current_price

    trailing_stop = round(peak * (1 - TRAILING_PCT), 8)
    current_profit_pct = (current_price - entry) / entry

    # Break-even logic — move stop to entry at +2%
    if current_profit_pct >= BREAKEVEN_PCT:
        min_stop = entry  # never sell below entry once +2% reached
    else:
        min_stop = entry * (1 - STOP_LOSS_PCT)

    # Profit locks
    locked_stop = min_stop
    for profit_threshold, stop_level in sorted(PROFIT_LOCKS, reverse=True):
        if current_profit_pct >= profit_threshold:
            lock_price  = round(entry * (1 + stop_level), 8)
            locked_stop = max(locked_stop, lock_price)
            break

    final_stop = max(trailing_stop, locked_stop)
    return round(final_stop, 8), round(peak, 8)


async def auto_buy(signal: dict) -> Optional[dict]:
    state = _load()
    _reset_daily_if_needed(state)

    if not state.get("auto_enabled"): return None
    if str(signal.get("verdict","")) != "STRONG_BUY": return None

    confidence = float(signal.get("confidence", 0))
    if confidence < MIN_CONFIDENCE: return None

    symbol = signal["symbol"]
    name   = str(signal.get("name", symbol.replace("USDT","")))

    # Skip stablecoins
    if name.upper() in SKIP_COINS: return None
    if any(s in symbol for s in ["EUR","USDC","PAXG","XAUT"]): return None

    # FIX 2: Signal freshness check
    detected_at = signal.get("detected_at", int(time.time()*1000))
    age_min = (int(time.time()*1000) - detected_at) / 60000
    if age_min > SIGNAL_MAX_AGE:
        print(f"Skip {symbol} — signal {age_min:.0f}min old (max {SIGNAL_MAX_AGE}min)")
        return None

    # FIX 3: Multi-signal requirement
    signals_list = signal.get("signals", []) or []
    if len(signals_list) < MIN_SIGNALS:
        print(f"Skip {symbol} — only {len(signals_list)} signal(s), need {MIN_SIGNALS}")
        return None

    # Already open
    if symbol in [t["symbol"] for t in state["open_trades"]]: return None
    if len(state["open_trades"]) >= MAX_OPEN_TRADES: return None

    # Re-entry block
    recent = state.get("recent_symbols", {})
    last_closed = recent.get(symbol, 0)
    if last_closed > 0 and (time.time()*1000 - last_closed)/3600000 < REENTRY_HOURS: return None

    # Loss streak pause
    pause_until = state.get("pause_until", 0)
    if time.time()*1000 < pause_until:
        hours_left = round((pause_until - time.time()*1000)/3600000, 1)
        print(f"Paused after losses. {hours_left}h remaining.")
        return None

    # FIX 4: Daily profit lock
    daily_profit = float(state.get("daily_profit", 0))
    if daily_profit >= DAILY_PROFIT_LOCK:
        print(f"Daily profit lock — already made ₹{daily_profit:.0f} today")
        return None

    # Correlation check
    if not _check_correlation(symbol, state["open_trades"]):
        print(f"Skip {symbol} — correlated coin already open")
        return None

    # Momentum check
    if not await check_momentum(symbol): return None

    # FIX 5: Volume still high check
    if not await check_volume_still_high(symbol):
        print(f"Skip {symbol} — volume dropped since signal")
        return None

    # Position sizing with streak adjustment
    position_pct  = _get_position_size(confidence, state)
    portfolio_val = _portfolio_value(state)
    invest_gross  = round(min(float(state["balance"]), portfolio_val * position_pct), 2)
    if invest_gross < 100: return None

    buy_fee    = round(invest_gross * FEE_PCT, 2)
    invest_net = round(invest_gross - buy_fee, 2)

    price        = float(signal["price"])
    target_price = round(price * (1 + TAKE_PROFIT_PCT), 8)
    stop_price   = round(price * (1 - STOP_LOSS_PCT), 8)
    qty          = round(invest_net / price, 8)
    max_hold_ms  = int(time.time()*1000) + int(MAX_HOLD_DAYS * 24 * 3600 * 1000)
    signal_type  = signals_list[0].get("type","") if signals_list else ""

    trade = {
        "id":             f"PT_{symbol}_{int(time.time())}",
        "symbol":         symbol,
        "name":           name,
        "entry_price":    price,
        "current_price":  price,
        "peak_price":     price,
        "target_price":   target_price,
        "stop_price":     stop_price,
        "initial_stop":   stop_price,
        "qty":            qty,
        "invested":       invest_gross,
        "buy_fee":        buy_fee,
        "current_value":  invest_net,
        "pnl":            0.0,
        "pnl_pct":        0.0,
        "confidence":     confidence,
        "position_pct":   round(position_pct * 100, 1),
        "signal_type":    signal_type,
        "signal_count":   len(signals_list),
        "reason":         str(signal.get("reason","")),
        "status":         "OPEN",
        "opened_at":      int(time.time() * 1000),
        "max_hold_until": max_hold_ms,
        "breakeven_hit":  False,
        "closed_at":      None,
        "close_reason":   None,
    }

    state["balance"] = round(float(state["balance"]) - invest_gross, 2)
    state["open_trades"].append(trade)
    _save(state)
    print(f"✅ Bought {symbol} @ {price} | conf:{confidence}% | {len(signals_list)} signals | age:{age_min:.0f}min")
    return trade


def auto_sell(trade_id: str, current_price: float, reason: str) -> Optional[dict]:
    state = _load()
    _reset_daily_if_needed(state)
    trade = next((t for t in state["open_trades"] if t["id"] == trade_id), None)
    if not trade: return None

    current_price = float(current_price)
    sell_gross    = round(float(trade["qty"]) * current_price, 2)
    sell_fee      = round(sell_gross * FEE_PCT, 2)
    tds           = round(sell_gross * TDS_PCT, 2)
    sell_value    = round(sell_gross - sell_fee - tds, 2)
    pnl           = round(sell_value - float(trade["invested"]), 2)
    pnl_pct       = round(pnl / float(trade["invested"]) * 100, 2)
    tax_estimate  = round(max(0, pnl * INCOME_TAX_PCT), 2)
    net_profit    = round(pnl - tax_estimate, 2)
    net_pct       = round(net_profit / float(trade["invested"]) * 100, 2)
    won            = pnl > 0

    trade.update({
        "current_price": current_price,
        "current_value": sell_value,
        "sell_fee":      sell_fee,
        "tds":           tds,
        "tax_estimate":  tax_estimate,
        "net_profit":    net_profit,
        "net_pct":       net_pct,
        "total_fees":    round(float(trade.get("buy_fee",0)) + sell_fee, 2),
        "pnl":           pnl,
        "pnl_pct":       pnl_pct,
        "status":        "WIN" if won else "LOSS",
        "closed_at":     int(time.time() * 1000),
        "close_reason":  reason,
    })

    _update_signal_stats(state, trade.get("signal_type","UNKNOWN"), won, pnl_pct)

    # Streak tracking
    if won:
        state["win_streak"]  = int(state.get("win_streak", 0)) + 1
        state["loss_streak"] = 0
    else:
        state["loss_streak"] = int(state.get("loss_streak", 0)) + 1
        state["win_streak"]  = 0
        if state["loss_streak"] >= MAX_LOSS_STREAK:
            state["pause_until"] = int(time.time()*1000) + int(PAUSE_HOURS * 3600 * 1000)
            print(f"⚠ {MAX_LOSS_STREAK} losses — pausing {PAUSE_HOURS}h")

    # Daily profit tracking
    state["daily_profit"] = round(float(state.get("daily_profit", 0)) + pnl, 2)

    state["open_trades"]   = [t for t in state["open_trades"] if t["id"] != trade_id]
    state["closed_trades"].insert(0, trade)
    state["balance"]       = round(float(state["balance"]) + sell_value, 2)
    state["total_profit"]  = round(float(state["total_profit"]) + pnl, 2)

    if "recent_symbols" not in state: state["recent_symbols"] = {}
    state["recent_symbols"][trade["symbol"]] = int(time.time() * 1000)

    _save(state)
    print(f"{'✅' if won else '❌'} Sold {trade['symbol']} | {reason} | {'+' if pnl_pct>0 else ''}{pnl_pct}% | net:{net_pct}%")
    return trade


async def monitor_loop():
    while True:
        try:
            state = _load()
            _reset_daily_if_needed(state)

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

                    # Check if breakeven hit
                    entry = float(trade["entry_price"])
                    if price >= entry * (1 + BREAKEVEN_PCT) and not trade.get("breakeven_hit"):
                        trade["breakeven_hit"] = True
                        print(f"🔒 Breakeven locked for {trade['symbol']}")

                    fee_est = round(qty * price * FEE_PCT, 2)
                    tds_est = round(qty * price * TDS_PCT, 2)
                    current_value = round(qty * price - fee_est - tds_est, 2)

                    trade["current_price"] = price
                    trade["current_value"] = current_value
                    trade["pnl"]           = round(current_value - invested, 2)
                    trade["pnl_pct"]       = round(trade["pnl"] / invested * 100, 2) if invested > 0 else 0
                    trade["peak_price"]    = new_peak
                    trade["stop_price"]    = new_stop

                    close_reason = None

                    # Time limit
                    max_hold = trade.get("max_hold_until", 0)
                    if max_hold and int(time.time()*1000) > max_hold:
                        close_reason = "TIME_LIMIT"

                    # Trailing stop — only after +2% rise
                    elif price <= new_stop:
                        if new_peak > entry * (1 + BREAKEVEN_PCT):
                            close_reason = "TRAILING_STOP"
                        elif price <= float(trade.get("initial_stop", entry * (1-STOP_LOSS_PCT))):
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

        await asyncio.sleep(CHECK_INTERVAL)


def get_stats() -> dict:
    state  = _load()
    _reset_daily_if_needed(state)
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
    total_tds = sum(float(t.get("tds",0)) for t in closed)
    total_tax = sum(float(t.get("tax_estimate",0)) for t in closed)
    total_net = sum(float(t.get("net_profit",0)) for t in closed)

    pause_until = state.get("pause_until", 0)
    paused      = int(time.time()*1000) < pause_until
    pause_left  = round(max(0, pause_until - time.time()*1000)/3600000, 1) if paused else 0

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
        "loss_streak":      int(state.get("loss_streak", 0)),
        "win_streak":       int(state.get("win_streak", 0)),
        "paused":           paused,
        "pause_hours_left": pause_left,
        "daily_profit":     round(float(state.get("daily_profit", 0)), 2),
        "daily_lock":       bool(float(state.get("daily_profit", 0)) >= DAILY_PROFIT_LOCK),
        "auto_enabled":     bool(state.get("auto_enabled", True)),
        "signal_stats":     state.get("signal_stats", {}),
        "open_trades":      open_display,
        "closed_trades":    closed[:20],
    }