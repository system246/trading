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

# ML scorer — imported lazily to avoid circular imports
_ml_scorer = None
def _get_ml():
    global _ml_scorer
    if _ml_scorer is None:
        try:
            from ml_scorer import MLScorer
            _ml_scorer = MLScorer()
        except Exception as e:
            print(f"ML import error: {e}")
    return _ml_scorer

PAPER_FILE        = "paper_trades.json"
STARTING_BALANCE  = 20000.0
MAX_PER_TRADE     = 0.10
MAX_OPEN_TRADES   = 3
MIN_CONFIDENCE    = 92
TAKE_PROFIT_PCT   = 0.08      # #1: raised to 8% (trailing captures whatever it gives)
STOP_LOSS_PCT     = 0.025     # #9: tighter stop — risk only 2.5%
MAX_HOLD_DAYS     = 3.0       # #3: allow up to 3 days if still trending
FLAT_EXIT_HOURS   = 8         # #3: exit if coin flat for 8h (no movement)
FLAT_PCT          = 0.005     # #3: "flat" = less than 0.5% move from entry in 8h
REENTRY_HOURS     = 24
CHECK_INTERVAL    = 15        # 15-second checks
TRAILING_PCT      = 0.025     # slightly wider trail — more room to breathe
BREAKEVEN_PCT     = 0.03      # lock breakeven after +3% (was 4%)
PARTIAL_EXIT_PCT  = 0.05      # #1: sell 50% of position at +5%
MIN_HOLD_HOURS    = 2         # don't fire stop within first 2h (unless drop > 5%)
TDS_PCT           = 0.01      # 1% TDS on sell (reality check)
INCOME_TAX_PCT    = 0.30      # 30% income tax on profit
DAILY_PROFIT_LOCK = 500.0     # Stop trading for the day after ₹500 profit
MAX_LOSS_STREAK   = 3         # Pause after 3 consecutive losses
PAUSE_HOURS       = 6         # Pause duration after loss streak
SIGNAL_MAX_AGE    = 30        # Signal must be < 30 min old
MIN_SIGNALS       = 3         # Need at least 3 signal types
SIGNAL_COOLDOWN_H = 4         # #7: don't re-signal same coin for 4h after a loss
# #9: Real trade safety limits
MAX_REAL_POSITION_PCT = 0.02  # never risk more than 2% of real balance per trade
MAX_REAL_TRADES       = 2     # max concurrent real trades (conservative)

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


async def get_market_regime() -> dict:
    """
    #2: Market regime filter — only trade in uptrending markets.
    Checks BTC 4h RSI + 20-day MA. Returns regime info for ML feature too.
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # BTC 4h candles (last 60 = 10 days)
            r4h = await client.get(
                f"{BINANCE_REST}/klines?symbol=BTCUSDT&interval=4h&limit=60"
            )
            # BTC daily candles (last 25 = MA20)
            r1d = await client.get(
                f"{BINANCE_REST}/klines?symbol=BTCUSDT&interval=1d&limit=25"
            )
        if r4h.status_code != 200 or r1d.status_code != 200:
            return {"regime": "UNKNOWN", "tradeable": True, "reason": "data_error"}

        closes_4h = [float(k[4]) for k in r4h.json()]
        closes_1d = [float(k[4]) for k in r1d.json()]

        price     = closes_4h[-1]
        ma20      = sum(closes_1d[-20:]) / 20

        # Simple RSI on 4h
        gains  = [max(closes_4h[i]-closes_4h[i-1], 0) for i in range(1, len(closes_4h))]
        losses = [max(closes_4h[i-1]-closes_4h[i], 0) for i in range(1, len(closes_4h))]
        ag = sum(gains[-14:]) / 14
        al = sum(losses[-14:]) / 14
        rsi_4h = round(100 - (100 / (1 + ag/al)), 1) if al > 0 else 50.0

        above_ma20   = price > ma20
        rsi_bullish  = rsi_4h > 45   # not oversold/neutral-bullish
        rsi_overbought = rsi_4h > 75  # don't chase overbought

        if above_ma20 and rsi_bullish and not rsi_overbought:
            regime    = "BULL"
            tradeable = True
            reason    = f"BTC above MA20, RSI {rsi_4h}"
        elif rsi_overbought:
            regime    = "OVERBOUGHT"
            tradeable = False          # don't buy tops
            reason    = f"BTC RSI {rsi_4h} — overbought, wait for pullback"
        elif not above_ma20:
            regime    = "BEAR"
            tradeable = False
            reason    = f"BTC below 20-day MA — downtrend"
        else:
            regime    = "NEUTRAL"
            tradeable = True           # allow in neutral, scanner will filter
            reason    = f"BTC neutral, RSI {rsi_4h}"

        return {
            "regime":    regime,
            "tradeable": tradeable,
            "reason":    reason,
            "btc_price": price,
            "ma20":      round(ma20, 2),
            "rsi_4h":    rsi_4h,
            "above_ma20": above_ma20,
        }
    except Exception as e:
        print(f"Market regime error: {e}")
        return {"regime": "UNKNOWN", "tradeable": True, "reason": str(e)}


def _is_signal_on_cooldown(symbol: str, state: dict) -> bool:
    """
    #7: Duplicate signal filter — if coin closed as LOSS recently,
    don't trade it again for SIGNAL_COOLDOWN_H hours.
    """
    recent = state.get("recent_symbols", {})
    info   = recent.get(symbol, {})
    if isinstance(info, (int, float)):
        # Legacy format — just a timestamp, no loss info
        return False
    last_loss_at = info.get("last_loss_at", 0)
    if last_loss_at:
        hours_since = (time.time() * 1000 - last_loss_at) / 3600000
        if hours_since < SIGNAL_COOLDOWN_H:
            return True
    return False


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

    # #2: Market regime filter — skip if BTC in downtrend or overbought
    regime = await get_market_regime()
    if not regime.get("tradeable", True):
        print(f"Skip {symbol} — market regime: {regime.get('reason','')}")
        return None

    # Correlation filter — no 2 similar coins open
    for open_trade in state["open_trades"]:
        if _similar_symbol(open_trade["symbol"], symbol):
            return None

    symbols_open = [t["symbol"] for t in state["open_trades"]]
    if symbol in symbols_open:
        return None
    if len(state["open_trades"]) >= MAX_OPEN_TRADES:
        return None

    # #7: Duplicate/cooldown filter — skip if coin lost recently
    if _is_signal_on_cooldown(symbol, state):
        print(f"Skip {symbol} — on loss cooldown ({SIGNAL_COOLDOWN_H}h)")
        return None

    # Re-entry cooldown (24h after any close)
    recent = state.get("recent_symbols", {})
    info   = recent.get(symbol, {})
    last_closed = info.get("last_closed_at", 0) if isinstance(info, dict) else info
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
        # Save signal features for ML training later
        "ch3h":           float(signal.get("ch3h") or 0),
        "ch24h":          float(signal.get("ch24h") or 0),
        "ch7d":           float(signal.get("ch7d") or 0),
        "rsi_h":          signal.get("rsi_h"),
        "rsi_d":          signal.get("rsi_d"),
        "rsi_rising":     bool(signal.get("rsi_rising", False)),
        "macd_hist":      signal.get("macd_hist"),
        "macd_crossed":   bool(signal.get("macd_crossed", False)),
        "vol_ratio":      signal.get("vol_ratio"),
        "vol_spike":      bool(signal.get("vol_spike", False)),
        "supertrend_bull":bool(signal.get("supertrend_bull", False)),
        "ema_bullish":    bool(signal.get("ema_bullish", False)),
        "ema_pullback":   bool(signal.get("ema_pullback", False)),
        "above_ema21":    bool(signal.get("above_ema21", False)),
        "bb_width":       signal.get("bb_width"),
        "bb_squeeze":     bool(signal.get("bb_squeeze", False)),
        "obv_trend":      signal.get("obv_trend"),
        "adx":            signal.get("adx"),
        "stoch_k":        signal.get("stoch_k"),
        "mtf_score":      signal.get("mtf_score"),
        "book":           signal.get("book"),
        "funding":        signal.get("funding"),
        "vol_support":    signal.get("vol_support"),
        "quality":        signal.get("quality"),
    }

    state["balance"] = round(float(state["balance"]) - invest, 2)
    state["open_trades"].append(trade)
    _save(state)
    return trade


def manual_buy(body: dict) -> Optional[dict]:
    """
    Manual paper buy — user chose the signal themselves.
    Same tracking as auto_buy but skips confidence/signal filters.
    """
    state = _load()
    _reset_daily_if_needed(state)

    symbol = str(body.get("symbol", "")).upper()
    if not symbol: return None

    # Check balance and duplicate
    if symbol in [t["symbol"] for t in state["open_trades"]]: return None
    if len(state["open_trades"]) >= MAX_OPEN_TRADES + 2: return None  # allow 2 extra for manual

    invest = float(body.get("invest", 0))
    if invest < 100 or invest > float(state["balance"]): return None

    price        = float(body.get("price", 0))
    if price <= 0: return None

    target_price = round(price * (1 + TAKE_PROFIT_PCT), 8)
    stop_price   = round(price * (1 - STOP_LOSS_PCT), 8)
    qty          = round(invest / price, 8)
    max_hold_ms  = int(time.time() * 1000) + int(MAX_HOLD_DAYS * 24 * 3600 * 1000)

    signals_list = body.get("signals", []) or []
    signal_type  = body.get("signal_type", "") or (signals_list[0].get("type","") if signals_list else "MANUAL")

    trade = {
        "id":             f"MPT_{symbol}_{int(time.time())}",
        "symbol":         symbol,
        "name":           str(body.get("name", symbol.replace("USDT",""))),
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
        "confidence":     float(body.get("confidence", 0)),
        "signal_type":    signal_type,
        "reason":         str(body.get("reason", "Manual buy")),
        "status":         "OPEN",
        "opened_at":      int(time.time() * 1000),
        "max_hold_until": max_hold_ms,
        "closed_at":      None,
        "close_reason":   None,
        "trailing":       True,
        "manual":         True,   # flag so we can show it differently in UI
        # ML features
        "ch3h":           body.get("ch3h", 0),
        "ch24h":          body.get("ch24h", 0),
        "ch7d":           body.get("ch7d", 0),
        "rsi_h":          body.get("rsi_h"),
        "rsi_d":          body.get("rsi_d"),
        "rsi_rising":     bool(body.get("rsi_rising", False)),
        "macd_hist":      body.get("macd_hist"),
        "macd_crossed":   bool(body.get("macd_crossed", False)),
        "vol_ratio":      body.get("vol_ratio"),
        "vol_spike":      bool(body.get("vol_spike", False)),
        "supertrend_bull":bool(body.get("supertrend_bull", False)),
        "ema_bullish":    bool(body.get("ema_bullish", False)),
        "ema_pullback":   bool(body.get("ema_pullback", False)),
        "above_ema21":    bool(body.get("above_ema21", False)),
        "bb_width":       body.get("bb_width"),
        "bb_squeeze":     bool(body.get("bb_squeeze", False)),
        "obv_trend":      body.get("obv_trend"),
        "adx":            body.get("adx"),
        "stoch_k":        body.get("stoch_k"),
        "mtf_score":      body.get("mtf_score"),
        "book":           body.get("book"),
        "funding":        body.get("funding"),
        "vol_support":    body.get("vol_support"),
        "quality":        body.get("quality"),
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
    tax_amount  = round(max(0, pnl) * INCOME_TAX_PCT, 2)
    net_profit  = round(pnl - tax_amount, 2)   # FIX: actual take-home after tax

    trade.update({
        "current_price": current_price,
        "current_value": sell_value,
        "net_sell":      net_sell,
        "tds":           tds_amount,
        "tax":           tax_amount,
        "pnl":           pnl,
        "pnl_pct":       pnl_pct,
        "net_profit":    net_profit,             # FIX: per-trade net shown in history
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

    # #7: Track last close + last loss per coin for duplicate/cooldown filter
    if "recent_symbols" not in state:
        state["recent_symbols"] = {}
    sym_info = {
        "last_closed_at": int(time.time() * 1000),
        "last_status":    trade["status"],
    }
    if trade["status"] == "LOSS":
        sym_info["last_loss_at"] = int(time.time() * 1000)
    else:
        # Keep old loss time if it existed
        old = state["recent_symbols"].get(trade["symbol"], {})
        if isinstance(old, dict):
            sym_info["last_loss_at"] = old.get("last_loss_at", 0)
    state["recent_symbols"][trade["symbol"]] = sym_info

    _save(state)

    # Trigger ML retraining check after every closed trade
    try:
        ml = _get_ml()
        if ml:
            ml.maybe_retrain()
    except Exception as e:
        print(f"ML retrain error: {e}")

    return trade


def _do_partial_exit(trade: dict, current_price: float, state: dict) -> bool:
    """
    #1: Partial exit — sell 50% at PARTIAL_EXIT_PCT (+5%).
    Returns True if partial exit was done.
    Only fires once per trade (partial_exited flag).
    """
    if trade.get("partial_exited"):
        return False
    entry      = float(trade["entry_price"])
    profit_pct = (current_price - entry) / entry
    if profit_pct < PARTIAL_EXIT_PCT:
        return False

    # Sell half the position
    half_qty   = round(float(trade["qty"]) / 2, 8)
    sell_value = round(half_qty * current_price, 2)
    tds        = round(sell_value * TDS_PCT, 2)
    net_sell   = round(sell_value - tds, 2)
    pnl        = round(net_sell - float(trade["invested"]) / 2, 2)

    # Update trade: reduce qty and invested, return half balance
    trade["qty"]             = round(float(trade["qty"]) - half_qty, 8)
    trade["invested"]        = round(float(trade["invested"]) / 2, 2)
    trade["partial_exited"]  = True
    trade["partial_exit_at"] = int(time.time() * 1000)
    trade["partial_exit_pct"]= round(profit_pct * 100, 2)
    trade["partial_pnl"]     = pnl

    state["balance"]      = round(float(state["balance"]) + net_sell, 2)
    state["daily_profit"] = round(state.get("daily_profit", 0) + pnl, 2)
    state["total_profit"] = round(float(state.get("total_profit", 0)) + pnl, 2)

    print(f"📊 PARTIAL EXIT {trade['symbol']} +{profit_pct*100:.1f}% — locked ₹{pnl:.0f}")
    return True


async def monitor_loop():
    """
    Background loop — runs every 15s.
    Features: partial exit, market regime, flat exit, BTC crash guard.
    """
    while True:
        try:
            state = _load()

            if state["open_trades"]:
                btc_change = await get_btc_change()
                market_bad = btc_change < -3.5

                for trade in list(state["open_trades"]):
                    price = await get_live_price(trade["symbol"])
                    if not price:
                        continue

                    price    = float(price)
                    invested = float(trade["invested"])
                    qty      = float(trade["qty"])
                    entry    = float(trade["entry_price"])
                    hold_h   = (int(time.time() * 1000) - trade["opened_at"]) / 3600000

                    new_stop, new_peak = _calc_trailing_stop(trade, price)

                    trade["current_price"] = price
                    trade["current_value"] = round(qty * price, 2)
                    trade["pnl"]           = round(trade["current_value"] - invested, 2)
                    trade["pnl_pct"]       = round(trade["pnl"] / invested * 100, 2)
                    trade["peak_price"]    = new_peak
                    trade["stop_price"]    = new_stop

                    # #1: Partial exit — lock profit at +5% before full exit
                    did_partial = _do_partial_exit(trade, price, state)
                    if did_partial:
                        _save(state)
                        await asyncio.sleep(0.5)
                        continue

                    close_reason = None
                    drop_pct     = (entry - price) / entry

                    # Time limit (max hold days)
                    max_hold = trade.get("max_hold_until", 0)
                    if max_hold and int(time.time() * 1000) > max_hold:
                        close_reason = "TIME_LIMIT"

                    # #3: Flat exit — coin going nowhere after FLAT_EXIT_HOURS
                    elif hold_h >= FLAT_EXIT_HOURS and not trade.get("partial_exited"):
                        move_pct = abs((price - entry) / entry)
                        if move_pct < FLAT_PCT:
                            close_reason = "FLAT_EXIT"

                    # Take profit
                    elif price >= float(trade["target_price"]):
                        close_reason = "TAKE_PROFIT"

                    # Stop loss / trailing
                    elif price <= new_stop:
                        if drop_pct >= 0.05 or hold_h >= MIN_HOLD_HOURS:
                            if new_peak > entry * (1 + BREAKEVEN_PCT):
                                close_reason = "TRAILING_STOP"
                            else:
                                close_reason = "STOP_LOSS"

                    # BTC market crash
                    elif market_bad:
                        if hold_h >= MIN_HOLD_HOURS:
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
        td["hold_hours"]         = round(hold_h, 1)
        # #6: show stop price info clearly in UI
        td["stop_price_display"] = round(float(t.get("stop_price", 0)), 6)
        td["stop_pct_from_entry"]= round((float(t.get("stop_price",0)) - float(t.get("entry_price",1))) / float(t.get("entry_price",1)) * 100, 2)
        td["partial_exited"]     = bool(t.get("partial_exited", False))
        open_display.append(td)

    return {
        "balance":          round(float(state["balance"]), 2),
        "starting":         float(state["starting"]),
        "portfolio_value":  portfolio,
        "total_return_pct": total_ret,
        "total_profit":     round(total_pnl, 2),
        "total_tds":        round(total_tds, 2),
        "total_tds_paid":   round(total_tds, 2),
        "total_tax":        round(total_tax, 2),
        "total_tax_est":    round(total_tax, 2),
        "net_after_tax":    round(total_pnl - total_tax, 2),
        "total_net_profit": round(total_pnl - total_tax, 2),
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


# #9: Real trade safety checks
def validate_real_trade(symbol: str, amount: float, real_balance: float) -> dict:
    """
    Safety checks before placing any real money trade.
    Returns {ok: bool, reason: str}
    """
    errors = []

    # Never risk more than 2% of real balance per trade
    max_allowed = real_balance * MAX_REAL_POSITION_PCT
    if amount > max_allowed:
        errors.append(f"Amount ₹{amount:.0f} exceeds 2% of balance (₹{max_allowed:.0f})")

    # Minimum meaningful trade size
    if amount < 100:
        errors.append("Minimum trade size is ₹100")

    # Must have a symbol
    if not symbol or not symbol.endswith("USDT"):
        errors.append("Invalid symbol — must end in USDT")

    # Balance sanity
    if real_balance < 500:
        errors.append("Real balance too low — minimum ₹500 to trade safely")

    if errors:
        return {"ok": False, "reason": " | ".join(errors)}
    return {"ok": True, "reason": "All checks passed", "max_per_trade": max_allowed}


async def get_regime_status() -> dict:
    """Expose market regime for dashboard display."""
    return await get_market_regime()