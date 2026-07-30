"""
backtester.py — Historical signal validation using 'ta' library
"""

import asyncio
import httpx
import pandas as pd
import numpy as np
import ta
from typing import Optional

BINANCE_REST = "https://api.binance.com/api/v3"

async def _fetch_daily(symbol, days):
    limit = min(days + 50, 1000)
    url   = f"{BINANCE_REST}/klines?symbol={symbol}&interval=1d&limit={limit}"
    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.get(url)
        if r.status_code != 200: return None
    klines = r.json()
    if len(klines) < 60: return None
    df = pd.DataFrame(klines, columns=[
        "open_time","open","high","low","close","volume",
        "close_time","quote_vol","trades","taker_base","taker_quote","ignore"
    ])
    for col in ["open","high","low","close","volume"]:
        df[col] = pd.to_numeric(df[col])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df.set_index("open_time", inplace=True)
    return df.iloc[-days:]

def _run_backtest(df):
    c = df["close"]
    h = df["high"]
    l = df["low"]
    v = df["volume"]

    try:
        rsi_s    = ta.momentum.RSIIndicator(c, window=14).rsi()
        macd_obj = ta.trend.MACD(c)
        macd_h   = macd_obj.macd_diff()
        vol_avg  = v.rolling(20).mean()
        vol_spike= v > vol_avg * 1.8
    except:
        return None

    trades   = []
    in_trade = False
    entry_price = 0.0
    entry_idx   = 0
    closes = c.values
    n = len(closes)

    for i in range(25, n - 5):
        if in_trade:
            ret      = (closes[i] - entry_price) / entry_price * 100
            days_held= i - entry_idx
            if ret >= 8.0 or ret <= -4.0 or days_held >= 5:
                trades.append({"entry":entry_price,"exit":closes[i],
                               "return_pct":round(ret,2),"days":days_held,"win":ret>0})
                in_trade = False
        else:
            rsi_ok   = rsi_s.iloc[i] < 35
            macd_ok  = macd_h.iloc[i] > 0 and macd_h.iloc[i-1] <= 0
            vol_ok   = vol_spike.iloc[i]
            if rsi_ok and macd_ok and vol_ok:
                entry_price = closes[i]; entry_idx = i; in_trade = True

    # Looser fallback
    if not trades:
        in_trade = False
        for i in range(25, n - 5):
            if in_trade:
                ret = (closes[i] - entry_price) / entry_price * 100
                if ret >= 8.0 or ret <= -4.0 or (i-entry_idx) >= 5:
                    trades.append({"entry":entry_price,"exit":closes[i],
                                   "return_pct":round(ret,2),"days":i-entry_idx,"win":ret>0})
                    in_trade = False
            else:
                if rsi_s.iloc[i] < 38 and macd_h.iloc[i] > 0 and macd_h.iloc[i-1] <= 0:
                    entry_price = closes[i]; entry_idx = i; in_trade = True

    if not trades:
        return {"trades":0,"win_rate":0,"avg_return":0,
                "note":"No clear setups found in this period"}

    wins    = [t for t in trades if t["win"]]
    losses  = [t for t in trades if not t["win"]]
    returns = [t["return_pct"] for t in trades]
    wr      = round(len(wins)/len(trades)*100, 1)
    avg_r   = round(np.mean(returns), 2)
    std_r   = np.std(returns) if len(returns) > 1 else 1
    return {
        "trades":       len(trades),
        "wins":         len(wins),
        "losses":       len(losses),
        "win_rate":     wr,
        "avg_return":   avg_r,
        "avg_win":      round(np.mean([t["return_pct"] for t in wins]),2) if wins else 0,
        "avg_loss":     round(np.mean([t["return_pct"] for t in losses]),2) if losses else 0,
        "best_trade":   round(max(returns),2),
        "worst_trade":  round(min(returns),2),
        "sharpe":       round(avg_r/std_r,2) if std_r>0 else 0,
        "avg_days_held":round(np.mean([t["days"] for t in trades]),1),
        "recent_trades":trades[-10:],
    }

async def backtest_symbol(symbol, days=180):
    df = await _fetch_daily(symbol, days)
    if df is None: return None
    result = _run_backtest(df)
    if result is None: return None
    return {"symbol":symbol,"name":symbol.replace("USDT",""),"period_days":days,**result}
