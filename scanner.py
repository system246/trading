"""
scanner.py v2 — Enhanced signal detection
NEW: Multi-timeframe, ADX trend strength, volume profile,
     EMA pullback entry, coin quality filter
"""

import asyncio
import time
import httpx
import pandas as pd
import numpy as np
import ta
from typing import Optional


def _clean(obj):
    if isinstance(obj, dict): return {k: _clean(v) for k, v in obj.items()}
    elif isinstance(obj, list): return [_clean(i) for i in obj]
    elif isinstance(obj, np.bool_): return bool(obj)
    elif isinstance(obj, np.integer): return int(obj)
    elif isinstance(obj, np.floating): return float(obj)
    elif isinstance(obj, np.ndarray): return obj.tolist()
    return obj


BINANCE_REST = "https://api.binance.com/api/v3"
BINANCE_FUT  = "https://fapi.binance.com/fapi/v1"
TIMEOUT      = httpx.Timeout(15.0)

SKIP = {
    "BTCUSDT","ETHUSDT","BNBUSDT","USDCUSDT","BUSDUSDT","DAIUSDT",
    "TUSDUSDT","WBTCUSDT","STETHUSDT","WSTETHUSDT","FRAXUSDT",
    "USDPUSDT","USTCUSDT","PAXUSDT","HUSDUSDT","FDUSDUSDT",
    "XRPUSDT","SOLUSDT","ADAUSDT","AVAXUSDT","DOTUSDT","MATICUSDT",
    "LTCUSDT","LINKUSDT","ATOMUSDT","UNIUSDT","XLMUSDT","NEARUSDT",
    "APTUSDT","ALGOUSDT","VETUSDT","ICPUSDT","FILUSDT","ETCUSDT",
    "TRXUSDT","DOGEUSDT","SHIBUSDT","PEPEUSDT","WIFUSDT","BONKUSDT",
}

MIN_VOL    = 300_000
MAX_VOL    = 60_000_000
SCAN_DELAY = 0.15
MAX_COINS  = 200


async def _get(client, url):
    try:
        r = await client.get(url, timeout=TIMEOUT)
        if r.status_code == 200:
            return r.json()
    except:
        pass
    return None


def klines_to_df(klines):
    df = pd.DataFrame(klines, columns=[
        "open_time","open","high","low","close","volume",
        "close_time","quote_vol","trades","taker_base","taker_quote","ignore"
    ])
    for col in ["open","high","low","close","volume","quote_vol"]:
        df[col] = pd.to_numeric(df[col])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df.set_index("open_time", inplace=True)
    return df


def calc_adx(high, low, close, window=14):
    try:
        adx = ta.trend.ADXIndicator(high, low, close, window=window)
        return round(float(adx.adx().iloc[-1]), 1)
    except:
        return None


def calc_volume_profile(df, bins=10):
    try:
        price_range = df["close"].max() - df["close"].min()
        if price_range == 0: return None
        df2 = df.copy()
        df2["bin"] = pd.cut(df2["close"], bins=bins, labels=False)
        profile = df2.groupby("bin")["volume"].sum()
        current_price = df["close"].iloc[-1]
        current_bin = int((current_price - df["close"].min()) / price_range * (bins - 1))
        current_bin = max(0, min(bins - 1, current_bin))
        vol_at_price = float(profile.get(current_bin, 0))
        max_vol = float(profile.max())
        return round(vol_at_price / max_vol, 2) if max_vol > 0 else 0
    except:
        return None


def calc_coin_quality(df_d, vol_usd):
    try:
        days_data = len(df_d)
        closes = df_d["close"].values
        volumes = df_d["volume"].values
        active_days = int(np.sum(volumes > 0))
        consistency = active_days / days_data if days_data > 0 else 0
        returns = np.diff(closes) / closes[:-1]
        volatility = float(np.std(returns) * 100)
        daily_changes = np.abs(returns * 100)
        max_single_day = float(np.max(daily_changes)) if len(daily_changes) > 0 else 0
        pump_risk = bool(max_single_day > 50)
        quality_score = consistency * 100
        if volatility > 20: quality_score -= 20
        if pump_risk: quality_score -= 30
        if vol_usd > 1_000_000: quality_score += 10
        return {
            "score":       round(max(0.0, min(100.0, quality_score)), 1),
            "consistency": round(float(consistency * 100), 1),
            "volatility":  round(volatility, 1),
            "pump_risk":   pump_risk,
        }
    except:
        return None


def calc_indicators(df):
    if len(df) < 30: return {}
    c = df["close"]; h = df["high"]; l = df["low"]; v = df["volume"]
    out = {}

    try:
        out["rsi"] = round(float(ta.momentum.RSIIndicator(c, window=14).rsi().iloc[-1]), 1)
    except: pass

    try:
        macd = ta.trend.MACD(c)
        hist = float(macd.macd_diff().iloc[-1])
        prev = float(macd.macd_diff().iloc[-2])
        out["macd_hist"]    = round(hist, 8)
        out["macd_crossed"] = bool(prev <= 0 and hist > 0)
        out["macd_rising"]  = bool(hist > prev)
    except: pass

    try:
        bb    = ta.volatility.BollingerBands(c, window=20)
        upper = float(bb.bollinger_hband().iloc[-1])
        mid   = float(bb.bollinger_mavg().iloc[-1])
        lower = float(bb.bollinger_lband().iloc[-1])
        price = float(c.iloc[-1])
        bw    = (upper - lower) / mid if mid > 0 else 0
        out["bb_upper"]   = round(upper, 8)
        out["bb_mid"]     = round(mid, 8)
        out["bb_lower"]   = round(lower, 8)
        out["bb_width"]   = round(bw, 4)
        out["bb_squeeze"] = bool(bw < 0.05)
        out["bb_pct"]     = round((price - lower) / (upper - lower), 3) if upper != lower else 0.5
    except: pass

    try:
        out["atr"] = round(float(ta.volatility.AverageTrueRange(h, l, c, window=14).average_true_range().iloc[-1]), 8)
    except: pass

    try:
        ema9  = ta.trend.EMAIndicator(c, window=9).ema_indicator()
        ema21 = ta.trend.EMAIndicator(c, window=21).ema_indicator()
        ema50 = ta.trend.EMAIndicator(c, window=50).ema_indicator()
        out["ema9"]        = round(float(ema9.iloc[-1]), 8)
        out["ema21"]       = round(float(ema21.iloc[-1]), 8)
        out["ema50"]       = round(float(ema50.iloc[-1]), 8)
        out["ema_bullish"] = bool(float(ema9.iloc[-1]) > float(ema21.iloc[-1]))
        price = float(c.iloc[-1])
        ema21_val = float(ema21.iloc[-1])
        prev_price = float(c.iloc[-2]) if len(c) > 1 else price
        out["ema_pullback"] = bool(
            out["ema_bullish"] and
            prev_price <= ema21_val * 1.02 and
            price > prev_price
        )
    except: pass

    try:
        stoch = ta.momentum.StochasticOscillator(h, l, c)
        out["stoch_k"] = round(float(stoch.stoch().iloc[-1]), 1)
        out["stoch_d"] = round(float(stoch.stoch_signal().iloc[-1]), 1)
    except: pass

    try:
        obv_s = ta.volume.OnBalanceVolumeIndicator(c, v).on_balance_volume()
        if len(obv_s) >= 14:
            now = float(obv_s.iloc[-1]); ago = float(obv_s.iloc[-7])
            out["obv_trend"] = round((now / ago - 1) * 100, 1) if ago != 0 else 0.0
    except: pass

    try:
        if len(v) >= 30:
            recent = float(v.iloc[-6:].mean())
            base   = float(v.iloc[-30:-6].mean())
            ratio  = round(recent / base, 2) if base > 0 else 1.0
            out["vol_ratio_6h"] = ratio
            out["vol_spike"]    = bool(ratio > 1.8)
    except: pass

    closes = c.values
    def pct(n):
        return round(float((closes[-1] / closes[-1-n] - 1) * 100), 2) if len(closes) > n else 0.0

    out["ch3h"]  = pct(3)
    out["ch6h"]  = pct(6)
    out["ch24h"] = pct(24)
    out["ch7d"]  = pct(min(168, len(closes) - 1))
    out["adx"]   = calc_adx(h, l, c)

    try:
        atr_s = ta.volatility.AverageTrueRange(h, l, c, window=10).average_true_range()
        hl2   = (h + l) / 2
        upper_band = hl2 + 3.0 * atr_s
        lower_band = hl2 - 3.0 * atr_s
        direction  = pd.Series(0, index=c.index, dtype=int)
        for i in range(1, len(c)):
            if float(c.iloc[i]) > float(upper_band.iloc[i-1]):
                direction.iloc[i] = 1
            elif float(c.iloc[i]) < float(lower_band.iloc[i-1]):
                direction.iloc[i] = -1
            else:
                direction.iloc[i] = int(direction.iloc[i-1])
        out["supertrend_bull"] = bool(int(direction.iloc[-1]) == 1)
    except:
        out["supertrend_bull"] = False

    return out


def calc_mtf_score(ind_15m, ind_1h, ind_4h, ind_1d):
    score = 0
    for ind in [ind_15m, ind_1h, ind_4h, ind_1d]:
        bullish = 0
        if float(ind.get("macd_hist") or 0) > 0: bullish += 1
        if float(ind.get("rsi") or 50) < 60:     bullish += 1
        if ind.get("ema_bullish"):                bullish += 1
        if bullish >= 2: score += 1
    return score


def detect_signals(ind_h, ind_d, ind_4h, ind_15m, price, funding, book, quality):
    signals = []
    rsi_h = ind_h.get("rsi")
    rsi_d = ind_d.get("rsi")
    ch24h = float(ind_h.get("ch24h") or 0)
    ch3h  = float(ind_h.get("ch3h") or 0)
    ch7d  = float(ind_d.get("ch7d") or 0)
    adx   = ind_h.get("adx")

    if abs(ch24h) > 40 or abs(ch3h) > 15: return []
    if quality and quality.get("pump_risk"): return []
    if quality and float(quality.get("score") or 100) < 30: return []

    mtf = calc_mtf_score(ind_15m, ind_h, ind_4h, ind_d)
    mtf_bonus = mtf * 8
    adx_ok    = adx is not None and float(adx) > 20
    adx_bonus = 15 if adx_ok else 0
    pullback_bonus = 12 if ind_h.get("ema_pullback") else 0
    vol_support = float(ind_h.get("vol_support") or 0)
    vp_bonus = 10 if vol_support > 0.6 else 5 if vol_support > 0.4 else 0

    # SIGNAL 1: EARLY VOLUME
    score, conf, reasons = 0, 0, []
    if ind_h.get("vol_spike"):
        score += 35; conf += 1
        reasons.append(f"Volume {ind_h['vol_ratio_6h']}x normal in last 6h")
    if abs(ch24h) < 8 and abs(ch3h) < 4:
        score += 20; conf += 1
        reasons.append("Price quiet — volume moved first (early signal)")
    if float(ind_h.get("obv_trend") or 0) > 10:
        score += 15; conf += 1
        reasons.append(f"OBV rising {ind_h['obv_trend']:.1f}% — real buyers")
    if rsi_h and float(rsi_h) < 60:
        score += 8; reasons.append(f"RSI {rsi_h} — room to move up")
    if book and float(book) > 1.3:
        score += 10; conf += 1; reasons.append(f"Buy orders stacking ({book}x)")
    if ind_h.get("macd_crossed"):
        score += 12; conf += 1; reasons.append("MACD just turned bullish")
    if mtf >= 3:
        score += mtf_bonus; conf += 1; reasons.append(f"MTF confirmed on {mtf}/4 timeframes")
    score += adx_bonus + pullback_bonus + vp_bonus
    if conf >= 2 and score >= 55:
        signals.append({"type":"EARLY VOLUME","icon":"⚡","score":score,"conf":conf,
                        "reasons":reasons,"timeframe":"Hours to 1 day","early_score":min(96,80+score//5)})

    # SIGNAL 2: REVERSAL
    score, conf, reasons = 0, 0, []
    if rsi_h and float(rsi_h) < 35:
        score += 30; conf += 1; reasons.append(f"Hourly RSI oversold at {rsi_h}")
    if rsi_d and float(rsi_d) < 40:
        score += 20; conf += 1; reasons.append(f"Daily RSI oversold at {rsi_d}")
    if ind_h.get("macd_crossed"):
        score += 30; conf += 1; reasons.append("MACD just turned bullish")
    elif ind_h.get("macd_rising"):
        score += 12; reasons.append("MACD improving")
    if float(ind_h.get("vol_ratio_6h") or 1) > 1.5:
        score += 10; conf += 1; reasons.append("Volume picking up on bounce")
    if ch7d < -15:
        score += 8; reasons.append(f"Down {abs(ch7d):.1f}% this week — oversold")
    if ind_h.get("stoch_k") and float(ind_h["stoch_k"]) < 20:
        score += 10; conf += 1; reasons.append(f"Stochastic oversold at {ind_h['stoch_k']:.0f}")
    if mtf >= 2:
        score += mtf_bonus; conf += 1; reasons.append(f"Reversal confirmed on {mtf} timeframes")
    score += adx_bonus + vp_bonus
    if conf >= 2 and score >= 52:
        signals.append({"type":"REVERSAL","icon":"📡","score":score,"conf":conf,
                        "reasons":reasons,"timeframe":"1–3 days","early_score":min(96,76+score//5)})

    # SIGNAL 3: SQUEEZE
    score, conf, reasons = 0, 0, []
    if ind_d.get("bb_squeeze"):
        score += 35; conf += 1; reasons.append("Bollinger Bands compressed — energy building")
    mid_dist = abs(price - float(ind_d.get("bb_mid") or price)) / price * 100 if ind_d.get("bb_mid") else 999
    if mid_dist < 3:
        score += 18; conf += 1; reasons.append("Price coiling at BB midline")
    if float(ind_h.get("vol_ratio_6h") or 1) > 1.4:
        score += 20; conf += 1; reasons.append(f"Volume waking up ({ind_h.get('vol_ratio_6h',1)}x)")
    if rsi_h and 45 < float(rsi_h) < 62:
        score += 10; reasons.append(f"RSI building momentum at {rsi_h}")
    if mtf >= 2:
        score += mtf_bonus; conf += 1; reasons.append(f"Squeeze confirmed on {mtf} timeframes")
    score += adx_bonus
    if conf >= 2 and score >= 55:
        signals.append({"type":"SQUEEZE","icon":"🔒","score":score,"conf":conf,
                        "reasons":reasons,"timeframe":"1–4 days","early_score":min(96,74+score//5)})

    # SIGNAL 4: ACCUMULATION
    score, conf, reasons = 0, 0, []
    if float(ind_d.get("obv_trend") or 0) > 15:
        score += 25; conf += 1; reasons.append(f"OBV rising {ind_d['obv_trend']:.1f}% on daily")
    if abs(ch7d) < 8 and abs(ch24h) < 5:
        score += 20; conf += 1; reasons.append("Price flat for a week — accumulation phase")
    if book and float(book) > 1.4:
        score += 12; conf += 1; reasons.append("Heavy bids in order book")
    if rsi_d and 35 < float(rsi_d) < 55:
        score += 10; reasons.append(f"Daily RSI neutral at {rsi_d}")
    if ind_d.get("ema_bullish"):
        score += 8; conf += 1; reasons.append("EMA9 above EMA21 — uptrend forming")
    if vol_support > 0.5:
        score += 15; conf += 1; reasons.append("High volume accumulated at current price")
    score += mtf_bonus // 2
    if conf >= 2 and score >= 60:
        signals.append({"type":"ACCUMULATION","icon":"🐋","score":score,"conf":conf,
                        "reasons":reasons,"timeframe":"3–7 days","early_score":min(96,78+score//5)})

    # SIGNAL 5: SUPERTREND
    score, conf, reasons = 0, 0, []
    if ind_d.get("supertrend_bull") and ind_h.get("supertrend_bull"):
        score += 40; conf += 2; reasons.append("Supertrend bullish on daily + hourly")
    elif ind_h.get("supertrend_bull"):
        score += 22; conf += 1; reasons.append("Supertrend turned bullish on hourly")
    if ind_h.get("ema_bullish"):
        score += 15; conf += 1; reasons.append("EMA stack aligned bullish")
    if ind_h.get("vol_spike"):
        score += 15; conf += 1; reasons.append("Volume surge confirms breakout")
    if rsi_h and 50 < float(rsi_h) < 70:
        score += 8; reasons.append(f"RSI in momentum zone at {rsi_h}")
    if adx_ok:
        score += adx_bonus; conf += 1; reasons.append(f"ADX {adx} — strong trend confirmed")
    if mtf >= 3:
        score += mtf_bonus; conf += 1; reasons.append(f"Trend confirmed on {mtf}/4 timeframes")
    if ind_h.get("ema_pullback"):
        score += pullback_bonus; reasons.append("Perfect pullback to EMA — ideal entry")
    if conf >= 2 and score >= 58:
        signals.append({"type":"SUPERTREND","icon":"🚀","score":score,"conf":conf,
                        "reasons":reasons,"timeframe":"1–5 days","early_score":min(96,75+score//5)})

    # SIGNAL 6: FUNDING
    if funding is not None and float(funding) < -0.0005:
        score, conf, reasons = 40, 1, []
        reasons.append(f"Funding {float(funding)*100:.4f}% — shorts paying longs")
        if rsi_h and float(rsi_h) < 45:
            score += 22; conf += 1; reasons.append(f"RSI oversold at {rsi_h}")
        if float(ind_h.get("obv_trend") or 0) > 0:
            score += 18; conf += 1; reasons.append("Spot buyers absorbing")
        if conf >= 1 and score >= 50:
            signals.append({"type":"FUNDING","icon":"💰","score":score,"conf":conf,
                            "reasons":reasons,"timeframe":"Hours to 1 day","early_score":min(96,72+score//5)})

    signals.sort(key=lambda x: -x["early_score"])
    return signals


def calc_verdict(signals, ind_h, ind_d, ch24h, ch7d):
    ch24h = float(ch24h or 0)
    ch7d  = float(ch7d or 0)
    if ch24h > 20:
        return {"verdict":"DANGER","confidence":95,"reason":"Already pumped hard. Skip.","risk":"VERY_HIGH"}
    if ch7d > 40:
        return {"verdict":"DANGER","confidence":90,"reason":"Up 40%+ this week. Easy money gone.","risk":"VERY_HIGH"}
    if not signals:
        return {"verdict":"AVOID","confidence":20,"reason":"No clear signal right now.","risk":"HIGH"}

    best  = signals[0]
    score = float(best["early_score"])
    adx   = ind_h.get("adx")

    if ind_h.get("macd_crossed"):     score += 10
    rsi_d = ind_h.get("rsi")
    rsi_d2 = ind_d.get("rsi")
    if rsi_d2 and float(rsi_d2) < 30:  score += 12
    if rsi_d2 and float(rsi_d2) > 75:  score -= 15
    if ind_h.get("supertrend_bull"):   score += 8
    if ind_d.get("supertrend_bull"):   score += 5
    if adx and float(adx) > 25:        score += 8
    if ind_h.get("ema_pullback"):      score += 6
    score = min(98.0, max(5.0, score))

    n = len(signals)
    if score >= 82 and n >= 2:
        v, risk = "STRONG_BUY", "LOW"
        pts = []
        if ind_h.get("vol_spike"):    pts.append(f"Volume {ind_h.get('vol_ratio_6h',0)}x above normal.")
        if ind_h.get("macd_crossed"): pts.append("Momentum just turned bullish.")
        if adx and float(adx) > 25:   pts.append(f"Strong trend (ADX {adx}).")
        if ind_h.get("ema_pullback"): pts.append("Perfect pullback entry.")
        reason = " ".join(pts[:3]) or "Multiple strong signals align."
    elif score >= 68:
        v, risk = "BUY", "MEDIUM"
        reason = f"Good setup — {n} signal(s). Consider moderate position."
    elif score >= 52:
        v, risk = "WATCH", "MEDIUM"
        reason = "Signs forming. Wait for volume confirmation."
    else:
        v, risk = "AVOID", "HIGH"
        reason = "No clear signal. Better opportunities will come."

    return {"verdict": v, "confidence": round(score), "reason": reason, "risk": risk}


def calc_targets(price, ind_h, ind_d):
    atr      = float(ind_h.get("atr") or ind_d.get("atr") or price * 0.03)
    bb_upper = float(ind_d.get("bb_upper") or price * 1.10)
    bb_lower = float(ind_d.get("bb_lower") or price * 0.93)
    target1  = round(price + 2.0 * atr, 8)
    target2  = round(max(bb_upper, price + 3.0 * atr), 8)
    stop     = round(max(price - 1.5 * atr, bb_lower), 8)
    up1  = round((target1 - price) / price * 100, 1)
    up2  = round((target2 - price) / price * 100, 1)
    down = round((price - stop) / price * 100, 1)
    rr   = round(up1 / down, 1) if down > 0 else 0
    return {"target1":target1,"target2":target2,"stop":stop,
            "upside1":f"+{up1}%","upside2":f"+{up2}%","downside":f"-{down}%","risk_reward":rr}


async def fetch_coin_data(client, symbol):
    urls = [
        f"{BINANCE_REST}/klines?symbol={symbol}&interval=15m&limit=60",
        f"{BINANCE_REST}/klines?symbol={symbol}&interval=1h&limit=96",
        f"{BINANCE_REST}/klines?symbol={symbol}&interval=4h&limit=60",
        f"{BINANCE_REST}/klines?symbol={symbol}&interval=1d&limit=120",
        f"{BINANCE_FUT}/fundingRate?symbol={symbol}&limit=1",
        f"{BINANCE_REST}/depth?symbol={symbol}&limit=10",
    ]
    results = await asyncio.gather(*[_get(client, u) for u in urls], return_exceptions=True)
    m15_raw, h1_raw, h4_raw, d1_raw, f_raw, b_raw = results

    if not isinstance(h1_raw, list) or len(h1_raw) < 24: return None
    if not isinstance(d1_raw, list) or len(d1_raw) < 30: return None

    df_15m = klines_to_df(m15_raw) if isinstance(m15_raw, list) and len(m15_raw) >= 20 else None
    df_1h  = klines_to_df(h1_raw)
    df_4h  = klines_to_df(h4_raw) if isinstance(h4_raw, list) and len(h4_raw) >= 20 else None
    df_1d  = klines_to_df(d1_raw)

    funding = None
    if isinstance(f_raw, list) and f_raw:
        try: funding = float(f_raw[0].get("fundingRate", 0))
        except: pass

    book = None
    if isinstance(b_raw, dict):
        bids = sum(float(x[1]) for x in b_raw.get("bids", []))
        asks = sum(float(x[1]) for x in b_raw.get("asks", []))
        book = round(bids / asks, 2) if asks > 0 else None

    return {"df_15m":df_15m,"df_1h":df_1h,"df_4h":df_4h,"df_1d":df_1d,"funding":funding,"book":book}


def _build_result(symbol, price, df_15m, df_1h, df_4h, df_1d, funding, book, ch24h, vol_usd=0):
    ind_15m = calc_indicators(df_15m) if df_15m is not None else {}
    ind_1h  = calc_indicators(df_1h)
    ind_4h  = calc_indicators(df_4h) if df_4h is not None else {}
    ind_1d  = calc_indicators(df_1d)
    ind_1h["ch24h"] = float(ch24h)

    vol_support = calc_volume_profile(df_1h)
    if vol_support is not None:
        ind_1h["vol_support"] = vol_support

    quality = calc_coin_quality(df_1d, vol_usd)

    signals = detect_signals(ind_1h, ind_1d, ind_4h, ind_15m, price, funding, book, quality)
    if not signals:
        return None

    verdict = calc_verdict(signals, ind_1h, ind_1d, ch24h, ind_1d.get("ch7d", 0))
    targets = calc_targets(price, ind_1h, ind_1d)

    result = {
        "symbol":          symbol,
        "name":            symbol.replace("USDT", ""),
        "price":           round(float(price), 8),
        "ch24h":           float(ch24h),
        "ch7d":            float(ind_1d.get("ch7d") or 0),
        "ch3h":            float(ind_1h.get("ch3h") or 0),
        "rsi_h":           ind_1h.get("rsi"),
        "rsi_d":           ind_1d.get("rsi"),
        "macd_hist":       ind_1h.get("macd_hist"),
        "macd_crossed":    bool(ind_1h.get("macd_crossed", False)),
        "bb_upper":        ind_1d.get("bb_upper"),
        "bb_lower":        ind_1d.get("bb_lower"),
        "bb_width":        ind_1d.get("bb_width"),
        "bb_squeeze":      bool(ind_1d.get("bb_squeeze", False)),
        "atr":             ind_1h.get("atr"),
        "adx":             ind_1h.get("adx"),
        "supertrend_bull": bool(ind_1h.get("supertrend_bull", False)),
        "obv_trend":       ind_1h.get("obv_trend"),
        "vol_ratio":       ind_1h.get("vol_ratio_6h"),
        "vol_spike":       bool(ind_1h.get("vol_spike", False)),
        "vol_support":     vol_support,
        "stoch_k":         ind_1h.get("stoch_k"),
        "ema_bullish":     bool(ind_1h.get("ema_bullish", False)),
        "ema_pullback":    bool(ind_1h.get("ema_pullback", False)),
        "mtf_score":       calc_mtf_score(ind_15m, ind_1h, ind_4h, ind_1d),
        "quality":         quality,
        "funding":         funding,
        "book":            book,
        "signals":         signals,
        "targets":         targets,
        "closes":          [float(x) for x in df_1d["close"].tolist()[-60:]],
        "verdict":         verdict["verdict"],
        "confidence":      verdict["confidence"],
        "reason":          verdict["reason"],
        "risk":            verdict["risk"],
        "detected_at":     int(time.time() * 1000),
    }
    return _clean(result)


async def scan_single(symbol):
    async with httpx.AsyncClient() as client:
        data = await fetch_coin_data(client, symbol)
    if not data: return None
    price = float(data["df_1h"]["close"].iloc[-1])
    return _build_result(symbol, price, data["df_15m"], data["df_1h"],
                         data["df_4h"], data["df_1d"], data["funding"], data["book"], 0)


async def get_all_pairs(client):
    data = await _get(client, f"{BINANCE_REST}/ticker/24hr")
    if not data: return []
    pairs = []
    for t in data:
        sym = t.get("symbol", "")
        if not sym.endswith("USDT"): continue
        if sym in SKIP: continue
        if any(x in sym for x in ["UP","DOWN","BULL","BEAR","3L","3S"]): continue
        vol   = float(t.get("quoteVolume", 0))
        price = float(t.get("lastPrice", 0))
        if vol < MIN_VOL or vol > MAX_VOL: continue
        if price < 0.000001: continue
        pairs.append({"symbol":sym,"price":price,
                      "ch24h":float(t.get("priceChangePercent", 0)),
                      "vol":vol,"count":int(t.get("count", 0))})
    pairs.sort(key=lambda x: -x["count"])
    return pairs[:MAX_COINS]


async def full_scan():
    sem = asyncio.Semaphore(5)
    results = []
    async with httpx.AsyncClient() as client:
        pairs = await get_all_pairs(client)

        async def scan_one(ticker):
            async with sem:
                data = await fetch_coin_data(client, ticker["symbol"])
                await asyncio.sleep(SCAN_DELAY)
                if not data: return
                price = float(data["df_1h"]["close"].iloc[-1])
                r = _build_result(ticker["symbol"], price, data["df_15m"],
                                  data["df_1h"], data["df_4h"], data["df_1d"],
                                  data["funding"], data["book"],
                                  ticker.get("ch24h", 0), ticker.get("vol", 0))
                if r: results.append(r)

        await asyncio.gather(*[scan_one(t) for t in pairs])

    order = {"STRONG_BUY":0,"BUY":1,"WATCH":2,"AVOID":3,"DANGER":4,"STALE":5}
    results.sort(key=lambda r: (order.get(r.get("verdict","AVOID"), 6), -r.get("confidence", 0)))
    return results


async def get_btc_health():
    async with httpx.AsyncClient() as client:
        tick   = await _get(client, f"{BINANCE_REST}/ticker/24hr?symbol=BTCUSDT")
        klines = await _get(client, f"{BINANCE_REST}/klines?symbol=BTCUSDT&interval=1d&limit=30")
    if not tick or not klines:
        return {"mood":"UNKNOWN","label":"Checking…","color":"gray"}
    closes = [float(k[4]) for k in klines]
    ch24h  = float(tick.get("priceChangePercent", 0))
    price  = float(tick.get("lastPrice", 0))
    gains  = [max(closes[i]-closes[i-1], 0) for i in range(1, len(closes))]
    losses = [max(closes[i-1]-closes[i], 0) for i in range(1, len(closes))]
    ag = float(np.mean(gains[-14:]))
    al = float(np.mean(losses[-14:]))
    rsi_v = round(100 - 100 / (1 + ag/al), 1) if al > 0 else 100.0
    if ch24h < -5 or rsi_v < 30:   mood, label, color = "BAD",     "Bad — Stay Careful",  "red"
    elif ch24h < -2:                mood, label, color = "WEAK",    "Weak Market",          "orange"
    elif rsi_v > 75:                mood, label, color = "HOT",     "Market Overheated",    "yellow"
    elif ch24h > 3 and rsi_v < 68: mood, label, color = "GOOD",    "Good Day for Crypto",  "green"
    else:                           mood, label, color = "NEUTRAL", "Market is Calm",       "gray"
    return {"mood":mood,"label":label,"color":color,"btc_price":price,"ch24h":ch24h,"rsi":rsi_v}
