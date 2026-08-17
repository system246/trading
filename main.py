"""
MarketPulse Python Backend — Final Version v4
"""

import asyncio
import time
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from scanner import full_scan, scan_single, get_btc_health
from backtester import backtest_symbol
from ml_scorer import MLScorer
import paper_trading as pt

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("mp")

cached_results: list = []
cache_time: float    = 0
CACHE_TTL            = 290

scorer    = MLScorer()
scheduler = AsyncIOScheduler()


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Starting MarketPulse backend v4")
    asyncio.create_task(refresh_cache())
    asyncio.create_task(pt.monitor_loop())
    scheduler.add_job(refresh_cache, "interval", minutes=5, id="auto_scan")
    scheduler.start()
    yield
    scheduler.shutdown()


app = FastAPI(title="MarketPulse API", version="4.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


async def refresh_cache():
    global cached_results, cache_time
    log.info("Running full scan...")
    try:
        results = await full_scan()
        for r in results:
            try:
                r["ml_score"] = scorer.score(r)
            except:
                r["ml_score"] = {"probability": 50, "grade": "C", "note": ""}

            if r.get("verdict") == "STRONG_BUY" and float(r.get("confidence", 0)) >= 88:
                try:
                    btc_change = await pt.get_btc_change()
                    if btc_change > -5.0:
                        trade = await pt.auto_buy(r)
                        if trade:
                            log.info(f"Paper trade opened: {r['name']} @ {r['price']}")
                    else:
                        log.info(f"Skipping {r['name']} — BTC down {btc_change:.1f}%")
                except Exception as e:
                    log.error(f"Auto-buy error: {e}")

        cached_results = results
        cache_time     = time.time()
        log.info(f"Scan complete — {len(results)} signals found")
    except Exception as e:
        log.error(f"Scan failed: {e}")


@app.get("/")
def root():
    return {"status": "ok", "version": "4.0.0", "signals": len(cached_results)}


@app.get("/health")
async def health():
    try:
        btc = await get_btc_health()
    except:
        btc = {}
    return {
        "btc":               btc,
        "cache_age_seconds": int(time.time() - cache_time),
        "signals_count":     len(cached_results),
        "next_scan_in":      max(0, CACHE_TTL - int(time.time() - cache_time)),
    }


@app.get("/scan")
async def scan():
    global cached_results, cache_time
    if time.time() - cache_time > CACHE_TTL or not cached_results:
        await refresh_cache()
    return {
        "results":   cached_results,
        "cache_age": int(time.time() - cache_time),
        "count":     len(cached_results),
    }


@app.get("/scan/{symbol}")
async def scan_one(symbol: str):
    symbol = symbol.upper()
    if not symbol.endswith("USDT"):
        symbol += "USDT"
    try:
        result = await scan_single(symbol)
    except Exception as e:
        raise HTTPException(500, str(e))
    if not result:
        raise HTTPException(404, f"No signal for {symbol}")
    try:
        result["ml_score"] = scorer.score(result)
    except:
        result["ml_score"] = {"probability": 50, "grade": "C", "note": ""}
    return result


@app.get("/backtest/{symbol}")
async def backtest(symbol: str, days: int = 180):
    symbol = symbol.upper()
    if not symbol.endswith("USDT"):
        symbol += "USDT"
    try:
        result = await backtest_symbol(symbol, min(max(days, 30), 365))
    except Exception as e:
        raise HTTPException(500, str(e))
    if not result:
        raise HTTPException(404, f"Not enough data for {symbol}")
    return result


@app.get("/rescan")
async def rescan():
    asyncio.create_task(refresh_cache())
    return {"status": "scan_started"}


@app.get("/btc")
async def btc_status():
    return await get_btc_health()


# ── PAPER TRADING ─────────────────────────────────────────────

@app.get("/paper/stats")
def paper_stats():
    try:
        return pt.get_stats()
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/paper/state")
def paper_state():
    return pt.get_state()


@app.post("/paper/toggle")
def paper_toggle(enabled: bool = True):
    return pt.toggle_auto(enabled)


@app.post("/paper/reset")
def paper_reset():
    return pt.reset_portfolio()


@app.post("/paper/sell/{trade_id}")
async def paper_sell(trade_id: str):
    state = pt.get_state()
    trade = next((t for t in state["open_trades"] if t["id"] == trade_id), None)
    if not trade:
        raise HTTPException(404, "Trade not found")
    price = await pt.get_live_price(trade["symbol"])
    if not price:
        raise HTTPException(500, "Could not get live price")
    result = pt.auto_sell(trade_id, price, "MANUAL")
    if not result:
        raise HTTPException(500, "Sell failed")
    return result