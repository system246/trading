"""
ml_scorer.py — FINAL VERSION
Real ML training from actual closed trades.
Uses LightGBM when 30+ trades available.
Falls back to rule-based when not enough data.
"""

import json
import os
import numpy as np

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

PAPER_FILE = "paper_trades.json"
MODEL_FILE = "ml_model.txt"
MIN_TRADES = 30  # minimum closed trades to train


def _load_trades() -> list:
    """Load closed trades from paper trading history."""
    if not os.path.exists(PAPER_FILE):
        return []
    try:
        data = json.load(open(PAPER_FILE))
        return data.get("closed_trades", [])
    except:
        return []


def _extract_features_from_trade(trade: dict) -> np.ndarray:
    """Extract features from a closed trade for training."""
    def safe(key, default=0.0):
        v = trade.get(key)
        try: return float(v) if v is not None else float(default)
        except: return float(default)

    signal_type = str(trade.get("signal_type", ""))
    features = np.array([
        safe("confidence", 50),
        float("EARLY VOLUME" in signal_type),
        float("REVERSAL"    in signal_type),
        float("SQUEEZE"     in signal_type),
        float("SUPERTREND"  in signal_type),
        float("ACCUMULATION"in signal_type),
        float("FUNDING"     in signal_type),
        safe("position_pct", 10),
    ], dtype=np.float32)
    return np.nan_to_num(features, nan=0.0)


def _extract_features_from_signal(result: dict) -> np.ndarray:
    """Extract features from a live signal for scoring."""
    def safe(key, default=0.0):
        v = result.get(key)
        try: return float(v) if v is not None else float(default)
        except: return float(default)

    signals    = result.get("signals", []) or []
    types      = [s.get("type", "") for s in signals]
    confidence = safe("confidence", 50)

    # Position pct based on confidence
    if confidence >= 95:   pos = 15.0
    elif confidence >= 90: pos = 12.0
    else:                  pos = 10.0

    features = np.array([
        confidence,
        float("EARLY VOLUME" in types),
        float("REVERSAL"     in types),
        float("SQUEEZE"      in types),
        float("SUPERTREND"   in types),
        float("ACCUMULATION" in types),
        float("FUNDING"      in types),
        pos,
    ], dtype=np.float32)
    return np.nan_to_num(features, nan=0.0)


class MLScorer:
    def __init__(self):
        self.model       = None
        self.trained     = False
        self.trade_count = 0
        self._try_load_model()
        # Try to train on startup if enough trades exist
        self._try_train()

    def _try_train(self):
        """Train on existing closed trades if enough data."""
        trades = _load_trades()
        if len(trades) < MIN_TRADES:
            print(f"ML: {len(trades)} trades — need {MIN_TRADES} to train")
            return

        X, y = [], []
        for trade in trades:
            features = _extract_features_from_trade(trade)
            won      = 1 if trade.get("status") == "WIN" else 0
            X.append(features)
            y.append(won)

        X = np.array(X)
        y = np.array(y)

        if len(X) >= MIN_TRADES and HAS_LGB:
            try:
                params = {
                    "objective":     "binary",
                    "metric":        "binary_logloss",
                    "num_leaves":    15,
                    "learning_rate": 0.05,
                    "n_estimators":  50,
                    "verbose":       -1,
                    "min_data_in_leaf": 5,
                }
                self.model   = lgb.LGBMClassifier(**params)
                self.model.fit(X, y)
                self.trained     = True
                self.trade_count = len(trades)
                self._try_save_model()
                print(f"ML: Trained on {len(trades)} trades ✅")

                # Log signal type performance
                self._log_signal_performance(trades)
            except Exception as e:
                print(f"ML training error: {e}")

    def _log_signal_performance(self, trades: list):
        """Log which signal types perform best."""
        stats = {}
        for trade in trades:
            sig_type = str(trade.get("signal_type", "UNKNOWN"))
            won      = trade.get("status") == "WIN"
            pnl      = float(trade.get("pnl_pct", 0))
            if sig_type not in stats:
                stats[sig_type] = {"wins":0,"losses":0,"total_pnl":0}
            if won: stats[sig_type]["wins"]   += 1
            else:   stats[sig_type]["losses"] += 1
            stats[sig_type]["total_pnl"] += pnl

        print("=== Signal Performance ===")
        for sig_type, data in stats.items():
            total = data["wins"] + data["losses"]
            wr    = round(data["wins"]/total*100, 1) if total > 0 else 0
            avg   = round(data["total_pnl"]/total, 2) if total > 0 else 0
            print(f"{sig_type}: {wr}% WR, avg {avg}%, {total} trades")

    def retrain(self):
        """Force retrain with latest trades."""
        self.trained = False
        self.model   = None
        self._try_train()
        return {"trained": self.trained, "trade_count": self.trade_count}

    def score(self, result: dict) -> dict:
        if not result:
            return {"probability":0,"grade":"D","note":"No data","trained":False}
        try:
            features = _extract_features_from_signal(result)
            if self.trained and self.model is not None and HAS_LGB:
                prob = float(self.model.predict_proba(
                    features.reshape(1, -1)
                )[0][1]) * 100
                source = f"ML ({self.trade_count} trades)"
            else:
                prob   = self._heuristic_score(result)
                source = "Rules"

            prob  = round(min(99.0, max(1.0, prob)), 1)
            grade = "A" if prob >= 75 else "B" if prob >= 60 else "C" if prob >= 45 else "D"
            note  = self._grade_note(grade, result, source)
            return {
                "probability": prob,
                "grade":       grade,
                "note":        note,
                "trained":     self.trained,
                "source":      source,
            }
        except Exception as e:
            return {"probability":50,"grade":"C","note":f"Error: {e}","trained":False}

    def _heuristic_score(self, result: dict) -> float:
        """Rule-based fallback."""
        base  = float(result.get("confidence") or 50)
        score = base
        rsi_h = float(result.get("rsi_h") or 50)
        rsi_d = float(result.get("rsi_d") or 50)
        adx   = float(result.get("adx") or 0)
        mtf   = int(result.get("mtf_score") or 0)

        if result.get("macd_crossed"):    score += 8
        if result.get("vol_spike"):       score += 6
        if result.get("bb_squeeze"):      score += 5
        if result.get("supertrend_bull"): score += 7
        if result.get("ema_bullish"):     score += 4
        if result.get("ema_pullback"):    score += 5
        if result.get("whale") and result["whale"].get("whale_buying"): score += 8
        if result.get("speed") and result["speed"].get("fast_mover"):   score += 5
        if rsi_h < 35:  score += 8
        if rsi_d < 40:  score += 6
        if adx > 25:    score += 6
        if mtf >= 3:    score += 8
        if float(result.get("funding") or 0) < -0.0005: score += 5
        if float(result.get("book") or 1) > 1.3:        score += 4
        if len(result.get("signals") or []) >= 3:        score += 6
        if float(result.get("ch24h") or 0) > 15: score -= 15
        if float(result.get("ch7d") or 0) > 30:  score -= 12
        if rsi_h > 70:                            score -= 10

        # Risk score adjustment
        risk = result.get("risk_score", {})
        if risk.get("level") == "HIGH": score -= 10
        if risk.get("level") == "LOW":  score += 5

        return min(95.0, max(5.0, score))

    def _grade_note(self, grade: str, result: dict, source: str) -> str:
        n   = len(result.get("signals") or [])
        mtf = int(result.get("mtf_score") or 0)
        src = f"[{source}]"
        if grade == "A": return f"{src} Strong — {n} signals, {mtf}/4 timeframes. High probability."
        if grade == "B": return f"{src} Good setup — signals aligning. Moderate position."
        if grade == "C": return f"{src} Mixed signals. Wait for confirmation."
        return f"{src} Weak. Better setups available."

    def _try_save_model(self):
        if not HAS_LGB or not self.model: return
        try:
            self.model.booster_.save_model(MODEL_FILE)
        except: pass

    def _try_load_model(self):
        if not HAS_LGB: return
        try:
            if os.path.exists(MODEL_FILE):
                booster      = lgb.Booster(model_file=MODEL_FILE)
                self.model   = booster
                self.trained = True
                print("ML: Loaded existing model ✅")
        except: pass

    def get_insights(self) -> dict:
        """Return ML insights about signal performance."""
        trades = _load_trades()
        if not trades:
            return {"message": "No trades yet"}

        total  = len(trades)
        wins   = [t for t in trades if t.get("status") == "WIN"]
        losses = [t for t in trades if t.get("status") == "LOSS"]

        # Signal type breakdown
        sig_stats = {}
        for trade in trades:
            sig = str(trade.get("signal_type", "UNKNOWN"))
            won = trade.get("status") == "WIN"
            pnl = float(trade.get("pnl_pct", 0))
            if sig not in sig_stats:
                sig_stats[sig] = {"wins":0,"losses":0,"total_pnl":0.0,"trades":0}
            sig_stats[sig]["trades"]    += 1
            sig_stats[sig]["total_pnl"] += pnl
            if won: sig_stats[sig]["wins"]   += 1
            else:   sig_stats[sig]["losses"] += 1

        for sig, data in sig_stats.items():
            t = data["trades"]
            data["win_rate"] = round(data["wins"]/t*100, 1) if t > 0 else 0
            data["avg_pnl"]  = round(data["total_pnl"]/t, 2) if t > 0 else 0

        # Best signal type
        best_sig = max(sig_stats.items(), key=lambda x: x[1]["win_rate"], default=(None,{}))

        return {
            "total_trades":   total,
            "win_rate":       round(len(wins)/total*100, 1) if total > 0 else 0,
            "avg_win":        round(sum(float(t.get("pnl_pct",0)) for t in wins)/len(wins), 2) if wins else 0,
            "avg_loss":       round(sum(float(t.get("pnl_pct",0)) for t in losses)/len(losses), 2) if losses else 0,
            "signal_stats":   sig_stats,
            "best_signal":    best_sig[0],
            "model_trained":  self.trained,
            "trades_trained": self.trade_count,
            "needs_more":     total < MIN_TRADES,
        }
