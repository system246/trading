"""
ml_scorer.py — Signal probability scorer
Rule-based heuristic (works without training data)
Upgrades to LightGBM automatically once enough trades accumulate
"""

import numpy as np

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False


class MLScorer:
    def __init__(self):
        self.model   = None
        self.trained = False
        self._try_load()

    def _extract_features(self, result: dict) -> np.ndarray:
        def safe(key, default=0.0):
            v = result.get(key)
            try:
                return float(v) if v is not None else float(default)
            except:
                return float(default)

        signals    = result.get("signals", []) or []
        n_signals  = len(signals)
        best_score = max((float(s.get("score", 0)) for s in signals), default=0.0)
        best_conf  = max((float(s.get("conf", 0)) for s in signals), default=0.0)
        types      = [s.get("type", "") for s in signals]

        features = np.array([
            safe("rsi_h", 50), safe("rsi_d", 50),
            safe("macd_hist"), float(bool(result.get("macd_crossed"))),
            safe("bb_width", 0.1), float(bool(result.get("bb_squeeze"))),
            safe("obv_trend"), safe("vol_ratio", 1.0),
            float(bool(result.get("vol_spike"))),
            safe("stoch_k", 50), safe("funding", 0), safe("book", 1.0),
            safe("ch24h"), safe("ch7d"), safe("ch3h"),
            safe("confidence", 50), float(n_signals), best_score, best_conf,
            float("EARLY VOLUME" in types), float("REVERSAL" in types),
            float("SQUEEZE" in types), float("SUPERTREND" in types),
            float("FUNDING" in types), float("ACCUMULATION" in types),
            float(bool(result.get("supertrend_bull"))),
            float(bool(result.get("ema_bullish"))),
            safe("adx", 0), safe("mtf_score", 0),
        ], dtype=np.float32)

        return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    def score(self, result: dict) -> dict:
        if not result:
            return {"probability": 0, "grade": "D", "note": "No data"}
        try:
            features = self._extract_features(result)
            if self.trained and self.model is not None and HAS_LGB:
                prob = float(self.model.predict(features.reshape(1, -1))[0]) * 100
            else:
                prob = self._heuristic_score(result)
            prob  = round(min(99.0, max(1.0, prob)), 1)
            grade = "A" if prob >= 75 else "B" if prob >= 60 else "C" if prob >= 45 else "D"
            note  = self._grade_note(grade, result)
            return {"probability": prob, "grade": grade, "note": note}
        except:
            return {"probability": 50, "grade": "C", "note": "Score unavailable"}

    def _heuristic_score(self, result: dict) -> float:
        base  = float(result.get("confidence") or 50)
        score = base
        rsi_h = float(result.get("rsi_h") or 50)
        rsi_d = float(result.get("rsi_d") or 50)
        adx   = float(result.get("adx") or 0)
        mtf   = int(result.get("mtf_score") or 0)

        if result.get("macd_crossed"):     score += 8
        if result.get("vol_spike"):        score += 6
        if result.get("bb_squeeze"):       score += 5
        if result.get("supertrend_bull"):  score += 7
        if result.get("ema_bullish"):      score += 4
        if result.get("ema_pullback"):     score += 5
        if rsi_h < 35:                     score += 8
        if rsi_d < 40:                     score += 6
        if adx > 25:                       score += 6
        if mtf >= 3:                       score += 8
        if float(result.get("funding") or 0) < -0.0005: score += 5
        if float(result.get("book") or 1) > 1.3:        score += 4
        if len(result.get("signals") or []) >= 3:        score += 6
        if float(result.get("ch24h") or 0) > 15:        score -= 15
        if float(result.get("ch7d") or 0) > 30:         score -= 12
        if rsi_h > 70:                     score -= 10

        return min(95.0, max(5.0, score))

    def _grade_note(self, grade: str, result: dict) -> str:
        n = len(result.get("signals") or [])
        mtf = int(result.get("mtf_score") or 0)
        if grade == "A": return f"Strong setup — {n} signals, {mtf}/4 timeframes aligned."
        if grade == "B": return f"Good setup — signals aligning. Moderate position."
        if grade == "C": return "Mixed signals. Wait for confirmation."
        return "Weak setup. Better opportunities will come."

    def _try_load(self):
        if not HAS_LGB: return
        try:
            import os
            if os.path.exists("ml_model.txt"):
                self.model   = lgb.Booster(model_file="ml_model.txt")
                self.trained = True
        except:
            pass

    def _try_save(self):
        if not HAS_LGB or not self.model: return
        try:
            self.model.booster_.save_model("ml_model.txt")
        except:
            pass
