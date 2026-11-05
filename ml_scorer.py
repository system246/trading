"""
ml_scorer.py — Self-learning ML signal scorer
- Learns from every closed paper trade (win/loss)
- Retrains automatically every 10 new trades
- Falls back to rule-based scoring until 20 trades exist
- Uses RandomForest (no GPU needed, works on Railway)
- Saves model + training data to disk (persists across restarts)
"""

import json
import os
import time
import pickle
import numpy as np
from typing import Optional

# Try to import sklearn — if not installed, falls back to rule-based
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score
    from sklearn.calibration import CalibratedClassifierCV
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False
    print("⚠️  sklearn not installed — pip install scikit-learn")

PAPER_FILE      = "paper_trades.json"
MODEL_FILE      = "ml_model.pkl"
DATA_FILE       = "ml_training_data.json"
BLACKLIST_FILE  = "ml_blacklist.json"
MIN_TRADES      = 20      # need this many closed trades before ML kicks in
RETRAIN_EVERY   = 10      # retrain after every N new trades
MIN_COIN_TRADES = 3       # need at least 3 trades on a coin before auto-blacklisting
BLACKLIST_WINRATE = 0.20  # auto-blacklist if coin win rate < 20%


# ── Auto-blacklist (ML learns bad coins automatically) ──────────

def _load_blacklist() -> dict:
    if os.path.exists(BLACKLIST_FILE):
        try:
            return json.load(open(BLACKLIST_FILE))
        except:
            pass
    return {"coins": {}, "patterns": [], "updated_at": 0}


def _save_blacklist(bl: dict):
    bl["updated_at"] = int(time.time())
    json.dump(bl, open(BLACKLIST_FILE, "w"), indent=2)


def update_blacklist():
    """
    Automatically learn which coins/patterns consistently lose.
    Called after each retrain. No manual work needed.
    """
    if not os.path.exists(PAPER_FILE):
        return

    try:
        state = json.load(open(PAPER_FILE))
    except:
        return

    closed = state.get("closed_trades", [])
    if not closed:
        return

    # Count wins/losses per coin
    coin_stats = {}
    for t in closed:
        sym = t.get("symbol", "").replace("USDT", "")
        if not sym:
            continue
        if sym not in coin_stats:
            coin_stats[sym] = {"wins": 0, "losses": 0, "reasons": []}
        if t.get("status") == "WIN":
            coin_stats[sym]["wins"] += 1
        else:
            coin_stats[sym]["losses"] += 1
            coin_stats[sym]["reasons"].append(t.get("close_reason", ""))

    bl = _load_blacklist()

    for coin, stats in coin_stats.items():
        total = stats["wins"] + stats["losses"]
        if total < MIN_COIN_TRADES:
            continue

        win_rate = stats["wins"] / total
        if win_rate < BLACKLIST_WINRATE:
            if coin not in bl["coins"]:
                print(f"ML AUTO-BLACKLIST: {coin} — {total} trades, {win_rate*100:.0f}% win rate")
                bl["coins"][coin] = {
                    "wins":     stats["wins"],
                    "losses":   stats["losses"],
                    "win_rate": round(win_rate, 3),
                    "top_reason": max(set(stats["reasons"]), key=stats["reasons"].count) if stats["reasons"] else "",
                    "blacklisted_at": int(time.time()),
                }
        else:
            # Remove from blacklist if it recovers
            if coin in bl["coins"]:
                print(f"ML UNBLACKLIST: {coin} — recovered to {win_rate*100:.0f}% win rate")
                del bl["coins"][coin]

    _save_blacklist(bl)
    return bl


def get_blacklist() -> set:
    """Return set of blacklisted coin base symbols (no USDT)."""
    bl = _load_blacklist()
    return set(bl.get("coins", {}).keys())


def is_blacklisted(symbol: str) -> bool:
    """Check if a coin symbol is ML-blacklisted."""
    base = symbol.replace("USDT", "").replace("BUSD", "")
    return base in get_blacklist()


# ── Feature extraction ─────────────────────────────────────────

def extract_features(signal: dict) -> Optional[np.ndarray]:
    """
    Extract numeric features from a scanner signal.
    These are the inputs the model learns from.
    """
    try:
        f = [
            # Price momentum
            float(signal.get("ch3h") or 0),
            float(signal.get("ch24h") or 0),
            float(signal.get("ch7d") or 0),

            # RSI
            float(signal.get("rsi_h") or 50),
            float(signal.get("rsi_d") or 50),
            1.0 if signal.get("rsi_rising") else 0.0,

            # MACD
            float(signal.get("macd_hist") or 0),
            1.0 if signal.get("macd_crossed") else 0.0,

            # Volume
            float(signal.get("vol_ratio") or 1.0),
            1.0 if signal.get("vol_spike") else 0.0,

            # Trend
            1.0 if signal.get("supertrend_bull") else 0.0,
            1.0 if signal.get("ema_bullish") else 0.0,
            1.0 if signal.get("ema_pullback") else 0.0,
            1.0 if signal.get("above_ema21") else 0.0,

            # Bollinger
            float(signal.get("bb_width") or 0),
            1.0 if signal.get("bb_squeeze") else 0.0,

            # OBV
            float(signal.get("obv_trend") or 0),

            # ADX
            float(signal.get("adx") or 0),

            # Stochastic
            float(signal.get("stoch_k") or 50),

            # Multi-timeframe
            float(signal.get("mtf_score") or 0),

            # Signals count + confidence
            float(len(signal.get("signals") or [])),
            float(signal.get("confidence") or 0),

            # Order book
            float(signal.get("book") or 1.0),

            # Funding rate
            float(signal.get("funding") or 0),

            # Vol support at price
            float(signal.get("vol_support") or 0),

            # Coin quality
            float((signal.get("quality") or {}).get("score") or 50),
            float((signal.get("quality") or {}).get("volatility") or 10),
        ]

        # Signal type one-hot
        sig_types = ["REVERSAL", "SQUEEZE", "ACCUMULATION", "SUPERTREND", "FUNDING"]
        active = [s.get("type","") for s in (signal.get("signals") or [])]
        for st in sig_types:
            f.append(1.0 if st in active else 0.0)

        return np.array(f, dtype=np.float32)

    except Exception as e:
        print(f"Feature extraction error: {e}")
        return None


# ── Training data management ────────────────────────────────────

def _load_training_data() -> dict:
    if os.path.exists(DATA_FILE):
        try:
            return json.load(open(DATA_FILE))
        except:
            pass
    return {"samples": [], "last_trade_count": 0}


def _save_training_data(data: dict):
    try:
        json.dump(data, open(DATA_FILE, "w"), indent=2)
    except Exception as e:
        print(f"Training data save error: {e}")


def update_training_data():
    """
    Called after trades close — extracts features from closed trades
    and saves them as training samples (features + label).
    """
    if not os.path.exists(PAPER_FILE):
        return 0

    try:
        state = json.load(open(PAPER_FILE))
    except:
        return 0

    closed = state.get("closed_trades", [])
    td     = _load_training_data()
    existing_ids = {s["trade_id"] for s in td["samples"]}

    added = 0
    for trade in closed:
        tid = trade.get("id", "")
        if tid in existing_ids:
            continue
        if trade.get("status") not in ("WIN", "LOSS"):
            continue

        # Reconstruct signal-like dict from trade fields
        signal_proxy = {
            "ch3h":           trade.get("ch3h", 0),
            "ch24h":          trade.get("ch24h", 0),
            "ch7d":           trade.get("ch7d", 0),
            "rsi_h":          trade.get("rsi_h", 50),
            "rsi_d":          trade.get("rsi_d", 50),
            "rsi_rising":     trade.get("rsi_rising", False),
            "macd_hist":      trade.get("macd_hist", 0),
            "macd_crossed":   trade.get("macd_crossed", False),
            "vol_ratio":      trade.get("vol_ratio", 1.0),
            "vol_spike":      trade.get("vol_spike", False),
            "supertrend_bull":trade.get("supertrend_bull", False),
            "ema_bullish":    trade.get("ema_bullish", False),
            "ema_pullback":   trade.get("ema_pullback", False),
            "above_ema21":    trade.get("above_ema21", False),
            "bb_width":       trade.get("bb_width", 0),
            "bb_squeeze":     trade.get("bb_squeeze", False),
            "obv_trend":      trade.get("obv_trend", 0),
            "adx":            trade.get("adx", 0),
            "stoch_k":        trade.get("stoch_k", 50),
            "mtf_score":      trade.get("mtf_score", 0),
            "confidence":     trade.get("confidence", 0),
            "book":           trade.get("book", 1.0),
            "funding":        trade.get("funding", 0),
            "vol_support":    trade.get("vol_support", 0),
            "quality":        trade.get("quality", {}),
            "signals":        [{"type": trade.get("signal_type", "")}],
        }

        features = extract_features(signal_proxy)
        if features is None:
            continue

        label = 1 if trade["status"] == "WIN" else 0
        td["samples"].append({
            "trade_id": tid,
            "features": features.tolist(),
            "label":    label,
            "pnl_pct":  float(trade.get("pnl_pct", 0)),
            "reason":   trade.get("close_reason", ""),
            "symbol":   trade.get("symbol", ""),
            "added_at": int(time.time() * 1000),
        })
        added += 1

    if added > 0:
        _save_training_data(td)
        print(f"ML: added {added} new training samples ({len(td['samples'])} total)")

    return len(td["samples"])


# ── Model training ──────────────────────────────────────────────

def train_model() -> bool:
    """Train/retrain the model on all available samples."""
    if not SKLEARN_OK:
        return False

    td = _load_training_data()
    samples = td["samples"]

    if len(samples) < MIN_TRADES:
        print(f"ML: only {len(samples)} samples — need {MIN_TRADES} to train")
        return False

    X = np.array([s["features"] for s in samples], dtype=np.float32)
    y = np.array([s["label"] for s in samples], dtype=int)

    wins   = int(y.sum())
    losses = len(y) - wins
    print(f"ML: training on {len(y)} trades ({wins}W/{losses}L)")

    # Handle class imbalance
    class_weight = "balanced"

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Use GradientBoosting for better accuracy with small datasets
    model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        min_samples_leaf=2,
        random_state=42,
    )

    # Wrap with calibration for better probability estimates
    calibrated = CalibratedClassifierCV(model, cv=min(3, len(y)//5 + 1), method="isotonic")

    try:
        calibrated.fit(X_scaled, y)
    except Exception:
        # Fallback to simple RF if calibration fails
        calibrated = RandomForestClassifier(
            n_estimators=100, max_depth=4,
            class_weight=class_weight, random_state=42
        )
        calibrated.fit(X_scaled, y)

    # Cross-val accuracy
    try:
        cv_scores = cross_val_score(
            RandomForestClassifier(n_estimators=50, random_state=42),
            X_scaled, y, cv=min(3, len(y)//5 + 1)
        )
        cv_acc = round(float(cv_scores.mean()) * 100, 1)
    except:
        cv_acc = 0.0

    # Feature importance (only for RF)
    feature_importance = {}
    try:
        if hasattr(calibrated, "feature_importances_"):
            fi = calibrated.feature_importances_
        elif hasattr(calibrated, "base_estimator") and hasattr(calibrated.base_estimator, "feature_importances_"):
            fi = calibrated.base_estimator.feature_importances_
        else:
            fi = None
        if fi is not None:
            names = [
                "ch3h","ch24h","ch7d","rsi_h","rsi_d","rsi_rising",
                "macd_hist","macd_crossed","vol_ratio","vol_spike",
                "supertrend","ema_bullish","ema_pullback","above_ema21",
                "bb_width","bb_squeeze","obv_trend","adx","stoch_k",
                "mtf_score","signals_count","confidence","book","funding",
                "vol_support","quality_score","quality_vol",
                "sig_reversal","sig_squeeze","sig_accum","sig_super","sig_funding"
            ]
            feature_importance = {names[i]: round(float(fi[i]), 4) for i in range(min(len(names), len(fi)))}
    except:
        pass

    # Save model
    try:
        pickle.dump({
            "model":    calibrated,
            "scaler":   scaler,
            "trained_at": int(time.time()),
            "n_samples":  len(samples),
            "cv_accuracy": cv_acc,
            "feature_importance": feature_importance,
        }, open(MODEL_FILE, "wb"))
        print(f"ML: model saved — CV accuracy {cv_acc}%")
        return True
    except Exception as e:
        print(f"ML: model save failed: {e}")
        return False


def _load_model() -> Optional[dict]:
    if not SKLEARN_OK:
        return None
    if not os.path.exists(MODEL_FILE):
        return None
    try:
        return pickle.load(open(MODEL_FILE, "rb"))
    except:
        return None


# ── Scoring ─────────────────────────────────────────────────────

class MLScorer:
    def __init__(self):
        self._model_cache = None
        self._cache_time  = 0
        self._trade_count_at_last_train = 0

    def _get_model(self):
        # Reload model every 10 min in case it was retrained
        if time.time() - self._cache_time > 600:
            self._model_cache = _load_model()
            self._cache_time  = time.time()
        return self._model_cache

    def maybe_retrain(self):
        """
        Call this after each trade closes.
        Automatically retrains if enough new data.
        Also updates the auto-blacklist — no manual coin filtering needed.
        """
        n = update_training_data()
        # Always update blacklist (even before ML activates)
        try:
            update_blacklist()
        except Exception as e:
            print(f"ML blacklist update error: {e}")

        if n >= MIN_TRADES and (n - self._trade_count_at_last_train) >= RETRAIN_EVERY:
            print(f"ML: auto-retraining on {n} samples…")
            success = train_model()
            if success:
                self._trade_count_at_last_train = n
                self._model_cache = None  # force reload

    def score(self, signal: dict) -> dict:
        """
        Score a signal. Returns probability + grade.
        Falls back to rule-based if not enough data yet.
        """
        m = self._get_model()

        # Check if we should auto-retrain
        try:
            self.maybe_retrain()
        except Exception as e:
            print(f"ML retrain check error: {e}")

        if m is None:
            return self._rule_based_score(signal)

        features = extract_features(signal)
        if features is None:
            return self._rule_based_score(signal)

        try:
            X = m["scaler"].transform(features.reshape(1, -1))
            prob = float(m["model"].predict_proba(X)[0][1]) * 100
            prob = round(prob, 1)

            if prob >= 70:   grade = "A"
            elif prob >= 55: grade = "B"
            elif prob >= 40: grade = "C"
            else:            grade = "D"

            return {
                "probability":   prob,
                "grade":         grade,
                "mode":          "ML",
                "n_samples":     m.get("n_samples", 0),
                "cv_accuracy":   m.get("cv_accuracy", 0),
                "trained_at":    m.get("trained_at", 0),
                "note":          f"Trained on {m.get('n_samples',0)} trades",
            }
        except Exception as e:
            print(f"ML score error: {e}")
            return self._rule_based_score(signal)

    def _rule_based_score(self, signal: dict) -> dict:
        """Simple rule-based fallback until ML has enough data."""
        td = _load_training_data()
        n  = len(td["samples"])
        needed = max(0, MIN_TRADES - n)

        score = float(signal.get("confidence") or 50)
        sigs  = len(signal.get("signals") or [])
        if sigs >= 3: score += 5
        if signal.get("rsi_rising"): score += 3
        if signal.get("above_ema21"): score += 3
        if signal.get("vol_spike"): score += 4
        score = min(98.0, score)

        if score >= 70:   grade = "A"
        elif score >= 55: grade = "B"
        elif score >= 40: grade = "C"
        else:             grade = "D"

        return {
            "probability": round(score, 1),
            "grade":       grade,
            "mode":        "RULES",
            "n_samples":   n,
            "note":        f"Need {needed} more trades to activate ML" if needed > 0 else "ML ready — retraining…",
        }

    def get_model_stats(self) -> dict:
        """Return model info for the dashboard."""
        m  = self._get_model()
        td = _load_training_data()
        n  = len(td["samples"])
        wins   = sum(1 for s in td["samples"] if s["label"] == 1)
        losses = n - wins

        bl = _load_blacklist()
        blacklisted_coins = list(bl.get("coins", {}).keys())

        if m:
            top_features = sorted(
                m.get("feature_importance", {}).items(),
                key=lambda x: -x[1]
            )[:5]
            return {
                "active":           True,
                "mode":             "ML",
                "n_samples":        n,
                "wins":             wins,
                "losses":           losses,
                "cv_accuracy":      m.get("cv_accuracy", 0),
                "trained_at":       m.get("trained_at", 0),
                "top_features":     top_features,
                "blacklisted_coins": blacklisted_coins,
                "blacklist_count":  len(blacklisted_coins),
            }
        else:
            return {
                "active":            False,
                "mode":              "RULES",
                "n_samples":         n,
                "wins":              wins,
                "losses":            losses,
                "needed":            max(0, MIN_TRADES - n),
                "note":              f"Need {max(0, MIN_TRADES - n)} more closed trades to activate ML",
                "blacklisted_coins": blacklisted_coins,
                "blacklist_count":   len(blacklisted_coins),
            }