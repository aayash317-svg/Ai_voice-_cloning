"""
Classifier Architecture & Training Wrapper for Voice Integrity Verification.
Provides Random Forest Baseline Classifier conforming to training_file.md Section 8.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False

from backend.config import BASELINE_MODEL_PATH, MODELS_DIR


class AntiSpoofClassifier:
    """Baseline binary classifier for Genuine vs Spoof audio detection."""

    def __init__(self, model_type: str = "random_forest", model_params: Optional[dict] = None, calibrated_threshold: float = 0.5):
        self.model_type = model_type.lower()
        self.model_params = model_params or {}
        self.feature_names: List[str] = []
        self.calibrated_threshold: float = calibrated_threshold
        self.model = self._init_model()

    def _init_model(self):
        if self.model_type == "random_forest":
            params = {
                "n_estimators": 200,
                "random_state": 42,
                "class_weight": "balanced",
                "n_jobs": -1,
                **self.model_params
            }
            return RandomForestClassifier(**params)
        elif self.model_type == "xgboost":
            if not HAS_XGBOOST:
                raise ImportError("XGBoost is not available or libomp runtime is missing.")
            params = {
                "n_estimators": 200,
                "max_depth": 6,
                "learning_rate": 0.05,
                "random_state": 42,
                "n_jobs": -1,
                "eval_metric": "logloss",
                **self.model_params
            }
            return XGBClassifier(**params)
        elif self.model_type == "logistic_regression":
            params = {
                "max_iter": 1000,
                "random_state": 42,
                "class_weight": "balanced",
                **self.model_params
            }
            return LogisticRegression(**params)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

    def train(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None):
        """Train classifier on feature matrix X and ground truth labels y."""
        if feature_names:
            self.feature_names = feature_names
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary class labels using calibrated threshold."""
        probs = self.predict_proba(X)[:, 1]
        return (probs >= self.calibrated_threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities: [P(genuine), P(spoof)]."""
        return self.model.predict_proba(X)

    def predict_spoof_risk(self, x: np.ndarray) -> float:
        """Predict single instance spoof probability (0.0 to 1.0)."""
        if x.ndim == 1:
            x = x.reshape(1, -1)
        proba = self.model.predict_proba(x)[0, 1]
        return float(proba)

    def predict_sample(self, x: np.ndarray) -> Dict[str, Any]:
        """Predict single instance returning structured genuine/spoof confidence using calibrated threshold."""
        if x.ndim == 1:
            x = x.reshape(1, -1)
        probs = self.model.predict_proba(x)[0]
        genuine_prob = float(probs[0])
        spoof_prob = float(probs[1])
        predicted_label = "spoof" if spoof_prob >= self.calibrated_threshold else "genuine"

        return {
            "prediction": predicted_label,
            "spoof_probability": round(spoof_prob, 4),
            "genuine_probability": round(genuine_prob, 4),
            "confidence": round(max(genuine_prob, spoof_prob), 4),
            "calibrated_threshold": self.calibrated_threshold
        }

    def save(self, filepath: Union[str, Path] = BASELINE_MODEL_PATH):
        """Save model checkpoint with feature names, config metadata, and calibrated threshold."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "model_type": self.model_type,
            "feature_names": self.feature_names,
            "model_params": self.model_params,
            "calibrated_threshold": self.calibrated_threshold,
            "model": self.model
        }
        joblib.dump(checkpoint, filepath)

    @classmethod
    def load(cls, filepath: Union[str, Path] = BASELINE_MODEL_PATH) -> "AntiSpoofClassifier":
        """Load serialized model checkpoint."""
        filepath = Path(filepath)
        checkpoint = joblib.load(filepath)
        instance = cls(
            model_type=checkpoint["model_type"],
            model_params=checkpoint.get("model_params", {}),
            calibrated_threshold=checkpoint.get("calibrated_threshold", 0.5)
        )
        instance.feature_names = checkpoint.get("feature_names", [])
        instance.model = checkpoint["model"]
        return instance
