"""
Unit Tests for Baseline Classifier and Evaluation Pipeline.
"""

import sys
import tempfile
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from backend.classifier import AntiSpoofClassifier
from backend.evaluate import evaluate_classifier


def test_classifier_and_eval():
    np.random.seed(42)
    # Generate mock features (100 samples, 45 dimensions)
    X = np.random.randn(100, 45).astype(np.float32)
    # Binary labels (0 = bonafide, 1 = spoof)
    y = np.random.randint(0, 2, size=100).astype(np.int32)
    feature_names = [f"feature_{i}" for i in range(45)]

    clf = AntiSpoofClassifier(model_type="random_forest")
    clf.train(X[:80], y[:80], feature_names=feature_names)

    preds = clf.predict(X[80:])
    probs = clf.predict_proba(X[80:])[:, 1]

    with tempfile.TemporaryDirectory() as tmp_dir:
        metrics = evaluate_classifier(y_true=y[80:], y_pred=preds, y_prob=probs, output_dir=Path(tmp_dir))
        assert "equal_error_rate_eer" in metrics
        assert "accuracy" in metrics
        assert (Path(tmp_dir) / "metrics.json").exists()
        assert (Path(tmp_dir) / "confusion_matrix.png").exists()

        # Save and reload model
        model_save_path = Path(tmp_dir) / "model.joblib"
        clf.save(model_save_path)

        loaded_clf = AntiSpoofClassifier.load(model_save_path)
        single_risk = loaded_clf.predict_spoof_risk(X[80])
        assert 0.0 <= single_risk <= 1.0

    print("[+] Test passed: Classifier training, prediction, serialization, and metrics evaluation verified!")


if __name__ == "__main__":
    test_classifier_and_eval()
