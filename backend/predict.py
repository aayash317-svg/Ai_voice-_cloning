"""
Prediction Pipeline conforming to training_file.md Section 12.
Loads an input audio file, runs preprocessing & feature extraction,
and predicts genuine vs spoof probability using the trained baseline model.
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Union
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import BASELINE_MODEL_PATH, SAMPLE_RATE
from backend.audio_preprocessing import AudioPreprocessor
from backend.features import FeatureExtractor
from backend.classifier import AntiSpoofClassifier
from backend.risk_engine import RiskEngine


class VoicePredictor:
    """End-to-end voice authenticity prediction pipeline."""

    def __init__(self, model_path: Union[str, Path] = BASELINE_MODEL_PATH):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Trained model not found at {self.model_path}. Please train the model first using backend/train.py.")

        self.classifier = AntiSpoofClassifier.load(self.model_path)
        self.preprocessor = AudioPreprocessor(target_sr=SAMPLE_RATE)
        self.extractor = FeatureExtractor(sample_rate=SAMPLE_RATE)
        self.risk_engine = RiskEngine()

    def predict_audio(self, audio_source: Union[str, Path, bytes, np.ndarray]) -> Dict[str, Any]:
        """
        Run complete inference on single audio file or buffer.
        Returns:
            {
                "prediction": "genuine" | "spoof",
                "spoof_probability": float,
                "genuine_probability": float,
                "confidence": float,
                "risk_score": float (0-100),
                "risk_level": "LOW" | "MEDIUM" | "HIGH" | "CRITICAL"
            }
        """
        # 1. Preprocess
        audio = self.preprocessor.load_audio(audio_source)
        audio = self.preprocessor.normalize_amplitude(audio)

        # 2. Extract Features
        feat_dict = self.extractor.extract_all(audio)
        feat_vec = self.extractor.to_vector(feat_dict, self.classifier.feature_names)

        # 3. Model Prediction
        res = self.classifier.predict_sample(feat_vec)

        # 4. Risk Engine Score
        assessment = self.risk_engine.compute_risk(
            classifier_spoof_prob=res["spoof_probability"],
            acoustic_anomaly_score=feat_dict.get("acoustic_phase_derivative_var", 0.0),
            spectral_anomaly_score=feat_dict.get("spectral_hf_energy_ratio", 0.0),
            prosody_anomaly_score=feat_dict.get("prosody_jitter", 0.0)
        )

        res["risk_score"] = assessment.risk_score
        res["risk_level"] = assessment.risk_level
        res["alert_triggered"] = assessment.alert_triggered
        res["explanation"] = assessment.explanation
        return res


def main():
    parser = argparse.ArgumentParser(description="Predict authenticity of an audio file")
    parser.add_argument("--audio", type=str, required=True, help="Path to input audio file (.wav / .flac)")
    parser.add_argument("--model", type=str, default=str(BASELINE_MODEL_PATH), help="Path to model checkpoint")

    args = parser.parse_args()
    predictor = VoicePredictor(model_path=args.model)
    result = predictor.predict_audio(args.audio)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
