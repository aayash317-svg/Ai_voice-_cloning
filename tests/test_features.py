import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from backend.features import FeatureExtractor


def test_feature_extraction():
    extractor = FeatureExtractor(sample_rate=16000)
    # Generate 3.0s synthetic test signal (mixture of sines + white noise)
    t = np.linspace(0, 3.0, 16000 * 3, endpoint=False)
    synthetic_speech = (
        0.5 * np.sin(2 * np.pi * 220 * t) +
        0.3 * np.sin(2 * np.pi * 440 * t) +
        0.05 * np.random.randn(len(t))
    ).astype(np.float32)

    # 1. Acoustic features
    acoustic = extractor.extract_acoustic_features(synthetic_speech)
    assert "acoustic_kurtosis" in acoustic
    assert "acoustic_phase_derivative_var" in acoustic
    assert "acoustic_zcr_mean" in acoustic

    # 2. Spectral features
    spectral = extractor.extract_spectral_features(synthetic_speech)
    assert "spectral_flatness" in spectral
    assert "spectral_centroid_mean" in spectral
    assert "spectral_hf_energy_ratio" in spectral
    assert "mfcc_0_mean" in spectral

    # 3. Prosody features
    prosody = extractor.extract_prosody_features(synthetic_speech)
    assert "prosody_voiced_ratio" in prosody
    assert "prosody_f0_mean" in prosody
    assert "prosody_jitter" in prosody

    # 4. Composite extraction and vector conversion
    all_feats = extractor.extract_all(synthetic_speech)
    vec = extractor.to_vector(all_feats)
    assert len(vec) > 30
    assert not np.isnan(vec).any()
    print(f"[+] Test passed: Extracted {len(vec)} feature dimensions successfully without NaN values!")


if __name__ == "__main__":
    test_feature_extraction()
