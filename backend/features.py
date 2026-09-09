"""
Feature Extraction Engine for Voice Integrity Verification Framework.
Extracts 4 independently scored signal families:
1. MFCCs (20 coefficients: mean, std)
2. Spectral Features (Centroid, Bandwidth, Rolloff, Contrast, Flatness, HF energy ratio, ZCR)
3. Prosody & Dynamics (Pitch/F0 stats, Jitter, Shimmer, Voiced ratio, RMS energy stats)
4. Waveform Moments (Kurtosis, Skewness, Instantaneous Phase Variance)
"""

import numpy as np
import scipy.stats
import scipy.signal
import librosa
from typing import Dict, List, Optional, Tuple, Any

from backend.config import SAMPLE_RATE, N_FFT, HOP_LENGTH, N_MFCC


class FeatureExtractor:
    """Extracts fixed-length, validated numerical feature vectors from audio waveforms."""

    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.sr = sample_rate

    def extract_acoustic_features(self, y: np.ndarray) -> Dict[str, float]:
        """Waveform / Phase domain moments and Zero-Crossing Rate."""
        if len(y) < 128:
            return {
                "acoustic_kurtosis": 0.0,
                "acoustic_skewness": 0.0,
                "acoustic_zcr_mean": 0.0,
                "acoustic_zcr_std": 0.0,
                "acoustic_phase_derivative_var": 0.0
            }

        kurt = float(scipy.stats.kurtosis(y))
        skew = float(scipy.stats.skew(y))

        zcr = librosa.feature.zero_crossing_rate(y, frame_length=512, hop_length=256)[0]
        zcr_mean = float(np.mean(zcr))
        zcr_std = float(np.std(zcr))

        # Instantaneous Phase derivative variance via Hilbert transform
        try:
            analytic_signal = scipy.signal.hilbert(y)
            instantaneous_phase = np.unwrap(np.angle(analytic_signal))
            phase_var = float(np.var(np.diff(instantaneous_phase)))
        except Exception:
            phase_var = 0.0

        return {
            "acoustic_kurtosis": kurt if not np.isnan(kurt) else 0.0,
            "acoustic_skewness": skew if not np.isnan(skew) else 0.0,
            "acoustic_zcr_mean": zcr_mean if not np.isnan(zcr_mean) else 0.0,
            "acoustic_zcr_std": zcr_std if not np.isnan(zcr_std) else 0.0,
            "acoustic_phase_derivative_var": phase_var if not np.isnan(phase_var) else 0.0
        }

    def extract_spectral_features(self, y: np.ndarray) -> Dict[str, float]:
        """Frequency-Domain Spectral features and MFCCs."""
        if len(y) < N_FFT:
            y = np.pad(y, (0, N_FFT - len(y)))

        # Spectral Flatness
        flatness = librosa.feature.spectral_flatness(y=y, n_fft=N_FFT, hop_length=HOP_LENGTH)[0]
        flatness_mean = float(np.mean(flatness))

        # Spectral Centroid, Bandwidth, Rolloff
        centroid = librosa.feature.spectral_centroid(y=y, sr=self.sr, n_fft=N_FFT, hop_length=HOP_LENGTH)[0]
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=self.sr, n_fft=N_FFT, hop_length=HOP_LENGTH)[0]
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=self.sr, n_fft=N_FFT, hop_length=HOP_LENGTH)[0]

        # Spectral Contrast (mean & std across sub-bands)
        contrast = librosa.feature.spectral_contrast(y=y, sr=self.sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
        contrast_mean = float(np.mean(contrast))
        contrast_std = float(np.std(contrast))

        # High-frequency energy ratio (> 4kHz)
        fft_mag = np.abs(np.fft.rfft(y))
        freqs = np.fft.rfftfreq(len(y), 1.0 / self.sr)
        hf_mask = freqs >= 4000
        total_energy = np.sum(fft_mag ** 2) + 1e-10
        hf_energy = np.sum(fft_mag[hf_mask] ** 2)
        hf_ratio = float(hf_energy / total_energy)

        # MFCCs (20 coefficients: mean and std)
        mfccs = librosa.feature.mfcc(y=y, sr=self.sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
        mfcc_means = np.mean(mfccs, axis=1)
        mfcc_stds = np.std(mfccs, axis=1)

        features = {
            "spectral_flatness": flatness_mean,
            "spectral_centroid_mean": float(np.mean(centroid)),
            "spectral_centroid_std": float(np.std(centroid)),
            "spectral_bandwidth_mean": float(np.mean(bandwidth)),
            "spectral_bandwidth_std": float(np.std(bandwidth)),
            "spectral_rolloff_mean": float(np.mean(rolloff)),
            "spectral_rolloff_std": float(np.std(rolloff)),
            "spectral_contrast_mean": contrast_mean,
            "spectral_contrast_std": contrast_std,
            "spectral_hf_energy_ratio": hf_ratio,
        }

        for i in range(N_MFCC):
            features[f"mfcc_{i}_mean"] = float(mfcc_means[i])
            features[f"mfcc_{i}_std"] = float(mfcc_stds[i])

        return features

    def extract_prosody_features(self, y: np.ndarray) -> Dict[str, float]:
        """Prosody, Pitch / F0 Dynamics, Jitter, Shimmer, and RMS Energy."""
        # Fast Pitch tracking via YIN algorithm
        try:
            f0 = librosa.yin(
                y,
                fmin=librosa.note_to_hz('C2'),  # ~65 Hz
                fmax=librosa.note_to_hz('C7'),  # ~2093 Hz
                sr=self.sr,
                frame_length=N_FFT,
                hop_length=HOP_LENGTH
            )
            # Voiced frames threshold
            voiced_mask = (f0 > 65.0) & (f0 < 2000.0)
            valid_f0 = f0[voiced_mask]
            voiced_ratio = float(np.mean(voiced_mask.astype(float)))
        except Exception:
            valid_f0 = np.array([])
            voiced_ratio = 0.0

        if len(valid_f0) > 2:
            f0_mean = float(np.mean(valid_f0))
            f0_std = float(np.std(valid_f0))
            f0_range = float(np.ptp(valid_f0))
            periods = 1.0 / (valid_f0 + 1e-6)
            jitter = float(np.mean(np.abs(np.diff(periods))) / (np.mean(periods) + 1e-6))
        else:
            f0_mean, f0_std, f0_range, jitter = 0.0, 0.0, 0.0, 0.0

        # RMS Energy & Shimmer
        rms = librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP_LENGTH)[0]
        rms_mean = float(np.mean(rms))
        rms_std = float(np.std(rms))

        if len(rms) > 2 and rms_mean > 1e-6:
            shimmer = float(np.mean(np.abs(np.diff(rms))) / (rms_mean + 1e-6))
        else:
            shimmer = 0.0

        return {
            "prosody_voiced_ratio": voiced_ratio,
            "prosody_f0_mean": f0_mean,
            "prosody_f0_std": f0_std,
            "prosody_f0_range": f0_range,
            "prosody_jitter": jitter,
            "prosody_shimmer": shimmer,
            "prosody_rms_mean": rms_mean,
            "prosody_rms_std": rms_std
        }

    def extract_all(self, y: np.ndarray) -> Dict[str, float]:
        """Extract composite dictionary of all 4 signal families with NaN/Inf sanitation."""
        features = {}
        features.update(self.extract_acoustic_features(y))
        features.update(self.extract_spectral_features(y))
        features.update(self.extract_prosody_features(y))

        # Sanitize all values
        for k, v in features.items():
            if np.isnan(v) or np.isinf(v):
                features[k] = 0.0
            else:
                features[k] = float(v)

        return features

    def to_vector(self, feature_dict: Dict[str, float], feature_names: Optional[List[str]] = None) -> np.ndarray:
        """Convert feature dictionary to a sorted numpy 1D vector."""
        if feature_names is None:
            feature_names = sorted(feature_dict.keys())
        vec = np.array([feature_dict.get(k, 0.0) for k in feature_names], dtype=np.float32)
        # Replace any NaNs or Infs in vector
        vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
        return vec
