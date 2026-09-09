"""
Audio Quality Analysis Subsystem
Evaluates SNR, clipping distortion, dynamic range, and overall suitability for biometrics.
"""

from typing import Dict, Any
import numpy as np


class AudioQualityAnalyzer:
    """Analyzes audio quality metrics to ensure signal validity before inference."""

    @staticmethod
    def compute_snr_db(audio: np.ndarray) -> float:
        """
        Estimates Signal-to-Noise Ratio (SNR) in dB using percentile energy comparison.
        Speech energy is estimated by the 90th percentile of frame energy,
        and noise floor is estimated by the 10th percentile.
        """
        if len(audio) < 160:
            return 0.0

        frame_len = 320  # 20ms at 16kHz
        n_frames = len(audio) // frame_len
        if n_frames < 2:
            return 0.0

        frames = audio[: n_frames * frame_len].reshape(n_frames, frame_len)
        frame_energies = np.mean(frames**2, axis=1) + 1e-12

        signal_energy = np.percentile(frame_energies, 90)
        noise_energy = np.percentile(frame_energies, 10)

        if noise_energy <= 1e-12:
            return 40.0  # Cap at +40 dB for pristine/silent noise floor

        ratio = signal_energy / noise_energy
        snr_db = 10.0 * np.log10(max(ratio, 1e-3))
        return float(round(np.clip(snr_db, -10.0, 50.0), 2))

    @staticmethod
    def compute_clipping_ratio(audio: np.ndarray, threshold: float = 0.99) -> float:
        """Computes the fraction of samples saturated near digital maximum."""
        if len(audio) == 0:
            return 0.0
        clipped = np.sum(np.abs(audio) >= threshold)
        return float(round(clipped / len(audio), 4))

    @staticmethod
    def compute_rms(audio: np.ndarray) -> float:
        """Computes Root Mean Square (RMS) amplitude."""
        if len(audio) == 0:
            return 0.0
        return float(round(np.sqrt(np.mean(audio**2)), 4))

    def analyze(self, audio: np.ndarray, sr: int = 16000) -> Dict[str, Any]:
        """Runs comprehensive quality checks on audio segment."""
        duration_sec = round(len(audio) / sr, 3) if sr > 0 else 0.0
        snr_db = self.compute_snr_db(audio)
        clipping_ratio = self.compute_clipping_ratio(audio)
        rms = self.compute_rms(audio)

        # Determine quality tier
        if clipping_ratio > 0.05 or snr_db < 5.0 or rms < 0.005:
            rating = "DEGRADED"
            is_usable = snr_db >= 0.0 and rms >= 0.002
        elif snr_db < 15.0 or clipping_ratio > 0.01:
            rating = "ACCEPTABLE"
            is_usable = True
        else:
            rating = "EXCELLENT"
            is_usable = True

        return {
            "duration_sec": duration_sec,
            "sample_rate": sr,
            "snr_db": snr_db,
            "clipping_ratio": clipping_ratio,
            "rms_energy": rms,
            "quality_rating": rating,
            "is_usable": is_usable,
        }
