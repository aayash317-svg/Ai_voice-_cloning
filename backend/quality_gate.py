"""
Audio Quality Gate & Integrity Subsystem for Conversational Call Audio.
Evaluates usable speech duration, Signal-to-Noise Ratio (SNR), clipping,
and telephony channel degradation before authentication analysis.
"""

from dataclasses import dataclass
from typing import Dict, Any, Tuple
import numpy as np
import librosa


@dataclass
class QualityReport:
    """Detailed audio quality metrics and usability gate determination."""
    usable_speech_seconds: float
    total_duration_seconds: float
    speech_ratio: float
    snr_db: float
    clipping_percent: float
    telephony_bandwidth: bool
    quality_level: str              # 'good', 'moderate', 'poor', 'unusable'
    is_usable: bool
    rejection_reason: str = ""


class AudioQualityGate:
    """
    Evaluates whether input conversational audio satisfies minimum quality criteria
    for safe forensic evaluation. If quality is insufficient, returns explicit
    'insufficient_evidence' instead of guessing on distorted audio.
    """

    def __init__(
        self,
        min_speech_duration: float = 1.5,
        min_snr_db: float = 4.0,
        max_clipping_pct: float = 8.0,
    ):
        self.min_speech_duration = min_speech_duration
        self.min_snr_db = min_snr_db
        self.max_clipping_pct = max_clipping_pct

    def evaluate_quality(self, audio: np.ndarray, sr: int = 16000) -> QualityReport:
        """
        Assess acoustic signal quality and usability.
        """
        total_len = len(audio)
        if total_len == 0:
            return QualityReport(
                usable_speech_seconds=0.0,
                total_duration_seconds=0.0,
                speech_ratio=0.0,
                snr_db=0.0,
                clipping_percent=0.0,
                telephony_bandwidth=False,
                quality_level="unusable",
                is_usable=False,
                rejection_reason="Audio buffer is empty."
            )

        total_duration = total_len / float(sr)

        # 1. Clipping detection (samples near full dynamic range +/- 0.99)
        clipped_samples = np.sum(np.abs(audio) >= 0.985)
        clipping_pct = float((clipped_samples / total_len) * 100.0)

        # 2. Energy-based Voice Activity Detection (VAD)
        # Using 30ms frames with 15ms hop
        frame_len = int(sr * 0.030)
        hop_len = int(sr * 0.015)
        rms = librosa.feature.rms(y=audio, frame_length=frame_len, hop_length=hop_len)[0]

        if len(rms) > 0 and np.max(rms) > 1e-5:
            # Adaptive threshold: 22 dB below 95th percentile speech energy
            p95 = np.percentile(rms, 95)
            speech_thresh = max(1e-4, p95 * 0.08)
            active_frames = np.sum(rms >= speech_thresh)
            speech_ratio = float(active_frames / len(rms))
            usable_speech_sec = float(active_frames * hop_len / sr)

            # 3. WADA-SNR proxy: speech energy vs noise floor (10th percentile)
            noise_floor = np.percentile(rms, 10) + 1e-6
            snr_db = float(20.0 * np.log10((p95 + 1e-6) / noise_floor))
        else:
            usable_speech_sec = 0.0
            speech_ratio = 0.0
            snr_db = 0.0

        # 4. Telephony Bandwidth check (energy confined to 300 - 3400 Hz)
        try:
            fft_mag = np.abs(np.fft.rfft(audio[: min(total_len, sr * 4)]))
            freqs = np.fft.rfftfreq(min(total_len, sr * 4), 1.0 / sr)
            pots_mask = (freqs >= 300) & (freqs <= 3400)
            pots_energy = np.sum(fft_mag[pots_mask] ** 2)
            total_energy = np.sum(fft_mag ** 2) + 1e-9
            telephony_bandwidth = bool((pots_energy / total_energy) > 0.88)
        except Exception:
            telephony_bandwidth = False

        # 5. Quality Level Classification
        is_usable = True
        rejection_reason = ""

        if usable_speech_sec < self.min_speech_duration:
            quality_level = "unusable"
            is_usable = False
            rejection_reason = f"Usable speech duration ({usable_speech_sec:.1f}s) is below required minimum ({self.min_speech_duration}s)."
        elif clipping_pct > self.max_clipping_pct:
            quality_level = "poor"
            is_usable = False
            rejection_reason = f"Severe microphone clipping detected ({clipping_pct:.1f}% of samples distorted)."
        elif snr_db < self.min_snr_db:
            quality_level = "poor"
            is_usable = False
            rejection_reason = f"Excessive background noise or low SNR ({snr_db:.1f} dB)."
        elif snr_db < 10.0 or telephony_bandwidth or clipping_pct > 3.0:
            quality_level = "moderate"
        else:
            quality_level = "good"

        return QualityReport(
            usable_speech_seconds=round(usable_speech_sec, 2),
            total_duration_seconds=round(total_duration, 2),
            speech_ratio=round(speech_ratio, 3),
            snr_db=round(snr_db, 1),
            clipping_percent=round(clipping_pct, 2),
            telephony_bandwidth=telephony_bandwidth,
            quality_level=quality_level,
            is_usable=is_usable,
            rejection_reason=rejection_reason
        )
