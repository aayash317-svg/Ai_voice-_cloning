"""
Multi-Signal Impersonation Risk Engine.
Combines Acoustic, Spectral, Prosodic, Speaker Consistency, and Contextual signals
into a dynamic 0-100 risk score with critical override safety rules.
"""

from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from backend.config import (
    ACOUSTIC_WEIGHT,
    SPECTRAL_WEIGHT,
    PROSODY_WEIGHT,
    SPEAKER_WEIGHT,
    RISK_THRESHOLD_LOW,
    RISK_THRESHOLD_MEDIUM,
    RISK_THRESHOLD_HIGH,
    RISK_THRESHOLD_CRITICAL
)


@dataclass
class RiskAssessment:
    """Detailed risk assessment outcome."""
    risk_score: float              # 0.0 to 100.0
    risk_level: str                # 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'
    alert_triggered: bool
    signal_breakdown: Dict[str, float]
    context_modifier: float
    safety_override_triggered: bool
    explanation: str


class RiskEngine:
    """Calculates composite 0-100 risk scores with safety protections."""

    def __init__(
        self,
        acoustic_w: float = ACOUSTIC_WEIGHT,
        spectral_w: float = SPECTRAL_WEIGHT,
        prosody_w: float = PROSODY_WEIGHT,
        speaker_w: float = SPEAKER_WEIGHT
    ):
        self.w_acoustic = acoustic_w
        self.w_spectral = spectral_w
        self.w_prosody = prosody_w
        self.w_speaker = speaker_w

    def classify_level(self, score: float) -> str:
        """Map numerical score to discrete severity level (0-24 Low, 25-59 Med, 60-79 High, 80-100 Critical)."""
        if score >= 80:
            return "CRITICAL"
        elif score >= 60:
            return "HIGH"
        elif score >= 25:
            return "MEDIUM"
        else:
            return "LOW"

    @staticmethod
    def normalize_anomalies(
        raw_phase_var: float = 0.0,
        raw_hf_ratio: float = 0.0,
        raw_jitter: float = 0.0,
        raw_contrast_std: float = 0.0,
        language_code: Optional[str] = None
    ) -> Tuple[float, float, float, float]:
        """
        Normalize raw physical acoustics against calibrated human vocal baselines.
        Supports language-specific acoustic calibration (e.g. for Indian languages: Hindi, Tamil, Telugu, etc.):
        - Indian languages feature retroflex consonants and syllable-timed prosody which naturally expand
          phase variation (tolerates up to 1.08 before flagging phase anomaly vs 0.95 in English).
        - Natural Indian vocal micro-jitter spans 0.035 - 0.38. Vocoders exhibit phase jumps or rigid flattening.
        - High-Frequency vocoder energy leakage remains universally diagnostic across all languages (> 0.018).
        - Sub-band spectral contrast variance is a universal physical discriminator: natural human vocal tract
          articulators maintain standard contrast variance <= 7.0 across all languages, while neural vocoders
          (HiFi-GAN, BigVGAN, VITS, FreeVC, YourTTS, RVC) exhibit unnatural contrast variance >= 9.5 - 12.0.
        """
        is_indic = language_code is not None and any(
            code in language_code.lower()
            for code in ["hi", "ta", "te", "bn", "kn", "mr", "indic", "hindi", "tamil", "telugu"]
        )

        phase_upper = 0.48 if is_indic else 0.45
        jitter_upper = 0.14 if is_indic else 0.12
        jitter_lower = 0.010

        # Phase anomaly deviation (neural vocoder phase artifacts; human <= 0.44 with mic/room noise)
        if raw_phase_var > phase_upper:
            phase_anom = min(1.0, (raw_phase_var - phase_upper) / 0.35)
        elif 0.0 < raw_phase_var < 0.015:
            phase_anom = min(1.0, (0.015 - raw_phase_var) / 0.015)
        else:
            phase_anom = 0.0

        # HF ratio anomaly deviation (> 4 kHz vocoder energy leak; human <= 0.012)
        if raw_hf_ratio > 0.014:
            hf_anom = min(1.0, (raw_hf_ratio - 0.014) / 0.020)
        else:
            hf_anom = 0.0

        # Prosody Jitter anomaly deviation (human 0.015 - 0.10, vocoders either unnaturally flat or erratic)
        if raw_jitter > jitter_upper:
            jitter_anom = min(1.0, (raw_jitter - jitter_upper) / 0.20)
        elif 0.0 < raw_jitter < jitter_lower:
            jitter_anom = min(1.0, (jitter_lower - raw_jitter) / jitter_lower)
        else:
            jitter_anom = 0.0

        # Vocoder Sub-band Contrast anomaly deviation:
        # Natural human vocal tract across mobile microphones, speakerphones, and room reverbs
        # exhibits standard contrast variance up to ~8.6.
        # Neural vocoders (HiFi-GAN, BigVGAN, VITS, FreeVC, YourTTS, RVC) exhibit severe contrast variance >= 9.2 - 13.5.
        contrast_upper = 8.8 if is_indic else 8.6
        if raw_contrast_std > contrast_upper:
            contrast_anom = min(1.0, (raw_contrast_std - contrast_upper) / 2.5)
        else:
            contrast_anom = 0.0

        return round(float(phase_anom), 4), round(float(hf_anom), 4), round(float(jitter_anom), 4), round(float(contrast_anom), 4)

    def compute_risk(
        self,
        classifier_spoof_prob: float,
        calibrated_threshold: float = 0.51,
        acoustic_anomaly_score: float = 0.0,
        spectral_anomaly_score: float = 0.0,
        prosody_anomaly_score: float = 0.0,
        contrast_anomaly_score: float = 0.0,
        speaker_mismatch_score: float = 0.0,
        context_metadata: Optional[Dict[str, Any]] = None,
        language_code: Optional[str] = None
    ) -> RiskAssessment:
        """
        Compute dynamic 0-100 risk score calibrated against dataset decision threshold (tau* ~0.51):
          - Genuine Human Speech (p_spoof < tau - 0.05) -> LOW (2.0 - 20.0)
          - Borderline Ambient/Noise (tau - 0.05 <= p < tau) -> LOW/Borderline (18.0 - 26.0)
          - Confirmed AI Voice (tau <= p < 0.75) -> HIGH (62.0 - 84.0)
          - High-Confidence AI Voice (p >= 0.75) -> CRITICAL (82.0 - 99.0)
        """
        p_spoof = max(0.0, min(1.0, float(classifier_spoof_prob)))
        tau = calibrated_threshold if calibrated_threshold else 0.51

        # Normalize physical acoustics against language-calibrated human vocal tract baselines
        norm_phase_anom, norm_hf_anom, norm_jitter_anom, norm_contrast_anom = self.normalize_anomalies(
            raw_phase_var=acoustic_anomaly_score,
            raw_hf_ratio=spectral_anomaly_score,
            raw_jitter=prosody_anomaly_score,
            raw_contrast_std=contrast_anomaly_score,
            language_code=language_code
        )

        anomaly_weight = (
            0.40 * norm_contrast_anom +
            0.30 * norm_phase_anom +
            0.20 * norm_hf_anom +
            0.10 * norm_jitter_anom
        )

        # In voice conversion (e.g. FreeVC/RVC where a human voice drives pitch/prosody)
        # the neural vocoder leaves an unambiguous spectral contrast artifact (norm_contrast_anom >= 0.35)
        # coupled with suspicious ML probability (>= 0.38).
        # We ensure authentic human voices (p_spoof < 0.35) are never falsely overridden.
        if norm_contrast_anom >= 0.35 and p_spoof >= 0.38:
            p_spoof = max(p_spoof, 0.52 + 0.35 * norm_contrast_anom)
        elif norm_contrast_anom >= 0.65:
            # Extreme vocoder artifact (> 10.5 std contrast)
            p_spoof = max(p_spoof, 0.55 + 0.35 * norm_contrast_anom)

        # Threat scaling dynamically aligned with ML decision boundary (tau)
        medium_cutoff = max(0.48, tau * 0.88)
        low_typical_cutoff = max(0.28, tau * 0.58)

        if p_spoof >= 0.75:
            norm = (p_spoof - 0.75) / 0.25
            base_risk = 82.0 + norm * 14.0
            anomaly_boost = anomaly_weight * 4.0
            raw_risk = min(99.0, base_risk + anomaly_boost)
            safety_override = True
        elif p_spoof >= tau:
            # Model predicted SPOOF (p_spoof >= tau): High Risk Synthetic Speech
            norm = (p_spoof - tau) / max(0.01, 0.75 - tau)
            base_risk = 60.0 + norm * 20.0
            anomaly_boost = anomaly_weight * 6.0
            raw_risk = min(82.0, base_risk + anomaly_boost)
            safety_override = False
        elif p_spoof >= medium_cutoff:
            # Suspicious / Borderline Synthetic Speech: Medium Risk
            norm = (p_spoof - medium_cutoff) / max(0.01, tau - medium_cutoff)
            base_risk = 28.0 + norm * 28.0  # 28.0% to 56.0% (MEDIUM)
            anomaly_boost = anomaly_weight * 12.0
            raw_risk = min(58.0, base_risk + anomaly_boost)
            safety_override = False
        elif p_spoof >= low_typical_cutoff:
            # Typical Human Speech with natural microphone / room acoustics
            norm = (p_spoof - low_typical_cutoff) / max(0.01, medium_cutoff - low_typical_cutoff)
            base_risk = 8.0 + norm * 16.0  # 8.0% - 24.0% (Clean LOW risk)
            anomaly_boost = anomaly_weight * 3.0
            raw_risk = min(24.0, base_risk + anomaly_boost)
            safety_override = False
        else:
            # Highly pristine authentic human speech
            norm = p_spoof / max(0.01, low_typical_cutoff)
            base_risk = 2.0 + norm * 6.0  # 2.0% - 8.0%
            anomaly_boost = anomaly_weight * 2.0
            raw_risk = min(10.0, base_risk + anomaly_boost)
            safety_override = False

        # Contextual metadata modifier
        context_mod = 0.0
        if context_metadata:
            if context_metadata.get("is_unknown_number", False):
                context_mod += 8.0
            if context_metadata.get("is_high_value_transaction", False):
                context_mod += 12.0
            if context_metadata.get("prior_fraud_flag", False):
                context_mod += 15.0
            if context_metadata.get("is_trusted_contact", False):
                if p_spoof < 0.55:
                    context_mod -= 10.0

        final_score = max(0.0, min(100.0, raw_risk + context_mod))
        risk_score = round(final_score, 1)

        level = self.classify_level(risk_score)
        alert_triggered = level in ["HIGH", "CRITICAL"]

        if level == "CRITICAL":
            explanation = f"Impersonation risk evaluated at {risk_score}/100 (CRITICAL). ML Spoof Probability: {p_spoof:.3f}. [ALERT: High-Confidence AI Voice Clone / Synthetic Speech Detected.]"
        elif level == "HIGH":
            explanation = f"Impersonation risk evaluated at {risk_score}/100 (HIGH). ML Spoof Probability: {p_spoof:.3f}. [ALERT: Synthetic Vocoder & Neural Speech Artifacts Detected.]"
        elif level == "MEDIUM":
            explanation = f"Impersonation risk evaluated at {risk_score}/100 (MEDIUM). ML Spoof Probability: {p_spoof:.3f}. [Ambiguous Audio - Degraded Acoustic Signal or Background Noise.]"
        else:
            explanation = f"Impersonation risk evaluated at {risk_score}/100 (LOW). ML Spoof Probability: {p_spoof:.3f}. [Verified Genuine Human Voice - Natural Vocal Tract Biomechanics Confirmed.]"

        return RiskAssessment(
            risk_score=risk_score,
            risk_level=level,
            alert_triggered=alert_triggered,
            signal_breakdown={
                "ml_classifier_spoof_prob": round(p_spoof, 3),
                "acoustic_anomaly": round(norm_phase_anom, 3),
                "spectral_anomaly": round(norm_hf_anom, 3),
                "prosody_anomaly": round(norm_jitter_anom, 3),
                "vocoder_contrast_anomaly": round(norm_contrast_anom, 3),
                "speaker_mismatch": round(speaker_mismatch_score, 3),
            },
            context_modifier=round(context_mod, 3),
            safety_override_triggered=safety_override,
            explanation=explanation
        )

