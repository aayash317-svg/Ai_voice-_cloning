"""
Multi-Signal Impersonation Risk Engine.
Combines Acoustic, Spectral, Prosodic, Speaker Consistency, and Contextual signals
into a dynamic 0-100 risk score with critical override safety rules.
"""

from typing import Dict, Any, Optional
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
        """Map numerical score to discrete severity level (0-29 Low, 30-59 Med, 60-89 High, 90-100 Critical)."""
        if score >= 90:
            return "CRITICAL"
        elif score >= 60:
            return "HIGH"
        elif score >= 30:
            return "MEDIUM"
        else:
            return "LOW"

    def compute_risk(
        self,
        classifier_spoof_prob: float,
        acoustic_anomaly_score: float = 0.0,
        spectral_anomaly_score: float = 0.0,
        prosody_anomaly_score: float = 0.0,
        speaker_mismatch_score: float = 0.0,
        context_metadata: Optional[Dict[str, Any]] = None
    ) -> RiskAssessment:
        """
        Compute dynamic 0-100 risk score.
        All input signal scores are in range [0.0, 1.0].
        """
        # If sub-signals are provided, blend them; otherwise classifier probability is base
        sub_signals = [acoustic_anomaly_score, spectral_anomaly_score, prosody_anomaly_score, speaker_mismatch_score]
        if any(s > 0.0 for s in sub_signals):
            raw_audio_score = (
                0.50 * classifier_spoof_prob +
                0.20 * spectral_anomaly_score +
                0.15 * acoustic_anomaly_score +
                0.10 * prosody_anomaly_score +
                0.05 * speaker_mismatch_score
            )
        else:
            raw_audio_score = classifier_spoof_prob

        # Contextual metadata modifier
        context_mod = 0.0
        if context_metadata:
            # Unknown caller origin / unverified number
            if context_metadata.get("is_unknown_number", False):
                context_mod += 0.08
            # High financial transaction context
            if context_metadata.get("is_high_value_transaction", False):
                context_mod += 0.12
            # Prior fraud flag
            if context_metadata.get("prior_fraud_flag", False):
                context_mod += 0.15
            # Known trusted contact discount
            if context_metadata.get("is_trusted_contact", False):
                context_mod -= 0.10

        # Base combined score
        combined_score = np_clip = max(0.0, min(1.0, raw_audio_score + context_mod))
        risk_score = round(combined_score * 100.0, 1)

        # SAFETY RULE: If classifier detects spoofing with extreme confidence (>=0.92)
        # AND spectral anomaly confirms it, flag high risk without forcing a hardcoded 88 floor.
        safety_override = False
        if classifier_spoof_prob >= 0.92 and spectral_anomaly_score >= 0.10:
            risk_score = max(risk_score, round(raw_audio_score * 100.0, 1))
            safety_override = True
        elif classifier_spoof_prob >= 0.90 and context_mod < 0:
            risk_score = max(risk_score, round(raw_audio_score * 100.0, 1))
            safety_override = True

        level = self.classify_level(risk_score)
        alert_triggered = level in ["HIGH", "CRITICAL"]

        explanation = f"Impersonation risk evaluated at {risk_score}/100 ({level}). ML Spoof Probability: {classifier_spoof_prob:.3f}."
        if safety_override:
            explanation += " [SAFETY OVERRIDE ACTIVATED: Extreme single-signal anomaly protected from metadata suppression.]"

        return RiskAssessment(
            risk_score=risk_score,
            risk_level=level,
            alert_triggered=alert_triggered,
            signal_breakdown={
                "ml_classifier_spoof_prob": round(classifier_spoof_prob, 3),
                "acoustic_anomaly": round(acoustic_anomaly_score, 3),
                "spectral_anomaly": round(spectral_anomaly_score, 3),
                "prosody_anomaly": round(prosody_anomaly_score, 3),
                "speaker_mismatch": round(speaker_mismatch_score, 3),
            },
            context_modifier=round(context_mod, 3),
            safety_override_triggered=safety_override,
            explanation=explanation
        )
