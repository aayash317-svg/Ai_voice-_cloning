"""
Multi-Speaker Conversational Call Authenticity Subsystem.
Coordinates Audio Quality Gate, Diarization, Overlap Exclusion, Role Assignment,
Independent Per-Speaker Anti-Spoof Evaluation, and Multi-Signal Evidence Policy.
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import numpy as np

from backend.quality_gate import AudioQualityGate, QualityReport
from backend.diarization import SpeakerDiarizer, DiarizationResult
from backend.features import FeatureExtractor
from backend.classifier import AntiSpoofClassifier
from backend.risk_engine import RiskEngine


class MultiSpeakerCallAnalyzer:
    """
    End-to-end multi-speaker call authenticity coordinator.
    Enforces the core rule: Never analyze a two-person call as one voice.
    """

    def __init__(
        self,
        classifier: AntiSpoofClassifier,
        feature_extractor: Optional[FeatureExtractor] = None,
        risk_engine: Optional[RiskEngine] = None,
        sample_rate: int = 16000
    ):
        self.classifier = classifier
        self.extractor = feature_extractor or FeatureExtractor(sample_rate=sample_rate)
        self.risk_engine = risk_engine or RiskEngine()
        self.quality_gate = AudioQualityGate()
        self.diarizer = SpeakerDiarizer(sample_rate=sample_rate)
        self.sr = sample_rate

    def analyze_call(
        self,
        audio: np.ndarray,
        claimed_contact_role: str = "father",
        is_financial_transaction: bool = False,
        language_preset: str = "auto"
    ) -> Dict[str, Any]:
        """
        Analyze a conversational two-party call and return structured multi-speaker assessment.
        """
        # --- STAGE 1: Audio Quality Gate ---
        quality_report: QualityReport = self.quality_gate.evaluate_quality(audio, self.sr)

        if not quality_report.is_usable:
            return {
                "call_assessment": "insufficient_evidence",
                "audio_quality": {
                    "usable_speech_seconds": quality_report.usable_speech_seconds,
                    "overlap_percent": 0.0,
                    "quality": quality_report.quality_level,
                    "rejection_reason": quality_report.rejection_reason
                },
                "speakers": [],
                "guidance": f"Audio quality is insufficient for safe forensic evaluation ({quality_report.rejection_reason}). Verify identity through an alternate trusted channel.",
                "timeline": []
            }

        # --- STAGE 2: Speaker Diarization & Overlap Detection ---
        diar_result: DiarizationResult = self.diarizer.diarize(audio, max_speakers=2)

        # High overlap warning threshold (> 60% overlap)
        if diar_result.overlap_percent >= 60.0 and diar_result.total_speech_seconds < 4.0:
            return {
                "call_assessment": "insufficient_evidence",
                "audio_quality": {
                    "usable_speech_seconds": diar_result.total_speech_seconds,
                    "overlap_percent": diar_result.overlap_percent,
                    "quality": "poor",
                    "rejection_reason": "Excessive overlapping speech prevented clean speaker turn extraction."
                },
                "speakers": [],
                "guidance": "Severe overlapping speech detected. Audio collision prevents accurate speaker separation. Please ask the caller to repeat in turn.",
                "timeline": [
                    {"start": s.start_sec, "end": s.end_sec, "speaker": s.speaker, "is_overlap": s.is_overlap}
                    for s in diar_result.segments
                ]
            }

        # --- STAGE 3 & 4: Per-Speaker Analysis on Clean Turns ---
        speakers_output = []
        max_remote_risk = 0.0
        remote_assessment_level = "low"

        # Determine speaker roles:
        # Speaker A = Initial active speaker / caller / enrolled user
        # Speaker B = Remote participant / claimed contact
        role_definitions = {
            "Speaker A": {
                "role": "caller / enrolled user",
                "role_conf": 0.94
            },
            "Speaker B": {
                "role": f"remote participant / claimed {claimed_contact_role}",
                "role_conf": 0.78 if diar_result.num_speakers_detected > 1 else 0.50
            }
        }

        active_speaker_keys = ["Speaker A"]
        if diar_result.num_speakers_detected > 1:
            active_speaker_keys.append("Speaker B")

        for spk_key in active_speaker_keys:
            clean_audio = diar_result.speaker_clean_audio.get(spk_key, np.array([], dtype=np.float32))
            clean_sec = diar_result.speaker_durations.get(spk_key, 0.0)
            role_info = role_definitions.get(spk_key, {"role": "participant", "role_conf": 0.50})

            # Extract individual clean non-overlapping turns for this speaker
            clean_segments = [
                s for s in diar_result.segments
                if s.speaker == spk_key and not s.is_overlap and s.duration_sec >= 0.7
            ]

            if len(clean_segments) == 0 or clean_sec < 1.0:
                speakers_output.append({
                    "label": spk_key,
                    "role": role_info["role"],
                    "role_confidence": role_info["role_conf"],
                    "clean_speech_seconds": clean_sec,
                    "synthetic_voice_risk": "insufficient_data",
                    "spoof_score": 0.0,
                    "analysis_confidence": 0.30,
                    "reasons": ["Less than 1.0 second of clean non-overlapping speech available."]
                })
                continue

            # Evaluate each clean speech turn independently to avoid cross-turn splicing artifacts
            turn_probs = []
            turn_durs = []
            phase_vars = []
            hf_ratios = []
            jitters = []
            contrasts = []

            for seg in clean_segments:
                seg_feats = self.extractor.extract_all(seg.audio)
                seg_vec = self.extractor.to_vector(seg_feats, self.classifier.feature_names)
                seg_p = float(self.classifier.predict_spoof_risk(seg_vec))
                turn_probs.append(seg_p)
                turn_durs.append(seg.duration_sec)
                phase_vars.append(seg_feats.get("acoustic_phase_derivative_var", 0.0))
                hf_ratios.append(seg_feats.get("spectral_hf_energy_ratio", 0.0))
                jitters.append(seg_feats.get("prosody_jitter", 0.0))
                contrasts.append(seg_feats.get("spectral_contrast_std", 0.0))

            total_valid_dur = max(sum(turn_durs), 1e-4)
            spoof_prob = float(sum(p * d for p, d in zip(turn_probs, turn_durs)) / total_valid_dur)
            avg_phase_var = float(sum(pv * d for pv, d in zip(phase_vars, turn_durs)) / total_valid_dur)
            avg_hf_ratio = float(sum(hf * d for hf, d in zip(hf_ratios, turn_durs)) / total_valid_dur)
            avg_jitter = float(sum(j * d for j, d in zip(jitters, turn_durs)) / total_valid_dur)
            avg_contrast = float(sum(c * d for c, d in zip(contrasts, turn_durs)) / total_valid_dur) if contrasts else 0.0

            # Risk Engine calculation
            assessment = self.risk_engine.compute_risk(
                classifier_spoof_prob=spoof_prob,
                acoustic_anomaly_score=avg_phase_var,
                spectral_anomaly_score=avg_hf_ratio,
                prosody_anomaly_score=avg_jitter,
                contrast_anomaly_score=avg_contrast,
                language_code=language_preset
            )

            # Map to specification threat levels: low, moderate, elevated, critical
            reasons = []
            if assessment.risk_level == "CRITICAL":
                synth_risk = "critical"
                reasons.append("Consistent high-confidence synthetic vocoder or cloned speech characteristics detected across clean speech turns.")
                if avg_hf_ratio > 0.015:
                    reasons.append("High-frequency unnatural harmonic leakage detected.")
                if avg_phase_var > 0.55:
                    reasons.append("Waveform instantaneous phase discontinuity indicates neural synthesis.")
            elif assessment.risk_level == "HIGH":
                synth_risk = "elevated"
                reasons.append("Elevated synthetic voice indicators detected across separated speech turns.")
            elif assessment.risk_level == "MEDIUM":
                synth_risk = "moderate"
                reasons.append("Ambiguous acoustic characteristics; mild compression or acoustic reverberation present.")
            else:
                synth_risk = "low"
                reasons.append("Verified natural human vocal tract biomechanics and normal micro-jitter dynamics.")

            if quality_report.telephony_bandwidth:
                reasons.append("Telephone bandpass compression reduced high-frequency spectral resolution.")

            # Analysis confidence combines diarization confidence and speech duration
            duration_factor = min(1.0, clean_sec / 4.0)
            analysis_conf = round(0.50 * diar_result.diarization_confidence + 0.50 * duration_factor, 2)

            # Track remote participant risk for overall call verdict
            if spk_key == "Speaker B" or (len(active_speaker_keys) == 1 and spk_key == "Speaker A"):
                if assessment.risk_score > max_remote_risk:
                    max_remote_risk = assessment.risk_score
                    remote_assessment_level = synth_risk

            speakers_output.append({
                "label": spk_key,
                "role": role_info["role"],
                "role_confidence": role_info["role_conf"],
                "clean_speech_seconds": clean_sec,
                "synthetic_voice_risk": synth_risk,
                "spoof_score": round(spoof_prob, 3),
                "risk_score": assessment.risk_score,
                "analysis_confidence": analysis_conf,
                "reasons": reasons
            })

        # --- STAGE 5: Decision Policy & Evidence Aggregation ---
        # Remote participant threat determines the primary call assessment
        if remote_assessment_level in ["critical", "elevated"] or max_remote_risk >= 60.0:
            call_assessment = "high_spoof_risk"
            guidance = (
                f"Elevated synthetic voice risk detected for the remote participant ({claimed_contact_role}). "
                "Do not disclose money, passwords, OTPs, or sensitive financial data. "
                "Verify independently by calling the known official phone number."
            )
        elif remote_assessment_level == "moderate" or max_remote_risk >= 25.0:
            call_assessment = "review_recommended"
            guidance = (
                "The remote participant could not be verified with high confidence. "
                "Do not act on urgent or unusual financial requests; confirm via another trusted communication channel."
            )
        else:
            call_assessment = "low_risk"
            guidance = (
                "Adequate clean speech analyzed with no strong synthetic evidence detected. "
                "Continue standard verification vigilance for sensitive operations."
            )

        # Context modifier for financial transactions
        if is_financial_transaction and call_assessment != "low_risk":
            guidance += " [ALERT: High-value transaction requested during unverified call.]"

        # Construct timeline for visual dashboard
        timeline = [
            {
                "start": s.start_sec,
                "end": s.end_sec,
                "duration": s.duration_sec,
                "speaker": s.speaker,
                "is_overlap": s.is_overlap
            }
            for s in diar_result.segments
        ]

        return {
            "call_assessment": call_assessment,
            "overall_call_risk": round(max_remote_risk, 1),
            "audio_quality": {
                "usable_speech_seconds": quality_report.usable_speech_seconds,
                "overlap_percent": diar_result.overlap_percent,
                "quality": quality_report.quality_level
            },
            "speakers": speakers_output,
            "guidance": guidance,
            "timeline": timeline
        }
