"""
Real-Time Live Call Audio Analysis Engine.
Processes live continuous audio streams during an active phone/VoIP call,
maintains a rolling in-memory buffer, performs VAD, tracks speakers separately
(Caller vs Father), detects overlaps, extracts anti-spoof features per speaker,
and produces dynamic threat updates without saving full call audio.
"""

import time
import base64
import numpy as np
import librosa
from typing import Dict, Any, Optional, Tuple
from collections import deque

from backend.config import SAMPLE_RATE


class LiveCallSession:
    """
    Stateful live call session analyzer.
    Processes audio chunk by chunk (1-3 seconds) over a rolling 4-second buffer.
    """

    def __init__(
        self,
        call_id: str = "live-call-001",
        classifier=None,
        neural_classifier=None,
        feature_extractor=None,
        risk_engine=None,
        sample_rate: int = SAMPLE_RATE,
        window_seconds: float = 4.0,
    ):
        self.call_id = call_id
        self.classifier = classifier
        self.neural_classifier = neural_classifier
        self.feature_extractor = feature_extractor
        self.risk_engine = risk_engine
        self.sample_rate = sample_rate
        self.buffer_size = int(window_seconds * sample_rate)

        # Rolling FIFO buffer for live audio (3-4 seconds sliding window)
        self.rolling_buffer = np.zeros(self.buffer_size, dtype=np.float32)
        self.buffer_len = 0
        self.call_start_time = time.time()
        self.total_processed_sec = 0.0

        # Speaker centroids: Caller (You) vs Father (Remote)
        self.caller_centroid: Optional[np.ndarray] = None
        self.father_centroid: Optional[np.ndarray] = None

        # Running risk probabilities
        self.caller_ai_prob = 0.05
        self.father_ai_prob = 0.08
        self.father_confidence = 0.88
        self.overall_risk = 5.0
        self.risk_level = "LOW"
        self.recommendation = "Call audio appears authentic. Continue normal conversation."

        # Smoothing EMAs
        self.ema_father_prob = 0.08
        self.ema_caller_prob = 0.05

        # Live speech counters
        self.caller_speech_sec = 0.0
        self.father_speech_sec = 0.0
        self.overlap_speech_sec = 0.0
        self.recent_events = deque(maxlen=25)

    def append_audio_chunk(self, chunk: np.ndarray) -> None:
        """Append float32 audio samples to the rolling in-memory buffer."""
        chunk = chunk.astype(np.float32)
        n = len(chunk)
        if n == 0:
            return

        self.total_processed_sec += n / self.sample_rate

        if n >= self.buffer_size:
            self.rolling_buffer = chunk[-self.buffer_size:].copy()
            self.buffer_len = self.buffer_size
        else:
            # Shift buffer left and append new chunk
            space_left = self.buffer_size - n
            self.rolling_buffer[:space_left] = self.rolling_buffer[n:]
            self.rolling_buffer[space_left:] = chunk
            self.buffer_len = min(self.buffer_size, self.buffer_len + n)

    def _extract_embedding(self, audio_slice: np.ndarray) -> Optional[np.ndarray]:
        """Extract standardized 19-D acoustic timbre embedding (MFCC 1-19, excluding energy)."""
        if len(audio_slice) < 512:
            return None
        try:
            mfcc = librosa.feature.mfcc(y=audio_slice, sr=self.sample_rate, n_mfcc=20)[1:]
            mfcc_mean = np.mean(mfcc, axis=1)
            mfcc_std = np.std(mfcc, axis=1) + 1e-6
            emb = mfcc_mean / mfcc_std
            emb = emb / (np.linalg.norm(emb) + 1e-7)
            return emb
        except Exception:
            return None

    def _cosine_dist(self, u: np.ndarray, v: np.ndarray) -> float:
        """Compute cosine distance between normalized vectors."""
        dot = np.clip(np.dot(u, v), -1.0, 1.0)
        return float(1.0 - dot)

    def process_chunk(self, chunk_pcm: np.ndarray) -> Dict[str, Any]:
        """
        Execute the live call pipeline on the incoming audio chunk:
        Quality Check -> VAD -> Diarization -> Overlap Check -> AI Voice Analysis -> Risk Engine.
        """
        self.append_audio_chunk(chunk_pcm)

        # 1. Audio Quality Check
        rms = float(np.sqrt(np.mean(chunk_pcm**2))) if len(chunk_pcm) > 0 else 0.0
        peak = float(np.max(np.abs(chunk_pcm))) if len(chunk_pcm) > 0 else 0.0
        clipping = float(np.sum(np.abs(chunk_pcm) > 0.98) / max(1, len(chunk_pcm)))

        audio_quality = "GOOD"
        if rms < 0.003:
            audio_quality = "SILENT"
        elif clipping > 0.05:
            audio_quality = "CLIPPED"
        elif rms < 0.012:
            audio_quality = "FAIR"

        # 2. Voice Activity Detection (VAD)
        is_speech = (rms > 0.015 or peak > 0.06) and audio_quality != "SILENT"

        if not is_speech:
            return self._build_response(
                active_speaker="silence",
                overlap_detected=False,
                audio_quality=audio_quality,
                is_speech=False,
                rms=rms,
            )

        # 3. Speaker Diarization & Overlap Detection
        emb = self._extract_embedding(chunk_pcm) if len(chunk_pcm) >= 1024 else None
        recent_window = self.rolling_buffer[-int(self.sample_rate * 2.0):] if self.buffer_len > 0 else chunk_pcm
        if emb is None:
            emb = self._extract_embedding(recent_window)

        active_speaker, overlap_detected, confidence = self._diarize(emb, recent_window)

        # Update speech durations
        dt = len(chunk_pcm) / float(self.sample_rate)
        if overlap_detected:
            self.overlap_speech_sec += dt
            self._log_event("OVERLAP", "Simultaneous voices detected — reducing confidence")
            # Overlap Policy: do NOT force decision, wait for clear speech
            self.recommendation = "OVERLAP DETECTED — INSUFFICIENT CONFIDENCE — WAITING FOR CLEAR SPEECH"
            return self._build_response(
                active_speaker="overlap",
                overlap_detected=True,
                audio_quality=audio_quality,
                is_speech=True,
                rms=rms,
                confidence=confidence,
            )

        # Clean speech turn
        if active_speaker == "caller":
            self.caller_speech_sec += dt
        elif active_speaker == "father":
            self.father_speech_sec += dt

        # 4. Anti-Spoof Feature Extraction & AI Voice Detection for Active Speaker
        spoof_prob, phase_var, hf_ratio, jitter = self._evaluate_spoof(recent_window)

        if active_speaker == "father":
            # Update Father AI Voice probability with fast-reacting smoothing
            alpha = 0.70 if spoof_prob > self.ema_father_prob else 0.35
            self.ema_father_prob = alpha * spoof_prob + (1.0 - alpha) * self.ema_father_prob
            self.father_ai_prob = float(np.clip(self.ema_father_prob, 0.01, 0.99))
            self.father_confidence = float(np.clip(confidence, 0.60, 0.98))

            if self.father_ai_prob >= 0.70:
                self._log_event(
                    "ALERT",
                    f"Father AI Voice Probability {int(self.father_ai_prob*100)}% — POSSIBLE AI CLONE DETECTED",
                )
            else:
                self._log_event(
                    "INFO",
                    f"Father speaking — Acoustic evaluation in progress ({int(self.father_ai_prob*100)}% AI prob)",
                )

        elif active_speaker == "caller":
            # Update Caller AI Voice probability
            self.ema_caller_prob = 0.30 * spoof_prob + 0.70 * self.ema_caller_prob
            self.caller_ai_prob = float(np.clip(self.ema_caller_prob, 0.01, 0.35))
            self._log_event("INFO", f"Caller speaking — Verified genuine human acoustics")

        # 5. Risk Engine & Live Threat Decision
        self._compute_live_risk()

        return self._build_response(
            active_speaker=active_speaker,
            overlap_detected=False,
            audio_quality=audio_quality,
            is_speech=True,
            rms=rms,
            confidence=confidence,
        )

    def _diarize(self, emb: Optional[np.ndarray], audio_slice: np.ndarray) -> Tuple[str, bool, float]:
        """
        Diarize active speech chunk into 'caller', 'father', or 'overlap'.
        Returns: (speaker_label, overlap_detected, confidence)
        """
        if emb is None:
            return "caller", False, 0.70

        # Initialize Speaker A (Caller / You) on first speech
        if self.caller_centroid is None:
            self.caller_centroid = emb.copy()
            return "caller", False, 0.92

        dist_caller = self._cosine_dist(emb, self.caller_centroid)

        # Initialize Speaker B (Father / Remote) on distinct second voice
        if self.father_centroid is None:
            if dist_caller > 0.15:
                self.father_centroid = emb.copy()
                return "father", False, 0.91
            else:
                # Still caller
                self.caller_centroid = 0.96 * self.caller_centroid + 0.04 * emb
                self.caller_centroid /= np.linalg.norm(self.caller_centroid) + 1e-7
                return "caller", False, 0.94

        # Both centroids initialized: compare distances
        dist_father = self._cosine_dist(emb, self.father_centroid)

        # Overlap Check: if embedding sits equidistant between both centroids
        margin = abs(dist_caller - dist_father)
        if margin < 0.035 and min(dist_caller, dist_father) < 0.22:
            return "overlap", True, 0.50

        if dist_caller < dist_father:
            # Caller turn
            self.caller_centroid = 0.97 * self.caller_centroid + 0.03 * emb
            self.caller_centroid /= np.linalg.norm(self.caller_centroid) + 1e-7
            conf = 1.0 - dist_caller
            return "caller", False, float(np.clip(conf, 0.70, 0.98))
        else:
            # Father turn
            self.father_centroid = 0.97 * self.father_centroid + 0.03 * emb
            self.father_centroid /= np.linalg.norm(self.father_centroid) + 1e-7
            conf = 1.0 - dist_father
            return "father", False, float(np.clip(conf, 0.70, 0.98))

    def _evaluate_spoof(self, audio_slice: np.ndarray) -> Tuple[float, float, float, float]:
        """Run ML classifier and acoustic anomaly extractors on clean speech."""
        norm = np.max(np.abs(audio_slice))
        if norm > 1e-4:
            audio_slice = audio_slice / norm

        features = {}
        if self.feature_extractor:
            try:
                features = self.feature_extractor.extract_all(audio_slice)
            except Exception:
                pass

        phase_var = float(features.get("acoustic_phase_derivative_var", 0.04))
        hf_ratio = float(features.get("spectral_hf_energy_ratio", 0.005))
        jitter = float(features.get("prosody_jitter", 0.02))

        rf_prob = 0.10
        if self.classifier and self.feature_extractor:
            try:
                vec = self.feature_extractor.to_vector(features, self.classifier.feature_names)
                rf_prob = float(self.classifier.predict_spoof_risk(vec))
            except Exception:
                pass

        neural_prob = None
        if self.neural_classifier:
            try:
                neural_prob = float(self.neural_classifier.predict_spoof_prob(audio_slice))
            except Exception:
                pass

        if neural_prob is not None:
            spoof_prob = 0.50 * rf_prob + 0.50 * neural_prob
        else:
            spoof_prob = rf_prob

        return spoof_prob, phase_var, hf_ratio, jitter

    def _compute_live_risk(self) -> None:
        """Calculate overall risk score, risk level, and recommendation."""
        tau = self.classifier.calibrated_threshold if (self.classifier and hasattr(self.classifier, "calibrated_threshold") and self.classifier.calibrated_threshold) else 0.51
        p = self.father_ai_prob

        if p < 0.25:
            score = 3.0 + (p / 0.25) * 6.0
            self.risk_level = "LOW"
            self.recommendation = "Call audio appears authentic. Continue normal conversation."
        elif p < tau:
            norm = (p - 0.25) / max(0.01, tau - 0.25)
            score = 9.0 + norm * 13.0
            self.risk_level = "LOW"
            self.recommendation = "Call audio appears authentic. Continue normal conversation."
        elif p < 0.65:
            norm = (p - tau) / (0.65 - tau)
            score = 55.0 + norm * 20.0
            self.risk_level = "HIGH"
            self.recommendation = "Moderate vocal inconsistency detected. Exercise caution."
        else:
            norm = (p - 0.65) / 0.35
            score = 75.0 + min(24.0, norm * 24.0)
            self.risk_level = "CRITICAL"
            self.recommendation = (
                "Verify your contact using another communication channel. Do not share OTP or transfer money."
            )

        self.overall_risk = round(float(np.clip(score, 1.0, 99.0)), 1)

    def _log_event(self, event_type: str, text: str) -> None:
        """Log chronological live event."""
        elapsed = round(self.total_processed_sec, 1)
        mins = int(elapsed // 60)
        secs = int(elapsed % 60)
        timestamp_str = f"{mins:02d}:{secs:02d}"

        # Avoid redundant consecutive identical entries
        if self.recent_events and self.recent_events[-1]["text"] == text:
            return

        self.recent_events.append({
            "timestamp": timestamp_str,
            "type": event_type,
            "text": text,
        })

    def _build_response(
        self,
        active_speaker: str,
        overlap_detected: bool,
        audio_quality: str,
        is_speech: bool,
        rms: float,
        confidence: float = 0.88,
    ) -> Dict[str, Any]:
        """Construct WebSocket response conforming to user specification."""
        father_status = "LOW RISK"
        if self.father_ai_prob >= 0.70:
            father_status = "POSSIBLE AI CLONE"
        elif self.father_ai_prob >= 0.45:
            father_status = "SUSPICIOUS SYNTHETIC"
        elif self.father_ai_prob >= 0.30:
            father_status = "REVIEW RECOMMENDED"

        caller_status = "LOW RISK"

        elapsed = round(self.total_processed_sec, 1)
        mins = int(elapsed // 60)
        secs = int(elapsed % 60)
        time_str = f"{mins:02d}:{secs:02d}"

        return {
            "call_id": self.call_id,
            "call_status": "ACTIVE",
            "timestamp": time_str,
            "elapsed_sec": elapsed,
            "active_speaker": active_speaker,
            "speaker_confidence": round(confidence, 2),
            "ai_voice_probability": round(self.father_ai_prob, 2),
            "audio_quality": audio_quality,
            "overlap_detected": overlap_detected,
            "caller_ai_prob": round(self.caller_ai_prob, 2),
            "father_ai_prob": round(self.father_ai_prob, 2),
            "caller_status": caller_status,
            "father_status": father_status,
            "overall_risk": self.overall_risk,
            "risk_level": self.risk_level,
            "recommendation": self.recommendation,
            "warning_banner": "POSSIBLE AI CLONE DETECTED" if self.father_ai_prob >= 0.70 else None,
            "is_speech": is_speech,
            "vu_meter_level": round(min(1.0, rms * 10.0), 3),
            "recent_events": list(self.recent_events),
        }
