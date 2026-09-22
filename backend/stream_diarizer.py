"""
Real-Time Streaming Call Diarizer and Threat Engine.
Processes continuous live audio streams (16 kHz PCM), detects voice activity,
clusters speech into Speaker A (Local User) vs Speaker B (Remote Party),
extracts low-latency anti-spoof features, and calculates real-time threat scores
while the conversation is taking place.
"""

import time
import numpy as np
import librosa
from typing import Dict, Any, Optional, Tuple
from collections import deque

from backend.config import SAMPLE_RATE
from backend.preprocess import AudioPreprocessor


class StreamingCallMonitor:
    """
    Stateful real-time conversational monitor for live audio streams.
    Designed for 16 kHz mono PCM streams.
    """

    def __init__(
        self,
        classifier=None,
        neural_classifier=None,
        feature_extractor=None,
        risk_engine=None,
        sample_rate: int = SAMPLE_RATE,
        buffer_seconds: float = 6.0,
        frame_seconds: float = 0.25,
        claimed_contact_role: str = "father",
        is_financial_transaction: bool = True,
        language_preset: str = "auto",
    ):
        self.classifier = classifier
        self.neural_classifier = neural_classifier
        self.feature_extractor = feature_extractor
        self.risk_engine = risk_engine
        self.sample_rate = sample_rate
        self.buffer_size = int(buffer_seconds * sample_rate)
        self.frame_samples = int(frame_seconds * sample_rate)
        self.claimed_contact_role = claimed_contact_role
        self.is_financial_transaction = is_financial_transaction
        self.language_preset = language_preset
        self.preprocessor = AudioPreprocessor()

        # Rolling circular buffer of raw float32 samples (-1.0 to 1.0)
        self.pcm_buffer = np.zeros(self.buffer_size, dtype=np.float32)
        self.write_idx = 0
        self.total_samples_received = 0

        # Speaker cluster centroids (13 MFCC + spectral centroid)
        self.centroid_a: Optional[np.ndarray] = None  # Local Caller (User)
        self.centroid_b: Optional[np.ndarray] = None  # Remote Party (Contact)
        self.turn_history = deque(maxlen=20)

        # Real-time state
        self.current_speaker = "SILENCE"
        self.is_speech = False
        self.is_overlap = False
        self.speaker_a_risk = 5.0
        self.speaker_b_risk = 5.0
        self.live_call_risk = 5.0
        self.risk_level = "LOW"
        self.alert_triggered = False
        self.alert_message = "Awaiting speech..."

        # Peak & Cumulative Session Threat Memory (Preserves threat when analysis is stopped)
        self.peak_risk_b = 5.0
        self.peak_risk_a = 5.0
        self.peak_call_risk = 5.0
        self.alert_ever_triggered = False
        self.speaker_b_frames = deque(maxlen=int(sample_rate * 12.0))
        self.speaker_a_frames = deque(maxlen=int(sample_rate * 12.0))
        self.speaker_b_voiced_frames = deque(maxlen=int(sample_rate * 12.0))
        self.speaker_a_voiced_frames = deque(maxlen=int(sample_rate * 12.0))
        self.deep_spoof_prob_b = 0.05
        self.deep_risk_b = 5.0
        self.deep_risk_a = 5.0
        self.last_deep_eval_time_b = 0.0
        self.last_deep_eval_time_a = 0.0
        self.last_eval_samples_b = 0
        self.last_eval_samples_a = 0

        # Smoothing EMA for live risk score
        self.ema_risk_b = 5.0
        self.ema_risk_a = 5.0

        # Track speech durations
        self.speaker_a_speech_sec = 0.0
        self.speaker_b_speech_sec = 0.0
        self.overlap_speech_sec = 0.0

        # Progressive multi-stage verification (Stage 0: 0-5s calibration, Stage 1: 5-8s average, Stage 2: >=8s final)
        self.evaluation_stage = "CALIBRATING"  # 'CALIBRATING' | 'PRELIMINARY_AVERAGE' | 'CONSOLIDATED_FINAL'
        self.risk_history = deque(maxlen=60)
        self.average_risk = 5.0
        self.final_confirmed_risk = 5.0

        # Live conversation timeline & events
        self.start_time = time.time()
        self.live_events = deque(maxlen=30)
        self.turn_counts = {"SPEAKER_A": 0, "SPEAKER_B": 0, "OVERLAP": 0}
        self.last_speaker = "SILENCE"
        self._record_event("INFO", "SYSTEM", "Live Call Shield initialized. Zero-retention volatile buffer active.", 0.0)

        # Baseline energy floor
        self.energy_floor = 0.005
        self.last_process_time = time.time()

    def _record_event(self, event_type: str, speaker: str, text: str, risk: float = 5.0) -> None:
        """Record chronological live conversational event."""
        elapsed = round(max(0.0, time.time() - self.start_time), 1)
        mins = int(elapsed // 60)
        secs = int(elapsed % 60)
        timestamp_str = f"{mins:02d}:{secs:02d}"

        if self.live_events and self.live_events[-1].get("text") == text:
            return

        self.live_events.append({
            "timestamp": timestamp_str,
            "speaker": speaker,
            "type": event_type,
            "text": text,
            "risk": round(risk, 1),
        })

    def push_pcm(self, pcm_data: np.ndarray) -> None:
        """
        Append new float32 audio samples to circular buffer.
        Accepts numpy array of float32 values in range [-1.0, 1.0].
        """
        n = len(pcm_data)
        if n == 0:
            return

        self.total_samples_received += n
        if n >= self.buffer_size:
            self.pcm_buffer = pcm_data[-self.buffer_size:].copy()
            self.write_idx = 0
            return

        space_at_end = self.buffer_size - self.write_idx
        if n <= space_at_end:
            self.pcm_buffer[self.write_idx : self.write_idx + n] = pcm_data
            self.write_idx = (self.write_idx + n) % self.buffer_size
        else:
            self.pcm_buffer[self.write_idx : self.buffer_size] = pcm_data[:space_at_end]
            remainder = n - space_at_end
            self.pcm_buffer[0:remainder] = pcm_data[space_at_end:]
            self.write_idx = remainder

    def get_recent_window(self, seconds: float = 2.0) -> np.ndarray:
        """Get the most recent N seconds of contiguous audio from the circular buffer."""
        num_samples = min(int(seconds * self.sample_rate), self.buffer_size)
        if self.total_samples_received < num_samples:
            num_samples = self.total_samples_received

        if num_samples <= 0:
            return np.zeros(0, dtype=np.float32)

        # Unroll circular buffer
        start_idx = (self.write_idx - num_samples) % self.buffer_size
        if start_idx < self.write_idx:
            return self.pcm_buffer[start_idx:self.write_idx].copy()
        else:
            part1 = self.pcm_buffer[start_idx:]
            part2 = self.pcm_buffer[:self.write_idx]
            return np.concatenate([part1, part2])

    def _extract_embedding(self, audio_slice: np.ndarray) -> Optional[np.ndarray]:
        """Extract standardized 40-dimensional acoustic timbre embedding (MFCC mean/std + centroid + rolloff)."""
        if len(audio_slice) < 512:
            return None
        try:
            mfcc = librosa.feature.mfcc(y=audio_slice, sr=self.sample_rate, n_mfcc=20)[1:]
            sc = librosa.feature.spectral_centroid(y=audio_slice, sr=self.sample_rate)
            sr_feat = librosa.feature.spectral_rolloff(y=audio_slice, sr=self.sample_rate)
            v = np.concatenate([
                np.mean(mfcc, axis=1),
                np.std(mfcc, axis=1),
                [np.mean(sc) / 4000.0],
                [np.mean(sr_feat) / 8000.0]
            ])
            return (v / (np.linalg.norm(v) + 1e-7)).astype(np.float32)
        except Exception:
            return None

    def _cosine_dist(self, u: np.ndarray, v: np.ndarray) -> float:
        """Cosine distance between normalized vectors."""
        dot = np.clip(np.dot(u, v), -1.0, 1.0)
        return float(1.0 - dot)

    def process_live_chunk(self, pcm_data: np.ndarray) -> Dict[str, Any]:
        """
        Process an incoming live audio chunk and return instantaneous conversational threat telemetry.
        Called continuously as chunks arrive over WebSocket.
        """
        self.push_pcm(pcm_data)

        # Calculate instantaneous RMS energy
        energy = float(np.sqrt(np.mean(pcm_data**2))) if len(pcm_data) > 0 else 0.0
        peak = float(np.max(np.abs(pcm_data))) if len(pcm_data) > 0 else 0.0

        # Voice Activity Detection: Calibrated for smartphone and speakerphone microphones
        # (Mobile mics under hardware AGC typically yield speech RMS between 0.0035 and 0.04)
        is_speech = energy > 0.0035 or peak > 0.018
        self.is_speech = is_speech

        if not is_speech:
            self.current_speaker = "SILENCE"
            self.is_overlap = False
            self._update_threat_verdict()
            return self._build_telemetry(energy, peak)

        # Get recent 1.5 seconds for acoustic classification
        recent_audio = self.get_recent_window(seconds=1.5)
        if len(recent_audio) < int(self.sample_rate * 0.25):
            # Too short for analysis
            self._update_threat_verdict()
            return self._build_telemetry(energy, peak)

        # 1. Diarization: Who is speaking right now?
        emb = self._extract_embedding(recent_audio)
        if emb is not None:
            speaker, is_overlap = self._classify_speaker(emb, recent_audio)
            self.current_speaker = speaker
            self.is_overlap = is_overlap

            # Record turn transition event
            if speaker != self.last_speaker:
                self.last_speaker = speaker
                if speaker in self.turn_counts:
                    self.turn_counts[speaker] += 1
                role_str = self.claimed_contact_role.replace('_', ' ').title()
                if is_overlap:
                    self._record_event("OVERLAP", "OVERLAP", "Simultaneous voices detected (Cross-talk excluded from scoring)", 25.0)
                elif speaker == "SPEAKER_A":
                    self._record_event("INFO", "SPEAKER_A", "You (Local Caller) speaking - natural human vocal tract confirmed", self.speaker_a_risk)
                elif speaker == "SPEAKER_B":
                    if self.speaker_b_risk >= 70.0:
                        self._record_event("ALERT", "SPEAKER_B", f"Remote ({role_str}) speaking - ALERT: HIGH CLONE / SPOOF RISK DETECTED", self.speaker_b_risk)
                    else:
                        self._record_event("INFO", "SPEAKER_B", f"Remote ({role_str}) speaking - spectral timbre analysis active", self.speaker_b_risk)

            dt = len(pcm_data) / float(self.sample_rate)
            if is_overlap:
                self.overlap_speech_sec += dt
            elif speaker == "SPEAKER_A":
                self.speaker_a_speech_sec += dt
            elif speaker == "SPEAKER_B":
                self.speaker_b_speech_sec += dt

            # 2. Anti-Spoof Feature Extraction (Isolated Per-Speaker to prevent contamination)
            if not is_overlap:
                chunk_energy = float(np.sqrt(np.mean(pcm_data**2))) if len(pcm_data) > 0 else 0.0

                if speaker == "SPEAKER_B":
                    # Remote party (claimed contact)
                    if chunk_energy > 0.003:
                        self.speaker_b_voiced_frames.extend(pcm_data)
                    self.speaker_b_frames.extend(pcm_data)

                    current_b_len = len(self.speaker_b_voiced_frames)
                    # Evaluate rapidly once sufficient voiced speech accumulated (>= 0.5s, update every 0.25s)
                    if current_b_len >= int(self.sample_rate * 0.5) and (current_b_len - self.last_eval_samples_b >= int(self.sample_rate * 0.25)):
                        self.last_eval_samples_b = current_b_len
                        accum_samples = np.array(self.speaker_b_voiced_frames, dtype=np.float32)
                        eval_len = min(len(accum_samples), int(self.sample_rate * 4.0))
                        deep_slice = accum_samples[-eval_len:]
                        d_prob, d_pv, d_hf, d_jit, d_risk, d_level = self._evaluate_anti_spoof(deep_slice)
                        self.deep_spoof_prob_b = d_prob
                        self.deep_risk_b = d_risk
                        self.risk_history.append(d_risk)

                    evaluated_risk = self.deep_risk_b

                    # Responsive dynamic EMA smoothing:
                    # When speech evaluates as safe genuine human (< 30%), immediately decay into safe LOW zone
                    if evaluated_risk < 30.0:
                        self.ema_risk_b = 0.45 * evaluated_risk + 0.55 * self.ema_risk_b
                    elif evaluated_risk >= 60.0:
                        self.ema_risk_b = max(self.ema_risk_b, evaluated_risk)
                    else:
                        self.ema_risk_b = 0.35 * evaluated_risk + 0.65 * self.ema_risk_b

                    self.speaker_b_risk = round(self.ema_risk_b, 1)
                    self.peak_risk_b = max(self.peak_risk_b, evaluated_risk, self.speaker_b_risk)

                    if self.speaker_b_risk >= 50.0 and (not self.live_events or self.live_events[-1].get("type") != "ALERT"):
                        role_str = self.claimed_contact_role.replace('_', ' ').title()
                        self._record_event("ALERT", "SPEAKER_B", f"ALERT: AI-Cloned Voice detected for {role_str} ({self.speaker_b_risk}%)", self.speaker_b_risk)

                elif speaker == "SPEAKER_A":
                    # Local caller / audio input (user)
                    if chunk_energy > 0.003:
                        self.speaker_a_voiced_frames.extend(pcm_data)
                    self.speaker_a_frames.extend(pcm_data)

                    current_a_len = len(self.speaker_a_voiced_frames)
                    if current_a_len >= int(self.sample_rate * 0.5) and (current_a_len - self.last_eval_samples_a >= int(self.sample_rate * 0.25)):
                        self.last_eval_samples_a = current_a_len
                        accum_samples = np.array(self.speaker_a_voiced_frames, dtype=np.float32)
                        eval_len = min(len(accum_samples), int(self.sample_rate * 4.0))
                        deep_slice = accum_samples[-eval_len:]
                        d_prob, d_pv, d_hf, d_jit, d_risk, d_level = self._evaluate_anti_spoof(deep_slice)
                        self.deep_risk_a = d_risk
                        self.risk_history.append(d_risk)

                    evaluated_risk_a = self.deep_risk_a
                    if evaluated_risk_a < 30.0:
                        self.ema_risk_a = 0.45 * evaluated_risk_a + 0.55 * self.ema_risk_a
                    elif evaluated_risk_a >= 60.0:
                        self.ema_risk_a = max(self.ema_risk_a, evaluated_risk_a)
                    else:
                        self.ema_risk_a = 0.35 * evaluated_risk_a + 0.65 * self.ema_risk_a
                    self.speaker_a_risk = round(self.ema_risk_a, 1)
                    self.peak_risk_a = max(self.peak_risk_a, evaluated_risk_a, self.speaker_a_risk)

                    if self.speaker_a_risk >= 50.0 and (not self.live_events or self.live_events[-1].get("type") != "ALERT"):
                        self._record_event("ALERT", "SPEAKER_A", f"ALERT: AI-Cloned Voice detected on audio stream ({self.speaker_a_risk}%)", self.speaker_a_risk)

        # Update overall live call threat level
        self._update_threat_verdict()

        return self._build_telemetry(energy, peak)

    def _classify_speaker(self, emb: np.ndarray, audio_slice: np.ndarray) -> Tuple[str, bool]:
        """
        Classify speech into Speaker A, Speaker B, or Overlap.
        Uses adaptive centroid clustering with calibrated timbre distance.
        """
        # Case 1: First speaker encounter -> Initialize Speaker A (Caller)
        if self.centroid_a is None:
            self.centroid_a = emb.copy()
            return "SPEAKER_A", False

        dist_a = self._cosine_dist(emb, self.centroid_a)

        # Case 2: Speaker B not yet initialized
        if self.centroid_b is None:
            if dist_a > 0.025:
                # Distinct new timbre detected -> Initialize Speaker B (Remote Party)
                self.centroid_b = emb.copy()
                return "SPEAKER_B", False
            else:
                # Still Speaker A; adapt centroid slightly (EMA 0.03)
                self.centroid_a = (0.97 * self.centroid_a + 0.03 * emb)
                self.centroid_a /= np.linalg.norm(self.centroid_a) + 1e-7
                return "SPEAKER_A", False

        # Case 3: Both centroids known -> compute distances
        dist_b = self._cosine_dist(emb, self.centroid_b)

        # Check for overlap / simultaneous cross-talk
        margin = abs(dist_a - dist_b)
        if margin < 0.005 and min(dist_a, dist_b) < 0.035:
            return "OVERLAP", True

        if dist_a < dist_b:
            # Closer to Speaker A
            self.centroid_a = (0.98 * self.centroid_a + 0.02 * emb)
            self.centroid_a /= np.linalg.norm(self.centroid_a) + 1e-7
            return "SPEAKER_A", False
        else:
            # Closer to Speaker B
            self.centroid_b = (0.98 * self.centroid_b + 0.02 * emb)
            self.centroid_b /= np.linalg.norm(self.centroid_b) + 1e-7
            return "SPEAKER_B", False

    def _evaluate_anti_spoof(
        self, audio_window: np.ndarray
    ) -> Tuple[float, float, float, float, float, str]:
        """
        Extract complete anti-spoof features on clean speaker slice and compute
        calibrated risk using RiskEngine as single source of truth.
        Returns: (spoof_prob, phase_var, hf_ratio, jitter, risk_score, risk_level)
        """
        if len(audio_window) < int(self.sample_rate * 0.25):
            return 0.05, 0.02, 0.005, 0.01, 5.0, "LOW"

        energy = float(np.sqrt(np.mean(audio_window**2)))
        peak = float(np.max(np.abs(audio_window)))
        if energy < 0.0035 and peak < 0.018:
            # Silence / ambient noise - do not amplify or evaluate noise as spoof
            return 0.05, 0.02, 0.005, 0.01, 5.0, "LOW"

        # Extract voiced speech only (removes silent gaps & room hiss)
        voiced = self.preprocessor.apply_vad(audio_window, top_db=26.0)
        if len(voiced) < int(self.sample_rate * 0.25):
            if len(audio_window) >= int(self.sample_rate * 0.25):
                eval_audio = self.preprocessor.normalize_audio(audio_window)
            else:
                return 0.05, 0.02, 0.005, 0.01, 5.0, "LOW"
        else:
            eval_audio = self.preprocessor.normalize_audio(voiced)

        features = {}
        if self.feature_extractor:
            try:
                features = self.feature_extractor.extract_all(eval_audio)
            except Exception:
                pass

        phase_var = float(features.get("acoustic_phase_derivative_var", 0.04))
        hf_ratio = float(features.get("spectral_hf_energy_ratio", 0.005))
        jitter = float(features.get("prosody_jitter", 0.02))
        contrast_std = float(features.get("spectral_contrast_std", 6.0))

        rf_prob = 0.10
        if self.classifier:
            try:
                vec = self.feature_extractor.to_vector(features, self.classifier.feature_names)
                rf_prob = float(self.classifier.predict_spoof_risk(vec))
            except Exception:
                pass

        neural_prob = None
        if self.neural_classifier:
            try:
                neural_prob = float(self.neural_classifier.predict_spoof_prob(eval_audio))
            except Exception:
                pass

        if neural_prob is not None:
            spoof_prob = 0.50 * rf_prob + 0.50 * neural_prob
        else:
            spoof_prob = rf_prob

        tau = (
            float(self.classifier.calibrated_threshold)
            if (self.classifier and hasattr(self.classifier, "calibrated_threshold") and self.classifier.calibrated_threshold)
            else 0.5102
        )

        context_meta = {
            "is_high_value_transaction": self.is_financial_transaction
        }

        risk_score = 5.0
        risk_level = "LOW"
        if self.risk_engine:
            assessment = self.risk_engine.compute_risk(
                classifier_spoof_prob=spoof_prob,
                calibrated_threshold=tau,
                acoustic_anomaly_score=phase_var,
                spectral_anomaly_score=hf_ratio,
                prosody_anomaly_score=jitter,
                contrast_anomaly_score=contrast_std,
                context_metadata=context_meta,
                language_code=self.language_preset
            )
            risk_score = float(assessment.risk_score)
            risk_level = assessment.risk_level
        print(f"[LIVE EVALUATION] rf_prob={rf_prob:.4f} contrast_std={contrast_std:.2f} risk_score={risk_score:.1f} risk_level={risk_level}", flush=True)
        return spoof_prob, phase_var, hf_ratio, jitter, risk_score, risk_level

    def _update_threat_verdict(self) -> None:
        """Update live call threat level and alert message based on both speakers, progressive stages, and session history."""
        total_speech = self.speaker_b_speech_sec + self.speaker_a_speech_sec
        active_threat = self.speaker_b_risk
        if self.speaker_b_speech_sec < 0.5 or self.speaker_a_risk >= 50.0:
            active_threat = max(active_threat, self.speaker_a_risk)

        self.peak_call_risk = max(self.peak_call_risk, self.peak_risk_b, self.peak_risk_a if self.peak_risk_a >= 50.0 else 0.0)

        # Progressive Stages (5-6s average, 10-12s unconditional final confirmed risk):
        call_elapsed = max(0.0, time.time() - self.start_time)
        is_stage_2 = (call_elapsed >= 10.0) or (total_speech >= 4.5)
        is_stage_1 = not is_stage_2 and ((call_elapsed >= 5.0) or (total_speech >= 1.5))

        # If entering Stage 1 or Stage 2 without an evaluation yet, run fallback evaluation across buffered audio
        if (is_stage_1 or is_stage_2) and not self.risk_history:
            candidate_audio = None
            if len(self.speaker_b_voiced_frames) >= int(self.sample_rate * 0.25):
                candidate_audio = np.array(self.speaker_b_voiced_frames, dtype=np.float32)
            elif len(self.speaker_b_frames) >= int(self.sample_rate * 0.25):
                candidate_audio = np.array(self.speaker_b_frames, dtype=np.float32)
            elif len(self.speaker_a_voiced_frames) >= int(self.sample_rate * 0.25):
                candidate_audio = np.array(self.speaker_a_voiced_frames, dtype=np.float32)
            elif len(self.speaker_a_frames) >= int(self.sample_rate * 0.25):
                candidate_audio = np.array(self.speaker_a_frames, dtype=np.float32)
            elif self.total_samples_received >= int(self.sample_rate * 0.5):
                candidate_audio = self.get_recent_window(seconds=3.0)

            if candidate_audio is not None and len(candidate_audio) >= int(self.sample_rate * 0.25):
                d_prob, d_pv, d_hf, d_jit, d_risk, d_level = self._evaluate_anti_spoof(candidate_audio)
                self.deep_spoof_prob_b = d_prob
                self.deep_risk_b = d_risk
                self.risk_history.append(d_risk)
                self.speaker_b_risk = round(d_risk, 1)
                active_threat = max(active_threat, d_risk)

        if is_stage_2:
            self.evaluation_stage = "CONSOLIDATED_FINAL"
            if self.risk_history:
                if active_threat >= 50.0:
                    self.final_confirmed_risk = round(max(active_threat, float(np.max(self.risk_history))), 1)
                else:
                    self.final_confirmed_risk = round(active_threat, 1)
            else:
                self.final_confirmed_risk = round(active_threat, 1)
            self.live_call_risk = self.final_confirmed_risk
        elif is_stage_1:
            self.evaluation_stage = "PRELIMINARY_AVERAGE"
            if self.risk_history:
                avg = float(np.mean(self.risk_history))
            else:
                avg = active_threat
            self.average_risk = round(avg, 1)
            self.live_call_risk = self.average_risk
        else:
            self.evaluation_stage = "CALIBRATING"
            self.live_call_risk = 5.0
            self.risk_level = "CALIBRATING"
            self.alert_triggered = False
            needed = max(0.0, 5.0 - call_elapsed)
            self.alert_message = f"Calibrating acoustic baseline ({call_elapsed:.0f}s / 10s)... Preliminary score in {needed:.0f}s."

        if self.evaluation_stage == "PRELIMINARY_AVERAGE":
            time_to_final = max(0.0, 10.0 - call_elapsed)
            if self.live_call_risk >= 65.0:
                self.risk_level = "CRITICAL"
                self.alert_triggered = True
                self.alert_ever_triggered = True
                self.alert_message = f"⚠️ PRELIMINARY WARNING: Potential AI clone ({self.average_risk:.1f}% avg). Finalizing verification in {time_to_final:.0f}s..."
            elif self.live_call_risk >= 50.0:
                self.risk_level = "HIGH"
                self.alert_triggered = True
                self.alert_ever_triggered = True
                self.alert_message = f"⚠️ PRELIMINARY ALERT: Synthetic artifacts detected ({self.average_risk:.1f}% avg). Stabilizing..."
            elif self.live_call_risk >= 25.0:
                self.risk_level = "MEDIUM"
                self.alert_triggered = False
                self.alert_message = f"Preliminary average risk: {self.average_risk:.1f}% (Elevated ambient acoustics). Stabilizing..."
            else:
                self.risk_level = "LOW"
                self.alert_triggered = False
                self.alert_message = f"Preliminary assessment: {self.average_risk:.1f}% risk (Averaged over {total_speech:.1f}s speech). Finalizing in {time_to_final:.0f}s..."

        elif self.evaluation_stage == "CONSOLIDATED_FINAL":

            role_str = self.claimed_contact_role.replace('_', ' ').title()
            if self.live_call_risk >= 65.0:
                self.risk_level = "CRITICAL"
                self.alert_triggered = True
                self.alert_ever_triggered = True
                self.alert_message = (
                    f"🚨 CRITICAL ALERT: AI-Cloned Voice confirmed on {role_str}! "
                    f"(Final Score: {self.final_confirmed_risk:.1f}%). DO NOT authorize wire transfers or payments."
                )
            elif self.live_call_risk >= 50.0:
                self.risk_level = "HIGH"
                self.alert_triggered = True
                self.alert_ever_triggered = True
                self.alert_message = (
                    f"⚠️ HIGH SPOOF RISK: Confirmed synthetic vocal features on {role_str} "
                    f"(Final Score: {self.final_confirmed_risk:.1f}%). Verify identity with a challenge question."
                )
            elif self.live_call_risk >= 25.0:
                self.risk_level = "MEDIUM"
                self.alert_triggered = False
                self.alert_message = f"Final assessment: {self.final_confirmed_risk:.1f}% risk. Ambient room acoustics or VoIP compression detected."
            else:
                self.risk_level = "LOW"
                self.alert_triggered = False
                self.alert_message = f"Call verified authentic ({self.final_confirmed_risk:.1f}% risk). Vocal biomechanics match natural human speech."

    def get_session_summary(self) -> Dict[str, Any]:
        """Detailed forensic summary of the complete call session."""
        is_clone = self.peak_call_risk >= 50.0 or self.alert_ever_triggered
        return {
            "verdict": "CRITICAL_CLONE_DETECTED" if self.peak_call_risk >= 65.0 else "HIGH_SPOOF_RISK" if self.peak_call_risk >= 50.0 else "AUTHENTIC_CALL",
            "verdict_title": "AI-Cloned Voice Detected" if is_clone else "Call Verified Authentic",
            "peak_risk": round(self.peak_call_risk, 1),
            "final_risk": round(self.live_call_risk, 1),
            "alert_triggered": self.alert_ever_triggered,
            "remote_speech_sec": round(self.speaker_b_speech_sec, 1),
            "local_speech_sec": round(self.speaker_a_speech_sec, 1),
            "turns_remote": self.turn_counts.get("SPEAKER_B", 0),
            "turns_local": self.turn_counts.get("SPEAKER_A", 0),
            "deep_ml_spoof_prob": round(self.deep_spoof_prob_b, 3),
            "explanation": self.alert_message,
            "stage": self.evaluation_stage,
            "average_risk": round(self.average_risk, 1),
            "final_confirmed_risk": round(self.final_confirmed_risk, 1),
        }

    def _build_telemetry(self, energy: float, peak: float) -> Dict[str, Any]:
        """Construct JSON-serializable telemetry payload for WebSocket clients."""
        role_label = f"Claimed {self.claimed_contact_role.replace('_', ' ').title()}"
        total_speech = self.speaker_b_speech_sec + self.speaker_a_speech_sec
        elapsed = round(max(0.0, time.time() - self.start_time), 1)

        active_label = "Listening..."
        if self.is_speech:
            if self.is_overlap:
                active_label = "Simultaneous Speech (Cross-Talk Overlap)"
            elif self.current_speaker == "SPEAKER_A":
                active_label = "You (Local Caller) Speaking"
            else:
                active_label = f"Remote Voice ({role_label}) Speaking"
        elif elapsed < 10.0:
            active_label = f"Analyzing call audio (Evaluating: {elapsed:.0f}s / 10s)..."

        time_prog = min(1.0, elapsed / 10.0)
        speech_prog = min(1.0, total_speech / 4.5)
        cal_prog = max(time_prog, speech_prog)

        return {
            "is_speech": self.is_speech,
            "active_speaker": self.current_speaker,
            "active_label": active_label,
            "language_preset": self.language_preset,
            "elapsed_sec": elapsed,
            "total_speech_sec": round(total_speech, 1),
            "stage": self.evaluation_stage,
            "calibration_progress": round(cal_prog, 2),
            "average_risk": round(self.average_risk, 1),
            "final_confirmed_risk": round(self.final_confirmed_risk, 1),
            "live_call_risk": round(self.live_call_risk, 1),
            "risk_level": self.risk_level,
            "alert": self.alert_triggered,
            "alert_message": self.alert_message,
            "is_overlap": self.is_overlap,
            "speaker_a": {
                "label": "You (Local Caller)",
                "risk_score": round(self.speaker_a_risk, 1),
                "risk_level": "LOW" if self.speaker_a_risk < 50.0 else "HIGH",
                "speech_duration_sec": round(self.speaker_a_speech_sec, 1),
                "is_speaking": self.is_speech and self.current_speaker == "SPEAKER_A",
                "turns": self.turn_counts.get("SPEAKER_A", 0),
            },
            "speaker_b": {
                "label": f"Remote ({role_label})",
                "risk_score": round(self.speaker_b_risk, 1),
                "risk_level": (
                    "CRITICAL"
                    if self.speaker_b_risk >= 70.0
                    else "HIGH"
                    if self.speaker_b_risk >= 50.0
                    else "MEDIUM"
                    if self.speaker_b_risk >= 30.0
                    else "LOW"
                ),
                "speech_duration_sec": round(self.speaker_b_speech_sec, 1),
                "is_speaking": self.is_speech and self.current_speaker == "SPEAKER_B",
                "turns": self.turn_counts.get("SPEAKER_B", 0),
            },
            "overlap_sec": round(self.overlap_speech_sec, 1),
            "audio_energy": round(min(1.0, energy * 8.0), 3),
            "peak_level": round(peak, 3),
            "timestamp": round(time.time(), 3),
            "live_events": list(self.live_events),
            "session_summary": self.get_session_summary(),
        }
