"""
Speaker Diarization & Overlap Detection Subsystem.
Identifies active speaker turns (who spoke when), detects overlapping cross-talk,
and segments audio in volatile RAM without saving unconsented biometric recordings to disk.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import librosa
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class SpeechSegment:
    """Individual conversational speech turn segment."""
    start_sec: float
    end_sec: float
    duration_sec: float
    speaker: str                   # 'Speaker A', 'Speaker B', or 'Overlap'
    is_overlap: bool
    confidence: float
    audio: np.ndarray = field(repr=False)


@dataclass
class DiarizationResult:
    """Aggregated diarization outcome for conversational audio."""
    segments: List[SpeechSegment]
    total_speech_seconds: float
    overlap_seconds: float
    overlap_percent: float
    speaker_durations: Dict[str, float]
    speaker_clean_audio: Dict[str, np.ndarray]
    diarization_confidence: float
    num_speakers_detected: int


class SpeakerDiarizer:
    """
    In-memory speaker diarization and overlap detection engine.
    Uses energy/flux VAD, multi-pitch harmonic entropy for overlap detection,
    and cosine-distance acoustic clustering for turn attribution.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        min_segment_len_sec: float = 0.5,
        max_segment_len_sec: float = 2.5,
        min_silence_len_sec: float = 0.25,
    ):
        self.sr = sample_rate
        self.min_seg_len = min_segment_len_sec
        self.max_seg_len = max_segment_len_sec
        self.min_silence_len = min_silence_len_sec

    def detect_overlap(self, audio_chunk: np.ndarray) -> Tuple[bool, float]:
        """
        Detect simultaneous overlapping speech via harmonic collision and spectral entropy.
        When two distinct speakers talk simultaneously:
          1. Multiple competing pitch tracks exist simultaneously (harmonic collision).
          2. Spectral flatness and entropy rise compared to harmonic single-speaker phonation.
        """
        if len(audio_chunk) < int(self.sr * 0.4):
            return False, 0.0

        try:
            # 1. Pitch tracking using YIN
            f0 = librosa.yin(
                audio_chunk,
                fmin=librosa.note_to_hz('C2'),  # ~65 Hz
                fmax=librosa.note_to_hz('C7'),  # ~2093 Hz
                sr=self.sr,
                frame_length=1024,
                hop_length=256
            )
            valid_f0 = f0[(f0 > 65.0) & (f0 < 1500.0)]

            # 2. Spectral Flatness & Sub-band energy
            flatness = librosa.feature.spectral_flatness(y=audio_chunk, n_fft=1024, hop_length=256)[0]
            flatness_std = float(np.std(flatness)) if len(flatness) > 0 else 0.0

            # 3. High pitch variance + high spectral fluctuation indicates overlapping voices
            if len(valid_f0) > 8:
                f0_diff = np.abs(np.diff(valid_f0))
                pitch_jump_ratio = float(np.mean(f0_diff > 75.0))
                if (pitch_jump_ratio >= 0.065 and flatness_std >= 0.040) or (pitch_jump_ratio >= 0.22):
                    overlap_conf = min(1.0, max(0.65, pitch_jump_ratio * 3.0 + flatness_std * 8.0))
                    return True, round(float(overlap_conf), 3)

            return False, 0.0
        except Exception:
            return False, 0.0

    def extract_speaker_embedding(self, audio_chunk: np.ndarray) -> np.ndarray:
        """
        Extract compact 28-D acoustic representation for speaker clustering:
        13 MFCC means + 13 MFCC stds + pitch mean & std.
        """
        if len(audio_chunk) < int(self.sr * 0.2):
            audio_chunk = np.pad(audio_chunk, (0, int(self.sr * 0.2) - len(audio_chunk)))

        # MFCCs
        mfccs = librosa.feature.mfcc(y=audio_chunk, sr=self.sr, n_mfcc=13, n_fft=1024, hop_length=256)
        mfcc_mean = np.mean(mfccs, axis=1)
        mfcc_std = np.std(mfccs, axis=1)

        # Pitch proxy
        try:
            f0 = librosa.yin(audio_chunk, fmin=70, fmax=500, sr=self.sr, frame_length=1024, hop_length=256)
            valid_f0 = f0[(f0 > 70) & (f0 < 500)]
            if len(valid_f0) > 2:
                f0_mean = float(np.mean(valid_f0)) / 500.0
                f0_std = float(np.std(valid_f0)) / 250.0
            else:
                f0_mean, f0_std = 0.0, 0.0
        except Exception:
            f0_mean, f0_std = 0.0, 0.0

        feat = np.concatenate([mfcc_mean, mfcc_std, [f0_mean, f0_std]])
        norm = np.linalg.norm(feat)
        if norm > 1e-6:
            feat = feat / norm
        return feat.astype(np.float32)

    def diarize(self, audio: np.ndarray, max_speakers: int = 2) -> DiarizationResult:
        """
        Full in-memory conversational diarization pipeline:
        1. Energy-based turn boundary segmentation.
        2. Overlap cross-talk detection.
        3. Acoustic embedding extraction.
        4. Cosine Agglomerative Clustering into Speaker A & Speaker B.
        """
        total_len = len(audio)
        min_samples = int(self.min_seg_len * self.sr)
        max_samples = int(self.max_seg_len * self.sr)
        silence_samples = int(self.min_silence_len * self.sr)

        if total_len < min_samples:
            # Single brief segment fallback
            return DiarizationResult(
                segments=[SpeechSegment(0.0, total_len / self.sr, total_len / self.sr, "Speaker A", False, 0.9, audio)],
                total_speech_seconds=round(total_len / self.sr, 2),
                overlap_seconds=0.0,
                overlap_percent=0.0,
                speaker_durations={"Speaker A": round(total_len / self.sr, 2), "Speaker B": 0.0},
                speaker_clean_audio={"Speaker A": audio, "Speaker B": np.array([], dtype=np.float32)},
                diarization_confidence=0.85,
                num_speakers_detected=1
            )

        # 1. Voice Activity Detection (VAD) Intervals with natural phonetic margin
        intervals = librosa.effects.split(audio, top_db=36.0, frame_length=1024, hop_length=256)
        if len(intervals) == 0:
            intervals = np.array([[0, total_len]])

        # 2. Add natural speech margin (120ms) and merge closely spaced intervals
        pad_samples = int(self.sr * 0.12)
        padded_intervals = []
        for s, e in intervals:
            padded_intervals.append((max(0, s - pad_samples), min(total_len, e + pad_samples)))

        merged = []
        cur_start, cur_end = padded_intervals[0]
        for s, e in padded_intervals[1:]:
            if s <= cur_end + silence_samples:
                cur_end = max(cur_end, e)
            else:
                merged.append((cur_start, cur_end))
                cur_start, cur_end = s, e
        merged.append((cur_start, cur_end))

        # Chunk large intervals into digestible turn segments
        raw_chunks = []
        for s, e in merged:
            dur = e - s
            if dur <= max_samples:
                if dur >= min_samples:
                    raw_chunks.append((s, e))
            else:
                for sub_s in range(s, e, max_samples):
                    sub_e = min(e, sub_s + max_samples)
                    if sub_e - sub_s >= min_samples:
                        raw_chunks.append((sub_s, sub_e))

        if len(raw_chunks) == 0:
            raw_chunks = [(0, total_len)]

        # 3. Process each segment: overlap detection + embedding extraction
        embeddings = []
        chunk_metadata = []
        overlap_dur = 0.0

        for s_idx, e_idx in raw_chunks:
            chunk_audio = audio[s_idx:e_idx]
            start_s = round(s_idx / float(self.sr), 2)
            end_s = round(e_idx / float(self.sr), 2)
            dur_s = round(end_s - start_s, 2)

            is_overlap, overlap_conf = self.detect_overlap(chunk_audio)
            if is_overlap:
                overlap_dur += dur_s

            emb = self.extract_speaker_embedding(chunk_audio)
            embeddings.append(emb)
            chunk_metadata.append({
                "start": start_s,
                "end": end_s,
                "dur": dur_s,
                "audio": chunk_audio,
                "is_overlap": is_overlap,
                "overlap_conf": overlap_conf
            })

        # 4. Cluster embeddings into Speaker A vs Speaker B
        embeddings = np.array(embeddings)
        n_chunks = len(embeddings)

        if n_chunks >= 2 and max_speakers >= 2:
            try:
                # Cosine distance agglomerative clustering
                clustering = AgglomerativeClustering(
                    n_clusters=2,
                    metric='cosine',
                    linkage='average'
                )
                labels = clustering.fit_predict(embeddings)

                # Compute cluster separation confidence
                c0 = embeddings[labels == 0]
                c1 = embeddings[labels == 1]
                if len(c0) > 0 and len(c1) > 0:
                    mean0 = np.mean(c0, axis=0, keepdims=True)
                    mean1 = np.mean(c1, axis=0, keepdims=True)
                    inter_sim = float(cosine_similarity(mean0, mean1)[0, 0])
                    # Higher divergence -> higher confidence
                    cluster_conf = min(0.98, max(0.55, 1.0 - (inter_sim + 1.0) / 2.0 + 0.35))
                else:
                    cluster_conf = 0.75

                # Enforce Speaker A as the first active speaker
                first_label = labels[0]
                speaker_map = {first_label: "Speaker A", 1 - first_label: "Speaker B"}
                mean_spk_a = mean0 if first_label == 0 else mean1
                mean_spk_b = mean1 if first_label == 0 else mean0

            except Exception:
                labels = np.zeros(n_chunks, dtype=int)
                speaker_map = {0: "Speaker A"}
                mean_spk_a = np.mean(embeddings, axis=0, keepdims=True)
                mean_spk_b = mean_spk_a
                cluster_conf = 0.70
        else:
            labels = np.zeros(n_chunks, dtype=int)
            speaker_map = {0: "Speaker A"}
            mean_spk_a = np.mean(embeddings, axis=0, keepdims=True)
            mean_spk_b = mean_spk_a
            cluster_conf = 0.90

        # 5. Build speech segments and collect clean audio buffers
        segments: List[SpeechSegment] = []
        speaker_audio_lists: Dict[str, List[np.ndarray]] = {"Speaker A": [], "Speaker B": []}
        speaker_durations: Dict[str, float] = {"Speaker A": 0.0, "Speaker B": 0.0}
        total_speech_sec = 0.0
        overlap_dur = 0.0

        for i, meta in enumerate(chunk_metadata):
            lbl = labels[i]
            spk = speaker_map.get(lbl, "Speaker A")
            total_speech_sec += meta["dur"]

            # Compute margin between Speaker A and Speaker B
            sim_a = float(cosine_similarity([embeddings[i]], mean_spk_a)[0, 0])
            sim_b = float(cosine_similarity([embeddings[i]], mean_spk_b)[0, 0])
            margin = abs(sim_a - sim_b)

            # Segment is overlap if acoustic margin between Speaker A and B is near zero (cross-talk collision)
            is_overlap = bool(margin < 0.007 and n_chunks >= 3)
            overlap_conf = round(float(1.0 - margin / 0.010), 3) if is_overlap else 0.0

            if is_overlap:
                overlap_dur += meta["dur"]
                seg = SpeechSegment(
                    start_sec=meta["start"],
                    end_sec=meta["end"],
                    duration_sec=meta["dur"],
                    speaker="Overlap (A+B)",
                    is_overlap=True,
                    confidence=max(0.70, overlap_conf),
                    audio=meta["audio"]
                )
                # Overlap is strictly EXCLUDED from clean audio buffers
            else:
                # Assign to dominant speaker by similarity
                assigned_spk = "Speaker A" if sim_a >= sim_b else "Speaker B"
                seg = SpeechSegment(
                    start_sec=meta["start"],
                    end_sec=meta["end"],
                    duration_sec=meta["dur"],
                    speaker=assigned_spk,
                    is_overlap=False,
                    confidence=round(cluster_conf, 3),
                    audio=meta["audio"]
                )
                speaker_audio_lists[assigned_spk].append(meta["audio"])
                speaker_durations[assigned_spk] = round(speaker_durations[assigned_spk] + meta["dur"], 2)

            segments.append(seg)

        # Merge clean turns in RAM for each speaker
        speaker_clean_audio = {
            "Speaker A": np.concatenate(speaker_audio_lists["Speaker A"]) if len(speaker_audio_lists["Speaker A"]) > 0 else np.array([], dtype=np.float32),
            "Speaker B": np.concatenate(speaker_audio_lists["Speaker B"]) if len(speaker_audio_lists["Speaker B"]) > 0 else np.array([], dtype=np.float32)
        }

        overlap_pct = round((overlap_dur / max(total_speech_sec, 1e-4)) * 100.0, 1)
        num_speakers = 2 if speaker_durations.get("Speaker B", 0.0) > 0.5 else 1

        return DiarizationResult(
            segments=segments,
            total_speech_seconds=round(total_speech_sec, 2),
            overlap_seconds=round(overlap_dur, 2),
            overlap_percent=overlap_pct,
            speaker_durations=speaker_durations,
            speaker_clean_audio=speaker_clean_audio,
            diarization_confidence=round(cluster_conf, 3),
            num_speakers_detected=num_speakers
        )
