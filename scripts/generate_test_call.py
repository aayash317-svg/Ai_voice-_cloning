"""
Generate a realistic synthetic two-speaker conversation test call.
Simulates:
- Turn 1: Speaker A (Genuine Caller / You asking a question)
- Turn 2: Speaker B (Synthetic AI-cloned voice / Claimed father responding)
- Turn 3: Speaker A (Genuine Caller follow-up)
- Turn 4: Overlapping speech (Both speaking simultaneously - cross-talk collision)
- Turn 5: Speaker B (Synthetic AI-cloned closing)
"""

import os
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa

TARGET_SR = 16000

def create_two_speaker_call(output_path: str = "test_samples/simulated_call_father_clone.wav"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 1. Load samples and resample to 16kHz mono
    g1, sr1 = librosa.load("test_samples/test_genuine.wav", sr=TARGET_SR, mono=True)
    s1, sr2 = librosa.load("test_samples/chatterbox_tts_sample.wav", sr=TARGET_SR, mono=True)
    g2, sr3 = librosa.load("test_samples/genuine_sample.wav", sr=TARGET_SR, mono=True)

    # Normalize amplitudes
    g1 = (g1 / (np.max(np.abs(g1)) + 1e-6)) * 0.85
    s1 = (s1 / (np.max(np.abs(s1)) + 1e-6)) * 0.85
    g2 = (g2 / (np.max(np.abs(g2)) + 1e-6)) * 0.85

    # 2. Construct segments
    # Turn 1: Speaker A (Caller) - ~3.8s
    turn1 = g1[: int(TARGET_SR * 3.8)]

    # Pause between turns (0.6s natural pause)
    pause1 = np.zeros(int(TARGET_SR * 0.6), dtype=np.float32)

    # Turn 2: Speaker B (Remote cloned father) - ~4.5s
    turn2 = s1[: int(TARGET_SR * 4.5)]

    # Pause between turns (0.6s)
    pause2 = np.zeros(int(TARGET_SR * 0.6), dtype=np.float32)

    # Turn 3: Speaker A (Caller) - ~3.5s
    turn3 = g2[: int(TARGET_SR * 3.5)]

    # Pause between turn 3 and overlap cross-talk (0.6s)
    pause_before_ov = np.zeros(int(TARGET_SR * 0.6), dtype=np.float32)

    # Overlap cross-talk segment (1.8s) - Speaker A and Speaker B speaking simultaneously
    ov_len = int(TARGET_SR * 1.8)
    spk_a_ov = g1[int(TARGET_SR * 1.5) : int(TARGET_SR * 1.5) + ov_len] * 0.65
    spk_b_ov = s1[int(TARGET_SR * 1.0) : int(TARGET_SR * 1.0) + ov_len] * 0.65
    overlap_segment = spk_a_ov + spk_b_ov

    # Pause between overlap and next turn (0.6s)
    pause3 = np.zeros(int(TARGET_SR * 0.6), dtype=np.float32)

    # Turn 4: Speaker B (Remote cloned father closing) - ~3.5s
    turn4 = s1[int(TARGET_SR * 1.5) : int(TARGET_SR * 5.0)]

    # 3. Concatenate call waveform
    full_call = np.concatenate([
        turn1,            # 0.0s - 3.8s (Speaker A)
        pause1,           # 3.8s - 4.4s (Pause)
        turn2,            # 4.4s - 8.9s (Speaker B - Clone)
        pause2,           # 8.9s - 9.5s (Pause)
        turn3,            # 9.5s - 13.0s (Speaker A)
        pause_before_ov,  # 13.0s - 13.6s (Pause)
        overlap_segment,  # 13.6s - 15.4s (OVERLAP CROSS-TALK)
        pause3,           # 15.4s - 16.0s (Pause)
        turn4             # 16.0s - 19.5s (Speaker B - Clone)
    ]).astype(np.float32)

    # Retain natural acoustic properties without artificial noise masking
    full_call = np.clip(full_call, -0.98, 0.98)

    sf.write(output_path, full_call, TARGET_SR)
    print(f"Generated realistic test call: {output_path}")
    print(f"Total Duration: {len(full_call) / TARGET_SR:.2f} seconds")
    print(f"Expected Structure:")
    print(f"  • 0.0s - 3.5s: Speaker A (Caller / Genuine)")
    print(f"  • 4.1s - 8.6s: Speaker B (Remote Father / AI Clone)")
    print(f"  • 9.1s - 12.1s: Speaker A (Caller / Genuine)")
    print(f"  • 12.1s - 13.9s: Overlapping Cross-talk (Both)")
    print(f"  • 14.3s - 17.3s: Speaker B (Remote Father / AI Clone)")
    return output_path

if __name__ == "__main__":
    create_two_speaker_call()
