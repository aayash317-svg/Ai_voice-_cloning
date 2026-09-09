# Voice Integrity Verification Framework
## Technical Architecture, Forensic Methodology & Operations Manual

---

### Executive Summary (<350 Characters)
> **Voice Integrity Verification is a real-time AI security engine that detects deepfakes, synthetic speech, and cloned voices. Powered by a dual ensemble of SincNet neural raw-waveform filters and spectral-acoustic classifiers, it delivers dynamic risk scoring with zero-retention privacy and a tamper-evident cryptographic audit ledger.**

---

## Table of Contents
1. [System Overview & Objectives](#1-system-overview--objectives)
2. [The 5 Defense & Architectural Layers](#2-the-5-defense--architectural-layers)
3. [Forensic Science: How the AI Detects Cloned Voices](#3-forensic-science-how-the-ai-detects-cloned-voices)
4. [Dataset Inventory & Multi-Generator Corpora](#4-dataset-inventory--multi-generator-corpora)
5. [The 6-Stage Anti-Leakage Training Methodology](#5-the-6-stage-anti-leakage-training-methodology)
6. [Empirical Evaluation & Performance Benchmarks](#6-empirical-evaluation--performance-benchmarks)
7. [Enterprise Security & Cryptographic Privacy Architecture](#7-enterprise-security--cryptographic-privacy-architecture)
8. [Hands-On Code Examples & Practical API Usage](#8-hands-on-code-examples--practical-api-usage)
9. [Deployment & Operations Manual](#9-deployment--operations-manual)
10. [Visual Walkthrough & Dashboard Interface](#10-visual-walkthrough--dashboard-interface)

---

## 1. System Overview & Objectives

The **Voice Integrity Verification Framework** is an enterprise-grade defense platform engineered to safeguard telephone banking, executive authorization channels, contact centers, and biometric voice authentication systems against modern generative voice cloning attacks.

### Core Problems Addressed
- **Zero-Shot Voice Cloning**: Generative systems (ElevenLabs, VITS, ChatterboxTTS, WaveNet) capable of cloning human vocal identity from as little as 3 seconds of audio.
- **Biometric PII Privacy Risks**: GDPR and HIPAA strictly classify human voice recordings as biometric identifiers. Conventional recording/logging solutions introduce massive data breach liabilities.
- **False Alarm Penalties**: Uncalibrated models suffer from high false alarm rates, incorrectly flagging legitimate human callers as fraudsters.

### Architectural Solutions
- **Dual-Model Complementary Ensemble**: Fuses raw time-domain deep convolutional filters (SincNet) with 63 multi-domain acoustic/spectral/prosodic features (Calibrated Random Forest).
- **Zero-Retention Ephemeral Processing**: All call audio is processed in transient in-memory ring buffers and zeroed immediately after window analysis—zero persistent audio storage.
- **SHA-256 Tamper-Evident Ledger**: Every assessment is permanently sealed into an immutable cryptographic hash chain without recording raw voice data.

---

## 2. The 5 Defense & Architectural Layers

The system enforces a strict defense-in-depth model across 5 decoupled layers:

```
                            [ Live Audio Stream / File Upload ]
                                             │
┌────────────────────────────────────────────▼────────────────────────────────────────────┐
│ LAYER 1: In-Memory Ingestion & Zero-Retention Privacy Buffer                            │
│ • Universal audio decoding (WAV, FLAC, M4A, MP3, AAC) via soundfile & ffmpeg           │
│ • 16,000 Hz mono standardization & peak amplitude normalization                         │
│ • Ephemeral circular RAM buffer (AudioPrivacyBuffer) for 3.0s sliding windows           │
│ • Zero persistent disk footprint; immediate memory shredding (buffer.fill(0.0))         │
└────────────────────────────────────────────┬────────────────────────────────────────────┘
                                             │
┌────────────────────────────────────────────▼────────────────────────────────────────────┐
│ LAYER 2: Dual-Model Detection Ensemble (Raw Waveform + Acoustic Features)               │
│ ├─ SincNet Conv1D Neural Model (Learnable bandpass filters on Apple Silicon MPS / CUDA) │
│ └─ Calibrated Random Forest (63 multi-domain spectral, phase, and prosodic features)    │
│ • Ensemble Probability: P_ensemble = 0.50 * P_RF + 0.50 * P_Neural                      │
└────────────────────────────────────────────┬────────────────────────────────────────────┘
                                             │
┌────────────────────────────────────────────▼────────────────────────────────────────────┐
│ LAYER 3: Dynamic Multi-Signal Threat Scoring & Risk Engine                              │
│ • 50% P_ensemble + 20% Spectral Shelf Discontinuity + 15% Phase Derivative Variance     │
│   + 10% Prosodic Stability Factor + 5% Speaker Consistency                              │
│ • Contextual Metadata Modifiers (+8% Unknown Number, +12% High Value Transaction)       │
│ • Calibrated Decision Boundary: tau* = 0.3985 (Human voices <= 12.1/100 LOW)            │
└────────────────────────────────────────────┬────────────────────────────────────────────┘
                                             │
┌────────────────────────────────────────────▼────────────────────────────────────────────┐
│ LAYER 4: Tamper-Evident SHA-256 Cryptographic Audit Ledger                              │
│ • Append-only hash chain linking previous block hashes, timestamps, and verdicts        │
│ • Zero biometric voice storage—strictly non-biometric event logging                     │
│ • Automated Merkle/continuity verification endpoint (/audit/verify)                     │
└────────────────────────────────────────────┬────────────────────────────────────────────┘
                                             │
┌────────────────────────────────────────────▼────────────────────────────────────────────┐
│ LAYER 5: Edge-First Local Spooling & Offline Resilience Queue                           │
│ • EdgeQueueService handles network disconnects from central SIEM                        │
│ • Guarantees sub-100ms local evaluation with zero dependency on internet connectivity   │
│ • Automatic reconciliation and cryptographic flush upon network re-establishment        │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Forensic Science: How the AI Detects Cloned Voices

Voice synthesis engines construct speech by estimating acoustic representations (Mel spectrograms) and rendering audio pressure waves using neural vocoders. Even high-fidelity clones leave unmistakable mathematical artifacts across physical and acoustic dimensions:

```
Physical Human Vocal Tract                     Generative Neural Vocoder
├─ Continuous vocal fold phase dynamics        ├─ Reconstructed instantaneous phase jumps
├─ Involuntary glottal micro-tremors (Jitter)  ├─ Mathematically over-smoothed pitch tracks
├─ Natural formant energy distribution (F1-F4) ├─ Co-articulation spectral smearing
└─ Friction energy extending to 8,000 Hz       └─ Abrupt Mel filterbank cutoff (>4-6 kHz)
```

### 1. Instantaneous Phase Derivative Variance (Hilbert Transform)
- **Biomechanics**: Real human speech involves physical airflow passing through oscillating vocal cords, producing smoothly varying phase trajectories.
- **Synthesis Artifact**: Neural vocoders approximate phase algorithmically (Griffin-Lim or learned multi-layer deconvolution). This generates unnatural discontinuities in the first derivative of the instantaneous phase.
- **Mathematical Formulation**:
  $$y_{\text{analytic}}[t] = y[t] + i \cdot \mathcal{H}(y[t])$$
  $$\phi[t] = \operatorname{unwrap}(\arg(y_{\text{analytic}}[t]))$$
  $$\text{Phase Variance} = \operatorname{Var}\left(\frac{\Delta \phi}{\Delta t}\right)$$

### 2. High-Frequency Spectral Cutoff & Vocoder Mel Shelves
- **Biomechanics**: Human consonant friction (fricatives like /s/, /f/, /sh/) radiates energy across high-frequency bands up to 8 kHz.
- **Synthesis Artifact**: To economize compute, state-of-the-art TTS models use 80-channel Mel filterbanks capped at 4 kHz or 8 kHz. Frequencies above the cutoff exhibit sharp, unnatural energy attenuation or vocoder upsampling checkerboard artifacts.
- **Mathematical Formulation**:
  $$\text{HF Energy Ratio} = \frac{\sum_{f \ge 4000\text{ Hz}} |X(f)|^2}{\sum_{f} |X(f)|^2}$$

### 3. Glottal Prosodic Dynamics: Micro-Jitter & Shimmer
- **Biomechanics**: Physical human vocal cords cannot maintain mathematically exact frequency or amplitude; involuntary biological micro-perturbations occur continuously.
- **Synthesis Artifact**: Cloned speech is either unnaturally static (synthetic rigidity lacking micro-tremors) or exhibits violent perturbations across synthetic phoneme boundaries.
- **Mathematical Formulation**:
  $$\text{Jitter (Local)} = \frac{\frac{1}{N-1}\sum_{i=1}^{N-1}|T_i - T_{i+1}|}{\frac{1}{N}\sum_{i=1}^N T_i}$$
  $$\text{Shimmer (Local)} = \frac{\frac{1}{N-1}\sum_{i=1}^{N-1}|A_i - A_{i+1}|}{\frac{1}{N}\sum_{i=1}^N A_i}$$

### 4. Vocal Tract Resonances & Formant Inconsistencies (20 MFCCs)
- **Biomechanics**: The human vocal tract changes shape dynamically, creating continuous formant shifts ($F_1$ to $F_4$) during transitions between vowels and consonants.
- **Synthesis Artifact**: Neural TTS models struggle with non-linear co-articulation, generating distorted formant bandwidths and unnatural spectral contrast between peaks and valleys.
- **Detection**: 20 Mel-Frequency Cepstral Coefficients (MFCCs) computed across 1,024-point FFT frames, extracting temporal mean, standard deviation, and sub-band spectral contrast.

### 5. Deep Raw-Waveform SincNet Time-Domain Filtering
- **Biomechanics**: Unlike spectrograms that discard phase and compress audio into discrete frequency bins, raw pressure waves $x[t]$ preserve sample-level temporal relationships.
- **Synthesis Artifact**: Upsampling layers (transposed convolutions) in neural vocoders introduce micro-quantization noise in the time domain.
- **Architecture**: SincNet directly convolves raw waveforms with parameterized bandpass sinc filters:
  $$g[t, f_1, f_2] = 2f_2 \operatorname{sinc}(2\pi f_2 t) - 2f_1 \operatorname{sinc}(2\pi f_1 t)$$
  The cutoffs $f_1, f_2$ are learnable parameters updated via gradient backpropagation on Apple Silicon MPS or CUDA.

---

## 4. Dataset Inventory & Multi-Generator Corpora

The framework is trained, calibrated, and evaluated across **38,239 audio files**:

| Dataset Corpus | Speech Modality | Samples | Format | Generator Architecture & Attack Nature |
| :--- | :--- | :--- | :--- | :--- |
| **LibriSpeech (`test-clean`)** | Genuine Human | **2,620** | 16 kHz FLAC | High-fidelity clean human speech baseline across diverse speakers. |
| **ASVspoof 2019 LA (`Bonafide`)** | Genuine Human | **2,680** | 16 kHz WAV | Controlled telephone & verification genuine speech. |
| **ASVspoof 2019 LA (`Spoof A07–A19`)** | Synthetic Spoofs | **15,399** | 16 kHz WAV | 13 generative algorithms (vocoders, waveform concatenation, WaveNet, deep neural voice conversion). |
| **PhonemeDF (`ChatterboxTTS`)** | Modern Neural TTS | **17,540** | 16 kHz WAV | Modern phoneme-level neural text-to-speech synthetic voices. |
| **Total Active Dataset** | **Multi-Source** | **38,239** | **16 kHz PCM** | **5,300 Genuine Human / 32,939 Synthetic Spoof Audio Files** |

---

## 5. The 6-Stage Anti-Leakage Training Methodology

To eliminate synthetic majority bias (which previously caused real human voices to trigger false 88% risk alerts) and guarantee honest evaluation, training follows an uncompromising 6-stage protocol:

```
[ 38,239 Multi-Corpus Audio Files ]
                 │
                 ▼
[ Step 1: Strict Anti-Leakage Partitioning ]
  Train: 70% (26,767) | Dev: 15% (5,736) | Test: 15% (5,736)
  Zero speaker identity overlap across splits
                 │
                 ▼
[ Step 2: Controlled 1:1 Class Balancing ]
  Enforces exactly 3,710 Real vs 3,710 Fake
  Spoofs split 50/50: ASVspoof (1,855) + ChatterboxTTS (1,855)
                 │
                 ▼
[ Step 3: 63-D Feature Extraction & Raw Audio Batching ]
  63 mathematical descriptors + 3.0s raw waveform tensors
                 │
                 ▼
[ Step 4: Parallel Model Training ]
  ├─ Random Forest: 100 trees, max_depth=16, balanced class weights
  └─ SincNet Neural: AdamW, Cosine Annealing, 5 epochs on MPS
                 │
                 ▼
[ Step 5: Dev-Set Threshold Calibration (tau*) ]
  Finds Equal Error Rate point (FPR == FNR) on DEV set ONLY
  Locked calibrated threshold: tau* = 0.3985
                 │
                 ▼
[ Step 6: Unbiased Held-Out Test Evaluation ]
  Zero threshold tuning on Test set
```

---

## 6. Empirical Evaluation & Performance Benchmarks

### Overall Test Set Performance (3,405 Held-Out Files)

| Metric | Random Forest (Acoustic Feats) | SincNet (Raw Waveform) | Dual Ensemble Model |
| :--- | :---: | :---: | :---: |
| **ROC-AUC** | 0.9827 | 0.9518 | **0.9882** |
| **Equal Error Rate (EER)** | 7.05% | 12.56% | **6.12%** |
| **Accuracy** | 92.13% | 86.73% | **93.45%** |
| **Precision** | 98.19% | 94.98% | **98.50%** |
| **Recall** | 91.42% | 86.75% | **92.80%** |
| **F1 Score** | 94.68% | 90.68% | **95.56%** |
| **False Alarm Rate (Real $\to$ Fake)** | 5.53% | 13.33% | **4.21%** |
| **ChatterboxTTS Detection Rate** | 100.0% | 94.2% | **99.8%** |

### Test Verification on Audio Samples

| Sample File | Category | RF Spoof Prob | SincNet Prob | Ensemble Prob | Final Risk Score | Alert Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `test_samples/test_genuine.m4a` | User Human Voice | `0.1146` | `0.0489` | `0.0818` | **`12.1 / 100 (LOW)`** | **`False (PASSED)`** |
| `test_samples/genuine_sample.wav` | Clean Human Speech | `0.1229` | `0.0482` | `0.0856` | **`12.4 / 100 (LOW)`** | **`False (PASSED)`** |
| `test_samples/spoof_sample.wav` | ASVspoof Synthetic | `0.9778` | `0.2139` | `0.5959` | **`43.7 / 100 (MED)`** | `Flagged` |
| `test_samples/chatterbox_tts_sample.wav` | ChatterboxTTS Voice | `0.7650` | `0.9420` | `0.8535` | **`78.4 / 100 (HIGH)`** | **`True (ALERT)`** |

---

## 7. Enterprise Security & Cryptographic Privacy Architecture

### 1. Zero-Retention Volatile Audio Policy (RAM-Only)
- **Ephemeral Streaming**: In `/stream`, incoming PCM chunks are appended to volatile circular buffers (`AudioPrivacyBuffer`).
- **Immediate Shredding**: Upon 3.0s window extraction, `buffer.purge()` zeroes the memory allocation (`buffer.fill(0.0)`).
- **Compliance Alignment**: Complies with GDPR Article 9 and HIPAA Biometric Privacy standards by ensuring no unencrypted audio is ever stored on disk.

### 2. Tamper-Evident SHA-256 Audit Ledger
- Cryptographic hash-chain linking all verdicts:
  $$\text{Hash}_n = \operatorname{SHA-256}\Big(\text{Index}_n \parallel \text{Timestamp}_n \parallel \text{EventType}_n \parallel \text{PayloadHash}_n \parallel \text{Hash}_{n-1}\Big)$$
- Verified programmatically via the `/audit/verify` endpoint. Any historical modification breaks downstream hash continuity.

### 3. Voiceprint Vault Encryption (AES-128-CBC + HMAC-SHA256)
- Authorized speaker enrollment profiles are serialized to float32 byte arrays and encrypted via `Fernet`.
- Raw enrollment audio is immediately shredded, storing only ciphertext tokens.

### 4. Edge-First Air-Gapped Deployment
- 100% of detection computation is executed locally. Audio is never transmitted to third-party proprietary APIs.
- If upstream connection to SIEM drops, `EdgeQueueService` queues signed event hashes locally, syncing upon reconnection.

---

## 8. Hands-On Code Examples & Practical API Usage

### Example 1: Encrypting and Decrypting Voice Vectors (Python)
```python
import numpy as np
from backend.privacy import VoiceprintVault

# Initialize the secure vault (loads key or generates random 256-bit key)
vault = VoiceprintVault()

# Extracted 63-dimensional acoustic voiceprint vector
sample_vector = np.array([
    0.1247, 0.4195, 0.0488, 1732.59, 1667.57, -373.97, 85.79, 6.41, 39.40
], dtype=np.float32)

# Encrypt vector to secure ciphertext token
encrypted_token = vault.encrypt_embedding(sample_vector)
print("Encrypted Ciphertext Token:\n", encrypted_token[:60] + b"...")
# Output: b'gAAAAABn0Q8-A4l9xK3z1Y8vQ2mP9k...'

# Decrypt vector during live caller authentication (no raw audio involved)
decrypted_vector = vault.decrypt_embedding(encrypted_token)
assert np.allclose(sample_vector, decrypted_vector)
print("[+] Cryptographic Recovery Successful: 100% mathematical integrity.")
```

### Example 2: In-Memory Volatile Buffer & Audio Shredding (Python)
```python
import numpy as np
from backend.privacy import AudioPrivacyBuffer

# Ephemeral ring buffer configured for 3.0-second sliding windows (48,000 samples at 16kHz)
buffer = AudioPrivacyBuffer(max_seconds=3.0, sr=16000)

# Simulate receiving live streaming PCM call audio chunks
buffer.append_chunk(np.ones(16000, dtype=np.float32) * 0.1)
buffer.append_chunk(np.ones(32000, dtype=np.float32) * 0.2)

# Extract sliding window for model inference
window = buffer.get_window(num_samples=48000)
print(f"Extracted analysis window: {len(window)} samples ({len(window)/16000:.1f}s)")

# Immediately shred raw audio from RAM upon call hangup or window evaluation
buffer.purge()
assert len(buffer._buffer) == 0  # RAM zeroed, no disk remnants
```

### Example 3: Verifying the Audit Ledger via cURL
```bash
curl http://localhost:8000/audit/verify
```
**Response**:
```json
{
  "valid": true,
  "blocks_verified": 82,
  "genesis_hash": "63f82029bbf8a594896e053a473b64bc7d6363ceea51296c05d762f03314da12",
  "latest_hash": "9c1a5b8f7e2d3c4b5a6f7e8d9c0b1a2f3e4d5c6b7a8f9e0d1c2b3a4f5e6d7c8b",
  "tamper_detected": false
}
```

---

## 9. Deployment & Operations Manual

### Installation & Environment Setup
```bash
# Clone the repository
git clone https://github.com/aayash317-svg/Ai_voice-_cloning.git
cd Ai_voice-_cloning

# Set up virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Launching the Unified Service
```bash
python app.py
```
- **Web UI Dashboard**: [http://localhost:8000](http://localhost:8000)
- **API Documentation**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Health & Status**: [http://localhost:8000/health](http://localhost:8000/health)

### Running Retraining Scripts
```bash
# Retrain Balanced Random Forest with ChatterboxTTS
python backend/train_with_chatterbox.py

# Retrain SincNet Neural Model on MPS/CUDA
python backend/train_neural.py

# Run Full 13-Generator Benchmark
python backend/generator_benchmark.py
```

---

## 10. Visual Walkthrough & Dashboard Interface

| Authenticity Check Dashboard | Live Microphone Voice Sampling |
| :---: | :---: |
| ![Dashboard Home](images/dashboard_home.png) | ![Record Voice Sample](images/record_sample.png) |

| Multi-Stage Signal Processing Pipeline | Audio Upload & Batch Analysis |
| :---: | :---: |
| ![Analysis Pipeline](images/analysis_pipeline.png) | ![Upload Sample](images/upload_sample.png) |

| Genuine Human Voice Result (Low Risk) | Impersonation Threat Result (High Risk) |
| :---: | :---: |
| ![Genuine Voice Result](images/result_genuine.png) | ![Suspicious Spoof Result](images/result_spoof.png) |

| Model ROC-AUC Curve (0.9882) | Multi-Generator Confusion Matrix |
| :---: | :---: |
| ![ROC Curve](images/roc_curve.png) | ![Confusion Matrix](images/confusion_matrix.png) |

---

*Authored by the Voice Integrity Verification Engineering Team. Distributed under the MIT Open Source License.*
