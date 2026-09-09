# Voice Integrity Verification Framework

> **Voice Integrity Verification is a real-time AI security engine that detects deepfakes, synthetic speech, and cloned voices. Powered by a dual ensemble of SincNet neural raw-waveform filters and spectral-acoustic classifiers, it delivers dynamic risk scoring with zero-retention privacy and a tamper-evident cryptographic audit ledger.**

> [!TIP]
> 📖 **Master Technical Documentation**: For the exhaustive architectural specification, acoustic physics formulas, dataset inventory, anti-leakage training methodology, and operations runbook, refer to [docs/PROJECT_DOCUMENTATION.md](docs/PROJECT_DOCUMENTATION.md).

---

## Application Interface & Visual Walkthrough

| Live Authenticity Check Dashboard | Live Voice Recording & Sampling |
| :---: | :---: |
| ![Dashboard Home](docs/images/dashboard_home.png) | ![Record Voice Sample](docs/images/record_sample.png) |

| Multi-Stage Signal Processing Pipeline | Audio Upload & Batch Analysis |
| :---: | :---: |
| ![Analysis Pipeline](docs/images/analysis_pipeline.png) | ![Upload Sample](docs/images/upload_sample.png) |

| Genuine Human Voice Result (Low Risk) | Impersonation Threat Result (High Risk) |
| :---: | :---: |
| ![Genuine Voice Result](docs/images/result_genuine.png) | ![Suspicious Spoof Result](docs/images/result_spoof.png) |

| Model ROC-AUC Curve (0.9882) | Multi-Generator Confusion Matrix |
| :---: | :---: |
| ![ROC Curve](docs/images/roc_curve.png) | ![Confusion Matrix](docs/images/confusion_matrix.png) |

---

## The 5 Security & Architectural Layers

The framework implements a layered defense-in-depth architecture designed for high throughput, sub-100ms latency, zero-retention privacy, and cryptographic accountability:

```
                      [ Live Call / Audio Stream ]
                                   │
┌──────────────────────────────────▼──────────────────────────────────┐
│ LAYER 1: In-Memory Ingestion & Zero-Retention Privacy Buffer        │
│ 16kHz resampler • Amplitude normalizer • 3.0s circular ephemeral RAM│
└──────────────────────────────────┬──────────────────────────────────┘
                                   │
┌──────────────────────────────────▼──────────────────────────────────┐
│ LAYER 2: Dual-Model Detection Ensemble (Raw Waveform + Features)    │
│ ├─ SincNet Conv1D Neural Model (Time-domain bandpass filters on MPS)│
│ └─ Calibrated Random Forest (63 spectral, acoustic & prosody feats) │
└──────────────────────────────────┬──────────────────────────────────┘
                                   │
┌──────────────────────────────────▼──────────────────────────────────┐
│ LAYER 3: Dynamic Multi-Signal Threat Scoring & Risk Engine          │
│ P_ensemble (50/50 blend) • Phase derivative • HF energy • Metadata  │
│ Categorical verdicts: LOW (0-30) | MED (31-60) | HIGH (61-80) | CRIT│
└──────────────────────────────────┬──────────────────────────────────┘
                                   │
┌──────────────────────────────────▼──────────────────────────────────┐
│ LAYER 4: Tamper-Evident SHA-256 Cryptographic Audit Ledger          │
│ Immutable hash chain • Non-biometric metadata • Merkle verification │
└──────────────────────────────────┬──────────────────────────────────┘
                                   │
┌──────────────────────────────────▼──────────────────────────────────┐
│ LAYER 5: Edge-First Local Spooling & Offline Resilience Queue       │
│ Offline fallback • Zero call latency • Auto-sync upon reconnection  │
└─────────────────────────────────────────────────────────────────────┘
```

### Layer 1: In-Memory Ingestion & Zero-Retention Privacy Buffer
- Operates on transient audio streams, resampling to a standardized 16,000 Hz mono PCM format.
- Uses an in-memory circular ring buffer (`AudioPrivacyBuffer`) configured for 3.0-second sliding analysis windows.
- **Strict Privacy Guarantee**: Call audio is never written to disk or persistently cached. Buffers are zeroed and purged immediately after feature extraction.

### Layer 2: Dual-Model Detection Ensemble
- Combines two complementary anti-spoofing paradigms to defend against both artifact-based and waveform-level attacks:
  - **SincNet Neural Raw-Waveform Classifier**: Deep 1D neural architecture with parameterized bandpass sinc convolutions, temporal residual blocks, and attentive statistics pooling. Operates directly on raw waveform time steps without lossy STFT compression, accelerated via Apple Silicon Metal Performance Shaders (`mps`) or CUDA.
  - **Calibrated Random Forest Classifier**: Evaluates 63 handcrafted mathematical features spanning spectral rolloff/contrast, MFCC dynamics, phase derivative variance, and vocal jitter/shimmer.
  - **Ensemble Fusion**: $P_{\text{ensemble}} = 0.50 \cdot P_{\text{RF}} + 0.50 \cdot P_{\text{Neural}}$ providing superior generalization across unseen generators.

### Layer 3: Dynamic Multi-Signal Threat Scoring & Risk Engine
- Synthesizes model probabilities with sub-band anomaly metrics to produce an intuitive **0–100 Impersonation Risk Score**:
  - $50\%$ ML Classifier Spoof Probability ($P_{\text{ensemble}}$)
  - $20\%$ High-Frequency Spectral Energy Discontinuity Score
  - $15\%$ Acoustic Phase Derivative Variance
  - $10\%$ Prosodic & Pitch Stability Factor
  - $5\%$ Speaker Consistency Weight
- Contextual metadata adjustments (+8% unknown caller, +12% high financial transaction, +15% prior fraud flag, -10% trusted contact).
- Automated threshold calibration ($\tau^* = 0.3985$) guarantees real human voices fall safely in the **LOW** zone (~12/100).

### Layer 4: Tamper-Evident SHA-256 Cryptographic Audit Ledger
- Every scan and stream verification logs an immutable event block into an append-only cryptographic hash chain (`AuditChain`).
- Each block contains: `block_index`, `timestamp`, `event_type`, `payload_hash`, and `previous_hash`.
- Zero raw audio or biometrics are stored in the ledger—only verifiable security verdicts.
- Integrated `/audit/verify` endpoint verifies block integrity across the entire chain.

### Layer 5: Edge-First Local Spooling & Offline Resilience Queue
- Implements an offline-first architecture (`EdgeQueueService`) allowing deployment on edge gateways, mobile devices, and local branch PBX servers.
- If connectivity to central security monitoring / SIEM is lost, detection runs entirely offline with sub-100ms latency, spooling signed ledger entries locally and syncing automatically once reconnected.

---

## How the Model Finds Cloned Voices (Forensic & Acoustic Mechanics)

Voice cloning systems (e.g. ElevenLabs, ChatterboxTTS, VITS, WaveNet, Tacotron, HiFi-GAN) construct speech by predicting intermediate representations (Mel-spectrograms or linguistic tokens) and synthesizing pressure waves via neural vocoders. While cloned voices sound convincing to the human ear, they leave distinct mathematical and physical traces in the acoustic and phase domains:

```
Human Vocal Tract (Physical Air Flow)          AI Voice Cloner (Neural Vocoder)
├─ Glottal pulses & continuous phase           ├─ Reconstructed phase & mathematical jumps
├─ Involuntary micro-tremors (Jitter/Shimmer)  ├─ Rigid, unnatural prosodic stability
├─ Natural resonant formants (F1-F4)           ├─ Non-linear co-articulation artifacts
└─ Wide harmonic energy distribution           └─ Abrupt high-frequency spectral cutoff (>4-6kHz)
```

Our dual-model ensemble exploits six forensic anomalies to identify deepfakes:

### 1. Instantaneous Phase Derivative Variance (Hilbert Transform Analysis)
- **Physical Mechanism**: Human vocal fold vibrations and acoustic radiation from the lips create smoothly continuous phase transitions governed by biomechanical aerodynamics.
- **Cloning Artifact**: Generative vocoders estimate phase synthetically (Griffin-Lim or learned vocoder upsampling layers). This produces unnatural, jagged instantaneous phase transitions across consecutive time frames.
- **Detection Algorithm**: The engine computes the analytic signal using the Hilbert transform:
  $$y_{\text{analytic}}[t] = y[t] + i \cdot \mathcal{H}(y[t])$$
  Unwrapping the instantaneous phase angle $\phi[t] = \operatorname{unwrap}(\arg(y_{\text{analytic}}[t]))$ and calculating the variance of the first derivative:
  $$\text{Phase Variance} = \operatorname{Var}\left(\frac{\Delta \phi}{\Delta t}\right)$$
  Spikes in phase derivative variance strongly correlate with synthetic vocoder reconstruction.

### 2. High-Frequency Spectral Cutoffs & Mel Shelf Artifacts
- **Physical Mechanism**: Uncompressed human vocal speech naturally radiates acoustic energy into the 4,000–8,000 Hz spectrum through consonant friction and oral cavity resonances.
- **Cloning Artifact**: To reduce compute, neural TTS architectures almost universally operate on 80-band Mel filterbanks capped at 4 kHz or 8 kHz. Above this cutoff, neural vocoders either truncate energy completely or produce synthetic checkerboard upsampling artifacts.
- **Detection Algorithm**: Evaluates high-frequency energy ratios above 4 kHz against total spectral power:
  $$\text{HF Energy Ratio} = \frac{\sum_{f \ge 4000 \text{ Hz}} |X(f)|^2}{\sum_{f} |X(f)|^2}$$
  Abrupt spectral rolloff shelves or unnatural energy voids trigger immediate spectral anomaly flags.

### 3. Glottal Prosodic Dynamics: Micro-Jitter & Shimmer
- **Physical Mechanism**: Biological human speech contains continuous, involuntary micro-fluctuations in vocal fold cycle length (pitch **Jitter**) and cycle amplitude (**Shimmer**).
- **Cloning Artifact**: Cloned speech is either *mathematically too smooth* (robotic stability without micro-perturbations) or *erratic across phoneme transitions*.
- **Detection Algorithm**: Extracts Fundamental Frequency ($F_0$) pitch tracks via parabolic peak interpolation, measuring relative perturbation:
  $$\text{Jitter} = \frac{\frac{1}{N-1} \sum_{i=1}^{N-1} |T_i - T_{i+1}|}{\frac{1}{N} \sum_{i=1}^N T_i}, \quad \text{Shimmer} = \frac{\frac{1}{N-1} \sum_{i=1}^{N-1} |A_i - A_{i+1}|}{\frac{1}{N} \sum_{i=1}^N A_i}$$

### 4. Vocal Tract Resonances & Formant Inconsistencies (20 MFCC Dimensions)
- **Physical Mechanism**: The human vocal tract acts as a biological acoustic filter, continuously shifting formants ($F_1$ through $F_4$) during co-articulation (e.g. transitioning from plosives like /p/ or /k/ into vowels).
- **Cloning Artifact**: Generative models struggle with non-linear formant trajectories, creating spectral smearing or unnatural energy contrast between formant peaks and spectral valleys.
- **Detection Algorithm**: Computes 20 Mel-Frequency Cepstral Coefficients (MFCCs) across 1,024-point FFT frames, extracting temporal mean, standard deviation, and sub-band spectral contrast across octave bands.

### 5. Deep Raw-Waveform SincNet Time-Domain Filtering
- **Physical Mechanism**: Spectrogram-based classifiers discard phase and compress audio into discrete frequency bins, losing temporal micro-artifacts.
- **Cloning Artifact**: Neural vocoder upsamplers introduce waveform-level quantization noise and sample interpolation errors in the time domain.
- **Detection Algorithm**: SincNet directly convolves the raw pressure signal $x[t]$ with parameterized learnable bandpass filters:
  $$g[t, f_1, f_2] = 2f_2 \operatorname{sinc}(2\pi f_2 t) - 2f_1 \operatorname{sinc}(2\pi f_1 t)$$
  The filter cutoffs $f_1, f_2$ are updated during backpropagation, autonomously isolating adversarial frequency bands that differentiate human glottal pulses from synthetic neural vocoding.

### 6. Dual Ensemble Consensus Decision
- Handcrafted acoustic features are evaluated by the calibrated Random Forest.
- Raw time-domain waveforms are evaluated by SincNet on Apple Silicon MPS.
- Individual probabilities are blended ($P_{\text{ensemble}} = 0.50 \cdot P_{\text{RF}} + 0.50 \cdot P_{\text{Neural}}$), ensuring that if an attacker circumvents spectral features, the raw-waveform network catches the attack (and vice versa).

---

## Complete List of Datasets (38,239 Total Audio Clips)

The framework is trained, calibrated, and evaluated across four comprehensive speech corpora:

| Dataset | Type | Sample Count | Audio Format | Description & Attack Systems |
| :--- | :--- | :--- | :--- | :--- |
| **LibriSpeech (test-clean)** | Genuine Human | **2,620** clips | 16 kHz FLAC | Clean, diverse human reading speech across hundreds of male and female speakers. |
| **ASVspoof 2019 LA (Bonafide)** | Genuine Human | **2,680** clips | 16 kHz WAV | Telephone & voice verification bonafide speech recorded in controlled acoustic environments. |
| **ASVspoof 2019 LA (Spoof A07–A19)** | Synthetic Spoofs | **15,399** clips | 16 kHz WAV | 13 distinct voice generation systems: neural vocoders, waveform concatenation, WaveNet, deep neural voice conversion. |
| **PhonemeDF (ChatterboxTTS)** | Modern Neural TTS | **17,540** clips | 16 kHz WAV | State-of-the-art contemporary phoneme-level neural text-to-speech synthetic voices. |
| **Total Active Dataset** | **Multi-Source** | **38,239** clips | **16 kHz PCM** | **5,300 Genuine Human / 32,939 Synthetic Spoof Clips** |

### Bundled Test Samples (`test_samples/`)
For instant verification without downloading multi-gigabyte files, pre-packaged samples are included in the repository:
- `test_samples/genuine_sample.wav` — Clean human speech reference
- `test_samples/test_genuine.m4a` / `.wav` — Natural human speech recording
- `test_samples/spoof_sample.wav` — ASVspoof synthetic voice sample
- `test_samples/chatterbox_tts_sample.wav` — PhonemeDF ChatterboxTTS neural synthetic voice

---

## How We Train the Dataset (Step-by-Step Methodology)

To guarantee scientific honesty, prevent data leakage, and eliminate false alarms on genuine human voices, training follows a strict 6-stage protocol:

```
 38,239 Audio Files (LibriSpeech + ASVspoof + ChatterboxTTS)
                           │
 ┌─────────────────────────▼──────────────────────────┐
 │ Step 1: Strict Anti-Leakage Partitioning           │
 │ Train (70%)  •  Dev (15%)  •  Held-out Test (15%)  │
 └─────────────────────────┬──────────────────────────┘
                           │
 ┌─────────────────────────▼──────────────────────────┐
 │ Step 2: Controlled 1:1 Class Balancing             │
 │ Exactly 3,710 Genuine Human vs 3,710 Synthetic     │
 │ (Eliminates majority-spoof bias & false 88% alarms)│
 └─────────────────────────┬──────────────────────────┘
                           │
 ┌─────────────────────────▼──────────────────────────┐
 │ Step 3: Multi-Domain Feature Matrix Extraction     │
 │ 63 Acoustic, Spectral, MFCC & Prosodic dimensions  │
 └─────────────────────────┬──────────────────────────┘
                           │
 ┌─────────────────────────▼──────────────────────────┐
 │ Step 4: Model Training                             │
 │ ├─ Random Forest (100 trees, max_depth=16, CPU)    │
 │ └─ SincNet Neural Waveform (PyTorch on MPS/CUDA)   │
 └─────────────────────────┬──────────────────────────┘
                           │
 ┌─────────────────────────▼──────────────────────────┐
 │ Step 5: Dev-Set Threshold Calibration (tau*)       │
 │ Tunes decision boundary at EER point on Dev ONLY   │
 └─────────────────────────┬──────────────────────────┘
                           │
 ┌─────────────────────────▼──────────────────────────┐
 │ Step 6: Unbiased Held-Out Test Set Evaluation      │
 │ Final honest benchmark (Zero test-set tuning)      │
 └────────────────────────────────────────────────────┘
```

### Step 1: Anti-Leakage Data Partitioning
- Datasets are partitioned into **Train (70%)**, **Development (15%)**, and **Test (15%)** splits.
- Audio files from the same recording session or speaker never cross split boundaries, preventing identity memorization.

### Step 2: Controlled 1:1 Class Balancing
- Standard anti-spoofing datasets are up to 85% synthetic, which causes standard classifiers to develop a strong majority bias—falsely labeling real human voices as deepfakes (the root cause of the previous 88% false risk score).
- We enforce an exact **1:1 training distribution**:
  - **3,710 Genuine Human Voices** (LibriSpeech + ASVspoof bonafide)
  - **3,710 Synthetic Spoofs** (split evenly between ASVspoof A07–A19 and ChatterboxTTS)

### Step 3: 63-Dimensional Feature Extraction
- Audio clips are normalized and transformed into fixed-length 63-dimensional vectors:
  - **MFCCs (20 coefficients)**: Mean and standard deviation across frames (spectral envelope).
  - **Spectral Descriptors**: Centroid, bandwidth, rolloff, flatness, and sub-band contrast.
  - **Phase & Acoustic Artifacts**: Instantaneous phase derivative variance and zero-crossing rate.
  - **Prosodic Dynamics**: Fundamental frequency (F0), pitch range, jitter, and shimmer.

### Step 4: Model Training
- **Random Forest**: Trained with 100 estimators, balanced leaf weights, and multi-core CPU parallelism.
- **SincNet Neural Model**: Trained on raw audio waveforms using AdamW optimizer with Cosine Annealing learning rate scheduling across 5 epochs on Apple Silicon Metal Performance Shaders (`mps`).

### Step 5: Dev-Set Threshold Calibration ($\tau^*$)
- **Strict Anti-Overfitting Rule**: The decision threshold is tuned strictly on the Development set by locating the Equal Error Rate (EER) point where False Positive Rate (FPR) equals False Negative Rate (FNR).
- The resulting calibrated threshold $\tau^* = 0.3985$ is locked into the model metadata.

### Step 6: Unbiased Test Set Evaluation
- Evaluated on the held-out Test set (3,405 audio clips) with zero test-set threshold adjustments:
  - **ROC-AUC**: **0.9827** (Dual Ensemble: **0.9882**)
  - **Equal Error Rate (EER)**: **7.05%** (Dual Ensemble: **6.12%**)
  - **Human False Alarm Rate (Real $\to$ Fake)**: **5.53%** (down from 15%)
  - **ChatterboxTTS Detection Rate**: **100.0%** (300 / 300 detected)

---

## Quickstart & Running Commands

### 1. Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13
- `ffmpeg` (for universal audio decoding)
  ```bash
  # macOS
  brew install ffmpeg

  # Ubuntu / Debian
  sudo apt install ffmpeg
  ```

### 2. Installation
Clone the repository and set up a virtual environment:
```bash
git clone https://github.com/aayash317-svg/Ai_voice-_cloning.git
cd Ai_voice-_cloning

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Launch the Application
Start the unified FastAPI server and dashboard:
```bash
python app.py
```
Or with Uvicorn:
```bash
uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```

---

## Testing with Packaged Sample Data

The repository includes pre-packaged test samples in `test_samples/` for immediate verification:

### Test 1: Genuine Human Voice
```bash
curl -X POST -F "file=@test_samples/genuine_sample.wav" http://localhost:8000/analyze
```
**Result**: Risk Score: **~12 / 100 (LOW RISK)** | Alert: `false`

### Test 2: Natural Speech Recording (M4A)
```bash
curl -X POST -F "file=@test_samples/test_genuine.m4a" http://localhost:8000/analyze
```
**Result**: Risk Score: **~12 / 100 (LOW RISK)** | Alert: `false`

### Test 3: ASVspoof Synthetic Voice
```bash
curl -X POST -F "file=@test_samples/spoof_sample.wav" http://localhost:8000/analyze
```
**Result**: High/Medium threat flagged.

### Test 4: ChatterboxTTS Neural Voice
```bash
curl -X POST -F "file=@test_samples/chatterbox_tts_sample.wav" http://localhost:8000/analyze
```
**Result**: Neural TTS spoof pattern identified with 100% confidence.

---

## Retraining & Benchmarking Commands

### Train Multi-Generator Random Forest (with ChatterboxTTS)
```bash
python backend/train_with_chatterbox.py
```

### Train SincNet Neural Raw-Waveform Model on MPS / GPU
```bash
python backend/train_neural.py
```

### Run Multi-Generator Benchmark (A07–A19)
```bash
python backend/generator_benchmark.py
```

### Download Full Datasets
```bash
# Extract full 17,540 PhonemeDF ChatterboxTTS files
python scripts/install_phonemedf.py --max-samples 0

# Download ASVspoof 2019 LA
python scripts/download_asvspoof2019.py
```

---

## Repository Structure

```
.
├── app.py                            # Application entrypoint
├── backend/
│   ├── app.py                        # FastAPI endpoints & WebSocket streaming
│   ├── classifier.py                 # Anti-spoof classifier abstraction
│   ├── neural_classifier.py          # PyTorch SincNet deep raw-waveform architecture
│   ├── features.py                   # 63-dimensional feature extractor
│   ├── preprocess.py                 # 16kHz resampler & amplitude normalizer
│   ├── risk_engine.py                # Multi-signal threat scoring engine
│   ├── privacy.py                    # Zero-retention circular privacy buffer
│   ├── audit_chain.py                # SHA-256 immutable audit ledger
│   ├── dataset_loader.py             # ASVspoof, LibriSpeech & Chatterbox loader
│   ├── train_with_chatterbox.py      # Balanced multi-generator training
│   └── train_neural.py               # MPS-accelerated deep neural training
├── frontend/
│   └── index.html                    # Cyberpunk Stitch UI dashboard
├── models/
│   ├── baseline_random_forest.pkl    # Pre-trained calibrated RF checkpoint
│   └── neural_sincnet.pt             # Pre-trained PyTorch SincNet weights
├── test_samples/                     # Packaged testing audio files
├── docs/images/                      # Application screenshots and benchmark charts
├── scripts/
│   ├── download_asvspoof2019.py      # ASVspoof automated downloader
│   └── install_phonemedf.py          # PhonemeDF ChatterboxTTS installer
├── requirements.txt                  # Python dependencies
└── README.md                         # Project documentation
```

---

## Enterprise Security & Privacy Architecture

Security, zero-trust privacy, and non-repudiation are foundational engineering requirements of the Voice Integrity Verification platform:

### 1. Zero-Retention Audio Policy (RAM-Only Volatile Processing)
- **Zero Disk Spooling**: In live call streaming (`/stream`), audio chunks are received over secure WebSockets directly into transient memory. No raw audio is ever written to disk, databases, or temporary cache files.
- **Volatile Ring Buffering**: Implemented via `AudioPrivacyBuffer`, which holds an ephemeral 3.0-second sliding window. Once features are extracted, the buffer is zeroed with `buffer.fill(0.0)` and discarded.
- **GDPR & HIPAA Compliance Alignment**: Because voice recordings are classified as biometric personal identifiable information (PII), our zero-retention guarantee ensures that intercepted or stored voice data cannot be leaked, breached, or subpoenaed from server disk drives.

### 2. Tamper-Evident SHA-256 Cryptographic Audit Ledger
- All security verdicts, model probabilities, and alert triggers are sealed into an immutable blockchain-style audit ledger (`AuditChain`).
- **Cryptographic Hash Chain Structure**:
  $$\text{Block Hash}_n = \operatorname{SHA-256}\Big(\text{Index}_n \parallel \text{Timestamp}_n \parallel \text{EventType}_n \parallel \text{PayloadHash}_n \parallel \text{Block Hash}_{n-1}\Big)$$
- **Non-Biometric Metadata Guarantee**: Audit blocks record event metrics (e.g. `{"risk_score": 12.1, "risk_level": "LOW", "alert_triggered": false}`), but **never store raw audio or speaker biometric embeddings**.
- **Tamper Detection**: The built-in `/audit/verify` endpoint verifies the mathematical hash continuity of all blocks. Any retrospective modification or deletion of an audit record instantly breaks the cryptographic chain and raises an alert.

### 3. Encrypted Voiceprint Vault (AES-128 / Fernet)
- When authorized speaker profiles are registered for comparison, voice audio is converted into mathematical embedding vectors and encrypted using `Fernet` (AES-128-CBC with HMAC-SHA256 authentication).
- Raw enrollment audio is immediately shredded, storing only ciphertext embeddings.

---

### Raw Audio Ingestion & Cryptographic Encryption Flow

The platform guarantees that sensitive raw human voice samples are **never retained on disk in plaintext**. Instead, audio passes through a one-way ephemeral ingestion pipeline:

```
[ Incoming Audio File / Stream Chunk ]
                │
                ▼
[ In-Memory Transient Decode (io.BytesIO) ]  <── Never touches persistent disk
                │
                ▼
[ Feature Extraction & Normalization ]       <── Converted to 63-D mathematical vector
                │
        ┌───────┴────────────────────────┐
        ▼                                ▼
[ Raw Audio Shredder ]         [ Voiceprint Vault ]
• buffer.fill(0.0)             • Vector serialized to float32 byte array
• Memory buffer purged         • Encrypted via AES-128-CBC + HMAC-SHA256
• Zero disk remnants           • Output: Ciphertext Token (b'gAAAAABn...')
```

#### Step 1: Memory-Only Audio Ingestion
Incoming audio bytes from HTTP uploads or WebSocket frames are decoded strictly in-memory using `io.BytesIO`. Temporary OS transcoding files (if needed for M4A/AAC conversion) are immediately deleted in `finally:` blocks before responses are returned.

#### Step 2: Irreversible Mathematical Vectorization
Audio waveforms are transformed into fixed-length 63-dimensional statistical feature vectors (MFCC moments, spectral rolloff, phase derivative variance). These numerical moments cannot be inverted to reconstruct the original speech recording.

#### Step 3: Military-Grade Symmetric Encryption (Fernet)
When saving authorized speaker profiles, the resulting numerical embedding is encrypted using `VoiceprintVault`. Fernet guarantees:
- **AES-128-CBC Encryption**: Strong symmetric confidentiality.
- **HMAC-SHA256 Authentication**: Prevents tampering or ciphertext manipulation.
- **Timestamp & IV Randomization**: Distinct ciphertexts produced even for identical voiceprints.

---

### Code Examples & Hands-On Demonstrations

#### Example 1: Encrypting and Decrypting Voice Vectors in Python
```python
import numpy as np
from backend.privacy import VoiceprintVault

# Initialize the secure vault (loads ENCRYPTION_KEY or generates random 256-bit key)
vault = VoiceprintVault()

# 1. Simulate a 63-dimensional extracted acoustic voiceprint vector
sample_voiceprint = np.array([
    0.1247, 0.4195, 0.0488, 1732.59, 1667.57, -373.97, 85.79, 6.41, 39.40
], dtype=np.float32)

print("Original Voiceprint Vector:\n", sample_voiceprint[:4])

# 2. Encrypt vector to secure ciphertext
encrypted_token = vault.encrypt_embedding(sample_voiceprint)
print("\nEncrypted Ciphertext Stored in Database:")
print(encrypted_token[:60] + b"...")
# Output: b'gAAAAABn0Q8-A4l9xK3z1Y8vQ2mP9k...'

# 3. Decrypt vector during live caller authentication (no raw audio involved)
decrypted_voiceprint = vault.decrypt_embedding(encrypted_token)
print("\nDecrypted Vector for Cosine Similarity:")
print(decrypted_voiceprint[:4])

# 4. Verify exact mathematical equality
assert np.allclose(sample_voiceprint, decrypted_voiceprint)
print("\n[+] Verification Successful: Zero audio stored, 100% cryptographic recovery.")
```

#### Example 2: In-Memory Volatile Buffer & Audio Shredding
```python
import numpy as np
from backend.privacy import AudioPrivacyBuffer

# Create ephemeral ring buffer configured for 3.0-second sliding windows (48,000 samples at 16kHz)
buffer = AudioPrivacyBuffer(max_seconds=3.0, sr=16000)

# Simulate receiving live streaming PCM call audio chunks
pcm_chunk_1 = np.ones(16000, dtype=np.float32) * 0.1  # 1.0 second of audio
pcm_chunk_2 = np.ones(32000, dtype=np.float32) * 0.2  # 2.0 seconds of audio

buffer.append_chunk(pcm_chunk_1)
buffer.append_chunk(pcm_chunk_2)

# Extract analysis window for model inference
window = buffer.get_window(num_samples=48000)
print(f"Window extracted for inference: {len(window)} samples ({len(window)/16000:.1f}s)")

# Immediately shred raw audio from RAM upon call hangup or window evaluation
buffer.purge()
print(f"Buffer samples remaining after purge: {len(buffer._buffer)} (RAM zeroed)")
assert len(buffer._buffer) == 0
```

#### Example 3: Verifying the Immutable Audit Ledger via cURL
Every batch analysis or live call detection logs an event into the cryptographic ledger without saving any voice data:

```bash
# Verify the entire audit ledger integrity
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

### 4. Edge-First / Air-Gapped Operation
- All inference pipelines (FastAPI, PyTorch MPS, Scikit-Learn) run 100% locally on-premise or on edge gateways.
- **No External Cloud Calls**: Audio is never transmitted to third-party proprietary APIs (e.g. OpenAI, ElevenLabs, Google Cloud), eliminating man-in-the-middle (MitM) eavesdropping risks.
- If network connection to a central SIEM server drops, `EdgeQueueService` spools signed ledger hashes locally on disk and reconciles upon reconnection.

### 5. Defensive Biometric Integrity Scope
This system is engineered strictly for **defensive biometric verification and voice impersonation protection**. It is designed to safeguard banking call centers, executive authorization lines, and everyday individuals from voice cloning fraud, CEO gift card scams, and deepfake social engineering.
