# Voice Integrity Verification Framework

> **Voice Integrity Verification is a real-time AI security engine that detects deepfakes, synthetic speech, and cloned voices. Powered by a dual ensemble of SincNet neural raw-waveform filters and spectral-acoustic classifiers, it delivers dynamic risk scoring with zero-retention privacy and a tamper-evident cryptographic audit ledger.**

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

## Key Features

- **Dual-Model Detection Ensemble**:
  - **SincNet Neural Raw-Waveform Model**: Learnable parameterized bandpass filters operating directly on raw time-domain audio samples with Apple Silicon Metal Performance Shaders (`mps`) / CUDA acceleration.
  - **Calibrated Random Forest Classifier**: Extracts 63 multi-domain features across spectral, phase derivative, and prosodic acoustic spaces.
- **Dynamic Threat Scoring**: Computes an adaptive 0–100 risk score and categorical threat classification (`LOW`, `MEDIUM`, `HIGH`, `CRITICAL`).
- **Real-Time Sliding Window Streaming**: Full-duplex WebSocket endpoint (`/stream`) analyzing 3.0-second sliding audio windows with zero lag.
- **Zero-Retention Privacy**: Circular in-memory audio buffers (`AudioPrivacyBuffer`) purged immediately after window analysis—zero persistent voice storage.
- **Cryptographic Audit Ledger**: Non-biometric security events are hashed into a tamper-evident SHA-256 blockchain-style ledger (`/audit/chain`, `/audit/verify`).
- **Interactive Cyberpunk Dashboard**: Built with Stitch UI aesthetics, featuring real-time frequency visualizers, animated threat gauges, direct microphone recording, and drag-and-drop batch analysis.

---

## Benchmark Performance

Evaluated on held-out test splits across 38,239 audio clips (LibriSpeech human speech, ASVspoof 2019 A07–A19 generators, and PhonemeDF ChatterboxTTS):

| Metric | Random Forest (Features) | SincNet (Raw Waveform) | Dual Ensemble |
| :--- | :---: | :---: | :---: |
| **ROC-AUC** | 0.9827 | 0.9518 | **0.9882** |
| **Equal Error Rate (EER)** | 7.05% | 12.56% | **6.12%** |
| **Accuracy** | 92.13% | 86.73% | **93.45%** |
| **Precision** | 98.19% | 94.98% | **98.50%** |
| **False Alarm Rate (Real $\to$ Fake)** | 5.53% | 13.33% | **4.21%** |
| **ChatterboxTTS Detection Rate** | 100.0% | 94.2% | **99.8%** |

---

## Quickstart & Running Commands

### 1. Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13
- `ffmpeg` (for audio decoding)
  ```bash
  # macOS
  brew install ffmpeg

  # Ubuntu / Debian
  sudo apt install ffmpeg
  ```

### 2. Installation
Clone the repository and set up a virtual environment:
```bash
git clone https://github.com/aayash317-svg/Ai_voice-_cloning_deduction-.git
cd Ai_voice-_cloning_deduction-

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Launch the Application
Start the unified FastAPI server and dashboard:
```bash
python app.py
```
Or directly with Uvicorn:
```bash
uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```

Once started:
- **Interactive Web UI**: Open [http://localhost:8000](http://localhost:8000)
- **API Swagger Documentation**: Open [http://localhost:8000/docs](http://localhost:8000/docs)
- **System Health & Ensemble Status**: [http://localhost:8000/health](http://localhost:8000/health)

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

## Model Training & Custom Experiments

To retrain or benchmark the models:

### Train Random Forest with 1:1 Class Balancing
```bash
python backend/train_with_chatterbox.py
```

### Train SincNet Neural Raw-Audio Model on MPS / GPU
```bash
python backend/train_neural.py
```

### Run Multi-Generator Benchmark (A07–A19)
```bash
python backend/generator_benchmark.py
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

## Security & Ethical Disclosure
This project is engineered strictly for **defensive biometric integrity and voice impersonation protection**. All audio data processed in live streams is handled exclusively in transient RAM buffers and purged immediately upon analysis.
