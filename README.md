# Voice Shield AI: Voice Integrity Verification Framework

[![GitHub Repo](https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github)](https://github.com/aayash317-svg/Ai_voice-_cloning)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110.0-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.0-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](LICENSE)
[![Tests Passing](https://img.shields.io/badge/Tests-5%2F5%20Passing-success?style=for-the-badge)](tests/)

> **One-line pitch**: A real-time cybersecurity engine that detects AI-cloned voices, synthetic speech, and deepfakes during live phone conversations and audio uploads with zero-retention privacy and cryptographic proof.

> [!TIP]
> 📖 **Master Technical Documentation**: For the exhaustive architectural specification, acoustic physics formulas, dataset inventory, anti-leakage training methodology, and operations runbook, refer to [docs/PROJECT_DOCUMENTATION.md](docs/PROJECT_DOCUMENTATION.md).

> [!IMPORTANT]
> 📑 **Presentation Guide & Architecture Brief**: For the slide-by-slide presentation deck content, visual working flow diagrams, and ML algorithms breakdown, refer to [APPLICATION_DOCUMENTATION.md](APPLICATION_DOCUMENTATION.md).

---


## 🎯 Goals

### What problem does this project solve?
Generative voice cloning systems (e.g. ElevenLabs, ChatterboxTTS, VITS, HiFi-GAN) can now clone an individual's vocal identity from as little as **3 seconds** of audio. Fraudsters exploit this to commit authorized push-payment fraud, emergency family scams, and executive impersonation over phone calls. Legacy forensic tools only process audio *after* the call finishes—when the money has already been stolen. 

**Voice Shield AI** solves this by evaluating live two-way conversational audio in real time over WebSockets with sub-90ms latency, delivering a **guaranteed verified verdict at 10–12 seconds** without saving user audio to disk.

### Who is it for?
- **Banking & Fintech Security Teams**: To prevent fraudulent telephone wire transfer authorizations.
- **Enterprise Call Centers**: To verify inbound customer voice biometric integrity in real time.
- **Smartphone Users**: To protect against AI-powered extortion and family emergency voice scams.
- **Digital Forensics & Security Researchers**: To audit synthetic speech artifacts using explainable physics.

### What does "done" look like for v1?
- Real-time conversational two-speaker diarization isolating the local caller from the remote party.
- Progressive 3-stage threat scoring: Stage 0 (0–5s calibration) $\to$ Stage 1 (5–6s preliminary average) $\to$ Stage 2 (10–12s guaranteed final verdict).
- Dual-model consensus ensemble combining PyTorch SincNet 1D raw-waveform neural filters with 63-D calibrated Random Forest classifiers.
- Zero-retention ephemeral RAM ring buffer (no audio stored on disk) and SHA-256 cryptographic audit ledger.
- Validated performance: **0.9882 ROC-AUC**, **6.12% EER**, and **100% detection rate on modern neural TTS**.

---

## 🏗️ System Architecture

<p align="center">
  <img src="docs/images/architecture_flowchart.png" alt="Voice Shield AI System Architecture" width="360" />
</p>

Voice Shield AI processes real-time two-way audio through a 6-stage defensive pipeline:
1. **Client Layer**: Captures & preprocesses mobile / VoIP audio at 16 kHz.
2. **Ingestion & Privacy**: High-speed WebSocket streaming with zero disk retention (RAM-only ring buffer).
3. **Real-Time Diarization**: Separates speakers by vocal timbre using online cosine clustering.
4. **AI Detection Ensemble**: Late fusion of SincNet 1D raw-waveform neural network and 63-D Random Forest.
5. **Risk Engine**: Multi-signal threat scoring with contextual transaction awareness.
6. **Audit & Output**: Immutable SHA-256 audit ledger, live dashboard telemetry, and instant freeze alerts.

---

## ✨ Features


### Implemented & Ready in v1
- [x] **Two-Way Live Conversational Diarization**: Isolates Speaker A (Local user) from Speaker B (Remote caller) using 40-D timbre embeddings and online cosine clustering.
- [x] **Cross-Talk Overlap Quarantine**: Automatically detects and quarantines simultaneous speech to prevent false contamination.
- [x] **Progressive 3-Stage Threat Evaluation**:
  - *Stage 0 (0–5s)*: Baseline ambient room and mic gain calibration with active `(Xs / 10s)` countdown ticker.
  - *Stage 1 (5–6s)*: Rolling 5-second preliminary average risk score.
  - *Stage 2 (10–12s)*: **Guaranteed unconditional verified verdict** (`CALL AUTHENTIC` or `AI CLONE DETECTED`).
- [x] **Dual-Model AI Ensemble**: Blends raw time-domain neural filters with physical acoustic heuristics ($P_{\text{ensemble}} = 0.50 \cdot P_{\text{RF}} + 0.50 \cdot P_{\text{Neural}}$).
- [x] **Explainable Acoustic Physics**:
  - *Hilbert Phase Derivative Variance*: Detects artificial vocoder phase jumps.
  - *High-Frequency Spectral Cutoff*: Identifies 4–6 kHz Mel filterbank energy drop-offs.
  - *Glottal Pitch Jitter & Shimmer*: Catches unnatural robotic pitch smoothness.
  - *Spectral Contrast Standard Deviation*: Evaluates formant peak vs valley dynamics across 6 octave bands.
- [x] **Zero-Retention Ephemeral Privacy Buffer**: Audio lives exclusively in a 3.0-second circular RAM ring buffer and is shredded immediately after feature extraction.
- [x] **SHA-256 Cryptographic Audit Ledger (`AuditChain`)**: Seals every completed call inspection into an append-only hash chain with block verification.
- [x] **Glassmorphic Cyber UI**: 60 FPS HTML5 canvas oscilloscope, dynamic per-speaker forensic cards, and color-coded alert banners.
- [x] **Mobile Optimization & Cloud Tunnels**: Audio keep-alive feedback loop (`0.00001` gain) and integrated Cloudflare tunnel (`cloudflared`) for live cellular phone testing.
- [x] **Batch Audio Forensic Upload**: Drag-and-drop analysis for `.wav`, `.mp3`, `.m4a`, and `.flac` files.
- [x] **1-Click Cloud Deployment**: Pre-configured `Dockerfile`, `Procfile`, and `render.yaml` for Render, Hugging Face Spaces, and Railway.

### Planned Roadmap
- [ ] **Native Mobile Call Screening**: Android `CallScreeningService` background telephony dialer hook.
- [ ] **On-Device Keyword Fraud Spotter**: Real-time keyword spotting for high-risk fraud trigger phrases (*"OTP"*, *"urgent wire transfer"*, *"police warrant"*).
- [ ] **Enterprise SIP / PBX Proxy**: Hardware-accelerated SIP trunk inspection for corporate banking call centers.
- [ ] **Adversarial Codec Data Augmentation**: Fine-tuning against low-bitrate WhatsApp VoIP and AMR-WB mobile codecs.

---

## 🗺️ Roadmap

Track ongoing engineering tasks, sprint items, and upcoming milestones on our live issue board:
👉 **[GitHub Issues & Milestones](https://github.com/aayash317-svg/Ai_voice-_cloning/issues)**

---

## 📦 Requirements

- **Operating System**: Windows 10/11, macOS (Apple Silicon or Intel), or Linux (Ubuntu 20.04+)
- **Python**: 3.10, 3.11, 3.12, or 3.13
- **Hardware Acceleration**:
  - Apple Silicon Metal Performance Shaders (`mps`) supported
  - NVIDIA CUDA GPU supported
  - CPU multi-core fallback enabled by default
- **Key Dependencies**:
  - `torch >= 2.2.0` & `torchaudio >= 2.2.0` (SincNet neural waveform network)
  - `librosa >= 0.10.1` & `scipy >= 1.12.0` (Acoustic physics & Hilbert transform)
  - `scikit-learn >= 1.4.0` (Calibrated Random Forest classifier)
  - `fastapi >= 0.110.0` & `uvicorn[standard]` (High-throughput async server)
  - `websockets >= 12.0` (Low-latency binary audio streaming)
  - `soundfile >= 0.12.1` & `pydub` (PCM audio decoding)

---

## 🚀 Installation

### 1. Prerequisites (Install FFmpeg)
FFmpeg is required for universal audio decoding:
```bash
# Windows (via Chocolatey or Scoop)
choco install ffmpeg
# or: scoop install ffmpeg

# macOS (via Homebrew)
brew install ffmpeg

# Ubuntu / Debian Linux
sudo apt update && sudo apt install -y ffmpeg
```

### 2. Clone Repository & Setup Virtual Environment

#### Windows (PowerShell):
```powershell
# Clone the repository
git clone https://github.com/aayash317-svg/Ai_voice-_cloning.git
cd Ai_voice-_cloning

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

#### Linux / macOS:
```bash
# Clone the repository
git clone https://github.com/aayash317-svg/Ai_voice-_cloning.git
cd Ai_voice-_cloning

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🔧 Usage

### Workflow 1: Launch the Interactive Dashboard & Call Shield
Start the unified FastAPI server and dashboard:
```bash
python app.py
```
- Open **`http://127.0.0.1:8000`** (or `http://127.0.0.1:8050`) in your web browser.
- Click **"Start Live Call Shield"** to monitor audio live.

### Workflow 2: Live Mobile Testing via Cloudflare Tunnel
To test live cellular phone calls from your smartphone over an encrypted public HTTPS/WSS URL:
```bash
.\cloudflared.exe tunnel --url http://127.0.0.1:8000
```
Open the generated `https://*.trycloudflare.com` URL in Chrome on Android or Safari on iOS.

> [!TIP]
> **Android Cellular Call Tip**: When placing a live cellular call on Android, turn on **Speakerphone** so Android's audio system permits Chrome to capture both caller voices simultaneously.

### Workflow 3: Command-Line Audio File Prediction
Run forensic inspection directly on an audio file from the terminal:
```bash
python backend/predict.py --audio test_samples/spoof_sample.wav
```

### Workflow 4: Run Automated Tests
Execute the unit test suite covering the classifier, features, privacy shredder, and audit ledger:
```bash
pytest tests/ -v
```

### Workflow 5: Cloud & Docker Deployment

#### A. Free 1-Click Deployment on Render.com (Recommended)
This repository includes a pre-configured [`render.yaml`](render.yaml) blueprint:
1. Log into [Render.com](https://render.com).
2. Go to **Blueprints** → Click **New Blueprint Instance**.
3. Connect your repository: `https://github.com/aayash317-svg/Ai_voice-_cloning.git`.
4. Render automatically configures the Docker web service, sets dynamic port binding (`$PORT`), and mounts the `/health` check.
5. Click **Apply** — your instance will build and be live in under 2 minutes!

> [!TIP]
> **Manual Web Service on Render**: If creating a manual Web Service on Render:
> - **Environment**: Select `Docker` (or select `Python 3`).
> - **Build Command (if Python 3)**: `./build.sh`
> - **Start Command (if Python 3)**: `python main.py`
> - **Health Check Path**: `/health`

#### B. Local or Cloud Docker Deployment
```bash
# Build the optimized production Docker image (CPU-only PyTorch, ~95% smaller)
docker build -t voice-shield-ai .

# Run container locally on port 8000
docker run -p 8000:8000 voice-shield-ai
```
Visit `http://localhost:8000` to access the dashboard.


---

## 📁 Project Structure

```
Ai_voice-_cloning/
├── backend/
│   ├── app.py                     # FastAPI WebSocket & REST endpoints
│   ├── classifier.py              # Calibrated Random Forest (63 features)
│   ├── neural_classifier.py       # PyTorch SincNet 1D raw-waveform model
│   ├── stream_diarizer.py         # Real-time streaming diarizer & progressive stages
│   ├── risk_engine.py             # Multi-signal risk fusion & context engine
│   ├── features.py                # 63-D acoustic & physics feature extractor
│   ├── privacy.py                 # Ephemeral RAM ring buffer (zero disk retention)
│   └── audit_chain.py             # SHA-256 cryptographic audit ledger
├── frontend/
│   ├── index.html                 # Complete responsive UI & Web Audio streamer
│   └── assets/                    # Styling, emblems, and visual assets
├── models/
│   ├── baseline_random_forest.pkl # Calibrated Random Forest model checkpoint
│   └── neural_sincnet.pt          # PyTorch SincNet weights
├── docs/
│   ├── PROJECT_DOCUMENTATION.md   # Master technical and mathematical specification
│   └── images/                    # UI walkthrough screenshots and ROC curves
├── test_samples/                  # Reference genuine & clone audio test samples
├── tests/                         # pytest automated unit test suite (5/5 passing)
├── .github/
│   ├── ISSUE_TEMPLATE/            # Bug report, feature request & task templates
│   └── pull_request_template.md   # Standardized PR checklist
├── Dockerfile & Procfile          # Container and cloud deployment definitions
├── render.yaml                    # 1-click cloud deployment blueprint for Render.com
├── requirements.txt               # Pinned Python dependencies
├── CONTRIBUTING.md                # Developer guidelines and issue tracking workflow
├── APPLICATION_DOCUMENTATION.md   # Comprehensive project & presentation deck brief
├── LICENSE                        # MIT Open Source License
└── app.py                         # Root entrypoint with port collision auto-resolver
```

---

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on reporting bugs, requesting features, and submitting pull requests.

We use **[GitHub Issues](https://github.com/aayash317-svg/Ai_voice-_cloning/issues)** for project tracking:
- **Bug Reports**: Use the `[BUG]` template for reproducible acoustic or UI anomalies.
- **Feature Requests**: Use the `[FEAT]` template to suggest new model architectures or integrations.
- **Project Tasks**: Use the `[TASK]` template for milestone and sprint deliverables.

---

## ⚖️ Ethical Use

Voice Shield AI is engineered strictly for **defensive biometric verification, fraud prevention, and voice integrity security**. It is designed to safeguard individuals, banks, and enterprises from malicious voice cloning extortion, CEO fraud, and unauthorized audio impersonation. Do not misuse this technology for non-consensual surveillance or malicious exploitation.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- **[ASVspoof Consortium](https://www.asvspoof.org/)**: For the ASVspoof 2019 LA and 2021 DF anti-spoofing benchmark datasets.
- **[LibriSpeech ASR Corpus](https://www.openslr.org/12/)**: For clean, diverse human voice reference speech.
- **[Mirco Ravanelli et al.](https://arxiv.org/abs/1808.00158)**: For the foundational research and mathematical formulation of SincNet raw-waveform bandpass convolutions.
- **PhonemeDF / ChatterboxTTS Team**: For contemporary neural TTS deepfake reference samples.
