# Voice Shield AI — Complete Jury & PPT Presentation Diagrams Portfolio

> **Project Title**: Voice Shield AI — Real-Time Conversational AI Voice Cloning Detection & Call Defense Engine  
> **Target Audience**: Hackathon / Academic / Enterprise Jury, PPT Slides, and Technical Documentation  
> **Format**: High-Resolution GitHub-Flavored & PowerPoint-Ready Mermaid Diagrams  

---

## 1. Problem → Solution Diagram
*Immediate high-level comparison showing the threat landscape and Voice Shield AI's solution.*

```mermaid
flowchart TD
    subgraph PROBLEM[" THE GROWING PROBLEM: GENERATIVE AI VOICE FRAUD "]
        direction TB
        P1["🚨 Hyper-Realistic AI Clones<br/>(ElevenLabs, Chatterbox, VALL-E)"]
        P2["📞 Live Vishing & Social Engineering<br/>(Urgent CEO Fraud, Fake Family Emergencies)"]
        P3["💰 Banking & Wire Transfer Theft<br/>(Bypassing Voice Biometrics & Verbal Authorizations)"]
        P4["⚠️ Existing Limitations<br/>• Human ear deception rate > 80%<br/>• Cloud APIs introduce fatal latency (> 2s)<br/>• Violates privacy by uploading calls to 3rd parties"]
        P1 --> P2 --> P3 --> P4
    end

    subgraph GAP[" CRITICAL DEFENSE GAP "]
        direction TB
        G1["❌ Traditional Voice Auth:<br/>Only verifies 'Who is speaking'<br/>Blind to whether voice is synthetic"]
        G2["❌ Batch Audio Checkers:<br/>Require complete audio file after call ends<br/>Cannot stop fraud in-flight"]
    end

    subgraph SOLUTION[" THE SOLUTION: VOICE SHIELD AI "]
        direction TB
        S1["🛡️ Sub-Second Real-Time Call Defense<br/>250ms latency via streaming WebSockets"]
        S2["👥 Real-Time Conversational Diarization<br/>Separates User from Remote Caller automatically"]
        S3["🔬 Dual Forensic ML Ensemble<br/>Calibrated Random Forest + Raw-Waveform SincNet"]
        S4["🔒 Zero-Retention RAM Architecture<br/>No call audio written to disk (GDPR / HIPAA compliant)"]
        S5["📜 Cryptographic SHA-256 Audit Trail<br/>Tamper-evident legal evidence ledger"]
        S1 --> S2 --> S3 --> S4 --> S5
    end

    PROBLEM ==> GAP ==> SOLUTION

    style PROBLEM fill:#ffebee,stroke:#c62828,stroke-width:2px,color:#b71c1c
    style GAP fill:#fff3e0,stroke:#e65100,stroke-width:2px,color:#bf360c
    style SOLUTION fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,color:#1b5e20
```

---

## 2. Technical Architecture Diagram
*Detailed component breakdown across the 6 system layers.*

```mermaid
graph TB
    subgraph LAYER1[" 1. CLIENT & INGESTION LAYER "]
        MIC["🎙️ Microphone / VoIP Call Stream"]
        WA["🌐 Web Audio API (16,000 Hz, 16-bit Mono PCM)"]
        KA["⚡ Inaudible Keep-Alive Loop (gain=0.00001)"]
        MIC --> WA --> KA
    end

    subgraph LAYER2[" 2. STREAMING & PRIVACY BUFFER LAYER "]
        WS["🔌 FastAPI WebSocket (/stream/call)"]
        RING["🧠 AudioPrivacyBuffer (Circular RAM Ring - 3.0s)"]
        SHRED["🧹 Memory Auto-Purge (Zero Disk Retention)"]
        KA -->|Binary Chunks 250ms| WS --> RING --> SHRED
    end

    subgraph LAYER3[" 3. CONVERSATIONAL DIARIZATION ENGINE "]
        VAD["Detect Speech / Silence (VAD)"]
        TIMBRE["40-D Timbre Embedding (MFCC + Spectral)"]
        CLUSTER["Adaptive Centroid Clustering (Cosine Distance)"]
        SPK_A["Speaker A (Local User) -> Whitelisted Baseline"]
        SPK_B["Speaker B (Remote Caller) -> Isolated for ML"]
        OVERLAP["Overlapping Cross-Talk -> Quarantined"]
        
        RING --> VAD --> TIMBRE --> CLUSTER
        CLUSTER --> SPK_A
        CLUSTER --> SPK_B
        CLUSTER --> OVERLAP
    end

    subgraph LAYER4[" 4. DUAL FORENSIC ML DETECTION ENSEMBLE "]
        direction TB
        subgraph BRANCH_A[" Acoustic Physics Engine "]
            FEAT["63 Physics-Informed Descriptors<br/>• Phase Derivative Variance (Hilbert)<br/>• High-Frequency Cutoff (>4kHz)<br/>• Glottal Micro-Jitter & Shimmer<br/>• 20 MFCCs & Spectral Contrast"]
            RF["🌲 Calibrated Random Forest<br/>(100 Trees | tau* = 0.3985)"]
            FEAT --> RF
        end

        subgraph BRANCH_B[" Deep Raw-Waveform Engine "]
            RAW["Raw 1D Audio Tensor (48k samples)"]
            SINC["🌊 SincNet Neural Waveform Model<br/>(Learnable Bandpass Sinc Filters)"]
            RAW --> SINC
        end

        FUSION["⚖️ Consensus Late Fusion<br/>P_ensemble = 0.50 * P_RF + 0.50 * P_Neural"]
        RF --> FUSION
        SINC --> FUSION
    end

    SPK_B -->|3.0s Isolated Window| FEAT
    SPK_B -->|3.0s Raw Waveform| RAW

    subgraph LAYER5[" 5. PROGRESSIVE RISK ENGINE "]
        FORMULA["Dynamic Multi-Signal Risk Score<br/>50% ML + 20% HF + 15% Phase + 10% Jitter + 5% Cons"]
        SENS["⚠️ Context Multiplier (+12% Wire/Banking Flags)"]
        STAGES["3-Stage Timeline (Stage 0: 0-5s | Stage 1: 5s | Stage 2: 10s Final)"]
        FUSION --> FORMULA --> SENS --> STAGES
    end

    subgraph LAYER6[" 6. TELEMETRY & CRYPTOGRAPHIC AUDIT LAYER "]
        AUDIT["🔗 SHA-256 Tamper-Evident AuditChain"]
        UI_GAUGE["🎯 Live Threat Gauge (0% - 100%)"]
        UI_BANNER["🚨 Dynamic Call Defense Banner"]
        UI_OSC["📈 60 FPS Real-Time Oscilloscope"]
        
        STAGES --> AUDIT
        STAGES -->|JSON Telemetry| UI_GAUGE
        STAGES -->|JSON Telemetry| UI_BANNER
        STAGES -->|JSON Telemetry| UI_OSC
    end

    style LAYER1 fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    style LAYER2 fill:#e0f7fa,stroke:#00838f,stroke-width:2px
    style LAYER3 fill:#fff8e1,stroke:#f57f17,stroke-width:2px
    style LAYER4 fill:#e8eaf6,stroke:#283593,stroke-width:2px
    style LAYER5 fill:#fbe9e7,stroke:#d84315,stroke-width:2px
    style LAYER6 fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
```

---

## 3. End-to-End Workflow Diagram
*Sequential path of audio data from capture to final defensive verdict.*

```mermaid
sequenceDiagram
    autonumber
    actor Caller as 👤 Remote Caller / Clone
    actor User as 🛡️ Local User (Victim)
    participant Client as 💻 Web Client (WebAudio)
    participant WS as ⚡ WebSocket Server (FastAPI)
    participant RAM as 🧠 Ephemeral RAM Buffer
    participant Diar as 👥 Diarization & VAD
    participant ML as 🔬 Dual Ensemble (RF + SincNet)
    participant Engine as ⚙️ Progressive Risk Engine
    participant Audit as 🔒 SHA-256 Audit Ledger
    participant UI as 🖥️ Live UI Dashboard

    User->>Client: Speaks into microphone
    Caller->>Client: Incoming caller voice audio
    Client->>WS: Stream 250ms binary PCM chunks (16kHz)
    WS->>RAM: Push into 3.0s Circular RAM Ring
    RAM->>Diar: Extract 3.0s window for speaker clustering
    
    alt Overlap / Cross-talk
        Diar-->>RAM: Quarantine mixed audio (prevents false alerts)
    else Local User Voice
        Diar-->>UI: Update User Card (Baseline Authentic)
    else Remote Caller Voice
        Diar->>ML: Send isolated remote speaker segment
        par Feature Extraction & RF
            ML->>ML: Calculate 63 acoustic features (Hilbert, Jitter, HF)
            ML->>ML: RF inference -> P_RF
        and Raw Waveform & SincNet
            ML->>ML: Pass raw waveform -> SincNet filters
            ML->>ML: Neural inference -> P_Neural
        end
        ML->>Engine: Late Fusion P_ensemble (50/50)
        Engine->>Engine: Evaluate 3-Stage timeline & context flags
        Engine->>Audit: Seal verdict into SHA-256 AuditChain
        Engine->>UI: Broadcast JSON telemetry
        
        opt Threat Score > 50%
            UI->>User: 🚨 TRIGGER RED BANNER: "CRITICAL CLONE DETECTED - FREEZE PAYMENTS"
        end
    end
    RAM->>RAM: Memory auto-purged (Zero disk footprint)
```

---

## 4. Real-Time Call / Two-Speaker Workflow
*How Voice Shield AI solves the hardest real-world problem: isolating the attacker from the victim.*

```mermaid
flowchart TD
    IN["🎙️ Continuous Incoming Call Stream (Mixed Audio)"] --> VAD{"Voice Activity<br/>Detection (VAD)"}
    
    VAD -- Silence / Background Noise --> DROP["🗑️ Discard (No Processing)"]
    VAD -- Active Speech Frame --> EMBED["📐 Extract 40-D Timbre Representation<br/>• 19 MFCC Means<br/>• 19 MFCC Standard Deviations<br/>• Spectral Centroid<br/>• Spectral Rolloff"]
    
    EMBED --> CLUST{"Adaptive Centroid Clustering<br/>(Cosine Distance Threshold)"}
    
    CLUST -- Matches Local Profile --> SPK1["👤 SPEAKER A: Local User<br/>• Biomechanical vocal tract established<br/>• Tagged as verified genuine human<br/>• UI: User Turn Counter ++"]
    
    CLUST -- Matches Remote Caller --> SPK2["📞 SPEAKER B: Remote Party<br/>• Unknown acoustic profile<br/>• Isolated into 3.0s forensic window<br/>• UI: Remote Turn Counter ++"]
    
    CLUST -- Ambiguous / Simultaneous --> SPK_MIX["⚠️ Cross-Talk / Overlap<br/>• Quarantined from classification<br/>• Prevents false positive contamination"]
    
    SPK1 --> SAFE["✅ No Alert Triggered for User"]
    SPK_MIX --> HOLD["⏳ Await Disentangled Speech"]
    SPK2 ==> FORENSIC["🔬 Streamed Directly to Dual AI Forensic Ensemble"]

    style IN fill:#e1f5fe,stroke:#0288d1,stroke-width:2px
    style VAD fill:#fff9c4,stroke:#fbc02d,stroke-width:2px
    style CLUST fill:#ffe0b2,stroke:#f57c00,stroke-width:2px
    style SPK1 fill:#e8f5e9,stroke:#388e3c,stroke-width:2px
    style SPK2 fill:#ffebee,stroke:#d32f2f,stroke-width:3px
    style FORENSIC fill:#ede7f6,stroke:#512da8,stroke-width:3px
```

---

## 5. AI Detection Pipeline (RF + SincNet + Ensemble)
*The dual-branch forensic machine learning architecture.*

```mermaid
flowchart TD
    INPUT["🎵 3.0s Isolated Audio Stream (48,000 Samples @ 16 kHz)"]
    
    INPUT --> BRANCH1["🔬 BRANCH 1: Acoustic Physics Descriptors"]
    INPUT --> BRANCH2["🌊 BRANCH 2: Raw Waveform Time-Domain"]

    subgraph B1[" Physics-Informed Feature Engine "]
        direction TB
        F1["1. Instantaneous Phase Derivative Variance<br/>(Hilbert Transform unwrap - vocoder phase glitch)"]
        F2["2. High-Frequency Energy Ratio<br/>(Energy above 4 kHz cutoff / vocoder Mel-shelf)"]
        F3["3. Glottal Micro-Perturbations<br/>(Vocal cord local Jitter & Shimmer)"]
        F4["4. Spectral Formants & Contrast<br/>(20 MFCCs mean/std across 1024-pt FFT)"]
        RF_MODEL["🌲 Calibrated Random Forest<br/>100 Decision Trees | max_depth=16<br/>Trained on balanced corpora"]
        F1 & F2 & F3 & F4 --> RF_MODEL
        RF_MODEL --> P_RF["P_RF (Probability Spoof)"]
    end

    subgraph B2[" Deep SincNet Waveform Engine "]
        direction TB
        S1["Learnable Parameterized Sinc Bandpass Filters<br/>g[t, f1, f2] = 2f2*sinc(2πf2 t) - 2f1*sinc(2πf1 t)"]
        S2["1D Waveform Convolutions + LayerNorm + LeakyReLU"]
        S3["Dense Classification Layers + Dropout"]
        SINC_MODEL["🧠 SincNet Waveform Neural Network<br/>Direct time-domain sample analysis"]
        S1 --> S2 --> S3 --> SINC_MODEL
        SINC_MODEL --> P_SINC["P_Neural (Probability Spoof)"]
    end

    BRANCH1 --> B1
    BRANCH2 --> B2

    P_RF & P_SINC --> FUSION["⚖️ CONSENSUS LATE FUSION<br/>P_ensemble = 0.50 * P_RF + 0.50 * P_Neural"]

    FUSION --> CALIB{"Threshold Decision<br/>Calibrated tau* = 0.3985"}
    
    CALIB -- P_ensemble < 0.3985 --> RES_REAL["✅ GENUINE HUMAN VOICE<br/>Natural glottal jitter & continuous phase"]
    CALIB -- P_ensemble >= 0.3985 --> RES_FAKE["🚨 SYNTHETIC AI VOICE CLONE<br/>Vocoder artifacts & synthetic phase detected"]

    style INPUT fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    style B1 fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2px
    style B2 fill:#e0f2f1,stroke:#00695c,stroke-width:2px
    style FUSION fill:#fff3e0,stroke:#e65100,stroke-width:2px
    style RES_REAL fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    style RES_FAKE fill:#ffebee,stroke:#c62828,stroke-width:2px
```

---

## 6. Security & Privacy Architecture
*Proving zero data leakage, GDPR/HIPAA compliance, and cryptographic auditability.*

```mermaid
flowchart LR
    subgraph CLIENT[" 1. CLIENT PRIVACY "]
        C1["Web Audio Capture"]
        C2["No Local Recording File"]
        C1 --> C2
    end

    subgraph TRANS[" 2. ENCRYPTED TRANSPORT "]
        T1["TLS 1.3 / WSS WebSocket"]
        T2["Binary PCM Payload"]
        T1 --> T2
    end

    subgraph MEMORY[" 3. ZERO-RETENTION VOLATILE RAM "]
        M1["Circular AudioPrivacyBuffer (48,000 float32)"]
        M2["Extraction into Feature Vector"]
        M3["buffer.purge() -> buffer.fill(0.0)"]
        M4["ZERO DISK WRITES<br/>(GDPR Art 9 & HIPAA Compliant)"]
        M1 --> M2 --> M3 --> M4
    end

    subgraph VAULT[" 4. ENCRYPTED EMBEDDINGS "]
        V1["Voiceprint Profiles"]
        V2["Fernet AES-128-CBC + HMAC-SHA256"]
        V1 --> V2
    end

    subgraph AUDIT[" 5. IMMUTABLE AUDIT LEDGER "]
        A1["Event Payload: Timestamp + Verdict + Threat Score"]
        A2["SHA-256 Hash Chain:<br/>Hash_n = SHA-256(Index || Time || Payload || Hash_n-1)"]
        A3["Tamper-Evident Verification Endpoint (/audit/verify)"]
        A1 --> A2 --> A3
    end

    CLIENT --> TRANS --> MEMORY
    MEMORY -.-> VAULT
    MEMORY --> AUDIT

    style CLIENT fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    style TRANS fill:#e8eaf6,stroke:#3f51b5,stroke-width:2px
    style MEMORY fill:#e0f2f1,stroke:#00796b,stroke-width:2px
    style VAULT fill:#fff3e0,stroke:#e65100,stroke-width:2px
    style AUDIT fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
```

---

## 7. Dataset & Anti-Leakage Training Pipeline
*How the system was trained across 38,239 samples with zero speaker overlap.*

```mermaid
flowchart TD
    subgraph DATASETS[" 38,239 MULTI-CORPUS AUDIO SAMPLES "]
        D1["LibriSpeech test-clean<br/>(2,620 Genuine Human)"]
        D2["ASVspoof 2019 LA Bonafide<br/>(2,680 Genuine Human)"]
        D3["ASVspoof 2019 LA Spoof<br/>(15,399 A07-A19 Algorithms)"]
        D4["PhonemeDF ChatterboxTTS<br/>(17,540 Modern Neural TTS)"]
    end

    subgraph STAGE1[" STEP 1: Strict Anti-Leakage Partitioning "]
        SPLIT["Disjoint Speaker ID Splitting<br/>Train: 70% (26,767) | Dev: 15% (5,736) | Test: 15% (5,736)<br/>⚠️ Zero Speaker Identity Overlap"]
    end

    subgraph STAGE2[" STEP 2: Controlled 1:1 Class Balancing "]
        BAL["Exact 1:1 Balance (3,710 Real vs 3,710 Fake)<br/>• 3,710 Human (LibriSpeech + ASVspoof)<br/>• 1,855 ASVspoof + 1,855 ChatterboxTTS<br/>Eliminates synthetic majority bias"]
    end

    subgraph STAGE3[" STEP 3: Parallel Ingestion & Training "]
        direction LR
        FEAT_GEN["63-D Feature Extraction"] --> RF_TRAIN["Random Forest Training<br/>100 Trees, max_depth=16"]
        RAW_GEN["3.0s Raw Waveforms"] --> SINC_TRAIN["SincNet Waveform Training<br/>AdamW, Cosine Annealing, MPS/CUDA"]
    end

    subgraph STAGE4[" STEP 4: Dev-Set Threshold Calibration "]
        CALIB["Equal Error Rate (EER) Calibration on DEV Set ONLY<br/>Locked Calibrated Threshold: tau* = 0.3985"]
    end

    subgraph STAGE5[" STEP 5: Held-Out Test Evaluation "]
        EVAL["Final Test on 3,405 Held-Out Files<br/>Zero threshold tuning | Unbiased Benchmark"]
    end

    DATASETS --> STAGE1 --> STAGE2 --> STAGE3 --> STAGE4 --> STAGE5

    style DATASETS fill:#e1f5fe,stroke:#0288d1,stroke-width:2px
    style STAGE1 fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    style STAGE2 fill:#fff8e1,stroke:#fbc02d,stroke-width:2px
    style STAGE3 fill:#ede7f6,stroke:#512da8,stroke-width:2px
    style STAGE4 fill:#fbe9e7,stroke:#d84315,stroke-width:2px
    style STAGE5 fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
```

---

## 8. Progressive Risk Engine
*How multiple windows evolve through 3 progressive stages into an infallible verdict.*

```mermaid
flowchart TD
    STREAM["Continuous 250ms Audio Inflow"] --> WIN["3.0s Ephemeral Analysis Window"]
    
    subgraph SIGNALS[" Multi-Signal Formula Engine "]
        direction TB
        S1["ML Ensemble Prob (50%)"]
        S2["HF Energy Ratio (20%)"]
        S3["Phase Deriv Var (15%)"]
        S4["Jitter/Shimmer (10%)"]
        S5["Window Consistency (5%)"]
        CALC["Composite Raw Threat Score (0 - 100)"]
        S1 & S2 & S3 & S4 & S5 --> CALC
    end
    WIN --> SIGNALS

    subgraph CONTEXT[" Contextual Sensitivity Modifiers "]
        C_FLAG["High-Value Transaction / Banking Keyword Detected?<br/>('wire transfer', 'OTP', 'send money')"]
        C_FLAG -- Yes --> BOOST["Apply Sensitivity Boost (+12% Threat)"]
        C_FLAG -- No --> NO_BOOST["Standard Weighting"]
    end
    CALC --> CONTEXT

    subgraph TIMELINE[" 3-Stage Progressive Timeline Evaluation "]
        direction TB
        T0["STAGE 0: Calibration (0.0s - 5.0s)<br/>• Accumulating acoustic baseline<br/>• UI Displays Calibrating Countdown (Xs / 10s)"]
        T1["STAGE 1: Preliminary Indicator (5.0s - 6.0s)<br/>• Early rolling 5-second average<br/>• Early advisory alert if threat > 80%"]
        T2["STAGE 2: Consolidated Final Verdict (10.0s - 12.0s)<br/>• Minimum 20 evaluated sliding windows<br/>• Guaranteed verified final verdict"]
        T0 --> T1 --> T2
    end
    BOOST & NO_BOOST --> TIMELINE

    subgraph VERDICT[" Final Decision Tiers "]
        direction LR
        V_GREEN["🟢 0% - 25%<br/>AUTHENTIC HUMAN<br/>(Safe to proceed)"]
        V_YELLOW["🟡 26% - 50%<br/>SUSPICIOUS / INCONCLUSIVE<br/>(Monitor call)"]
        V_ORANGE["🟠 51% - 75%<br/>HIGH RISK SPOOF<br/>(Step-up verification)"]
        V_RED["🔴 76% - 100%<br/>CRITICAL AI CLONE<br/>(TERMINATE / FREEZE)"]
    end
    T2 --> VERDICT

    style STREAM fill:#e1f5fe,stroke:#0288d1,stroke-width:2px
    style SIGNALS fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    style CONTEXT fill:#fff9c4,stroke:#fbc02d,stroke-width:2px
    style TIMELINE fill:#ede7f6,stroke:#512da8,stroke-width:2px
    style V_GREEN fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    style V_YELLOW fill:#fffde7,stroke:#fbc02d,stroke-width:2px
    style V_ORANGE fill:#fff3e0,stroke:#e65100,stroke-width:2px
    style V_RED fill:#ffebee,stroke:#c62828,stroke-width:2px
```

---

## 9. Results / Benchmark Visualization
*Clear side-by-side performance proving the superiority of the dual ensemble.*

```mermaid
flowchart TD
    subgraph BENCHMARK[" EMPIRICAL TEST BENCHMARKS (3,405 Held-Out Files) "]
        direction TB
        
        subgraph MODELS[" Comparative Detection Models "]
            direction LR
            M_RF["🌲 Random Forest (63-D Physics)<br/>• ROC-AUC: 0.9827<br/>• Accuracy: 92.13%<br/>• Precision: 98.19%<br/>• EER: 7.05%<br/>• False Alarm: 5.53%"]
            M_SINC["🌊 SincNet (Raw Waveform)<br/>• ROC-AUC: 0.9518<br/>• Accuracy: 86.73%<br/>• Precision: 94.98%<br/>• EER: 12.56%<br/>• False Alarm: 13.33%"]
            M_ENS["⭐ DUAL ENSEMBLE (Consensus)<br/>• ROC-AUC: 0.9882 (HIGHEST)<br/>• Accuracy: 93.45% (BEST)<br/>• Precision: 98.50% (LOWEST ERROR)<br/>• EER: 6.12% (LOWEST)<br/>• False Alarm: 4.21% (LOWEST)"]
        end

        subgraph SAMPLES[" Empirical Real-World Sample Verifications "]
            direction LR
            S_USER["User Real Voice (m4a)<br/>Risk: 12.1% -> PASSED ✅"]
            S_LIBRI["Clean Human (wav)<br/>Risk: 12.4% -> PASSED ✅"]
            S_ASV["ASVspoof Synthetic<br/>Risk: 43.7% -> FLAGGED ⚠️"]
            S_CHAT["ChatterboxTTS Clone<br/>Risk: 78.4% -> CRITICAL ALERT 🚨"]
        end
        
        MODELS --> SAMPLES
    end

    style BENCHMARK fill:#ffffff,stroke:#1a237e,stroke-width:2px
    style M_RF fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    style M_SINC fill:#e0f2f1,stroke:#00695c,stroke-width:2px
    style M_ENS fill:#e8f5e9,stroke:#1b5e20,stroke-width:3px
    style S_USER fill:#f1f8e9,stroke:#33691e,stroke-width:2px
    style S_LIBRI fill:#f1f8e9,stroke:#33691e,stroke-width:2px
    style S_ASV fill:#fffde7,stroke:#f57f17,stroke-width:2px
    style S_CHAT fill:#ffebee,stroke:#b71c1c,stroke-width:2px
```

---

## 10. Deployment Architecture
*Production cloud, on-premises, and live web app deployment topology.*

```mermaid
graph TB
    subgraph CLIENT_EDGE[" CLIENT / EDGE DEVICES "]
        BROWSER["💻 Browser Client (Chrome / Safari / Edge)"]
        MOBILE["📱 Mobile Phone (iOS / Android WebRTC)"]
        SOFTPHONE["🎧 Call Center Softphone / VoIP PBX"]
    end

    subgraph NETWORK[" SECURE EDGE & INGRESS LAYER "]
        CF["☁️ Cloudflare CDN / SSL Termination"]
        LB["⚖️ Load Balancer & Reverse Proxy (Render / Nginx)"]
        WSS_ROUTE["⚡ WSS Routing (/stream/call)"]
        HTTP_ROUTE["🌐 HTTP/REST Routing (/api, /audit)"]
        CF --> LB
        LB --> WSS_ROUTE
        LB --> HTTP_ROUTE
    end

    CLIENT_EDGE -->|TLS 1.3 / WSS| CF

    subgraph SERVER[" FASTAPI CONTAINERIZED ENGINE (Docker) "]
        APP["🚀 FastAPI Application Instance (Uvicorn ASGI)"]
        WS_HANDLER["WebSocket Streaming Ingest Controller"]
        MEM_RING["🧠 Ephemeral RAM Ring (No Disk Storage)"]
        DIAR_PROC["👥 Fast Conversational Diarizer"]
        ML_WORKER["🔬 PyTorch / Scikit-Learn Inference Engine<br/>(Apple MPS / NVIDIA CUDA / Optimized CPU)"]
        AUDIT_CHAIN["🔒 In-Memory SHA-256 Audit Chain"]
        
        WSS_ROUTE --> WS_HANDLER --> MEM_RING --> DIAR_PROC --> ML_WORKER
        HTTP_ROUTE --> APP
        ML_WORKER --> AUDIT_CHAIN
    end

    subgraph TELEMETRY[" REAL-TIME CLIENT FEEDBACK "]
        AUDIT_CHAIN -->|Telemetry Events| WS_HANDLER
        WS_HANDLER -->|JSON Telemetry Frame| BROWSER
    end

    style CLIENT_EDGE fill:#e8eaf6,stroke:#283593,stroke-width:2px
    style NETWORK fill:#e0f7fa,stroke:#006064,stroke-width:2px
    style SERVER fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    style TELEMETRY fill:#e8f5e9,stroke:#1b5e20,stroke-width:2px
```

---

## 11. Use-Case Diagram
*Enterprise, banking, and executive defense interaction matrix.*

```mermaid
flowchart LR
    subgraph ACTORS[" ACTORS "]
        BANK["🏦 Banking Customer"]
        AGENT["🎧 Call Center Agent"]
        FRAUD["🦹 Remote Fraudster / Clone Bot"]
        EXEC["👔 Corporate Executive / CFO"]
        AUDITOR["⚖️ Compliance Auditor"]
    end

    subgraph SYSTEM[" VOICE SHIELD AI DEFENSE PLATFORM "]
        direction TB
        UC1(["UC-1: Live Call Screening & Real-Time Alerting"])
        UC2(["UC-2: Two-Speaker Conversation Diarization"])
        UC3(["UC-3: High-Value Wire Transfer Authorization Guard"])
        UC4(["UC-4: CEO Fraud & Impersonation Defense"])
        UC5(["UC-5: In-Memory Volatile Zero-Retention Processing"])
        UC6(["UC-6: Cryptographic SHA-256 Audit Trail Export"])
        UC7(["UC-7: Live Threat Gauge & Forensic Waveform Visualizer"])
    end

    FRAUD -.->|Attempts clone attack via phone| UC1
    BANK --- UC1
    AGENT --- UC1
    AGENT --- UC2
    AGENT --- UC7
    EXEC --- UC4
    EXEC --- UC3
    BANK --- UC3
    AUDITOR --- UC6
    SYSTEM -.-> UC5

    style ACTORS fill:#f5f5f5,stroke:#616161,stroke-width:2px
    style SYSTEM fill:#e8f5e9,stroke:#2e7d32,stroke-width:3px
    style FRAUD fill:#ffebee,stroke:#b71c1c,stroke-width:2px
    style UC1 fill:#fff9c4,stroke:#fbc02d,stroke-width:2px
    style UC3 fill:#ffe0b2,stroke:#f57c00,stroke-width:2px
    style UC4 fill:#ffcdd2,stroke:#d32f2f,stroke-width:2px
    style UC6 fill:#e1f5fe,stroke:#0288d1,stroke-width:2px
```
