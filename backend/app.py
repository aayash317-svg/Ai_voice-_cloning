"""
FastAPI & WebSocket Streaming Server for Voice Integrity Verification.
Provides real-time streaming audio analysis, batch verification,
and tamper-evident audit endpoints.
"""

import json
import io
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
import soundfile as sf
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from backend.config import SAMPLE_RATE, WINDOW_SAMPLES, MODELS_DIR, BASE_DIR
from backend.preprocess import AudioPreprocessor
from backend.features import FeatureExtractor
from backend.classifier import AntiSpoofClassifier
from backend.neural_classifier import SincNetClassifier, NEURAL_MODEL_PATH
from backend.risk_engine import RiskEngine
from backend.privacy import AudioPrivacyBuffer
from backend.audit_chain import AuditChain

FRONTEND_DIR = BASE_DIR / "frontend"

app = FastAPI(
    title="Voice Integrity Verification Framework",
    description="Real-Time AI Impersonation & Cloned Voice Detection Engine",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

@app.get("/")
def serve_dashboard():
    """Serve the interactive web application dashboard."""
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {"message": "Voice Integrity Verification API is online. Visit /docs for Swagger UI."}

# Global Framework State
preprocessor = AudioPreprocessor()
feature_extractor = FeatureExtractor()
risk_engine = RiskEngine()
audit_chain = AuditChain()
classifier: Optional[AntiSpoofClassifier] = None
neural_classifier: Optional[SincNetClassifier] = None


def get_or_load_classifier() -> Optional[AntiSpoofClassifier]:
    """Load latest available Random Forest model checkpoint (.pkl or .joblib)."""
    global classifier
    if classifier is not None:
        return classifier

    for model_path in list(MODELS_DIR.glob("**/*.pkl")) + list(MODELS_DIR.glob("**/*.joblib")):
        try:
            classifier = AntiSpoofClassifier.load(model_path)
            print(f"[+] Loaded model checkpoint from: {model_path}")
            return classifier
        except Exception as e:
            print(f"[!] Could not load {model_path}: {e}")
    return None


def get_or_load_neural_classifier() -> Optional[SincNetClassifier]:
    """Load latest SincNet neural raw-waveform model checkpoint (.pt)."""
    global neural_classifier
    if neural_classifier is not None:
        return neural_classifier

    if NEURAL_MODEL_PATH.exists():
        try:
            neural_classifier = SincNetClassifier.load(NEURAL_MODEL_PATH)
            print(f"[+] Loaded neural raw-waveform model from: {NEURAL_MODEL_PATH}")
            return neural_classifier
        except Exception as e:
            print(f"[!] Could not load neural model {NEURAL_MODEL_PATH}: {e}")
    return None


async def _decode_audio(audio_bytes: bytes, filename: str):
    """
    Decode audio bytes to numpy float32 array + sample rate.
    Tries 3 strategies in order:
      1. soundfile  — fast, handles WAV/FLAC/OGG/MP3/CAF
      2. ffmpeg     — handles M4A/AAC/MP4/AIFF and almost everything else
      3. afconvert  — macOS built-in fallback (no extra install needed)
    """
    import tempfile, os, subprocess

    # --- Strategy 1: soundfile (in-memory, fastest) ---
    try:
        data, sr = sf.read(io.BytesIO(audio_bytes))
        return data, sr
    except Exception:
        pass

    suffix = Path(filename).suffix.lower() or ".tmp"

    # Write raw bytes to a temp file for external tools
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_in:
        tmp_in.write(audio_bytes)
        tmp_in_path = tmp_in.name

    tmp_wav_path = tmp_in_path + "_converted.wav"

    try:
        # --- Strategy 2: ffmpeg (handles M4A, AAC, MP4, AIFF, etc.) ---
        ffmpeg_bin = "/opt/homebrew/bin/ffmpeg"
        if not os.path.exists(ffmpeg_bin):
            ffmpeg_bin = "ffmpeg"  # system PATH fallback
        try:
            result = subprocess.run(
                [ffmpeg_bin, "-y", "-i", tmp_in_path, "-ar", "16000", "-ac", "1", "-f", "wav", tmp_wav_path],
                capture_output=True, timeout=30
            )
            if result.returncode == 0:
                data, sr = sf.read(tmp_wav_path)
                return data, sr
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        # --- Strategy 3: macOS afconvert (built-in, no install needed) ---
        try:
            result = subprocess.run(
                ["afconvert", "-f", "WAVE", "-d", "LEF32@16000", tmp_in_path, tmp_wav_path],
                capture_output=True, timeout=30
            )
            if result.returncode == 0:
                data, sr = sf.read(tmp_wav_path)
                return data, sr
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format '{suffix}'. Please use WAV, FLAC, MP3, OGG, M4A, or AAC."
        )
    finally:
        for p in [tmp_in_path, tmp_wav_path]:
            try:
                os.unlink(p)
            except Exception:
                pass




@app.get("/health")
def health_check():
    """System health check and active model status."""
    active_clf = get_or_load_classifier()
    active_neural = get_or_load_neural_classifier()
    return {
        "status": "online",
        "random_forest_loaded": active_clf is not None,
        "neural_sincnet_loaded": active_neural is not None,
        "ensemble_active": active_clf is not None and active_neural is not None,
        "model_type": "Dual-Ensemble (RandomForest + SincNet Raw Audio)",
        "features_dimension": len(active_clf.feature_names) if active_clf else 0,
        "audit_chain_blocks": len(audit_chain.chain)
    }


@app.post("/analyze")
async def analyze_audio_file(
    file: UploadFile = File(...),
    is_unknown_number: bool = Query(False),
    is_high_value_transaction: bool = Query(False),
    prior_fraud_flag: bool = Query(False)
):
    """
    Analyze uploaded audio file, extract 4-family feature vectors + raw audio,
    compute dual ensemble impersonation risk score, and log security audit event.
    """
    audio_bytes = await file.read()
    try:
        data, sr = await _decode_audio(audio_bytes, file.filename or "audio.tmp")
        audio = preprocessor.load_audio(data, sr=sr)
        audio = preprocessor.normalize_audio(audio)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Audio decoding error: {e}")

    # Extract features
    features = feature_extractor.extract_all(audio)
    active_clf = get_or_load_classifier()
    active_neural = get_or_load_neural_classifier()

    rf_spoof_prob: Optional[float] = None
    if active_clf:
        vec = feature_extractor.to_vector(features, active_clf.feature_names)
        rf_spoof_prob = float(active_clf.predict_spoof_risk(vec))

    neural_spoof_prob: Optional[float] = None
    if active_neural:
        try:
            neural_spoof_prob = float(active_neural.predict_spoof_prob(audio))
        except Exception as e:
            print(f"[!] Neural prediction error: {e}")

    # Dual Ensemble Fusion
    if rf_spoof_prob is not None and neural_spoof_prob is not None:
        spoof_prob = 0.50 * rf_spoof_prob + 0.50 * neural_spoof_prob
    elif rf_spoof_prob is not None:
        spoof_prob = rf_spoof_prob
    elif neural_spoof_prob is not None:
        spoof_prob = neural_spoof_prob
    else:
        spoof_prob = float(np.clip(features.get("spectral_flatness", 0.0) * 1.5, 0.05, 0.95))

    context_meta = {
        "is_unknown_number": is_unknown_number,
        "is_high_value_transaction": is_high_value_transaction,
        "prior_fraud_flag": prior_fraud_flag
    }

    assessment = risk_engine.compute_risk(
        classifier_spoof_prob=spoof_prob,
        acoustic_anomaly_score=features.get("acoustic_phase_derivative_var", 0.0),
        spectral_anomaly_score=features.get("spectral_hf_energy_ratio", 0.0),
        prosody_anomaly_score=features.get("prosody_jitter", 0.0),
        context_metadata=context_meta
    )

    # Append non-biometric security event to audit chain
    audit_chain.append_event("BATCH_ANALYSIS", {
        "filename": file.filename,
        "risk_score": assessment.risk_score,
        "risk_level": assessment.risk_level,
        "alert_triggered": assessment.alert_triggered
    })

    return {
        "filename": file.filename,
        "assessment": assessment,
        "model_breakdown": {
            "random_forest_prob": round(rf_spoof_prob, 4) if rf_spoof_prob is not None else None,
            "neural_sincnet_prob": round(neural_spoof_prob, 4) if neural_spoof_prob is not None else None,
            "ensemble_spoof_prob": round(spoof_prob, 4)
        },
        "features_extracted": {k: round(v, 4) for k, v in features.items()}
    }


@app.websocket("/stream")
async def websocket_stream_endpoint(websocket: WebSocket):
    """
    Real-time streaming WebSocket endpoint.
    Receives continuous PCM 16-bit / float audio chunks, analyzes sliding windows,
    returns continuous dynamic risk scores, and discards audio buffers immediately.
    """
    await websocket.accept()
    buffer = AudioPrivacyBuffer()
    active_clf = get_or_load_classifier()
    active_neural = get_or_load_neural_classifier()

    try:
        while True:
            # Receive raw binary PCM or JSON audio data
            data = await websocket.receive_bytes()
            chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            buffer.append_chunk(chunk)

            # Analyze sliding window (3.0 seconds = 48,000 samples at 16kHz)
            window = buffer.get_window(WINDOW_SAMPLES)
            if window is not None:
                norm_window = preprocessor.normalize_audio(window)
                features = feature_extractor.extract_all(norm_window)

                rf_prob = None
                if active_clf:
                    vec = feature_extractor.to_vector(features, active_clf.feature_names)
                    rf_prob = float(active_clf.predict_spoof_risk(vec))

                neural_prob = None
                if active_neural:
                    try:
                        neural_prob = float(active_neural.predict_spoof_prob(norm_window))
                    except Exception:
                        pass

                if rf_prob is not None and neural_prob is not None:
                    spoof_prob = 0.50 * rf_prob + 0.50 * neural_prob
                elif rf_prob is not None:
                    spoof_prob = rf_prob
                elif neural_prob is not None:
                    spoof_prob = neural_prob
                else:
                    spoof_prob = float(np.clip(features.get("spectral_flatness", 0.0) * 1.5, 0.05, 0.95))

                assessment = risk_engine.compute_risk(
                    classifier_spoof_prob=spoof_prob,
                    acoustic_anomaly_score=features.get("acoustic_phase_derivative_var", 0.0),
                    spectral_anomaly_score=features.get("spectral_hf_energy_ratio", 0.0),
                    prosody_anomaly_score=features.get("prosody_jitter", 0.0)
                )

                if assessment.alert_triggered:
                    audit_chain.append_event("STREAM_ALERT", {
                        "risk_score": assessment.risk_score,
                        "risk_level": assessment.risk_level
                    })

                await websocket.send_json({
                    "risk_score": assessment.risk_score,
                    "risk_level": assessment.risk_level,
                    "alert": assessment.alert_triggered,
                    "signal_breakdown": assessment.signal_breakdown,
                    "safety_override": assessment.safety_override_triggered
                })

    except WebSocketDisconnect:
        pass
    finally:
        buffer.purge()


@app.get("/audit/chain")
def get_audit_chain():
    """Retrieve immutable audit chain entries."""
    return {"chain": [vars(b) for b in audit_chain.chain], "length": len(audit_chain.chain)}


@app.get("/audit/verify")
def verify_audit_integrity():
    """Cryptographically verify the integrity of the audit ledger."""
    is_valid, error = audit_chain.verify_integrity()
    return {
        "valid": is_valid,
        "status": "SECURE_AND_INTACT" if is_valid else "TAMPERING_DETECTED",
        "error_detail": error,
        "blocks_verified": len(audit_chain.chain)
    }
