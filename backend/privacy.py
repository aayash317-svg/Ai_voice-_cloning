"""
Privacy & Security Engine for Voice Integrity Verification.
Implements zero-retention in-memory rolling audio buffers and
Fernet-encrypted speaker voiceprint storage.
"""

import os
import io
import time
from typing import Optional, Tuple
import numpy as np
from cryptography.fernet import Fernet

from backend.config import AUDIO_BUFFER_MAX_SECONDS, SAMPLE_RATE, ENCRYPTION_KEY_ENV


class AudioPrivacyBuffer:
    """
    Volatile in-memory rolling buffer for streaming audio analysis.
    Guarantees automatic eviction and zero permanent retention of raw audio data.
    """

    def __init__(self, max_seconds: float = AUDIO_BUFFER_MAX_SECONDS, sr: int = SAMPLE_RATE):
        self.max_seconds = max_seconds
        self.sr = sr
        self.max_samples = int(max_seconds * sr)
        self._buffer = np.zeros(0, dtype=np.float32)

    def append_chunk(self, chunk: np.ndarray):
        """Append incoming audio samples and discard overflow samples beyond max window."""
        if chunk.ndim > 1:
            chunk = np.mean(chunk, axis=1)
        self._buffer = np.concatenate([self._buffer, chunk.astype(np.float32)])
        if len(self._buffer) > self.max_samples:
            self._buffer = self._buffer[-self.max_samples:]

    def get_window(self, num_samples: int) -> Optional[np.ndarray]:
        """Extract latest window copy."""
        if len(self._buffer) < num_samples:
            return None
        return self._buffer[-num_samples:].copy()

    def purge(self):
        """Zero-out and purge buffer immediately from memory."""
        self._buffer.fill(0.0)
        self._buffer = np.zeros(0, dtype=np.float32)


class VoiceprintVault:
    """
    Encrypts and protects enrolled speaker embeddings using AES-128-CBC / HMAC-SHA256 (Fernet).
    Raw enrollment audio is discarded, storing only encrypted numerical embeddings.
    """

    def __init__(self, key: Optional[bytes] = None):
        if key is None:
            env_key = os.getenv(ENCRYPTION_KEY_ENV)
            if env_key:
                self.key = env_key.encode("utf-8")
            else:
                self.key = Fernet.generate_key()
        else:
            self.key = key
        self.cipher = Fernet(self.key)

    def encrypt_embedding(self, embedding: np.ndarray) -> bytes:
        """Serialize and encrypt speaker embedding vector."""
        raw_bytes = embedding.astype(np.float32).tobytes()
        return self.cipher.encrypt(raw_bytes)

    def decrypt_embedding(self, encrypted_data: bytes) -> np.ndarray:
        """Decrypt and deserialize speaker embedding vector."""
        decrypted_bytes = self.cipher.decrypt(encrypted_data)
        return np.frombuffer(decrypted_bytes, dtype=np.float32)
