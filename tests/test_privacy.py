import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from backend.privacy import AudioPrivacyBuffer, VoiceprintVault


def test_privacy_buffer_and_vault():
    # 1. Privacy Buffer eviction test
    buf = AudioPrivacyBuffer(max_seconds=1.0, sr=16000)
    chunk1 = np.ones(8000, dtype=np.float32)
    chunk2 = np.ones(12000, dtype=np.float32) * 2.0

    buf.append_chunk(chunk1)
    buf.append_chunk(chunk2)

    window = buf.get_window(16000)
    assert window is not None
    assert len(window) == 16000
    # The oldest 4000 samples should have been evicted automatically
    assert np.all(window[-12000:] == 2.0)

    buf.purge()
    assert len(buf._buffer) == 0

    # 2. Encrypted Voiceprint Vault test
    vault = VoiceprintVault()
    dummy_embedding = np.random.randn(192).astype(np.float32)
    encrypted = vault.encrypt_embedding(dummy_embedding)
    assert isinstance(encrypted, bytes)

    decrypted = vault.decrypt_embedding(encrypted)
    np.testing.assert_almost_equal(dummy_embedding, decrypted, decimal=5)
    print("[+] Test passed: Audio privacy buffer and encrypted voiceprint vault functioning properly!")


if __name__ == "__main__":
    test_privacy_buffer_and_vault()
