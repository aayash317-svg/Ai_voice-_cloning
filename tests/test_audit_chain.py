"""
Unit Tests for Tamper-Evident Security Audit Chain.
Verifies event chaining and tamper detection mechanisms.
"""

import sys
import tempfile
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.audit_chain import AuditChain


def test_audit_chain_creation_and_tamper_detection():
    with tempfile.TemporaryDirectory() as tmp_dir:
        chain_path = Path(tmp_dir) / "test_chain.json"
        audit = AuditChain(chain_file=chain_path)

        # 1. Verify initial genesis block
        is_valid, err = audit.verify_integrity()
        assert is_valid is True
        assert err is None
        assert len(audit.chain) == 1

        # 2. Append events
        audit.append_event("ENROLLMENT", {"user_id": "speaker_101"})
        audit.append_event("RISK_ASSESSMENT", {"risk_score": 92.5, "level": "CRITICAL"})

        is_valid, err = audit.verify_integrity()
        assert is_valid is True
        assert len(audit.chain) == 3

        # 3. Simulate unauthorized historical tampering (changing risk from 92.5 to 22.0)
        audit.chain[2].event_data["risk_score"] = 22.0

        is_valid, err = audit.verify_integrity()
        assert is_valid is False
        assert "Tampering detected" in err
        print("[+] Test passed: Tamper detection successfully flagged unauthorized ledger modification!")


if __name__ == "__main__":
    test_audit_chain_creation_and_tamper_detection()
