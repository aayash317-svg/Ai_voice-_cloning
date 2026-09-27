"""
Comprehensive Hardening & Tamper-Evident Test Suite for Voice Shield AI Audit Ledger.
Tests all failure and tampering scenarios:
1. Genesis block initialization
2. Legitimate event appending & verification
3. Tamper event_data (e.g. altering fraud risk score)
4. Tamper current_hash (direct hash mutation)
5. Tamper previous_hash (broken continuity)
6. Delete a block (sequence & linkage break)
7. Reorder blocks (index & chronological disruption)
8. Change block index (sequence mismatch)
9. Corrupted JSON (refuses silent genesis reset, raises AUDIT_LEDGER_CORRUPTED)
10. Empty ledger file (0 bytes, refuses silent genesis reset)
11. Duplicate block (index and hash repetition)
12. Atomic write safety & non-corrupting disk updates
13. External trusted checkpoint creation & verification against rewritten ledger
14. Thread-safe concurrent event appending
"""

import sys
import json
import time
import shutil
import tempfile
import threading
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.audit_chain import AuditChain, AuditLedgerCorruptedError, AuditBlock


def test_1_genesis_creation():
    """Test genesis block creation and valid structure."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        chain_file = Path(tmp_dir) / "audit_chain.json"
        ac = AuditChain(chain_file=chain_file)

        assert len(ac.chain) == 1
        assert ac.chain[0].index == 0
        assert ac.chain[0].event_type == "GENESIS"
        assert ac.chain[0].previous_hash == "0" * 64

        res = ac.verify_integrity()
        assert res.valid is True
        assert res.blocks_verified == 1
        assert res.status == "CHAIN_INTEGRITY_OK"
        print("  [PASSED] Test 1: Genesis creation and structure")


def test_2_valid_event_appending():
    """Test normal sequential appending of security events."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        chain_file = Path(tmp_dir) / "audit_chain.json"
        ac = AuditChain(chain_file=chain_file)

        b1 = ac.append_event("BATCH_ANALYSIS", {"risk_score": 12.4, "alert": False})
        b2 = ac.append_event("CALL_ANALYSIS", {"overall_call_risk": 84.2, "verdict": "CLONE_DETECTED"})
        b3 = ac.append_event("STREAM_ALERT", {"risk_score": 96.0, "risk_level": "CRITICAL"})

        assert len(ac.chain) == 4
        assert b1.index == 1
        assert b2.index == 2
        assert b3.index == 3

        res = ac.verify_integrity()
        assert res.valid is True
        assert res.blocks_verified == 4
        assert res.status == "CHAIN_INTEGRITY_OK"
        print("  [PASSED] Test 2: Valid event sequence appending")


def test_3_tamper_event_data():
    """Test modifying payload metadata (e.g. changing 88.0 risk score to 10.0)."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        chain_file = Path(tmp_dir) / "audit_chain.json"
        ac = AuditChain(chain_file=chain_file)
        ac.append_event("CALL_ANALYSIS", {"risk_score": 88.0, "verdict": "SPOOF"})
        ac.append_event("CALL_ANALYSIS", {"risk_score": 12.0, "verdict": "AUTHENTIC"})

        # Tamper event_data in block 1
        ac.chain[1].event_data["risk_score"] = 10.0

        res = ac.verify_integrity()
        assert res.valid is False
        assert res.status == "CHAIN_TAMPER_DETECTED"
        assert "Tampering detected" in res.error_detail
        print("  [PASSED] Test 3: Tamper event_data detected")


def test_4_tamper_current_hash():
    """Test modifying the current_hash field directly."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        chain_file = Path(tmp_dir) / "audit_chain.json"
        ac = AuditChain(chain_file=chain_file)
        ac.append_event("CALL_ANALYSIS", {"risk_score": 75.0})

        # Mutate current_hash of block 1
        orig = ac.chain[1].current_hash
        ac.chain[1].current_hash = "a" * 64

        res = ac.verify_integrity()
        assert res.valid is False
        assert res.status == "CHAIN_TAMPER_DETECTED"
        print("  [PASSED] Test 4: Tamper current_hash detected")


def test_5_tamper_previous_hash():
    """Test mutating the previous_hash link between blocks."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        chain_file = Path(tmp_dir) / "audit_chain.json"
        ac = AuditChain(chain_file=chain_file)
        ac.append_event("BATCH_ANALYSIS", {"risk_score": 25.0})
        ac.append_event("BATCH_ANALYSIS", {"risk_score": 30.0})

        # Break linkage between block 1 and block 2
        ac.chain[2].previous_hash = "f" * 64

        res = ac.verify_integrity()
        assert res.valid is False
        assert res.status in ("BROKEN_CHAIN_LINK", "CHAIN_TAMPER_DETECTED")
        print("  [PASSED] Test 5: Tamper previous_hash detected")


def test_6_tamper_delete_block():
    """Test deleting an intermediate block."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        chain_file = Path(tmp_dir) / "audit_chain.json"
        ac = AuditChain(chain_file=chain_file)
        ac.append_event("CALL_1", {"data": 1})
        ac.append_event("CALL_2", {"data": 2})
        ac.append_event("CALL_3", {"data": 3})

        # Delete intermediate block 1 (leaving 0, 2, 3)
        del ac.chain[1]

        res = ac.verify_integrity()
        assert res.valid is False
        assert res.status in ("SEQUENCE_BROKEN", "BROKEN_CHAIN_LINK")
        print("  [PASSED] Test 6: Block deletion detected")


def test_7_tamper_reorder_blocks():
    """Test swapping two valid blocks."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        chain_file = Path(tmp_dir) / "audit_chain.json"
        ac = AuditChain(chain_file=chain_file)
        ac.append_event("CALL_1", {"data": 1})
        ac.append_event("CALL_2", {"data": 2})

        # Swap blocks 1 and 2
        ac.chain[1], ac.chain[2] = ac.chain[2], ac.chain[1]

        res = ac.verify_integrity()
        assert res.valid is False
        assert res.status in ("SEQUENCE_BROKEN", "BROKEN_CHAIN_LINK", "TIMESTAMP_OUT_OF_ORDER")
        print("  [PASSED] Test 7: Block reordering detected")


def test_8_tamper_change_block_index():
    """Test changing a block index."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        chain_file = Path(tmp_dir) / "audit_chain.json"
        ac = AuditChain(chain_file=chain_file)
        ac.append_event("CALL_1", {"data": 1})

        # Alter index of block 1 to 99
        ac.chain[1].index = 99

        res = ac.verify_integrity()
        assert res.valid is False
        assert res.status == "SEQUENCE_BROKEN"
        print("  [PASSED] Test 8: Block index modification detected")


def test_9_corrupted_json_refuses_silent_genesis():
    """
    CRITICAL REQUIREMENT 1:
    If audit_chain.json is corrupted, it must NOT silently create a new genesis block.
    It must report AUDIT_LEDGER_CORRUPTED and require administrator intervention.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        chain_file = Path(tmp_dir) / "audit_chain.json"

        # Create corrupted JSON content
        with open(chain_file, "w") as f:
            f.write("CORRUPTED_NON_JSON_DATA_[[[[{{{{")

        # In strict mode, must raise AuditLedgerCorruptedError
        try:
            AuditChain(chain_file=chain_file, strict=True)
            assert False, "Should have raised AuditLedgerCorruptedError"
        except AuditLedgerCorruptedError as e:
            assert "AUDIT_LEDGER_CORRUPTED" in str(e)

        # In standard mode, must flag corrupted state and refuse to overwrite
        ac = AuditChain(chain_file=chain_file, strict=False)
        assert ac.is_corrupted is True
        assert ac.status == "AUDIT_LEDGER_CORRUPTED"
        assert len(ac.chain) == 0

        # Attempting to append an event to corrupted chain must fail
        try:
            ac.append_event("ILLEGAL_EVENT", {})
            assert False, "Append should have failed on corrupted ledger"
        except AuditLedgerCorruptedError:
            pass

        # Verifying integrity must report corruption
        res = ac.verify_integrity()
        assert res.valid is False
        assert res.status == "AUDIT_LEDGER_CORRUPTED"

        # The file on disk must NOT have been silently overwritten with a new genesis
        with open(chain_file, "r") as f:
            content = f.read()
            assert "CORRUPTED_NON_JSON_DATA" in content

        print("  [PASSED] Test 9: Corrupted JSON handled safely (no silent genesis wipe)")


def test_10_empty_ledger_refuses_silent_genesis():
    """Test that a 0-byte file triggers AUDIT_LEDGER_CORRUPTED without overwriting."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        chain_file = Path(tmp_dir) / "audit_chain.json"

        # Create 0-byte empty file
        chain_file.touch()

        ac = AuditChain(chain_file=chain_file, strict=False)
        assert ac.is_corrupted is True
        assert ac.status == "AUDIT_LEDGER_CORRUPTED"

        res = ac.verify_integrity()
        assert res.valid is False
        assert res.status == "AUDIT_LEDGER_CORRUPTED"
        print("  [PASSED] Test 10: Empty (0-byte) ledger safely detected")


def test_11_tamper_duplicate_block():
    """Test duplicate block insertion."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        chain_file = Path(tmp_dir) / "audit_chain.json"
        ac = AuditChain(chain_file=chain_file)
        ac.append_event("CALL_1", {"data": 1})

        # Duplicate block 1
        ac.chain.append(ac.chain[1])

        res = ac.verify_integrity()
        assert res.valid is False
        assert res.status in ("SEQUENCE_BROKEN", "BROKEN_CHAIN_LINK")
        print("  [PASSED] Test 11: Duplicate block detected")


def test_12_atomic_write_safety():
    """Test atomic file write and content persistence across reloads."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        chain_file = Path(tmp_dir) / "audit_chain.json"
        ac = AuditChain(chain_file=chain_file)

        for i in range(10):
            ac.append_event(f"EVENT_{i}", {"seq": i, "val": i * 10})

        # Reload from disk into a new AuditChain instance
        ac_reloaded = AuditChain(chain_file=chain_file)
        assert len(ac_reloaded.chain) == 11
        res = ac_reloaded.verify_integrity()
        assert res.valid is True
        assert res.blocks_verified == 11
        assert res.status == "CHAIN_INTEGRITY_OK"
        print("  [PASSED] Test 12: Atomic write persistence and reload verified")


def test_13_trusted_external_checkpoint():
    """
    Test external trusted checkpoint mechanism.
    If an attacker regenerates the entire chain with new valid hashes,
    the trusted checkpoint catches the discrepancy.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        chain_file = Path(tmp_dir) / "audit_chain.json"
        checkpoint_file = Path(tmp_dir) / "audit_checkpoint.json"

        ac = AuditChain(chain_file=chain_file, checkpoint_file=checkpoint_file)
        ac.append_event("EVENT_A", {"score": 50.0})
        ac.append_event("EVENT_B", {"score": 90.0})

        # Create trusted checkpoint of valid chain
        checkpoint = ac.create_checkpoint()
        assert checkpoint["blocks_count"] == 3
        assert checkpoint["last_block_index"] == 2

        # Verify against checkpoint passes
        cp_res = ac.verify_against_checkpoint(checkpoint)
        assert cp_res.valid is True
        assert cp_res.status == "CHECKPOINT_VERIFIED_OK"

        # Attacker scenario: An attacker with server write access creates a completely
        # different valid chain (rewriting history from block 0)
        attacker_chain_file = Path(tmp_dir) / "forged_chain.json"
        attacker_ac = AuditChain(chain_file=attacker_chain_file)
        attacker_ac.append_event("FORGED_EVENT_A", {"score": 0.0}) # Cleared fraud score
        attacker_ac.append_event("FORGED_EVENT_B", {"score": 0.0})

        # Attacker's chain is internally valid
        assert attacker_ac.verify_integrity().valid is True

        # BUT testing attacker's chain against the trusted checkpoint FAILS!
        cp_check = attacker_ac.verify_against_checkpoint(checkpoint)
        assert cp_check.valid is False
        assert "CHECKPOINT" in cp_check.status
        print("  [PASSED] Test 13: Trusted external checkpoint blocks chain rewrite attacks")


def test_14_concurrent_thread_safety():
    """Test thread-safety with concurrent event appends."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        chain_file = Path(tmp_dir) / "audit_chain.json"
        ac = AuditChain(chain_file=chain_file)

        errors = []

        def worker(worker_id):
            for i in range(15):
                try:
                    ac.append_event(f"WORKER_{worker_id}", {"step": i})
                    time.sleep(0.001)
                except Exception as ex:
                    errors.append(ex)

        threads = [threading.Thread(target=worker, args=(w,)) for w in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(ac.chain) == 1 + (4 * 15) # Genesis + 60 events

        # Verify chain integrity after 60 concurrent writes
        res = ac.verify_integrity()
        assert res.valid is True
        assert res.blocks_verified == 61
        assert res.status == "CHAIN_INTEGRITY_OK"
        print("  [PASSED] Test 14: Concurrent thread-safe appends verified")


def run_all_tamper_tests():
    print("=" * 70)
    print("RUNNING VOICE SHIELD AI AUDIT-CHAIN HARDENING & TAMPER TEST SUITE")
    print("=" * 70)

    test_1_genesis_creation()
    test_2_valid_event_appending()
    test_3_tamper_event_data()
    test_4_tamper_current_hash()
    test_5_tamper_previous_hash()
    test_6_tamper_delete_block()
    test_7_tamper_reorder_blocks()
    test_8_tamper_change_block_index()
    test_9_corrupted_json_refuses_silent_genesis()
    test_10_empty_ledger_refuses_silent_genesis()
    test_11_tamper_duplicate_block()
    test_12_atomic_write_safety()
    test_13_trusted_external_checkpoint()
    test_14_concurrent_thread_safety()

    print("=" * 70)
    print("ALL 14 AUDIT-CHAIN HARDENING & TAMPER TESTS PASSED!")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tamper_tests()
