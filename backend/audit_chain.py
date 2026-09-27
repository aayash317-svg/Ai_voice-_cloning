"""
Tamper-Evident Security Audit Trail (Hardened v2.0).
Maintains a cryptographic SHA-256 hash-chain of security, model risk assessments, and forensic events.
Detects any unauthorized historical modification, block deletion, reordering, or corruption of audit records.

Hardening features:
1. Strict Corruption Handling: Never overwrites corrupted files with a silent genesis block.
2. Comprehensive Chain Verification: Genesis check, strict index sequence, field typing, monotonic timestamps,
   previous hash linkage, and SHA-256 recalculation.
3. Atomic File Writes: Uses temporary file creation, fsync, and atomic rename to prevent half-written ledgers.
4. Thread-Safe: RLock protected for concurrent multi-request environments.
5. Trusted External Checkpointing: Enables anchoring cumulative chain hashes to detect whole-ledger rewriting.
"""

import json
import time
import hashlib
import os
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict

from backend.config import AUDIT_CHAIN_FILE, AUDIT_CHECKPOINT_FILE


class AuditLedgerCorruptedError(RuntimeError):
    """Raised when the audit ledger file is missing, empty, unparseable, or cryptographically invalid."""
    pass


@dataclass
class AuditBlock:
    """A single immutable block in the security audit chain."""
    index: int
    timestamp: float
    event_type: str            # e.g., 'GENESIS', 'BATCH_ANALYSIS', 'CALL_ANALYSIS', 'LIVE_CALL_CLONE_DETECTED'
    event_data: Dict[str, Any] # Non-biometric metadata: risk score, level, audio hash reference
    previous_hash: str
    current_hash: str


@dataclass
class VerificationResult:
    """Structured report returned by chain verification."""
    valid: bool
    blocks_verified: int
    status: str                # e.g., 'CHAIN_INTEGRITY_OK', 'CHAIN_TAMPER_DETECTED', 'AUDIT_LEDGER_CORRUPTED'
    error_detail: Optional[str] = None

    def __iter__(self):
        """Allows unpacking as (is_valid, error_detail) for backwards compatibility."""
        return iter((self.valid, self.error_detail))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "blocks_verified": self.blocks_verified,
            "status": self.status,
            "error_detail": self.error_detail
        }


class AuditChain:
    """Cryptographically chained event ledger with atomic persistence and tamper verification."""

    GENESIS_PREV_HASH = "0" * 64

    def __init__(
        self,
        chain_file: Path = AUDIT_CHAIN_FILE,
        checkpoint_file: Path = AUDIT_CHECKPOINT_FILE,
        auto_init: bool = True,
        strict: bool = False
    ):
        self.chain_file = Path(chain_file)
        self.checkpoint_file = Path(checkpoint_file)
        self.chain: List[AuditBlock] = []
        self._lock = threading.RLock()
        self.is_corrupted: bool = False
        self.corruption_error: Optional[str] = None
        self.status: str = "UNINITIALIZED"

        if auto_init:
            self._load_or_init_chain(strict=strict)

    def _hash_payload(
        self,
        index: int,
        timestamp: float,
        event_type: str,
        event_data: dict,
        prev_hash: str
    ) -> str:
        """Compute canonical SHA-256 hash of block payload."""
        payload = {
            "index": index,
            "timestamp": timestamp,
            "event_type": event_type,
            "event_data": event_data,
            "previous_hash": prev_hash
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(',', ':')).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _validate_blocks(self, blocks: List[AuditBlock]) -> VerificationResult:
        """
        Comprehensive multi-phase chain verification:
        1. Genesis block existence and structure.
        2. Strict 0-indexed sequential progression.
        3. Field types and non-empty hashes.
        4. Monotonic timestamps.
        5. Cryptographic previous-hash linkage.
        6. Payload SHA-256 recalculation.
        """
        if not blocks:
            return VerificationResult(
                valid=False,
                blocks_verified=0,
                status="EMPTY_LEDGER",
                error_detail="Audit ledger contains 0 blocks."
            )

        # 1. Genesis Block Verification
        genesis = blocks[0]
        if genesis.index != 0:
            return VerificationResult(
                valid=False,
                blocks_verified=0,
                status="GENESIS_BLOCK_INVALID",
                error_detail=f"Block 0 index must be 0, found {genesis.index}."
            )
        if genesis.event_type != "GENESIS":
            return VerificationResult(
                valid=False,
                blocks_verified=0,
                status="GENESIS_BLOCK_INVALID",
                error_detail=f"Block 0 event_type must be 'GENESIS', found '{genesis.event_type}'."
            )
        if genesis.previous_hash != self.GENESIS_PREV_HASH:
            return VerificationResult(
                valid=False,
                blocks_verified=0,
                status="GENESIS_BLOCK_INVALID",
                error_detail="Genesis block previous_hash must be 64 zeros."
            )

        # 2. Iterate each block and verify continuity & cryptography
        for i, block in enumerate(blocks):
            # Sequence check
            if block.index != i:
                return VerificationResult(
                    valid=False,
                    blocks_verified=i,
                    status="SEQUENCE_BROKEN",
                    error_detail=f"Sequential index broken at position {i}: expected index {i}, found {block.index}."
                )

            # Field schema check
            if not isinstance(block.timestamp, (int, float)) or block.timestamp <= 0:
                return VerificationResult(
                    valid=False,
                    blocks_verified=i,
                    status="INVALID_BLOCK_SCHEMA",
                    error_detail=f"Block {i} has invalid timestamp: {block.timestamp}."
                )
            if not isinstance(block.event_type, str) or not block.event_type:
                return VerificationResult(
                    valid=False,
                    blocks_verified=i,
                    status="INVALID_BLOCK_SCHEMA",
                    error_detail=f"Block {i} has invalid event_type: {block.event_type}."
                )
            if not isinstance(block.event_data, dict):
                return VerificationResult(
                    valid=False,
                    blocks_verified=i,
                    status="INVALID_BLOCK_SCHEMA",
                    error_detail=f"Block {i} event_data must be a dict."
                )

            # Monotonic timestamp check (allow up to 2.0s negative jitter for minor NTP skew)
            if i > 0 and block.timestamp < (blocks[i - 1].timestamp - 2.0):
                return VerificationResult(
                    valid=False,
                    blocks_verified=i,
                    status="TIMESTAMP_OUT_OF_ORDER",
                    error_detail=f"Block {i} timestamp ({block.timestamp}) precedes block {i-1} timestamp ({blocks[i-1].timestamp})."
                )

            # Cryptographic chain linkage check
            if i > 0:
                prev_block = blocks[i - 1]
                if block.previous_hash != prev_block.current_hash:
                    return VerificationResult(
                        valid=False,
                        blocks_verified=i,
                        status="BROKEN_CHAIN_LINK",
                        error_detail=(
                            f"Broken chain link between block {prev_block.index} and block {block.index}. "
                            f"Block {block.index} expected prev_hash '{prev_block.current_hash}', found '{block.previous_hash}'."
                        )
                    )

            # SHA-256 payload recalculation
            # Note: For backwards compatibility, support both compact separators and standard json dumps
            expected_hash = self._hash_payload(
                block.index, block.timestamp, block.event_type, block.event_data, block.previous_hash
            )
            if block.current_hash != expected_hash:
                # Fallback check with legacy formatting in case serialized with default whitespace
                legacy_encoded = json.dumps({
                    "index": block.index,
                    "timestamp": block.timestamp,
                    "event_type": block.event_type,
                    "event_data": block.event_data,
                    "previous_hash": block.previous_hash
                }, sort_keys=True).encode("utf-8")
                legacy_hash = hashlib.sha256(legacy_encoded).hexdigest()

                if block.current_hash != legacy_hash:
                    return VerificationResult(
                        valid=False,
                        blocks_verified=i,
                        status="CHAIN_TAMPER_DETECTED",
                        error_detail=(
                            f"Tampering detected: Hash mismatch at block {block.index}. "
                            f"Recalculated SHA-256 does not match recorded current_hash."
                        )
                    )

        return VerificationResult(
            valid=True,
            blocks_verified=len(blocks),
            status="CHAIN_INTEGRITY_OK",
            error_detail=None
        )

    def _load_or_init_chain(self, strict: bool = False):
        """
        Load existing ledger from disk or create Genesis block if file does not exist.
        CRITICAL: If the file exists and is corrupted/unreadable/invalid, DO NOT create a new chain!
        Instead, enter an explicit AUDIT_LEDGER_CORRUPTED state and require manual recovery.
        """
        with self._lock:
            if self.chain_file.exists():
                # Check for 0-byte or truncated file
                try:
                    size = self.chain_file.stat().st_size
                    if size == 0:
                        self.is_corrupted = True
                        self.status = "AUDIT_LEDGER_CORRUPTED"
                        self.corruption_error = (
                            f"AUDIT_LEDGER_CORRUPTED: Ledger file '{self.chain_file}' is empty (0 bytes). "
                            f"Refusing to overwrite with new Genesis block. Manual recovery required."
                        )
                        if strict:
                            raise AuditLedgerCorruptedError(self.corruption_error)
                        return
                except OSError as e:
                    self.is_corrupted = True
                    self.status = "AUDIT_LEDGER_CORRUPTED"
                    self.corruption_error = f"AUDIT_LEDGER_CORRUPTED: Cannot access '{self.chain_file}': {e}."
                    if strict:
                        raise AuditLedgerCorruptedError(self.corruption_error)
                    return

                # Parse JSON
                try:
                    with open(self.chain_file, "r", encoding="utf-8") as f:
                        raw_data = json.load(f)
                except Exception as e:
                    self.is_corrupted = True
                    self.status = "AUDIT_LEDGER_CORRUPTED"
                    self.corruption_error = (
                        f"AUDIT_LEDGER_CORRUPTED: Unparseable/corrupted JSON in '{self.chain_file}': {e}. "
                        f"Refusing to recreate Genesis. Manual recovery required."
                    )
                    if strict:
                        raise AuditLedgerCorruptedError(self.corruption_error)
                    return

                # Validate list structure
                if not isinstance(raw_data, list):
                    self.is_corrupted = True
                    self.status = "AUDIT_LEDGER_CORRUPTED"
                    self.corruption_error = (
                        f"AUDIT_LEDGER_CORRUPTED: Expected list of blocks in '{self.chain_file}', "
                        f"found {type(raw_data).__name__}."
                    )
                    if strict:
                        raise AuditLedgerCorruptedError(self.corruption_error)
                    return

                # Parse block schemas
                parsed_blocks: List[AuditBlock] = []
                for i, b in enumerate(raw_data):
                    try:
                        block = AuditBlock(
                            index=int(b["index"]),
                            timestamp=float(b["timestamp"]),
                            event_type=str(b["event_type"]),
                            event_data=dict(b["event_data"]),
                            previous_hash=str(b["previous_hash"]),
                            current_hash=str(b["current_hash"])
                        )
                        parsed_blocks.append(block)
                    except Exception as e:
                        self.is_corrupted = True
                        self.status = "AUDIT_LEDGER_CORRUPTED"
                        self.corruption_error = (
                            f"AUDIT_LEDGER_CORRUPTED: Malformed block schema at index {i}: {e}. "
                            f"Manual recovery required."
                        )
                        if strict:
                            raise AuditLedgerCorruptedError(self.corruption_error)
                        return

                # Validate full cryptographic chain integrity
                validation = self._validate_blocks(parsed_blocks)
                if not validation.valid:
                    self.is_corrupted = True
                    self.status = "AUDIT_LEDGER_CORRUPTED"
                    self.corruption_error = (
                        f"AUDIT_LEDGER_CORRUPTED: Existing ledger failed integrity verification: {validation.error_detail} "
                        f"(Status: {validation.status}). Manual recovery required."
                    )
                    if strict:
                        raise AuditLedgerCorruptedError(self.corruption_error)
                    return

                # Successfully loaded intact ledger
                self.chain = parsed_blocks
                self.is_corrupted = False
                self.corruption_error = None
                self.status = "CHAIN_INTEGRITY_OK"
                return

            # If file does NOT exist, initialize fresh Genesis block
            genesis_time = time.time()
            genesis_event = {"msg": "Audit chain initialized", "version": "2.0"}
            genesis_hash = self._hash_payload(
                0, genesis_time, "GENESIS", genesis_event, self.GENESIS_PREV_HASH
            )
            genesis_block = AuditBlock(
                index=0,
                timestamp=genesis_time,
                event_type="GENESIS",
                event_data=genesis_event,
                previous_hash=self.GENESIS_PREV_HASH,
                current_hash=genesis_hash
            )
            self.chain = [genesis_block]
            self.is_corrupted = False
            self.corruption_error = None
            self.status = "CHAIN_INTEGRITY_OK"
            self._save_chain()

    def _save_chain(self):
        """
        Persist chain to disk using an atomic write pattern.
        Writes to a temporary sibling file, flushes/fsyncs to disk, then atomically replaces target.
        """
        with self._lock:
            self.chain_file.parent.mkdir(parents=True, exist_ok=True)
            # Create a unique temporary file in the same directory (ensures same filesystem)
            tmp_filename = f"{self.chain_file.name}.tmp.{os.getpid()}.{threading.get_ident()}.{int(time.time()*1000)}"
            tmp_path = self.chain_file.parent / tmp_filename

            try:
                with open(tmp_path, "w", encoding="utf-8") as f:
                    json.dump([asdict(b) for b in self.chain], f, indent=2)
                    f.flush()
                    os.fsync(f.fileno())

                # Atomic replace
                os.replace(tmp_path, self.chain_file)
            finally:
                if tmp_path.exists():
                    try:
                        tmp_path.unlink()
                    except OSError:
                        pass

    def append_event(self, event_type: str, event_data: Dict[str, Any]) -> AuditBlock:
        """
        Append a new security or forensic event to the tamper-evident chain.
        Guarantees thread-safety and atomic disk commitment.
        """
        with self._lock:
            if self.is_corrupted:
                raise AuditLedgerCorruptedError(
                    f"AUDIT_LEDGER_CORRUPTED: Cannot append event. {self.corruption_error}"
                )

            if not self.chain:
                raise AuditLedgerCorruptedError(
                    "AUDIT_LEDGER_CORRUPTED: Ledger has no blocks. Cannot append event."
                )

            last_block = self.chain[-1]
            new_index = len(self.chain)
            timestamp = time.time()
            prev_hash = last_block.current_hash

            # Ensure event_data is JSON-serializable
            sanitized_data = json.loads(json.dumps(event_data, default=str))

            curr_hash = self._hash_payload(new_index, timestamp, event_type, sanitized_data, prev_hash)

            block = AuditBlock(
                index=new_index,
                timestamp=timestamp,
                event_type=event_type,
                event_data=sanitized_data,
                previous_hash=prev_hash,
                current_hash=curr_hash
            )
            self.chain.append(block)
            self._save_chain()
            return block

    def verify_integrity(self) -> VerificationResult:
        """
        Cryptographically verify every block and link in the audit ledger.
        Returns a VerificationResult object detailing verification outcome.
        """
        with self._lock:
            if self.is_corrupted:
                return VerificationResult(
                    valid=False,
                    blocks_verified=0,
                    status="AUDIT_LEDGER_CORRUPTED",
                    error_detail=self.corruption_error
                )
            return self._validate_blocks(self.chain)

    def create_checkpoint(self, checkpoint_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Create a trusted cryptographic checkpoint file of the current ledger head.
        Computes a cumulative SHA-256 digest over the entire chain history.
        """
        with self._lock:
            val = self.verify_integrity()
            if not val.valid:
                raise ValueError(f"Cannot checkpoint corrupted ledger: {val.error_detail}")

            target_path = Path(checkpoint_path) if checkpoint_path else self.checkpoint_file
            target_path.parent.mkdir(parents=True, exist_ok=True)

            # Cumulative hash combining all block hashes in sequence
            cumulative_hasher = hashlib.sha256()
            for b in self.chain:
                cumulative_hasher.update(b.current_hash.encode("utf-8"))
            cumulative_hash = cumulative_hasher.hexdigest()

            last_block = self.chain[-1]
            checkpoint_data = {
                "checkpoint_timestamp": time.time(),
                "blocks_count": len(self.chain),
                "last_block_index": last_block.index,
                "last_block_hash": last_block.current_hash,
                "cumulative_hash": cumulative_hash,
                "version": "2.0"
            }

            # Atomic save checkpoint
            tmp_path = target_path.with_name(f"{target_path.name}.tmp.{os.getpid()}")
            try:
                with open(tmp_path, "w", encoding="utf-8") as f:
                    json.dump(checkpoint_data, f, indent=2)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp_path, target_path)
            finally:
                if tmp_path.exists():
                    try:
                        tmp_path.unlink()
                    except OSError:
                        pass

            return checkpoint_data

    def verify_against_checkpoint(self, checkpoint_data: Dict[str, Any]) -> VerificationResult:
        """
        Verify the local chain against an external trusted checkpoint.
        Guarantees that an attacker with local write access hasn't rewritten historical blocks.
        """
        with self._lock:
            # First verify internal integrity
            local_val = self.verify_integrity()
            if not local_val.valid:
                return local_val

            cp_count = checkpoint_data.get("blocks_count", 0)
            cp_last_index = checkpoint_data.get("last_block_index", -1)
            cp_last_hash = checkpoint_data.get("last_block_hash")
            cp_cumulative = checkpoint_data.get("cumulative_hash")

            if len(self.chain) < cp_count:
                return VerificationResult(
                    valid=False,
                    blocks_verified=len(self.chain),
                    status="CHECKPOINT_MISMATCH",
                    error_detail=f"Local chain has {len(self.chain)} blocks, fewer than checkpoint ({cp_count} blocks)."
                )

            if cp_last_index >= len(self.chain):
                return VerificationResult(
                    valid=False,
                    blocks_verified=len(self.chain),
                    status="CHECKPOINT_MISMATCH",
                    error_detail=f"Checkpoint index {cp_last_index} exceeds local chain length {len(self.chain)}."
                )

            target_block = self.chain[cp_last_index]
            if target_block.current_hash != cp_last_hash:
                return VerificationResult(
                    valid=False,
                    blocks_verified=cp_last_index,
                    status="CHECKPOINT_MISMATCH",
                    error_detail=(
                        f"Hash mismatch at checkpoint block {cp_last_index}. "
                        f"Local hash '{target_block.current_hash[:16]}...' does not match checkpoint '{cp_last_hash[:16]}...'."
                    )
                )

            # Recompute cumulative hash up to checkpoint block
            cumulative_hasher = hashlib.sha256()
            for b in self.chain[:cp_last_index + 1]:
                cumulative_hasher.update(b.current_hash.encode("utf-8"))
            local_cumulative = cumulative_hasher.hexdigest()

            if local_cumulative != cp_cumulative:
                return VerificationResult(
                    valid=False,
                    blocks_verified=cp_last_index,
                    status="CHECKPOINT_CUMULATIVE_MISMATCH",
                    error_detail="Cumulative chain hash differs from external trusted checkpoint."
                )

            return VerificationResult(
                valid=True,
                blocks_verified=len(self.chain),
                status="CHECKPOINT_VERIFIED_OK",
                error_detail=None
            )

    def admin_recover_corrupted(self, backup: bool = True, force_new_genesis: bool = False) -> Dict[str, Any]:
        """
        Administrative recovery workflow for corrupted ledger files.
        Requires explicit confirmation flag. Backs up corrupted file before creating fresh Genesis.
        """
        with self._lock:
            backup_path = None
            if backup and self.chain_file.exists():
                backup_path = self.chain_file.with_name(f"{self.chain_file.stem}_corrupt_{int(time.time())}.bak")
                try:
                    self.chain_file.rename(backup_path)
                except OSError as e:
                    raise RuntimeError(f"Failed to backup corrupted ledger: {e}")

            if not force_new_genesis:
                return {
                    "recovered": False,
                    "backup_path": str(backup_path) if backup_path else None,
                    "message": "Corrupted ledger backed up. 'force_new_genesis=True' required to reinitialize."
                }

            # Initialize fresh Genesis block
            genesis_time = time.time()
            genesis_event = {"msg": "Audit chain reinitialized by administrator", "version": "2.0"}
            genesis_hash = self._hash_payload(
                0, genesis_time, "GENESIS", genesis_event, self.GENESIS_PREV_HASH
            )
            genesis_block = AuditBlock(
                index=0,
                timestamp=genesis_time,
                event_type="GENESIS",
                event_data=genesis_event,
                previous_hash=self.GENESIS_PREV_HASH,
                current_hash=genesis_hash
            )
            self.chain = [genesis_block]
            self.is_corrupted = False
            self.corruption_error = None
            self.status = "CHAIN_INTEGRITY_OK"
            self._save_chain()

            return {
                "recovered": True,
                "backup_path": str(backup_path) if backup_path else None,
                "message": "Corrupted ledger recovered. Fresh Genesis block created."
            }
