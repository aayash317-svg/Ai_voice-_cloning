"""
Tamper-Evident Security Audit Trail.
Maintains a cryptographic SHA-256 hash-chain of security and risk assessment events.
Detects any unauthorized historical modification of audit records.
"""

import json
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict

from backend.config import AUDIT_CHAIN_FILE


@dataclass
class AuditBlock:
    """A single immutable block in the security audit chain."""
    index: int
    timestamp: float
    event_type: str            # e.g., 'RISK_ASSESSMENT', 'ALERT_TRIGGERED', 'ENROLLMENT'
    event_data: Dict[str, Any] # Non-biometric metadata: risk score, level, flags
    previous_hash: str
    current_hash: str


class AuditChain:
    """Cryptographically chained event ledger."""

    def __init__(self, chain_file: Path = AUDIT_CHAIN_FILE):
        self.chain_file = Path(chain_file)
        self.chain: List[AuditBlock] = []
        self._load_or_init_chain()

    def _hash_payload(self, index: int, timestamp: float, event_type: str, event_data: dict, prev_hash: str) -> str:
        """Compute SHA-256 hash of block payload."""
        payload = {
            "index": index,
            "timestamp": timestamp,
            "event_type": event_type,
            "event_data": event_data,
            "previous_hash": prev_hash
        }
        encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _load_or_init_chain(self):
        """Load existing ledger from disk or create Genesis block."""
        if self.chain_file.exists():
            try:
                with open(self.chain_file, "r", encoding="utf-8") as f:
                    raw_blocks = json.load(f)
                    self.chain = [AuditBlock(**b) for b in raw_blocks]
                    return
            except Exception:
                pass

        # Initialize Genesis Block
        genesis_time = time.time()
        genesis_hash = self._hash_payload(0, genesis_time, "GENESIS", {"msg": "Audit chain initialized"}, "0" * 64)
        genesis_block = AuditBlock(
            index=0,
            timestamp=genesis_time,
            event_type="GENESIS",
            event_data={"msg": "Audit chain initialized"},
            previous_hash="0" * 64,
            current_hash=genesis_hash
        )
        self.chain = [genesis_block]
        self._save_chain()

    def _save_chain(self):
        """Persist chain to disk."""
        self.chain_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.chain_file, "w", encoding="utf-8") as f:
            json.dump([asdict(b) for b in self.chain], f, indent=2)

    def append_event(self, event_type: str, event_data: Dict[str, Any]) -> AuditBlock:
        """Append a new security event to the tamper-evident chain."""
        last_block = self.chain[-1]
        new_index = len(self.chain)
        timestamp = time.time()
        prev_hash = last_block.current_hash

        curr_hash = self._hash_payload(new_index, timestamp, event_type, event_data, prev_hash)

        block = AuditBlock(
            index=new_index,
            timestamp=timestamp,
            event_type=event_type,
            event_data=event_data,
            previous_hash=prev_hash,
            current_hash=curr_hash
        )
        self.chain.append(block)
        self._save_chain()
        return block

    def verify_integrity(self) -> Tuple[bool, Optional[str]]:
        """
        Verify every link in the chain. Returns (True, None) if intact,
        or (False, error_description) if historical tampering is detected.
        """
        if not self.chain:
            return False, "Audit chain is empty."

        for i, block in enumerate(self.chain):
            # Check re-calculated hash
            expected_hash = self._hash_payload(
                block.index, block.timestamp, block.event_type, block.event_data, block.previous_hash
            )
            if block.current_hash != expected_hash:
                return False, f"Tampering detected at block {block.index}: Hash mismatch. Expected {expected_hash}, found {block.current_hash}"

            # Check chain linkage with previous block
            if i > 0:
                prev_block = self.chain[i - 1]
                if block.previous_hash != prev_block.current_hash:
                    return False, f"Broken chain link between block {prev_block.index} and {block.index}."

        return True, None
