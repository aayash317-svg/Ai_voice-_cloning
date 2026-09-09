"""
Edge & Offline Queue Subsystem
Implements Layer 5 edge-first operation:
- Queues security events locally when network connectivity to central SIEM is severed.
- Automatically flushes and syncs events once connectivity is reestablished.
- Guarantees detection and local audit logging never block on internet availability.
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, List
from sqlalchemy.ext.asyncio import AsyncSession

from config.settings import settings
from backend.app.audit.hash_chain import HashChainService


class EdgeQueueService:
    """Manages offline local security event queue and sync-on-reconnect."""

    def __init__(self, queue_dir: str = None):
        self.queue_dir = Path(queue_dir or settings.EDGE_QUEUE_DIR)
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        self.queue_file = self.queue_dir / "pending_events.jsonl"
        self._is_offline = settings.OFFLINE_MODE
        self.last_sync_time = None

    @property
    def is_offline(self) -> bool:
        return self._is_offline

    def set_offline_mode(self, offline: bool):
        self._is_offline = offline

    def enqueue_event(self, event_type: str, payload: Dict[str, Any], session_id: str = None) -> Dict[str, Any]:
        """Appends event to local disk queue when offline."""
        item = {
            "event_type": event_type,
            "payload": payload,
            "session_id": session_id,
            "queued_at": time.time(),
        }
        with open(self.queue_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(item) + "\n")
        return item

    def get_pending_count(self) -> int:
        """Returns number of events currently awaiting synchronization."""
        if not self.queue_file.exists():
            return 0
        with open(self.queue_file, "r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())

    def get_status(self) -> Dict[str, Any]:
        """Returns edge node operational status and pending sync queue."""
        pending_count = self.get_pending_count()
        return {
            "node_mode": "EDGE_OFFLINE" if self._is_offline else "ONLINE",
            "offline_mode": self._is_offline,
            "pending_sync_events": pending_count,
            "last_synced_at": self.last_sync_time,
            "edge_queue_path": str(self.queue_file),
            "status": "HEALTHY",
        }

    async def sync_events(self, db: AsyncSession) -> Dict[str, Any]:
        """Flushes queued local events into persistent hash chain upon reconnect."""
        if not self.queue_file.exists():
            return {"synced_count": 0, "status": "NO_PENDING_EVENTS"}

        lines: List[str] = []
        with open(self.queue_file, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]

        if not lines:
            return {"synced_count": 0, "status": "NO_PENDING_EVENTS"}

        hash_svc = HashChainService(db)
        synced_count = 0

        for line in lines:
            item = json.loads(line)
            await hash_svc.record_event(
                event_type=item["event_type"],
                payload={**item["payload"], "edge_synced": True, "original_queued_at": item["queued_at"]},
                session_id=item.get("session_id"),
            )
            synced_count += 1

        # Clear queue file
        self.queue_file.unlink(missing_ok=True)
        self.last_sync_time = time.time()

        return {
            "synced_count": synced_count,
            "status": "SYNC_COMPLETE",
            "timestamp": self.last_sync_time,
        }
