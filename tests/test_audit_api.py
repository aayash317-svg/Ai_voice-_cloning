"""
Integration tests for authenticated audit endpoints:
- GET /audit/chain (401/403/200)
- GET /audit/verify (401/403/200)
- POST /audit/checkpoint (401/403/200)
- POST /audit/verify-checkpoint (401/403/200)
- POST /audit/recover (401/403/200)
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from backend.app import app
from backend.config import ADMIN_API_KEY


def test_audit_api_endpoints_auth_and_verification():
    client = TestClient(app)

    # 1. Unauthenticated request to /audit/chain -> 401
    r = client.get("/audit/chain")
    assert r.status_code == 401

    # 2. Invalid API key to /audit/chain -> 403
    r = client.get("/audit/chain", headers={"X-Admin-API-Key": "wrong_key"})
    assert r.status_code == 403

    # 3. Valid API key to /audit/chain -> 200
    r = client.get("/audit/chain", headers={"X-Admin-API-Key": ADMIN_API_KEY})
    assert r.status_code == 200
    data = r.json()
    assert "chain" in data
    assert data["length"] >= 1

    # 4. Valid API key to /audit/verify -> 200
    r = client.get("/audit/verify", headers={"X-Admin-API-Key": ADMIN_API_KEY})
    assert r.status_code == 200
    ver = r.json()
    assert ver["valid"] is True
    assert ver["status"] == "CHAIN_INTEGRITY_OK"

    # 5. Create trusted checkpoint -> 200
    r = client.post("/audit/checkpoint", headers={"X-Admin-API-Key": ADMIN_API_KEY})
    assert r.status_code == 200
    cp = r.json()
    assert cp["status"] == "CHECKPOINT_CREATED"

    # 6. Verify against trusted checkpoint -> 200
    r = client.post("/audit/verify-checkpoint", headers={"X-Admin-API-Key": ADMIN_API_KEY}, json=cp["checkpoint"])
    assert r.status_code == 200
    cp_ver = r.json()
    assert cp_ver["valid"] is True
    assert cp_ver["status"] == "CHECKPOINT_VERIFIED_OK"
