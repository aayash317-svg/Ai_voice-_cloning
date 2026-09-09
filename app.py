"""
Voice Integrity Verification Framework Application Entrypoint.
Runs the FastAPI + WebSocket backend server.
"""

import uvicorn

if __name__ == "__main__":
    print("Starting Voice Integrity Verification Server on http://localhost:8000 ...")
    uvicorn.run("backend.app:app", host="0.0.0.0", port=8000, reload=True)
