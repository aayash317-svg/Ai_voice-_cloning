"""
Root ASGI entrypoint for Voice Integrity Verification Framework.
Enables commands like `uvicorn main:app` and default cloud deployment platforms
(Render, Railway, Hugging Face, AWS, Heroku) without module path conflicts.
"""

import os
import sys
from pathlib import Path

# Ensure workspace root is in sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from backend.app import app

if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host=host, port=port, reload=False)
