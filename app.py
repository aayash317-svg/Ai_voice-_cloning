"""
Voice Integrity Verification Framework Application Entrypoint.
Runs the FastAPI + WebSocket backend server.
"""

import os
import socket
import uvicorn

def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0

if __name__ == "__main__":
    host = os.getenv("HOST", "127.0.0.1")
    requested_port = int(os.getenv("PORT", 8000))
    port = requested_port

    if is_port_in_use(port, host):
        # Check what process is occupying the port
        occupant = "another running process"
        try:
            import psutil
            for conn in psutil.net_connections():
                if conn.laddr.port == port and conn.status == 'LISTEN':
                    proc = psutil.Process(conn.pid)
                    occupant = f"PID {conn.pid} ('{proc.name()}')"
                    break
        except Exception:
            pass

        alt_port = 8050
        print("\n" + "!" * 64)
        print(f"NOTICE: Port {port} is already occupied by {occupant}.")
        print(f"Switching Voice Shield to alternate port {alt_port} to prevent conflict.")
        print("!" * 64)
        port = alt_port

    print("\n" + "=" * 60)
    print("Voice Shield AI -- Authentic Audio Intelligence Dashboard:")
    print(f"  * http://localhost:{port}")
    print(f"  * http://127.0.0.1:{port}")
    uvicorn.run("backend.app:app", host=host, port=port, reload=True, reload_dirs=["backend"])
