"""
Convenience Runner for Curriculum Effectiveness Analysis Platform
Starts the FastAPI application and automatically launches your browser.
"""

import sys
import os
import time
import webbrowser
import uvicorn

# Ensure project root is in sys.path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


def open_browser(url: str, delay_sec: float = 1.2):
    import threading
    def _open():
        time.sleep(delay_sec)
        print(f"\n[Browser] Opening {url} ...")
        webbrowser.open(url)
    t = threading.Thread(target=_open, daemon=True)
    t.start()


def main():
    host = "127.0.0.1"
    port = 8000
    url = f"http://{host}:{port}"

    print("=" * 70)
    print("  Curriculum Effectiveness Analysis - Learning Analytics Platform")
    print("=" * 70)
    print(f"  [Server]  Starting FastAPI on: {url}")
    print(f"  [Docs]    Interactive API docs: {url}/docs")
    print(f"  [Browser] Launching web browser automatically...")
    print("=" * 70)

    # Automatically open browser once server begins
    open_browser(url)

    # Run Uvicorn
    uvicorn.run("src.api.main:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
