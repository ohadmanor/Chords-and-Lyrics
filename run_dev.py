"""
Development runner.

- FastAPI backend on :8000 with --reload (auto-restart on .py changes)
- Angular dev server (`ng serve`) on :4200 with live HMR
- Opens the browser to the Angular dev server

Run:  python run_dev.py
"""
import os
import sys
import time
import threading
import subprocess
import webbrowser

from _run_common import (
    BACKEND_DIR,
    BACKEND_HOST,
    BACKEND_PORT,
    FRONTEND_DIR,
    FRONTEND_PORT_DEV,
    banner,
    install_backend_dependencies,
    install_frontend_dependencies,
)


def run_backend() -> None:
    banner("Starting FastAPI backend (DEV, --reload)")
    subprocess.run(
        [
            sys.executable, "-m", "uvicorn", "app:app",
            "--host", BACKEND_HOST,
            "--port", str(BACKEND_PORT),
            "--reload",
            "--log-level", "debug",
        ],
        cwd=BACKEND_DIR,
    )


def run_frontend() -> None:
    banner("Starting Angular dev server (ng serve)")
    subprocess.run(
        f"npm run start -- --port {FRONTEND_PORT_DEV}",
        shell=True,
        cwd=FRONTEND_DIR,
    )


def main() -> None:
    install_backend_dependencies()
    install_frontend_dependencies()

    threading.Thread(target=run_backend, daemon=True).start()
    threading.Thread(target=run_frontend, daemon=True).start()

    print("\nWaiting for servers to come up...")
    time.sleep(5)

    url = f"http://localhost:{FRONTEND_PORT_DEV}"
    banner(f"DEV mode running -> {url}    (Ctrl+C to stop)")
    webbrowser.open(url)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down dev servers...")
        sys.exit(0)


if __name__ == "__main__":
    main()
