"""
Release runner.

- Builds the Angular app in production mode (`ng build --configuration production`)
- Serves the built static bundle from `frontend/dist/frontend/browser` on :4300
  using Python's built-in http.server (SPA-friendly fallback to index.html)
- Starts FastAPI backend on :8000 without --reload, single worker, info logging
- Opens the browser to the production frontend

Run:  python run_prod.py [--skip-build]
"""
import os
import sys
import time
import threading
import subprocess
import webbrowser
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

from _run_common import (
    BACKEND_DIR,
    BACKEND_HOST,
    BACKEND_PORT,
    FRONTEND_DIR,
    FRONTEND_DIST,
    FRONTEND_PORT_PROD,
    banner,
    install_backend_dependencies,
    install_frontend_dependencies,
)


def build_frontend() -> None:
    banner("Building Angular production bundle")
    subprocess.run(
        "npm run build -- --configuration production",
        shell=True,
        cwd=FRONTEND_DIR,
        check=True,
    )
    if not os.path.isdir(FRONTEND_DIST):
        raise RuntimeError(f"Build finished but {FRONTEND_DIST} was not produced.")


class SpaRequestHandler(SimpleHTTPRequestHandler):
    """Static file handler that falls back to /index.html for client-side routes."""

    def do_GET(self) -> None:  # noqa: N802 (stdlib signature)
        requested = self.translate_path(self.path)
        if not os.path.exists(requested) and "." not in os.path.basename(self.path):
            self.path = "/index.html"
        super().do_GET()


def run_backend() -> None:
    banner("Starting FastAPI backend (PROD)")
    subprocess.run(
        [
            sys.executable, "-m", "uvicorn", "app:app",
            "--host", BACKEND_HOST,
            "--port", str(BACKEND_PORT),
            "--log-level", "info",
            "--no-access-log",
        ],
        cwd=BACKEND_DIR,
    )


def run_static_server() -> None:
    banner(f"Serving production frontend from {FRONTEND_DIST}")
    handler = partial(SpaRequestHandler, directory=FRONTEND_DIST)
    httpd = ThreadingHTTPServer(("0.0.0.0", FRONTEND_PORT_PROD), handler)
    httpd.serve_forever()


def main() -> None:
    skip_build = "--skip-build" in sys.argv

    install_backend_dependencies()
    install_frontend_dependencies()

    if skip_build and os.path.isdir(FRONTEND_DIST):
        print("Skipping build (--skip-build) — using existing dist.")
    else:
        build_frontend()

    threading.Thread(target=run_backend, daemon=True).start()
    threading.Thread(target=run_static_server, daemon=True).start()

    print("\nWaiting for servers to come up...")
    time.sleep(3)

    url = f"http://localhost:{FRONTEND_PORT_PROD}"
    banner(f"PROD mode running -> {url}    (Ctrl+C to stop)")
    webbrowser.open(url)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down production servers...")
        sys.exit(0)


if __name__ == "__main__":
    main()
