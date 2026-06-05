"""
Unified runner / packager for Harmonix.

Usage:
    python run.py                      # dev mode (default)
    python run.py dev                  # dev mode (FastAPI --reload + ng serve)
    python run.py prod                 # prod mode (ng build + static serve + uvicorn)
    python run.py prod --skip-build    # prod mode, reuse existing frontend build
    python run.py build                # package a standalone Harmonix.exe (PyInstaller)
    python run.py build --skip-frontend  # package, reuse existing frontend build

Modes:
  dev   - FastAPI backend on :8000 with --reload (auto-restart on .py changes)
          and the Angular dev server (`ng serve`) on :4200 with live HMR.
  prod  - Builds the Angular production bundle, serves the static files on :4300
          with an SPA fallback, and runs uvicorn without --reload.
  build - Builds the Angular bundle, downloads FFmpeg if needed, then runs
          PyInstaller to produce a single dist/Harmonix.exe.
"""
import os
import sys
import time
import threading
import subprocess
import webbrowser
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer


# ---------------------------------------------------------------------------
# Shared paths / config
# ---------------------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(ROOT_DIR, "backend")
BACKEND_APP = os.path.join(BACKEND_DIR, "app.py")
FRONTEND_DIR = os.path.join(ROOT_DIR, "frontend")
FRONTEND_DIST = os.path.join(FRONTEND_DIR, "dist", "frontend", "browser")
FFMPEG_DIR = os.path.join(ROOT_DIR, "ffmpeg")

BACKEND_HOST = "127.0.0.1"
BACKEND_PORT = 8000
FRONTEND_PORT_DEV = 4200
FRONTEND_PORT_PROD = 4300


def banner(title: str) -> None:
    line = "=" * 60
    print(f"\n{line}\n{title}\n{line}")


def install_backend_dependencies() -> None:
    banner("Installing/verifying Python backend dependencies")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
            cwd=BACKEND_DIR,
            check=True,
        )
        print("Backend dependencies OK.\n")
    except subprocess.CalledProcessError as e:
        print(f"Warning: pip install failed: {e}")


def install_frontend_dependencies() -> None:
    node_modules = os.path.join(FRONTEND_DIR, "node_modules")
    if os.path.isdir(node_modules):
        return
    banner("Installing frontend npm dependencies (first run)")
    subprocess.run("npm install", shell=True, cwd=FRONTEND_DIR, check=True)


# ---------------------------------------------------------------------------
# DEV mode
# ---------------------------------------------------------------------------
def _dev_run_backend() -> None:
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


def _dev_run_frontend() -> None:
    banner("Starting Angular dev server (ng serve)")
    subprocess.run(
        f"npm run start -- --port {FRONTEND_PORT_DEV}",
        shell=True,
        cwd=FRONTEND_DIR,
    )


def run_dev(args) -> None:
    install_backend_dependencies()
    install_frontend_dependencies()

    threading.Thread(target=_dev_run_backend, daemon=True).start()
    threading.Thread(target=_dev_run_frontend, daemon=True).start()

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


# ---------------------------------------------------------------------------
# PROD mode
# ---------------------------------------------------------------------------
def _prod_build_frontend() -> None:
    banner("Building Angular production bundle")
    subprocess.run(
        "npm run build -- --configuration production",
        shell=True,
        cwd=FRONTEND_DIR,
        check=True,
    )
    if not os.path.isdir(FRONTEND_DIST):
        raise RuntimeError(f"Build finished but {FRONTEND_DIST} was not produced.")


class _SpaRequestHandler(SimpleHTTPRequestHandler):
    """Static file handler that falls back to /index.html for client-side routes."""

    def do_GET(self) -> None:  # noqa: N802 (stdlib signature)
        requested = self.translate_path(self.path)
        if not os.path.exists(requested) and "." not in os.path.basename(self.path):
            self.path = "/index.html"
        super().do_GET()


def _prod_run_backend() -> None:
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


def _prod_run_static_server() -> None:
    banner(f"Serving production frontend from {FRONTEND_DIST}")
    handler = partial(_SpaRequestHandler, directory=FRONTEND_DIST)
    httpd = ThreadingHTTPServer(("0.0.0.0", FRONTEND_PORT_PROD), handler)
    httpd.serve_forever()


def run_prod(args) -> None:
    skip_build = "--skip-build" in args

    install_backend_dependencies()
    install_frontend_dependencies()

    if skip_build and os.path.isdir(FRONTEND_DIST):
        print("Skipping build (--skip-build) — using existing dist.")
    else:
        _prod_build_frontend()

    threading.Thread(target=_prod_run_backend, daemon=True).start()
    threading.Thread(target=_prod_run_static_server, daemon=True).start()

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


# ---------------------------------------------------------------------------
# BUILD mode (standalone .exe via PyInstaller)
# ---------------------------------------------------------------------------
def _ensure_ffmpeg():
    """Make sure ffmpeg.exe/ffprobe.exe live in ./ffmpeg, downloading if needed.

    yt-dlp needs FFmpeg to extract/convert audio. To keep the packaged exe
    self-contained we ship the binaries inside it, so they must exist in the
    project before PyInstaller runs. Returns the folder holding the binaries.
    """
    ffmpeg_exe = os.path.join(FFMPEG_DIR, "ffmpeg.exe")
    ffprobe_exe = os.path.join(FFMPEG_DIR, "ffprobe.exe")
    if os.path.isfile(ffmpeg_exe) and os.path.isfile(ffprobe_exe):
        print(f"FFmpeg already present in {FFMPEG_DIR}.")
        return FFMPEG_DIR

    if os.name != "nt":
        print("Error: automatic FFmpeg download is only implemented for Windows.")
        print(f"Please place ffmpeg/ffprobe binaries in {FFMPEG_DIR} manually.")
        sys.exit(1)

    import urllib.request
    import zipfile

    # Try several mirrors; GitHub (BtbN) is usually the fastest/most reliable.
    urls = [
        "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip",
        "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip",
    ]
    os.makedirs(FFMPEG_DIR, exist_ok=True)
    archive_path = os.path.join(FFMPEG_DIR, "_ffmpeg_download.zip")

    downloaded = False
    for url in urls:
        print(f"=== Downloading FFmpeg from {url} ===")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                total = int(resp.headers.get("Content-Length", 0))
                read = 0
                last_pct = -1
                with open(archive_path, "wb") as out:
                    while True:
                        chunk = resp.read(1 << 16)
                        if not chunk:
                            break
                        out.write(chunk)
                        read += len(chunk)
                        if total:
                            pct = int(read * 100 / total)
                            if pct >= last_pct + 10:
                                last_pct = pct
                                print(f"  {pct}% ({read // (1 << 20)} MB / {total // (1 << 20)} MB)")
            downloaded = True
            break
        except Exception as e:
            print(f"  download failed: {e}")

    if not downloaded:
        print("Error: failed to download FFmpeg from all mirrors.")
        print(f"Please download a Windows build manually and place ffmpeg.exe and "
              f"ffprobe.exe in {FFMPEG_DIR}.")
        sys.exit(1)

    print("Extracting ffmpeg.exe and ffprobe.exe...")
    with zipfile.ZipFile(archive_path) as zf:
        for member in zf.namelist():
            name = os.path.basename(member)
            if name in ("ffmpeg.exe", "ffprobe.exe"):
                with zf.open(member) as src, open(os.path.join(FFMPEG_DIR, name), "wb") as dst:
                    dst.write(src.read())

    try:
        os.remove(archive_path)
    except OSError:
        pass

    if not (os.path.isfile(ffmpeg_exe) and os.path.isfile(ffprobe_exe)):
        print("Error: FFmpeg archive did not contain the expected binaries.")
        sys.exit(1)
    print(f"FFmpeg ready in {FFMPEG_DIR}.")
    return FFMPEG_DIR


def _build_frontend_exe():
    npm = "npm.cmd" if os.name == "nt" else "npm"
    if not os.path.isdir(os.path.join(FRONTEND_DIR, "node_modules")):
        print("=== Installing frontend dependencies (npm install) ===")
        subprocess.run([npm, "install"], cwd=FRONTEND_DIR, check=True)
    print("=== Building Angular production bundle ===")
    subprocess.run(
        [npm, "run", "build", "--", "--configuration", "production"],
        cwd=FRONTEND_DIR,
        check=True,
    )
    if not os.path.isdir(FRONTEND_DIST):
        print(f"Error: build finished but {FRONTEND_DIST} was not produced.")
        sys.exit(1)


def run_build(args) -> None:
    print("=== Harmonix Packaging Script ===")

    skip_frontend = "--skip-frontend" in args

    # 1. Build (or reuse) the frontend static assets.
    if skip_frontend and os.path.isdir(FRONTEND_DIST):
        print(f"Skipping frontend build, using existing: {FRONTEND_DIST}")
    else:
        _build_frontend_exe()
    print(f"Found frontend static assets at: {FRONTEND_DIST}")

    # 1b. Ensure FFmpeg binaries are available to bundle (yt-dlp needs them).
    _ensure_ffmpeg()

    # 2. Verify PyInstaller is available.
    try:
        import PyInstaller  # noqa: F401
        print(f"PyInstaller version {PyInstaller.__version__} is available.")
    except ImportError:
        print("Error: pyinstaller is not installed. Install it with: pip install pyinstaller")
        sys.exit(1)

    # Windows add-data syntax: source;dest (os.pathsep is ';' on Windows).
    add_data_arg = f"{FRONTEND_DIST}{os.pathsep}static"

    # Bundle FFmpeg binaries into an "ffmpeg" subfolder of the app bundle so
    # yt-dlp can find them at runtime (see get_ffmpeg_location in app.py).
    ffmpeg_binaries = [
        f"--add-binary={os.path.join(FFMPEG_DIR, 'ffmpeg.exe')}{os.pathsep}ffmpeg",
        f"--add-binary={os.path.join(FFMPEG_DIR, 'ffprobe.exe')}{os.pathsep}ffmpeg",
    ]

    # madmom is an OPTIONAL backend dependency (chord_extractor falls back to a
    # template chroma when it is absent). Only ask PyInstaller to bundle it when
    # it is actually installed, otherwise --collect-data=madmom aborts the build.
    try:
        import madmom  # noqa: F401
        madmom_available = True
        print("madmom detected: bundling its models and submodules.")
    except ImportError:
        madmom_available = False
        print("madmom not installed: building without the DeepChroma front-end.")

    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--onefile",
        "--name=Harmonix",
        "--noconfirm",
        "--clean",
        # Run without a console window: the app is a background web server that
        # opens the UI in the browser, so the CLI window should stay hidden.
        "--noconsole",
        f"--add-data={add_data_arg}",
        *ffmpeg_binaries,
        # Search the backend folder so `import chord_extractor` resolves.
        f"--paths={BACKEND_DIR}",
        "--hidden-import=chord_extractor",
    ]

    if madmom_available:
        # --- madmom: needs its bundled NN model files (data) plus all submodules,
        #     because it resolves processors/models dynamically at runtime. The
        #     other scientific libs (librosa, scipy, sklearn, numba, soundfile,
        #     soxr, audioread) are handled automatically by PyInstaller's built-in
        #     hooks, so we deliberately do NOT collect-all them (that would drag in
        #     their multi-hundred-module test suites and bloat the exe).
        cmd += [
            "--collect-data=madmom",
            "--collect-submodules=madmom",
        ]

    cmd += [
        # Trim test suites that occasionally get pulled in transitively.
        "--exclude-module=numba.tests",
        "--exclude-module=scipy.io.tests",
        "--exclude-module=sklearn.tests",
        "--exclude-module=matplotlib",
        "--exclude-module=torch",
        # --- Web server runtime imports (resolved dynamically by uvicorn) ---
        "--hidden-import=uvicorn.logging",
        "--hidden-import=uvicorn.loops",
        "--hidden-import=uvicorn.loops.auto",
        "--hidden-import=uvicorn.protocols",
        "--hidden-import=uvicorn.protocols.http",
        "--hidden-import=uvicorn.protocols.http.auto",
        "--hidden-import=uvicorn.protocols.websockets",
        "--hidden-import=uvicorn.protocols.websockets.auto",
        "--hidden-import=uvicorn.lifespan",
        "--hidden-import=uvicorn.lifespan.on",
        "--hidden-import=uvicorn.lifespan.off",
        # Entry point.
        BACKEND_APP,
    ]

    print("Running PyInstaller (this can take several minutes)...")
    print(" ".join(cmd))
    try:
        subprocess.run(cmd, check=True, cwd=ROOT_DIR)
        print("\n=== Success! Harmonix.exe generated in dist/Harmonix.exe ===")
    except subprocess.CalledProcessError as e:
        print(f"\nError: PyInstaller build failed: {e}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------
def main() -> None:
    args = sys.argv[1:]
    mode = (args[0].lower() if args else "dev")
    rest = args[1:]

    if mode in ("dev", "debug", "d"):
        run_dev(rest)
    elif mode in ("prod", "release", "p"):
        run_prod(rest)
    elif mode in ("build", "exe", "package", "b"):
        run_build(rest)
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
