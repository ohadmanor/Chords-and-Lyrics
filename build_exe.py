"""
Build a standalone Windows release of Harmonix as a single .exe.

What it does:
  1. Builds the Angular production bundle (npm run build) unless --skip-frontend.
  2. Runs PyInstaller, bundling the backend, the Angular static assets, and the
     heavy scientific dependencies (madmom models, librosa data, sklearn, etc.).

The resulting binary is dist/Harmonix.exe. Running it starts the FastAPI backend
on http://127.0.0.1:8000 and opens a browser to the app.

Usage:
  python build_exe.py                 # build frontend, then package
  python build_exe.py --skip-frontend # reuse existing frontend/dist bundle
"""
import os
import subprocess
import sys

# Paths
root_dir = os.path.dirname(os.path.abspath(__file__))
frontend_dir = os.path.join(root_dir, "frontend")
frontend_dist = os.path.join(root_dir, "frontend", "dist", "frontend", "browser")
backend_dir = os.path.join(root_dir, "backend")
backend_app = os.path.join(backend_dir, "app.py")
ffmpeg_dir = os.path.join(root_dir, "ffmpeg")


def ensure_ffmpeg():
    """Make sure ffmpeg.exe/ffprobe.exe live in ./ffmpeg, downloading if needed.

    yt-dlp needs FFmpeg to extract/convert audio. To keep the packaged exe
    self-contained we ship the binaries inside it, so they must exist in the
    project before PyInstaller runs. Returns the folder holding the binaries.
    """
    ffmpeg_exe = os.path.join(ffmpeg_dir, "ffmpeg.exe")
    ffprobe_exe = os.path.join(ffmpeg_dir, "ffprobe.exe")
    if os.path.isfile(ffmpeg_exe) and os.path.isfile(ffprobe_exe):
        print(f"FFmpeg already present in {ffmpeg_dir}.")
        return ffmpeg_dir

    if os.name != "nt":
        print("Error: automatic FFmpeg download is only implemented for Windows.")
        print(f"Please place ffmpeg/ffprobe binaries in {ffmpeg_dir} manually.")
        sys.exit(1)

    import urllib.request
    import zipfile

    # Try several mirrors; GitHub (BtbN) is usually the fastest/most reliable.
    urls = [
        "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip",
        "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip",
    ]
    os.makedirs(ffmpeg_dir, exist_ok=True)
    archive_path = os.path.join(ffmpeg_dir, "_ffmpeg_download.zip")

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
        print(f"Error: failed to download FFmpeg from all mirrors.")
        print(f"Please download a Windows build manually and place ffmpeg.exe and "
              f"ffprobe.exe in {ffmpeg_dir}.")
        sys.exit(1)

    print("Extracting ffmpeg.exe and ffprobe.exe...")
    with zipfile.ZipFile(archive_path) as zf:
        for member in zf.namelist():
            name = os.path.basename(member)
            if name in ("ffmpeg.exe", "ffprobe.exe"):
                with zf.open(member) as src, open(os.path.join(ffmpeg_dir, name), "wb") as dst:
                    dst.write(src.read())

    try:
        os.remove(archive_path)
    except OSError:
        pass

    if not (os.path.isfile(ffmpeg_exe) and os.path.isfile(ffprobe_exe)):
        print("Error: FFmpeg archive did not contain the expected binaries.")
        sys.exit(1)
    print(f"FFmpeg ready in {ffmpeg_dir}.")
    return ffmpeg_dir



def build_frontend():
    npm = "npm.cmd" if os.name == "nt" else "npm"
    if not os.path.isdir(os.path.join(frontend_dir, "node_modules")):
        print("=== Installing frontend dependencies (npm install) ===")
        subprocess.run([npm, "install"], cwd=frontend_dir, check=True)
    print("=== Building Angular production bundle ===")
    subprocess.run(
        [npm, "run", "build", "--", "--configuration", "production"],
        cwd=frontend_dir,
        check=True,
    )
    if not os.path.isdir(frontend_dist):
        print(f"Error: build finished but {frontend_dist} was not produced.")
        sys.exit(1)


def main():
    print("=== Harmonix Packaging Script ===")

    skip_frontend = "--skip-frontend" in sys.argv

    # 1. Build (or reuse) the frontend static assets.
    if skip_frontend and os.path.isdir(frontend_dist):
        print(f"Skipping frontend build, using existing: {frontend_dist}")
    else:
        build_frontend()
    print(f"Found frontend static assets at: {frontend_dist}")

    # 1b. Ensure FFmpeg binaries are available to bundle (yt-dlp needs them).
    ensure_ffmpeg()

    # 2. Verify PyInstaller is available.
    try:
        import PyInstaller  # noqa: F401
        print(f"PyInstaller version {PyInstaller.__version__} is available.")
    except ImportError:
        print("Error: pyinstaller is not installed. Install it with: pip install pyinstaller")
        sys.exit(1)

    # Windows add-data syntax: source;dest (os.pathsep is ';' on Windows).
    add_data_arg = f"{frontend_dist}{os.pathsep}static"

    # Bundle FFmpeg binaries into an "ffmpeg" subfolder of the app bundle so
    # yt-dlp can find them at runtime (see get_ffmpeg_location in app.py).
    ffmpeg_binaries = [
        f"--add-binary={os.path.join(ffmpeg_dir, 'ffmpeg.exe')}{os.pathsep}ffmpeg",
        f"--add-binary={os.path.join(ffmpeg_dir, 'ffprobe.exe')}{os.pathsep}ffmpeg",
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
        f"--paths={backend_dir}",
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
        backend_app,
    ]

    print("Running PyInstaller (this can take several minutes)...")
    print(" ".join(cmd))
    try:
        subprocess.run(cmd, check=True, cwd=root_dir)
        print("\n=== Success! Harmonix.exe generated in dist/Harmonix.exe ===")
    except subprocess.CalledProcessError as e:
        print(f"\nError: PyInstaller build failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
