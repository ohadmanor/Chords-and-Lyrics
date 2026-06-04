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

    # 2. Verify PyInstaller is available.
    try:
        import PyInstaller  # noqa: F401
        print(f"PyInstaller version {PyInstaller.__version__} is available.")
    except ImportError:
        print("Error: pyinstaller is not installed. Install it with: pip install pyinstaller")
        sys.exit(1)

    # Windows add-data syntax: source;dest (os.pathsep is ';' on Windows).
    add_data_arg = f"{frontend_dist}{os.pathsep}static"

    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--onefile",
        "--name=Harmonix",
        "--noconfirm",
        "--clean",
        f"--add-data={add_data_arg}",
        # Search the backend folder so `import chord_extractor` resolves.
        f"--paths={backend_dir}",
        "--hidden-import=chord_extractor",
        # --- madmom: needs its bundled NN model files (data) plus all submodules,
        #     because it resolves processors/models dynamically at runtime. The
        #     other scientific libs (librosa, scipy, sklearn, numba, soundfile,
        #     soxr, audioread) are handled automatically by PyInstaller's built-in
        #     hooks, so we deliberately do NOT collect-all them (that would drag in
        #     their multi-hundred-module test suites and bloat the exe).
        "--collect-data=madmom",
        "--collect-submodules=madmom",
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
