"""
Shared helpers for the dev and prod runners.
"""
import os
import sys
import subprocess

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(ROOT_DIR, "backend")
FRONTEND_DIR = os.path.join(ROOT_DIR, "frontend")
FRONTEND_DIST = os.path.join(FRONTEND_DIR, "dist", "frontend", "browser")

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
