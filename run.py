import os
import sys
import subprocess
import time
import webbrowser
import threading

# Resolve directory paths
root_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.join(root_dir, "backend")
frontend_dir = os.path.join(root_dir, "frontend")

def install_backend_dependencies():
    print("==================================================")
    print("Installing/verifying Python backend dependencies...")
    print("==================================================")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
            cwd=backend_dir,
            check=True
        )
        print("Backend dependencies check complete.\n")
    except subprocess.CalledProcessError as e:
        print(f"Warning: Failed to automatically install dependencies: {e}")
        print("Ensure pip is installed and you have internet access, then run pip manually.")

def run_backend():
    print("Starting FastAPI backend server...")
    try:
        # Run Uvicorn directly from python -m module execution to ensure using correct environment
        subprocess.run(
            [sys.executable, "-m", "uvicorn", "app:app", "--host", "127.0.0.1", "--port", "8000"],
            cwd=backend_dir
        )
    except Exception as e:
        print(f"Backend server error: {e}")

def run_frontend():
    print("Starting Angular dev server...")
    try:
      # On Windows, shell=True is needed to run npm/npx scripts
      subprocess.run("npm run start", shell=True, cwd=frontend_dir)
    except Exception as e:
        print(f"Frontend server error: {e}")

def main():
    # Install backend packages first
    install_backend_dependencies()
    
    # Spawn servers in separate daemon threads
    backend_thread = threading.Thread(target=run_backend, daemon=True)
    backend_thread.start()
    
    frontend_thread = threading.Thread(target=run_frontend, daemon=True)
    frontend_thread.start()
    
    # Wait for servers to initialize
    print("Initializing servers... please wait...")
    time.sleep(4)
    
    url = "http://localhost:4200"
    print(f"\n==================================================")
    print(f"App is running! Opening browser to: {url}")
    print(f"Press CTRL+C in this console to shut down.")
    print(f"==================================================\n")
    
    # Open the default web browser to the React dev server url
    webbrowser.open(url)
    
    try:
        # Keep the main process alive to monitor threads
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down application...")
        # Since threads are daemon threads, they will exit automatically with the main thread
        sys.exit(0)

if __name__ == "__main__":
    main()
