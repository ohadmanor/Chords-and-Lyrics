import os
import subprocess
import sys

# Paths
root_dir = os.path.dirname(os.path.abspath(__file__))
frontend_dist = os.path.join(root_dir, "frontend", "dist", "frontend", "browser")
backend_app = os.path.join(root_dir, "backend", "app.py")

def main():
    print("=== Harmonix Packaging Script ===")
    
    # 1. Verify frontend dist exists
    if not os.path.exists(frontend_dist):
        print(f"Error: Static assets not found at: {frontend_dist}")
        print("Please build the frontend first using: npm run build in frontend/")
        sys.exit(1)
        
    print(f"Found frontend static assets at: {frontend_dist}")
    
    # 2. Check if pyinstaller is installed
    try:
        import PyInstaller
        print(f"PyInstaller version {PyInstaller.__version__} is available.")
    except ImportError:
        print("Error: pyinstaller is not installed. Please install it using: pip install pyinstaller")
        sys.exit(1)
        
    # 3. Assemble PyInstaller command
    # Windows syntax: source_dir;dest_dir
    add_data_arg = f"{frontend_dist};static"
    
    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--onefile",
        "--name=Harmonix",
        f"--add-data={add_data_arg}",
        # Hidden imports for uvicorn, librosa, and soundfile
        "--hidden-import=soundfile",
        "--hidden-import=librosa",
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
        # Clean build folders before starting
        "--clean",
        # Source script path
        backend_app
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        print("\n=== Success! Harmonix.exe generated in dist/Harmonix.exe ===")
    except subprocess.CalledProcessError as e:
        print(f"\nError: PyInstaller build failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
