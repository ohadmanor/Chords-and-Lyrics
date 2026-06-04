"""
Convenience dispatcher.

Usage:
    python run.py            # dev mode (default)
    python run.py dev        # dev mode (FastAPI --reload + ng serve)
    python run.py prod       # prod mode (ng build + static serve + uvicorn)
    python run.py prod --skip-build
"""
import sys
import runpy


def main() -> None:
    args = sys.argv[1:]
    mode = (args[0].lower() if args else "dev")

    if mode in ("dev", "debug", "d"):
        sys.argv = ["run_dev.py"] + args[1:]
        runpy.run_module("run_dev", run_name="__main__")
    elif mode in ("prod", "release", "p"):
        sys.argv = ["run_prod.py"] + args[1:]
        runpy.run_module("run_prod", run_name="__main__")
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
