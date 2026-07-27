"""
UniBot – all-in-one launcher
=============================
Run this ONE file to do everything:

    python start.py

What it does, in order:
  1. Creates a virtual-environment (.venv) if one doesn't exist
  2. Installs / upgrades all required packages into that venv
  3. Copies .env.example → .env if .env is missing
  4. Re-executes itself inside the venv (so the installed packages are usable)
  5. Starts the FastAPI server on http://localhost:8000
     - GET /          → serves index.html (the React frontend)
     - GET /images/*  → serves the images/ folder
     - /rag-chat etc. → the backend API

No separate terminal needed. One command, fully self-contained.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

# Force UTF-8 output on Windows so Unicode chars don't crash
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── paths ─────────────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent          # project root
VENV_DIR = HERE / ".venv"
BACKEND_REQ = HERE / "agentic_rag" / "requirements.txt"
ENV_EXAMPLE = HERE / "agentic_rag" / ".env.example"
ENV_FILE = HERE / "agentic_rag" / ".env"

# The python / pip executables that live inside the venv
if sys.platform == "win32":
    VENV_PYTHON = VENV_DIR / "Scripts" / "python.exe"
    VENV_PIP    = VENV_DIR / "Scripts" / "pip.exe"
else:
    VENV_PYTHON = VENV_DIR / "bin" / "python"
    VENV_PIP    = VENV_DIR / "bin" / "pip"

# ── helpers ───────────────────────────────────────────────────────────────────

def _banner(text: str) -> None:
    width = 60
    print()
    print("-" * width)
    print(f"  {text}")
    print("-" * width)


def _run(*cmd, cwd: Path = HERE, check: bool = True) -> int:
    """Run a subprocess, streaming output live."""
    result = subprocess.run(
        [str(a) for a in cmd],
        cwd=str(cwd),
        check=check,
    )
    return result.returncode


def _inside_venv() -> bool:
    """True when this script is already running inside the project venv."""
    return Path(sys.executable).resolve().parent == (VENV_DIR / "Scripts").resolve() or \
           Path(sys.executable).resolve().parent == (VENV_DIR / "bin").resolve()


# ── step 1 – virtual environment ──────────────────────────────────────────────

def ensure_venv() -> None:
    if VENV_DIR.exists():
        print(f"[OK] Virtual environment found at {VENV_DIR.name}/")
        return

    _banner("Step 1 - Creating virtual environment")
    _run(sys.executable, "-m", "venv", str(VENV_DIR))
    print(f"[OK] Virtual environment created at {VENV_DIR.name}/")


# ── step 2 – install dependencies ─────────────────────────────────────────────

def install_dependencies() -> None:
    _banner("Step 2 - Installing / verifying dependencies")
    _run(
        VENV_PYTHON, "-m", "pip", "install", "--upgrade", "pip",
        "--quiet",
    )
    _run(
        VENV_PYTHON, "-m", "pip", "install", "-r", str(BACKEND_REQ),
    )
    print("[OK] Dependencies ready")


# ── step 3 – environment file ─────────────────────────────────────────────────

def ensure_env_file() -> None:
    if ENV_FILE.exists():
        print("[OK] Config file found: agentic_rag/.env")
        return

    _banner("Step 3 - Creating .env from .env.example")
    if not ENV_EXAMPLE.exists():
        print("[!] No .env.example found - skipping (you may need to create agentic_rag/.env manually)")
        return

    shutil.copy(ENV_EXAMPLE, ENV_FILE)
    print("[OK] Copied .env.example -> agentic_rag/.env")
    print()
    print("  !! IMPORTANT: open  agentic_rag/.env  and fill in your API keys,")
    print("     then re-run  python start.py")
    print()
    sys.exit(0)   # exit so the user can fill in keys before the server starts


# ── step 4 – re-execute inside venv ───────────────────────────────────────────

def relaunch_in_venv() -> None:
    """If we are NOT running from the venv python, restart using it."""
    if _inside_venv():
        return  # already inside - continue to server startup

    _banner("Step 4 - Switching to virtual-environment Python")
    print(f"  Using: {VENV_PYTHON}")
    # Spawn a new process using the venv Python and wait for it to finish.
    # (os.execv is unreliable on Windows, so we use subprocess + sys.exit)
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    result = subprocess.run(
        [str(VENV_PYTHON), __file__] + sys.argv[1:],
        cwd=str(HERE),
        env=env,
    )
    sys.exit(result.returncode)


# ── step 5 – start the server ─────────────────────────────────────────────────

def _open_browser() -> None:
    """Wait for the server to be ready, then open the browser."""
    import threading  # noqa: PLC0415
    import time       # noqa: PLC0415
    import webbrowser # noqa: PLC0415

    def _launch():
        time.sleep(1.5)  # give uvicorn a moment to start
        webbrowser.open("http://localhost:8000")

    threading.Thread(target=_launch, daemon=True).start()


def start_server() -> None:
    # These imports only work once we are inside the venv (step 4 ensures that)
    import uvicorn  # noqa: PLC0415

    # Make the project root importable
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))

    # Patch app.py to serve the frontend before importing it
    _ensure_frontend_served()

    print()
    print("=" * 60)
    print("  UniBot is running!")
    print()
    print("  Opening browser -> http://localhost:8000")
    print()
    print("  API docs: http://localhost:8000/docs")
    print("  Press Ctrl+C to stop")
    print("=" * 60)
    print()

    # Open the browser in the background while the server starts
    _open_browser()

    uvicorn.run(
        "agentic_rag.app:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )


def _ensure_frontend_served() -> None:
    """
    Idempotently make sure FastAPI serves index.html at GET /.
    This monkey-patches the app object only if the route doesn't exist yet,
    so it's safe to call multiple times.
    """
    from fastapi.responses import FileResponse          # noqa: PLC0415
    from fastapi.staticfiles import StaticFiles         # noqa: PLC0415
    from agentic_rag.app import app                     # noqa: PLC0415

    # Check if we already registered the frontend route
    existing_paths = {r.path for r in app.routes if hasattr(r, "path")}

    index_file = HERE / "index.html"
    images_dir = HERE / "images"

    if "/images" not in existing_paths and images_dir.exists():
        app.mount("/images", StaticFiles(directory=str(images_dir)), name="images")

    if "/" not in existing_paths:
        @app.get("/", include_in_schema=False)
        async def serve_frontend() -> FileResponse:
            return FileResponse(str(index_file), media_type="text/html")


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    print()
    print("=" * 60)
    print("         UniBot - All-in-One Launcher")
    print("=" * 60)

    if not _inside_venv():
        # Steps 1-4 run with the system python (no dependencies needed)
        ensure_venv()
        install_dependencies()
        ensure_env_file()
        relaunch_in_venv()   # spawns venv python, then exits
    else:
        # We're inside the venv - dependencies are available
        ensure_env_file()    # double-check .env exists
        start_server()       # start FastAPI + frontend


if __name__ == "__main__":
    main()
