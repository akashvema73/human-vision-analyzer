import os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))


def _ensure_env():
    """Galat Python se chala to .venv312 se dobara chalao."""
    try:
        import fastapi, uvicorn  # noqa: F401
        return
    except ImportError:
        pass
    for cand in (os.path.join(HERE, ".venv312", "Scripts", "python.exe"),
                 os.path.join(HERE, "..", ".venv312", "Scripts", "python.exe"),
                 os.path.join(HERE, ".venv", "Scripts", "python.exe")):
        cand = os.path.abspath(cand)
        if os.path.exists(cand) and cand != os.path.abspath(sys.executable):
            print(f"[run] {cand} se chala raha hoon...")
            sys.exit(subprocess.call([cand, os.path.abspath(__file__)], cwd=HERE))
    print("Packages nahi mile. Python 3.12 venv me: pip install -r requirements.txt")
    sys.exit(1)


_ensure_env()
import uvicorn  # noqa: E402

if __name__ == "__main__":
    os.chdir(HERE)
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=False)
