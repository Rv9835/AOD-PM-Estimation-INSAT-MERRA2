#!/usr/bin/env python
"""Run the FastAPI server on localhost."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.append(str(PROJECT_ROOT / "backend"))
sys.path.append(str(PROJECT_ROOT / "backend" / "src"))

import uvicorn

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    print(f"Starting server on http://localhost:{port}")
    print(f"API docs available at http://localhost:{port}/docs")
    print("Press Ctrl+C to stop\n")

    uvicorn.run(
        "airpollution.app:app",
        host="127.0.0.1",
        port=port,
        reload=True,
        log_level="info",
    )
