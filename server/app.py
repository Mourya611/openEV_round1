from __future__ import annotations

import os

import uvicorn
from app import app as root_app


def main() -> None:
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run(root_app, host=host, port=port)


if __name__ == "__main__":
    main()
