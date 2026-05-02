"""Diagnostic: dump paired_devices as JSON.

Use to inspect what a peer has stored about us (X25519 key present?
right IP? right port?). Run via the .venv's python so backend imports
work:

    & .\\.venv\\Scripts\\python.exe scripts/dump_paired.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend import db  # noqa: E402


def main() -> int:
    db.init()
    print(json.dumps(db.list_paired_devices(), indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
