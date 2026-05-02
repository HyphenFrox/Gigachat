"""One-shot backfill: set port=8000 for any paired_devices row that
came in via the older pair-notify shape (which omitted host_port,
leaving port stored as NULL).

After 2026-05 the host's pair-notify includes host_port so the
receiver stores it directly. This script is a one-time cleanup for
records created before that change. Idempotent — re-running it on a
fresh DB is a no-op (UPDATE matches zero rows).

Run:
    & .\\.venv\\Scripts\\python.exe scripts/backfill_paired_port.py
"""
import os
import sqlite3
import sys

# Bootstrap path so `from backend import ...` works when run from
# the project root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend import db  # noqa: E402


def main() -> int:
    db.init()
    conn = sqlite3.connect(db.DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute(
            "UPDATE paired_devices SET port = 8000 WHERE port IS NULL"
        )
        conn.commit()
        print(f"backfilled {cur.rowcount} row(s) -> port=8000")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
