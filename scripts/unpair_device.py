"""One-shot: drop a paired_devices row by device_id (or label).

Used during P2P debugging to purge a stale / phantom pair record so
the next pair attempt starts clean. Idempotent — refusing to drop a
non-existent record is a no-op.

Usage:

    & .\\.venv\\Scripts\\python.exe scripts/unpair_device.py <device_id_or_label>

Drops the row whose device_id equals the argument; if no match,
falls back to label match.
"""
import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend import db  # noqa: E402


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: unpair_device.py <device_id_or_label>")
        return 2
    needle = argv[1]
    db.init()
    conn = sqlite3.connect(db.DB_PATH)
    try:
        cur = conn.cursor()
        # Match by device_id first; if zero rows, fall through to label.
        cur.execute(
            "DELETE FROM paired_devices WHERE device_id = ?",
            (needle,),
        )
        n = cur.rowcount
        if n == 0:
            cur.execute(
                "DELETE FROM paired_devices WHERE label = ?",
                (needle,),
            )
            n = cur.rowcount
        # Cascade: if a compute_worker is keyed on this device, drop
        # that too so the UI doesn't keep trying to probe a peer we
        # just unpaired.
        cur.execute(
            "DELETE FROM compute_workers WHERE gigachat_device_id = ?",
            (needle,),
        )
        cw_n = cur.rowcount
        conn.commit()
        print(f"dropped {n} paired_devices row(s), {cw_n} compute_workers row(s) for {needle!r}")
    finally:
        conn.close()
    return 0 if n > 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
