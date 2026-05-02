"""Reset a split_models row to stopped. Pass the row id, or no
argument to reset every row."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend import db  # noqa: E402

db.init()
sid = sys.argv[1] if len(sys.argv) > 1 else None
if not sid:
    for r in db.list_split_models():
        print(f"resetting {r['id']} ({r['label']}) status={r['status']}")
        db.update_split_model_status(
            r["id"], status="stopped", last_error="reset by test",
        )
else:
    db.update_split_model_status(
        sid, status="stopped", last_error="reset by test",
    )
    print(f"reset {sid}")
