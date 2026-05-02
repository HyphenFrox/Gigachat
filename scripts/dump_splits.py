"""Dump split_models rows + in-memory _running registry."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend import db, split_lifecycle  # noqa: E402

db.init()
print("--- split_models DB rows ---")
print(json.dumps(db.list_split_models(), indent=2, default=str))
print()
print("--- in-memory _running registry ---")
for sid, rp in split_lifecycle._running.items():
    print(f"  {sid}: pid={rp.proc.pid} port={rp.port} ngl={rp.ngl} alive={rp.proc.poll() is None}")
if not split_lifecycle._running:
    print("  (empty)")
