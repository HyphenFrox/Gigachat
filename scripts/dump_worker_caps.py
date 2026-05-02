"""Print key capabilities for each worker for debugging."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend import db
db.init()
for w in db.list_compute_workers():
    caps = w.get("capabilities") or {}
    print(json.dumps({
        "label": w["label"],
        "current_rpc_backend": caps.get("current_rpc_backend"),
        "rpc_server_reachable": caps.get("rpc_server_reachable"),
        "ram_free_gb": caps.get("ram_free_gb"),
    }))
