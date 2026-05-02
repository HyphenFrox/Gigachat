"""Print rpc_endpoints capability for each enabled worker."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend import db
db.init()
for w in db.list_compute_workers():
    if not w.get("enabled"):
        continue
    caps = w.get("capabilities") or {}
    print(w["label"], "->", json.dumps({
        "rpc_endpoints": caps.get("rpc_endpoints"),
        "current_rpc_backend": caps.get("current_rpc_backend"),
        "rpc_server_reachable": caps.get("rpc_server_reachable"),
    }))
