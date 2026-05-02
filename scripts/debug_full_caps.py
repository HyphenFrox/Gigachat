"""Print every interesting field per worker for heartbeat / routing
diagnosis."""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend import db
db.init()
for w in db.list_compute_workers():
    if not w.get("enabled"): continue
    caps = w.get("capabilities") or {}
    print(json.dumps({
        "label": w["label"],
        "last_seen": w.get("last_seen"),
        "last_seen_age_s": (time.time() - (w.get("last_seen") or 0)) if w.get("last_seen") else None,
        "last_error": w.get("last_error"),
        "rpc_server_reachable": caps.get("rpc_server_reachable"),
        "current_rpc_backend": caps.get("current_rpc_backend"),
        "ram_free_gb": caps.get("ram_free_gb"),
        "ram_total_gb": caps.get("ram_total_gb"),
        "gpu_kind": caps.get("gpu_kind"),
        "bandwidth_mbps": caps.get("bandwidth_mbps"),
        "ram_free_probed_at_age_s": (time.time() - (caps.get("ram_free_probed_at") or 0)) if caps.get("ram_free_probed_at") else None,
    }, default=str))
