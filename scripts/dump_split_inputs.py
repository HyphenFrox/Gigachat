"""Show what _compute_optimal_ngl + _compute_tensor_split_ratios +
_resolve_rpc_endpoints produce for the current pool state."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend import db, split_lifecycle as sl
db.init()
workers = [w for w in db.list_compute_workers() if w.get("enabled")]
wids = [w["id"] for w in workers]

gguf = r"C:\Users\gauta\.gigachat\llama-cpp\models\gemma4-31b.gguf"
print(f"workers: {[w['label'] for w in workers]}")

endpoints = sl._resolve_rpc_endpoints(wids)
print(f"resolved endpoints ({len(endpoints)}): {endpoints}")

ts = sl._compute_tensor_split_ratios(gguf, wids)
print(f"tensor_split weights ({len(ts) if ts else 'None'}): {ts}")

ngl = sl._compute_optimal_ngl(gguf, wids)
print(f"ngl: {ngl}")

# Per-worker rpc_endpoints capability
for w in workers:
    eps = (w.get("capabilities") or {}).get("rpc_endpoints")
    print(f"  {w['label']} rpc_endpoints: {json.dumps(eps)}")
