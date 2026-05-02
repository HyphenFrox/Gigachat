"""Build the EXACT command split_lifecycle would launch llama-server
with for the latest split row. Lets us see -ngl + --tensor-split +
--rpc as actually emitted, without spawning the process."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend import db, split_lifecycle as sl
db.init()
rows = db.list_split_models()
if not rows:
    print("no split rows; create one first")
    sys.exit(1)
row = rows[-1]
gguf = row["gguf_path"]
worker_ids = row.get("worker_ids") or []
print(f"row: {row['id']}  workers={worker_ids}")
print(f"gguf: {gguf}")
endpoints = sl._resolve_rpc_endpoints(worker_ids)
ts = sl._compute_tensor_split_ratios(gguf, worker_ids)
ngl = sl._compute_optimal_ngl(gguf, worker_ids)
ctx = sl._compute_optimal_ctx_size(
    gguf, worker_ids, parallel=1, cache_type=None,
    target_size_bytes=os.path.getsize(gguf),
)
print(f"resolved endpoints ({len(endpoints)}): {endpoints}")
print(f"tensor_split weights ({len(ts) if ts else 'None'}): {ts}")
print(f"ngl: {ngl}")
print(f"ctx: {ctx}")
from backend.split_runtime import LLAMA_CPP_INSTALL_DIR
import platform
exe_name = "llama-server.exe" if platform.system() == "Windows" else "llama-server"
server = LLAMA_CPP_INSTALL_DIR / exe_name
cmd = sl._build_command(
    llama_server=server,
    gguf_path=gguf,
    port=row["llama_port"],
    rpc_endpoints=endpoints,
    ngl=ngl,
    parallel=1,
    tensor_split=ts,
    split_mode=None,
    cache_type=None,
    ctx_size=ctx,
)
print()
print("FINAL CMD:")
for i, a in enumerate(cmd):
    print(f"  [{i:02d}] {a}")
