"""Force the auto-fallback selector to pick SYCL0,CPU hybrid by
marking SYCL0-alone and Vulkan0-alone as failed; clearing hybrid
flags. Used to live-test whether a newer llama.cpp build fixes
the ggml-rpc.cpp hybrid-allocator layout-mismatch crash."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend import db, compute_pool as p
db.init()
for w in db.list_compute_workers():
    if not w.get("enabled"):
        continue
    caps = dict(w.get("capabilities") or {})
    caps["sycl_split_failed_at"] = time.time()
    caps["vulkan_split_failed_at"] = time.time()
    caps.pop("sycl_hybrid_split_failed_at", None)
    caps.pop("vulkan_hybrid_split_failed_at", None)
    db.update_compute_worker_capabilities(w["id"], capabilities=caps)
print("flags set; selector decisions:")
for w in db.list_compute_workers():
    if w.get("enabled"):
        print(" ", w["label"], "->", p._select_worker_backend(w, in_split=True))
