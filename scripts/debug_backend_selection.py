"""Print per-worker backend-selection decision so we can debug why
the auto-fallback chain picks what it picks."""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend import db, compute_pool as p
db.init()
for w in db.list_compute_workers():
    if not w.get("enabled"): continue
    caps = w.get("capabilities") or {}
    sycl = caps.get("sycl_split_failed_at")
    vulk = caps.get("vulkan_split_failed_at")
    print(json.dumps({
        "label": w["label"],
        "sycl_fail_age_s": (time.time() - sycl) if sycl else None,
        "vulkan_fail_age_s": (time.time() - vulk) if vulk else None,
        "current_rpc_backend": caps.get("current_rpc_backend"),
        "vendor": p._worker_gpu_vendor(w),
        "selector_in_split": p._select_worker_backend(w, in_split=True),
        "gpus_present": bool(caps.get("gpus")),
        "gpu_kind": caps.get("gpu_kind"),
    }))
