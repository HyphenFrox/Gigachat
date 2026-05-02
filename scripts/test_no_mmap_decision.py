"""Quick: print the no-mmap decision for various model sizes."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend import db, split_lifecycle
db.init()
wids = [w["id"] for w in db.list_compute_workers() if w.get("enabled")]
for size_gb, name in [
    (26.44, "dolphin-mixtral:8x7b"),
    (19.87, "gemma4:31b"),
    (9.61, "gemma4:e4b"),
    (4.92, "llama3.1:8b"),
    (1.6, "dolphin-phi"),
]:
    nm = split_lifecycle._should_disable_mmap(int(size_gb * 1e9), wids)
    print(f"  {name:<25s} ({size_gb:>5.2f} GB)  no_mmap={nm}")
