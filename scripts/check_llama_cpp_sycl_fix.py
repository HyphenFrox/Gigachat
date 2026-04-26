"""Periodic upstream-watch for the llama.cpp SYCL+RPC bug fix.

The current build (`b8934`) crashes when pushing tensors over RPC to
a worker's SYCL backend. We work around it by switching workers to
`-d CPU` during split (commits 35-36 / 38). When upstream fixes the
bug, we want to bump `LLAMA_CPP_VERSION` and reclaim worker iGPU
acceleration in split mode.

This script:
  1. Reads the current `LLAMA_CPP_VERSION` from `backend/split_runtime.py`.
  2. Queries GitHub Releases API for `ggml-org/llama.cpp` to find
     newer builds.
  3. Greps each newer build's release notes / closed-issue references
     for any of the bug-tracking issues (#21420, #20259, #21474,
     #21006, #21030).
  4. Reports findings: "newer build X available — bug Y referenced
     as fixed; consider bumping" or "no fix yet".

Run manually (sanity check) or on a recurring schedule (cron / Task
Scheduler / Claude scheduled task). It does NOT auto-bump — surfaces
the recommendation so the user can vet the new build first.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import httpx


REPO = "ggml-org/llama.cpp"
BUG_ISSUES = ("#21420", "#20259", "#21474", "#21006", "#21030", "#21893")


def _read_current_version() -> str | None:
    """Read `LLAMA_CPP_VERSION` from `backend/split_runtime.py`."""
    sr = Path(__file__).resolve().parent.parent / "backend" / "split_runtime.py"
    try:
        for line in sr.read_text(encoding="utf-8").splitlines():
            m = re.match(r'\s*LLAMA_CPP_VERSION\s*=\s*"(b\d+)"', line)
            if m:
                return m.group(1)
    except Exception:
        pass
    return None


def _newer_releases(current: str) -> list[dict]:
    """List GitHub releases of ggml-org/llama.cpp newer than `current`,
    most recent first. Up to 30 newest entries scanned (covers
    ~3 weeks of daily releases).
    """
    cur_n = int(current.lstrip("b")) if current.startswith("b") else 0
    r = httpx.get(
        f"https://api.github.com/repos/{REPO}/releases?per_page=30",
        timeout=30.0, follow_redirects=True,
    )
    r.raise_for_status()
    out = []
    for rel in r.json() or []:
        tag = rel.get("tag_name") or ""
        if not tag.startswith("b"):
            continue
        try:
            n = int(tag[1:])
        except ValueError:
            continue
        if n > cur_n:
            out.append({
                "tag": tag,
                "n": n,
                "name": rel.get("name") or "",
                "body": rel.get("body") or "",
                "url": rel.get("html_url") or "",
                "published_at": rel.get("published_at") or "",
            })
    out.sort(key=lambda x: x["n"])
    return out


def _scan_release_for_fix(rel: dict) -> list[str]:
    """Return list of bug-issue refs mentioned in this release's body
    (likely indicating a fix). Empty list if none referenced.
    """
    body = rel.get("body") or ""
    return [issue for issue in BUG_ISSUES if issue in body]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--quiet", action="store_true",
        help="Only print when a likely fix is detected; otherwise silent.",
    )
    args = parser.parse_args()

    current = _read_current_version()
    if not current:
        print("ERROR: could not read LLAMA_CPP_VERSION from split_runtime.py",
              file=sys.stderr)
        return 2

    if not args.quiet:
        print(f"Current: {current}")

    try:
        newer = _newer_releases(current)
    except Exception as e:
        print(f"WARNING: GitHub API call failed: {e}", file=sys.stderr)
        return 1

    if not newer:
        if not args.quiet:
            print(f"No newer builds since {current}.")
        return 0

    # Walk newer releases looking for any that reference bug fixes.
    fixed_in: list[tuple[str, list[str]]] = []
    for rel in newer:
        refs = _scan_release_for_fix(rel)
        if refs:
            fixed_in.append((rel["tag"], refs))

    if fixed_in:
        print(f"POTENTIAL FIX detected since {current}:")
        for tag, refs in fixed_in:
            print(f"  {tag}: references {', '.join(refs)}")
        latest = newer[-1]["tag"]
        print()
        print(f"Recommend bumping LLAMA_CPP_VERSION to {latest} and re-testing")
        print("the SYCL+RPC split path. If it works, drop the dynamic backend")
        print("workaround in compute_pool._select_worker_backend.")
        return 0

    if not args.quiet:
        print(f"No bug-fix references in {len(newer)} newer build(s) "
              f"({newer[0]['tag']} -> {newer[-1]['tag']}).")
        print("Workaround still needed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
