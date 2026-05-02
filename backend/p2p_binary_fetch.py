"""LAN-first binary fetcher: pull missing llama-cpp DLLs / EXEs from
a paired peer that already has them, instead of going to the
internet.

Why this exists
===============
Naresh's box was missing ``ggml-sycl.dll`` and ``ggml-rpc.dll`` — the
two DLLs llama.cpp's rpc-server needs to actually accept layer
pushes. Without them, ``rpc-server -d SYCL0,CPU`` exits with
"Failed to find RPC backend". The user had to scp them by hand.

A peer-to-peer install path means:
  * Zero internet bandwidth — same-LAN GB-class transfers in
    seconds, not minutes/hours.
  * No dependency on a release URL that could 404 or be rate-
    limited.
  * The peer that already has the binary is the most authoritative
    source for "the version that works on this machine class" —
    saves a quirky version mismatch.

API surface
===========
* ``GET /api/p2p/binary/list`` — returns the file list this peer
  currently has in its llama-cpp install dir, with sha256 + size.
* ``GET /api/p2p/binary/get/<filename>`` — streams the file body.

Both are whitelisted in the encrypted-proxy and routed to local
Gigachat (not Ollama). The streaming variant uses the existing
``serve_forward_stream`` envelope chain.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path

log = logging.getLogger(__name__)


# Where llama.cpp lives on this install. Same path as
# ``compute_pool._RPC_SERVER_BIN_DIR`` and
# ``p2p_rpc_server._RPC_SERVER_BIN_DIR``.
_LLAMA_CPP_DIR = Path.home() / ".gigachat" / "llama-cpp"

# Maximum file size we'll ever ship. 1 GB ceiling defends against
# accidentally streaming a huge GGUF model through this endpoint.
# DLLs and EXEs are well under this; the largest we ship is
# ggml-cuda.dll at ~480 MB.
_MAX_TRANSFERABLE_BYTES = 1024 * 1024 * 1024

# Whitelist of file extensions safe to ship. Refusing anything else
# means a malicious request can't extract source files or DBs even
# if the path-traversal guards below are bypassed somehow.
_TRANSFERABLE_EXTS = frozenset({".dll", ".exe", ".so", ".dylib"})


def list_local_binaries() -> dict:
    """Snapshot of what's in our llama-cpp install dir.

    Returns ``{"dir": str, "files": [{"name", "size", "sha256"}, ...]}``.
    Skips anything outside the transferable-extension whitelist so
    the response stays focused on what a peer might want to fetch.
    """
    out: dict = {"dir": str(_LLAMA_CPP_DIR), "files": []}
    if not _LLAMA_CPP_DIR.is_dir():
        return out
    files = []
    for p in _LLAMA_CPP_DIR.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in _TRANSFERABLE_EXTS:
            continue
        try:
            size = p.stat().st_size
        except OSError:
            continue
        if size > _MAX_TRANSFERABLE_BYTES:
            continue
        files.append({
            "name": p.name,
            "size": size,
            # sha256 of the file is cheap on small DLLs and lets the
            # caller verify integrity end-to-end (the encrypted
            # envelope already protects in transit, but the hash
            # also lets the caller pick a peer whose copy matches a
            # known-good build).
            "sha256": _file_sha256(p),
        })
    files.sort(key=lambda f: f["name"])
    out["files"] = files
    return out


def _file_sha256(p: Path) -> str:
    """Streaming sha256 — won't OOM on multi-hundred-MB DLLs."""
    h = hashlib.sha256()
    try:
        with p.open("rb") as f:
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                h.update(chunk)
    except OSError:
        return ""
    return h.hexdigest()


def open_local_binary(filename: str) -> tuple[Path, int] | None:
    """Resolve ``filename`` to a real file inside our llama-cpp dir,
    return ``(path, size)``. Returns None if the file isn't present
    OR isn't transferable (wrong extension, too big, traversal
    attempt, etc.).

    Intentionally strict path resolution — we accept only the
    basename, never any directory components. A request for
    ``../app.db`` becomes a lookup for ``..`` inside our llama-cpp
    dir and fails harmlessly.
    """
    if not filename:
        return None
    safe_name = Path(filename).name
    if safe_name != filename:
        # Caller tried to pass directory components.
        return None
    p = _LLAMA_CPP_DIR / safe_name
    if not p.is_file():
        return None
    if p.suffix.lower() not in _TRANSFERABLE_EXTS:
        return None
    try:
        size = p.stat().st_size
    except OSError:
        return None
    if size > _MAX_TRANSFERABLE_BYTES:
        return None
    # Final sanity: resolve and check it's still inside our dir
    # (defends against symlinks pointing outside).
    try:
        if _LLAMA_CPP_DIR.resolve() not in p.resolve().parents:
            return None
    except OSError:
        return None
    return p, size
