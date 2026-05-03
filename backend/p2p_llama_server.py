"""Local llama-server lifecycle, spawned by a paired peer over the
encrypted P2P channel.

Why this exists
===============
When a chat model is too big for any single peer's free RAM but a
paired LAN peer already has the GGUF on disk, the right architecture
is "peer-orchestrated split": the peer with the GGUF runs llama-server
locally and uses one or more remote `--rpc` backends (typically the
orchestrator host's rpc-server) to fan layer compute across the LAN.

* No GGUF transfer — the peer keeps its existing copy.
* No internet pull — model bytes never re-downloaded.
* No host disk pressure — large MoE / dense models that wouldn't fit
  the orchestrator's disk run anyway because they live on the peer.

Verified case: dolphin-mixtral:8x7b (26.4 GB) lives on FBS (16.6 GB
free, can't fit alone in plain Ollama). Host has 8 GB CUDA + 32 GB
RAM and rpc-server already listening on 50052. This module lets the
orchestrator drive FBS to start its own llama-server with
`--rpc <host>:50052` so layers fan across both nodes — model runs
without any data movement.

API surface
===========
Three functions, one for each lifecycle verb:
* ``start_local_llama_server(...)`` — idempotent spawn keyed by port.
* ``get_local_llama_server_status(port)`` — snapshot for orchestrator.
* ``stop_local_llama_server(port=None)`` — kill scoped to one port
  or every llama-server we own.

Authentication
==============
The matching `/api/p2p/llama-server/start|status|stop` endpoints are
whitelisted in ``p2p_secure_proxy._FORWARDABLE_PATHS`` and routed to
local Gigachat (not Ollama) via ``_GIGACHAT_INTERNAL_PATHS``. So the
only callers are (a) loopback, (b) a paired peer whose envelope
passes X25519+ChaCha20+Ed25519 verification.

Scope
=====
Spawn / status / stop only. Model resolution is handled here (we
look at the local Ollama blob store), but layer placement and
adaptive rebalance live in ``split_lifecycle`` on the orchestrator.
This module just makes sure the llama-server process is running
locally with the right `--model` and `--rpc` flags.
"""
from __future__ import annotations

import json
import logging
import os
import platform
import socket
import subprocess
import time
from pathlib import Path

log = logging.getLogger(__name__)


# Where llama-server.exe lives on this install. Same path as
# ``p2p_rpc_server._RPC_SERVER_BIN_DIR`` — the `p2p_binary_fetch`
# auto-installer drops everything llama.cpp into one directory, so
# both binaries are siblings.
_LLAMA_SERVER_BIN_DIR = Path.home() / ".gigachat" / "llama-cpp"
_LLAMA_SERVER_EXE = _LLAMA_SERVER_BIN_DIR / (
    "llama-server.exe" if platform.system() == "Windows" else "llama-server"
)

# Default port we bind llama-server to when the orchestrator doesn't
# specify one. Chosen above the rpc-server pair (50052/50053) so a
# single host can run both rpc-server (so other peers use it as a
# backend) AND llama-server (so it serves chats locally) without
# port conflict.
_DEFAULT_PORT = 8090

# Map (port -> {model, pid, started_at}) so the orchestrator can ask
# "what's running on port 8090?" and we can short-circuit
# already-running spawns. Module-level state, mirrored across the
# spawn / status / stop verbs.
_active_servers: dict[int, dict] = {}

# Stability env vars — same set p2p_rpc_server uses. Intel Xe2 / Meteor
# Lake SYCL paths are sensitive to upstream bugs; these knobs prevent
# silent weight corruption and warmup crashes.
_SPAWN_ENV: dict[str, str] = {
    "GGML_SYCL_DISABLE_OPT": "1",
    "GGML_SYCL_DISABLE_GRAPH": "1",
    "SYCL_CACHE_PERSISTENT": "1",
    # llama-server reads this to find DLLs / drivers when run from a
    # detached process whose CWD isn't the binary dir on every Windows
    # release. Belt-and-suspenders alongside subprocess `cwd=`.
    "PATH": str(_LLAMA_SERVER_BIN_DIR) + os.pathsep + os.environ.get("PATH", ""),
}

# How long to wait for the freshly-spawned llama-server to start
# listening on its TCP port. llama-server's startup includes loading
# the GGUF (mmap is fast) AND initializing all `--rpc` backends
# (slow — TCP handshake to each remote rpc-server, then their model-
# weight upload). For a 26 GB model with 2 RPC backends and SYCL JIT
# warmup, 60 s is the realistic cold-start budget.
_LISTEN_WAIT_SEC = 60.0


def _is_listening_on(port: int) -> bool:
    """Return True iff TCP port `port` has at least one listener."""
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.25):
            return True
    except OSError:
        return False


def _kill_running_llama_servers(only_listening_on_port: int | None = None) -> int:
    """Kill llama-server processes owned by this user.

    `only_listening_on_port`: when set, only kill the llama-server
    whose listener includes that port (sibling llama-servers on
    other ports keep running). When None, kills every llama-server
    we own — used by full-shutdown / unpair paths.

    Returns the count killed. Best-effort: psutil-based, silent
    on per-process errors.
    """
    killed = 0
    try:
        import psutil
    except ImportError:
        log.debug("p2p_llama_server: psutil missing, can't enumerate processes")
        return 0
    me = os.getlogin() if hasattr(os, "getlogin") else ""
    for p in psutil.process_iter(["pid", "name", "username"]):
        try:
            name = (p.info.get("name") or "").lower()
            if not name.startswith("llama-server"):
                continue
            if me and (p.info.get("username") or "").lower().split("\\")[-1] != me.lower():
                continue
            if only_listening_on_port is not None:
                try:
                    proc = psutil.Process(p.info["pid"])
                    ports = {
                        c.laddr.port for c in proc.net_connections(kind="inet")
                        if c.status == psutil.CONN_LISTEN
                    }
                    if only_listening_on_port not in ports:
                        continue
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            p.terminate()
            killed += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    if killed:
        # Give terminate a moment to settle so the next bind() of the
        # same port doesn't race a half-dead process.
        time.sleep(0.8)
    return killed


def _spawn_detached(cmd: list[str], cwd: str, env: dict[str, str]) -> int:
    """Spawn `cmd` so it survives this process exiting.

    Same detachment dance ``p2p_rpc_server._spawn_detached`` uses.
    stdout + stderr go into ``llama-server.log`` next to the binary
    so the orchestrator can diagnose load failures (e.g. "model file
    not found", "RPC backend connect failed") that wouldn't surface
    via the API otherwise.
    """
    log_path = _LLAMA_SERVER_BIN_DIR / "llama-server.log"
    try:
        _LLAMA_SERVER_BIN_DIR.mkdir(parents=True, exist_ok=True)
        log_f = open(log_path, "ab")
    except OSError:
        log_f = subprocess.DEVNULL  # type: ignore[assignment]
    if platform.system() == "Windows":
        DETACHED_PROCESS = 0x00000008
        CREATE_NEW_PROCESS_GROUP = 0x00000200
        CREATE_BREAKAWAY_FROM_JOB = 0x01000000
        flags = (
            DETACHED_PROCESS
            | CREATE_NEW_PROCESS_GROUP
            | CREATE_BREAKAWAY_FROM_JOB
        )
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            creationflags=flags,
            close_fds=True,
        )
        return proc.pid
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        close_fds=True,
    )
    return proc.pid


def _resolve_local_ollama_gguf(model_name: str) -> str | None:
    """Resolve a model name to its on-disk GGUF path via the local
    Ollama manifest store.

    Mirrors the resolution logic embedded in
    ``compute_pool.ensure_worker_chat_server``'s SSH PowerShell payload
    — but runs locally so we don't need PowerShell-via-SSH. Returns
    None when the model isn't installed in the local Ollama.

    Why we don't reuse ``compute_pool.resolve_ollama_model`` here:
    that function calls Ollama's HTTP /api/show endpoint, which on
    Windows includes a Modelfile parse step that fails for some
    older models. Reading the manifest directly is one fewer moving
    part — purely a filesystem walk.
    """
    base = Path.home() / ".ollama" / "models"
    if ":" in model_name:
        ns_model, tag = model_name.rsplit(":", 1)
        if "/" in ns_model:
            ns, model = ns_model.split("/", 1)
        else:
            ns, model = "library", ns_model
    else:
        ns, model, tag = "library", model_name, "latest"
    manifest = base / "manifests" / "registry.ollama.ai" / ns / model / tag
    if not manifest.exists():
        return None
    try:
        m = json.loads(manifest.read_text())
    except Exception as e:
        log.warning(
            "p2p_llama_server: failed to parse manifest %s: %s",
            manifest, e,
        )
        return None
    digest = None
    for layer in m.get("layers", []):
        if layer.get("mediaType") == "application/vnd.ollama.image.model":
            digest = layer.get("digest")
            break
    if not digest:
        return None
    blob = base / "blobs" / digest.replace(":", "-")
    return str(blob) if blob.exists() else None


def start_local_llama_server(
    *,
    model: str,
    port: int = _DEFAULT_PORT,
    rpc_targets: list[str] | None = None,
    n_gpu_layers: int = 99,
    context_size: int = 4096,
    parallel: int = 1,
) -> dict:
    """Bring up llama-server locally, idempotent w.r.t. (port, model).

    Args:
        model: Ollama model name (e.g. "dolphin-mixtral:8x7b"). We
            resolve to its on-disk GGUF via the local Ollama manifest
            store. Required.
        port: TCP port to bind. Defaults to 8090. Caller can pick a
            different port if 8090 is in use; we'll happily run
            multiple llama-servers concurrently on different ports.
        rpc_targets: Optional list of "host:port" strings for
            additional `--rpc` backends. Each target is a remote
            rpc-server we'll fan layers to. Empty list means
            single-node mode (this peer's GPU + CPU only).
        n_gpu_layers: How many layers to request on this peer's GPU
            (orchestrator can override based on its capacity model;
            99 means "as many as fit").
        context_size: --ctx-size value. 4096 is a safe default for
            chat; the orchestrator may bump to 8192 / 16384 for
            longer-context models.
        parallel: --parallel slot count for continuous batching.
            1 = single chat, no batching overhead. The orchestrator
            sets higher values when multiple subagents share the
            same llama-server.

    Returns:
        ``{"ok": bool, "port": int, "url": str, "pid": int|None,
           "model_path": str|None, "status": str, "rpc_targets":
           list[str], "log_path": str}``

        Status values:
        * ``already_running`` — port had this exact (model,
          rpc_targets) running; no spawn.
        * ``started`` — spawned, listening verified.
        * ``no_binary`` — llama-server.exe not installed.
        * ``model_not_found`` — couldn't resolve model GGUF locally.
        * ``spawn_failed`` — Popen raised; error in payload.
        * ``spawned_but_not_listening`` — process started but never
          opened the port; likely RPC connect or model-load failure
          (check log_path).
    """
    rpc_targets = list(rpc_targets or [])
    out: dict = {
        "ok": False,
        "binary_path": str(_LLAMA_SERVER_EXE),
        "model": model,
        "port": port,
        "rpc_targets": rpc_targets,
        "log_path": str(_LLAMA_SERVER_BIN_DIR / "llama-server.log"),
    }

    # Already running with the exact same model+rpc_targets on this
    # port? Re-verify TCP listener and short-circuit. Saves a
    # multi-second restart on every chat turn that re-routes here.
    current = _active_servers.get(port)
    if (
        current
        and current.get("model") == model
        and current.get("rpc_targets") == rpc_targets
        and _is_listening_on(port)
    ):
        out["ok"] = True
        out["status"] = "already_running"
        out["pid"] = current.get("pid")
        out["model_path"] = current.get("model_path")
        out["url"] = f"http://0.0.0.0:{port}"
        return out

    if not _LLAMA_SERVER_EXE.is_file():
        out["status"] = "no_binary"
        out["error"] = (
            f"llama-server binary missing at {_LLAMA_SERVER_EXE}; "
            "install llama.cpp on this peer (e.g. via the binary "
            "auto-fetch path or download from the upstream releases)."
        )
        return out

    # Resolve the model to an on-disk GGUF before spending the
    # process-spawn cycle. A fast-fail here saves 60 s of waiting
    # for a listener that'll never come up.
    model_path = _resolve_local_ollama_gguf(model)
    if not model_path:
        out["status"] = "model_not_found"
        out["error"] = (
            f"model {model!r} not found in local Ollama store. "
            "Pull it on this peer first (`ollama pull <model>`)."
        )
        return out
    out["model_path"] = model_path

    # Different model / different rpc_targets / not listening — kill
    # whatever's on this port and respawn fresh.
    killed = _kill_running_llama_servers(only_listening_on_port=port)
    out["killed_stale"] = killed
    _active_servers.pop(port, None)

    # Build the spawn command. Match the flags
    # ``compute_pool.ensure_worker_chat_server`` uses on host so
    # behaviour is consistent across peer-led and host-led splits:
    #   * --flash-attn on + KV q8_0 → halve KV memory at <1 % accuracy
    #     loss, frees room for layer weights.
    #   * --cache-reuse 256 → multi-turn chat reuses prefix KV.
    #   * --jinja → prompt template rendering for chat-completion API.
    #   * --no-warmup → skip the dummy forward pass; the first real
    #     request warms the engine implicitly. Saves 5-10 s of
    #     unnecessary GPU work on cold start.
    cmd = [
        str(_LLAMA_SERVER_EXE),
        "--model", model_path,
        "--host", "0.0.0.0",
        "--port", str(port),
        "--flash-attn", "on",
        "-ctk", "q8_0", "-ctv", "q8_0",
        "--cache-reuse", "256",
        "--jinja",
        "-ngl", str(n_gpu_layers),
        "-c", str(context_size),
        "--no-warmup",
        "--parallel", str(parallel),
    ]
    if rpc_targets:
        cmd += ["--rpc", ",".join(rpc_targets)]

    env = dict(os.environ)
    env.update(_SPAWN_ENV)
    try:
        pid = _spawn_detached(
            cmd, cwd=str(_LLAMA_SERVER_BIN_DIR), env=env,
        )
    except Exception as e:
        out["status"] = "spawn_failed"
        out["error"] = f"{type(e).__name__}: {e}"
        return out

    out["pid"] = pid

    # Wait for the listener. llama-server with `--rpc` backends takes
    # 30-60 s to come up: TCP handshake + Ollama-style metadata exchange
    # + initial layer-weight upload to each rpc-server. Worth the
    # blocking wait so the orchestrator gets a definitive
    # "ready / failed" signal in one round trip.
    deadline = time.time() + _LISTEN_WAIT_SEC
    listening = False
    while time.time() < deadline:
        if _is_listening_on(port):
            listening = True
            break
        time.sleep(0.5)

    if listening:
        out["ok"] = True
        out["status"] = "started"
        out["url"] = f"http://0.0.0.0:{port}"
        _active_servers[port] = {
            "model": model,
            "rpc_targets": rpc_targets,
            "pid": pid,
            "model_path": model_path,
            "started_at": time.time(),
        }
    else:
        out["status"] = "spawned_but_not_listening"
        out["error"] = (
            f"llama-server PID {pid} did not start listening on "
            f"127.0.0.1:{port} within {_LISTEN_WAIT_SEC}s. "
            f"Check {out['log_path']!r} for the spawn-side error — "
            "common causes: rpc-server not reachable on a target, "
            "model file missing on this peer, or out-of-memory "
            "during weight load."
        )
    return out


def get_local_llama_server_status(port: int = _DEFAULT_PORT) -> dict:
    """Snapshot of the local llama-server's state on `port`.

    Returns whether the port is listening, what model+rpc_targets we
    spawned (if known), and the PID. Used by orchestrators to decide
    whether to skip a redundant spawn.
    """
    state = _active_servers.get(port)
    return {
        "binary_path": str(_LLAMA_SERVER_EXE),
        "binary_present": _LLAMA_SERVER_EXE.is_file(),
        "port": port,
        "listening": _is_listening_on(port),
        "active_model": (state or {}).get("model"),
        "active_rpc_targets": (state or {}).get("rpc_targets"),
        "active_pid": (state or {}).get("pid"),
        "started_at": (state or {}).get("started_at"),
        "active_servers": dict(_active_servers),
    }


def stop_local_llama_server(port: int | None = None) -> dict:
    """Kill llama-server process(es).

    `port`: scope kill to one port (siblings on other ports keep
    running). When None, kills every llama-server we own.
    """
    if port is not None:
        killed = _kill_running_llama_servers(only_listening_on_port=port)
        _active_servers.pop(port, None)
        return {
            "killed": killed,
            "port": port,
            "listening": _is_listening_on(port),
        }
    killed = _kill_running_llama_servers()
    _active_servers.clear()
    return {"killed": killed, "listening": _is_listening_on(_DEFAULT_PORT)}
