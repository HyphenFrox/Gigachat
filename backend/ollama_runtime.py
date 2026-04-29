"""Auto-start + health-check for the Ollama daemon.

The frontend shows an "Ollama not reachable" toast the moment /api/models
returns without a `models` field, which means the user has to manually run
`ollama serve` in another terminal before anything works. That's friction
nobody should pay — Ollama is an *implementation detail* of Gigachat as far
as the end user is concerned.

This module fixes that by trying to bring Ollama up automatically whenever
the FastAPI backend starts:

  1. GET http://localhost:11434/api/tags with a short timeout. If it
     answers, Ollama is already running — nothing to do.
  2. Otherwise locate the `ollama` binary (shutil.which + a handful of
     OS-specific install paths Ollama's installer uses by default).
  3. Spawn `ollama serve` as a detached subprocess so it survives Gigachat
     exiting — users restart Gigachat all the time and shouldn't have
     Ollama rebooting + reloading models with them.
  4. Poll /api/tags for up to READY_TIMEOUT_SEC until it starts answering.

If any step fails, we log and return False. The existing UI toast handles
the user-facing message; this module is best-effort infrastructure.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

import httpx

log = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434"
# Short — this is called during startup and the user is blocked on it only
# indirectly (the UI can still render while we wait). But a non-responsive
# already-running instance should fail fast, not stall boot for a minute.
PROBE_TIMEOUT_SEC = 1.5
# Cold start time for Ollama on Windows is ~3-8 seconds on a fast SSD, but
# the first request after spawn triggers model-registry scanning which can
# add more. 30s is a generous ceiling before we give up and let the UI
# surface the "not reachable" toast.
READY_TIMEOUT_SEC = 30.0
# Poll interval while waiting for Ollama's HTTP surface to come up. Tighter
# than one second so a fast machine isn't sitting idle.
READY_POLL_SEC = 0.4


async def is_reachable() -> bool:
    """Return True if something answers /api/tags at :11434.

    Uses /api/tags rather than / because / is a static HTML page that any
    random server on that port could emit — /api/tags is an Ollama-specific
    JSON endpoint, so a 200 response is strong evidence Ollama is live.
    """
    try:
        async with httpx.AsyncClient(timeout=PROBE_TIMEOUT_SEC) as client:
            r = await client.get(f"{OLLAMA_URL}/api/tags")
        return r.status_code == 200
    except Exception:
        return False


def _windows_install_candidates() -> list[Path]:
    """Paths Ollama's Windows installer drops ollama.exe into.

    We check these in addition to PATH because Ollama's default Windows
    installer installs into the user's LocalAppData, which is NOT on PATH
    for a fresh-out-of-the-installer session until the user logs out/in.
    """
    candidates: list[Path] = []
    for env_var in ("LOCALAPPDATA", "PROGRAMFILES", "PROGRAMFILES(X86)"):
        base = os.environ.get(env_var)
        if not base:
            continue
        candidates.append(Path(base) / "Programs" / "Ollama" / "ollama.exe")
        candidates.append(Path(base) / "Ollama" / "ollama.exe")
    return candidates


def _posix_install_candidates() -> list[Path]:
    """Standard Unix install locations (Homebrew, the official installer)."""
    return [
        Path("/usr/local/bin/ollama"),
        Path("/usr/bin/ollama"),
        Path("/opt/homebrew/bin/ollama"),
        Path.home() / ".local" / "bin" / "ollama",
    ]


def find_ollama() -> str | None:
    """Locate the `ollama` executable. Returns an absolute path or None.

    Checks PATH first (covers users who installed via Homebrew, apt, or who
    manually added Ollama to PATH), then falls back to well-known installer
    locations per platform.
    """
    found = shutil.which("ollama")
    if found:
        return found
    candidates = _windows_install_candidates() if sys.platform == "win32" else _posix_install_candidates()
    for c in candidates:
        if c.is_file():
            return str(c)
    return None


def _recommend_ollama_num_parallel() -> int:
    """Pick a reasonable `OLLAMA_NUM_PARALLEL` based on host VRAM.

    Each parallel slot pre-allocates its own KV cache, so more slots
    means more VRAM consumed at model load. Tuned to match what
    `split_lifecycle._compute_optimal_parallel` would produce for the
    typical chat model on the same hardware:

      * <  6 GB VRAM (or no GPU)        → 1 slot   — tight, single-stream
      * 6 – 12 GB VRAM (4-7 B Q4 fits)  → 2 slots  — small headroom for one batched draft / subagent
      * 12 – 20 GB VRAM (8-13 B Q4)     → 4 slots  — comfortable for delegate_parallel
      * ≥ 20 GB VRAM (workstation tier) → 8 slots  — saturates compute

    Auto-adapts across NVIDIA / AMD / Intel iGPU / Apple Silicon /
    CPU-only because sysdetect normalizes all of them to a single
    `vram_gb` (for unified-memory / iGPU it reports the shared pool
    that's actually available to llama.cpp). Returning 1 on detection
    failure preserves the legacy single-slot behaviour.
    """
    try:
        from . import sysdetect
        spec = sysdetect.detect_system()
        vram_gb = float(spec.get("vram_gb") or 0)
    except Exception:
        return 1
    if vram_gb <= 0:
        return 1
    if vram_gb < 6:
        return 1
    if vram_gb < 12:
        return 2
    if vram_gb < 20:
        return 4
    return 8


def _spawn_ollama(executable: str) -> subprocess.Popen | None:
    """Launch `ollama serve` in a detached subprocess.

    We detach (CREATE_NEW_PROCESS_GROUP on Windows, setsid / start_new_session
    on POSIX) so that:
      - Ollama survives the Gigachat backend exiting cleanly (users who
        restart the app shouldn't evict model weights from RAM each time).
      - A Ctrl+C in the Gigachat terminal doesn't propagate to Ollama.

    Stdio is redirected to DEVNULL because we don't consume it and leaving
    it attached would fill OS pipe buffers over long sessions.

    Environment: we set `OLLAMA_NUM_PARALLEL` to a hardware-tuned value
    so concurrent chats (subagent fan-out, multiple browser tabs)
    actually batch in Ollama's runner instead of serializing on a
    single decoding slot. The user's existing env wins — we only set
    the var when it isn't already configured, so explicit overrides
    aren't clobbered.
    """
    env = os.environ.copy()
    if not env.get("OLLAMA_NUM_PARALLEL"):
        env["OLLAMA_NUM_PARALLEL"] = str(_recommend_ollama_num_parallel())
    # OLLAMA_KEEP_ALIVE controls how long models stay loaded after the
    # last request. Default is 5 minutes — generous for casual chat,
    # too short for long-form coding sessions where the user thinks
    # for 10+ minutes between turns. Bumping to 60 minutes keeps the
    # model warm across the typical "research → think → code" cycle
    # without consuming extra GPU memory beyond what was already
    # allocated. User-set value still wins.
    if not env.get("OLLAMA_KEEP_ALIVE"):
        env["OLLAMA_KEEP_ALIVE"] = "60m"
    # OLLAMA_NUM_THREAD overrides the runner's CPU thread count.
    # Default is logical core count which over-subscribes the FPU on
    # hyperthreaded CPUs — physical core count is typically faster
    # for AVX2 matmul. Same logic as split_lifecycle._recommend_thread_counts.
    if not env.get("OLLAMA_NUM_THREAD"):
        try:
            import psutil
            physical = psutil.cpu_count(logical=False)
            logical = psutil.cpu_count(logical=True)
            if physical and logical and logical > physical and physical > 1:
                env["OLLAMA_NUM_THREAD"] = str(min(32, int(physical)))
        except Exception:
            pass
    kwargs: dict = {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "close_fds": True,
        "env": env,
    }
    if sys.platform == "win32":
        # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP tells Windows to spawn
        # this without inheriting our console, so the user doesn't see a
        # black Ollama window pop up.
        DETACHED_PROCESS = 0x00000008
        CREATE_NEW_PROCESS_GROUP = 0x00000200
        CREATE_NO_WINDOW = 0x08000000
        kwargs["creationflags"] = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW
    else:
        kwargs["start_new_session"] = True
    try:
        return subprocess.Popen([executable, "serve"], **kwargs)
    except Exception as e:
        log.warning("ollama: spawn failed: %s", e)
        return None


async def _wait_until_ready(timeout: float = READY_TIMEOUT_SEC) -> bool:
    """Poll /api/tags until it responds or the timeout elapses."""
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if await is_reachable():
            return True
        await asyncio.sleep(READY_POLL_SEC)
    return False


async def ensure_running() -> dict:
    """Idempotent: start Ollama if it's not running, wait until healthy.

    Returns a small status dict the caller can log or surface in /api/system.
    Never raises — all failure modes collapse into {ok: False, reason: ...}.
    """
    if await is_reachable():
        return {"ok": True, "started": False, "reason": "already_running"}

    exe = find_ollama()
    if not exe:
        return {
            "ok": False,
            "started": False,
            "reason": "not_installed",
            "hint": "Install Ollama from https://ollama.com/download",
        }

    log.info("ollama: not running — auto-starting %s serve", exe)
    proc = _spawn_ollama(exe)
    if proc is None:
        return {"ok": False, "started": False, "reason": "spawn_failed"}

    if not await _wait_until_ready():
        # The subprocess is still detached — we don't kill it; maybe it needs
        # longer to come up, and the next /api/models call will succeed.
        return {"ok": False, "started": True, "reason": "ready_timeout"}

    log.info("ollama: serve is ready on %s", OLLAMA_URL)
    return {"ok": True, "started": True, "reason": "spawned"}


# ---------------------------------------------------------------------------
# Model catalog: listing and pulling
# ---------------------------------------------------------------------------
async def list_installed_models() -> list[str]:
    """Return tag names of every model Ollama currently has pulled.

    Empty list on any failure so callers can assume "nothing installed" and
    fall through to a pull — the common recovery for a cold environment.
    """
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{OLLAMA_URL}/api/tags")
            r.raise_for_status()
            data = r.json()
        return [m.get("name", "") for m in data.get("models", []) if m.get("name")]
    except Exception as e:
        log.warning("ollama: list_installed_models failed: %s", e)
        return []


async def pull_model(name: str, on_progress=None) -> dict:
    """Stream `ollama pull <name>` via /api/pull until complete.

    Ollama streams NDJSON progress events that look like:
      {"status":"pulling manifest"}
      {"status":"downloading digest sha256:…","total":1234,"completed":567}
      {"status":"success"}

    We optionally forward each event to `on_progress(event)` so the caller
    can log a progress bar. The pull itself can take MINUTES for a multi-GB
    model over a household connection, so the HTTP read timeout is None
    (let the stream run as long as it needs) but overall connection timeout
    stays short so a dead DNS doesn't hang us forever.
    """
    log.info("ollama: pulling model %r", name)
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, read=None)) as client:
            async with client.stream(
                "POST",
                f"{OLLAMA_URL}/api/pull",
                json={"name": name, "stream": True},
            ) as r:
                r.raise_for_status()
                last_status = ""
                async for line in r.aiter_lines():
                    if not line:
                        continue
                    try:
                        import json as _json
                        evt = _json.loads(line)
                    except Exception:
                        continue
                    if on_progress:
                        try:
                            on_progress(evt)
                        except Exception:
                            pass
                    status = evt.get("status") or ""
                    if status and status != last_status:
                        last_status = status
                        # Log only status transitions so we don't spam stdout
                        # with per-chunk download progress.
                        log.info("ollama pull %s: %s", name, status)
                    if evt.get("error"):
                        return {"ok": False, "error": evt["error"]}
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ---------------------------------------------------------------------------
# Auto-tune: pick and pull the right models for this hardware
# ---------------------------------------------------------------------------
# Recommendation stamp filled in by `auto_tune_models()` at startup and read
# by the /api/system/config route so the frontend defaults new conversations
# to the right model.
_RECOMMENDED: dict = {
    "chat_model": None,    # filled in at startup
    "embed_model": None,
    "pulling": False,
    "pull_status": "",
    "pull_error": "",
}


def get_recommendation() -> dict:
    """Snapshot of what the auto-tuner picked. Safe to read before startup —
    returns Nones until `auto_tune_models` has run at least once."""
    return dict(_RECOMMENDED)


async def auto_tune_models(disable_pull: bool = False) -> dict:
    """Pick optimal models for this hardware and pull any that are missing.

    Called during FastAPI startup after `ensure_running()` succeeds. Mutates
    the module-level `_RECOMMENDED` dict so the rest of the app can read
    "what should we default to" without re-probing the host.

    Pulls run in-band here: they can take several minutes but the rest of
    the server is already up (uvicorn treats startup hooks as concurrent),
    so the UI stays responsive. Progress is logged but the frontend also
    polls /api/system/config to surface a "pulling…" toast.

    If `disable_pull=True` (respects the GIGACHAT_NO_AUTO_PULL env var), we
    only record the recommendation; the user pulls manually. This is an
    escape hatch for metered connections.
    """
    # Late import — avoids a sysdetect → ollama_runtime cycle at module load.
    from . import sysdetect

    info = sysdetect.detect_system()
    installed = await list_installed_models()

    chat = sysdetect.recommend_chat_model(installed, info=info)
    embed = sysdetect.recommend_embed_model(installed)

    _RECOMMENDED["chat_model"] = chat["model"]
    _RECOMMENDED["embed_model"] = embed["model"]
    _RECOMMENDED["pull_error"] = ""

    if disable_pull or os.environ.get("GIGACHAT_NO_AUTO_PULL", "").strip() in ("1", "true", "yes"):
        _RECOMMENDED["pulling"] = False
        _RECOMMENDED["pull_status"] = "skipped (GIGACHAT_NO_AUTO_PULL set)"
        log.info(
            "ollama: auto-pull disabled via GIGACHAT_NO_AUTO_PULL. "
            "Recommended: chat=%s embed=%s",
            chat["model"],
            embed["model"],
        )
        return _RECOMMENDED

    # Figure out what actually needs to be fetched.
    to_pull: list[str] = []
    if chat["needs_pull"]:
        to_pull.append(chat["model"])
    if embed["needs_pull"]:
        to_pull.append(embed["model"])

    if not to_pull:
        _RECOMMENDED["pulling"] = False
        _RECOMMENDED["pull_status"] = "all models already installed"
        log.info(
            "ollama: auto-tune done. chat=%s embed=%s (no pulls needed)",
            chat["model"],
            embed["model"],
        )
        return _RECOMMENDED

    _RECOMMENDED["pulling"] = True
    _RECOMMENDED["pull_status"] = f"pulling {', '.join(to_pull)}"
    log.info(
        "ollama: auto-tune picked chat=%s embed=%s — pulling %s",
        chat["model"],
        embed["model"],
        to_pull,
    )

    errors: list[str] = []
    for m in to_pull:
        _RECOMMENDED["pull_status"] = f"pulling {m}"
        res = await pull_model(m)
        if not res.get("ok"):
            errors.append(f"{m}: {res.get('error', 'unknown error')}")

    _RECOMMENDED["pulling"] = False
    if errors:
        _RECOMMENDED["pull_status"] = "partial failure"
        _RECOMMENDED["pull_error"] = "; ".join(errors)
        log.warning("ollama: auto-pull errors: %s", errors)
    else:
        _RECOMMENDED["pull_status"] = "ready"

    return _RECOMMENDED


async def startup_autotune_background() -> None:
    """Fire-and-forget entry point for FastAPI's startup hook.

    We spawn the auto-tune as a background task rather than awaiting it
    synchronously so uvicorn finishes startup (serves /api/*) while a first-
    time user's multi-GB model pull runs. The UI reads the rolling status
    from /api/system/config.
    """
    try:
        await auto_tune_models()
    except Exception as e:
        log.warning("ollama: auto-tune crashed: %s", e)
        _RECOMMENDED["pulling"] = False
        _RECOMMENDED["pull_error"] = str(e)
