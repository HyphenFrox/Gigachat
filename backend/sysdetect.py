"""Hardware detection + context-window auto-tuning.

The single job of this module is to figure out how much room the host
actually has so the rest of the app can pick sensible defaults without the
user manually editing constants every time they move between a laptop and
a desktop.

It answers three questions:
  1. How much system RAM is installed?
  2. Is there a CUDA GPU, and how much VRAM does it have?
  3. Given those, what `num_ctx` should Ollama use?

All detection is best-effort — any probe that fails is skipped silently and
the recommendation falls back to the CPU tier. We never raise; a broken
probe must never keep the backend from starting.

Override: setting the environment variable `MM_NUM_CTX` forces a specific
value (useful for debugging / specialty hardware where the auto-tune misses).
"""

from __future__ import annotations

import os
import subprocess
import sys
from functools import lru_cache

try:
    import psutil  # type: ignore
    _HAVE_PSUTIL = True
except ImportError:  # pragma: no cover — psutil is listed in requirements.txt
    _HAVE_PSUTIL = False


# ---------------------------------------------------------------------------
# RAM
# ---------------------------------------------------------------------------
def _detect_ram_gb() -> float:
    """Return total physical RAM in GB, or 0.0 if detection fails."""
    if not _HAVE_PSUTIL:
        return 0.0
    try:
        return psutil.virtual_memory().total / (1024 ** 3)
    except Exception:
        return 0.0


def _detect_cpu_count() -> int:
    """Logical CPU count (includes hyperthreads). 0 on failure."""
    try:
        return os.cpu_count() or 0
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# GPU (NVIDIA only, via nvidia-smi — no CUDA runtime or torch dependency)
#
# We shell out to nvidia-smi because it's always present on machines that
# have the NVIDIA driver installed and needs no extra Python deps. On AMD /
# Intel / no-GPU systems the command is missing or errors out, and we fall
# through to 0 VRAM — the CPU tier in `recommend_num_ctx` handles the rest.
# ---------------------------------------------------------------------------
def _detect_nvidia_gpu() -> tuple[float, str]:
    """Return (vram_gb, gpu_name). Both zeroed out if no NVIDIA GPU found."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total,name",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=3.0,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return 0.0, ""
    if result.returncode != 0:
        return 0.0, ""
    # First GPU only — multi-GPU users can override via MM_NUM_CTX if needed.
    first_line = (result.stdout or "").strip().splitlines()
    if not first_line:
        return 0.0, ""
    parts = [p.strip() for p in first_line[0].split(",", 1)]
    if not parts or not parts[0].isdigit():
        return 0.0, ""
    vram_mb = int(parts[0])
    name = parts[1] if len(parts) > 1 else ""
    return vram_mb / 1024.0, name


def _detect_amd_gpu() -> tuple[float, str]:
    """Return (vram_gb, gpu_name) for an AMD GPU via rocm-smi, or (0,"")."""
    try:
        # rocm-smi --showmeminfo vram --showproductname -> JSON via --json on
        # recent releases; older builds print text. We use text + parsing
        # because --json is not universally available.
        result = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram", "--showproductname"],
            capture_output=True,
            text=True,
            timeout=3.0,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return 0.0, ""
    if result.returncode != 0:
        return 0.0, ""
    out = result.stdout or ""
    vram_bytes = 0
    name = ""
    # rocm-smi line shapes: "GPU[0] : vram Total Memory (B): 17163091968"
    # and "GPU[0] : Card Series: Radeon RX 7900 XT"
    for line in out.splitlines():
        low = line.lower()
        if "vram total memory" in low:
            tail = line.rsplit(":", 1)[-1].strip()
            if tail.isdigit():
                vram_bytes = max(vram_bytes, int(tail))
        elif "card series" in low or "card model" in low:
            if not name:
                name = line.rsplit(":", 1)[-1].strip()
    if vram_bytes == 0:
        return 0.0, ""
    return vram_bytes / (1024 ** 3), name or "AMD GPU"


def _detect_intel_gpu() -> tuple[float, str]:
    """Return (vram_gb, gpu_name) for an Intel GPU (iGPU or Arc dGPU)
    on Windows via WMI, else (0, "").

    nvidia-smi and rocm-smi don't see Intel hardware. WMI's
    Win32_VideoController class is the same probe the worker-side
    capability check uses, so the host's Intel detection is
    consistent with what we report for workers.

    For iGPUs the reported `AdapterRAM` is the BIOS-allocated shared
    memory pool (typically 128 MB - 1 GB), which understates the
    actual usable memory because Intel iGPUs dynamically grow into
    system RAM. Treat the reported number as the conservative
    floor — `recommend_num_ctx` already falls back to RAM tiers
    when VRAM is small.
    """
    if sys.platform != "win32":
        return 0.0, ""
    # Using PowerShell + Get-CimInstance avoids pulling the heavy
    # `wmi` Python package into requirements.txt for one-off probes.
    try:
        result = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                # Filter to Intel adapters; emit "<ram>|<name>" so we
                # parse without invoking JSON for one row.
                "Get-CimInstance Win32_VideoController "
                "| Where-Object { $_.Name -match 'Intel' } "
                "| Select-Object -First 1 "
                "| ForEach-Object { \"$($_.AdapterRAM)|$($_.Name)\" }",
            ],
            capture_output=True,
            text=True,
            timeout=4.0,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return 0.0, ""
    if result.returncode != 0:
        return 0.0, ""
    line = (result.stdout or "").strip().splitlines()
    if not line:
        return 0.0, ""
    parts = line[0].split("|", 1)
    if not parts or not parts[0].strip().isdigit():
        return 0.0, ""
    vram_bytes = int(parts[0].strip())
    name = parts[1].strip() if len(parts) > 1 else "Intel GPU"
    if vram_bytes <= 0:
        return 0.0, ""
    return vram_bytes / (1024 ** 3), name


def _detect_apple_silicon() -> tuple[float, str]:
    """Return (unified_ram_gb, chip_name) on Apple Silicon Macs, else (0,"").

    Apple Silicon exposes unified memory rather than dedicated VRAM — the GPU
    can address the same RAM the CPU uses. For our purposes that means the
    whole system RAM is effectively available to Ollama as "VRAM", so we
    surface it that way.
    """
    if sys.platform != "darwin":
        return 0.0, ""
    try:
        import platform
        if platform.machine() != "arm64":
            return 0.0, ""
        ram = _detect_ram_gb()
        chip = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            timeout=2.0,
        )
        name = (chip.stdout or "").strip() or "Apple Silicon"
        return ram, name
    except Exception:
        return 0.0, ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def detect_system() -> dict:
    """Probe the host and return a dict describing what we found.

    Cached so repeated calls across the codebase (startup, /api/system/config)
    don't re-shell-out to nvidia-smi every time. Hardware doesn't change mid
    process, so one probe at import time is plenty.

    GPU detection order:
      1. NVIDIA (nvidia-smi)            — most common on Windows + Linux desktops
      2. AMD (rocm-smi)                  — ROCm-capable Radeons
      3. Apple Silicon (sysctl)          — arm64 macOS unified memory
      4. Intel (Win32_VideoController)   — iGPUs + Arc dGPUs on Windows

    Sequential-with-early-exit is faster than parallel here because the
    Intel WMI probe is dramatically slower than the others (~700 ms
    spawning PowerShell on Windows). Running it last means a Windows
    NVIDIA / AMD host pays only the cheap probe; only Intel-only
    hosts pay the slow one. Parallel execution would regress NVIDIA
    Windows boots by always paying the Intel-WMI cost.

    The first one that reports nonzero memory wins. `gpu_kind` ∈
    {"nvidia", "amd", "apple", "intel", ""}; the empty string means
    no usable GPU detected (CPU-only host).
    """
    ram_gb = _detect_ram_gb()
    # Try each backend in turn. Apple Silicon is Mac-only and free (no shell
    # out), so we always try it — but _detect_apple_silicon returns 0 unless
    # we're actually on arm64 macOS.
    vram_gb, gpu_name = _detect_nvidia_gpu()
    gpu_kind = "nvidia" if vram_gb > 0 else ""
    if vram_gb == 0:
        vram_gb, gpu_name = _detect_amd_gpu()
        if vram_gb > 0:
            gpu_kind = "amd"
    if vram_gb == 0:
        vram_gb, gpu_name = _detect_apple_silicon()
        if vram_gb > 0:
            gpu_kind = "apple"
    if vram_gb == 0:
        vram_gb, gpu_name = _detect_intel_gpu()
        if vram_gb > 0:
            gpu_kind = "intel"
    return {
        "ram_gb": round(ram_gb, 1),
        "vram_gb": round(vram_gb, 1),
        "gpu_name": gpu_name,
        "gpu_kind": gpu_kind,
        "cpu_count": _detect_cpu_count(),
    }


def recommend_num_ctx(info: dict | None = None) -> int:
    """Pick a context-window size that should fit on this host without OOM.

    Tiers were hand-tuned against gemma4:e4b (~3.5 GB weights at 4-bit + KV
    cache that scales linearly with num_ctx). The KV budget for GQA models is
    roughly 100-150 KB per token, so a 16K window eats ~1.7 GB VRAM on top of
    the model weights; a 32K window eats ~3.4 GB, etc. We leave a buffer for
    the embedding model (~300 MB) and for intermediate activations.

    Overrides:
      - `MM_NUM_CTX` env var forces a specific integer (bypasses auto-tune).
    """
    # Hard override from env — useful for testing or unusual hardware.
    forced = os.environ.get("MM_NUM_CTX", "").strip()
    if forced.isdigit():
        return max(2048, min(int(forced), 262144))

    info = info or detect_system()
    vram = info.get("vram_gb", 0.0) or 0.0
    ram = info.get("ram_gb", 0.0) or 0.0

    # GPU tiers — assume weights + KV cache both live in VRAM.
    if vram >= 24:
        return 65536   # RTX 3090/4090, A6000, etc.
    if vram >= 16:
        return 32768   # RTX 4080, 4070 Ti Super, 7900 XT-class
    if vram >= 10:
        return 24576   # RTX 3080 10GB, 4070
    if vram >= 7:
        return 16384   # RTX 3060 Ti, 3070, 4060 Ti (default target)
    if vram >= 5:
        return 8192    # RTX 3050 8GB, 1660 Ti

    # CPU / integrated GPU / sub-5GB VRAM — KV cache lives in system RAM,
    # and we're also competing with the OS and whatever else is running.
    if ram >= 32:
        return 8192
    if ram >= 16:
        return 4096
    return 2048


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------
# Gemma 4 variants we know how to pick between, smallest → largest, with the
# approximate resident footprint at 4-bit quant. The KV cache + activations
# add another 1-4 GB on top depending on num_ctx.
#
# We stay inside the Gemma 4 family because the rest of the app (prompts,
# tool-call formatting, multimodal image support) is validated against it;
# a different base model would need matching system-prompt tweaks.
_CHAT_MODEL_TIERS = (
    # (model_tag, approx_resident_gb, min_vram_gb, min_ram_gb_cpu_fallback)
    ("gemma4:e2b",   3.5,  4,  12),
    ("gemma4:e4b",   6.0,  8,  16),
    ("gemma4:26b",  18.0, 24,  48),
    ("gemma4:31b",  20.0, 40,  64),
)

# Embedding model used for semantic search + doc_search. Small (~275 MB) so
# we always want it present regardless of hardware tier.
EMBED_MODEL = "nomic-embed-text"


def _normalise_model_tag(tag: str) -> str:
    """Strip a `:latest` suffix so equality checks are robust.

    Ollama returns tags as e.g. "gemma4:e4b" or "gemma4:latest" — users who
    pulled the default variant get the latter but the rest of the app uses
    the explicit tag. Collapsing both to the base saves false mismatches.
    """
    t = (tag or "").strip()
    if t.endswith(":latest"):
        return t[: -len(":latest")]
    return t


def recommend_chat_model(
    installed: list[str] | None = None,
    info: dict | None = None,
) -> dict:
    """Pick the best Gemma 4 variant for this hardware.

    Logic:
      1. Compute the "best possible" tier given the detected VRAM/RAM.
      2. If the user already has a model at that tier (or any lower tier)
         installed, prefer the highest-tier installed model that fits.
         Saves a big download when the user already did it once.
      3. If nothing suitable is installed, return the highest-tier model
         that fits so the caller can pull it.

    Returns `{model, needs_pull, reason}` so callers can both set the
    default and decide whether to kick off a pull.
    """
    info = info or detect_system()
    vram = info.get("vram_gb", 0.0) or 0.0
    ram = info.get("ram_gb", 0.0) or 0.0
    installed_norm = {_normalise_model_tag(t) for t in (installed or [])}

    # Highest tier the hardware can run. When there's no GPU we fall back to
    # the RAM-based column (roughly 2x the VRAM number, because CPU inference
    # is slower but model weights can be paged out of core more aggressively).
    affordable: list[str] = []
    for tag, _size, min_vram, min_ram in _CHAT_MODEL_TIERS:
        fits_gpu = vram >= min_vram
        fits_cpu = vram == 0 and ram >= min_ram
        if fits_gpu or fits_cpu:
            affordable.append(tag)
    if not affordable:
        # Very constrained hardware — force the smallest model and hope.
        affordable = [_CHAT_MODEL_TIERS[0][0]]

    # Prefer an already-installed affordable model (user saved the download).
    for tag in reversed(affordable):
        if tag in installed_norm:
            return {"model": tag, "needs_pull": False, "reason": "installed"}

    # Nothing suitable installed — pull the biggest that fits.
    return {
        "model": affordable[-1],
        "needs_pull": True,
        "reason": "best_for_hardware",
    }


def recommend_embed_model(installed: list[str] | None = None) -> dict:
    """Always nomic-embed-text, but tell the caller whether to pull it."""
    installed_norm = {_normalise_model_tag(t) for t in (installed or [])}
    return {
        "model": EMBED_MODEL,
        "needs_pull": EMBED_MODEL not in installed_norm,
    }


def describe_host() -> str:
    """Human-readable one-liner for logs and the UI footer."""
    info = detect_system()
    ctx = recommend_num_ctx(info)
    parts = [f"{info['ram_gb']:.1f} GB RAM"]
    if info["gpu_name"]:
        parts.append(f"{info['gpu_name']} ({info['vram_gb']:.1f} GB VRAM)")
    else:
        parts.append("no discrete GPU detected")
    parts.append(f"num_ctx={ctx}")
    return " | ".join(parts)
