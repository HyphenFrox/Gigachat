"""GPU auto-recovery — un-stick an iGPU after `UR_RESULT_ERROR_DEVICE_LOST`
(or any other backend handle invalidation) without rebooting.

Why this exists:
    Intel Iris Xe iGPUs (and to a lesser extent NVIDIA / Vulkan stacks)
    sometimes hit Windows' GPU watchdog (TDR — Timeout Detection and
    Recovery) on long llama.cpp matmul kernels. Windows kills + restarts
    the display driver, and every level_zero handle the SYCL runtime was
    holding becomes stale. The next call returns DEVICE_LOST and stays
    that way until either:
      (a) the host process restarts (not enough — the driver itself may
          still be in a bad state),
      (b) the user hits Win+Ctrl+Shift+B (soft graphics-stack reset),
      (c) the user disables + re-enables the display adapter
          (`pnputil /restart-device`), or
      (d) the user reboots.

    Without this module, our auto-demote path falls through to CPU after
    one failure and stays demoted for 24 h — even if the GPU could have
    been brought back in 5 seconds. Now we try (b) → (c) → demote, in
    that order, before giving up on the device.

Public surface:
  * ``try_soft_reset()``  — Win+Ctrl+Shift+B keystroke. No admin needed.
  * ``try_hard_reset()``  — pnputil-based device restart. Needs admin
    (install.bat already runs elevated; we assume the same elevation
    context is available to the rpc-server's parent).
  * ``run_recovery(...)`` — orchestrates soft → probe → hard → probe →
    return success/failure. Used by the HTTP endpoint and the
    orchestrator's record_backend_failure pre-demote hook.

This module is best-effort by design. Every step that can fail is
caught and logged; nothing here ever raises out to the caller. A
caller checking the return value can decide whether to demote.
"""
from __future__ import annotations

import logging
import platform
import subprocess
import time
from typing import Any

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step 1 — soft reset (Win+Ctrl+Shift+B keystroke)
# ---------------------------------------------------------------------------
def try_soft_reset() -> bool:
    """Send Win+Ctrl+Shift+B to the OS.

    Windows treats this as "please soft-reset the graphics driver" —
    the display blinks black for ~1 second, the driver gets reloaded,
    and SYCL/CUDA contexts can be re-acquired. NO admin needed; works
    in a normal user session. Returns True iff the keystroke was sent
    cleanly (we can't actually confirm the driver reset happened —
    callers should re-probe the device after sleeping a bit).

    On non-Windows hosts this is a no-op returning False — the keystroke
    only means anything to Windows' DXGK subsystem.
    """
    if platform.system() != "Windows":
        return False
    try:
        import ctypes
        from ctypes import wintypes
    except Exception as e:
        log.info("gpu_recovery.soft_reset: ctypes unavailable: %s", e)
        return False

    user32 = ctypes.WinDLL("user32", use_last_error=True)

    # SendInput structures — minimal subset for KEYDOWN/KEYUP.
    KEYEVENTF_KEYUP = 0x0002
    INPUT_KEYBOARD = 1
    VK_LWIN, VK_CONTROL, VK_SHIFT, VK_B = 0x5B, 0x11, 0x10, 0x42

    class KEYBDINPUT(ctypes.Structure):
        _fields_ = [
            ("wVk", wintypes.WORD), ("wScan", wintypes.WORD),
            ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD),
            ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
        ]

    class _INPUT_UNION(ctypes.Union):
        _fields_ = [("ki", KEYBDINPUT)]

    class INPUT(ctypes.Structure):
        _anonymous_ = ("_u",)
        _fields_ = [("type", wintypes.DWORD), ("_u", _INPUT_UNION)]

    def _make(vk: int, up: bool) -> "INPUT":
        ki = KEYBDINPUT(wVk=vk, wScan=0,
                        dwFlags=(KEYEVENTF_KEYUP if up else 0),
                        time=0, dwExtraInfo=None)
        i = INPUT(); i.type = INPUT_KEYBOARD; i.ki = ki
        return i

    # Press in order, release in reverse — like a human.
    seq = [
        _make(VK_LWIN, False),
        _make(VK_CONTROL, False),
        _make(VK_SHIFT, False),
        _make(VK_B, False),
        _make(VK_B, True),
        _make(VK_SHIFT, True),
        _make(VK_CONTROL, True),
        _make(VK_LWIN, True),
    ]
    arr = (INPUT * len(seq))(*seq)
    n_sent = user32.SendInput(len(seq), arr, ctypes.sizeof(INPUT))
    ok = (n_sent == len(seq))
    if ok:
        log.info("gpu_recovery.soft_reset: Win+Ctrl+Shift+B sent")
    else:
        err = ctypes.get_last_error()
        log.info(
            "gpu_recovery.soft_reset: SendInput sent %d/%d events "
            "(GetLastError=%d)", n_sent, len(seq), err,
        )
    return ok


# ---------------------------------------------------------------------------
# Step 2 — hard reset (pnputil disable + enable)
# ---------------------------------------------------------------------------
def try_hard_reset(vendor_filter: str = "Intel") -> bool:
    """Disable + re-enable every Display-class device that matches
    `vendor_filter` in its friendly name.

    Equivalent to the Device Manager → right-click → Disable + Enable
    flow, but scripted. Resets the GPU adapter much more thoroughly
    than the soft Win+Ctrl+Shift+B — comes back with fresh queues,
    fresh memory pools, fresh handles. Display blinks for 1-3 seconds.
    Requires admin elevation (install.bat already runs elevated; if
    rpc-server's parent isn't elevated this returns False without
    pretending success).

    `vendor_filter` defaults to Intel because that's the iGPU we see
    DEVICE_LOST on most. Pass "NVIDIA" or "AMD" to target those
    vendors. Pass "" to match all Display adapters (rarely what you
    want — would also reset the host's primary display).
    """
    if platform.system() != "Windows":
        return False
    # PowerShell one-liner: find Display class devices whose friendly
    # name contains the vendor filter, disable them, wait briefly,
    # re-enable them. Returns the count of devices restarted so the
    # caller knows whether anything actually matched.
    ps_cmd = (
        f"$devs = Get-PnpDevice -Class Display -Status OK -ErrorAction SilentlyContinue | "
        f"Where-Object {{ $_.FriendlyName -like '*{vendor_filter}*' }}; "
        f"if (-not $devs) {{ Write-Output 'NONE'; exit 0 }}; "
        f"foreach ($d in $devs) {{ "
        f"  try {{ Disable-PnpDevice -InstanceId $d.InstanceId -Confirm:$false -ErrorAction Stop }} "
        f"  catch {{ Write-Output ('FAIL_DISABLE: ' + $_.Exception.Message); exit 1 }} "
        f"}}; "
        f"Start-Sleep -Seconds 2; "
        f"foreach ($d in $devs) {{ "
        f"  try {{ Enable-PnpDevice -InstanceId $d.InstanceId -Confirm:$false -ErrorAction Stop }} "
        f"  catch {{ Write-Output ('FAIL_ENABLE: ' + $_.Exception.Message); exit 1 }} "
        f"}}; "
        f"Write-Output ('OK ' + $devs.Count)"
    )
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps_cmd],
            capture_output=True, text=True, timeout=30,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        log.info("gpu_recovery.hard_reset: powershell call failed: %s", e)
        return False
    out = (result.stdout or "").strip()
    err = (result.stderr or "").strip()
    if result.returncode != 0:
        log.info(
            "gpu_recovery.hard_reset: rc=%d stdout=%r stderr=%r",
            result.returncode, out[:200], err[:200],
        )
        return False
    if out.startswith("NONE"):
        log.info(
            "gpu_recovery.hard_reset: no Display device matched vendor=%r",
            vendor_filter,
        )
        return False
    if out.startswith("OK"):
        log.info("gpu_recovery.hard_reset: %s", out)
        return True
    log.info("gpu_recovery.hard_reset: unexpected stdout=%r", out[:200])
    return False


# ---------------------------------------------------------------------------
# Step 3 — orchestrate (soft → probe → hard → probe)
# ---------------------------------------------------------------------------
def _probe_sycl_device_present(timeout_sec: float = 8.0) -> bool:
    """Best-effort: ask llama.cpp's sycl-ls (or llama-server --list-devices)
    whether at least one SYCL device enumerates.

    Used to verify that a soft/hard reset actually un-stuck the GPU.
    Returns True iff the probe found a SYCL device. False on any error
    (probe binary missing, timeout, no SYCL devices, etc.) — caller
    treats False as "not yet recovered, try the next step".
    """
    from pathlib import Path
    install_dir = Path.home() / ".gigachat" / "llama-cpp"
    sycl_ls = install_dir / "sycl-ls.exe"
    if not sycl_ls.is_file():
        # Fall through to llama-server --list-devices, which always
        # exists in our install (any backend at all).
        sycl_ls = install_dir / "llama-server.exe"
        if not sycl_ls.is_file():
            return False
        args = [str(sycl_ls), "--list-devices"]
    else:
        args = [str(sycl_ls)]
    try:
        result = subprocess.run(
            args, capture_output=True, text=True, timeout=timeout_sec,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    text = (result.stdout or "") + (result.stderr or "")
    # Be conservative: any line mentioning SYCL or level_zero with
    # a "gpu" qualifier counts as "device present".
    haystack = text.lower()
    return ("sycl" in haystack and "gpu" in haystack) or "level_zero:gpu" in haystack


def run_recovery(
    *,
    skip_hard: bool = False,
    probe_after_soft: bool = True,
    probe_after_hard: bool = True,
) -> dict[str, Any]:
    """Orchestrate the full recovery cascade.

    Returns a structured summary so callers (HTTP endpoint, orchestrator
    pre-demote hook) can decide how to act:

      {
        "ok":           bool,   # device reachable at the end
        "soft_sent":    bool,   # Win+Ctrl+Shift+B was issued
        "soft_worked":  bool,   # post-soft probe found a device
        "hard_sent":    bool,   # pnputil restart-device was attempted
        "hard_worked":  bool,   # post-hard probe found a device
        "elapsed_sec":  float,  # total time spent recovering
      }

    `skip_hard` short-circuits before the hard reset (useful for tests
    or non-admin contexts where pnputil is guaranteed to fail).
    `probe_after_*` lets callers skip the probe step if they have their
    own probe path (the orchestrator does — it re-runs its full
    rpc-server start probe, which is more authoritative than sycl-ls).
    """
    t0 = time.time()
    summary: dict[str, Any] = {
        "ok": False,
        "soft_sent": False, "soft_worked": False,
        "hard_sent": False, "hard_worked": False,
        "elapsed_sec": 0.0,
    }
    # --- soft reset ---
    summary["soft_sent"] = try_soft_reset()
    if summary["soft_sent"]:
        # Display flickers ~1s; give the SYCL runtime a moment to re-init.
        time.sleep(3.0)
        if probe_after_soft and _probe_sycl_device_present():
            summary["soft_worked"] = True
            summary["ok"] = True
            summary["elapsed_sec"] = round(time.time() - t0, 1)
            log.info("gpu_recovery.run: soft reset recovered the device")
            return summary

    # --- hard reset ---
    if skip_hard:
        summary["elapsed_sec"] = round(time.time() - t0, 1)
        return summary
    summary["hard_sent"] = try_hard_reset()
    if summary["hard_sent"]:
        # Hard reset takes longer to settle — adapter re-enumerates,
        # SYCL runtime needs to find it again.
        time.sleep(5.0)
        if probe_after_hard and _probe_sycl_device_present():
            summary["hard_worked"] = True
            summary["ok"] = True
            summary["elapsed_sec"] = round(time.time() - t0, 1)
            log.info("gpu_recovery.run: hard reset recovered the device")
            return summary

    summary["elapsed_sec"] = round(time.time() - t0, 1)
    log.info(
        "gpu_recovery.run: device NOT recovered after soft=%s hard=%s "
        "(elapsed %.1fs)",
        summary["soft_sent"], summary["hard_sent"], summary["elapsed_sec"],
    )
    return summary
