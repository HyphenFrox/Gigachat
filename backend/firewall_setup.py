"""Idempotent Windows firewall rule installer.

Why this exists
===============
The user's complaint: "Why do we need to fix firewall over and over
again. When we install the app it should just set up whatever we
need to use the app".

`install.bat` opens the necessary ports at install time, but:
  * Existing deployments installed before the fuller port list (only
    8000 was opened originally) need to migrate without re-running
    install.bat.
  * If the OS reclassifies a network adapter (Public ↔ Private),
    Private-profile rules silently stop applying. Re-asserting at
    boot time covers that.
  * New ports added by the app over time should "just work" without
    the user knowing they exist.

Strategy
========
At backend boot we attempt to install the firewall rules. Each
``netsh advfirewall firewall add rule`` call is best-effort:
  * Without elevation, the call fails with a clean "access denied"
    error → we log INFO (not ERROR) and continue. The user sees no
    crash, just falls back to whatever rules already exist.
  * With elevation, the rule is created if missing OR updated to
    match. ``netsh ... add rule`` happily creates duplicates so we
    delete first then add to keep the rule set clean.

We DON'T trigger the UAC prompt on backend boot — that would be
hostile UX for a daemon process. Users who installed pre-fuller-
ports get a one-time message in the log directing them to re-run
install.bat (admin) for the migration. Future installs cover all
ports out of the gate.
"""
from __future__ import annotations

import logging
import os
import platform
import subprocess

log = logging.getLogger(__name__)


# Ports the app needs reachable from paired-LAN peers.
# Each entry is (port, display_name, description).
_REQUIRED_PORTS: tuple[tuple[int, str, str], ...] = (
    (8000, "Gigachat backend (port 8000)",
     "P2P endpoints (encrypted compute proxy + pair handshake)"),
    (50052, "Gigachat rpc-server SYCL (port 50052)",
     "llama.cpp rpc-server (SYCL/CUDA backend) for split-mode layer push"),
    (50053, "Gigachat rpc-server CPU (port 50053)",
     "llama.cpp rpc-server (CPU/RAM backend) for split-mode layer push"),
    (8090, "Gigachat llama-server (port 8090)",
     "llama-server for peer-orchestrated split (when this device holds the GGUF)"),
)


def _is_admin() -> bool:
    """Return True iff the current process can write firewall rules.

    Uses the canonical "can read SYSTEM-only registry" check, which
    matches what `net session` does in install.bat. False positives
    are impossible (a non-admin can't pass this); false negatives
    only happen when the elevation token is funky (rare in practice).
    """
    if platform.system() != "Windows":
        return False
    try:
        import ctypes
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False


def _rule_exists(display_name: str) -> bool:
    """Best-effort: True iff a firewall rule with this DisplayName
    is already installed.

    `netsh advfirewall firewall show rule name="<n>"` exits 0 when
    the rule exists, non-zero ("No rules match the specified
    criteria") otherwise. Bypasses PowerShell entirely so we don't
    pay the ~1 s startup tax per port.
    """
    if platform.system() != "Windows":
        return False
    try:
        proc = subprocess.run(
            ["netsh", "advfirewall", "firewall", "show", "rule",
             f"name={display_name}"],
            capture_output=True, text=True, timeout=5.0,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return proc.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


def _install_rule(port: int, display_name: str, description: str) -> bool:
    """Attempt to add a Private-profile inbound TCP allow for `port`.

    Idempotent: deletes any existing rule with the same DisplayName
    first so we never accumulate duplicates after rule edits across
    versions.

    Returns True on success, False on any failure (including missing
    elevation). Failures are logged at INFO so a non-admin daemon
    boot doesn't spam ERROR.
    """
    if platform.system() != "Windows":
        return False
    try:
        # Best-effort delete first. `netsh advfirewall firewall delete
        # rule name=X` is a no-op (exit 1, "No rules match...") when
        # the rule doesn't exist; we ignore the return code.
        subprocess.run(
            ["netsh", "advfirewall", "firewall", "delete", "rule",
             f"name={display_name}"],
            capture_output=True, text=True, timeout=5.0,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        proc = subprocess.run(
            ["netsh", "advfirewall", "firewall", "add", "rule",
             f"name={display_name}",
             "dir=in", "action=allow", "protocol=TCP",
             f"localport={port}",
             "profile=private",
             f"description={description}"],
            capture_output=True, text=True, timeout=10.0,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if proc.returncode == 0:
            return True
        log.info(
            "firewall_setup: netsh add rule for port %d returned %d: %s",
            port, proc.returncode,
            (proc.stdout or proc.stderr or "").strip()[:200],
        )
        return False
    except (subprocess.TimeoutExpired, OSError) as e:
        log.info(
            "firewall_setup: netsh invocation failed for port %d: %s",
            port, e,
        )
        return False


def ensure_firewall_rules() -> dict:
    """Idempotent: ensure every port in ``_REQUIRED_PORTS`` has its
    firewall rule installed. Safe to call on every boot.

    Returns ``{"installed": [<port>...], "skipped": [<port>...],
    "missing_admin": bool}``.

    Behaviour:
      * Linux / macOS → no-op (no Windows firewall to manage).
      * Already-correct rules → counted as "skipped" (not re-installed).
      * Missing rules + admin → installed.
      * Missing rules + no admin → INFO log directing the user to
        re-run install.bat.

    Called from `app.py` at startup so installs predating the
    fuller port list migrate automatically without user action.
    """
    out: dict = {"installed": [], "skipped": [], "failed": [],
                 "missing_admin": False}
    if platform.system() != "Windows":
        return out
    have_admin = _is_admin()
    rules_to_install: list[tuple[int, str, str]] = []
    for port, name, description in _REQUIRED_PORTS:
        if _rule_exists(name):
            out["skipped"].append(port)
        else:
            rules_to_install.append((port, name, description))
    if not rules_to_install:
        return out
    if not have_admin:
        out["missing_admin"] = True
        log.info(
            "firewall_setup: %d firewall rule(s) missing for ports %s "
            "but backend isn't running as Administrator. Re-run "
            "install.bat (UAC) to install them, or split-mode + "
            "peer-orchestrated chat across LAN devices may fail.",
            len(rules_to_install),
            [p for p, _n, _d in rules_to_install],
        )
        out["failed"] = [p for p, _n, _d in rules_to_install]
        return out
    for port, name, description in rules_to_install:
        if _install_rule(port, name, description):
            out["installed"].append(port)
            log.info(
                "firewall_setup: installed Private-profile inbound "
                "TCP allow for port %d (%s)", port, name,
            )
        else:
            out["failed"].append(port)
    return out
