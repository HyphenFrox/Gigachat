"""Runtime for user-defined tools.

Settings → Tools (POST /api/user-tools) stores a new Python tool in the
``user_tools`` SQLite table — the LLM itself cannot create tools. This module
is the execution side: it owns a dedicated virtualenv at ``data/tools_venv/``,
installs dependencies into it on demand, and runs each user tool in an
isolated subprocess that communicates with the agent over JSON stdin/stdout.

Design invariants:
  * **No in-process exec.** User code never runs inside the backend's Python
    interpreter. If a user tool segfaults or `sys.exit()`s, only the sandboxed
    subprocess dies; the backend keeps serving.
  * **Single shared venv.** One venv for all user tools keeps the install cost
    reasonable — tool creation only pays for packages that are genuinely new.
    Cross-tool package conflicts are the user's problem; the venv is an
    upgrade channel, not a per-tool sandbox. (If that becomes painful we can
    switch to per-tool venvs later without changing the public API.)
  * **Hard-validated dep specs.** Every entry in ``deps`` is matched against
    a PEP 508 subset regex before being passed to pip. This is the only
    defence against a hostile dep like ``-r /etc/passwd`` or
    ``--index-url http://attacker/`` — pip treats those as flags if they
    land on the argv.
  * **Timeout + output cap.** Execution reuses the same pattern as
    ``python_exec`` — asyncio subprocess with a hard timeout, falling back to
    blocking subprocess when the event loop can't spawn children (the
    WindowsSelectorEventLoop case). Output is capped at ``MAX_OUTPUT_CHARS``.
  * **Kill switch.** Setting ``GIGACHAT_DISABLE_USER_TOOLS=1`` in the
    environment turns the whole feature off — creation is refused,
    execution errors out. Lets a paranoid deployment keep the rest of
    Gigachat while disabling runtime code generation.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import sys
import venv
from pathlib import Path

MAX_OUTPUT_CHARS = 20000
DEFAULT_TIMEOUT_SEC = 60
PIP_INSTALL_TIMEOUT_SEC = 300  # pip can be slow on first compile of scientific wheels

# Shared venv location. Sits next to the SQLite DB so the user's data dir
# stays self-contained and portable. ``data/`` is already in .gitignore.
USER_TOOLS_DIR = Path(__file__).resolve().parent.parent / "data" / "tools_venv"

# PEP 508 subset: name [extras] [version-spec]. We intentionally refuse
# direct-URL requirements (``foo @ git+https://...``) and local paths — those
# would let an attacker pull arbitrary code from a URL they control or
# exfiltrate files via `pip install .`. Pinned versions use a handful of the
# common comparison ops.
_PKG_NAME = r"[A-Za-z][A-Za-z0-9_\-\.]{0,99}"
_PKG_EXTRAS = r"(?:\[[A-Za-z0-9_,\-\.\s]+\])?"
_PKG_VERSION = r"(?:(?:==|>=|<=|~=|!=|>|<)[A-Za-z0-9\.\-\+\*!]{1,50})?"
_DEP_SPEC_RE = re.compile(rf"^{_PKG_NAME}{_PKG_EXTRAS}{_PKG_VERSION}$")

# Packages we refuse to install at any version. `pip` itself is fine — we
# invoke it via `-m pip` which always uses the venv's copy — but a user tool
# shouldn't be able to shadow the agent's own deps under the venv's name.
# Keep the list short; this is defence-in-depth, not a policy engine.
_DEP_BLOCKLIST = {
    "pip",
    "setuptools",
    "wheel",
    "distribute",
}


def is_disabled() -> bool:
    """Return True when the operator has turned the feature off via env var."""
    return os.environ.get("GIGACHAT_DISABLE_USER_TOOLS", "").strip() in {"1", "true", "yes"}


def _venv_python() -> Path:
    """Absolute path to the user-tool venv's Python interpreter."""
    if sys.platform == "win32":
        return USER_TOOLS_DIR / "Scripts" / "python.exe"
    return USER_TOOLS_DIR / "bin" / "python"


def ensure_venv() -> Path:
    """Create the shared venv if missing. Returns the venv's python path.

    Reuses Python's stdlib ``venv`` rather than shelling out to `python -m
    venv` — simpler to reason about and avoids picking up a stray PATH
    entry. ``with_pip=True`` guarantees the venv has its own pip so
    subsequent ``install_deps`` calls don't accidentally invoke the
    backend's pip.
    """
    py = _venv_python()
    if py.is_file():
        return py
    USER_TOOLS_DIR.parent.mkdir(parents=True, exist_ok=True)
    # clear=False so a broken-but-partially-created venv is detected by
    # ``py.is_file()`` above; if someone manually deletes the interpreter we
    # rebuild from scratch on next call.
    venv.create(str(USER_TOOLS_DIR), with_pip=True, clear=False)
    return py


def _validate_dep(spec: str) -> str:
    """Return the cleaned spec if valid; raise ValueError otherwise.

    This is the security-critical check. Pip treats any argv entry that
    starts with ``-`` as a flag, so without this an attacker could slip in
    ``--index-url http://evil/`` or ``-r /etc/passwd``. We enforce:

      * spec must match the PEP 508 subset regex
      * package name must not appear in ``_DEP_BLOCKLIST``
      * no shell metacharacters whatsoever
    """
    s = (spec or "").strip()
    if not s:
        raise ValueError("dep spec must not be empty")
    if len(s) > 200:
        raise ValueError("dep spec too long (max 200 chars)")
    if not _DEP_SPEC_RE.match(s):
        raise ValueError(
            f"dep spec {s!r} is not a valid PEP 508 subset — "
            "only `name[extras]version` forms are accepted, "
            "no URLs / local paths / VCS references"
        )
    # Extract bare package name for the blocklist check.
    bare = re.split(r"[\[=<>!~]", s, maxsplit=1)[0].strip().lower()
    if bare in _DEP_BLOCKLIST:
        raise ValueError(f"dep {bare!r} is blocklisted for safety")
    return s


def validate_deps(deps: list[str]) -> list[str]:
    """Validate every entry and return the cleaned list in input order."""
    if not isinstance(deps, list):
        raise ValueError("deps must be a list of strings")
    return [_validate_dep(d) for d in deps]


async def install_deps(deps: list[str]) -> dict:
    """Pip-install the given deps into the shared venv. Returns a result dict.

    No-op (ok=True with an empty ``output``) if deps is empty. Pip is
    idempotent so re-installing an already-present package is cheap. We
    pass ``--disable-pip-version-check --no-input`` so pip can't prompt
    the user (it would hang the subprocess) and can't print a warning
    banner that confuses the agent.
    """
    if is_disabled():
        return {"ok": False, "output": "", "error": "user tools are disabled via GIGACHAT_DISABLE_USER_TOOLS"}
    cleaned = validate_deps(deps)
    if not cleaned:
        return {"ok": True, "output": ""}
    py = ensure_venv()
    cmd = [
        str(py),
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--no-input",
        *cleaned,
    ]
    try:
        return await _run_subprocess(cmd, timeout=PIP_INSTALL_TIMEOUT_SEC)
    except Exception as e:  # noqa: BLE001 — surface any install error to the agent
        return {"ok": False, "output": "", "error": f"pip install failed: {type(e).__name__}: {e}"}


# Wrapper script template. The user's tool source is loaded alongside a small
# bootstrapper that:
#   1. Reads a JSON arg payload from stdin.
#   2. Invokes ``run(args)`` (which the user code must define).
#   3. Emits the return value as a JSON object on stdout.
#
# We keep the wrapper tiny so the user sees exactly the error they caused
# rather than a traceback dominated by our glue code.
_WRAPPER = """
import json
import sys
import traceback

# USER CODE BEGIN
{user_code}
# USER CODE END

def _emit(obj):
    sys.stdout.write("__GIGACHAT_USER_TOOL_RESULT__\\n")
    sys.stdout.write(json.dumps(obj, ensure_ascii=False, default=str))
    sys.stdout.flush()

try:
    args = json.loads(sys.stdin.read() or "{{}}")
except Exception as e:
    _emit({{"ok": False, "output": "", "error": f"invalid args JSON: {{e}}"}})
    sys.exit(2)

if "run" not in globals() or not callable(globals()["run"]):
    _emit({{"ok": False, "output": "", "error": "tool must define `def run(args: dict) -> dict`"}})
    sys.exit(2)

try:
    result = run(args)
except Exception:
    _emit({{"ok": False, "output": "", "error": traceback.format_exc()}})
    sys.exit(1)

if not isinstance(result, dict):
    _emit({{"ok": False, "output": "", "error": f"run() must return a dict, got {{type(result).__name__}}"}})
    sys.exit(2)

# Normalise to the same shape built-in tools return.
result.setdefault("ok", True)
result.setdefault("output", "")
_emit(result)
"""


def _clip(s: str) -> str:
    """Mirror ``tools._clip`` — keeps head+tail when truncating long output."""
    if len(s) <= MAX_OUTPUT_CHARS:
        return s
    head = s[: MAX_OUTPUT_CHARS // 2]
    tail = s[-MAX_OUTPUT_CHARS // 2:]
    return f"{head}\n\n... [output truncated, {len(s) - MAX_OUTPUT_CHARS} chars omitted] ...\n\n{tail}"


def _parse_wrapper_output(raw: str) -> dict:
    """Split the subprocess stdout into (prints, result-dict).

    The wrapper emits a sentinel line (``__GIGACHAT_USER_TOOL_RESULT__``)
    right before the JSON payload so everything the user code printed during
    its run is available as ``output`` — same UX as ``python_exec`` — while
    the structured result still round-trips cleanly.

    Missing sentinel (e.g. process was killed before emit) yields a helpful
    error instead of a silent "ok=True, output=''".
    """
    sentinel = "__GIGACHAT_USER_TOOL_RESULT__\n"
    idx = raw.rfind(sentinel)
    if idx < 0:
        return {
            "ok": False,
            "output": _clip(raw),
            "error": "tool did not return a result (process exited early?)",
        }
    printed = raw[:idx]
    payload = raw[idx + len(sentinel):].strip()
    try:
        result = json.loads(payload)
    except Exception as e:  # noqa: BLE001 — bad result payload is a tool-author bug
        return {
            "ok": False,
            "output": _clip(raw),
            "error": f"tool result was not valid JSON: {e}",
        }
    if not isinstance(result, dict):
        return {
            "ok": False,
            "output": _clip(raw),
            "error": "tool result was not a JSON object",
        }
    # Merge the printed stdout into the `output` the model sees, preserving
    # any `output` the user explicitly returned (printed text goes first).
    existing = str(result.get("output") or "")
    combined = (printed.rstrip() + ("\n" if printed.rstrip() and existing else "") + existing).strip()
    result["output"] = _clip(combined)
    return result


async def _run_subprocess(argv: list[str], *, timeout: int, stdin_data: bytes | None = None, cwd: str | None = None) -> dict:
    """Shared subprocess runner — asyncio first, blocking fallback on Selector loop.

    Same pattern as ``tools.python_exec``, refactored into a reusable helper
    because we need it for both pip-install and tool-execution paths. Returns
    a ``{ok, output, error?, exit_code?}`` dict with the combined
    stdout+stderr.
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            cwd=cwd,
            stdin=asyncio.subprocess.PIPE if stdin_data is not None else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            stdout, _ = await asyncio.wait_for(
                proc.communicate(input=stdin_data), timeout=timeout
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return {"ok": False, "output": "", "error": f"timed out after {timeout}s"}
        out = stdout.decode("utf-8", errors="replace") if stdout else ""
        return {
            "ok": proc.returncode == 0,
            "output": out,  # caller clips or parses
            "exit_code": proc.returncode,
        }
    except NotImplementedError:
        # Windows Selector loop can't spawn subprocesses — fall back to sync.
        def _blocking() -> dict:
            try:
                cp = subprocess.run(
                    argv,
                    cwd=cwd,
                    input=stdin_data,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    timeout=timeout,
                )
                out = cp.stdout.decode("utf-8", errors="replace") if cp.stdout else ""
                return {"ok": cp.returncode == 0, "output": out, "exit_code": cp.returncode}
            except subprocess.TimeoutExpired:
                return {"ok": False, "output": "", "error": f"timed out after {timeout}s"}
            except Exception as e:  # noqa: BLE001
                return {"ok": False, "output": "", "error": f"{type(e).__name__}: {e}"}

        return await asyncio.to_thread(_blocking)
    except Exception as e:  # noqa: BLE001 — spawn failure (permissions etc.)
        return {"ok": False, "output": "", "error": f"{type(e).__name__}: {e}"}


async def execute_user_tool(
    *,
    code: str,
    args: dict | None,
    timeout: int,
    cwd: str,
) -> dict:
    """Run a user tool in the shared venv and return its structured result.

    Parameters
    ----------
    code
        The tool's Python source (must define ``def run(args: dict) -> dict``).
    args
        JSON-serialisable dict the model passed along with the tool call.
        Sent to the subprocess on stdin.
    timeout
        Hard wall-clock cap in seconds; the subprocess is killed on expiry.
    cwd
        Working directory for the subprocess — usually the conversation's
        ``cwd`` so the tool can read relative paths.
    """
    if is_disabled():
        return {"ok": False, "output": "", "error": "user tools are disabled via GIGACHAT_DISABLE_USER_TOOLS"}
    py = ensure_venv()
    wrapper_src = _WRAPPER.replace("{user_code}", code or "")
    t = max(1, min(int(timeout or DEFAULT_TIMEOUT_SEC), 600))
    try:
        payload = json.dumps(args or {}, ensure_ascii=False).encode("utf-8")
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "output": "", "error": f"args not JSON-serialisable: {e}"}
    # ``-I`` isolates the interpreter from PYTHON* env vars and the user-site
    # directory — same stance as python_exec. ``-c`` keeps the wrapper off
    # disk so there's no temp file to race with or clean up.
    raw_result = await _run_subprocess(
        [str(py), "-I", "-c", wrapper_src],
        timeout=t,
        stdin_data=payload,
        cwd=cwd,
    )
    # Subprocess-level failure (spawn error, timeout): propagate as-is.
    if raw_result.get("error") and not raw_result.get("output"):
        return raw_result
    return _parse_wrapper_output(raw_result.get("output") or "")
