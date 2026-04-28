"""Regression: `split_runtime` — find / install state for llama.cpp.

Phase 2 commit 2 ships the detection layer. Actual subprocess-spawning
lifecycle (commit 4) gets its own tests; this file pins:

  * `find_llama_server` / `find_rpc_server` resolve binaries from the
    private install dir AND from PATH, with private dir winning so a
    stale system install can't override our pinned version.
  * `get_install_status()` returns `installed=False` cleanly when no
    binaries are present (the boot-without-llama-cpp path).
  * `_extract_zip` refuses zip-slip paths.

Network is fully stubbed via `httpx.MockTransport` so these run
offline. Filesystem is isolated via tmp_path so the developer's real
`~/.gigachat/llama-cpp/` is never touched.
"""
from __future__ import annotations

import io
import platform
import zipfile
from pathlib import Path

import httpx
import pytest

from backend import split_runtime

pytestmark = pytest.mark.smoke


# `_resolve_binary` in split_runtime looks for `name.exe` on Windows and
# bare `name` everywhere else. Mirror that here so the fake binaries the
# tests drop on disk are discoverable by `find_llama_server` regardless of
# which OS the CI runner happens to be on.
_BINARY_SUFFIX = ".exe" if platform.system() == "Windows" else ""


# --- helpers --------------------------------------------------------------


def _redirect_install_dir(monkeypatch, tmp_path: Path) -> Path:
    """Point the module at a temp dir so tests don't write to user home."""
    install_dir = tmp_path / "llama-cpp"
    monkeypatch.setattr(split_runtime, "LLAMA_CPP_INSTALL_DIR", install_dir)
    # Also strip PATH so a developer's real llama-server (if any) doesn't
    # leak into the test result.
    monkeypatch.setenv("PATH", "")
    return install_dir


def _drop_fake_binary(install_dir: Path, name: str) -> Path:
    """Pretend an extracted llama.cpp install exists. Files are touched
    empty — they only need to exist for `is_file()` checks. Suffix is
    OS-aware so the binary is discovered on every CI matrix entry."""
    install_dir.mkdir(parents=True, exist_ok=True)
    p = install_dir / f"{name}{_BINARY_SUFFIX}"
    p.write_bytes(b"")
    return p


# --- find_llama_server / find_rpc_server ---------------------------------


def test_find_llama_server_returns_none_when_missing(monkeypatch, tmp_path):
    _redirect_install_dir(monkeypatch, tmp_path)
    assert split_runtime.find_llama_server() is None
    assert split_runtime.find_rpc_server() is None


def test_find_llama_server_picks_up_private_install(monkeypatch, tmp_path):
    install = _redirect_install_dir(monkeypatch, tmp_path)
    fake = _drop_fake_binary(install, "llama-server")
    assert split_runtime.find_llama_server() == fake


def test_find_rpc_server_separate_from_llama_server(monkeypatch, tmp_path):
    """Both binaries can be installed independently. The host has
    `llama-server` only (uses workers' rpc-server remotely); a worker
    drop has `rpc-server` only."""
    install = _redirect_install_dir(monkeypatch, tmp_path)
    _drop_fake_binary(install, "rpc-server")
    assert split_runtime.find_rpc_server() is not None
    assert split_runtime.find_llama_server() is None


def test_find_falls_through_to_path(monkeypatch, tmp_path):
    """A power user with their own llama.cpp build on PATH should still
    work — we don't require people to use our pinned install."""
    _redirect_install_dir(monkeypatch, tmp_path)
    path_dir = tmp_path / "elsewhere"
    path_dir.mkdir()
    (path_dir / f"llama-server{_BINARY_SUFFIX}").write_bytes(b"")
    monkeypatch.setenv("PATH", str(path_dir))
    found = split_runtime.find_llama_server()
    assert found is not None
    assert found.parent == path_dir


def test_private_install_wins_over_path(monkeypatch, tmp_path):
    """When the same binary exists in both the private install and on
    PATH, the private install wins so a stale system copy can't shadow
    our pinned version."""
    install = _redirect_install_dir(monkeypatch, tmp_path)
    private = _drop_fake_binary(install, "llama-server")
    path_dir = tmp_path / "elsewhere"
    path_dir.mkdir()
    (path_dir / f"llama-server{_BINARY_SUFFIX}").write_bytes(b"")
    monkeypatch.setenv("PATH", str(path_dir))
    assert split_runtime.find_llama_server() == private


# --- get_install_status --------------------------------------------------


def test_install_status_uninstalled(monkeypatch, tmp_path):
    _redirect_install_dir(monkeypatch, tmp_path)
    s = split_runtime.get_install_status()
    assert s.installed is False
    assert s.llama_server_path is None
    assert s.rpc_server_path is None
    assert s.version == split_runtime.LLAMA_CPP_VERSION


def test_install_status_with_llama_server_only(monkeypatch, tmp_path):
    install = _redirect_install_dir(monkeypatch, tmp_path)
    _drop_fake_binary(install, "llama-server")
    s = split_runtime.get_install_status()
    assert s.installed is True
    assert s.llama_server_path is not None
    # rpc-server is optional on the host (workers run it remotely).
    assert s.rpc_server_path is None


# --- zip extraction safety -----------------------------------------------


def _build_zip_in_memory(entries: list[tuple[str, bytes]]) -> bytes:
    """Helper: build an in-memory ZIP from (name, payload) pairs."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, payload in entries:
            zf.writestr(name, payload)
    return buf.getvalue()


def test_extract_zip_writes_files(tmp_path):
    z = tmp_path / "test.zip"
    z.write_bytes(_build_zip_in_memory([
        ("llama-server.exe", b"binary"),
        ("ggml-cuda.dll", b"library"),
    ]))
    out = tmp_path / "out"
    written = split_runtime._extract_zip(z, out)
    names = {p.name for p in written}
    assert names == {"llama-server.exe", "ggml-cuda.dll"}
    assert (out / "llama-server.exe").read_bytes() == b"binary"


def test_extract_zip_refuses_zip_slip(tmp_path):
    """A malicious zip with `../escape.txt` must not write outside the
    target dir. Defensive check — official llama.cpp zips don't do this,
    but we'd be on the hook if a user pointed us at a tampered URL."""
    z = tmp_path / "evil.zip"
    z.write_bytes(_build_zip_in_memory([
        ("../escape.txt", b"pwn"),
        ("legit.exe", b"ok"),
    ]))
    out = tmp_path / "out"
    written = split_runtime._extract_zip(z, out)
    # Only the legit file made it through.
    names = {p.name for p in written}
    assert names == {"legit.exe"}
    assert not (tmp_path / "escape.txt").exists()


# --- download_llama_cpp (network stubbed) --------------------------------


def test_download_rejects_unknown_variant(monkeypatch, tmp_path):
    _redirect_install_dir(monkeypatch, tmp_path)
    with pytest.raises(ValueError):
        split_runtime.download_llama_cpp(variant="bogus")


def test_download_happy_path_extracts_and_finds(monkeypatch, tmp_path):
    """End-to-end: stub the GitHub release URL, return a fake zip
    containing `llama-server` (with the OS-appropriate suffix), verify
    the install dir ends up with the binary and `find_llama_server`
    resolves it. The real downloads are Windows-only, but the extractor
    plus discovery layer are platform-agnostic — this test runs on every
    CI matrix entry, so the zip names match whatever `find_llama_server`
    will actually look for on the current OS."""
    install = _redirect_install_dir(monkeypatch, tmp_path)
    fake_zip_bytes = _build_zip_in_memory([
        (f"llama-server{_BINARY_SUFFIX}", b"binary"),
        (f"llama-cli{_BINARY_SUFFIX}", b"binary"),
    ])

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=fake_zip_bytes,
            headers={"content-length": str(len(fake_zip_bytes))},
        )

    transport = httpx.MockTransport(handler)
    # `httpx.stream` is a top-level helper that doesn't accept `transport` —
    # it builds its own client internally. We replace it with a stub that
    # creates a client wired to MockTransport, so `with httpx.stream(...)
    # as r` semantics still work.
    from contextlib import contextmanager

    @contextmanager
    def stub_stream(method, url, **kwargs):
        kwargs.pop("transport", None)
        with httpx.Client(transport=transport, timeout=kwargs.pop("timeout", None)) as c:
            with c.stream(method, url, **kwargs) as r:
                yield r

    monkeypatch.setattr(split_runtime.httpx, "stream", stub_stream)

    result = split_runtime.download_llama_cpp(variant="host")
    assert result == install
    assert split_runtime.find_llama_server() is not None
    # Zip cleaned up after extract.
    assert not (install / "llama-cpp-host.zip").exists()
