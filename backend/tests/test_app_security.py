"""Tests for the upload-name validator that guards image endpoints.

The `_safe_upload_name` helper in `backend.app` is the single chokepoint
between user-supplied filenames and the filesystem. These tests document
the expected behaviour so a future refactor doesn't accidentally widen the
attack surface.
"""

from __future__ import annotations

import pytest
from backend.app import _safe_upload_name

# Whole module is fast + offline — runs in the smoke tier.
pytestmark = pytest.mark.smoke


def test_accepts_uuid_hex_with_known_extension():
    """Filenames that match our own format pass through unchanged."""
    assert _safe_upload_name("0123456789abcdef0123456789abcdef.png") == \
        "0123456789abcdef0123456789abcdef.png"


def test_rejects_path_traversal():
    """Anything with a path component must be stripped or rejected."""
    # `..` plus some other parts should resolve to '..' (basename is '..')
    # which is rejected outright. Conservative: not accepting anything we
    # didn't write.
    assert _safe_upload_name("../etc/passwd") is None
    assert _safe_upload_name("/etc/passwd") is None
    assert _safe_upload_name("..\\windows\\system32\\cmd.exe") is None


def test_rejects_dotted_basenames():
    """`.` and `..` as basenames are rejected."""
    assert _safe_upload_name(".") is None
    assert _safe_upload_name("..") is None


def test_rejects_unknown_extension():
    """Only allow-listed extensions pass."""
    assert _safe_upload_name("0123456789abcdef0123456789abcdef.exe") is None
    assert _safe_upload_name("0123456789abcdef0123456789abcdef.txt") is None
    # No extension at all.
    assert _safe_upload_name("0123456789abcdef0123456789abcdef") is None


def test_rejects_non_hex_stem():
    """Stems that aren't hex (i.e. could be arbitrary names) are rejected."""
    assert _safe_upload_name("not-hex.png") is None
    assert _safe_upload_name("hello world.jpg") is None


def test_accepts_jpeg_webp_gif_extensions():
    """All four allow-listed image extensions should pass."""
    stem = "deadbeefdeadbeefdeadbeefdeadbeef"
    assert _safe_upload_name(f"{stem}.jpg") == f"{stem}.jpg"
    assert _safe_upload_name(f"{stem}.webp") == f"{stem}.webp"
    assert _safe_upload_name(f"{stem}.gif") == f"{stem}.gif"
