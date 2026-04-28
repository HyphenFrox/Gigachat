"""Regression: speculative-decoding draft picker + spawn-flag wiring.

Three things pinned here:

  * `pick_draft_for(target)` walks the pool's combined model inventory
    and returns the smallest, same-family, dramatically-smaller chat
    model whose GGUF is locally resolvable. It MUST refuse cross-family
    pairs, drafts ≥ 30 % of target size, and embedding-only models.

  * `_host_has_vram_for_speculative` honours the configured headroom
    multiplier so the router never engages speculative on a node that
    couldn't actually hold both models.

  * `_build_command` wires `-md`, `--draft-max`, `--draft-min`, and
    `-ngld` exactly when ``draft_gguf_path`` is set — and emits none
    of those flags when the caller opts out.

The pool inventory walker is fully stubbed; nothing here touches Ollama
or the disk beyond the temp DB.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from backend import compute_pool, split_lifecycle

pytestmark = pytest.mark.smoke


# --- _build_command speculative wiring -----------------------------------


def test_build_command_omits_draft_flags_when_not_set():
    """Vanilla single-model spawn: zero draft-related flags in argv."""
    cmd = split_lifecycle._build_command(
        llama_server=Path("/tmp/llama-server"),
        gguf_path="/tmp/m.gguf",
        port=11500,
        rpc_endpoints=[],
    )
    assert "-md" not in cmd
    assert "--draft-max" not in cmd
    assert "--draft-min" not in cmd
    assert "-ngld" not in cmd


def test_build_command_emits_draft_flags_when_set():
    """`draft_gguf_path` produces the documented flag set in argv."""
    cmd = split_lifecycle._build_command(
        llama_server=Path("/tmp/llama-server"),
        gguf_path="/tmp/target.gguf",
        port=11500,
        rpc_endpoints=[],
        draft_gguf_path="/tmp/draft.gguf",
    )
    assert "-md" in cmd
    # Adjacent positional value.
    assert cmd[cmd.index("-md") + 1] == "/tmp/draft.gguf"
    assert "--draft-max" in cmd and cmd[cmd.index("--draft-max") + 1] == "8"
    assert "--draft-min" in cmd and cmd[cmd.index("--draft-min") + 1] == "1"
    # Same -ngl-style "max layers to GPU" hint, but for the draft.
    assert "-ngld" in cmd and cmd[cmd.index("-ngld") + 1] == "99"


def test_build_command_speculative_works_with_rpc_workers():
    """Speculative decoding stacks on top of layer-split — both flag
    families coexist when both are configured."""
    cmd = split_lifecycle._build_command(
        llama_server=Path("/tmp/llama-server"),
        gguf_path="/tmp/big.gguf",
        port=11500,
        rpc_endpoints=["worker1.local:50052", "worker2.local:50052"],
        draft_gguf_path="/tmp/draft.gguf",
    )
    assert "--rpc" in cmd
    assert "-md" in cmd


# --- _host_has_vram_for_speculative --------------------------------------


def test_vram_check_rejects_when_budget_zero(monkeypatch):
    """A host with no VRAM-budget signal (no GPU detected) cannot serve
    speculative — return False so the router falls back to Ollama."""
    monkeypatch.setattr(compute_pool, "_host_vram_budget_bytes", lambda: 0)
    assert compute_pool._host_has_vram_for_speculative(1_000_000_000, 200_000_000) is False


def test_vram_check_rejects_when_pair_overruns_budget(monkeypatch):
    """Target + draft × 1.30 must fit the host's VRAM budget."""
    monkeypatch.setattr(compute_pool, "_host_vram_budget_bytes", lambda: 7_000_000_000)
    # 6 GB target + 1 GB draft = 7 GB pair × 1.30 = 9.1 GB ⊁ 7 GB budget.
    assert compute_pool._host_has_vram_for_speculative(
        6_000_000_000, 1_000_000_000,
    ) is False


def test_vram_check_accepts_when_pair_fits(monkeypatch):
    """A target + draft pair that fits with headroom returns True."""
    monkeypatch.setattr(compute_pool, "_host_vram_budget_bytes", lambda: 16_000_000_000)
    # 7 GB target + 1 GB draft = 8 GB pair × 1.30 = 10.4 GB ≤ 16 GB.
    assert compute_pool._host_has_vram_for_speculative(
        7_000_000_000, 1_000_000_000,
    ) is True


def test_vram_check_rejects_zero_or_negative_sizes(monkeypatch):
    """Defensive — bogus 0 sizes from a probe miss must not pass the gate."""
    monkeypatch.setattr(compute_pool, "_host_vram_budget_bytes", lambda: 16_000_000_000)
    assert compute_pool._host_has_vram_for_speculative(0, 1_000_000_000) is False
    assert compute_pool._host_has_vram_for_speculative(7_000_000_000, 0) is False


# --- pick_draft_for: positive matches ------------------------------------


def _stub_inventory(monkeypatch, models: list[dict]) -> None:
    """Inject a fake pool inventory so the picker tests don't need a
    real Ollama install or network probe."""
    monkeypatch.setattr(
        compute_pool, "_pool_model_inventory", lambda: list(models),
    )


def _stub_resolve(monkeypatch, by_name: dict[str, dict]) -> None:
    """Inject a fake `resolve_ollama_model` return so the picker can
    look up GGUF paths without disk I/O."""
    monkeypatch.setattr(
        compute_pool, "resolve_ollama_model",
        lambda name: by_name.get(name),
    )


def test_pick_draft_returns_smallest_same_family_host_resident(monkeypatch):
    """Two same-family candidates on host — picker returns the smaller."""
    _stub_resolve(monkeypatch, {
        "llama3.1:70b": {"family": "llama", "size_bytes": 40_000_000_000, "gguf_path": "/m/big.gguf"},
        "llama3.2:1b": {"family": "llama", "size_bytes": 1_000_000_000, "gguf_path": "/m/draft1.gguf"},
        "llama3.2:3b": {"family": "llama", "size_bytes": 3_000_000_000, "gguf_path": "/m/draft3.gguf"},
    })
    _stub_inventory(monkeypatch, [
        {"name": "llama3.2:3b", "family": "llama", "size_bytes": 3_000_000_000, "source": "host"},
        {"name": "llama3.2:1b", "family": "llama", "size_bytes": 1_000_000_000, "source": "host"},
    ])
    pick = compute_pool.pick_draft_for("llama3.1:70b")
    assert pick is not None
    assert pick["name"] == "llama3.2:1b"
    assert pick["gguf_path"] == "/m/draft1.gguf"
    assert pick["match"] == "family"


def test_pick_draft_promotes_cross_family_when_tokenizer_matches(monkeypatch):
    """A different-family candidate whose tokenizer fingerprint matches
    the target IS accepted — the family heuristic alone would reject
    it, but identical token IDs mean speculative decoding works fine."""
    _stub_resolve(monkeypatch, {
        "llama3.1:8b": {
            "family": "llama", "size_bytes": 5_000_000_000,
            "gguf_path": "/m/target.gguf",
        },
        "mistral-tiny:1b": {
            # Different family label, but ships the Llama tokenizer.
            "family": "mistral", "size_bytes": 1_000_000_000,
            "gguf_path": "/m/mistral.gguf",
        },
    })
    _stub_inventory(monkeypatch, [
        {"name": "mistral-tiny:1b", "family": "mistral", "size_bytes": 1_000_000_000, "source": "host"},
    ])
    # Stub fingerprint to return the SAME hash for both paths.
    monkeypatch.setattr(
        compute_pool, "_gguf_tokenizer_fingerprint",
        lambda p: "shared-llama-tokenizer-hash",
    )
    pick = compute_pool.pick_draft_for("llama3.1:8b")
    assert pick is not None
    assert pick["name"] == "mistral-tiny:1b"
    assert pick["match"] == "tokenizer"


def test_pick_draft_rejects_cross_family_when_fingerprint_differs(monkeypatch):
    """Different family + different fingerprint → the picker correctly
    refuses. Two different tokenizers can't share token IDs, so
    speculative decoding wouldn't work."""
    _stub_resolve(monkeypatch, {
        "llama3.1:8b": {
            "family": "llama", "size_bytes": 5_000_000_000,
            "gguf_path": "/m/target.gguf",
        },
        "qwen2.5:0.5b": {
            "family": "qwen2", "size_bytes": 400_000_000,
            "gguf_path": "/m/qwen.gguf",
        },
    })
    _stub_inventory(monkeypatch, [
        {"name": "qwen2.5:0.5b", "family": "qwen2", "size_bytes": 400_000_000, "source": "host"},
    ])
    # Different fingerprint per path — Llama vs Qwen tokenizer.
    fingerprints = {
        "/m/target.gguf": "llama-tokenizer-hash",
        "/m/qwen.gguf": "qwen-tokenizer-hash",
    }
    monkeypatch.setattr(
        compute_pool, "_gguf_tokenizer_fingerprint",
        lambda p: fingerprints.get(p),
    )
    assert compute_pool.pick_draft_for("llama3.1:8b") is None


def test_pick_draft_skips_cross_family_when_fingerprint_unavailable(monkeypatch):
    """If the GGUF parser can't read either model's tokenizer (no
    `gguf` package installed, or malformed file), cross-family
    candidates are silently rejected — `None` from the fingerprint
    means "can't verify", not "implicit match"."""
    _stub_resolve(monkeypatch, {
        "llama3.1:8b": {
            "family": "llama", "size_bytes": 5_000_000_000,
            "gguf_path": "/m/target.gguf",
        },
        "mistral-tiny:1b": {
            "family": "mistral", "size_bytes": 1_000_000_000,
            "gguf_path": "/m/mistral.gguf",
        },
    })
    _stub_inventory(monkeypatch, [
        {"name": "mistral-tiny:1b", "family": "mistral", "size_bytes": 1_000_000_000, "source": "host"},
    ])
    # Both fingerprint reads fail.
    monkeypatch.setattr(compute_pool, "_gguf_tokenizer_fingerprint", lambda p: None)
    assert compute_pool.pick_draft_for("llama3.1:8b") is None


def test_pick_draft_manual_override_bypasses_safety_checks(monkeypatch, isolated_db):
    """A user-pinned override is trusted unconditionally — size check,
    family check, and fingerprint check are all skipped. Misuse
    produces low accept rates but never crashes."""
    monkeypatch.setattr(compute_pool, "db", isolated_db)
    _stub_resolve(monkeypatch, {
        "llama3.1:8b": {
            "family": "llama", "size_bytes": 5_000_000_000,
            "gguf_path": "/m/target.gguf",
        },
        # The user wants this exotic Qwen as the draft for a Llama
        # target, even though normal checks would reject the pair.
        "qwen2.5:0.5b": {
            "family": "qwen2", "size_bytes": 400_000_000,
            "gguf_path": "/m/qwen.gguf",
        },
    })
    _stub_inventory(monkeypatch, [])  # Nothing in normal inventory.

    import json as _json
    isolated_db.set_setting(
        "compute_pool_speculative_overrides",
        _json.dumps({"llama3.1:8b": "qwen2.5:0.5b"}),
    )
    pick = compute_pool.pick_draft_for("llama3.1:8b")
    assert pick is not None
    assert pick["name"] == "qwen2.5:0.5b"
    assert pick["match"] == "override"


def test_pick_draft_falls_through_when_override_target_unresolvable(monkeypatch, isolated_db):
    """Stale override (user removed the pinned model) doesn't kill the
    picker — log + fall through to the auto picker so the chat layer
    still gets a viable draft."""
    monkeypatch.setattr(compute_pool, "db", isolated_db)
    _stub_resolve(monkeypatch, {
        "llama3.1:8b": {
            "family": "llama", "size_bytes": 5_000_000_000,
            "gguf_path": "/m/target.gguf",
        },
        "llama3.2:1b": {
            "family": "llama", "size_bytes": 1_000_000_000,
            "gguf_path": "/m/draft.gguf",
        },
        # Note: deleted-model:1b is intentionally NOT in this map.
    })
    _stub_inventory(monkeypatch, [
        {"name": "llama3.2:1b", "family": "llama", "size_bytes": 1_000_000_000, "source": "host"},
    ])
    import json as _json
    isolated_db.set_setting(
        "compute_pool_speculative_overrides",
        _json.dumps({"llama3.1:8b": "deleted-model:1b"}),
    )
    pick = compute_pool.pick_draft_for("llama3.1:8b")
    assert pick is not None
    # Auto picker took over with the family-matched draft.
    assert pick["name"] == "llama3.2:1b"
    assert pick["match"] == "family"


# --- pick_draft_for: negative matches ------------------------------------


def test_pick_draft_returns_none_for_unknown_target(monkeypatch):
    """resolve_ollama_model returning None for the target → no pick."""
    _stub_resolve(monkeypatch, {})
    _stub_inventory(monkeypatch, [])
    assert compute_pool.pick_draft_for("nonexistent:abc") is None


def test_pick_draft_returns_none_when_target_too_small(monkeypatch):
    """Target smaller than the speculative-decoding minimum (~1.5 GB)
    means the per-token overhead would eat the speedup. Skip silently."""
    _stub_resolve(monkeypatch, {
        "tiny:1b": {"family": "llama", "size_bytes": 800_000_000, "gguf_path": "/m/t.gguf"},
        "smaller:0.5b": {"family": "llama", "size_bytes": 300_000_000, "gguf_path": "/m/d.gguf"},
    })
    _stub_inventory(monkeypatch, [
        {"name": "smaller:0.5b", "family": "llama", "size_bytes": 300_000_000, "source": "host"},
    ])
    assert compute_pool.pick_draft_for("tiny:1b") is None


def test_pick_draft_returns_none_when_no_same_family_candidate(monkeypatch):
    """Cross-family pairs (qwen2 draft, llama target) are rejected —
    they share no tokenizer, accept rate would be ~0 %."""
    _stub_resolve(monkeypatch, {
        "llama3.1:70b": {"family": "llama", "size_bytes": 40_000_000_000, "gguf_path": "/m/big.gguf"},
        "qwen2.5:0.5b": {"family": "qwen2", "size_bytes": 400_000_000, "gguf_path": "/m/d.gguf"},
    })
    _stub_inventory(monkeypatch, [
        {"name": "qwen2.5:0.5b", "family": "qwen2", "size_bytes": 400_000_000, "source": "host"},
    ])
    assert compute_pool.pick_draft_for("llama3.1:70b") is None


def test_pick_draft_rejects_drafts_above_size_fraction(monkeypatch):
    """A draft ≥ 30 % of target costs more cycles than it saves."""
    _stub_resolve(monkeypatch, {
        "llama3:8b": {"family": "llama", "size_bytes": 5_000_000_000, "gguf_path": "/m/big.gguf"},
        "llama3:3b": {"family": "llama", "size_bytes": 2_000_000_000, "gguf_path": "/m/d.gguf"},
    })
    _stub_inventory(monkeypatch, [
        # 2 GB / 5 GB = 40 % — over the 30 % threshold.
        {"name": "llama3:3b", "family": "llama", "size_bytes": 2_000_000_000, "source": "host"},
    ])
    assert compute_pool.pick_draft_for("llama3:8b") is None


def test_pick_draft_rejects_target_as_its_own_draft(monkeypatch):
    """Defensive — same model name cannot serve as its own draft."""
    _stub_resolve(monkeypatch, {
        "llama3:8b": {"family": "llama", "size_bytes": 5_000_000_000, "gguf_path": "/m/big.gguf"},
    })
    _stub_inventory(monkeypatch, [
        {"name": "llama3:8b", "family": "llama", "size_bytes": 5_000_000_000, "source": "host"},
    ])
    assert compute_pool.pick_draft_for("llama3:8b") is None


def test_pick_draft_skips_worker_only_candidates(monkeypatch):
    """V1 only promotes host-resident drafts — the GGUF must be
    locally readable by llama-server. Worker-only candidates are
    documented as future work."""
    _stub_resolve(monkeypatch, {
        "llama3.1:70b": {"family": "llama", "size_bytes": 40_000_000_000, "gguf_path": "/m/big.gguf"},
        "llama3.2:1b": {"family": "llama", "size_bytes": 1_000_000_000, "gguf_path": "/m/d.gguf"},
    })
    _stub_inventory(monkeypatch, [
        # Same family, fits size budget, BUT not on host.
        {"name": "llama3.2:1b", "family": "llama", "size_bytes": 1_000_000_000, "source": "worker:abc"},
    ])
    assert compute_pool.pick_draft_for("llama3.1:70b") is None


# --- pick_draft_for: edge cases ------------------------------------------


def test_pick_draft_filters_out_embed_models_via_inventory(monkeypatch):
    """The inventory walker drops embed-named models before they reach
    the picker — defensive belt-and-braces, the picker itself doesn't
    need to re-check."""
    _stub_resolve(monkeypatch, {
        "llama3:8b": {"family": "llama", "size_bytes": 5_000_000_000, "gguf_path": "/m/big.gguf"},
    })
    # Inventory pre-filtered (no embed entries reach `pick_draft_for`).
    _stub_inventory(monkeypatch, [])
    assert compute_pool.pick_draft_for("llama3:8b") is None


# --- speculative_decoding_enabled flag -----------------------------------


def test_speculative_default_on(isolated_db, monkeypatch):
    """No setting written → speculative defaults to ON. The picker's
    own gates handle viability so leaving it on is a no-op for setups
    that can't benefit and a free speedup for everyone else."""
    monkeypatch.setattr(compute_pool, "db", isolated_db)
    assert compute_pool.speculative_decoding_enabled() is True


def test_speculative_flag_round_trip(isolated_db, monkeypatch):
    """Persisting the setting flips the helper's verdict — both
    directions, including the explicit-disable path the user takes
    when they want to force the legacy Ollama-only behaviour."""
    monkeypatch.setattr(compute_pool, "db", isolated_db)
    isolated_db.set_setting("compute_pool_speculative_decoding", "false")
    assert compute_pool.speculative_decoding_enabled() is False
    isolated_db.set_setting("compute_pool_speculative_decoding", "true")
    assert compute_pool.speculative_decoding_enabled() is True
    isolated_db.set_setting("compute_pool_speculative_decoding", "0")
    assert compute_pool.speculative_decoding_enabled() is False
