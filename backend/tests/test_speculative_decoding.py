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


# --- Auto-LAN-pull for worker-only drafts --------------------------------


def test_pick_draft_kicks_off_lan_pull_when_worker_candidate_wins(monkeypatch):
    """When the smallest viable draft sits only on a worker, we
    schedule a background pull so a future turn can promote it."""
    _stub_resolve(monkeypatch, {
        "llama3.1:8b": {
            "family": "llama", "size_bytes": 5_000_000_000,
            "gguf_path": "/m/target.gguf",
        },
        "llama3.2:1b": {
            "family": "llama", "size_bytes": 1_000_000_000,
            "gguf_path": "/m/draft.gguf",
        },
    })
    # Worker has the only viable draft.
    _stub_inventory(monkeypatch, [
        {"name": "llama3.2:1b", "family": "llama", "size_bytes": 1_000_000_000, "source": "worker:wid-A"},
    ])
    triggered: list[tuple] = []
    monkeypatch.setattr(
        compute_pool, "_maybe_kickoff_draft_lan_sync",
        lambda target, candidate, wid: triggered.append((target, candidate, wid)),
    )
    pick = compute_pool.pick_draft_for("llama3.1:8b")
    # No host candidate → return None for this turn.
    assert pick is None
    # But the background pull was scheduled.
    assert triggered == [("llama3.1:8b", "llama3.2:1b", "wid-A")]


def test_pick_draft_prefers_host_when_worker_candidate_is_larger(monkeypatch):
    """Host-resident candidates win on ties / when smaller than the
    worker-only candidate. No background pull is triggered."""
    _stub_resolve(monkeypatch, {
        "llama3.1:8b": {
            "family": "llama", "size_bytes": 5_000_000_000,
            "gguf_path": "/m/target.gguf",
        },
        "llama3.2:1b": {
            "family": "llama", "size_bytes": 1_000_000_000,
            "gguf_path": "/m/host.gguf",
        },
        "llama3.2:3b": {
            "family": "llama", "size_bytes": 3_000_000_000,
            "gguf_path": "/m/worker.gguf",
        },
    })
    _stub_inventory(monkeypatch, [
        {"name": "llama3.2:1b", "family": "llama", "size_bytes": 1_000_000_000, "source": "host"},
        {"name": "llama3.2:3b", "family": "llama", "size_bytes": 3_000_000_000, "source": "worker:wid-A"},
    ])
    triggered: list = []
    monkeypatch.setattr(
        compute_pool, "_maybe_kickoff_draft_lan_sync",
        lambda *a, **k: triggered.append(a),
    )
    pick = compute_pool.pick_draft_for("llama3.1:8b")
    assert pick is not None
    assert pick["name"] == "llama3.2:1b"  # host candidate wins
    assert triggered == []  # no pull needed


# --- Adaptive split-vs-host routing --------------------------------------


def test_should_force_split_uses_measured_verdict(monkeypatch):
    """When both host and split TPS are cached and fresh, the heuristic
    is bypassed and the router engages split iff split_tps > host_tps."""
    compute_pool._ROUTE_TPS_CACHE.clear()
    compute_pool._record_route_tps("llama3:8b", kind="host", tps=20.0)
    compute_pool._record_route_tps("llama3:8b", kind="split", tps=35.0)
    monkeypatch.setattr(compute_pool, "_eligible_split_workers", lambda: [{"id": "w"}])
    # Heuristic-irrelevant: both samples present, split is faster.
    assert compute_pool._should_force_split_for(
        "llama3:8b", strongest_single_vram=8_000_000_000, pool_vram_total=8_500_000_000,
    ) is True


def test_should_force_split_respects_measured_loss(monkeypatch):
    """Measured split slower than host → don't engage even if pool
    capacity heuristic would otherwise want it."""
    compute_pool._ROUTE_TPS_CACHE.clear()
    compute_pool._record_route_tps("llama3:8b", kind="host", tps=40.0)
    compute_pool._record_route_tps("llama3:8b", kind="split", tps=15.0)
    monkeypatch.setattr(compute_pool, "_eligible_split_workers", lambda: [{"id": "w"}])
    assert compute_pool._should_force_split_for(
        "llama3:8b", strongest_single_vram=8_000_000_000, pool_vram_total=40_000_000_000,
    ) is False


def test_should_force_split_falls_back_to_capacity_heuristic(monkeypatch):
    """No TPS cache → engage when pool VRAM ≥ 1.5× host VRAM and rpc
    workers are eligible."""
    compute_pool._ROUTE_TPS_CACHE.clear()
    monkeypatch.setattr(compute_pool, "_eligible_split_workers", lambda: [{"id": "w"}])
    # Pool 16GB vs host 8GB → 2× factor → engage.
    assert compute_pool._should_force_split_for(
        "llama3:8b", strongest_single_vram=8_000_000_000, pool_vram_total=16_000_000_000,
    ) is True
    # Pool 9GB vs host 8GB → 1.125× factor → below threshold, skip.
    assert compute_pool._should_force_split_for(
        "llama3:8b", strongest_single_vram=8_000_000_000, pool_vram_total=9_000_000_000,
    ) is False


def test_should_force_split_returns_false_without_rpc_workers(monkeypatch):
    """Even with measured split-wins or capacity-wins, no rpc workers =
    no engagement. Phase 2 needs at least one `--rpc` target."""
    compute_pool._ROUTE_TPS_CACHE.clear()
    monkeypatch.setattr(compute_pool, "_eligible_split_workers", lambda: [])
    assert compute_pool._should_force_split_for(
        "llama3:8b", strongest_single_vram=8_000_000_000, pool_vram_total=40_000_000_000,
    ) is False


def test_route_tps_cache_round_trips_recent_measurement():
    """Recording a TPS measurement makes it readable through the
    paired getter — within the freshness window."""
    compute_pool._ROUTE_TPS_CACHE.clear()
    compute_pool._record_route_tps("llama3:8b", kind="host", tps=42.5)
    assert compute_pool._route_tps_for("llama3:8b", "host") == 42.5
    assert compute_pool._route_tps_for("llama3:8b", "split") is None


def test_route_tps_cache_ignores_zero_or_negative():
    """A failed bench (TPS=0) must not clobber a previously-recorded
    valid measurement."""
    compute_pool._ROUTE_TPS_CACHE.clear()
    compute_pool._record_route_tps("llama3:8b", kind="host", tps=42.5)
    compute_pool._record_route_tps("llama3:8b", kind="host", tps=0)
    compute_pool._record_route_tps("llama3:8b", kind="host", tps=-1)
    assert compute_pool._route_tps_for("llama3:8b", "host") == 42.5


def test_route_tps_cache_expires_stale_samples(monkeypatch):
    """A sample older than the TTL is treated as missing — the router
    re-benches rather than trusting an obsolete number."""
    compute_pool._ROUTE_TPS_CACHE.clear()
    compute_pool._record_route_tps("llama3:8b", kind="split", tps=20.0)
    # Stamp the entry as hour-2-old (TTL is 24h, so still fresh).
    compute_pool._ROUTE_TPS_CACHE["llama3:8b"]["split_measured_at"] = (
        compute_pool.time.time() - 7200
    )
    assert compute_pool._route_tps_for("llama3:8b", "split") == 20.0
    # Now stamp it as 25-hour-old → expired.
    compute_pool._ROUTE_TPS_CACHE["llama3:8b"]["split_measured_at"] = (
        compute_pool.time.time() - 25 * 3600
    )
    assert compute_pool._route_tps_for("llama3:8b", "split") is None


# --- Round-robin embeddings ----------------------------------------------


def test_pick_embed_target_rotates_across_workers(monkeypatch):
    """Three eligible workers → three back-to-back calls hit each one
    in turn instead of pinning to the first."""
    fake_workers = [
        {"id": "w1", "address": "w1.local", "ollama_port": 11434, "label": "w1"},
        {"id": "w2", "address": "w2.local", "ollama_port": 11434, "label": "w2"},
        {"id": "w3", "address": "w3.local", "ollama_port": 11434, "label": "w3"},
    ]
    monkeypatch.setattr(
        compute_pool, "_eligible_workers",
        lambda flag, model=None: list(fake_workers),
    )
    # All workers comparably "capable" — no exclusion at the threshold step.
    monkeypatch.setattr(compute_pool, "_capability_score", lambda w: (1.0, True, 100, 100))
    monkeypatch.setattr(
        compute_pool.db, "get_compute_worker_auth_token", lambda wid: None,
    )
    compute_pool._EMBED_TARGET_INDEX.clear()

    seen: list[str] = []
    for _ in range(6):
        result = compute_pool.pick_embed_target("nomic-embed-text")
        assert result is not None
        seen.append(result[0])
    # Round-robin over 3 workers across 6 calls → each worker hit twice.
    assert sorted(set(seen)) == ["http://w1.local:11434", "http://w2.local:11434", "http://w3.local:11434"]
    assert len([s for s in seen if s == "http://w1.local:11434"]) == 2


def test_pick_embed_target_returns_none_when_no_eligible_workers(monkeypatch):
    """No eligible workers → caller falls back to host (None return)."""
    monkeypatch.setattr(compute_pool, "_eligible_workers", lambda *a, **k: [])
    assert compute_pool.pick_embed_target("nomic-embed-text") is None


# --- Pool inventory + dedup advisor --------------------------------------


def test_pool_inventory_summary_aggregates_per_model(monkeypatch):
    """Two nodes hold the same model → summary reports it once with
    `copies=2` and `redundant_bytes = size_bytes`."""
    _stub_inventory(monkeypatch, [
        {"name": "llama3:8b", "family": "llama", "size_bytes": 5_000_000_000, "source": "host"},
        {"name": "llama3:8b", "family": "llama", "size_bytes": 5_000_000_000, "source": "worker:wA"},
        {"name": "qwen2.5:0.5b", "family": "qwen2", "size_bytes": 400_000_000, "source": "host"},
    ])
    summary = compute_pool.pool_inventory_summary()
    by_name = {m["name"]: m for m in summary["models"]}
    assert by_name["llama3:8b"]["copies"] == 2
    assert by_name["llama3:8b"]["redundant_bytes"] == 5_000_000_000
    assert sorted(by_name["llama3:8b"]["locations"]) == ["host", "worker:wA"]
    assert by_name["qwen2.5:0.5b"]["copies"] == 1
    assert by_name["qwen2.5:0.5b"]["redundant_bytes"] == 0


def test_pool_inventory_totals_account_for_all_copies(monkeypatch):
    """`total_pool_bytes` counts every copy; `total_unique_bytes`
    counts each model once; `total_redundant_bytes` is the difference
    between them — the disk footprint a perfect dedup would reclaim."""
    _stub_inventory(monkeypatch, [
        {"name": "A:1", "family": "fam", "size_bytes": 1_000_000_000, "source": "host"},
        {"name": "A:1", "family": "fam", "size_bytes": 1_000_000_000, "source": "worker:wA"},
        {"name": "A:1", "family": "fam", "size_bytes": 1_000_000_000, "source": "worker:wB"},
        {"name": "B:1", "family": "fam", "size_bytes": 500_000_000, "source": "host"},
    ])
    s = compute_pool.pool_inventory_summary()
    assert s["total_unique_bytes"] == 1_500_000_000
    assert s["total_pool_bytes"] == 3_500_000_000
    assert s["total_redundant_bytes"] == 2_000_000_000


def test_pool_dedup_recommends_keeping_host_copy(monkeypatch, isolated_db):
    """Dedup advisor prefers keeping host's copy because it's the
    most-likely chat target and a Phase 1 chat from another node would
    have to round-trip back anyway."""
    monkeypatch.setattr(compute_pool, "db", isolated_db)
    _stub_inventory(monkeypatch, [
        {"name": "A:1", "family": "fam", "size_bytes": 1_000_000_000, "source": "host"},
        {"name": "A:1", "family": "fam", "size_bytes": 1_000_000_000, "source": "worker:wA"},
        {"name": "A:1", "family": "fam", "size_bytes": 1_000_000_000, "source": "worker:wB"},
    ])
    # Workers exist for the dedup advisor to rank.
    isolated_db.create_compute_worker(label="wA", address="a", enabled=True)
    isolated_db.create_compute_worker(label="wB", address="b", enabled=True)

    recs = compute_pool.pool_dedup_recommendations()
    assert len(recs) == 1
    assert recs[0]["model"] == "A:1"
    assert recs[0]["keep_at"] == "host"
    assert sorted(recs[0]["remove_from"]) == ["worker:wA", "worker:wB"]
    assert recs[0]["bytes_reclaimed"] == 2_000_000_000


def test_pool_dedup_skips_models_with_single_copy(monkeypatch):
    """Models that exist on only one node have nothing to reclaim —
    dedup advisor omits them entirely."""
    _stub_inventory(monkeypatch, [
        {"name": "solo:1", "family": "fam", "size_bytes": 1_000_000_000, "source": "host"},
    ])
    assert compute_pool.pool_dedup_recommendations() == []


# --- Distributed tool execution (fetch_url SSH dispatch) -----------------


async def test_dispatch_fetch_returns_none_when_no_eligible_workers(isolated_db, monkeypatch):
    """No workers configured → `None` so the caller falls back to host.
    We never block the chat on a missing pool. (Always-on by design;
    the gate is whether an eligible worker exists, not a setting flag.)"""
    monkeypatch.setattr(compute_pool, "db", isolated_db)
    result = await compute_pool.dispatch_fetch_url_to_worker("https://example.com")
    assert result is None


def test_pick_tool_dispatch_target_round_robins(isolated_db, monkeypatch):
    """Three eligible workers → three back-to-back picks rotate. Same
    pattern as embedding round-robin (#5)."""
    monkeypatch.setattr(compute_pool, "db", isolated_db)
    import time as _time
    fresh = _time.time()
    fake_workers = [
        {
            "id": f"w{i}", "label": f"w{i}",
            "ssh_host": f"alias{i}",
            "address": f"w{i}.local",
            "ollama_port": 11434,
            "last_seen": fresh,
            "enabled": True,
            "use_for_chat": True,
            "use_for_embeddings": True,
            "use_for_subagents": True,
        }
        for i in range(3)
    ]
    monkeypatch.setattr(
        compute_pool.db, "list_compute_workers",
        lambda enabled_only=False: list(fake_workers),
    )
    compute_pool._TOOL_DISPATCH_INDEX.clear()
    seen = []
    for _ in range(6):
        pick = compute_pool._pick_tool_dispatch_target()
        assert pick is not None
        seen.append(pick["id"])
    # All three workers hit, each twice.
    assert sorted(set(seen)) == ["w0", "w1", "w2"]


def test_pick_tool_dispatch_target_skips_workers_without_ssh(isolated_db, monkeypatch):
    """A worker without `ssh_host` configured can't be a dispatch
    target — there's no transport. Picker drops them."""
    monkeypatch.setattr(compute_pool, "db", isolated_db)
    import time as _time
    fresh = _time.time()
    rows = [
        {"id": "no-ssh", "ssh_host": "", "last_seen": fresh, "label": "no-ssh", "enabled": True},
        {"id": "no-probe", "ssh_host": "alias", "last_seen": None, "label": "no-probe", "enabled": True},
    ]
    monkeypatch.setattr(
        compute_pool.db, "list_compute_workers",
        lambda enabled_only=False: rows,
    )
    compute_pool._TOOL_DISPATCH_INDEX.clear()
    assert compute_pool._pick_tool_dispatch_target() is None


# --- Web-search worker selection -----------------------------------------


def test_pick_web_search_target_filters_by_ddgs(monkeypatch):
    """Workers without `ddgs` installed (capability flag missing) are
    skipped — dispatching to them would just error out on import."""
    import time as _time
    fresh = _time.time()
    workers = [
        {
            "id": "w-ready", "label": "w-ready", "ssh_host": "ready",
            "last_seen": fresh, "enabled": True,
            "capabilities": {"has_ddgs": True},
        },
        {
            "id": "w-no-ddgs", "label": "w-no-ddgs", "ssh_host": "noddgs",
            "last_seen": fresh, "enabled": True,
            "capabilities": {"has_ddgs": False},
        },
    ]
    monkeypatch.setattr(
        compute_pool.db, "list_compute_workers",
        lambda enabled_only=False: workers,
    )
    compute_pool._TOOL_DISPATCH_INDEX.clear()
    pick = compute_pool._pick_web_search_target()
    assert pick is not None
    assert pick["id"] == "w-ready"


# --- Read-doc worker selection -------------------------------------------


def test_pick_read_doc_target_filters_by_lib(monkeypatch):
    """Each format requires a specific library. Worker missing it gets
    skipped so we don't dispatch a doomed parse request."""
    import time as _time
    fresh = _time.time()
    workers = [
        {
            "id": "w-pdf", "label": "w-pdf", "ssh_host": "pdf",
            "last_seen": fresh, "enabled": True,
            "capabilities": {"read_doc_libs": ["pymupdf"]},
        },
        {
            "id": "w-docx", "label": "w-docx", "ssh_host": "docx",
            "last_seen": fresh, "enabled": True,
            "capabilities": {"read_doc_libs": ["docx"]},
        },
    ]
    monkeypatch.setattr(
        compute_pool.db, "list_compute_workers",
        lambda enabled_only=False: workers,
    )
    compute_pool._TOOL_DISPATCH_INDEX.clear()
    # PDF dispatch → pymupdf worker
    pdf_pick = compute_pool._pick_read_doc_target(".pdf")
    assert pdf_pick is not None and pdf_pick["id"] == "w-pdf"
    compute_pool._TOOL_DISPATCH_INDEX.clear()
    # DOCX dispatch → docx worker
    docx_pick = compute_pool._pick_read_doc_target(".docx")
    assert docx_pick is not None and docx_pick["id"] == "w-docx"
    # XLSX dispatch → no eligible worker
    compute_pool._TOOL_DISPATCH_INDEX.clear()
    assert compute_pool._pick_read_doc_target(".xlsx") is None


def test_pick_read_doc_target_returns_none_for_unknown_suffix():
    """Unknown extension → None. read_doc on host handles the error
    response itself."""
    assert compute_pool._pick_read_doc_target(".weird") is None


# --- Worker-side llama-server lifecycle ----------------------------------


def test_pick_worker_resident_draft_finds_smaller_same_family(monkeypatch):
    """Worker has both target and a smaller same-family model in its
    Ollama inventory — picker returns the smaller as draft."""
    _stub_resolve(monkeypatch, {
        "llama3.1:8b": {
            "family": "llama", "size_bytes": 5_000_000_000,
            "gguf_path": "/host/target.gguf",
        },
    })
    worker = {
        "id": "w", "label": "w",
        "capabilities": {
            "models": [
                {"name": "llama3.1:8b", "family": "llama", "size": 5_000_000_000},
                {"name": "llama3.2:1b", "family": "llama", "size": 1_000_000_000},
                {"name": "llama3.2:3b", "family": "llama", "size": 3_000_000_000},
            ],
        },
    }
    pick = compute_pool._pick_worker_resident_draft(worker, "llama3.1:8b")
    assert pick is not None
    # Smallest viable wins.
    assert pick["name"] == "llama3.2:1b"


def test_pick_worker_resident_draft_rejects_cross_family(monkeypatch):
    """Worker has a smaller model but it's a different family → no
    pick. Same tokenizer-vocab requirement as the host picker."""
    _stub_resolve(monkeypatch, {
        "llama3.1:8b": {
            "family": "llama", "size_bytes": 5_000_000_000,
            "gguf_path": "/host/target.gguf",
        },
    })
    worker = {
        "id": "w",
        "capabilities": {
            "models": [
                {"name": "qwen2.5:0.5b", "family": "qwen2", "size": 400_000_000},
            ],
        },
    }
    assert compute_pool._pick_worker_resident_draft(worker, "llama3.1:8b") is None


def test_worker_has_vram_for_pair_uses_proven_vram():
    """The headroom check uses `max_vram_seen_bytes` as the upper
    bound. A worker that's never loaded enough → can't host both.

    Headroom factor is `_WORKER_SPECULATIVE_VRAM_HEADROOM` (1.15 under
    the 5 %-margin policy: ~10 % real KV need + 5 % allocator buffer).
    The boundary needs to land between the two cases for the test to
    remain meaningful: pick a target+draft size that fits 16 GB at
    1.15× and a larger pair that doesn't.
    """
    worker = {"capabilities": {"max_vram_seen_bytes": 16_000_000_000}}
    # 5 GB target + 1 GB draft → 6 GB × 1.15 = 6.9 GB ≤ 16 GB.
    assert compute_pool._worker_has_vram_for_pair(worker, 5_000_000_000, 1_000_000_000) is True
    # 14 GB target + 1 GB draft → 15 GB × 1.15 = 17.25 GB > 16 GB.
    assert compute_pool._worker_has_vram_for_pair(worker, 14_000_000_000, 1_000_000_000) is False


def test_worker_has_vram_for_pair_refuses_unbenched_worker():
    """No probe data on max_vram_seen → can't verify, refuse rather
    than guess. Host-only path is the safe fallback."""
    worker = {"capabilities": {}}
    assert compute_pool._worker_has_vram_for_pair(
        worker, 1_000_000_000, 100_000_000,
    ) is False


# --- Active dedup execution ----------------------------------------------


async def test_execute_dedup_skips_host_locations(monkeypatch, isolated_db):
    """Recommendations include `host` as a remove-from candidate would
    be a no-op error path; the executor explicitly skips those.
    Operator handles host removals via Ollama directly."""
    monkeypatch.setattr(compute_pool, "db", isolated_db)
    # Stub the recommendations to include a host-only removal.
    monkeypatch.setattr(
        compute_pool, "pool_dedup_recommendations",
        lambda: [{
            "model": "M:1", "size_bytes": 1_000_000_000,
            "keep_at": "worker:w1", "remove_from": ["host"],
            "bytes_reclaimed": 1_000_000_000,
        }],
    )
    results = await compute_pool.execute_dedup_recommendations()
    # The host-only entry produces a "skipped" result, not a success.
    assert len(results) == 1
    assert results[0]["ok"] is False
    assert "host" in results[0]["error"].lower()


# --- Quant-variant grouping ----------------------------------------------


def test_strip_quant_suffix_extracts_known_quants():
    """Recognise the common Ollama / GGUF quant suffixes — Q4_0,
    Q4_K_M, Q5_K_S, Q8_0, IQ3_XS — and uppercase the result so the
    UI can group case-insensitively."""
    assert compute_pool._strip_quant_suffix("llama3:8b-q4_K_M") == ("llama3:8b", "Q4_K_M")
    assert compute_pool._strip_quant_suffix("qwen2.5:0.5b-q8_0") == ("qwen2.5:0.5b", "Q8_0")
    assert compute_pool._strip_quant_suffix("mistral:7b-iq3_xs") == ("mistral:7b", "IQ3_XS")
    assert compute_pool._strip_quant_suffix("gemma:2b-q4_0") == ("gemma:2b", "Q4_0")


def test_strip_quant_suffix_preserves_names_without_suffix():
    """A name with no recognisable quant (e.g. just `llama3:8b`)
    returns the original + ``None``."""
    assert compute_pool._strip_quant_suffix("llama3:8b") == ("llama3:8b", None)
    assert compute_pool._strip_quant_suffix("custom:my-tag") == ("custom:my-tag", None)
    # The underscore-but-not-quant case shouldn't match.
    assert compute_pool._strip_quant_suffix("model:special-edition") == ("model:special-edition", None)


def test_pool_inventory_groups_quant_variants(monkeypatch):
    """Two quant variants of the same base appear as one group with
    both variants listed; single-quant models don't show up in
    `quant_groups`."""
    _stub_inventory(monkeypatch, [
        {"name": "llama3:8b-q4_K_M", "family": "llama", "size_bytes": 5_000_000_000, "source": "host"},
        {"name": "llama3:8b-q8_0", "family": "llama", "size_bytes": 8_500_000_000, "source": "worker:wA"},
        # Single-quant model — no group expected.
        {"name": "qwen2.5:0.5b-q4_0", "family": "qwen2", "size_bytes": 400_000_000, "source": "host"},
    ])
    summary = compute_pool.pool_inventory_summary()
    groups = summary.get("quant_groups", [])
    assert len(groups) == 1
    g = groups[0]
    assert g["base"] == "llama3:8b"
    quants = sorted(v["quant"] for v in g["variants"])
    assert quants == ["Q4_K_M", "Q8_0"]


async def test_execute_dedup_filters_to_specific_model(monkeypatch, isolated_db):
    """`model_filter` restricts execution to one model; other recs
    are dropped before any SSH dispatch."""
    monkeypatch.setattr(compute_pool, "db", isolated_db)
    monkeypatch.setattr(
        compute_pool, "pool_dedup_recommendations",
        lambda: [
            {"model": "A:1", "size_bytes": 1_000_000_000, "keep_at": "host",
             "remove_from": ["host"], "bytes_reclaimed": 1_000_000_000},
            {"model": "B:1", "size_bytes": 2_000_000_000, "keep_at": "host",
             "remove_from": ["host"], "bytes_reclaimed": 2_000_000_000},
        ],
    )
    results = await compute_pool.execute_dedup_recommendations(model_filter="B:1")
    assert len(results) == 1
    assert results[0]["model"] == "B:1"


def test_pick_embed_target_excludes_dramatically_slower_worker(monkeypatch):
    """A worker measured at <50 % of the leader's TPS is excluded from
    the rotation — including it would slow the whole pool to its pace."""
    fast = {"id": "w-fast", "address": "fast.local", "ollama_port": 11434, "label": "fast"}
    slow = {"id": "w-slow", "address": "slow.local", "ollama_port": 11434, "label": "slow"}
    monkeypatch.setattr(
        compute_pool, "_eligible_workers",
        lambda flag, model=None: [fast, slow],
    )
    # Score schema: (tps, gpu_present, vram, last_seen). Slow is 30% of fast.
    scores = {"w-fast": (100.0, True, 8_000_000_000, 1.0), "w-slow": (30.0, True, 8_000_000_000, 1.0)}
    monkeypatch.setattr(compute_pool, "_capability_score", lambda w: scores[w["id"]])
    monkeypatch.setattr(
        compute_pool.db, "get_compute_worker_auth_token", lambda wid: None,
    )
    compute_pool._EMBED_TARGET_INDEX.clear()

    seen = set()
    for _ in range(4):
        seen.add(compute_pool.pick_embed_target("nomic-embed-text")[0])
    assert seen == {"http://fast.local:11434"}  # slow worker excluded


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
