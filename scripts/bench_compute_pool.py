"""All-metrics bench: every metric per model, with/without pool, side by side.

Run from project root:  python scripts/bench_compute_pool.py
"""
import asyncio
import json
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx

from backend import agent, compute_pool, db


WORKER_IDS = [w["id"] for w in db.list_compute_workers()]


def set_all_workers_enabled(enabled: bool) -> None:
    """Toggle ALL workers — needed when more than one is registered.
    Without this the bench falsely measures a half-enabled pool."""
    for wid in WORKER_IDS:
        db.update_compute_worker(wid, enabled=enabled)
EMBED_PROMPT = "The quick brown fox jumps over the lazy dog. " * 4
CHAT_PROMPT = "Count from 1 to 30, numbers only, separated by spaces."
N_PREDICT = 50
EMBED_RUNS = 3


def discover_models():
    manifests_root = os.path.expanduser(
        "~/.ollama/models/manifests/registry.ollama.ai/library"
    )
    models = []
    for name in sorted(os.listdir(manifests_root)):
        for tag in os.listdir(os.path.join(manifests_root, name)):
            if not os.path.isfile(os.path.join(manifests_root, name, tag)):
                continue
            full = f"{name}:{tag}"
            if "embed" in name.lower():
                continue
            if "vision" in name.lower():
                continue
            if name == "gemma4" and tag == "latest":
                continue  # duplicate of e4b
            models.append(full)
    return models


async def _warmup_chat(model: str, target: str) -> None:
    """Burn through one tiny call so the model is hot in VRAM/RAM
    when the timed call fires. Cold-load (multi-GB model paging in
    + Phase 2 layer push over LAN) gets accounted to warmup, NOT
    to the timed metrics. Bounded by the timeout the caller chose
    upstream."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "ok"}],
        "stream": True,
        "options": {"temperature": 0.0, "num_predict": 1},
    }
    timeout = httpx.Timeout(connect=10.0, read=900.0, write=30.0, pool=10.0)
    try:
        async with httpx.AsyncClient(timeout=timeout) as c:
            async with c.stream("POST", f"{target}/api/chat", json=payload) as r:
                async for line in r.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    if obj.get("done"):
                        break
    except Exception:
        pass  # warmup failures don't fail the bench; the timed call surfaces them


async def measure_chat(model):
    print(f"  starting {model} ...", flush=True)
    try:
        decision = await compute_pool.route_chat_for(model)
    except Exception as e:
        return {
            "engine": "ROUTING_ERR",
            "url": "",
            "error": str(e)[:80],
            "ttft_ms": 0,
            "total_s": 0,
            "tokens": 0,
            "tok_per_s": 0,
        }
    target = decision.get("base_url") or "http://localhost:11434"
    engine = decision.get("engine") or "?"

    # Pre-populate the host throughput cache so pick_chat_target
    # doesn't schedule a background bench that would contend for
    # host GPU during the timed call. Runs synchronously here.
    try:
        await compute_pool._measure_host_throughput(model)
    except Exception:
        pass

    # Warmup pass — loads the model into the chosen target's VRAM/RAM
    # so the timed measurement reflects hot inference, not cold-load
    # latency. For Phase 2 split this also pushes the layer weights
    # across LAN once before timing.
    print(f"    warmup ({engine}) ...", flush=True)
    t_warm = time.perf_counter()
    await _warmup_chat(model, target)
    print(f"    warmup done in {time.perf_counter() - t_warm:.1f}s", flush=True)

    # Small breather so any auto-sync SCP / probe / background tasks
    # can drain before we start the timed measurement.
    await asyncio.sleep(2.0)

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": CHAT_PROMPT}],
        "stream": True,
        "options": {"temperature": 0.0, "num_predict": N_PREDICT},
    }
    # 15-minute read timeout. After warmup the timed call is hot, so
    # this is plenty even for slow split paths.
    timeout = httpx.Timeout(connect=10.0, read=900.0, write=30.0, pool=10.0)
    t0 = time.perf_counter()
    ttft = None
    tok = 0
    try:
        async with httpx.AsyncClient(timeout=timeout) as c:
            async with c.stream("POST", f"{target}/api/chat", json=payload) as r:
                if r.status_code >= 400:
                    body = await r.aread()
                    return {
                        "engine": engine,
                        "url": target,
                        "error": f"HTTP {r.status_code}",
                        "ttft_ms": 0,
                        "total_s": time.perf_counter() - t0,
                        "tokens": 0,
                        "tok_per_s": 0,
                    }
                async for line in r.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    # Count any token, whether the model emits it as
                    # `content` (regular response) or `thinking` (chain-
                    # of-thought models like qwen3.5 / DeepSeek-R1 emit
                    # reasoning into a separate field before answering).
                    # Both stream out at the same rate, so either is a
                    # valid throughput signal — and ignoring `thinking`
                    # makes thinking models look like they crashed when
                    # they're actually working fine.
                    msg = obj.get("message") or {}
                    if msg.get("content") or msg.get("thinking"):
                        if ttft is None:
                            ttft = time.perf_counter()
                        tok += 1
                    if obj.get("done"):
                        break
        total = time.perf_counter() - t0
        if tok == 0 or ttft is None:
            return {
                "engine": engine,
                "url": target,
                "error": "no tokens",
                "ttft_ms": 0,
                "total_s": total,
                "tokens": 0,
                "tok_per_s": 0,
            }
        gen_s = time.perf_counter() - ttft
        return {
            "engine": engine,
            "url": target,
            "error": None,
            "ttft_ms": (ttft - t0) * 1000,
            "total_s": total,
            "tokens": tok,
            "tok_per_s": tok / gen_s if gen_s > 0 else 0,
        }
    except Exception as e:
        return {
            "engine": engine,
            "url": target,
            "error": str(e)[:80],
            "ttft_ms": 0,
            "total_s": 0,
            "tokens": 0,
            "tok_per_s": 0,
        }


async def measure_embed():
    target = compute_pool.pick_embed_target("nomic-embed-text")
    target_label = target[0] if target else "host"
    t0 = time.perf_counter()
    err = None
    for _ in range(EMBED_RUNS):
        try:
            v = await agent._embed_text(EMBED_PROMPT)
            if not v:
                err = "no vector"
                break
        except Exception as e:
            err = str(e)[:80]
            break
    total = time.perf_counter() - t0
    return {
        "target": target_label,
        "avg_ms": (total / EMBED_RUNS) * 1000,
        "total_s": total,
        "error": err,
    }


async def main():
    if not WORKER_IDS:
        print("No workers registered; nothing to compare.")
        return
    print(f"Pool size: {len(WORKER_IDS)} worker(s)")
    models = discover_models()
    print(f"Models tested: {len(models)} chat + 1 embed")
    print(f"  prompt: {CHAT_PROMPT!r} num_predict={N_PREDICT}")
    print(f"  embed: {EMBED_RUNS}x calls, prompt={len(EMBED_PROMPT)} chars")
    print()

    print("=== Phase A: WITHOUT compute pool ===")
    set_all_workers_enabled(False)
    await compute_pool.stop_all_running_splits()
    embed_off = await measure_embed()
    print(
        f"  embed: {embed_off['avg_ms']:.0f} ms/call, "
        f"total {embed_off['total_s']:.2f}s, target={embed_off['target']}"
    )
    chat_off = {}
    for m in models:
        r = await measure_chat(m)
        chat_off[m] = r
        if r["error"]:
            print(f"  {m:30}: ERR {r['error']}")
        else:
            print(
                f"  {m:30}: {r['tokens']} tok / {r['total_s']:.2f}s "
                f"| ttft {r['ttft_ms']:.0f}ms | {r['tok_per_s']:.1f} tok/s | {r['engine']}"
            )

    print("\n=== Phase B: WITH compute pool ===")
    set_all_workers_enabled(True)
    await compute_pool.probe_all_enabled()
    embed_on = await measure_embed()
    print(
        f"  embed: {embed_on['avg_ms']:.0f} ms/call, "
        f"total {embed_on['total_s']:.2f}s, target={embed_on['target']}"
    )
    chat_on = {}
    for m in models:
        r = await measure_chat(m)
        chat_on[m] = r
        if r["error"]:
            print(f"  {m:30}: ERR {r['error']}")
        else:
            print(
                f"  {m:30}: {r['tokens']} tok / {r['total_s']:.2f}s "
                f"| ttft {r['ttft_ms']:.0f}ms | {r['tok_per_s']:.1f} tok/s | {r['engine']}"
            )

    print("\n" + "=" * 124)
    print("SIDE-BY-SIDE — all metrics")
    print("=" * 124)
    hdr = (
        f"{'MODEL':28} | "
        f"{'tok':>4} {'tot_s':>6} {'ttft_ms':>7} {'t/s':>5} {'eng':>6} | "
        f"{'tok':>4} {'tot_s':>6} {'ttft_ms':>7} {'t/s':>5} {'eng':>6} | "
        f"{'Δ t/s':>7} {'Δ tot':>7}"
    )
    print(f"{'':28} | {'WITHOUT POOL':>40} | {'WITH POOL':>40} |")
    print(hdr)
    print("-" * 124)

    e_off, e_on = embed_off, embed_on
    delta_e = (
        (e_off["avg_ms"] - e_on["avg_ms"]) / e_off["avg_ms"] * 100
        if e_off["avg_ms"] > 0
        else 0
    )
    print(
        f"{'embed (ms/call)':28} | "
        f"{'':>4} {e_off['total_s']:>5.2f}s {'':>7} {e_off['avg_ms']:>4.0f}ms {'host':>6} | "
        f"{'':>4} {e_on['total_s']:>5.2f}s {'':>7} {e_on['avg_ms']:>4.0f}ms {'pool':>6} | "
        f"{delta_e:+6.1f}% {'':>7}"
    )

    for m in models:
        off = chat_off[m]
        on = chat_on[m]
        if off["error"] or on["error"]:
            print(
                f"{m:28} | error: off={off.get('error')!s} on={on.get('error')!s}"
            )
            continue
        d_tps = (
            (on["tok_per_s"] - off["tok_per_s"]) / off["tok_per_s"] * 100
            if off["tok_per_s"] > 0
            else 0
        )
        d_tot = (off["total_s"] - on["total_s"]) / off["total_s"] * 100 if off["total_s"] > 0 else 0
        print(
            f"{m:28} | "
            f"{off['tokens']:>4} {off['total_s']:>5.2f}s {off['ttft_ms']:>6.0f}ms "
            f"{off['tok_per_s']:>4.1f} {off['engine'][:6]:>6} | "
            f"{on['tokens']:>4} {on['total_s']:>5.2f}s {on['ttft_ms']:>6.0f}ms "
            f"{on['tok_per_s']:>4.1f} {on['engine'][:6]:>6} | "
            f"{d_tps:+6.1f}% {d_tot:+6.1f}%"
        )


asyncio.run(main())
