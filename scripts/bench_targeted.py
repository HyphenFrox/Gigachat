"""Targeted bench for the four models the user explicitly asked about:

  qwen3.5:9b
  gemma4:26b
  gemma4:31b
  dolphin-mixtral:8x7b

Runs each WITHOUT the compute pool (workers disabled, host-only via
Ollama) and WITH the compute pool (split path engaged when host can't
hold the model). Same metrics as `bench_compute_pool.py` — tokens,
ttft, total wall-time, sustained tok/s — measured on the hot path
after warmup, so cold-load and Phase 2 layer-push don't pollute the
numbers.

Counts both `message.content` and `message.thinking` tokens (the
qwen3.5:9b chain-of-thought stream goes to `thinking` first).
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx

from backend import compute_pool, db


MODELS = ["qwen3.5:9b", "gemma4:26b", "gemma4:31b", "dolphin-mixtral:8x7b"]
PROMPT = "Count from 1 to 30, numbers only, separated by spaces."
N_PREDICT = 50

WORKER_IDS = [w["id"] for w in db.list_compute_workers()]


def set_all_workers_enabled(enabled: bool) -> None:
    """Enable/disable every registered worker — needed when more than
    one is in the pool. Without this we'd be measuring a half-disabled
    pool, which is not a meaningful comparison."""
    for wid in WORKER_IDS:
        db.update_compute_worker(wid, enabled=enabled)


async def _warmup_chat(model: str, target: str, engine: str) -> None:
    """One-token call so the model is hot in VRAM/RAM (or split-loaded
    across the pool) before the timed measurement starts. Engine-aware
    so llama-server doesn't 404 on /api/chat."""
    if engine == "llama_server":
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": "ok"}],
            "stream": False,
            "max_tokens": 1,
            "temperature": 0.0,
        }
        url_path = "/v1/chat/completions"
    else:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": "ok"}],
            "stream": True,
            "options": {"temperature": 0.0, "num_predict": 1},
        }
        url_path = "/api/chat"
    timeout = httpx.Timeout(connect=10.0, read=900.0, write=30.0, pool=10.0)
    try:
        async with httpx.AsyncClient(timeout=timeout) as c:
            r = await c.post(f"{target}{url_path}", json=payload)
            _ = r.text  # drain
    except Exception:
        pass  # warmup failures don't fail the bench


async def measure_chat(model: str) -> dict:
    print(f"  starting {model} ...", flush=True)
    try:
        decision = await compute_pool.route_chat_for(model)
    except Exception as e:
        return {
            "engine": "ROUTE_ERR", "url": "", "error": str(e)[:120],
            "ttft_ms": 0, "total_s": 0, "tokens": 0, "tok_per_s": 0,
        }
    target = decision.get("base_url") or "http://localhost:11434"
    engine = decision.get("engine") or "?"

    # Pre-populate host TPS cache so background bench doesn't contend
    # with our timed call.
    try:
        await compute_pool._measure_host_throughput(model)
    except Exception:
        pass

    print(f"    warmup ({engine}) ...", flush=True)
    t_warm = time.perf_counter()
    await _warmup_chat(model, target, engine)
    print(f"    warmup done in {time.perf_counter() - t_warm:.1f}s", flush=True)
    await asyncio.sleep(2.0)

    # Route to the right API based on the engine. Ollama exposes
    # `/api/chat` with options.{temperature,num_predict}; llama-server
    # exposes the OpenAI-compatible `/v1/chat/completions` with
    # max_tokens / temperature at the top level. Mixing these produces
    # HTTP 404 from llama-server (no /api/chat route) — caused our
    # earlier dolphin-mixtral bench to spuriously fail.
    if engine == "llama_server":
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": PROMPT}],
            "stream": True,
            "temperature": 0.0,
            "max_tokens": N_PREDICT,
        }
        url_path = "/v1/chat/completions"
    else:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": PROMPT}],
            "stream": True,
            "options": {"temperature": 0.0, "num_predict": N_PREDICT},
        }
        url_path = "/api/chat"
    timeout = httpx.Timeout(connect=10.0, read=900.0, write=30.0, pool=10.0)
    t0 = time.perf_counter()
    ttft = None
    tok = 0
    try:
        async with httpx.AsyncClient(timeout=timeout) as c:
            async with c.stream("POST", f"{target}{url_path}", json=payload) as r:
                if r.status_code >= 400:
                    body = await r.aread()
                    return {
                        "engine": engine, "url": target,
                        "error": f"HTTP {r.status_code}",
                        "ttft_ms": 0, "total_s": time.perf_counter() - t0,
                        "tokens": 0, "tok_per_s": 0,
                    }
                async for line in r.aiter_lines():
                    line = line.strip()
                    if not line:
                        continue
                    # OpenAI-style streams prefix lines with "data: ";
                    # Ollama sends raw JSON. Detect + strip prefix.
                    if line.startswith("data:"):
                        body = line[5:].strip()
                        if body == "[DONE]":
                            break
                    else:
                        body = line
                    try:
                        obj = json.loads(body)
                    except Exception:
                        continue
                    # Ollama: {"message": {"content": "..."}, "done": ...}
                    # OpenAI: {"choices": [{"delta": {"content": "..."}}]}
                    msg = obj.get("message") or {}
                    delta_text = msg.get("content") or msg.get("thinking")
                    if not delta_text:
                        choices = obj.get("choices") or []
                        if choices:
                            delta = choices[0].get("delta") or {}
                            delta_text = delta.get("content")
                    if delta_text:
                        if ttft is None:
                            ttft = time.perf_counter()
                        tok += 1
                    if obj.get("done"):
                        break
        total = time.perf_counter() - t0
        if tok == 0 or ttft is None:
            return {
                "engine": engine, "url": target, "error": "no tokens",
                "ttft_ms": 0, "total_s": total, "tokens": 0, "tok_per_s": 0,
            }
        gen_s = time.perf_counter() - ttft
        return {
            "engine": engine, "url": target, "error": None,
            "ttft_ms": (ttft - t0) * 1000,
            "total_s": total,
            "tokens": tok,
            "tok_per_s": tok / gen_s if gen_s > 0 else 0,
        }
    except Exception as e:
        return {
            "engine": engine, "url": target, "error": str(e)[:120],
            "ttft_ms": 0, "total_s": 0, "tokens": 0, "tok_per_s": 0,
        }


async def main():
    if not WORKER_IDS:
        print("No workers registered; nothing to compare.")
        return
    print(f"Pool size: {len(WORKER_IDS)} worker(s)")
    print(f"Models: {len(MODELS)}")
    print(f"  prompt: {PROMPT!r}  num_predict={N_PREDICT}")
    print()

    print("=== Phase A: WITHOUT compute pool (workers disabled) ===")
    set_all_workers_enabled(False)
    await compute_pool.stop_all_running_splits()
    chat_off = {}
    for m in MODELS:
        r = await measure_chat(m)
        chat_off[m] = r
        if r["error"]:
            print(f"  {m:30}: ERR {r['error']}")
        else:
            print(
                f"  {m:30}: {r['tokens']} tok / {r['total_s']:.2f}s "
                f"| ttft {r['ttft_ms']:.0f}ms | {r['tok_per_s']:.1f} tok/s "
                f"| {r['engine']}"
            )

    print()
    print("=== Phase B: WITH compute pool (workers enabled) ===")
    set_all_workers_enabled(True)
    await compute_pool.probe_all_enabled()
    chat_on = {}
    for m in MODELS:
        r = await measure_chat(m)
        chat_on[m] = r
        if r["error"]:
            print(f"  {m:30}: ERR {r['error']}")
        else:
            print(
                f"  {m:30}: {r['tokens']} tok / {r['total_s']:.2f}s "
                f"| ttft {r['ttft_ms']:.0f}ms | {r['tok_per_s']:.1f} tok/s "
                f"| {r['engine']}"
            )

    # Side-by-side summary.
    print()
    print("=" * 124)
    print("SIDE-BY-SIDE - all metrics")
    print("=" * 124)
    print(
        f"{'MODEL':28} | "
        f"{'tok':>4} {'tot_s':>6} {'ttft_ms':>7} {'t/s':>5} {'eng':>6} | "
        f"{'tok':>4} {'tot_s':>6} {'ttft_ms':>7} {'t/s':>5} {'eng':>6} | "
        f"{'d t/s':>7} {'d_tot':>7}"
    )
    print(f"{'':28} | {'WITHOUT POOL':>40} | {'WITH POOL':>40} |")
    print("-" * 124)
    for m in MODELS:
        off = chat_off[m]
        on = chat_on[m]
        if off.get("error") or on.get("error"):
            print(f"{m:28} | error: off={off.get('error')!s} on={on.get('error')!s}")
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
