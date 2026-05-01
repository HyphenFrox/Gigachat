# Compute pool

Pair other Gigachat installs in **Settings → Compute pool** and the host automatically uses their CPU + RAM + GPU + VRAM alongside its own. Devices join the pool by PIN-pairing over the LAN (Bluetooth-style); after pairing, all compute traffic to them is end-to-end encrypted (see [P2P.md](./P2P.md) for the crypto details).

This document covers the routing internals — *how* the router decides which node runs each request, the `llama.cpp` flags it tunes, and the failure modes it auto-recovers from. For end-user setup see the README's [Compute pool & P2P](../README.md#compute-pool--p2p) section.

---

## Whole-request routing (Phase 1)

Each chat / embedding / parallel-subagent call goes to ONE machine — whichever is best for it. The router ranks every eligible node (host included) on a 6-axis capability score and picks the strongest. **The host is just one candidate**, not a privileged default — if a registered worker is strictly more capable than the host (faster measured tokens/sec, more VRAM, etc.), chat goes there.

The score, in priority order:

1. **Measured throughput** — real `tokens/sec` benchmarked via `/api/generate` (cached 1 hour). The bottom-line "how fast does this machine actually run this model" number; folds CPU + memory bandwidth + GPU compute into a single signal. Used as the primary key whenever both sides have measurements; falls back to the heuristic axes below otherwise.
2. **GPU presence** — binary signal from `/api/ps` (any loaded model with VRAM > 0).
3. **Proven VRAM** — `max(size_vram)` across loaded models. Hard lower bound on capacity.
4. **Total RAM** (workers via SSH probe; host via sysdetect).
5. **CPU threads** — for Phase 2 splits where rpc-server runs CPU layers.
6. **Last-seen freshness** — final tie-breaker; host always wins ties (no LAN hop).

Other gates the router applies before scoring:

- **Model availability** — the picked node has to have the model installed; otherwise it's skipped.
- **Strictly more capable than host** — among eligible workers, only one with a strictly higher score wins; ties go to host (KV cache stays warm).

Eligibility per worker is fine-grained: `Use for chat / embeddings / subagents` toggles per row let you split workloads (e.g. embeddings → laptop, chat stays on host).

### Locality-aware near-tie balancing

When N peers all have the same model installed, the router prefers **LAN-paired peers** (sub-millisecond RTT) over **public-pool peers** (hundreds of ms across the internet) — unless the public peer is measurably more capable. Within the same locality tier, the less-loaded peer wins so concurrent conversations spread across the pool instead of stacking on the strongest single node.

---

## Layer-split routing (Phase 2)

When the model is too big for any single node — e.g. a 27 B Q4 model that doesn't fit either machine alone — the router automatically engages [llama.cpp](https://github.com/ggml-org/llama.cpp)'s `--rpc` mechanism:

```
[Host CUDA + RAM]  ─RPC─►  [Worker iGPU/CPU + RAM]
                           [Worker dGPU + VRAM + RAM]
                           ...
       ▲                          ▲
       └── llama-server reads GGUF locally on host,
           streams layer weights to each rpc-server
           over LAN. Layers cascade VRAM → RAM → next
           node, native to llama.cpp.
```

Workers do **not** need the model installed for the split path — bytes stream over RPC from the host's local GGUF blob.

The split decision is automatic: if the model fits the strongest single node, Phase 1 wins (faster, no per-token LAN overhead). If it doesn't, llama-server is auto-spawned with `--rpc <worker>:50052,...` flags pointing at every reachable worker. Subsequent turns reuse the warm process. Switching to a different big model auto-stops the previous llama-server (one big model hot at a time — finite VRAM).

**Worker devices expose both iGPU + CPU.** rpc-server is launched with `-d SYCL0,CPU` on each worker so llama.cpp sees TWO devices per laptop — the Intel iGPU (via SYCL) and the system CPU (via system RAM). Net effect: a 2-laptop pool contributes ~14 GB of usable memory beyond what each laptop's iGPU alone would offer, AND the auto-distribution can place the heaviest layers on whichever device has free memory.

**Adaptive `-ngl`.** Before spawning llama-server the backend reads the GGUF's `block_count` metadata, divides total file size by layer count to estimate bytes-per-layer, and computes how many layers fit in the sum of pool free memory. **5 % safety margin** — small enough that nearly all VRAM still goes to inference, big enough to absorb allocator alignment overhead and avoid load-time OOM on overcommit. Layers beyond that stay on host CPU paged from the GGUF via mmap.

**Adaptive `--n-cpu-moe` for big-MoE models.** For Mixture-of-Experts models (DeepSeek-V3, Qwen3-MoE 235B, GPT-OSS 120B) that don't fit GPU even with `-ngl` tuned, expert FFN tensors are pinned to CPU/RAM while attention and dense weights stay on GPU. Each token only routes through a handful of experts per layer, so the GPU↔CPU activation hop costs less than streaming weights from disk via mmap.

### Inference-performance flags wired by default

| Flag | What it does |
|---|---|
| `--flash-attn on` | Fused attention kernel — 5-30 % faster generation; bigger gains at long context. Also a prerequisite for KV cache quantisation. |
| `-ctk q8_0 -ctv q8_0` | KV cache quantised to Q8 — halves the per-slot KV footprint at <1 % accuracy loss vs FP16. Frees VRAM for more parallel slots, longer context, or fewer layers offloaded to CPU. |
| `--cache-reuse 256` | KV-shift-based prompt-prefix reuse across follow-up turns. When a chat shares a prefix with the previous turn (same system prompt, same first N rounds), the server skips re-decoding that prefix and shifts the cached KV state forward. |
| `-b … -ub …` | Adaptive prompt-eval batch / ubatch sizes. Computed against free VRAM after weights + KV; ceiling raised to 8192. Big-VRAM workstations get 2-4× faster prefill on long prompts. |
| `--mlock` | Engaged when free RAM ≥ 2× the model file. Pins weights so the OS can't page them out under memory pressure. |
| `--threads N --threads-batch M` | Decode threads = physical core count (avoids SMT FPU contention); prefill threads = logical core count (batched matmul scales cleanly with SMT). |
| `--parallel N` | Continuous batching — N concurrent decoding slots sharing the same warm engine. Auto-tuned from GGUF metadata + free pool memory. |

Ollama path picks up the same family of optimisations via env vars: `OLLAMA_FLASH_ATTENTION=1`, `OLLAMA_KV_CACHE_TYPE=q8_0`, `OLLAMA_NUM_PARALLEL` (hardware-tuned), `OLLAMA_NUM_THREAD` (physical cores), `OLLAMA_KEEP_ALIVE=60m`, `OLLAMA_SCHED_SPREAD=1` (multi-GPU layer distribution), `OLLAMA_GPU_OVERHEAD` set to 5 % of detected VRAM. User-set values always win.

---

## Speculative decoding (recruit the rest of the pool)

Layer-split is for models the strongest single node can't fit. But the common case is the opposite: model fits one node, and the rest of the pool sits idle while that node runs the chat alone. Speculative decoding fixes that — the picker auto-selects a smaller vocab-compatible chat model from anywhere in the pool's combined inventory, then loads it as a draft alongside the target. The draft proposes a few cheap tokens per round, the target verifies them in a single batched pass, and net throughput on a single chat stream typically goes up 1.3–2× depending on accept rate.

**Default: ON.** The picker has its own gates so leaving it on is a no-op for setups that can't benefit. To force the legacy Ollama-only path:

```
POST /api/settings  {"compute_pool_speculative_decoding": "false"}
```

### Candidate matching

The picker is generic — nothing is hard-coded to a particular model or device. It walks every enabled worker's `/api/tags` snapshot plus the host's Ollama manifest store and accepts a candidate via three tiers, cheapest first:

1. **Manual override** — pinned via `compute_pool_speculative_overrides` (a JSON-encoded `{"<target>": "<draft>"}` map). Trusted unconditionally.
2. **Family match** — same Ollama-reported `details.family` (`llama` ↔ `llama`, `qwen2` ↔ `qwen2`, `gemma` ↔ `gemma`).
3. **Tokenizer fingerprint match** — different families but identical GGUF tokenizers, parsed via the official [`gguf`](https://pypi.org/project/gguf/) library. Catches cross-family pairs (some Mistral derivatives that adopt the Llama tokenizer).

A candidate that passes any tier still has to satisfy:

* dramatically smaller than the target (≤ 30 % of target size),
* lives on the host's local disk OR is auto-syncable from a worker.

### Engagement gates

Tier 1 of `route_chat_for` (the model fits one node) checks the picker before falling through to plain Ollama. Engagement requires:

* `compute_pool_speculative_decoding` is on (default).
* Picker returned a draft.
* Host has VRAM headroom for both models: `(target_size + draft_size) × 1.30 ≤ host VRAM budget`.
* `llama-server` is installed locally (one-click install in Settings → Compute pool).

Below any of those thresholds, the router stays on plain Ollama with no fanfare.

| Setup | Without speculative | With speculative | Notes |
|---|---|---|---|
| 8B target + 1B same-family draft on 16 GB GPU | 30 tok/s | ~50 tok/s | typical 1.5–2× on chat-heavy prompts |
| 13B target + 1B draft on 16 GB GPU | 18 tok/s | ~32 tok/s | scales with target/draft ratio |
| 70B target on host that can't even fit it | already Tier 2 split | layer-split + draft layered on top | speculative + RPC stack cleanly |

---

## Other pool-saturation features

All always-on. Each has its own engagement gate so it kicks in only when the pool would actually win.

- **Adaptive split-vs-host routing** — when a model fits one node, `route_chat_for` still considers Phase 2 (layer-split). It engages iff the pool VRAM is ≥ 1.5× the strongest single node's VRAM.
- **Round-robin embeddings** — `pick_embed_target` rotates across every worker that has the embedding model loaded and benches within 50 % of the leader's TPS.
- **Distributed `fetch_url` / `web_search` / `read_doc` / `python_exec` dispatch** — eligible tool calls SSH into a round-robin-picked worker via PowerShell. Frees host CPU for inference.
- **Worker-side llama-server** — when the strongest single node is a worker AND the worker has `llama-server.exe` installed AND a same-family draft GGUF is in the worker's Ollama inventory, the host SSH-spawns llama-server on the worker. Both target and draft live on the worker (no per-token RPC overhead).
- **Adaptive continuous batching** — `--parallel N` slots auto-tuned from GGUF metadata + free pool memory.
- **Tensor-split (`-ts`) auto-weights** — for heterogeneous pools where host VRAM and worker capacity differ by ≥ 1.5×.
- **Tensor-parallel row dispatch (`--split-mode row`)** — engaged on Gigabit-Ethernet-class LANs (every worker probe latency ≤ 4 ms).
- **MoE expert auto-pin to CPU** — for models where the geometry would OOM with default placement.
- **Pool-distributed embed fan-out** — `compute_pool.embed_concurrency_limit(model)` caps at 2× the number of usable embed backends, indexers run under that semaphore.
- **KV cache quantization** — `_decide_kv_precision_and_parallel` picks FP16 vs Q8 jointly with the `--parallel` slot count.
- **Adaptive context window** — `_compute_optimal_ctx_size` reads the GGUF's max context, the chosen `--parallel` count, and the bottleneck-node free memory.
- **Prompt-cache-to-disk reload** — `llama-server` is spawned with `--prompt-cache <path> --prompt-cache-all` keyed by GGUF + KV precision.
- **Per-conversation worker affinity** — `pick_chat_target(model, conv_id=...)` consults `_CONV_AFFINITY` so a chat that landed on worker-A on its previous turn keeps landing there as long as A is eligible.
- **Pool-distributed compaction** — auto-compaction summaries route through `pick_compaction_target(model)`.
- **Background idle re-indexing** — every 5-minute probe sweep walks every `status=ready` codebase index and re-embeds modified files.

---

## Auto-pull from official source (consumer side of public pool)

When the user picks a model and **neither host nor any pool peer has it**, the executing machine pulls it from the OFFICIAL Ollama registry (`registry.ollama.ai`) — NOT peer-to-peer. Model bytes never traverse another user's home internet, so the swarm doesn't bottleneck on the slowest uploader. Auto-pull progress streams to the UI via SSE so the user sees `Downloading model… 23% (1.2/5.4 GB)` instead of a silent multi-minute hang.

---

## KV-cache resume across worker restarts

When a worker dies mid-stream (HTTP error, connection reset), the routing layer marks it transiently unhealthy via `last_error`. The probe loop clears the flag once the worker comes back; until then `_is_fresh` excludes it from eligibility. The user's next message routes to a healthy node automatically — no manual retry-routing needed. The partial response is persisted with a `[stream interrupted]` marker, and a `stream_interrupted` SSE event surfaces a Sonner warning so the user knows what happened.

---

## Smaller-variant suggestions when a model exceeds the pool

When the user picks a model too big to run usefully even with split engaged (model bytes > combined pool memory), instead of silently degrading to disk-paged 0.1 tok/s inference, `route_chat_for` returns a `mega_model_warning` SSE event with up to 6 same-family alternatives the user already has at smaller sizes (sorted largest-fitting first). Quality is preserved — never silent quantization, only real same-family models designed at smaller sizes. See `find_smaller_variants_in_family()` in `compute_pool.py`.

---

## Override-file mechanism + auto-acquisition (Scope B)

Some Ollama-distributed multimodal models bundle the vision tower into the same GGUF as the text LLM (e.g. `gemma4:26b`'s 16.75 GB blob includes 354 vision tensors that stock `llama-server` doesn't recognize). The pool supports a generic GGUF override at `~/.gigachat/llama-cpp/models/<sanitized-model-name>.gguf` — when present, `resolve_ollama_model` returns this file instead of Ollama's blob.

Override files are produced/acquired automatically the first time the user requests an affected model:

1. **LAN copy** — if any enabled worker (`ssh_host` set) already has the override file in its own `~/.gigachat/llama-cpp/models/` from a previous distribution, it's pulled via SCP. Zero internet bandwidth.
2. **Local surgery / metadata patch** — when the Ollama blob is mostly correct but a structural attribute needs fixing, a tiny repack script produces the override:
   * `scripts/repack_text_only_gguf.py` — extracts the text LLM tensors from a bundled-multimodal blob (e.g. `gemma4:26b`, `gemma4:31b`). Bit-identical filter-and-copy.
   * `scripts/repack_qwen3_rope_fix.py` — extends `qwen3.5`'s `rope.dimension_sections` array. Lossless metadata patch.
   * `scripts/repack_gemma3_norm_fix.py` — injects the missing `gemma3.attention.layer_norm_rms_epsilon = 1e-6` key Ollama's `gemma3:*` blobs ship without. Lossless metadata patch.
3. **HuggingFace download** — fallback for cases that don't lend themselves to local surgery (e.g. `gemma4:e4b` / `gemma4:e2b` are really Gemma 3n with the wrong arch label and PLE structures Ollama can't trivially repack — Unsloth's clean `gemma-3n-E*B-it-GGUF` Q4_K_M is the source of truth). Pre-built mmproj GGUFs (CLIP format) also come from upstream.

After each successful local acquisition, the override files are distributed to every enabled worker in the pool (fire-and-forget background SCP).

The chat layer surfaces acquisition progress via a `preparing_model` SSE event with status (`surgery` / `downloading-main` / `downloading-mmproj`) and progress percentage.

---

## Known llama.cpp workarounds

### SYCL+RPC bug — dynamic worker backend

llama.cpp has an open bug ([#21420](https://github.com/ggml-org/llama.cpp/issues/21420), [#20259](https://github.com/ggml-org/llama.cpp/issues/20259), [#21474](https://github.com/ggml-org/llama.cpp/issues/21474)) where pushing layer tensors over RPC to a worker's SYCL backend crashes the rpc-server.

**Workaround in `compute_pool._ensure_split_running_for`**: when a Phase 2 split is about to spawn, every worker's rpc-server is restarted with `-d CPU` (no SYCL exposed). Workers contribute via system RAM + CPU compute; the host still does GPU compute via CUDA. After the split stops (`stop_all_running_splits`), workers restore to the default `-d SYCL0,CPU` so non-split paths get full iGPU acceleration.

### Gemma 3n PLE multi-flag workaround

Google's Gemma 3n E4B (Ollama-tagged `gemma4:e4b` / `gemma4:latest`) uses Per-Layer Embeddings + MatFormer + Gated Delta Net — a compute graph stock llama.cpp's RPC backend can't dispatch end-to-end. Three failure modes stack on top of each other (assertion in auto-fit pre-pass, in `sched_reserve`, and rpc-server crash during slot init).

`split_lifecycle._build_command` detects E4B via `_model_needs_fit_off()` and stacks the workarounds **only for that case**:

| Flag added | Reason |
|---|---|
| `-fit off` | Skip the auto-fit pre-pass. |
| Forced `-ngl 99` | Push every layer to a GPU/RPC device. |
| `-fa off` | Disable Flash Attention; no RPC-serializable counterpart for GDN. |
| `--parallel 1` | Single-slot dispatch — multi-slot init triggers the rpc-server crash. |
| `-ot ".*(altup\|laurel\|per_layer\|inp_gate).*=<host_dev>"` | Pin Gemma 3n's PLE / AltUp / Laurel / InputGate tensors to the host. |

---

## Empirical results

1-host + 2-laptop pool:

| Model | Without pool (host alone) | With pool (split path) | Note |
|---|---|---|---|
| gemma4:31b (~18.5 GB) | host OOM | 1.85 tok/s | only viable via pool |
| gemma4:26b A4B (~16.8 GB) | host OOM | 11.33 tok/s | only viable via pool |
| dolphin-mixtral:8x7b (24.6 GB) | 0.7 tok/s (CPU offload + disk paging) | 2.4 tok/s | **3.6×** |
| gemma4:e2b (3 GB) | host fits | 11.82 tok/s | pool engages cleanly |
| qwen3.5:9b (5.3 GB) | 15.0 tok/s (host fits) | 16.7 tok/s | +11% |

The pool's value is the top three rows — large models that the host can't load on its own become possible at meaningful speed via layer-split. Small models generally don't need the pool; the router keeps them host-only by default.

---

## Setup, per machine

- **Worker side** — Ollama installed and listening on `0.0.0.0:11434` (set `OLLAMA_HOST` env var, allow port 11434 on the Private firewall profile). Optional but enables Phase 2: install [llama.cpp's prebuilt build](https://github.com/ggml-org/llama.cpp/releases) at `~/.gigachat/llama-cpp/`, run `rpc-server.exe --host 0.0.0.0 --port 50052`. Allow port 50052 on the Private firewall profile.
- **Host side** — pair the worker via Settings → Compute pool → Pair new device. The PIN handshake auto-creates the routing entry; no manual IP entry. One-click "Install llama.cpp" in the Compute panel auto-detects the host's GPU vendor and fetches the matching upstream build (CUDA / ROCm / SYCL / Metal / CPU).

The Settings panel shows live status pills per device (Ollama version, model count, GPU detection). The panel polls capabilities every 5 minutes; click 🔄 on any row to probe immediately.

### Heads-up: Smart App Control on Windows 11

Windows 11's Smart App Control blocks unsigned executables — the prebuilt llama.cpp binaries won't run on a worker that has SAC enabled. Either disable SAC on the worker (Windows Security → App & browser control → Smart App Control → Off) or build llama.cpp from source and sign it yourself. Phase 1 routing (Ollama-only) works regardless of SAC since Ollama ships signed.
