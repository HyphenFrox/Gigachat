# llamapool-runner

Generic, **multi-tenant** compute-pool runner for llama.cpp workloads. Distribute LLM model layers across multiple machines on a LAN, with auto worker management, adaptive layer placement, per-vendor backend selection, and intelligent resource sharing when multiple apps engage the pool concurrently — without any of them needing to know about the others.

This is **just a folder of Python scripts** — no `pip install`, no package, no setup. Drop the folder anywhere, invoke the scripts directly with `python /path/to/llamapool-runner/<script>.py ...`.

## Why

Running large LLMs locally is bottlenecked by a single machine's RAM/VRAM. llama.cpp's RPC mode lets you split a model across machines, but driving it manually is tedious: start `rpc-server` on each worker with the right backend (`-d CUDA0,CPU` on NVIDIA, `-d Vulkan0,CPU` on AMD, `-d SYCL0,CPU` on Intel — except not Intel during a split, because of [an upstream SYCL+RPC bug][bug]), pick `-ngl` so layers actually fit, keep that consistent as workers come and go.

`llamapool-runner` automates that. Register your workers once. Any llama.cpp-based app on the host shells out to the wrapper, and the pool engages transparently.

[bug]: https://github.com/ggml-org/llama.cpp/issues/21420
[llamacpp]: https://github.com/ggml-org/llama.cpp

## Files

```
llamapool-runner/
├── README.md                # this file
├── run_llama_server.py      # drop-in `llama-server` wrapper — invoke this
├── cli.py                   # management CLI: add-worker, list-workers, status
├── core.py                  # engage()/disengage() — the public API for Python apps
├── config.py                # JSON-on-disk worker registry (~/.llamapool/config.json)
├── workers.py               # per-vendor backend selection, SSH+WMI rpc-server restart
├── ngl.py                   # adaptive `-ngl` from GGUF metadata + pool memory
└── registry.py              # cross-process claim tracking (~/.llamapool/active.json)
```

## Setup

You need [llama.cpp][llamacpp] installed: `llama-server` on the host, `rpc-server` on each worker. Place them on PATH or under `~/.llamapool/llama-cpp/` (the wrapper also picks up `~/.gigachat/llama-cpp/` if you have Gigachat installed).

Optional: `pip install gguf` to enable adaptive `-ngl` (without it the wrapper falls back to llama.cpp's default "fit as many as possible" probe).

## Configure your pool

Each worker needs to be running `rpc-server` (a Windows-targeted helper for restarting it via SSH+WMI ships in `workers.py`; for Linux/macOS workers manage the service yourself).

Register them on the host using the management CLI:

```bash
python /path/to/llamapool-runner/cli.py add-worker laptop-1 desktop-0692.local \
    --rpc-port 50052 --ssh-host laptop-1 --gpu-vendor intel

python /path/to/llamapool-runner/cli.py add-worker laptop-2 192.168.1.42 \
    --ssh-host gaming-rig --gpu-vendor nvidia

python /path/to/llamapool-runner/cli.py list-workers
```

Config persists at `~/.llamapool/config.json`. To auto-detect GPU vendor instead of passing `--gpu-vendor`, run `cli.py probe-worker <label>` (requires SSH access).

> Tip: alias the long path. On bash: `alias llamapool="python /path/to/llamapool-runner/cli.py"`. On Windows: a one-line `.bat` shim.

## Use it

### A) Drop-in CLI wrapper (any language)

Anywhere your stack invokes `llama-server`, swap the binary path to `run_llama_server.py`. The wrapper accepts the same args, engages the pool transparently, and forwards I/O.

```bash
# Before
llama-server -m model.gguf -c 8192 --port 11434

# After
python /path/to/llamapool-runner/run_llama_server.py -m model.gguf -c 8192 --port 11434
```

To bypass pool routing (e.g., debugging), pass `--no-pool`. To override which `llama-server` binary to spawn underneath, pass `--llama-server=/path/to/llama-server`. To set this workload's pool-share priority (see "Multi-tenant" below), pass `--pool-priority=<int>`.

### B) Python library (any Python app)

```python
import sys
sys.path.insert(0, "/path/to/llamapool-runner")
from core import engage, disengage

# `gguf_path` is your model file; pass None if you don't have it
# (you'll get rpc endpoints but no adaptive ngl).
# `priority` controls your share of pool memory when contested:
#   higher = bigger share. Default 100. Use 200 for user-facing work,
#   50 for background batch jobs.
rpc_endpoints, ngl, claim_id = engage(
    gguf_path="/path/to/model.gguf",
    priority=100,
)
try:
    import subprocess
    cmd = ["llama-server", "-m", "/path/to/model.gguf", "-c", "8192"]
    if rpc_endpoints:
        cmd += ["--rpc", ",".join(rpc_endpoints)]
    if ngl is not None:
        cmd += ["-ngl", str(ngl)]
    subprocess.run(cmd)
finally:
    # Releases this workload's share so the next workload can use it.
    # Idempotent — safe to skip on crash (the registry purges
    # dead-PID claims automatically on the next read).
    disengage(claim_id)
```

For async apps:

```python
import sys
sys.path.insert(0, "/path/to/llamapool-runner")
from core import engage_async, disengage_async

async def run():
    rpc, ngl, claim = await engage_async("/path/to/model.gguf")
    try:
        ...  # spawn and use llama-server
    finally:
        await disengage_async(claim)
```

### C) Shell scripts that build their own argv

If your script assembles `llama-server` flags procedurally, ask the runner directly:

```bash
$ python /path/to/llamapool-runner/cli.py engage /path/to/model.gguf
{
  "rpc": ["192.168.1.42:50052", "desktop-0692.local:50052"],
  "ngl": 28,
  "claim_id": "fefce8dda5774c659d3a8e96558c571b"
}
```

Pipe through `jq` to extract values into your script's command line. The claim is auto-released when the `cli.py engage` process exits — for long-running runs prefer the wrapper (Option A) which keeps the claim alive for the duration of `llama-server`.

## Multi-tenant: many apps sharing the pool

Multiple apps can engage the pool simultaneously. Each call to `engage()` registers a **claim** in `~/.llamapool/active.json` (cross-process state, file-locked). Subsequent claimants see what's already reserved and split the remainder by priority:

```
pool_free  = pool_total × safety_factor − sum(other_claims.estimated_bytes)
my_share   = pool_free × (my_priority / sum(my_priority + other_priorities))
my_ngl     = floor(my_share / avg_layer_bytes)
```

The "intelligent" part:

* **No overcommit.** A second workload computes `-ngl` against pool memory minus what the first already reserved.
* **Priority-weighted fair share.** Pass `priority=` to `engage()` (or `--pool-priority` to the wrapper). Higher priority = bigger slice of contested memory. Sensible defaults: 200 for user-facing chat, 100 for general batch, 50 for low-importance background jobs.
* **Graceful degradation.** When the pool can't fit even our share of one layer, we return `ngl=0` and an empty RPC list — caller runs **host-only** instead of overcommitting and crashing.
* **No backend bouncing.** When a peer's split is in flight, we don't restart its rpc-server out from under it: backend alignment is skipped if any peer claim is `in_split=True`.
* **Crash-safe cleanup.** Each claim records its PID; the registry auto-purges entries whose process is no longer alive on every read. A workload that exits without disengaging won't permanently reserve memory.
* **Inspect anytime:** `cli.py status` prints the live claim list (which workload, model, ngl, reserved bytes, priority).

Three apps running concurrently — Gigachat (priority 200), Peerful extraction (priority 100), a background batch (priority 50) — get pool memory roughly 4 : 2 : 1 with no manual coordination.

## Periodic rebalancing — `-ngl` adjusts as resources change

The wrapper runs a background **rebalance watcher** that re-evaluates the optimal `-ngl` on a configurable interval. When the value has drifted past a threshold AND a cooldown has elapsed, it gracefully respawns the underlying `llama-server` with the new value. The expensive part — terminate + cold-load + `/health` probe — is gated behind both checks so a noisy memory environment doesn't trigger a respawn storm.

A respawn fires only when **all three** are true:

1. `new_ngl` differs from the current value by at least `--pool-rebalance-threshold` layers (default **3**). A drift of 14 → 15 isn't worth a cold-load.
2. At least `--pool-rebalance-cooldown` seconds have elapsed since the last respawn (default **300** = 5 min). Stops oscillation when free memory bounces around a layer-count boundary.
3. The new value is non-zero — we don't tear down the running session into a "give up on the pool" state mid-flight; if the budget collapses completely, the workload keeps the layers it already has.

CLI flags:

| Flag | Default | What it does |
|---|---|---|
| `--pool-rebalance-interval=<seconds>` | 60 | How often the watcher ticks. `0` disables periodic rebalancing entirely. |
| `--pool-rebalance-threshold=<layers>` | 3 | Minimum `-ngl` delta that triggers a respawn. Raise to be more conservative. |
| `--pool-rebalance-cooldown=<seconds>` | 300 | Minimum time between respawns. Raise if you want fewer interruptions. |

What triggers a meaningful change in practice:

* A peer workload finishes (`disengage`) → its reservation returns to the pool → your share grows → ngl goes up next tick.
* A peer workload starts → reservation reduces your share → ngl goes down.
* A worker's free RAM changes (other apps quit / open on it) → next config-sync refreshes `ram_free_gb` → recompute reflects it.
* Worker comes online / goes offline → pool budget changes accordingly.

What rebalancing **doesn't** do (intentionally):

* It can't hot-migrate already-placed layers — that's an upstream llama.cpp limitation. Adjustment requires a respawn.
* It doesn't displace your other apps. The watcher reads what's free at tick time; it never asks the OS to evict another process's working set.
* It doesn't run forever in tight loops on a busy system — the cooldown ensures at least N minutes between any two respawns, so even pathological memory churn caps wrapper-induced overhead at one cold-load per cooldown window.

Resource source: the watcher reads `~/.llamapool/config.json` for worker free-memory. That JSON is refreshed by whatever owns it — Gigachat's `backend/llamapool_sync.py` runs on every CRUD operation plus a 5-minute capability probe. For purely-standalone use without Gigachat, keep the JSON fresh via your own probe loop or pass live values into the worker dicts before calling `engage`.

## Two modes: cooperative vs aggressive

There are two layers of adaptive resource usage in this runner, and they're controlled separately:

* **Inter-pool sharing** — between concurrent llama-server tasks that all use this runner. Always on. Each task registers a claim in `~/.llamapool/active.json` and gets a priority-weighted slice of the remaining pool memory. Cannot be disabled.
* **OS-cooperation** — between the whole pool and other apps the operator is running on the same machines (browsers, IDEs, the user's foreground work). Toggle with `--pool-os-cooperative=on|off`.

The toggle:

| | `--pool-os-cooperative=on` (default) | `--pool-os-cooperative=off` |
|---|---|---|
| Memory source | each machine's *free* RAM/VRAM | each machine's *total* (host VRAM ×0.95, worker RAM ×0.85) |
| Safety factor | 0.7 (30 % headroom) | 0.85 (15 % headroom — KV cache + buffers only) |
| Yields to other apps? | Yes — won't displace their working sets | No — will evict OS page cache to take what it wants |
| Rebalance watcher | On — periodically re-evaluates `-ngl` | **Off** — take what's available at start and hold it; no re-tick, no respawn |
| Right when | Operator is also using the host/workers for other work | Dedicated machines, batch jobs, operator explicitly accepts other-app slowdown |

Inter-pool sharing works identically in both modes. Two `--pool-os-cooperative=off` workloads still split the (now larger) pool memory between themselves by priority; they just both ignore non-pool OS apps. Mixing modes is allowed too — each workload's flag controls only its own engagement.

## How it works

1. **Worker discovery:** read `~/.llamapool/config.json` for the registered workers, TCP-probe each rpc-server port, drop unreachable ones.
2. **Per-vendor backend selection:** pick `-d` based on the worker's GPU vendor. NVIDIA gets `CUDA0,CPU`; AMD gets `Vulkan0,CPU`. Intel gets `SYCL0,CPU` when idle but `CPU` during a split run, because [#21420][bug] crashes rpc-server when tensors push to a SYCL backend over RPC.
3. **Lazy backend alignment:** each worker's current backend is tracked in the config. If it already matches the desired mode, we skip the SSH+WMI restart — re-engaging is cheap.
4. **Adaptive `-ngl`:** read `<arch>.block_count` from the GGUF, divide file size by layer count for avg layer bytes, sum (host VRAM via `nvidia-smi` + each worker's free RAM) with a 30 % safety margin and any peer reservations, set `-ngl` to the count that fits our priority-weighted share.
5. **Hand-off:** return `(rpc_endpoints, ngl, claim_id)` so the caller spawns `llama-server` itself. The wrapper script handles spawn + signal forwarding + auto-disengage on exit.

## Caveats

* The SSH+WMI rpc-server lifecycle is **Windows-target-specific**. On Linux/macOS workers, run `rpc-server` yourself (e.g., as a systemd service); the host will simply use whatever backend it's currently bound to.
* `gguf_path` is read from local disk — if your model file lives only on a worker, sync or mount it before calling `engage`.
* `nvidia-smi` is the only host-VRAM probe. AMD/Intel hosts will be treated as 0 host-VRAM and lean entirely on worker memory + CPU.
* **llama.cpp build alignment.** Mixed builds across host and workers can crash the RPC protocol when a `ggml_op` enum changes upstream. Keep host + every worker on the same release tag (we use `b8940` at time of writing).
* **Gemma 3n E4B (Ollama tag `gemma4:e4b` / `gemma4:latest`).** Stock llama.cpp's RPC backend can't dispatch its Gated Delta Net + PLE compute end-to-end. The host application (e.g. Gigachat) carries its own per-model spawn-flag stack to make E4B work via RPC: `-fit off` + forced `-ngl 99` + `-fa off` + `--parallel 1` + `-ot ".*(altup\|laurel\|per_layer\|inp_gate).*=<host_dev>"`. Other apps using this runner that intend to support E4B will need similar handling until the upstream RPC dispatch lands. E2B and the dense Gemmas (26B, 31B) need none of it — they engage the pool cleanly.

## Integration prompt for an external AI agent

The prompt below is self-contained — paste it to the AI agent that maintains the launcher script for any llama.cpp-based workload. Originally written for the Peerful resume-extraction agent.

> The host this script runs on has a shared LLM **compute pool** managed by `llamapool-runner` — a folder of Python scripts at a known location on the host. Other apps on the same host (e.g., Gigachat) may be running their own llama.cpp workloads concurrently with this script; pool memory must be shared without anyone tripping each other up. To use the pool, swap the `llama-server` binary path with the runner's wrapper script.
>
> **Steps:**
>
> 1. **Locate `llamapool-runner`.** It's a plain folder of `.py` files (no install, no package). Default location on this host: `%USERPROFILE%\Downloads\Gigachat\llamapool-runner` on Windows, or `~/llamapool-runner` on POSIX. Capture the path in a variable; if the folder is missing, run unchanged with the original `llama-server.exe` path (the wrapper is opt-in).
>
> 2. **Update the launcher (`scripts/run-extraction.bat`)** to invoke `python <runner>\run_llama_server.py` instead of the raw `llama-server.exe`. The wrapper accepts the *exact same* args (`-m`, `-c`, `--np`, `--batch`, `--port`, ...) and forwards stdin/stdout/stderr unchanged. It is fully transparent.
>
>    Before:
>    ```bat
>    set LLAMA_BIN=%USERPROFILE%\.gigachat\llama-cpp\llama-server.exe
>    %LLAMA_BIN% -m %MODEL% -c 14336 --np 2 --batch 2048 --port %PORT%
>    ```
>    After:
>    ```bat
>    set RUNNER=%USERPROFILE%\Downloads\Gigachat\llamapool-runner
>    if exist "%RUNNER%\run_llama_server.py" (
>        python "%RUNNER%\run_llama_server.py" --pool-priority=80 ^
>            -m %MODEL% -c 14336 --np 2 --batch 2048 --port %PORT%
>    ) else (
>        REM Runner not installed — fall back to the original binary.
>        "%USERPROFILE%\.gigachat\llama-cpp\llama-server.exe" ^
>            -m %MODEL% -c 14336 --np 2 --batch 2048 --port %PORT%
>    )
>    ```
>    Apply the same swap in any `.cjs`/`.js` that resolves the binary path itself (e.g., `extractWithLocalGemma.cjs`).
>
> 3. **Pool-share priority.** Resume extraction is a background batch job — it should yield memory to interactive workloads. Use `--pool-priority=80` (or 50). User-facing chat would use 200; default is 100. The wrapper consumes `--pool-priority` and strips it before forwarding to `llama-server`, so it never reaches llama-server's parser.
>
> 4. **(Optional)** Honor a `DEBUG_NO_POOL=1` env var by adding `--no-pool` to the args, for debug runs that should bypass the pool entirely.
>
> 5. **Verify** by running the script and watching stderr:
>    ```
>    [pool] engaging split: --rpc 192.168.1.42:50052,desktop-0692.local:50052
>    [pool] adaptive ngl: -ngl 28
>    ```
>    From a separate terminal, `python <runner>\cli.py status` should list this script's claim alongside any other active workloads.
>
> 6. **Do NOT** modify the existing `llama-server` flags, the model path, or any downstream parsing logic. The wrapper is fully transparent. Re-run the script's tests after the change to confirm extraction output is unchanged.
>
> **Why this matters:** Without the wrapper, this script would compete blindly with Gigachat (and any other concurrent llama.cpp workload) for pool memory, causing OOM crashes on whichever app started second. The wrapper registers a memory claim that other workloads see, computes `-ngl` against the *unreserved* portion of the pool, gracefully falls back to host-only inference if the pool is full, and skips bouncing rpc-server backends out from under a peer that's already mid-split.

## License

MIT.
