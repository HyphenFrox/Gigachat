# Gigachat patches for llama.cpp

These patches apply on top of upstream llama.cpp **b9030** (commit
`a09a00e` — `vendor : update cpp-httplib to 0.43.3 (#22686)`) and
ship the runtime resilience that the Gigachat app expects from its
bundled llama-server / rpc-server binaries. The same patch applied
cleanly to b9002 — RPC code is unchanged between those builds — so
the only thing changed in the b9030 rebuild is the underlying
llama.cpp tag.

**Pre-built Windows x64 binaries are published as a GitHub Release**
([`gigachat-llamacpp-b9030-1`](https://github.com/HyphenFrox/Gigachat/releases/tag/gigachat-llamacpp-b9030-1)).
`install.bat` / `install.sh` auto-fetches them at install time —
end users on Windows x64 don't need to build anything from source.
The instructions below are for users on Linux / macOS, or anyone who
wants to rebuild against a newer upstream tag (and update the release
sha256 in `backend/p2p_llama_server.py:_PATCHED_RELEASE_SHA256`).

The patched binaries live at `~/.gigachat/llama-cpp/` once installed,
identified by the presence of `gigachat_patch_marker.txt`. Gigachat
detects this marker at every spawn (see `backend/p2p_llama_server.py:
is_patched_llama_cpp_installed`); when absent the auto-installer is
re-invoked.

## What's patched

### `ggml/src/ggml-rpc/ggml-rpc.cpp`

`RPC_STATUS_ASSERT` no longer hard-aborts the process via
`GGML_ABORT` (which on Windows triggers `__fastfail` with code
`STATUS_STACK_BUFFER_OVERRUN` — a process kill with no recovery).
Instead it throws a recoverable `rpc_remote_failure` exception
(subclass of `std::runtime_error`) that the patched llama-server
catches around `llama_decode`. The chat layer in
`agent._stream_llama_server_chat` then auto-retries the request once
on the resulting 5xx response, so a transient RPC failure becomes a
brief stutter instead of a broken chat.

The buffer-free path is special-cased: it logs and continues silently
instead of throwing, because throwing through ggml's C-only backend
dispatcher would terminate the process.

### `ggml/src/ggml-rpc/transport.cpp`

The bare `recv()` / `send()` loops now retry on transient errno values
(`EAGAIN`, `EWOULDBLOCK`, `EINTR`, plus the WSA equivalents) with a
short backoff. Up to 8 attempts with 5–40 ms sleeps. Hard-fail errors
still propagate as `false`.

### `ggml/src/ggml-sycl/common.hpp`

`ggml_sycl_error` no longer hard-aborts via `GGML_ABORT("SYCL error")`
(which on Windows is `__fastfail(STATUS_STACK_BUFFER_OVERRUN)` — a
hard process kill that no exception handler can intercept). It now
throws `std::runtime_error("sycl_remote_failure: ...")` so the
exception can propagate to a higher-level handler. The `[[noreturn]]`
attribute is dropped (throw IS noreturn from the caller's view but
the attribute would prevent proper unwind table emission). This
addresses Intel level_zero `UR_RESULT_ERROR_DEVICE_LOST` events that
auto-recover within ~100-500 ms (Windows TDR / driver state restore)
— the previous abort gave them no chance.

### `ggml/src/ggml-sycl/ggml-sycl.cpp`

Every SYCL handler's catch block (`ggml_check_sycl`,
`ggml_backend_sycl_buffer_set_tensor`, `ggml_backend_sycl_buffer_get_tensor`,
`ggml_backend_sycl_compute_forward`, etc.) throws `std::runtime_error`
instead of calling `std::exit(1)`. Same rationale: let the exception
propagate to the higher-level retry / catch site instead of killing
the process.

### `tools/server/server-context.cpp`

The decode loop wraps `llama_decode(ctx, batch_view)` in a try/catch
for `std::runtime_error`. On `rpc_remote_failure`, every active slot
gets a clean error response and the next request can re-try.

### `ggml/src/ggml-rpc/ggml-rpc.cpp` `rpc_server::set_tensor`

Wraps the call to `ggml_backend_tensor_set` in a 3-attempt retry
loop with 250 ms / 500 ms backoff, catching `std::exception`. The
retry catches the patched `ggml_sycl_error`'s thrown
`std::runtime_error` AND any other backend exception. On final
failure, returns `false` cleanly — the dispatch loop in
`rpc_serve_client` closes the connection but the rpc-server PROCESS
stays alive, ready to serve the next chat. Without the retry, the
rpc-server would die on the first DEVICE_LOST and the orchestrator's
connection would break mid-load.

Net effect: a transient iGPU DEVICE_LOST during weight upload
becomes a 250-750 ms stall in the rpc-server while SYCL recovers,
then the upload succeeds. The orchestrator sees no failure and
the chat proceeds normally with all iGPUs (host + Naresh)
contributing to the split.

## Build

```cmd
git clone --depth 1 --branch b9030 https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
git apply ../vendor/llama.cpp-patches/gigachat-rpc-resilience.patch
mkdir build-sycl && cd build-sycl
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release ^
  "-DCMAKE_C_COMPILER=icx" "-DCMAKE_CXX_COMPILER=icx" ^
  -DGGML_SYCL=ON -DGGML_RPC=ON -DGGML_NATIVE=OFF ^
  -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_CURL=OFF
cmake --build . --config Release -j 6 --target llama-server rpc-server
```

After build, copy the contents of `build-sycl/bin/` into
`~/.gigachat/llama-cpp/`, layer in the matching Intel oneAPI runtime
DLLs (`sycl8.dll`, `mkl_*`, `tbb12.dll`, `ur_*`, etc.) from
`C:\Program Files (x86)\Intel\oneAPI\compiler\2025.3\bin\`, and drop
a `gigachat_patch_marker.txt` to mark the install as patched.

## Disabling at runtime

Define `GIGACHAT_RPC_STRICT_ABORT` at compile time to restore upstream
behaviour (every assertion crashes the process). Useful while
debugging the RPC protocol itself.
