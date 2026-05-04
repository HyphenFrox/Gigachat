# Gigachat patches for llama.cpp

These patches apply on top of upstream llama.cpp **b9002** (commit
`457e2288c` — `sync : ggml`) and ship the runtime resilience that the
Gigachat app expects from its bundled llama-server / rpc-server
binaries.

The patched binaries live at `~/.gigachat/llama-cpp/` once installed,
identified by the presence of `gigachat_patch_marker.txt`. Gigachat
detects this marker at every spawn and warns the user when stock
binaries are detected (see `backend/p2p_llama_server.py:
is_patched_llama_cpp_installed`).

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

### `tools/server/server-context.cpp`

The decode loop wraps `llama_decode(ctx, batch_view)` in a try/catch
for `std::runtime_error`. On `rpc_remote_failure`, every active slot
gets a clean error response and the next request can re-try.

## Build

```cmd
git clone --depth 1 --branch b9002 https://github.com/ggml-org/llama.cpp.git
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
