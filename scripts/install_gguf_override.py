"""Install a replacement GGUF that overrides Ollama's blob for Phase 2 split.

Why this exists
---------------
Some models can't be loaded by stock llama.cpp using Ollama's GGUF blob,
even when the architecture is supported. Common causes:

  - Fused vs unfused MoE expert tensors (Ollama's `gemma4:26b` packs
    `ffn_gate_up_exps` per layer; llama.cpp's loader spec expects them
    unfused as `ffn_gate_exps` + `ffn_up_exps` separately, leading to
    "wrong number of tensors; expected N, got M" on load).
  - Quantization formats Ollama uses internally that aren't in the
    stable GGUF format spec.
  - Older blobs that predate a bugfix in llama.cpp's loader.

The compute_pool's `resolve_ollama_model` checks for an override at
`~/.gigachat/llama-cpp/models/<sanitized-name>.gguf` BEFORE returning
Ollama's blob. Drop a working GGUF there and Phase 2 split picks it up
automatically - no other code changes.

What this script does
---------------------
1. Streams a GGUF from a given HuggingFace direct-download URL to
   `~/.gigachat/llama-cpp/models/<sanitized-model-name>.gguf`.
2. Resumes a partial download via Range header if `<dest>.part` exists,
   useful when a 20+ GB transfer drops mid-stream.
3. Renames `.part` -> final on success (atomic - never leaves a half-
   written file at the override path).

Usage
-----
    python scripts/install_gguf_override.py \\
        --model-name gemma4:26b \\
        --url https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF/resolve/main/gemma-4-26B-A4B-it-UD-Q5_K_M.gguf

    python scripts/install_gguf_override.py --dry-run --model-name foo:bar --url https://...

Verified URLs (copy-paste targets)
----------------------------------
    # Gemma 4 26B-A4B IT - Unsloth, all variants:
    #   UD-Q5_K_M (~19.7 GB) - default Unsloth recommendation
    #   UD-Q6_K   (~21.6 GB) - negligible quality drop
    #   Q8_0      (~25.0 GB) - non-UD, effectively lossless
    https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF/resolve/main/gemma-4-26B-A4B-it-UD-Q5_K_M.gguf

    # Dolphin Mixtral 2.7 8x7B - TheBloke, all variants:
    #   Q3_K_M    (~19.0 GB)
    #   Q4_K_M    (~24.6 GB) - same as Ollama's default blob
    #   Q5_K_M    (~30.0 GB)
    #   Q6_K      (~35.7 GB)
    #   Q8_0      (~46.2 GB)
    https://huggingface.co/TheBloke/dolphin-2.7-mixtral-8x7b-GGUF/resolve/main/dolphin-2.7-mixtral-8x7b.Q3_K_M.gguf
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Override directory - must match `_OVERRIDE_GGUF_DIR` in
# `backend/compute_pool.py`. We don't import compute_pool here so this
# script has zero backend-import surface (the file system is the entire
# contract between them).
OVERRIDE_DIR = Path.home() / ".gigachat" / "llama-cpp" / "models"


def _sanitize_model_name(name: str) -> str:
    """Strip characters that aren't legal in a Windows path. Mirrors
    `compute_pool._override_gguf_path_for` exactly so installer + reader
    agree on filenames."""
    return (name or "").strip().replace(":", "-").replace("/", "-")


def _format_bytes(n: int) -> str:
    """Pretty-print a byte count. GB for >1 GB, MB otherwise."""
    if n >= 1024 ** 3:
        return f"{n / 1024 ** 3:.2f} GB"
    if n >= 1024 ** 2:
        return f"{n / 1024 ** 2:.1f} MB"
    return f"{n} B"


def _download(url: str, dest: Path, *, resume: bool = True) -> None:
    """Stream `url` to `dest`, resuming from offset if `<dest>.part`
    already exists (HEAD-then-Range). Prints progress every ~2 seconds.

    `httpx` follows HuggingFace's 302 to its xet/cas-bridge CDN as long
    as `follow_redirects=True`. Connection-resume via Range works on
    the CDN's signed S3 URLs.
    """
    import httpx  # imported lazily so --dry-run doesn't require it

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    start_byte = 0
    if resume and tmp.is_file():
        start_byte = tmp.stat().st_size
        if start_byte > 0:
            print(f"  resuming from {_format_bytes(start_byte)}")

    headers = {}
    if start_byte > 0:
        headers["Range"] = f"bytes={start_byte}-"

    timeout = httpx.Timeout(connect=30.0, read=120.0, write=30.0, pool=30.0)
    mode = "ab" if start_byte > 0 else "wb"
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        with client.stream("GET", url, headers=headers) as r:
            if r.status_code not in (200, 206):
                raise RuntimeError(
                    f"download failed: HTTP {r.status_code} from {url}"
                )
            total = int(r.headers.get("content-length", 0))
            if start_byte > 0 and r.status_code == 206:
                total += start_byte  # 206 reports remaining-only
            print(f"  total size: {_format_bytes(total)}")

            written = start_byte
            last_print = time.time()
            t0 = last_print
            with tmp.open(mode) as f:
                for chunk in r.iter_bytes(chunk_size=1024 * 1024):
                    f.write(chunk)
                    written += len(chunk)
                    now = time.time()
                    if now - last_print >= 2.0:
                        elapsed = now - t0
                        rate = (written - start_byte) / elapsed if elapsed > 0 else 0
                        pct = (written / total * 100) if total > 0 else 0
                        eta_s = (total - written) / rate if rate > 0 else 0
                        print(
                            f"  {_format_bytes(written)} / "
                            f"{_format_bytes(total)} "
                            f"({pct:.1f}%) - "
                            f"{_format_bytes(int(rate))}/s, "
                            f"ETA {int(eta_s // 60)}m{int(eta_s % 60):02d}s",
                            flush=True,
                        )
                        last_print = now

    # Move .part -> final name only on full success. Crash-safe.
    tmp.rename(dest)
    print(f"  wrote {_format_bytes(dest.stat().st_size)} -> {dest}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--model-name", required=True,
        help="Ollama-style model name, e.g. 'gemma4:26b'. Determines the "
             "destination filename via the same sanitization rule as "
             "compute_pool._override_gguf_path_for.",
    )
    parser.add_argument(
        "--url", required=True,
        help="Direct-download URL for the GGUF on HuggingFace, "
             "e.g. https://huggingface.co/<repo>/resolve/main/<file>.gguf",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print URL, target path, and exit. No download.",
    )
    args = parser.parse_args()

    safe = _sanitize_model_name(args.model_name)
    if not safe or safe == ".gguf":
        print(f"ERROR: invalid model name {args.model_name!r}", file=sys.stderr)
        return 2
    dest = OVERRIDE_DIR / f"{safe}.gguf"

    print(f"Model name: {args.model_name}")
    print(f"URL:        {args.url}")
    print(f"Destination: {dest}")
    print()

    if args.dry_run:
        print("--dry-run: not downloading.")
        return 0

    if dest.is_file():
        print(f"NOTE: {dest} already exists ({_format_bytes(dest.stat().st_size)}).")
        print("Delete it manually first if you want to re-download.")
        print(f"Phase 2 split for {args.model_name} will use the existing file.")
        return 0

    print("Starting download. Resuming is supported - re-run if the connection drops.")
    print()
    try:
        _download(args.url, dest)
    except KeyboardInterrupt:
        print("\nInterrupted. Partial file kept at .part - re-run with the same flags to resume.")
        return 130
    except Exception as e:
        print(f"\nFAILED: {type(e).__name__}: {e}")
        return 1

    print()
    print(f"Done. Phase 2 split routing for `{args.model_name}` will now use this")
    print("GGUF instead of Ollama's blob.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
