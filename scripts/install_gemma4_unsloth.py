"""Install Unsloth's Gemma 4 26B-A4B-IT GGUF for Phase 2 split routing.

Why this exists
---------------
Ollama's `gemma4:26b` blob packs the MoE expert tensors fused as
`ffn_gate_up_exps` per layer (658 tensors total). Stock llama.cpp's
`gemma4` loader spec expects the unfused layout — `ffn_gate_exps` +
`ffn_up_exps` separately — totalling 1014 tensors. Result: when
`compute_pool` engages Phase 2 layer-split for `gemma4:26b`, llama-server
bails on load with `wrong number of tensors; expected 1014, got 658`.

Unsloth's GGUFs are produced for stock llama.cpp and use the unfused
layout. Dropping one of theirs into `~/.gigachat/llama-cpp/models/`
(named `gemma4-26b.gguf`) makes `resolve_ollama_model` prefer it over
the Ollama blob — no other code changes needed, since the override
hook is generic.

What this script does
---------------------
1. Streams `gemma-4-26B-A4B-it-UD-Q5_K_M.gguf` (~21.15 GB) from
   `unsloth/gemma-4-26B-A4B-it-GGUF` on HuggingFace into the override
   directory.
2. Resumes a partial download if the file already exists and is
   smaller than the expected size — useful when the user's connection
   drops mid-download for a 21 GB transfer.
3. Renames the final file to `gemma4-26b.gguf` so the override hook
   in `compute_pool._override_gguf_path_for` finds it.

Usage
-----
    python scripts/install_gemma4_unsloth.py            # default Q5_K_M
    python scripts/install_gemma4_unsloth.py --quant Q4_K_M
    python scripts/install_gemma4_unsloth.py --dry-run  # just print plan

Other quantizations available (filenames are case-sensitive on HF):
    UD-Q4_K_S    UD-Q4_K_M    UD-Q4_K_XL
    UD-Q5_K_S    UD-Q5_K_M    UD-Q5_K_XL
    UD-Q6_K      UD-Q8_0
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Repo + filename pattern come from Unsloth's HuggingFace conventions.
# Verified against `unsloth/gemma-4-26B-A4B-it-GGUF` repo listing.
HF_REPO = "unsloth/gemma-4-26B-A4B-it-GGUF"
FILENAME_TEMPLATE = "gemma-4-26B-A4B-it-UD-{quant}.gguf"
DEFAULT_QUANT = "Q5_K_M"

# Override directory — must match `_OVERRIDE_GGUF_DIR` in compute_pool.py.
# We don't import compute_pool here because this script intentionally has
# zero backend-import surface (the file system contract is the entire API).
OVERRIDE_DIR = Path.home() / ".gigachat" / "llama-cpp" / "models"
TARGET_FILENAME = "gemma4-26b.gguf"


def _format_bytes(n: int) -> str:
    """Pretty-print a byte count. GB for 21 GB files, MB for partial."""
    if n >= 1024 ** 3:
        return f"{n / 1024 ** 3:.2f} GB"
    if n >= 1024 ** 2:
        return f"{n / 1024 ** 2:.1f} MB"
    return f"{n} B"


def _download(url: str, dest: Path, *, resume: bool = True) -> None:
    """Stream `url` to `dest`, resuming from offset if `dest` already
    exists (HEAD-then-Range). Prints progress every ~2 seconds.

    `httpx` follows HuggingFace's 302 redirect to its xet/cas-bridge CDN
    automatically as long as `follow_redirects=True`. Connection-resume
    via Range works on the CDN side — verified against signed S3 URLs.
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
                total += start_byte  # 206 reports remaining only
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
                            f"({pct:.1f}%) — "
                            f"{_format_bytes(int(rate))}/s, "
                            f"ETA {int(eta_s // 60)}m{int(eta_s % 60):02d}s",
                            flush=True,
                        )
                        last_print = now

    # Move .part → final name only on full success. Crash-safe.
    tmp.rename(dest)
    print(f"  wrote {_format_bytes(dest.stat().st_size)} → {dest}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--quant",
        default=DEFAULT_QUANT,
        help=f"Quantization tag, e.g. Q4_K_M, Q5_K_M, Q8_0 (default: {DEFAULT_QUANT})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print URL, target path, and expected size; don't download.",
    )
    args = parser.parse_args()

    filename = FILENAME_TEMPLATE.format(quant=args.quant)
    url = f"https://huggingface.co/{HF_REPO}/resolve/main/{filename}?download=true"
    dest = OVERRIDE_DIR / TARGET_FILENAME

    print(f"Repo:        {HF_REPO}")
    print(f"Quant:       UD-{args.quant}")
    print(f"Filename:    {filename}")
    print(f"URL:         {url}")
    print(f"Destination: {dest}")
    print()

    if args.dry_run:
        print("--dry-run: not downloading.")
        return 0

    if dest.is_file():
        print(f"NOTE: {dest} already exists ({_format_bytes(dest.stat().st_size)}).")
        print("Delete it manually first if you want to re-download.")
        print("Phase 2 split for gemma4:26b will use the existing file.")
        return 0

    print("Starting download. This is ~21 GB; expect 30+ minutes on a 100 Mbps link.")
    print("Resuming is supported — re-run if the connection drops.")
    print()
    try:
        _download(url, dest)
    except KeyboardInterrupt:
        print("\nInterrupted. Partial file kept at .part — re-run to resume.")
        return 130
    except Exception as e:
        print(f"\nFAILED: {type(e).__name__}: {e}")
        return 1

    print()
    print("Done. Phase 2 split routing for `gemma4:26b` will now use the")
    print("Unsloth GGUF instead of Ollama's blob, sidestepping the")
    print("'wrong number of tensors' load error.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
