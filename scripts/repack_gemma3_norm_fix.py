"""Patch the missing `gemma3.attention.layer_norm_rms_epsilon` key in
Ollama's Gemma 3 GGUFs.

Why this exists
---------------
Ollama's `gemma3:*` blobs ship without the text RMS-norm epsilon
metadata key. They DO ship the vision-tower epsilon
(`gemma3.vision.attention.layer_norm_epsilon = 9.999...e-07`),
but stock llama.cpp's gemma3 loader requires the text variant too:

    error loading model hyperparameters:
    key not found in model: gemma3.attention.layer_norm_rms_epsilon

This script reads Ollama's blob, copies every tensor verbatim
(no dequantization, no requantization — bit-for-bit lossless),
copies every metadata key verbatim, AND injects the missing key
with the canonical Gemma RMS-norm epsilon value (1e-6 — matches the
vision tower's value in the same file and is the default for the
Gemma family).

Output goes to the override directory so `compute_pool` picks it
up automatically. Source blob is NEVER modified.

Usage
-----
    python scripts/repack_gemma3_norm_fix.py \\
        --src ~/.ollama/models/blobs/sha256-... \\
        --dst ~/.gigachat/llama-cpp/models/gemma3-4b.gguf

Same shape as `repack_qwen3_rope_fix.py` — a pure metadata patch,
zero quantization change, zero quality drop.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import gguf

# Canonical Gemma RMS-norm epsilon. Matches:
#   * the vision-tower epsilon stored in the same GGUF
#     (`gemma3.vision.attention.layer_norm_epsilon`)
#   * the upstream Gemma reference implementation
#   * the value Unsloth's clean GGUFs ship with for the same model
INJECT_KEY = "gemma3.attention.layer_norm_rms_epsilon"
INJECT_VALUE = 1e-6

# GGUF-structural keys the writer manages itself; skip when copying.
WRITER_MANAGED_KEYS = {
    "GGUF.version",
    "GGUF.tensor_count",
    "GGUF.kv_count",
    "general.architecture",
}


def _field_value(field):
    """Extract the Python value from a ReaderField, returning
    (value, value_type)."""
    types = list(field.types)
    if not types:
        return None, None
    head = types[0]
    if head == gguf.GGUFValueType.STRING:
        return field.parts[field.data[0]].tobytes().decode("utf-8", "replace"), head
    if head == gguf.GGUFValueType.ARRAY:
        elem_type = types[1] if len(types) > 1 else None
        if elem_type == gguf.GGUFValueType.STRING:
            values = [field.parts[i].tobytes().decode("utf-8", "replace") for i in field.data]
        else:
            values = [field.parts[i].tolist()[0] for i in field.data]
        return values, elem_type
    return field.parts[field.data[0]].tolist()[0], head


def _add_field(writer, name, value, value_type):
    """Re-emit a metadata field on the writer with its original type."""
    if isinstance(value, list):
        writer.add_array(name, value)
        return
    vt = gguf.GGUFValueType
    if value_type == vt.STRING:
        writer.add_string(name, value)
    elif value_type == vt.BOOL:
        writer.add_bool(name, bool(value))
    elif value_type == vt.UINT8:
        writer.add_uint8(name, int(value))
    elif value_type == vt.INT8:
        writer.add_int8(name, int(value))
    elif value_type == vt.UINT16:
        writer.add_uint16(name, int(value))
    elif value_type == vt.INT16:
        writer.add_int16(name, int(value))
    elif value_type == vt.UINT32:
        writer.add_uint32(name, int(value))
    elif value_type == vt.INT32:
        writer.add_int32(name, int(value))
    elif value_type == vt.UINT64:
        writer.add_uint64(name, int(value))
    elif value_type == vt.INT64:
        writer.add_int64(name, int(value))
    elif value_type == vt.FLOAT32:
        writer.add_float32(name, float(value))
    elif value_type == vt.FLOAT64:
        writer.add_float64(name, float(value))
    else:
        raise RuntimeError(f"unsupported field type {value_type} for key {name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--src", required=True, help="Source Ollama blob path.")
    parser.add_argument("--dst", required=True, help="Destination override path.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plan; don't write.")
    args = parser.parse_args()

    src = Path(args.src).expanduser()
    dst = Path(args.dst).expanduser()
    if not src.is_file():
        print(f"ERROR: source not found: {src}", file=sys.stderr)
        return 2

    print(f"src: {src}  ({src.stat().st_size / 1024 ** 3:.2f} GB)")
    print(f"dst: {dst}")
    print()

    reader = gguf.GGUFReader(str(src))

    arch_field = reader.fields.get("general.architecture")
    actual_arch, _ = _field_value(arch_field) if arch_field else (None, None)
    if actual_arch != "gemma3":
        print(f"ERROR: expected gemma3 arch, got {actual_arch!r}", file=sys.stderr)
        return 1

    if INJECT_KEY in reader.fields:
        print(f"NOTE: {INJECT_KEY} already present — passthrough copy.")
        already_present = True
    else:
        already_present = False
        print(f"  injecting {INJECT_KEY} = {INJECT_VALUE}")

    print(f"Source tensors:    {len(reader.tensors)}")
    print(f"Source metadata:   {len(reader.fields)}")
    print()

    if args.dry_run:
        print("--dry-run: not writing.")
        return 0

    if dst.is_file():
        print(f"ERROR: dst already exists: {dst}", file=sys.stderr)
        return 2

    print("Building output GGUF...")
    dst.parent.mkdir(parents=True, exist_ok=True)
    writer = gguf.GGUFWriter(str(dst), arch="gemma3")

    # Copy every metadata field verbatim (writer manages structural keys).
    for name, field in reader.fields.items():
        if name in WRITER_MANAGED_KEYS:
            continue
        value, vtype = _field_value(field)
        if value is None:
            continue
        try:
            _add_field(writer, name, value, vtype)
        except Exception as e:
            print(f"  WARNING: skipping {name} ({type(e).__name__}: {e})")

    # Inject the missing key (after the loop so it's never duplicated).
    if not already_present:
        writer.add_float32(INJECT_KEY, float(INJECT_VALUE))

    print(f"Copying {len(reader.tensors)} tensors lossless...")
    for i, t in enumerate(reader.tensors):
        data = np.asarray(t.data)
        if data.dtype == np.uint8:
            writer.add_tensor(t.name, data, raw_dtype=t.tensor_type)
        else:
            writer.add_tensor(t.name, data)
        if (i + 1) % 100 == 0 or i + 1 == len(reader.tensors):
            print(f"  {i + 1} / {len(reader.tensors)}")

    print("Writing...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"\nDone — wrote {dst.stat().st_size / 1024 ** 3:.2f} GB to {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
