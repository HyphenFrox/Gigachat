"""Patch the rope.dimension_sections metadata in a Qwen 3.5 GGUF.

Why this exists
---------------
Ollama's Qwen 3.5 9B GGUF stores `qwen35.rope.dimension_sections` as
a 3-element uint32 array `[11, 11, 10]`. Stock llama.cpp's qwen35
loader expects a 4-element array (the multimodal Qwen3-VL rope
convention with [time, height, width, vision] sections, where
text-only models use 0 for the vision section). Loading Ollama's
blob fails with:

    error loading model hyperparameters:
    key qwen35.rope.dimension_sections has wrong array length;
    expected 4, got 3

This script reads Ollama's blob, copies every tensor verbatim
(no dequantization, no requantization — bit-for-bit lossless),
copies every metadata key verbatim EXCEPT the three rope-section
arrays which we extend from length 3 to length 4 by appending 0.
Sum stays at 32 (half of dimension_count=64), so the rope behavior
is unchanged for text inputs — the appended 0 is the no-vision
sentinel the multimodal convention uses.

Output goes to the override directory so `compute_pool` picks it
up automatically. Source blob is NEVER modified.

Usage
-----
    python scripts/repack_qwen3_rope_fix.py \\
        --src ~/.ollama/models/blobs/sha256-... \\
        --dst ~/.gigachat/llama-cpp/models/qwen3.5-9b.gguf

Same shape as `repack_text_only_gguf.py` but a different transformation.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import gguf

# Metadata keys we extend from length 3 -> 4. Sum is preserved
# (the new last element is 0 = no-vision sentinel for the
# multimodal mrope convention).
ROPE_KEYS_TO_PATCH = (
    "qwen35.rope.dimension_sections",
    "qwen35.mrope_sections",
    "qwen35.rope.mrope_section",
)

# GGUF-structural keys the writer manages itself; skip when copying
# field-by-field.
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
    if dst.is_file():
        print(f"ERROR: dst already exists: {dst}", file=sys.stderr)
        return 2

    print(f"src: {src}  ({src.stat().st_size / 1024 ** 3:.2f} GB)")
    print(f"dst: {dst}")
    print()

    reader = gguf.GGUFReader(str(src))

    # Verify it's a qwen35 architecture (script is qwen-specific).
    arch_field = reader.fields.get("general.architecture")
    actual_arch, _ = _field_value(arch_field) if arch_field else (None, None)
    if actual_arch != "qwen35":
        print(f"ERROR: expected qwen35 arch, got {actual_arch!r}", file=sys.stderr)
        return 1

    print(f"Source tensors:    {len(reader.tensors)}")
    print(f"Source metadata:   {len(reader.fields)}")
    print()

    # Identify the rope-section keys we'll patch.
    patches = {}
    for key in ROPE_KEYS_TO_PATCH:
        f = reader.fields.get(key)
        if not f:
            continue
        value, vtype = _field_value(f)
        if isinstance(value, list) and len(value) == 3:
            new_value = list(value) + [0]
            patches[key] = (new_value, vtype)
            print(f"  patching {key}: {value} -> {new_value}")
        elif isinstance(value, list):
            print(f"  skipping {key}: already length {len(value)} (expected 3)")
    if not patches:
        print("ERROR: no rope-section keys at length 3 — nothing to patch.")
        return 1

    if args.dry_run:
        print("\n--dry-run: not writing.")
        return 0

    print()
    print("Building output GGUF...")
    dst.parent.mkdir(parents=True, exist_ok=True)
    writer = gguf.GGUFWriter(str(dst), arch="qwen35")

    # Copy metadata, applying the patches where applicable.
    for name, field in reader.fields.items():
        if name in WRITER_MANAGED_KEYS:
            continue
        if name in patches:
            new_value, vtype = patches[name]
            _add_field(writer, name, new_value, vtype)
            continue
        value, vtype = _field_value(field)
        if value is None:
            continue
        try:
            _add_field(writer, name, value, vtype)
        except Exception as e:
            print(f"  WARNING: skipping {name} ({type(e).__name__}: {e})")

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
