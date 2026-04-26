"""Extract just the text LLM from a multimodal GGUF (one-time, this host).

Why this exists
---------------
Ollama bundles some multimodal models (e.g. `gemma4:26b`) as a single
GGUF containing both the text LLM tensors AND a vision tower. Stock
`llama-server` (the binary our Phase 2 split path uses) doesn't load
the bundled multimodal layout — it expects the vision tower as a
SEPARATE `mmproj` GGUF passed via `--mmproj`. With the bundle, it
errors at load time with "wrong number of tensors; expected N, got
M".

This script extracts ONLY the text LLM tensors (everything not
prefixed with `v.` or `mm.`) into a new GGUF that `llama-server`
loads cleanly. It pairs with an mmproj GGUF (from Unsloth's repo or
a separate extraction) to enable full multimodal inference via
`llama-server --model <text>.gguf --mmproj <vision>.gguf --rpc ...`.

Lossless: tensor data is copied byte-for-byte with no
dequantization or requantization. Metadata is copied verbatim
EXCEPT for vision-specific keys (`*.vision.*`) which are dropped
since they describe a tower no longer in this file.

This script is intended for ONE-TIME use on the developer's machine.
End users should never need to run it — the app handles compatible-
GGUF acquisition automatically (see `compute_pool.ensure_compatible_gguf`).

Usage
-----
    python scripts/repack_text_only_gguf.py \\
        --src C:/Users/.../sha256-7121486771cbfe...     \\
        --dst ~/.gigachat/llama-cpp/models/gemma4-26b.gguf \\
        --arch gemma4

The architecture flag must match the source's `general.architecture`
metadata (gemma4, gemma3, llama, etc.) — the script verifies this
before writing.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import gguf

# Tensors with these prefixes are vision/multimodal and get filtered out.
VISION_TENSOR_PREFIXES = ("v.", "mm.")

# Metadata keys containing any of these substrings are dropped because
# they describe the vision tower we're removing. We DON'T drop generic
# tokenizer / general / arch keys — only true vision-specific ones.
VISION_METADATA_SUBSTRINGS = (".vision.",)

# Metadata keys the writer manages itself (we'd duplicate them via
# explicit calls). Skip so we don't fight the writer.
WRITER_MANAGED_KEYS = {
    "GGUF.version",
    "GGUF.tensor_count",
    "GGUF.kv_count",
    "general.architecture",
}


def _field_value(field: gguf.gguf_reader.ReaderField):
    """Extract the Python value from a ReaderField. Handles all common
    GGUF value types we encounter in real-world models. Returns
    (value, value_type) where value_type is the GGUFValueType used by
    the field (or for ARRAY fields, the element type)."""
    types = list(field.types)
    if not types:
        return None, None
    head = types[0]

    if head == gguf.GGUFValueType.STRING:
        # parts[data[0]] is the bytes of the string
        return field.parts[field.data[0]].tobytes().decode("utf-8", "replace"), head

    if head == gguf.GGUFValueType.ARRAY:
        # types[1] is the element type. data is a list of indices into
        # parts, one per element.
        elem_type = types[1] if len(types) > 1 else None
        if elem_type == gguf.GGUFValueType.STRING:
            values = [field.parts[i].tobytes().decode("utf-8", "replace") for i in field.data]
        else:
            values = []
            for i in field.data:
                # numeric arrays — parts[i] is a memoryview of one scalar
                values.append(field.parts[i].tolist()[0])
        return values, elem_type

    # Scalar numeric / bool
    return field.parts[field.data[0]].tolist()[0], head


def _add_field_to_writer(writer: gguf.GGUFWriter, name: str, value, value_type):
    """Add an arbitrary field to the writer using its low-level
    add_key_value API. Routes through the strongly-typed helpers when
    the type is known."""
    if isinstance(value, list):
        # Array field
        if value_type == gguf.GGUFValueType.STRING:
            writer.add_array(name, value)
            return
        # Numeric array — pick the right add_array variant
        writer.add_array(name, value)
        return

    if value_type == gguf.GGUFValueType.STRING:
        writer.add_string(name, value)
    elif value_type == gguf.GGUFValueType.BOOL:
        writer.add_bool(name, bool(value))
    elif value_type == gguf.GGUFValueType.UINT8:
        writer.add_uint8(name, int(value))
    elif value_type == gguf.GGUFValueType.INT8:
        writer.add_int8(name, int(value))
    elif value_type == gguf.GGUFValueType.UINT16:
        writer.add_uint16(name, int(value))
    elif value_type == gguf.GGUFValueType.INT16:
        writer.add_int16(name, int(value))
    elif value_type == gguf.GGUFValueType.UINT32:
        writer.add_uint32(name, int(value))
    elif value_type == gguf.GGUFValueType.INT32:
        writer.add_int32(name, int(value))
    elif value_type == gguf.GGUFValueType.UINT64:
        writer.add_uint64(name, int(value))
    elif value_type == gguf.GGUFValueType.INT64:
        writer.add_int64(name, int(value))
    elif value_type == gguf.GGUFValueType.FLOAT32:
        writer.add_float32(name, float(value))
    elif value_type == gguf.GGUFValueType.FLOAT64:
        writer.add_float64(name, float(value))
    else:
        raise RuntimeError(f"unsupported field type {value_type} for key {name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--src", required=True, help="Path to the source bundled GGUF.")
    parser.add_argument("--dst", required=True, help="Output path for the text-only GGUF.")
    parser.add_argument(
        "--arch", required=True,
        help="Expected general.architecture in source (e.g. gemma4). "
             "The script verifies this matches before writing.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Inspect only; don't write.")
    args = parser.parse_args()

    src = Path(args.src).expanduser()
    dst = Path(args.dst).expanduser()
    if not src.is_file():
        print(f"ERROR: source not found: {src}", file=sys.stderr)
        return 2
    if dst.is_file():
        print(f"ERROR: destination already exists: {dst}", file=sys.stderr)
        print("Delete it first if you want to re-extract.", file=sys.stderr)
        return 2

    print(f"src: {src}  ({src.stat().st_size / 1024 ** 3:.2f} GB)")
    print(f"dst: {dst}")
    print(f"arch: {args.arch}")
    print()

    reader = gguf.GGUFReader(str(src))

    # Verify architecture matches.
    arch_field = reader.fields.get("general.architecture")
    if arch_field is None:
        print("ERROR: source has no general.architecture metadata", file=sys.stderr)
        return 1
    actual_arch, _ = _field_value(arch_field)
    if actual_arch != args.arch:
        print(f"ERROR: arch mismatch — expected {args.arch!r}, got {actual_arch!r}", file=sys.stderr)
        return 1

    # Partition tensors.
    keep_tensors = [t for t in reader.tensors
                    if not any(t.name.startswith(p) for p in VISION_TENSOR_PREFIXES)]
    drop_tensors = [t for t in reader.tensors
                    if any(t.name.startswith(p) for p in VISION_TENSOR_PREFIXES)]

    print(f"Source tensor count:    {len(reader.tensors)}")
    print(f"  keep (text LLM):      {len(keep_tensors)}")
    print(f"  drop (vision/mm):     {len(drop_tensors)}")
    print()

    # Partition metadata.
    keep_fields = []
    drop_fields = []
    for name, field in reader.fields.items():
        if name in WRITER_MANAGED_KEYS:
            continue
        if any(s in name for s in VISION_METADATA_SUBSTRINGS):
            drop_fields.append(name)
            continue
        keep_fields.append((name, field))

    print(f"Source metadata count:  {len(reader.fields)}")
    print(f"  keep:                 {len(keep_fields)}")
    print(f"  drop (vision/mm):     {len(drop_fields)}")
    print(f"  managed by writer:    {len(reader.fields) - len(keep_fields) - len(drop_fields)}")
    if drop_fields:
        print(f"  dropped keys:")
        for k in drop_fields:
            print(f"    {k}")

    if args.dry_run:
        print("\n--dry-run: not writing.")
        return 0

    # Estimate output size = sum of kept tensor data sizes (rough, ignores metadata + alignment).
    est_size = sum(t.n_bytes for t in keep_tensors)
    print(f"\nEstimated output size: {est_size / 1024 ** 3:.2f} GB")
    print()

    # Build the writer. The arch parameter sets general.architecture
    # automatically. We DON'T pass tensor counts upfront — the writer
    # tallies them as we add tensors.
    dst.parent.mkdir(parents=True, exist_ok=True)
    writer = gguf.GGUFWriter(str(dst), arch=args.arch)

    print(f"Copying {len(keep_fields)} metadata fields...")
    n_field_skipped = 0
    for name, field in keep_fields:
        value, vtype = _field_value(field)
        if value is None:
            n_field_skipped += 1
            continue
        try:
            _add_field_to_writer(writer, name, value, vtype)
        except Exception as e:
            print(f"  WARNING: skipping {name} ({type(e).__name__}: {e})")
            n_field_skipped += 1
    print(f"  done ({n_field_skipped} skipped)")

    print(f"\nCopying {len(keep_tensors)} tensors...")
    for i, t in enumerate(keep_tensors):
        # Add the tensor with its raw bytes preserved — no dequant, no
        # requant. The reader exposes `t.data` as a uint8 numpy array
        # whose shape is the BYTE layout (e.g. for a Q4_K matrix
        # stored as logical (2816, 2048) elements, t.data.shape is
        # (2048, 1584) since each row of 2816 elements occupies
        # 2816/256 * 144 = 1584 bytes). We pass that byte-shape array
        # straight through plus `raw_dtype` so the writer's
        # `quant_shape_from_byte_shape` converts it back to the
        # correct logical shape, and the GGUF spec's dim-reversal
        # round-trips it to the original file layout.
        #
        # For F32 / F16 / BF16 tensors (not quantized), `t.data` has
        # the matching numpy dtype already, so we don't pass
        # `raw_dtype` — the writer infers everything correctly from
        # the array's dtype + shape.
        data = np.asarray(t.data)
        if data.dtype == np.uint8:
            writer.add_tensor(t.name, data, raw_dtype=t.tensor_type)
        else:
            writer.add_tensor(t.name, data)
        if (i + 1) % 50 == 0 or i + 1 == len(keep_tensors):
            print(f"  {i + 1} / {len(keep_tensors)}")

    print(f"\nWriting GGUF...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    out_size = dst.stat().st_size
    print(f"\nDone — wrote {out_size / 1024 ** 3:.2f} GB to {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
