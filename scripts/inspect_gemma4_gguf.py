"""Inspect Ollama's gemma4:26b GGUF — tensor list, shapes, quants, key metadata.

Read-only — does not modify anything. Output drives the surgery design
in `repack_gemma4_gguf.py` (TBD).
"""
from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

import gguf

BLOB_PATH = Path(r"C:\Users\gauta\.ollama\models\blobs\sha256-7121486771cbfe218851513210c40b35dbdee93ab1ef43fe36283c883980f0df")


def main() -> int:
    if not BLOB_PATH.is_file():
        print(f"NOT FOUND: {BLOB_PATH}")
        return 1
    reader = gguf.GGUFReader(str(BLOB_PATH))

    print("=" * 80)
    print("METADATA")
    print("=" * 80)
    for field in reader.fields.values():
        try:
            value = repr(field.parts[field.data[0]].tobytes() if field.types[0] == gguf.GGUFValueType.STRING else field.parts[field.data[0]])
        except Exception:
            value = "<complex>"
        # Only show the most useful keys at the top
        important = (
            "general.architecture", "general.name",
            "gemma4.block_count", "gemma4.embedding_length",
            "gemma4.feed_forward_length", "gemma4.attention.head_count",
            "gemma4.expert_count", "gemma4.expert_used_count",
            "gemma4.context_length",
            "general.quantization_version", "general.file_type",
        )
        if field.name in important:
            try:
                if field.types[0] == gguf.GGUFValueType.STRING:
                    val = field.parts[field.data[0]].tobytes().decode("utf-8", "replace")
                else:
                    val = list(field.parts[field.data[0]])
                    if len(val) == 1:
                        val = val[0]
                print(f"  {field.name:50} = {val}")
            except Exception as e:
                print(f"  {field.name:50} = <error: {e}>")

    print()
    print("=" * 80)
    print(f"TENSORS — total count: {len(reader.tensors)}")
    print("=" * 80)

    # Group by tensor name pattern.
    by_pattern = defaultdict(list)
    quant_counts = Counter()
    for t in reader.tensors:
        # Strip layer index for pattern grouping: "blk.5.ffn_gate.weight" -> "blk.N.ffn_gate.weight"
        parts = t.name.split(".")
        normalized = ".".join(["N" if p.isdigit() else p for p in parts])
        by_pattern[normalized].append((t.name, tuple(t.shape), str(t.tensor_type).split(".")[-1]))
        quant_counts[str(t.tensor_type).split(".")[-1]] += 1

    print(f"\nQuantization breakdown:")
    for qt, n in quant_counts.most_common():
        print(f"  {qt:15} = {n} tensors")

    print(f"\nTensor name patterns ({len(by_pattern)} unique):")
    for pattern in sorted(by_pattern.keys()):
        instances = by_pattern[pattern]
        # Show one example with its actual shape and type
        example_name, example_shape, example_type = instances[0]
        print(f"  {pattern:50} count={len(instances)}  shape={example_shape}  type={example_type}")

    # Look specifically for the fused MoE tensors that cause the "wrong number of tensors" error.
    print()
    print("=" * 80)
    print("MoE FUSED TENSORS (the ones that need splitting)")
    print("=" * 80)
    fused_patterns = [p for p in by_pattern if "ffn_gate_up" in p or "ffn_up_gate" in p]
    if not fused_patterns:
        print("  No fused ffn_gate_up_exps tensors found — checking for any 'expert' tensors:")
        expert_patterns = [p for p in by_pattern if "exp" in p or "moe" in p or "ffn" in p]
        for p in expert_patterns:
            print(f"  {p:50} (expert-related)")
    else:
        for pattern in fused_patterns:
            instances = by_pattern[pattern]
            example_name, example_shape, example_type = instances[0]
            print(f"  Pattern: {pattern}")
            print(f"  Count:   {len(instances)} (one per layer)")
            print(f"  Shape:   {example_shape}")
            print(f"  Type:    {example_type}")
            print(f"  All instances:")
            for name, shape, qt in instances[:5]:
                print(f"    {name:50}  shape={shape}  type={qt}")
            if len(instances) > 5:
                print(f"    ... ({len(instances) - 5} more)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
