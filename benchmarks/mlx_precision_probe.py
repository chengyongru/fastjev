"""Investigate BF16 drift using identical prompts and a Torch CPU FP32 reference.

Select all quality rows above the frozen review threshold, plus changed choices.
Run only after GPU timing benchmarks: CPU reference work shares Mac memory.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mlx_benchmark import choice, compare, read, write
from fastjev.runtime import mlx as backend
from fastjev.runtime.direct import encode_prompt, score as torch_score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mlx-cache-limit-mib", type=int, default=backend.DEFAULT_CACHE_LIMIT_MIB,
                        help="Inactive allocation cache in MiB (default: 256; 0 disables caching)")
    args = parser.parse_args()
    if args.mlx_cache_limit_mib < 0:
        parser.error("--mlx-cache-limit-mib must be nonnegative")
    if args.output.exists():
        parser.error("Output must be new")
    contract = json.loads(Path("manifests/mlx-validation.json").read_text())
    threshold = contract["bf16_probability_difference_review_threshold"]
    manifest = json.loads((args.run / "manifest.json").read_text())
    source, revision = manifest["model"]["source"], manifest["model"]["revision"]
    selected = []
    for dataset in ("authored144", "perturbations108"):
        gold = {row["id"]: row for row in read(f"benchmarks/data/{dataset}.jsonl")}
        old = {row["id"]: row for row in read(f"results/raw/predictions/direct-{dataset}.jsonl")}
        for row in read(args.run / f"{dataset}.jsonl"):
            reference = old[row["id"]]
            delta = max(abs(a - b) for a, b in zip(reference["probabilities"], row["probabilities"]))
            if delta > threshold or choice(row) != choice(reference):
                selected.append((gold[row["id"]], reference, row))
    print(f"Investigating {len(selected)} decisions", flush=True)
    model, tokenizer, metadata = backend.load_model(source, revision, cache_limit_mib=args.mlx_cache_limit_mib)
    bf16 = [backend.score(model, tokenizer, row, metadata) for row, _, _ in selected]
    serial = backend.SerialPrefixScorer(model, tokenizer, metadata)
    bf16_serial = [serial.score(row) for row, _, _ in selected]
    del serial
    import mlx.core as mx

    model.set_dtype(mx.float32)
    mx.eval(model.parameters())
    metadata = {**metadata, "dtype": ["mlx.core.float32"], "diagnostic_cast": "all floating weights to FP32"}
    mlx_fp32 = [backend.score(model, tokenizer, row, metadata) for row, _, _ in selected]
    serial = backend.SerialPrefixScorer(model, tokenizer, metadata)
    mlx_fp32_serial = [serial.score(row) for row, _, _ in selected]
    del serial

    # MLX folds Qwen's (1 + RMSNorm weight) into a conventional RMSNorm weight.
    # Folding in BF16 rounds that addition before a later diagnostic FP32 cast.
    # Use the same upstream sanitizer on FP32 source arrays to isolate this
    # weight-conversion rounding from execution rounding. This is diagnostic
    # only; the production loader continues to use the native MLX-LM path.
    from huggingface_hub import snapshot_download
    from mlx.utils import tree_flatten

    source_path = Path(source) if Path(source).is_dir() else Path(snapshot_download(
        source, revision=revision, local_files_only=True,
        allow_patterns=["*.json", "model*.safetensors", "*.jinja", "*.txt", "*.model"],
    ))
    raw = {}
    for path in source_path.glob("model*.safetensors"):
        raw.update(mx.load(path))
    sanitized = model.sanitize({key: value.astype(mx.float32) for key, value in raw.items()})
    suffixes = (".input_layernorm.weight", ".post_attention_layernorm.weight",
                "model.norm.weight", ".q_norm.weight", ".k_norm.weight")
    corrected = {key: value for key, value in sanitized.items() if key.endswith(suffixes)}
    del raw, sanitized
    current = dict(tree_flatten(model.parameters()))
    norm_differences = {key: float(mx.max(mx.abs(value - current[key])).item()) for key, value in corrected.items()}
    del current
    model.load_weights(list(corrected.items()), strict=False)
    mx.eval(model.parameters())
    corrected_metadata = {**metadata, "diagnostic_norm_folding": "FP32 source weights before upstream sanitizer"}
    mlx_fp32_source_norms = [backend.score(model, tokenizer, row, corrected_metadata) for row, _, _ in selected]
    long_rows = read("benchmarks/data/shape777.jsonl")[:4]
    long_fresh = [backend.score(model, tokenizer, row, corrected_metadata) for row in long_rows]
    serial = backend.SerialPrefixScorer(model, tokenizer, corrected_metadata)
    long_serial = [serial.score(row) for row in long_rows]
    long_shared, _ = backend.score_shared(model, tokenizer, long_rows, corrected_metadata)
    del serial
    del corrected, model
    mx.synchronize()
    mx.clear_cache()

    import torch
    import transformers

    torch.set_num_threads(8)
    config = transformers.AutoConfig.from_pretrained(source, revision=revision, trust_remote_code=False)
    torch_model, loading = transformers.Qwen3_5ForCausalLM.from_pretrained(
        source, revision=revision, config=config.get_text_config(), dtype=torch.float32,
        device_map={"": "cpu"}, low_cpu_mem_usage=True, output_loading_info=True, trust_remote_code=False,
    )
    if any(loading.get(key) for key in ("missing_keys", "mismatched_keys", "error_msgs")):
        raise RuntimeError(f"Incomplete CPU reference checkpoint: {loading}")
    torch_model.eval()
    torch_tokenizer = transformers.AutoTokenizer.from_pretrained(source, revision=revision, trust_remote_code=False)
    reference_metadata = {"source": source, "revision": revision, "dtype": "float32", "device": "cpu",
                          "torch_version": torch.__version__, "transformers_version": transformers.__version__}
    cpu = []
    for row, _, _ in selected:
        if encode_prompt(tokenizer, row, 4096) != encode_prompt(torch_tokenizer, row, 4096):
            raise AssertionError("Tokenization differs")
        cpu.append(torch_score(torch_model, torch_tokenizer, row, reference_metadata))
        print(f"CPU reference: {len(cpu)}/{len(selected)}", flush=True)
    published = [row for _, row, _ in selected]
    report = {"selection_run": str(args.run), "selection_threshold": threshold, "rows": len(selected),
              "comparison": {
                  "bf16_mlx_vs_published": compare(published, bf16),
                  "bf16_mlx_serial_vs_published": compare(published, bf16_serial),
                  "fp32_mlx_vs_cpu": compare(cpu, mlx_fp32),
                  "fp32_source_norms_mlx_vs_cpu": compare(cpu, mlx_fp32_source_norms),
                  "fp32_mlx_serial_vs_fresh": compare(mlx_fp32, mlx_fp32_serial),
                  "bf16_mlx_vs_fp32_mlx": compare(mlx_fp32, bf16),
              },
              "predictions": {"published_cuda_bf16": published, "mlx_bf16_direct": bf16,
                              "mlx_bf16_serial": bf16_serial, "mlx_fp32_direct": mlx_fp32,
                              "mlx_fp32_serial": mlx_fp32_serial, "torch_cpu_fp32": cpu},
              "source_norm_folding_max_weight_difference": norm_differences,
              "fp32_long_prompt_cache": {
                  "fresh": long_fresh, "serial": long_serial, "shared": long_shared,
                  "serial_vs_fresh": compare(long_fresh, long_serial),
                  "shared_vs_fresh": compare(long_fresh, long_shared),
              },
              "limitations": "Diagnostic selection from observed discrepancies, not a held-out quality benchmark. CPU timing is not compared to GPU timing."}
    report["predictions"]["mlx_fp32_source_norms"] = mlx_fp32_source_norms
    write(args.output, report)
    print(json.dumps({key: {"flips": len(value["argmax_flips"]), "max_difference": value["max_probability_difference"]}
                      for key, value in report["comparison"].items()}), flush=True)


if __name__ == "__main__":
    main()
