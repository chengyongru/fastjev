# 复现已报告结果

[English](README.md) | 简体中文

仓库包含精确的项目自有速度 fixture、基准 runner、逐行模型输出和源选择 ID。模型权重以及未经授权再分发的第三方记录仍保留在上游。

## 紧凑生成比较

```bash
CUDA_VISIBLE_DEVICES=0 python benchmarks/decision_vs_generation.py \
  --model Qwen/Qwen3.5-4B \
  --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --input benchmarks/data/shape777.jsonl \
  --output compact-array-run.json
```

该命令在第一个 21 行共享状态分组上，对每条路径各进行三次 warm 测量。生成基线只请求一个有序的 `"yes"`/`"no"` JSON 数组。提交的运行记录包含精确 prompt message 和 token 时间线，见 [decision-vs-compact-array.json](../results/raw/decision-vs-compact-array.json)。

## 稳定性扰动

提交的 108 行稳定性 fixture 由 36 个项目自有原始 case 确定性生成。按以下命令重建 fixture 和 manifest：

```bash
python benchmarks/build_perturbations.py \
  --source benchmarks/data/authored144.jsonl \
  --output perturbations108.jsonl \
  --manifest perturbations108-manifest.json
```

从已提交逐行预测重建 `results/raw/perturbation-comparison.json` 的完整命令见 `docs/REPRODUCE.zh-CN.md`。重新生成的报告与已提交报告逐字节一致。

## 完整 37×21 系统基准

```bash
CUDA_VISIBLE_DEVICES=0 python benchmarks/shape777.py \
  --model Qwen/Qwen3.5-4B \
  --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --input benchmarks/data/shape777.jsonl \
  --output shape777-run.json
```

该基准覆盖全新评分、串行前缀缓存复用和并行共享状态评分。原生 reranker 测量单独复现：

```bash
CUDA_VISIBLE_DEVICES=0 python benchmarks/shape777_reranker.py \
  --model Qwen/Qwen3-Reranker-4B \
  --revision 22e683669bc0f0bd69640a1354a6d0aebcfeede5 \
  --input benchmarks/data/shape777.jsonl \
  --pair-batch-sizes 1,4,8 \
  --output shape777-reranker-run.json
```

这个 6.7 MB fixture 由项目编写，SHA-256 为 `8dcf414b12fc2684e3c4ca5f3ebfd3f525f5346fec4a9bc67eb65138101f55f1`。两个 runner 都会写入汇总计时和逐行预测。

## 质量证据

- `data/authored144.jsonl` 是完整的项目自有标注工作负载。
- `manifests/evaluation-matrix.jsonl` 冻结了全部 706 个被评估的行 ID、任务族和分母。
- `manifests/source-selection.jsonl` 将 WANLI 行映射到固定 test-set ID，将 TypeSafe 行映射到 case/question ID 和源 hash，并将 Every 行映射到实验 item。
- `../results/raw/predictions/` 包含直接与 reranker 的逐行输出。
- `../results/raw/quality-comparison.json` 包含 README 表格背后的完整汇总报告。
- `evaluate.py` 重新计算硬标签准确率、平衡准确率、F1、概率指标和按源分组的配对 bootstrap 区间。

获取允许再分发的外部快照：

```bash
python benchmarks/fetch_sources.py --output /path/on/large-drive/semif-sources
```

仓库不包含 TypeSafe 源快照。如果你持有这些快照，请按 `build_typesafe.py` 预期的文件名，将本地副本放入同一源目录。

从已验证快照确定性重建评估行：

```bash
SRC=/path/on/large-drive/semif-sources
OUT=/path/on/large-drive/semif-built
mkdir -p "$OUT"

python benchmarks/build_wanli.py \
  --source "$SRC/wanli-test.jsonl" \
  --selection benchmarks/manifests/source-selection.jsonl \
  --output "$OUT/wanli256.jsonl"

python benchmarks/build_every.py \
  --archive "$SRC/every-source.zip" \
  --experiments "$SRC/every-experiments.json" \
  --selection benchmarks/manifests/source-selection.jsonl \
  --output-dir "$OUT/every"

python benchmarks/build_typesafe.py \
  --source-dir "$SRC" \
  --selection benchmarks/manifests/source-selection.jsonl \
  --output "$OUT/typesafe102.jsonl"
```

使用重建标签和已提交预测，重新计算公开对齐指标：

```bash
python benchmarks/evaluate_external.py --source typesafe \
  --gold "$OUT/typesafe102.jsonl" \
  --direct results/raw/predictions/direct-typesafe102.jsonl \
  --reranker results/raw/predictions/reranker-typesafe102.jsonl

python benchmarks/evaluate_external.py --source every \
  --gold "$OUT/every/gold154.jsonl" \
  --inference "$OUT/every/inference204.jsonl" \
  --firewall-actions "$OUT/every/firewall-actions.json" \
  --direct results/raw/predictions/direct-every204.jsonl \
  --reranker results/raw/predictions/reranker-every204.jsonl
```

TypeSafe evaluator 报告按 case 等权的众数一致率和总变差距离。Every evaluator 报告 judgment 准确率、检索 Recall@1/3、MRR，以及冻结的十 action firewall 组合结果。

使用已发布评分路径重新生成逐行预测。已提交直接文件使用串行状态前缀复用；缓存命中不会改变 prompt 约定。

```bash
score_set () {
  input=$1
  stem=$2
  CUDA_VISIBLE_DEVICES=0 fastjev-score --mode serial \
    --model Qwen/Qwen3.5-4B \
    --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
    --input "$input" --output "direct-$stem.jsonl"
  CUDA_VISIBLE_DEVICES=0 fastjev-score --mode reranker \
    --model Qwen/Qwen3-Reranker-4B \
    --revision 22e683669bc0f0bd69640a1354a6d0aebcfeede5 \
    --input "$input" --output "reranker-$stem.jsonl"
}

score_set benchmarks/data/authored144.jsonl authored144
score_set "$OUT/wanli256.jsonl" wanli256
score_set "$OUT/typesafe102.jsonl" typesafe102
score_set "$OUT/every/inference204.jsonl" every204
```

重新计算自编数据和 WANLI 的硬标签指标，包括按源分组的配对比较：

```bash
python benchmarks/evaluate.py \
  --gold benchmarks/data/authored144.jsonl \
  --predictions reranker-authored144.jsonl \
  --comparison direct-authored144.jsonl \
  --output authored-report.json

python benchmarks/evaluate.py \
  --gold "$OUT/wanli256.jsonl" \
  --predictions reranker-wanli256.jsonl \
  --comparison direct-wanli256.jsonl \
  --output wanli-report.json
```

fetcher 设有字节数限制，并验证每个下载文件的 SHA-256。仓库不包含 TypeSafe 源记录。WANLI 使用 CC-BY-4.0；Every 以直接公开下载形式提供实验 JSON 和源 archive。

按来源划分的转换过程见[方法](../docs/METHOD.zh-CN.md)。验证每个已提交原始结果及其与机器可读摘要的关联：

```bash
(cd results/raw && sha256sum -c SHA256SUMS)
python benchmarks/verify_published.py
```
