# 复现指南

[English](REPRODUCE.md) | 简体中文

## 环境

创建隔离虚拟环境，并把缓存放到有足够模型权重空间的磁盘：

```bash
python -m venv .venv
. .venv/bin/activate
export HF_HOME=/path/to/large-drive/huggingface
pip install -r requirements.txt
pip install -e .
pytest -q
```

每个评分进程只使用一块 GPU。测量环境为 Linux x86_64 上的 Ubuntu 22.04、Python 3.10.12、NVIDIA driver 595.71.05、CUDA 12.8、PyTorch 2.10.0+cu128、Transformers 5.17.0、BF16 和 RTX 3090。`requirements.txt` 固定了实测 Python runtime package；启用 CUDA 的 PyTorch wheel 仍需要兼容的 NVIDIA driver。精确模型 commit ID 见 [../manifests/models.json](../manifests/models.json)。

`pytest -q` 运行全部核心测试和浏览器源测试。计时对硬件敏感，BF16/kernel 差异也可能改变临界概率或选择。提交的行数、schema、源 hash 和校验和是精确验收标准；计时和模型输出则应与已提交的逐行证据比较，不应当作字节完全一致的 golden output。

## 对项目自有示例评分

```bash
CUDA_VISIBLE_DEVICES=0 fastjev-score --mode direct \
  --model Qwen/Qwen3.5-4B \
  --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --input examples/decisions.jsonl --output results-direct.jsonl

CUDA_VISIBLE_DEVICES=0 fastjev-score --mode serial \
  --model Qwen/Qwen3.5-4B \
  --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --input examples/decisions.jsonl --output results-serial.jsonl

CUDA_VISIBLE_DEVICES=0 fastjev-score --mode reranker \
  --model Qwen/Qwen3-Reranker-4B \
  --revision 22e683669bc0f0bd69640a1354a6d0aebcfeede5 \
  --input examples/decisions.jsonl --output results-reranker.jsonl
```

命令会拒绝已存在的输出路径，也拒绝静默截断输入。每条输出都嵌入精确 revision、库版本、prompt hash、token 数、计时和明确的概率状态警告。状态可以是非空字符串、JSON 对象或 JSON 数组。`serial` 缓存连续相同的状态；`shared` 要求所有输入行具有完全相同的状态，由下文的 37×21 runner 覆盖。

## 复现 RTX 5090 GGUF 验证

使用与 runtime 匹配的上游 CUDA wheel index 安装 FastJev。已提交测量使用 CUDA
13.0：

```bash
pip install -e '.[llama-cpp]' \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu130
```

从 `bartowski/Qwen_Qwen3.5-4B-GGUF` revision
`4168f45a16a1290d65a4ec0fa312ae917a4c15d6` 下载
`Qwen_Qwen3.5-4B-Q4_K_M.gguf`。确认文件大小为 3,013,027,808 字节，SHA-256 为
`13c16f426047e2de38cd075bdade4a7bcbc8c774384876f677740cda65f8a983`，
再把 `MODEL` 设为其本地路径。

运行固定的公共 SDK safeguard 基准：

```bash
CUDA_VISIBLE_DEVICES=0 python benchmarks/llama_cpp_sdk.py \
  --model "$MODEL" \
  --revision 4168f45a16a1290d65a4ec0fa312ae917a4c15d6 \
  --fastjev-commit "$(git rev-parse HEAD)" \
  --output llama-cpp-sdk-run.json
```

在新路径重新生成两个完整的项目自有质量输出及报告：

```bash
CUDA_VISIBLE_DEVICES=0 fastjev-score --backend llama-cpp --mode direct \
  --model "$MODEL" \
  --revision 4168f45a16a1290d65a4ec0fa312ae917a4c15d6 \
  --llama-cpp-n-gpu-layers -1 --llama-cpp-n-batch 512 \
  --input benchmarks/data/authored144.jsonl \
  --output llama-cpp-authored144.jsonl

CUDA_VISIBLE_DEVICES=0 fastjev-score --backend llama-cpp --mode direct \
  --model "$MODEL" \
  --revision 4168f45a16a1290d65a4ec0fa312ae917a4c15d6 \
  --llama-cpp-n-gpu-layers -1 --llama-cpp-n-batch 512 \
  --input benchmarks/data/perturbations108.jsonl \
  --output llama-cpp-perturbations108.jsonl

python benchmarks/evaluate.py \
  --gold benchmarks/data/authored144.jsonl \
  --predictions llama-cpp-authored144.jsonl \
  --comparison results/raw/predictions/direct-authored144.jsonl \
  --output llama-cpp-authored144-report.json

python benchmarks/evaluate.py \
  --gold benchmarks/data/perturbations108.jsonl \
  --predictions llama-cpp-perturbations108.jsonl \
  --comparison results/raw/predictions/direct-perturbations108.jsonl \
  --output llama-cpp-perturbations108-report.json
```

scorer 与 benchmark runner 都拒绝已有输出路径。硬件计时允许变化；行数、模型身份、
工件 hash 和指标定义是固定的。

## 第三方评估

仓库不包含 TypeSafe 源记录。要复现该比较，请在源目录中提供本地快照。辅助脚本会下载其余公开评估输入并验证 hash：

```bash
python benchmarks/fetch_sources.py --output /path/on/large-drive/semif-sources
```

冻结的 706 行矩阵和源 ID 位于 `benchmarks/manifests/`。直接与 reranker 的逐行输出位于 `results/raw/predictions/`。完整的项目自有 144 行标注工作负载随 `benchmarks/data/authored144.jsonl` 分发。

按照[基准指南](../benchmarks/README.zh-CN.md#质量证据)中的命令构建精确的外部评估行并重新计算指标。builder 会验证源 hash 和冻结选择 ID；TypeSafe 与 Every evaluator 接受重建后的 gold 行及已提交的逐行预测。

## 复现扰动证据

从 36 个项目自有原始 case 重建冻结的 108 行 fixture，再确认它与已提交 fixture 一致：

```bash
python benchmarks/build_perturbations.py \
  --source benchmarks/data/authored144.jsonl \
  --output /tmp/perturbations108.jsonl \
  --manifest /tmp/perturbations108-manifest.json
cmp /tmp/perturbations108.jsonl benchmarks/data/perturbations108.jsonl
```

分别使用 `fastjev-score --mode serial` 和 `--mode reranker` 重新生成直接与 reranker 预测；也可以从随附的逐行预测重新计算与提交版本完全一致的报告：

```bash
python benchmarks/evaluate_perturbations.py \
  --gold benchmarks/data/authored144.jsonl \
  --perturbations benchmarks/data/perturbations108.jsonl \
  --direct-base results/raw/predictions/direct-authored144.jsonl \
  --direct-perturbations results/raw/predictions/direct-perturbations108.jsonl \
  --reranker-base results/raw/predictions/reranker-authored144.jsonl \
  --reranker-perturbations results/raw/predictions/reranker-perturbations108.jsonl \
  --output perturbation-report.json
cmp perturbation-report.json results/raw/perturbation-comparison.json
```

## 复现主要速度结果

运行三次重复的直接路径与紧凑数组生成集中比较：

```bash
CUDA_VISIBLE_DEVICES=0 python benchmarks/decision_vs_generation.py \
  --model Qwen/Qwen3.5-4B \
  --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --input benchmarks/data/shape777.jsonl \
  --output compact-array-run.json
```

运行完整 777 项决策的全新、串行缓存和并行共享状态比较：

```bash
CUDA_VISIBLE_DEVICES=0 python benchmarks/shape777.py \
  --model Qwen/Qwen3.5-4B \
  --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --input benchmarks/data/shape777.jsonl \
  --output shape777-run.json
```

按已发布的 pair batch size 运行完整原生 reranker 比较：

```bash
CUDA_VISIBLE_DEVICES=0 python benchmarks/shape777_reranker.py \
  --model Qwen/Qwen3-Reranker-4B \
  --revision 22e683669bc0f0bd69640a1354a6d0aebcfeede5 \
  --input benchmarks/data/shape777.jsonl \
  --pair-batch-sizes 1,4,8 \
  --output shape777-reranker-run.json
```

所有脚本都要求使用新的输出路径。计时在 warmup 后进行，包含 prompt 构造、tokenization、传输、模型执行和 CPU 读取；不包含模型加载和最终结果文件写入。

验证已提交的证据包，并确认机器可读摘要中的每个选定标量都与其原始报告一致：

```bash
(cd results/raw && sha256sum -c SHA256SUMS)
python benchmarks/verify_published.py
```

前述按来源划分的质量命令会重新生成 `results/raw/quality-comparison.json` 中的指标。`verify_published.py` 对照该报告以及扰动、系统和生成报告，检查 69 个已发布摘要值。它特意不要求 GPU 重跑结果达到字节完全一致。
