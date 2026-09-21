# Apple Silicon / MLX

[English](MLX.md) | 简体中文

原生 MLX 后端在 macOS arm64 上运行 fastjev 的直接、串行前缀和并行共享决策模式。它使用 MLX-LM 的 Qwen3.5 实现，并采用与 Torch 后端相同的 prompt 和答案 token 检查，不生成答案 token。浏览器 demo 是独立实现。

## 安装与评分

在支持 Metal 的 Apple Silicon Mac 上使用隔离 Python 环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[test,mlx]'

fastjev-score --backend mlx --mode direct \
  --model Qwen/Qwen3.5-4B \
  --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --input examples/decisions.jsonl \
  --output results-mlx-direct.jsonl
```

首次运行会将固定 checkpoint 下载到 Hugging Face 缓存。权重占用约 9 GB 磁盘空间，GPU 执行还需要额外内存。源 checkpoint 包含视觉权重，MLX-LM 原生 sanitizer 会将它们排除在文本模型之外。基线保留源精度（BF16，部分参数为 FP32）。runtime 固定为 MLX 0.32.2，以及 commit `a63e24c389382619eb6d9af656e3b46024be217a` 的 MLX-LM（package version 0.32.0）。该源 revision 包含上游 Qwen recurrent q/k normalization 修复；0.31.3 版本对 L2 epsilon 的应用不正确。安装过程需要 Git。每条预测都会记录已安装 runtime 的源 commit。

使用 `--mode serial` 复用连续且完全相同的状态。所有输入行具有完全相同状态且决策 ID 唯一时，可使用 `--mode shared`。shared 模式在一个 batch 中评估所有问题；内存占用随 batch size 和后缀长度增长。输入限制强制执行，不做截断。

`--mlx-bits 8` 或 `--mlx-bits 4` 会在内存中应用确定性的 affine quantization，group size 为 64。它从同一个固定 checkpoint 开始，不写入另一份模型工件。结果会记录该转换和源文件 hash。量化会改变模型概率，必须相对未量化 MLX 运行单独评估。本地模型目录需要 revision 标签并同样计算 hash；远程 revision 必须是不可变的 40 字符 commit ID。

默认后端仍是 Torch/CUDA。MLX 明确不支持 reranker 模式。其他平台继续使用现有 Torch 路径，无需导入 MLX。MLX extra 只适用于 macOS arm64，不替代仓库已有的 Torch 依赖。

loader 默认将 MLX 非活跃分配缓存上限设为 256 MiB。使用 `--mlx-cache-limit-mib 512` 修改，或用 `--mlx-cache-limit-mib 0` 禁用非活跃分配缓存。基准和精度探测脚本接受同一参数；Python 调用者可向 `mlx_backend.load_model` 传入 `cache_limit_mib=512`。预测元数据会以字节记录有效上限。这是进程级 MLX allocator 设置，不是前缀缓存。MLX 默认缓存会在变长 prompt 之间保留几乎全部系统内存，不适合其他本地模型共享统一内存的场景。这个上限只约束分配缓存，不约束活动模型或 batch 内存需求。

## Apple Silicon demo

![Apple M5 Max 上的原生 MLX CLI](media/openjev-mlx.png)

[可重放终端记录与截图说明](media/README.md)。截图展示了使用原 OpenJev 名称执行的真实本地 CLI 完整录制；当前命令使用 `fastjev-score`。

## 缓存正确性

Qwen3.5 将 attention history 与 recurrent convolution/delta state 结合。串行评分会为每个问题深拷贝完整原生前缀缓存。并行评分合并原生缓存副本，对问题后缀右侧 padding，将真实长度提供给 recurrent cache，并读取各后缀最后一个真实 token。任何分支的状态都不会输入另一个问题。

保留缓存以精确的前缀 token ID 为键：调用者修改此前传入的 JSON 对象不会意外复用陈旧缓存。测试使用一个小型真实 Qwen3.5 hybrid 模型，覆盖变长后缀、问题重排、重复调用、状态变化和非法输入，而且不下载权重。

不同 kernel、prompt 和 batch shape 的 GPU 运算会有差异。pilot 与 review 阈值冻结在 `manifests/mlx-validation.json`。报告会列出每个发生变化的选择；类型化输出不保证语义正确，softmax 分数也不是校准置信度。

## 压缩保留证据

为控制贡献体积，大型历史报告使用无损 gzip 压缩。验证脚本和参考运行读取器可透明接受 `.gz` 文件；新的基准运行继续保留原始纯 JSON/JSONL 格式。原始 payload 校验和及复现细节见[证据索引](../results/mlx/README.md)。

## 复现证据

在仓库根目录、已激活环境中运行。每个输出目录必须是新目录；中断运行保留为部分证据，不会被覆盖。GPU 基准每次只运行一个进程。

```bash
python benchmarks/mlx_benchmark.py --suite diagnostic --output results/mlx/my-pilot
python benchmarks/mlx_benchmark.py --suite all --output results/mlx/my-bf16
python benchmarks/mlx_benchmark.py --suite quantization --bits 8 --reference-run results/mlx/my-bf16 --output results/mlx/my-q8
python benchmarks/mlx_benchmark.py --suite quantization --bits 4 --reference-run results/mlx/my-bf16 --output results/mlx/my-q4
```

runner 记录：

- **Diagnostic：** 精确 tokenizer 等价性；小型 pilot 上的全新、串行、并行和重排分数。
- **Quality：** 144 个自编 case 和 108 个扰动 case、现有 evaluator 指标、缺失证据行为，以及相对已发布 Torch 预测的逐行差异。已发布预测使用串行前缀复用，因此该比较同时包含后端差异和执行形状差异。
- **Systems：** 全部 777 项决策的全新、串行和并行模式，包括逐状态延迟、MLX 峰值分配量，以及相对全新路径发生变化的每个选择。
- **Generation：** 三次重复比较 21 个直接分布与同一模型写出的紧凑 yes/no 数组。记录原始输出、有效性、首 token 时间、完成时间和一致率。无效生成答案记为失败，不视为等价的更快或更慢答案。

量化 suite 运行 diagnostic、quality 和 generation 比较。使用 `--suite all --bits 8`（或 `4`）可额外让该精度下的全部 777 项决策经过三种模式。

模型加载、工件 hash 和初始 warmup 不在计时范围内。执行测量包含 prompt 渲染、tokenization、评估、同步和 CPU 读取。MLX 使用 lazy evaluation，因此必须显式求值数组并等待 GPU 完成。MLX 峰值分配包括模型权重和临时数组，不是 macOS 进程总内存，也不能直接等同于 CUDA allocator 指标。CUDA 和 Mac 计时描述的是不同硬件。

## 验证

```bash
pytest -q
(cd results/raw && shasum -a 256 -c SHA256SUMS)
(cd results/mlx && shasum -a 256 -c SHA256SUMS)
python benchmarks/mlx_evidence.py results/mlx/UNCOMPRESSED_SHA256SUMS
python benchmarks/verify_published.py
python benchmarks/verify_mlx.py results/mlx/my-bf16
```

原始 CUDA 摘要和证据继续保留。Mac 测量存放在 `results/mlx/` 下各自带日期的目录中。包括数值差异和已被取代实验在内的内容，见[测量结果与运行历史](../results/mlx/README.md)。

如需深入调查概率差异，请在计时 GPU 基准完成后运行 FP32 diagnostic：

```bash
python benchmarks/mlx_precision_probe.py \
  --run results/mlx/my-bf16 --output results/mlx/my-bf16/precision-probe.json
```

它会选择超过冻结概率 review 阈值的质量行和所有选择发生变化的行，比较原生 BF16、FP32 MLX 执行与 PyTorch CPU FP32 参考，并单独诊断 MLX-LM 折叠 Qwen offset RMSNorm 权重时产生的舍入差异。它不会改变生产模型加载方式，也不会以诊断结果替代基准。

## Runtime 参考

- [固定的 Qwen3.5 实现](https://github.com/ml-explore/mlx-lm/blob/a63e24c389382619eb6d9af656e3b46024be217a/mlx_lm/models/qwen3_5.py)
- [固定的原生缓存 API](https://github.com/ml-explore/mlx-lm/blob/a63e24c389382619eb6d9af656e3b46024be217a/mlx_lm/models/cache.py)
- [上游 normalization 修复](https://github.com/ml-explore/mlx-lm/commit/a63e24c389382619eb6d9af656e3b46024be217a)
- [MLX lazy evaluation](https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html)
