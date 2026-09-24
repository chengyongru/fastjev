# 跨 state 批处理测量

[English](BATCHING.md) | 简体中文

`FastJev.decide_batch(states, questions)` 对多个 state 应用同一组问题。Torch 通过
`batch_size` 显式开启张量批处理，默认保持顺序评分。[SDK 指南](SDK.zh-CN.md#批量处理多个-state)
说明结果顺序、内存、计时和概率语义。

## 混合长度的公共 SDK 工作负载

[原始报告](../results/raw/runtime/rtx5090-cross-state-batching.json)包含 24 个自编
state，每个 state 回答三个类型化问题。每种模式预热一次、完整测量五次，每次保存
全部 72 个决策。输入混合中英文消息，并附带不同长度的无关上下文；它用于测吞吐与
结果漂移，不用于衡量准确率。

三种模式共享一个 Qwen3.5-4B BF16 实例，revision 为
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`，运行在 WSL2 的单张 RTX 5090 上。
环境为 PyTorch 2.10.0+cu128、Transformers 5.17.0，未安装可选的 causal-conv1d
和 flash-linear-attention 内核。计时包含 SDK 验证、编码、推理和结果转换，排除
加载及预热，并同步 CUDA。依次测量顺序、batch 8、长度分组三种模式。

| 执行方式 | 整次调用中位耗时 | 决策/秒 | 峰值分配 GiB | 选择变化 | 最大概率漂移 |
|---|---:|---:|---:|---:|---:|
| 顺序 | 4.666 秒 | 15.43 | 8.05 | 基线 | 0 |
| Batch 8 | 3.991 秒 | 18.04 | 9.53 | 0/360 | 0.037524 |
| Batch 8，按长度分组 | 2.975 秒 | 24.20 | 9.53 | 0/360 | 0.030953 |

该任务上，长度分组的吞吐为顺序执行的 1.57 倍。360 次比较是 72 个决策的五次重复，
不是 360 个独立样本。选择相同不代表概率相同，也不能证明校准有效。峰值分配包含
模型张量，不等于整个进程的显存占用。

## 冻结质量数据与 this-that 对比

[模型对比报告](../results/raw/runtime/rtx5090-cross-state-model-comparison-verified.json)
在相同硬件和精度下覆盖全部 144 行自编决策及 108 行自编扰动。两套数据各有 36 个
来源组；扰动来自自编原始样本，不是独立语料。测试保留原始问题、选项顺序和证据。
每种模式预热八行，然后对各数据集测量一遍；所有路径均未生成输出 token，也未截断输入。

缓存的 `flock-io/this-that-model-1.0` 使用固定 revision
`3d927195c4f9845efe66c5715883a7a0f42b1239`，runner 会检查权重 SHA-256。
原生接口使用 `thisthat` 源码 revision
`542d445efa5f68b14bfbd1f8ed25aacd8379d839`、`state_first` 布局、温度 1、每次一个
问题，并显式检查 4,096-token 上限。通用接口使用 FastJev 原有的
`direct-options-v1` prompt 与完整词表读出。这是完整推理路径的比较，不能只归因于架构。

| 模型 / 执行方式 | 自编数据，决策族等权平衡准确率 | 扰动数据，决策族等权平衡准确率 |
|---|---:|---:|
| Qwen3.5-4B，顺序 | 0.8132 | 0.7720 |
| Qwen3.5-4B，batch 8 + 长度分组 | 0.8132 | 0.7640 |
| this-that，FastJev 顺序 | 0.8660 | 0.7483 |
| this-that，FastJev batch 8 + 长度分组 | 0.8676 | 0.7434 |
| this-that，原生顺序 | 0.8286 | 0.8556 |

Qwen 批处理保留全部自编数据的选择，但改变了一个扰动样本的选择，使正确数从
83/108 降至 82/108。因此批处理仍需由部署方显式开启。this-that 值得继续验证：
该质量任务上，其顺序推理峰值分配约 3.5 GiB，Qwen 约 7.9 GiB。它的质量排名受
prompt 路径和工作负载影响；与顺序 Qwen 比较的所有配对来源组 bootstrap 95%
区间都包含零，目前不能认定存在普遍的质量优势。

置信度同样需要独立验证。自编集 ECE 分别为 Qwen 0.0567、通用 this-that 0.0855、
原生 this-that 0.0917；扰动集上原生 this-that 为 0.0708，Qwen 为 0.1256。
这些数据没有额外拟合温度。质量报告中的单遍耗时只用于诊断；专门的吞吐测量采用
上面重复五次的 SDK 工作负载。

## 复现

从仓库根目录运行，在独立环境中安装 `pip install -e '.[test,torch]'`。每个评分进程
只能看到一块 GPU，输出路径必须是新路径。

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 python benchmarks/cross_state_batching.py \
  --output /path/to/new-batching-report.json

# 仅原生 this-that 对比需要这个可选依赖。
pip install --no-deps \
  'thisthat @ git+https://github.com/FLock-io/this-that-model@542d445efa5f68b14bfbd1f8ed25aacd8379d839'
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 python benchmarks/compare_decision_models.py \
  --flock /path/to/this-that-model-1.0-3d927195 \
  --output /path/to/new-model-comparison.json

(cd results/raw && sha256sum -c SHA256SUMS)
python benchmarks/verify_published.py
```

`--flock` 目录需要包含从 Hugging Face 下载的完整固定模型和 tokenizer 文件。模型和
缓存不进入仓库。报告记录精确输入或自编 fixture 哈希、模型 revision、实现哈希、逐行
预测、耗时和内存；验证器从已提交预测重新计算批处理耗时、漂移及质量指标。
