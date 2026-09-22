# 校准

[English](CALIBRATION.md) | 简体中文

直接评分器返回的是“以当前选项集合为条件”的原生概率，默认未校准。FastJev 提供按
工作负载绑定的后处理温度缩放，使概率阈值可以在经过验证的部署分布上使用。该步骤与
原生评分分离，不改变 prompt、原始预测或获胜选项。

## 方法

每个工作负载拟合一个标量 `T`，校准概率为
`softmax(option_logits / T)`。`T` 通过最小化该工作负载有标签样本的平均负对数似然
得到。正温度不会改变 logits 的顺序，因此 accuracy、balanced accuracy 和其他基于
argmax 的决策指标保持不变，只有置信度会变化。

`benchmarks/calibrate.py` 在 CPU 上读取已提交的 `option_logits`，拟合温度、执行分组
隔离的交叉验证，并可输出校准后的预测文件。公共 SDK 的
`TemperatureCalibration` 还能把 profile 绑定到 workload、backend、model、revision
和 prompt version，阻止跨身份误用。

## 冻结结果

全量样本拟合得到部署候选温度；ECE 使用按 `group_id` 隔离的五折交叉验证，避免同一
语义样本的变体同时进入拟合与评估。区间为按来源 group bootstrap 得到的 95% 区间。

| 工作负载 | 行数 | 模型准确率 | 拟合 `T` | 未校准 ECE | 自身 `T` 的 OOF ECE | 区间分离 |
|---|---:|---:|---:|---:|---:|:--:|
| 自编数据 | 144 | 0.806 | 1.23 | 0.068 | 0.038 | 否 |
| **WANLI（NLI）** | 256 | 0.637 | **2.50** | **0.208** | **0.069** | **是** |
| Every 有标签子集 | 154 | 0.942 | 1.71 | 0.050 | 0.047 | 否 |

主要结果来自 WANLI：原始分数明显过度自信，温度缩放将 ECE 从 0.208 降至 0.069，
bootstrap 区间不重叠。自编数据和 Every 的起始 ECE 已较低，校准前后区间重叠，不能
声称有可靠改善。

逐工作负载结果、区间、reliability bins、全量拟合温度及精确 fold 分配位于
`results/raw/calibration/{authored144,wanli256,every154}.json`；跨工作负载对照位于
`results/raw/calibration/summary.json`。

## 为什么按工作负载拟合

不同工作负载的候选温度差异明显。对照实验使用完全相同的逐行 fold 分配，只把
“每个工作负载单独拟合”改为“所有工作负载合并拟合”。结果如下：

| 工作负载 | 自身 `T` 的 OOF ECE | pooled `T` 的 OOF ECE | paired Δ 95% CI |
|---|---:|---:|---|
| 自编数据 | 0.038 | 0.081 | [-0.012, +0.073] |
| WANLI | 0.069 | 0.067 | [-0.043, +0.047] |
| Every | 0.047 | 0.053 | [-0.018, +0.030] |

三个 paired 区间都包含零，因此该对照没有证明 pooled 温度显著更差。按工作负载拟合的
理由是部署目标本身及温度差异，而不是 OOF ECE 必然占优。

## 使用边界

- 单一标量只能修正整体过度或不足自信，不能修正工作负载内部的形状误校准。
- 冻结评估只覆盖 hard-label 行；分布标签需要能感知目标分布的方法。
- 不得把这里的 Qwen3.5-4B BF16 温度直接用于 EXL3、GGUF、vLLM、MLX、其他 revision
  或其他 prompt。公共 SDK 会检查可验证的模型身份，但 workload 代表性仍需部署方负责。
- 校准置信度不能替代任务准确率、安全策略或人工复核。

## 复现

```bash
python benchmarks/fetch_sources.py --output build/sources
python benchmarks/build_wanli.py --source build/sources/wanli-test.jsonl --selection benchmarks/manifests/source-selection.jsonl --output build/gold-wanli256.jsonl
python benchmarks/build_every.py --archive build/sources/every-source.zip --experiments build/sources/every-experiments.json --selection benchmarks/manifests/source-selection.jsonl --output-dir build/every
python benchmarks/calibrate.py --gold benchmarks/data/authored144.jsonl --predictions results/raw/predictions/direct-authored144.jsonl --report build/authored144.json
python benchmarks/calibrate.py --gold build/gold-wanli256.jsonl --predictions results/raw/predictions/direct-wanli256.jsonl --report build/wanli256.json
python benchmarks/calibrate.py --gold build/every/gold154.jsonl --predictions results/raw/predictions/direct-every204.jsonl --report build/every154.json
python benchmarks/calibrate.py --manifest results/raw/calibration/workloads.json --summary build/summary.json
```

快速自检：

```bash
python -c "import sys; sys.path.insert(0,'benchmarks'); import calibrate; calibrate.demo()"
```

