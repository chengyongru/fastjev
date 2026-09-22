# 方法

[English](METHOD.md) | 简体中文

## 问题与系统

第一阶段检验开放、无生成的读取方式能否复现 Jev 公开主张中的实用部分：接受非结构化状态和运行时定义的自然语言决策，并以足够低的成本返回可嵌入普通软件的类型化分数。

直接系统向冻结的 Qwen3.5-4B 提供状态、标准和 2–16 个带说明的选项。它执行一次原生前向传播，只对固定大写答案 token 的 logits 做 softmax，不解码任何 token。

reranker 系统遵循 Qwen3-Reranker-4B 原生的 yes/no 约定。每个候选答案都变成独立的 query/document 相关性命题。系统为每个选项计算 `logit(yes) - logit(no)`，再在所有选项的 log-odds 上做 softmax。最后这一步归一化是本项目的比较规则，不属于上游 reranker 的校准约定。

## 冻结评估矩阵

在评估完整 reranker 输出前，prompt、ID、标签、任务语义、revision 和指标都已冻结。完整本地矩阵共 706 行：

| 来源 | 行数 | 用途 |
|---|---:|---|
| 项目自编 | 144 | 证据解释、规则应用、候选选择；包括原始 case 和缺失证据 case |
| WANLI | 256 | 外部自然语言推理 |
| TypeSafe 公开评估子集 | 102 | 20 个可用 case 上的分布/参考一致率 |
| Every 公开实验工件 | 204 | Judgment grid、检索、公司知识和组合 action policy |

不同性质的任务没有被压缩成一个准确率。硬标签任务使用完整分母准确率、平衡准确率、宏 F1、适用时的 NLL/Brier，以及按源分组的 bootstrap 区间。检索任务报告 query 排序指标。TypeSafe 行比较分布并按 case 等权宏平均，问题较多的 case 不会占据更高权重。

TypeSafe 比较只包含本地可对齐的 102 条公开记录，不是其宣传的 711 行汇总，也不是在线 Jev 运行。已发布的 Jev、Opus 和 Sol 数值读取自这些公开记录。这个候选版本不包含第三方原始 fixture。

精确评估 ID 已提交到 `benchmarks/manifests/source-selection.jsonl`。WANLI 使用 revision `61c95318fd71c55b6ba355d76253254615f387ec`：排除格式错误、超过 4,000 字符，以及与 pilot 训练源相连的行或 component；剩余 ID 排序后以 seed 291607 打乱，选取 86 条 entailment、85 条 contradiction、85 条 neutral，并保证每个相连 premise/pair-ID component 最多一行。premise 作为状态，hypothesis 作为标准；entailment/neutral/contradiction 分别映射到 supported/insufficient/contradicted。

TypeSafe 提取读取四个由本地提供且经 hash 验证的 `*-cases.js` 快照。只保留已发布且成功运行的 Choice/Noul 节点，同时要求 document/question 绑定无歧义、存在已发布参考答案、参考分布 argmax 唯一。选择过程不观察结果，按 workflow、case 和 primitive 轮询，各 bucket 内按 hash ID 排序。102 行比较不包含 Score primitive 和并列目标。Every 映射及原始实验数据可从直接链接的实验 JSON 和源 archive 获取。

## 扰动测试

36 个项目自有原始 case 各生成三种不观察输出的变体：反转展示选项顺序但保留语义 ID；用保留语义的措辞包裹标准；追加项目自有的无关上下文。另有 36 行缺失证据总体，用于检验系统是否选择 `insufficient`。稳定性指标在按语义选项 ID 对齐概率后计算。

## 浏览器模型梯度

Qwen3-0.6B、MiniCPM5-2B 和 Qwen3.5-4B 在 144 条自编数据、108 条扰动数据和 102 条选定 TypeSafe 数据上使用相同冻结 prompt 与原生 BF16 末位置选项 logits 评分器。TypeSafe 众数一致率先在 20 个源 case 内分别平均，再对 case 等权平均。浏览器工件是独立固定的 GGUF 量化版本。浏览器冒烟测试计时从页面发起每项操作后开始；模型文件由本地 SSD 提供，不计网络传输。成功冒烟要求模型加载、warmup、所有展示选项 logits 有限，并完成生成路径；它不能证明完整量化质量或可迁移延迟。

## llama.cpp GGUF 验证

服务端 GGUF 验证使用 `bartowski/Qwen_Qwen3.5-4B-GGUF` revision
`4168f45a16a1290d65a4ec0fa312ae917a4c15d6`、精确文件
`Qwen_Qwen3.5-4B-Q4_K_M.gguf`，其 SHA-256 为
`13c16f426047e2de38cd075bdade4a7bcbc8c774384876f677740cda65f8a983`。
commit `e9ee737` 的 FastJev 0.1.1 通过 `llama-cpp-python` 0.3.35 的 CUDA
13.0 wheel 运行，设置 `n_gpu_layers=-1`、`n_batch=512`，并在 WSL2 下只暴露
一块 RTX 5090。runtime 的 GPU-offload 探针必须返回 true。

质量测试与主要的已提交 Torch BF16 预测使用相同 prompt 约定和 evaluator。llama.cpp
direct mode 分别评分全部 144 行自编数据与 108 行扰动数据；缺失或无效行仍按失败计入。
配对的 BF16 文件使用原生状态前缀缓存。报告在这些完整 runtime 路径之间按语义 ID
对齐 argmax，因此没有把量化影响与 serving shape、kernel 影响相互隔离。

`benchmarks/llama_cpp_sdk.py` 定义另一项固定的三问题 shell-safeguard 冒烟测试，
不会执行任何命令。SDK 构造耗时包括 GGUF 加载和用于来源记录的 SHA-256 遍历。
一次不计时 warmup 后，围绕公共 API 测量七次完整 `decide_many` 调用。每次调用包含
相同三个问题，当前 llama.cpp 会按顺序执行它们。模型传输、进程启动、结果序列化和
warmup 不计入所报告的中位数。GPU 显存来自整卡 `nvidia-smi` 观测，不是仅限 allocator
的测量。

## 形状匹配的系统基准

项目自有 fixture 包含 37 个状态，每个状态使用 21 项固定二元标准，共 777 项决策。状态长度约 8,000 字符，用于测试重复上下文计算。它与公开 Every/Jev 演示具有相同的计数结构，但不复现其未发布文档、token 长度、硬件、API 路径或模型。因此，这是一项系统测量，不是与 Jev 的正面对比基准。

直接模式包括全新 batch-one 评分、一次状态预填充后的串行后缀，以及一次状态预填充后的并行后缀分支。对每项二元决策，reranker 为两个独立的 yes/no 选项对重复状态，并测试常规 pair batching。计时使用一块 RTX 3090 和 warm-loaded BF16 模型，包含 prompt 构造、tokenization、传输、前向传播和 CPU 读取；模型加载和结果文件写入不在计时范围内。

## 解释规则

- 强制类型化输出仍然可能在语义上错误。
- 对允许 token 做 softmax 得到的是以给定选项为条件的概率，不是经过校准的运行时置信度。
- 前缀缓存加速是实现结果，不是 Jev 已披露架构的证据。
- reranker 理应在排序上表现最好，不能把它的分类阈值指标与排序质量混为一谈。
