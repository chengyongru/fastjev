# 第三方材料

[English](THIRD_PARTY.md) | 简体中文

本仓库不分发模型权重或第三方原始评估记录。

| 项目 | 上游 | 固定 revision | 说明 |
|---|---|---|---|
| Qwen3.5-4B | https://huggingface.co/Qwen/Qwen3.5-4B | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` | 直接 logits 基线；请查看上游模型许可证。 |
| Qwen3-Reranker-4B | https://huggingface.co/Qwen/Qwen3-Reranker-4B | `22e683669bc0f0bd69640a1354a6d0aebcfeede5` | 原生 reranker 基线；其 model card 标注 Apache-2.0。 |
| TypeSafe 公开评估 | https://evals.typesafe.ai/ | — | 用于已发布比较；仓库不包含源记录。 |
| Every parallel judgment lab | https://typesafe-parallel-judgment-lab.every-4573.chatgpt.site/ | 公开下载 | 提供[实验 JSON](https://typesafe-parallel-judgment-lab.every-4573.chatgpt.site/downloads/experiments.json)和[源 archive](https://typesafe-parallel-judgment-lab.every-4573.chatgpt.site/downloads/typesafe-lab-source.zip)。 |
| WANLI | https://huggingface.co/datasets/alisawuffles/WANLI | `61c95318fd71c55b6ba355d76253254615f387ec` | CC-BY-4.0。 |
| wllama | https://github.com/ngxson/wllama | `3.6.1` | 随仓库提供的浏览器推理 runtime；MIT。 |
| Vue | https://github.com/vuejs/core | `3.5.21` | 浏览器 demo UI runtime；MIT。 |
| Material Symbols | https://fonts.google.com/icons | Google Fonts CDN | 浏览器 demo 图标；Apache-2.0。 |
| Jev Ultrafast | https://github.com/browser-use/jev-ultrafast | `1231850a0bf1a0c0341fe408ef1668dbbfdfac46` | 经过验证的浏览器 demo 会执行该版本；MIT；本仓库不重新分发其源码。检查日期：2026-09-22。 |
| Qwen3-0.6B GGUF | https://huggingface.co/Qwen/Qwen3-0.6B-GGUF | `23749fefcc72300e3a2ad315e1317431b06b590a` | 外部 Q8_0 浏览器模型；不在本仓库重新分发权重。 |
| MiniCPM5-2B GGUF | https://huggingface.co/openbmb/MiniCPM5-2B-GGUF | `2079a22f3beaa4e306449978533478fe0522f4b3` | 外部 Q4_K_M 浏览器模型；其 model card 标注 Apache-2.0；不在本仓库重新分发权重。 |
| Qwen3.5-4B GGUF | https://huggingface.co/bartowski/Qwen_Qwen3.5-4B-GGUF | `4168f45a16a1290d65a4ec0fa312ae917a4c15d6` | 外部 Q4_K_M 浏览器模型；不在本仓库重新分发权重。 |
| Qwen3.8-27B EXL3 | https://huggingface.co/turboderp/Qwen3.8-27B-exl3 | `a35e75a73baee51da709329d19294245cbeeb5d8`（5.00 bpw） | 可选 EXL3 验证 checkpoint；沿用上游条款与 Qwen 基础模型许可证。 |
| ExLlamaV3 | https://github.com/turboderp-org/exllamav3 | `1.5.1` | 可选 EXL3 推理 runtime；MIT。 |

原有 URL 于 2026-09-18 完成检查；EXL3 条目于 2026-09-22 检查。下载的评估输入由 SHA-256 固定。

fastjev 是 [TheoLeeCJ/SemIf](https://github.com/TheoLeeCJ/SemIf)（原名 OpenJev）的独立维护分支，与 TheoLeeCJ、TypeSafe 或 Jev 无隶属关系，也未获得其背书。Jev、TypeSafe 及其他名称和商标归各自权利人所有。继承代码与 fastjev 修改按仓库 MIT 许可证提供；第三方模型和材料沿用其上游条款。
