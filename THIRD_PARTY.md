# Third-party material

No model weights or third-party raw evaluation records are distributed here.

| Item | Upstream | Pinned revision | Note |
|---|---|---|---|
| Qwen3.5-4B | https://huggingface.co/Qwen/Qwen3.5-4B | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` | Direct-logit baseline; check upstream model license. |
| Qwen3-Reranker-4B | https://huggingface.co/Qwen/Qwen3-Reranker-4B | `22e683669bc0f0bd69640a1354a6d0aebcfeede5` | Native reranker baseline; Apache-2.0 on its model card. |
| TypeSafe public evaluations | https://evals.typesafe.ai/ | Referenced for the published comparison; source records are not included. |
| Every parallel judgment lab | https://typesafe-parallel-judgment-lab.every-4573.chatgpt.site/ | [Experiment JSON](https://typesafe-parallel-judgment-lab.every-4573.chatgpt.site/downloads/experiments.json) and [source archive](https://typesafe-parallel-judgment-lab.every-4573.chatgpt.site/downloads/typesafe-lab-source.zip) are public downloads. |
| WANLI | https://huggingface.co/datasets/alisawuffles/WANLI | Pinned at `61c95318fd71c55b6ba355d76253254615f387ec`; CC-BY-4.0. |
| wllama | https://github.com/ngxson/wllama | `3.6.1` | Vendored browser inference runtime; MIT. |
| Vue | https://github.com/vuejs/core | `3.5.21` | Browser demo UI runtime; MIT. |
| Material Symbols | https://fonts.google.com/icons | Google Fonts CDN | Browser demo icons; Apache-2.0. |
| Jev Ultrafast | https://github.com/browser-use/jev-ultrafast | `1231850a0bf1a0c0341fe408ef1668dbbfdfac46` | Executed by the verified browser demo; MIT; source is not redistributed. Checked 2026-09-22. |
| Qwen3-0.6B GGUF | https://huggingface.co/Qwen/Qwen3-0.6B-GGUF | `23749fefcc72300e3a2ad315e1317431b06b590a` | External Q8_0 browser model; weights are not redistributed. |
| MiniCPM5-2B GGUF | https://huggingface.co/openbmb/MiniCPM5-2B-GGUF | `2079a22f3beaa4e306449978533478fe0522f4b3` | External Q4_K_M browser model; Apache-2.0 on its model card; weights are not redistributed. |
| Qwen3.5-4B GGUF | https://huggingface.co/bartowski/Qwen_Qwen3.5-4B-GGUF | `4168f45a16a1290d65a4ec0fa312ae917a4c15d6` | External Q4_K_M browser model; weights are not redistributed. |
| Qwen3.8-27B EXL3 | https://huggingface.co/turboderp/Qwen3.8-27B-exl3 | `a35e75a73baee51da709329d19294245cbeeb5d8` (5.00 bpw) | Optional EXL3 validation checkpoint; upstream terms and the Qwen base-model license apply. |
| ExLlamaV3 | https://github.com/turboderp-org/exllamav3 | `1.5.1` | Optional EXL3 inference runtime; MIT. |
| this-that-model-1.0 | https://huggingface.co/flock-io/this-that-model-1.0 | `3d927195c4f9845efe66c5715883a7a0f42b1239` | Optional comparison checkpoint; MIT on its model card; weights are not redistributed. |

The original URLs were checked on 2026-09-18; the EXL3 entries were checked on
2026-09-22. Downloaded evaluation inputs are pinned by SHA-256.

fastjev is an independently maintained fork of [TheoLeeCJ/SemIf](https://github.com/TheoLeeCJ/SemIf), formerly OpenJev. It is not affiliated with or endorsed by TheoLeeCJ, TypeSafe, or Jev. Jev, TypeSafe, and other names and marks are the property of their respective owners. The inherited code and fastjev modifications are provided under the repository's MIT License; third-party models and material retain their upstream terms.
