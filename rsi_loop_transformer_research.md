# RSI 与 Looped Transformer 研究资料汇编

> 主题：递归自我改进（Recursive Self-Improvement, RSI）与循环 Transformer（Looped / Recurrent-Depth Transformer）的相关论文与开源项目汇总。
> 检索时间：2026-10-10。条目数：论文 33 篇 + 开源项目/模型 9 个 + 权威综述/博客 5 篇，合计 47 条。

## 1. 概述

两条技术脉络在 2025–2026 年出现交汇：

- **RSI 脉络**：让模型/智能体改进"改进自身的能力"。源头可追溯到 I.J. Good 的"超智能机器"设想[cite:18]与 Schmidhuber 的 Gödel Machine 理论[cite:5]，近期落地为 Darwin Gödel Machine、STOP、SEAL、Absolute Zero 等可运行系统[cite:1][cite:4][cite:6][cite:8]。
- **Looped Transformer 脉络**：通过共享权重、在深度方向反复迭代同一模块，在不增加参数的前提下扩展有效深度与测试时计算。源头为 Universal Transformer[cite:19]与 ALBERT 权重共享[cite:22]，近期落地为 Huginn、Ouro、Mixture-of-Recursions 等[cite:24][cite:25][cite:27]。
- **交汇点**：以"循环深度"作为自我改进载体的工作（如 Metan 的 emergent depth、DeepLoop 的深度缩放）[cite:11][cite:37]，以及把循环架构与自进化训练结合的系统[cite:22][cite:35]。

## 2. RSI（递归自我改进）论文与项目

### 2.1 奠基性工作

| # | 名称 | 类型 | 年份 | 关键贡献 |
|---|------|------|------|----------|
| 1 | Speculations concerning the first ultraintelligent machine (I.J. Good) | 论文 | 1965/66 | 提出"智能爆炸"概念，RSI 思想的源头[cite:18] |
| 2 | Gödel Machine (Schmidhuber) | 理论论文 | 2003 | 可证明自改进代理的理论基础，因形式化证明要求而难以实用[cite:5][cite:11] |
| 3 | STaR: Self-Taught Reasoner | 论文 | 2022 | 模型自举生成推理链并微调，自我改进推理的早期范式[cite:7] |

### 2.2 现代 RSI 系统（LLM/智能体）

| # | 名称 | 类型 | 年份 | 关键贡献 |
|---|------|------|------|----------|
| 4 | Darwin Gödel Machine (DGM) | 论文+代码 | 2025 | 开放式进化的自改进编码代理，重写自身代码并以基准验证，构建代理档案库[cite:1][cite:2][cite:3] |
| 5 | STOP: Self-Taught Optimizer | 论文 | 2023 | 用 LM 改进"改进器脚手架"并递归改进自身[cite:4] |
| 6 | Gödel Agent | 论文 | 2024 | 自指框架，同时修改任务策略 π 与元级改进逻辑 I[cite:5][cite:12] |
| 7 | SEAL: Self-Adapting Language Models | 论文 | 2025 | 模型生成自编辑（数据+超参+工具）做 SFT，实现持久权重更新[cite:6][cite:17] |
| 8 | Absolute Zero Reasoner (AZR) | 论文 | 2025 | 零外部数据，自提任务+自解+代码执行器验证的 RLVR 范式[cite:8] |
| 9 | R-Zero | 论文 | 2025 | Challenger–Solver 协同进化，从零数据自演化推理[cite:9] |
| 10 | Self-Reference in LLMs: Introspection Threshold for RSI | 论文 | 2026 | 提出自我改进的"内省阈值"，讨论 model collapse 风险[cite:10] |
| 11 | Metan: Recursive Self-Improvement through Emergent Depth | 论文 | 2026 | 把 RSI 与循环深度结合， emergent depth 作为自改进载体[cite:11] |
| 12 | Polaris: Gödel Agent Framework for Small LMs | 论文 | 2026 | 在小模型上实现经验抽象的策略修复式 RSI[cite:12] |
| 13 | Recursive Harness Self-Improvement for Frontier Reasoning | 论文 | 2026 | 改进"推理数据合成 harness"的递归自改进[cite:16] |
| 14 | From 0-to-1 to 1-to-N: MetaAI Recursive Self-Design | 论文 | 2026 | 可复现的元自设计证据，引用 STOP/Gödel Agent 作二级证据[cite:13] |

### 2.3 RSI 相关开源项目/模型

| # | 名称 | 链接 | 说明 |
|---|------|------|------|
| 15 | DGM 官方代码 | github.com/jennyzzt/dgm[cite:3] | Sakana AI 开源实现 |
| 16 | MetaAI-Mini | github.com/DunLi-Tsinghua/MetaAI-Mini[cite:13] | 最小可复现递归自设计实验 |
| 17 | RSIAI0 / DarwinGodel-Machine-ASI | github.com/BrandonDavidJones1/DarwinGodel-Machine-ASI[cite:14] | LLM 引导递归自修改的开源探索框架 |
| 18 | evoseal (PyPI) | pypi.org/project/evoseal[cite:17] | 结合 OpenEvolve + SEAL + DGM 的进化式代码改进库 |
| 19 | Karpathy `autoresearch` | 见 36Kr 报道[cite:16] | 2026 年开源的 AI 自动研究代理项目 |

## 3. Looped Transformer 论文与项目

### 3.1 奠基性工作

| # | 名称 | 类型 | 年份 | 关键贡献 |
|---|------|------|------|----------|
| 20 | Universal Transformers | 论文 | 2019 | 时序并行自注意力循环模型 + 动态逐位置 halting[cite:19][cite:20] |
| 21 | ALBERT | 论文 | 2019 | 全部 Transformer 层共享权重的权重共享思想[cite:22] |
| 22 | Looped Transformers are Better at Learning Learning Algorithms | 论文 | 2023/ICLR24 | 循环 Transformer 在 in-context 学习算法上优于标准 Transformer[cite:23] |
| 23 | On Expressive Power of Looped Transformers (Timestep Encoding) | 论文 | 2024 | 理论分析表达力，提出时间步编码增强[cite:30] |

### 3.2 现代循环架构（深度缩放 / 测试时计算）

| # | 名称 | 类型 | 年份 | 关键贡献 |
|---|------|------|------|----------|
| 24 | Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach (Huginn) | 论文+模型 | 2025 | 隐空间反复迭代 recurrent block，测试时任意展开深度；3.5B 模型 Huginn-0125[cite:24][cite:40] |
| 25 | Ouro: Scaling Latent Reasoning via Looped Language Models | 论文+模型 | 2025 | 开源 LoopLM 1.4B/2.6B，7.7T tokens，熵正则深度分配，匹敌 12B[cite:25][cite:26] |
| 26 | Mixture-of-Recursions (MoR) | 论文 | 2025/NeurIPS | 端到端训练轻量路由器，按 token 分配递归深度，统一参数效率+自适应计算[cite:27][cite:28] |
| 27 | MoR-ViT | 论文 | 2025 | 把 MoR 的 token 级动态递归引入视觉 Transformer[cite:29] |
| 28 | RecurrentGPT: Expressive Depth through Recurrent Modulation | 论文 | 2026 | 循环调制实现表达性深度，回顾 ALBERT/UT 脉络[cite:22] |
| 29 | Hyperloop Transformers | 论文 | 2026 | MIT 提出的新型循环 Transformer 变体[cite:31] |
| 30 | When Does Recurrence Become an Algorithm? (Convergence Selection) | 论文 | 2026 | 研究权重绑定循环 Transformer 的收敛选择[cite:32] |
| 31 | Looped Transformers under the Jacobian Lens | 论文 | 2026 | 用 Jacobian 透镜分析循环模型的全局工作空间[cite:33] |
| 32 | Depth through Recurrence: Looped Transformers for Flow-Matching TTS | 论文 | 2026 | 把循环深度组织方式用于流匹配语音合成[cite:34] |
| 33 | Encode, Think, Decode: Scaling Test-Time Reasoning with Recursive Latent Thoughts | 论文 | 2025 | 递归隐式思考 + 自适应深度测试时推理[cite:35] |
| 34 | Improving Test-Time Scaling with Adaptive Looped Transformers | 论文 | 2026 | 对比 Ouro 全栈循环与 Huginn 中段循环的自适应测试时扩展[cite:36] |
| 35 | DeepLoop: Depth Scaling for Looped Transformers | 论文 | 智源 | 研究循环 Transformer 的残差缩放问题[cite:37] |
| 36 | Universal Reasoning Model | 论文 | 2025 | 深度方向权重绑定 + 灵活步数 T 的推理模型[cite:42] |
| 37 | Beyond Memorization: Extending Reasoning Depth with Recurrence | 论文 | 2025 | 结合循环、记忆与测试时计算扩展推理深度[cite:41] |

### 3.3 Looped Transformer 相关开源项目/模型

| # | 名称 | 链接 | 说明 |
|---|------|------|------|
| 38 | Ouro 模型族 | ouro-llm.github.io[cite:26] | 1.4B/2.6B 开源 LoopLM |
| 39 | Huginn-0125 权重 | huggingface: tomg-group-umd/huginn-0125[cite:40] | 3.5B 循环深度模型检查点 |
| 40 | MoR 代码 | 见 NeurIPS 版[cite:28] | Mixture-of-Recursions 官方实现 |

## 4. 权威综述 / 解读博客

| # | 名称 | 来源 | 价值 |
|---|------|------|------|
| 41 | Harness Engineering for Self-Improvement | Lilian Weng, lilianweng.github.io[cite:15] | RSI 概念史（Good→Yudkowsky）与工程视角综述 |
| 42 | Sebastian Raschka: Looped Transformer 架构画廊 | sebastianraschka.com[cite:38] | 梳理 ALBERT→UT→Recurrent Depth→MoR→Ouro 谱系 |
| 43 | Raschka 论循环 Transformer 与 GPT-6 Astra | cere-bro 博客[cite:39] | 第三方复现两循环结果与 KV 共享失败分析 |
| 44 | What Is a Looped Transformer? (Astra 指南) | tosea.ai[cite:43] | 循环 Transformer 时间线与贡献表 |
| 45 | 当 AI 开始造 AI：递归自我改进的 2026 年 | 36Kr[cite:16] | 中文 RSI 进展与 Karpathy autoresearch 报道 |

## 5. 分析与观察

- **两条脉络正在融合**：Metan、DeepLoop、Encode-Think-Decode 等把"循环深度"作为模型在推理时自我精炼的机制，可视为 RSI 思想在架构层面的具象化[cite:11][cite:35][cite:37]。
- **RSI 的瓶颈**：DGM 等系统仍依赖外部基准与人工评测；Self-Reference 论文指出 model collapse 与内省阈值是主要障碍[cite:1][cite:10]。
- **循环架构的瓶颈**：残差缩放（DeepLoop）、收敛选择（When Does Recurrence…）、KV 共享失效（Raschka 复现）是开放问题[cite:32][cite:37][cite:39]。
- **开源可复现性**：DGM、Ouro、Huginn-0125、MoR、MetaAI-Mini 均提供公开代码/权重，便于后续实验[cite:3][cite:26][cite:40][cite:28][cite:13]。

## 6. 知识缺口声明

- ALBERT 原始论文（arXiv:1909.11942）未在检索中直接命中，仅经 RecurrentGPT 间接引用[cite:22]，建议补充核对。
- 部分 2026 年 arXiv 编号（如 2607/2608/2609 系列）来自检索结果快照，正式发表状态需二次确认。

## Sources

- [cite:1] Darwin Gödel Machine: Open-Ended Evolution of Self-Improving Agents — https://arxiv.org/pdf/2505.22954v3
- [cite:2] The Darwin Gödel Machine (Sakana AI blog) — https://sakana.ai/dgm/
- [cite:3] DGM 代码仓库 — https://github.com/jennyzzt/dgm
- [cite:4] Self-Taught Optimizer (STOP) — https://arxiv.org/pdf/2310.02304v2.pdf
- [cite:5] Gödel Agent: A Self-Referential Framework for RSI — https://arxiv.org/html/2410.04444
- [cite:6] Self-Adapting Language Models (SEAL) — https://arxiv.org/html/2506.10943
- [cite:7] STaR: Self-Taught Reasoner — https://arxiv.org/pdf/2203.14465v2.pdf
- [cite:8] Absolute Zero: Reinforced Self-play Reasoning with Zero Data — https://proceedings.neurips.cc/paper_files/paper/2025/file/9837dc00ff67d176373268ed48042d49-Paper-Conference.pdf
- [cite:9] R-Zero: Self-Evolving Reasoning LLM from Zero Data — https://arxiv.org/html/2508.05004v3
- [cite:10] Self-Reference in LLMs: Introspection Threshold for RSI — https://arxiv.org/pdf/2607.04277v1
- [cite:11] Metan: Recursive Self-Improvement through Emergent Depth — https://arxiv.org/html/2608.24735v1
- [cite:12] Polaris: A Gödel Agent Framework for Small LMs — https://arxiv.org/html/2603.23129v4
- [cite:13] From 0-to-1 to 1-to-N: MetaAI Recursive Self-Design — https://arxiv.org/html/2606.09663v1
- [cite:14] RSIAI0 / DarwinGodel-Machine-ASI — https://zenodo.org/records/15644670
- [cite:15] Harness Engineering for Self-Improvement (Lilian Weng) — https://lilianweng.github.io/posts/2026-07-04-harness/
- [cite:16] 当 AI 开始造 AI：递归自我改进的 2026 年 — https://m.36kr.com/p/4019262571663237
- [cite:17] evoseal (PyPI) — https://pypi.org/project/evoseal/
- [cite:18] I.J. Good, Speculations concerning the first ultraintelligent machine — 经 [cite:5] 引用
- [cite:19] Universal Transformers — https://arxiv.org/pdf/1807.03819.pdf
- [cite:20] Universal Transformers (DeepMind) — https://www.deepmind.com/publications/universal-transformers
- [cite:21] Looped Transformers are Better at Learning Learning Algorithms — https://arxiv.org/pdf/2311.12424v3.pdf
- [cite:22] RecurrentGPT: Expressive Depth through Recurrent Modulation — https://arxiv.org/pdf/2608.15062v2
- [cite:23] Looped Transformers are Better at Learning Learning Algorithms — https://arxiv.org/abs/2311.12424
- [cite:24] Scaling up Test-Time Compute with Latent Reasoning (Huginn) — https://arxiv.org/pdf/2502.05171
- [cite:25] Ouro: Scaling Latent Reasoning via Looped Language Models — https://arxiv.org/html/2510.25741
- [cite:26] Ouro 项目主页 — https://ouro-llm.github.io/
- [cite:27] Mixture-of-Recursions (MoR) — https://arxiv.org/html/2507.10524
- [cite:28] MoR (NeurIPS 2025) — https://papers.nips.cc/paper_files/paper/2025/file/8b08bbf8b420faa6eeb4020720582ec7-Paper-Conference.pdf
- [cite:29] MoR-ViT — https://arxiv.org/html/2507.21761v1/
- [cite:30] On Expressive Power of Looped Transformers — https://arxiv.org/pdf/2410.01405v7.pdf
- [cite:31] Hyperloop Transformers — https://arxiv.org/html/2604.21254v2
- [cite:32] When Does Recurrence Become an Algorithm? — https://arxiv.org/html/2607.20594v1
- [cite:33] Looped Transformers under the Jacobian Lens — https://arxiv.org/html/2609.01924v1
- [cite:34] Depth through Recurrence: Looped Transformers for Flow-Matching TTS — https://arxiv.org/html/2609.29768v1
- [cite:35] Encode, Think, Decode — https://arxiv.org/pdf/2510.07358
- [cite:36] Improving Test-Time Scaling with Adaptive Looped Transformers — https://fuvty.github.io/thinking_yard_project_page/projects/tah2/static/pdf/tah2-paper.pdf
- [cite:37] DeepLoop: Depth Scaling for Looped Transformers — https://hub-assets-cache.baai.ac.cn/paper/d61de1ab-5ccf-4144-894a-d710e249c988
- [cite:38] Raschka: Looped Transformer 架构画廊 — https://sebastianraschka.com/llm-architecture-gallery/looped-depth-sharing/
- [cite:39] Raschka on looped transformers (cere-bro) — https://bayesiansapien.github.io/cere-bro/llms-foundation-models/2026-09-10-raschka-looped-transformers-recurrent-depth/
- [cite:40] Huginn-0125 (HuggingFace) — https://huggingface.co/tomg-group-umd/huginn-0125
- [cite:41] Beyond Memorization: Extending Reasoning Depth with Recurrence — https://openreview.net/attachment?id=2pySNr77HZ&name=pdf
- [cite:42] Universal Reasoning Model — https://arxiv.org/html/2512.14693
- [cite:43] What Is a Looped Transformer? (Tosea.ai) — https://tosea.ai/blog/looped-transformer-recurrent-depth-astra-guide
