下面我会按你的要求做三件事（一次性完成，不留坑）：

1. **把你原 proposal（v0）的方法与实验，扩展成“可直接投 NeurIPS 的 Method + Experiments”级别**（包含算法细节、公式、伪代码、训练与推理全部超参、预算口径、评测协议、统计检验、鲁棒性、消融、失败案例计划——尽量做到“审稿人看完没有可抓的‘没说清楚’”）。原稿来自你上传的 PDF。
2. **进行 10 次迭代**：每一轮都以“NeurIPS reviewer”视角指出方法/实验设计的硬伤→给出必须修改项→形成新版本（v1…v9）。
3. **最后给出 v9 的完整 Method 与 Experiments 两节“全文内容”**（你可以直接搬进论文/LaTeX）。

在对比与实验设计中，我会显式对齐并准备 head-to-head 的 2025–2026 强相关路线：Triage（层级视觉预算）、FastAV（AV-LLM token pruning）、CaCoVID（RL 压缩）、Nüwa（pruning 破坏空间参考系问题）、FlexSelect、X2])

**硬伤**

* Stage‑1 只用 audio eventness，面对“问题依赖证据”时容易把预算花在“响但不相关”的片段。
* Experiments 缺少明确 QA 任务与 query-aware baseline 的 head-to-head。

**必须改**

* Method：加入 **Q‑L2L（query-aware reweight）**：用 ASR 或 audio-text embedding 将 query relevance 注入锚点权重。
* Experiments：加入 Video‑MME（含 long）作为主 QA 基准，并加入 FlexSelect、Triage 作为对照。([arXiv][3])

**v1 增量**

* 增加 “query relevance weight r_i” 与其成本记账（ASR/encoder 计入 cheap budget）。

---

### Round 2 (v1 → v2)：最大问题变成“音频不可靠时怎么办”，需要门控而不是固定 α

**Reviewer summary**
你有 uniform fallback（α），但 α 是固定超参会被认为是“调参旋钮”；需要一个可解释/可复现的可靠性驱动机制（尤其要覆盖静音、噪声、不同步）。

**必须改**

* Method：α 从常数改为 **α = f(reliability)ynamics anchors**（帧差/压缩域 motion vector），与音频锚点做融合，仍不调用重视觉编码器。
* Experiments：加入 “audio-only vs visual-only vs fusion+gating” 消融与分桶分析（耦合强/弱）。

**v3 增量**

* Stage‑1 变为双源锚点 + 门控融合。

---

### Round 4 (v3 → v4)：最大问题是 Stage‑2 “像启发式”，需要标准优化表述与可复现求解器

**Reviewer summary**
如果 Stage‑2 仍是比值贪心，会被认为 engineering；但 2026 Triage 把 budgeting 写成资源分配问题并给两阶段 token-level budgeting（core/context）。([arXiv][1])

**必须改**

* Method：把 Stage‑2 写成 **Multiple‑Choice Knapsack**（每窗口从离散配置集选一个），默认用拉格朗日松弛求解（确定性、可复现），并给复杂度。
* Experiments：加 “solver ablation”（greedy vs Lagrangian vs DP 近似），证明不是 solver trick。

**v4 增量**

* Stage‑2 形式化 + solver 伪代码。

---

###  增量**

* “core tokens + context tokens” 约束进入 Stage‑2。

---

### Round 6 (v5 → v6)：最大问题是“compute 口径不够硬”，必须拆 tokenization/prefill/decode

**Reviewer summary**
2025–2026 效率论文审稿越来越严：只报 Tok 不够，要拆真实 latency，并区分 tokenization、视觉编码器、LLM prefill 与 decode。

**必须改**

* Method：Compute Accounting Protocol 升级：明确记录（i）视觉编码器调用比例；（ii）prefill/decode；（iii）所有 cheap 模块计入。
* Experiments：报告同一硬件/精度下 latency，并给脚本与统计（median/p95）。

**v6 增量**

* 增加 compute breakdown 表格规范。

---

### Round 7 (v6 → v7)：最大问题是“缺少与 2026 强对手的正面对比”

**Reviewer summary**
如果你不 head-to-head：Triage（training-free budgeting）、FastAV（AV‑LLM token pruning）、CaCoVID（RL 贡献度压缩），会被认为不够强。([arXiv][1])

**必须改**

* Experiments：温度缩放/校准流程（仅用 val）。
* Experiments：按 “AV 耦合强/弱、静音比例、高 FPS 密集”分桶画 Pareto，外加校准指标（ECE）。

**v8 增量**

* 引入 AVCI（耦合指数）与 q 校准。

---

### Round 9 (v8 → v9)：最大问题是“实验还不够像 Oral”，需要 6 张“生死图”与统计检验预注册

**Reviewer summary**
Oral 往往需要“曲线统治力 + 机制可证伪 + 边界透明”。你需要把图表与检验写成“交付物”，并预注册显著性与置信区间。

**必须改**

* Experiments：固定预算点集合、固定 seeds、bootstrap CI、dominance 判据、以及删除证据（delete-and-predict）faithfulness 指标。
* Method：把 evidence alignment 指标与 proxy（无标注 QA）写成正式定义。

**v9 增量**

* 完整的 “MDE++ + Evidence/Faithfulness + Robustness + Compute”评估套件。

---

### Round 10：v9 评审结论（就 Method/Experiments 的“可 Oral 性”）

如果结果能做到ïMethod** 与 **完整 Experiments** 两节文本（不省略内容）。它在结构上兼容 NeurIPS 写法（3 Method / 4 Experiments），也能直接拆成 appendix。

---

## 3. Method

### 3.1 Problem setup and notation

给定一段长视频，其视觉帧序列记为
[
V={I_t}_{t=1}^{T_f},
]
同步音频波形记为 (A)。对视频问答/推理任务，还给定自然语言 query (q)（对分类/定位任务可令 (q=\varnothing)）。

我们关注“长视频理解”的核心瓶颈：视觉冗余导致昂贵视觉编码器与后端模型的计算量近似随时长线性增长，而有效证据往往时间稀疏（但我们也显式覆盖 dense 信息边界任务 DIVE）。 ([OpenReview][4])

我们的目标是在严格预算下最大化下游任务性能：
[
\max_{\pi} \ \mathbb{E}[ \text{Acc}(\pi(V,A,q))]\quad \tCost}(\pi)\le B,
]
其中策略 (\pi) 输出一个对全视频的“视觉 token 分配计划”（何处高采样/高分辨率/少压缩，何处低采样/低分辨率/强压缩）ã®义主预算轴为**输入视觉编码器的 patch token 数**（与原 proposal 一致）：
给定离散配置
[
c=(\text{fps},\text{res},r_{\text{keep}})\in\mathcal{C},
]
对于时长 (L) 的视频段，其视觉 token 成本为
[
\text{Tok}(c;L) = (\text{fps}\cdot L)\cdos}}{p}\Big)^2\cdot r_{\text{keep}},
]
其中 (p) 是 patch size（固定，如 14 或 16），(r_{\text{keep}}\in(0,1]) 表示在窗口内若启用 token pruning/merging 的保留比例；若不启用压缩则 (r_{\text{keep}}=1)。

总预算以 token 记为：
[
B = B_{\text{vis}} + B_{\text{cheap}},
]
其中 (B_{\text{vis}}) 由上式统计；(B_{\text{cheap}}) 将用真实 FLOPs/latency 与（若适用）等效 token 进行同时。

> 备注（与现有方法层级关系）：
>
> * Triage 把视频推理表述为层级视觉预算分配（frame-level + token-level）。([arXiv][1])
> * FastAV 在 AV‑LLM 内进行 token pruning 降 FLOPs。([arXiv][6])CaCoVID 用 RL 学 token 贡献度进行压缩。([arXiv][7])
>   我们的æ* **Stage‑2（Allocation）**：在严格视觉 token 预算 (B_{\text{vis}}) 下，把预算拆成背景兜底与锚点强化两部分，并把锚点侧的配置选择写成一个可复现的多选背包（Multiple‑Choice Knapsack）优化。

这保持了你原稿 v0 的核心（锚点覆盖 + 预算分配 + MDE 协议），并加入 2026 审稿最在意的三点：**query-aware、reliability gating、标准优化表述**。

---

### 3.4 Stage‑1: Multimodal anchor proposal with reliability gating

Stage‑1 输出一组时间窗口锚点：
[
W_i=[t_i-\Delta_i,, t_i+\Delta_i],\quad i=1,\dots,K,
]
以及窗口权重 (w_i) 与可靠性 (q_i)。

#### 3.4.1 Audio eventness stream

**输入与特征**
将音频 (A) 重采样至 16kHz，提取 log-mel 频谱（例如 64 或 80 mel bins），时间步长固定为 (\delta=0.2s)（预注册，不随数据集调整）。型**
使用轻量音频编码器 (f_a) 输出每个时间步的事件概率：
[
s_a(t)=\sigma(f_a(\text{mel}(A_{t:t+\delta})))\in[0,1],
 Triage 的 frame-level budgeting，我们显式加入廉价视觉动态流 (s_v(t([arXiv][1])

实现二选一（默认用 (a)，(b) 作为消融）：

(a) **Frame difference energy**
对低分辨率缩略帧（如 64×64）计算
[
s_v(t)=| \phi(I_{t})-\phi(I_{t-\delta})|_1,
]
其中 (\phi) 是固定的下采样与灰度化。该开销极低。

(b) **Compressed-domain motion vectors (MV)**
若视频源可取到压缩域 MV，则用 MV 幅值作为动态强度 proxy。该方案仍不需要重视觉编码器。

两者均将 (s_v(t)) 归一化到 [0,1]（min-max per-video 或全局 robust normalization）。

#### 3.4.3 Query-aware reweight (Q‑L2L, for QA/reasoning)

对于 video QA / reasoning，Stage‑1 需要考虑“响≠相关”。我们引入 query-aware 权重 (r_q(t))。

默认方案（两条都给，主结果选其一并固定）：

* **ASR‑TFIDF relevance（最便宜）**：对音频做 ASR 得到片段文本 (x(t))，计算 (r_q(t)=\text{BM25}(q,x(t))) 并归一化。
* **Audio/Text em学习版本（消融）。

**可复现确定性版本（默认）**
设四个可观测量：

1. **Silence ratio** (u_1(t))：窗口内低能量帧比例（静音越多越不可靠）。
2. **Estimated SNR** (u_2(t))：简易噪声估计得到 SNR（越低越不可靠）。
3. **Eventness confidence** (u_3(t))：对校准后概率 (\tilde s_a(t)-0.5|) 或熵的负值（越接近 0.5 越不可靠）。
4. **AV coupling proxy** (u_4(t))：局部相关 (\text{corr}(\tilde s_a, s_v))（音画峰值是否一致）。

将它们归一化到 [0,1] 后组合：
[
q(t)=\text{clip}\Big(\sum_{j=1}^4 \lambda_j u_j(t),,0,,1\Big),
]
其中 (\lambda_j) 预注册为 ([0.35,0.25,0.25,0.15]) 并跨数据集固定。仅允许在附录做敏感性分析，主结果不调。

**学习版（消融）**
用一个 2-layer MLP (g_\theta(\cdot)) 输入 ([u_1,u_2,u_3,u_4]) 输出 (\hat q)。训练目标是预测“锚点覆盖是否命中证据”（有边界数据上可监督）。该版本只作为 ablation，不作为主结论ï的候选集合 (\mathcal{P})。
2. Top‑K：按 (\hat s) 取分数最高的 (K_{\max}) 个候选（预注册 (K_{\max}=32)）。
3. 1D‑NMS：对候选中心 (t_i) 构造窗口 (W_i)，若 (\text{IoU}(W_i,W_j)>\eta)（(\eta=0.5) 固定）则保留高分者。
4. Window half-width：

   * 基础宽度 (\Delta_{\min}=2.0s)（固定）；
   * 置信度越低窗口越宽以提高ta_i = \Delta_{\min}\cdot (1+\kappa(1-q_i)),
     ]
     其中 (q_i) 是窗口内 (q(t)) 平均，(\kappa=1) 固定。
5. A/V 不同步鲁棒：再做对称扩张 (\Delta_i\leftarrow \Delta_i+\Delta_{\text{async}})，(\Delta_{\text{async}}=1.0s)（固定），并在鲁棒性实验显式扫偏移（§4.7）。

**窗口权重 (w_i)**
[
w_i=\underbrace{\max_{t\in W_i}\hat s(t)}*{\text{peak score}}\cdot \underbrace{\bar q_i}*{\text{avg reliability}},
]
其中 (\bar q_i=\frac{1}{|W_i|}\int_{t\in W_i} q(t)dt)。

---

### 3.5 Stage‑2: Reliability-aware budget allocation with a multiple-choice knapsack solver

Stage‑2 在固定视觉 token 预算 (B_{\text{vis}}) 下输出对全视频的离散配置分配 ({c_u})。

#### 3.5.1 Budget split: background vs anchors (α-mixture)

我们将预算拆成背景兜底与锚点强化两部分：
[
B_{\text{back}}=\alpha B_{\text{vis}},\quad B_{\text{anchor}}=(1-\alpha)B_{\text{vis}}.
]

关键：(\alpha) **不是自由调参**，而是由可靠性驱动：
[
\alpha = \text{clip}(\alpha_{\min} + (1-\bar q)\cdot (\alpha_{\max}-\alpha_{\min}),,\alpha_{\min},,\alpha_{\max}),
]
其中

* (\bar q) 为全视频 (q(t)) 平均；
* (\alpha_{\min}=0.10)、(\alpha_{\max}=0.60) 固定跨数据集；
* 解释：廉价索引越不可信，越接近 uniform 背景采样，保证不“归零”。

#### 3.5.2 Background schedule (uniform fallback)

背景配置 (c_{\text{back}}) 固定为最便宜的配置之一（预注册）：
[
c_{\text{back}}=(\text{fps}=0.5,, \text{res}=224,, r_{\text{keep}}=1).
]
即每 2 秒取一帧、低分辨率、不开启窗口内压缩（避免再引入 pruning 风险）。如è被审稿人接受（方法不是拍脑袋），也便于与 Triage 的 budgeting 叙事正面对比。([arXiv][1])

**效用函数 (U(c))（预注册，避免调参）**
我们不直接用下游准确率作为效用（那会引入学习/调参），而用单调先验：
[
U(c)=u_{\text{fps}}(\text{fps}) + u_{\text{res}}(\text{res}) + u_{\text{keep}}(r_{\text{keep}}),
]
其中预注册：

* (u_{\text{fps}}(1)=0,\ u_{\text{fps}}(2)=1,\ u_{\text{fps}}(4)=2)
* (u_{\text{res}}(224)=0,\ u_{\text{res}}(336)=1)
* (u_{\text{keep}}(1)=0,\ u_{\text{keep}}(0.5)=0.5)（仅在启用时）

这保证 (U(c)) 与复杂度单调，且跨数据集固定。我们只在附录做“替代效用”敏感性分析（如乘法形式）。

#### 3.5.4 Deterministic solver: Lagrangian relaxation (default)

我们用拉格朗日松弛将约束优化转为无约束：
[
\max_{{c_i}}\sum_i w_iU(c_i) - \lambda\sum_i \text{Tok}(c_i;|W_i|).
]
对固定 (\lambda)，每个窗口独立选择：
[
c_i(\lambda)=\arg\max_{c\in\mathcal{C}} \Context-preserving constraint for token pruning/merging (optional)

鉴于 2026 Nüwa 指出 pruning 可能破坏全局空间参考框架并导致 grounding 退化，我们对窗口内的 token 压缩引入“空间锚保留”约束。([arXiv][5])

当 (r_{\text{keep}}<1) 时，我们将保留 token 分成两部分：

* **Spatial anchor tokens**：从规则网格（如 4×4）中固定采样位置 token（与内容无关），保证空间参考存在。
* **Content tokens**：剩余预算按注意力/相似度选择（可用现有 pruning/merging）。

形式化：设每帧 token 集合为 (\mathcal{T})，保留集合 (\mathcal{T}'=\mathcal{T}*{\text{anchor}}\cup \mathcal{T}*{\text{content}})，并强制
[
|\mathcal{T}_{\text{anchor}}|\ge \rho |\mathcal{T}'|,
]
其中 (\rho=0.25) 预注册。

主结果默认 **不启用** (r_{\text{keep}}<1)，只在“与压缩方法可组合”实验中启用（避免审稿人把主贡献误解为 pruning）。

---

### 3.6 End-to-end inference pipeline

给定视频断（并降低审稿争议），我们沿用并扩展你原稿的 Oracle→Predicted MDE 协议。

#### 3.7.1 Coverage@τ (with boundary annotations)

给定证据区间集合 (\mathcal{E})，定义锚点覆盖率：
[
\text{Cov}@\tau = \frac{1}{|\mathcal{E}|}\sum_{e\in \mathcal{E}}\mathbf{1}\Big(\max_i \text{IoU}(W_i,e)\ge\tau\Big),
]
IoU 为 1D 时间区间 IoU。该指标用于解释：Oracle 成功/Predicted 失败是否由 Stage‑1 覆盖不足导致。

#### 3.7.2 Faithfulness proxy for QA (delete-and-predict)

对无显式证据标注的 QA，我们定义删除证据的 faithful proxy：

* 令 (\mathcal{S}) 为模型输入中由 Listen‑then‑Look++ 选择的 top‑M 锚点窗口集合（按 (w_i)）。
* 构造两种输入：
  (i) 原输入 (X)； token 的输入 (X\setminus \mathcal{S})（用背景采样替换或置空占位以保持长度）。
* 定义 faithfulness drop：
  [
  \Delta_{\text{del}} = \text{Acc}(X)-\text{Acc}(X\setminus \mathcal{S}).
  ]
  我们还报告对 M 的曲线 AUC制有效）；Predicted Anchors 接近 Oracle（说明 Stage‑1 可落地）。

H3（可靠性与边界透明）：在弱耦合/静音/噪声/密集信息（DIVE）条件下，α(q) 能自动回退，不出现灾难性性能崩溃。([OpenReview][4])

H4（与 2026 强对手的关系）：与 Triage/FlexSelect/XComp/FastAV/CaCoVID 等相比，我们要么在同预算下更强，要么给出清晰边界与可组合收益。([arXiv][1])

---

### 4.2 Benchmarks

我们采用“三层闭环 + 边界任务”的组合（每类都给清晰指标）：

1. **长视频 QA / reasoning（主战场）**

* Video‑MME（重点使用 long duration 子集/划分）。([arXiv][3])
* LVBench（极长视频理解）。([CVF Open Access][8])
* EgoSchema（长时程多选 QA）。([arXiv][9])作为超长影片级压力测试，若算力允许）。([arXiv][10])

2. **有时间边界标注的音画任务（用于 MDE 与 Coverage@τ）**

* 按你原 proposal：AVE、EPIC‑SOUNDS（用于 Oracle→Predicted 与 Cov@τ）L 系列中支持视频输入的版本（FlexSelect 声称可 plug‑and‑play 支持多种 VideoLLM，包括 LLaVA‑Video、InternVL、Qwen‑VL）。([NeurIPS][2])

我们固定：

* 提示词模板（prompt）
* 解码策略（greedy 或 temperature=0，top‑p=1）
* 最大生成长度（如 64 tokens）
  确保差异来自输入上下文选择，而非解码随机性。

> 若某些 baseline 仅在特定 backbone 有官方实现，我们在该 backbone 上做 head-to-head，并代实现”（见 §4.4）。

#### 4.3.2 Audio / cheap-visual modules

* 音频输入：16kHz，log‑mel（64 bins），(\delta=0.2s)。
* Audio eventness 模型：轻量 CNN/Transformer（参数规模与 FLOPs 固定），训练细节见 §4.8。
* Cheap‑visual：默认 frame-diff（64×64），MV 作为消融。
* ASR：默认使用同一离线 ASR（若启用），并将其时间/显存计入 (C_{\text{cheap}})。

#### 4.3.3 Configuration set and budgets (pre-registered)

配置集 (\mathcal{C}) 固定为（同 §3.5form**：在预算约束下均匀采样（可通过调整 fps/res 达到目标 Tok）。
* **Random Anchors**：随机选 K 个窗口做高配置，其余背景（用于反驳“随便挑窗口都涨”）。
* **Audio Energy Heuristic**：用音频能量峰值代替学习到的 eventness（用于验证 Stage‑1 学习必要性）。

#### 4.4.2 Cheap-cue selection baselines

* **Cheap‑Visual Anchors only**：只用 frame-diff/MV 生成锚点，不用音频（对标“视觉动态先验”）。
* **Audio Anchors only**：只用音频 eventness（你的原始主线）。
* **Fusion w/o gating**：融合但不使用 q（证明 reliability gating 价值）。

#### 4.4.3 Token selection / compression baselines (2024–2026 strong)

* **Triage**（层级 budgeting，training-free）。([arXiv][1])
* **FlexSelect**（reference-layer attention 的 token selection，plug‑and‑play）。([NeurIPS][2])
* **XComp**（极限压缩路线，一 token/帧）。([NeurIPS][12])
* **Video Token Merging (VTM)**（NeurIPS 2024，token merging for long video）。([NeurIPS Proceedings][13])
* **PruneVid**（训练无关 token pruning）。([arXiv][14])
* **CaCoVID**（RL 贡献度压缩；若无法完整复现，则使用其公开模型或报告“作者配置”结果并在附录声明差异）。([arXiv][7])

#### 4.4.4 AV‑LLM pruning baseline

* **FastAV**：当 backbone 属于 AV‑LLM 且可复现时，比较其 F验（§4.6）。([arXiv][6])

---

### 4.5 Main results: Pareto curves and dominance tests

#### 4.5.1 Metrics

* QA：Accuracy（多选）或 Exact Match / F1（如适用）。
* 定位/检测：mAP 或 IoU@threshold（按数据集标准）。
* Dense（DIVE）：官方/论文定义的 QA accuracy + throughput 指标。([OpenReview][4])

成本侧：

* Visual tokens（主轴）
* GFLOPs（拆到 cheap/vision/prefill/decode）
* Latency：在统一硬件上测量 median 与 p95（见 §4.9）

#### 4.5.2 Pareto evaluation

对每个预算点 (B\in\mathcal{B})，每个方法都必须产生满足 Tok≤B çCombination experiments (orthogonality)

为回应“你只是另一种压缩吗？”我们做组合实验：

* Listen‑then‑Look++（只做上游预算）
* FastAV / PruneVid / VTM / FlexSelect（只做 token 压缩/选择）
* Listen‑then‑Look++ +（上述任一）

验证二者是否可叠加：若可叠加，说明我们贡献位于更上游的杠杆点（减少视觉编码调用与输入长度），而不是替代关系。([arXiv][6])

---

### 4.7 Evidence alignment and faithfulness

#### 4.7.1 Coverage@τ on boundary datasets

报告 (\tau\in{0.3,0.5}) 的 Cov@τ，并绘制 Cov@τ vs Acc 的散点与相关系数，用于机制诊断：Oracle 成功但 Pred 失败 → Stage‑1 覆盖不足。

#### 4.7.2 Delete-and-predict on QA

在 Video‑MME/LVBench/EgoSchema 上，对 top‑M 锚点窗口进行删除，报告 (\Delta_{\text{del}}) 与 AUC，并与 Uniform/Random Anchors 对比，检验选择的证据是否“真有用”。

---

### 4.8 Training details (all hyperparameters fixed)

证据”（IoU≥0.3）

#### 4.8.3 No test-time tuning policy

所有阈值（例如 top‑K、NMS、温度、α_min/α_max、配置集）均在 paper 中预注册并固定；任何“按数据集微调”必须进入附录并标为 non-main。

---

### 4.9 Robustness protocols (pre-registered stress tests)

我们复用你原 proposal 的三类扰动，并把它们系统化为矩阵测试：

* **A/V offset**：(\Delta t\in{-2,-1,-0.5,0.5,1,2}) 秒
* **Noise**：SNR ∈ {20,10,0} dB
* **Silence**：静音比例 ∈ {0,25%,50%,75%}

每个扰动设置下报告：

* Acc–Tok 曲线（至少 3 个预算点）
* 自动得到的 (\alpha) 分布（证明确实回退）
* 与 Uniform 的差距（是否回到接近 uniform，而非崩溃）

---

### 4.10 Dense boundary evaluation on DIVE

DIVE 定义 dense video understanding：需要高 FPS、帧级推理，强调 token throughput 与 temporal alignment。([OpenReview][4])

在 DIVE 上，我们重点检查两点：

1. 在高 FPS 需求下，Listen‑thenâoken pruning 时：是否加入 context-preserving constraint（回应 Nüwa 风险）([arXiv][5])

---

### 4.12 Runtime measurement and reporting

硬件：统一 GPU（例如 A100 80GB 或 H100），固定 batch=1 测推理延迟；每个样本重复 5 次取 median，汇总全数据集样本的 median 与 p95。

必须报告并拆分：

* cheap front-end latency
* vision encoder latency（以及实际被调用的时间比例/帧数）
* LLM prefill latency
* LLM decode latency（固定输出长度上限）

这确保对比 Triage/FastAV/FlexSelect/XComp 等方法时没有“把成本藏在别处”的空间。([arXiv][1])

---

## 你接下来如何使用这份内容

* 如果你要写论文：上面需要把“我们将/我们会”改成“我们”并填上实验结果、图表与表格。
* 如果你要更像 NeurIPS Oral：我建议你把 **§4.5 主结果 + §4.6 MDE++ + §4.9 鲁棒 + §4.10 DIVE 边界** 做成正文核心，其他放 appendix。

---

如果你希望我进一步把 **最终21075 "https://arxiv.org/abs/2405.21075"
[4]: https://openreview.net/forum?id=njpwWTNGj5 "https://openreview.net/forum?id=njpwWTNGj5"
[5]: https://arxiv.org/abs/2602.02951?utm_source=chatgpt.com "Nüwa: Mending the Spatial Integrity Torn by VLM Token Pruning"
[6]: https://www.arxiv.org/abs/2601.13143?utm_source=chatgpt.com "FastAV: Efficient Token Pruning for Audio-Visual Large ..."
[7]: https://arxiv.org/abs/2602.01649?utm_source=chatgpt.com "Contribution-aware Token Compression for Efficient Video Understanding via Reinforcement Learning"
[8]: https://openaccess.thecvf.com/content/ICCV2025/papers/Wang_LVBench_An_Extreme_Long_Video_Understanding_Benchmark_ICCV_2025_paper.pdf "https://openaccess.thecvf.com/content/ICCV2025/papers/Wang_LVBench_An_Extreme_Long_Video_Understanding_Benchmark_ICCV_2025_paper.pdf"
[9]: https://arxiv.org/abs/2308.09126 "https://arxiv.org/abs/2308.09126"
[10]: https://arxiv.org/abs/2406.19875 "https://arxiv.org/abs/2406.19875"
[11]: https://arxiv.org/abs/2601.06097 "https://arxiv.og/abs/2601.06097"
[12]: https://neurips.cc/virtual/2025/poster/117133 "https://neurips.cc/virtual/2025/poster/117133"
[13]: https://proceedings.neurips.cc/paper_files/paper/2024/file/194fa4536bf36f35a4505d20cd5dd6fc-Paper-Conference.pdf "https://proceedings.neurips.cc/paper_files/paper/2024/file/194fa4536bf36f35a4505d20cd5dd6fc-Paper-Conference.pdf"
[14]: https://arxiv.org/abs/2412.16117 "https://arxiv.org/abs/2412.16117"

# Listen-then-Look：Audio-Anchored Budgeted Visual Token Allocation for Efficient Long-Video Understanding

> 这份 `plan.md` 是 paper-ready 的主线草稿；工程执行以 `docs/plan.md`（P####/C####）与 `docs/experiment.md`（E####）为准。
> 冲 oral 的最小决定性 checklist：`docs/oral_checklist.md`。

---

## Contributions（两句封口）

1. **Listen-then-Look**：提出一种在固定计算预算下的两阶段策略：用低成本音频在时间轴上产生高召回锚点，再对离散视觉采样配置进行预算受限分配，使昂贵视觉计算集中于高信息窗口，从而在 **accuracy-per-token / accuracy-per-FLOP** 上形成稳定 Pareto 优势。
2. **MDE + Evidence/Degradation Protocol**：提出预注册的 Oracle→Predicted 最小决定性实验协议，并配套证据一致性（有标注/无标注 proxy）与鲁棒退化（偏移/噪声/静音）评估与可计算下界，使审稿人难以用“heuristic / cherry-pick / 不可复现 / 只在强绑定有效”一票否决。

---

## Abstract（paper-ready 风格）

长视频理解的核心瓶颈往往并非模型表达能力，而是时间轴上的视觉冗余导致推理成本随时长急剧增长；对许多任务而言，答案/事件证据在时间上是稀疏的。我们提出 Listen-then-Look：先以低成本音频在时间轴上生成高召回锚点，再在固定视觉 token 预算下对离散视觉采样配置进行受限分配，使昂贵视觉计算集中于高信息窗口；在多个预算点上形成稳定的 Acc–Tok / Acc–FLOPs Pareto 优势，并通过预注册的 Oracle→Predicted 最小决定性实验与鲁棒退化协议给出可复现证据。

---

## 1. Motivation：硬瓶颈与杠杆点

**瓶颈**：长视频任务的计算成本随着帧数与分辨率增长，而“有效证据”在时间轴上稀疏。若对全局均匀采样提升分辨率/帧率，会把大量预算浪费在无关片段。

**现有路线的局限**：大量方法集中在“视觉 token 内部压缩/合并”，仍需要先把整段视频过一遍视觉编码器（即使低密度），并且压缩策略往往与具体架构耦合、成本口径容易争议。

**杠杆点**：音频作为更廉价的模态，往往能在时间轴给出“发生了什么”的稀疏提示。我们用音频先把时间搜索空间缩小，再把昂贵视觉预算投向少量候选片段。

一句话动机：用廉价音频做时间索引，让昂贵视觉计算只发生在少量“可能有证据”的窗口里。

---

## 2. Problem Setting：严格封口的预算定义与策略输出

### 2.1 输入/输出

- 输入：视频帧序列 `V` 与同步音频 `A`（允许缺失音频：走兜底策略）。
- 输出：时间轴上的若干窗口 `W_i` 及对应的视觉采样配置 `c_i`，满足视觉预算约束。

### 2.2 预算口径（视觉 token）

定义配置 `c=(fps, res, r_keep)`，其中 `r_keep∈(0,1]` 为（可选）token 保留比例（不用内部 token reduction 则固定为 1）。

对任一时长为 `L_u` 的段 `u`，配置 `c` 的 token 成本：

```
Tok(c; L_u) = (fps · L_u) · (res / p)^2 · r_keep
```

其中 `p` 为 patch size（固定）。全视频预算约束：

```
∑_{u∈U} Tok(c_u; L_u) ≤ B_vis
```

---

## 3. Method：Listen-then-Look

### 3.1 Stage-1：音频时间锚点（高召回证据覆盖）

音频模块以固定步长 `δ`（预注册，例如 0.2s）输出 eventness 分数序列 `s(t)∈[0,1]`。

**高召回的形式化度量：Coverage@τ**

给定证据区间集合 `E`，定义：

```
Cov@τ = (1/|E|) · ∑_{e∈E} 1[max_i IoU(W_i, e) ≥ τ]
```

其中 `W_i=[t_i-Δ_i, t_i+Δ_i]` 为锚点窗口，IoU 为 1D 时间区间交并比。

锚点生成算法（可复现）：

1. 在 `s(t)` 上取局部极大值候选；
2. 按 `s(t)` 排序取 top-`K`；
3. 做 1D NMS 去除过近锚点；
4. 以固定 `Δ` 扩窗为窗口集合 `W_i`（写死，不允许每数据集调参）。

### 3.2 Stage-2：预算受限的视觉采样分配

将视觉预算拆为两部分：

```
B_anchor = (1-α) · B_vis
```

- 兜底均匀采样：对非锚点区域使用固定低成本配置 `c_back`，保证弱耦合/无声事件下界；
- 锚点强化采样：对锚点窗口在 `B_anchor` 下选择更高 `fps/res`（或更低 merging）的配置。

**Budget Allocator（默认可复现版本）**

输入：锚点窗口 `{W_i}`，权重 `w_i=g(s_i)`，配置集 `C`，预算 `B_anchor`。

```
c_i = argmax_{c∈C}  (w_i · h(c)) / Tok(c; |W_i|)
  s.t.  ∑_i Tok(c_i; |W_i|) ≤ B_anchor
```

- `h(c)` 为与配置复杂度单调的先验效用（预注册，避免调参）；
- 若预算不足，按 `w_i` 从低到高降级/丢弃窗口，直到满足约束；
- 可选：学习 `h` 或 `ΔÛ`，但主结果以默认 allocator 为基线以保持透明可复现。

声画不同步处理（写死）：

- 窗口扩张：`Δ_i` 至少覆盖潜在偏移范围；
- 训练时加入 jitter；并在评估显式报告偏移曲线。

与内部 token reduction 的关系（正交杠杆）：

- 主结果默认不启用额外 token merging（`r_keep=1`），证明上游索引本身成立；
- 另做可选叠加实验：窗口内启用 `r_keep<1`，展示可组合性。

---

## 4. MDE：预注册最小决定性实验协议

### 4.1 MDE-1 Oracle Anchors（机制上限）

- 用 GT 时间边界/证据区间构造锚点窗口；
- 目的：只验证 Stage-2 的预算分配机制是否带来 Pareto 改善；
- 预注册预算点：`B∈{B1,…,Bm}`；
- 成功判据：在至少一半预算点上，Acc 相对 best baseline 提升 ≥ Δ 且 95% CI 不跨 0；并报告 Pareto dominance。

### 4.2 MDE-2 Predicted Anchors（落地可部署）

- 用 Stage-1 预测锚点重复 MDE-1；
- 成功判据：Predicted 与 Oracle 在 `B` 上平均差距 ≤ ε，且 Predicted 仍显著优于 best baseline。

### 4.3 失败可诊断（机制分解）

同时报告以下对照：

- Random Anchors（关键对照）
- Cheap-Visual Anchors（帧差/光流/压缩域 motion vector/低成本视觉 eventness）
- Text/ASR Anchors（若可得，成本计入）
- Visual-only Token Reduction（代表性 merging/pruning）
- Multimodal Token Pruning（alignment-guided pruning 类）
- Audio Energy Heuristic
- Oracle Upper Bound（非 baseline，用于上限）

所有方法在同一预算点与同一 backbone/训练设置下比较，禁止“训练更久/更大模型”偷优势。

---

## 5. Metrics：证据一致性与鲁棒退化

主结果：Pareto 曲线（Acc/mAP/QA acc vs Tok；附 GFLOPs/Latency）

证据一致性（Evidence Alignment）：

- 有标注：`Cov@τ`；
- QA 无标注：delete-and-predict proxy（固定分段、替换扰动、top-M 证据集、抽样子集离线分析）。

鲁棒退化：

- 声画偏移 `Δt`、噪声 SNR、静音比例 × α 曲线/热力图；
- 预注册范围：偏移 `Δt∈{-1.0,-0.5,0,+0.5,+1.0}` 秒；噪声 `SNR∈{20,10,0}dB`；静音比例 `∈{0,25%,50%,75%}`。

---

## 6. Expected Key Figures（审稿人一眼知道你要交付什么）

- Fig.1：Listen-then-Look pipeline 总览；
- Fig.2：Oracle vs Predicted vs 强 baselines 的 Acc–Tok Pareto（含 CI）；
- Fig.3：Evidence Alignment（有标注 Cov@τ + QA proxy Cov@τ）+ 与 Acc 的相关散点；
- Fig.4：鲁棒退化（偏移/噪声/静音 × α）；
- Table 1：固定预算点上 Acc/Tok/GFLOPs/Latency（LLM 部分拆分）。

---

## 7. Risks & Mitigations（写进 proposal 的兜底）

1. 无声关键事件：通过 α 兜底保证回到 uniform；并在静音扰动中实证。
2. 噪声误报浪费预算：top-K + NMS + 校准，且用 Random Anchors 对照排除“只要高 res 窗口就涨”。
3. 声画不同步：窗口扩张 + jitter + 偏移曲线显式报告。
4. “拼贴/增量有限”质疑：Oracle 上限证明 Stage-2 机制；Predicted 证明 Stage-1 可部署；协议透明。

---

## 最短路径执行顺序（直接可做）

1. AVE 上先跑 MDE-1 Oracle：把 Fig.2 的 Oracle 曲线做出来（机制立住）。
2. 再跑 Predicted Anchors：做 Oracle–Pred gap + coverage→Acc 分析。
3. 加 Random Anchors / Cheap-Visual Anchors：把“任何窗口都行？”的质疑一刀切掉。
4. EPIC-SOUNDS 先做强耦合子集：确保长时场景也有漂亮 Pareto。
5. 选一个长视频 QA（EgoSchema/IntentQA 二选一）做终局验证 + QA evidence proxy（抽样离线）。

---

## 工程落地映射（Repo 内文件）

- 代码与可执行验证：`docs/plan.md`（P####/C####）与 `docs/mohu.md`（M####）
- 实验账本与命令：`docs/experiment.md`（E####）
- 运行产物：`runs/`（日志、metrics、figures、cache）
