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
