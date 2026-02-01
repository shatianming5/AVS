下面给你一份**完整、可落地、最小不可分（MIU：Minimum Indivisible Units）**的 Demo Proposal。目标是你说的那条验证路径：

> 上传 3 篇 PDF → 自动生成「导师 intro 蓝图 + 3 个可直接用的段落模板 + figure storyboard」→ **每条结论都能点回 PDF 页码证据**。

我会按**第一性原理 → 系统级 I/O → 数据模型 → 模块拆解（每个模块的输入/输出/功能）→ 端到端流程 → 实现计划分阶段逐项清单**来写，不省略关键内容。

---

## 0) 第一性原理：我们到底在解决什么

### 0.1 用户真实目标（不可再分）

用户（freshman/组内成员）想要的不是“写作知识”，而是：

1. **生成物对齐**：我写出来的 introduction / 实验叙事 / 图表安排，**像导师/像组内代表作**
2. **可解释与可追责**：为什么说“导师喜欢这种写法”？要能指出**证据在 PDF 的哪一页哪段**
3. **可执行**：不是建议，是可以直接复制/套用/改写的模板与清单
4. **可复用与可升级**：同一个导师/实验室的风格会迭代；工具要能版本化、可比较、可更新

### 0.2 因果链（从目标推导出必需能力）

要实现“对齐 + 证据 + 可执行”，系统必须做到：

* **从少量样本（3 篇 PDF）中**抽取稳定模式（宏观结构 + 微观措辞 + 图表叙事）
* 把模式变成**结构化规则 + 模板**（可被机器应用）
* 每条模式必须绑定**Evidence Pointer**（页码/段落/高亮框）
* 生成结果要经过**自动校验**：是否真的来自样本风格，而不是 LLM 幻觉

---

## 1) Demo 产品定义（MVP 版本）

### 1.1 Demo 名称

**Mentor Intro SkillPack Demo v0**（SkillPack Studio 的第一个“爆火验证切片”）

### 1.2 用户可见输入（System Input）

**必需输入**

1. `PDF_1, PDF_2, PDF_3`：3 篇目标风格参考论文（导师/实验室代表作）
2. `pack_name`：技能包名称（例如“Prof.X Intro Style 2026Q1”）

**可选输入（不影响 Demo 核心闭环）**
3. `field_hint`：领域提示（CV/NLP/系统/生物等）
4. `target_venue_hint`：期刊/会议风格提示（可空）
5. `language`：默认 English

### 1.3 用户可见输出（System Output）

系统输出一个 **SkillPack Report + 可下载模板**，包含 3 个核心交付物，每一条结论都带证据链接：

1. **导师 Introduction 蓝图（Intro Blueprint）**

* 段落级结构：P1 做什么、P2 做什么…（按导师常见顺序）
* 每段的“功能标签”（Problem / Context / Gap / Key idea / Contribution / Roadmap）
* 导师典型“故事推进方式”（转折、强调、对比、承诺的写法）
* 每条结论：附 `evidence[]`（点击跳转 PDF 的页码并高亮）

2. **3 个可直接用的段落模板（3 Paragraph Templates）**

* Template A：开场背景/问题定义段（带可填槽位）
* Template B：Gap/Challenge + “我们做了什么”段
* Template C：Contributions + Roadmap 段
* 每个模板包含：常用句式库 + 连接词/hedging 风格约束 + 禁忌（don’t）

3. **Figure Storyboard（图表叙事分镜）**

* 建议图 1–图 N 的职责（图 1 讲故事、图 2 讲方法、图 3 讲核心实验…）
* 每个图的“应该出现在 intro 的哪一段后面”、caption 的典型句式
* 证据：样本论文中类似图的页码、caption 风格实例

**额外输出（机器可用）**
4) `skillpack.json` / `skillpack.yaml`：结构化技能包（规则、模板、清单、证据映射）
5) `evidence_index.json`：证据索引（用于 PDF viewer 高亮跳转）

---

## 2) 关键约束与不可妥协原则（Demo 也必须守）

1. **Evidence-first**：任何“导师风格结论”必须可追溯到 PDF（页码/段落/高亮框）；否则降级为“猜测建议”，并显式标记 `confidence=low`
2. **Anti-plagiarism**：模板必须是“结构与句式范式”，避免复制原句；可用短片段引用但必须很短且做引用标识（Demo 阶段也要做）
3. **Deterministic-ish**：同一组 PDF 多次构建，结果应高度一致（允许小幅措辞变化）
4. **可评测**：输出必须能跑自动检查（结构覆盖率、证据覆盖率、风格一致性评分）

---

## 3) 数据模型（最小可用 + 可扩展）

> 这是后续“最小不可分模块”的接口基础。

### 3.1 Document（PDF 文档）

* `pdf_id`
* `file_hash`
* `title`（可空）
* `num_pages`
* `created_at`

### 3.2 TextBlock（可定位文本块）

* `block_id`
* `pdf_id`
* `page_index`（从 1 开始）
* `text`
* `bbox`：`[x0,y0,x1,y1]`（用于高亮；没有 bbox 也要至少有 page+block_index）
* `block_type`：paragraph / heading / caption / table / reference
* `section_path`：例如 `["Introduction"]`

### 3.3 EvidencePointer（证据指针）

* `pdf_id`
* `page_index`
* `bbox`（可空，但强烈建议有）
* `block_id`（可空）
* `excerpt`（短摘录，用于 UI 提示）
* `reason`（为什么这段支持该结论）
* `confidence`：0–1

### 3.4 RhetoricalMove（修辞/功能标签）

* `move_id`
* `label`：Context / Problem / Gap / Approach / Contribution / Roadmap / RelatedWorkHook / Claim / Limitation …
* `block_id`
* `confidence`

### 3.5 StyleFeatures（风格特征）

* `sentence_len_stats`：均值/分位
* `hedging_profile`：may/likely/clearly 等比例
* `connector_profile`：However/Therefore/In summary 等分布
* `voice_profile`：we/this paper/passive
* `citation_density`：每段引用数

### 3.6 Pattern（跨文档共性模式）

* `pattern_id`
* `pattern_type`：MoveSequence / Lexicon / ClaimStyle / ParagraphLength / FigureNarrative …
* `description`
* `supporting_evidence[]`：EvidencePointer 列表
* `strength_score`（跨文档一致性强度）

### 3.7 Template（模板）

* `template_id`
* `template_type`：IntroOpening / GapApproach / ContributionsRoadmap
* `text_with_slots`：带槽位（{PROBLEM}、{GAP}…）
* `slot_schema`：每个槽位应该填什么、长度、语气
* `do_rules[]` / `dont_rules[]`
* `supporting_evidence[]`

### 3.8 StoryboardItem（分镜项）

* `item_id`
* `figure_role`：Overview / MethodPipeline / CoreModule / Ablation / SOTAComparison / Qualitative …
* `recommended_position`：Intro 段落编号或 section
* `caption_formula`
* `supporting_evidence[]`

### 3.9 SkillPack（最终产物）

* `pack_id`
* `pack_name`
* `pdf_ids[]`
* `intro_blueprint`
* `templates[]`
* `storyboard[]`
* `patterns[]`
* `version`
* `build_metadata`：模型版本、时间、参数
* `quality_report`：覆盖率、证据率、评分

---

## 4) 最小不可分模块（MIU）拆解：每个模块的输入/输出/功能

下面是**最小不可分**的功能单元列表（按构建链路顺序），每一个都写清楚：**输入是什么、输出是什么、功能是什么、失败模式是什么**。

> 你可以把它当成“研发任务的最小 ticket 粒度”。

---

### MIU-01：文件上传与校验

* **输入**：`PDF bytes + filename`
* **输出**：`pdf_id + file_hash + stored_path`
* **功能**：上传、校验文件类型、算 hash、落盘/对象存储
* **失败模式**：非 PDF / 文件过大 / 网络中断 → 可重试分片上传（可选）

### MIU-02：去重与版本策略

* **输入**：`file_hash`
* **输出**：`dedup_decision`（新建/复用已有解析结果）
* **功能**：避免重复解析，节省 token/算力
* **失败模式**：hash 冲突极低概率（可忽略或加 size+hash）

### MIU-03：PDF 元信息抽取

* **输入**：`stored_path`
* **输出**：`num_pages + title(optional) + toc(optional)`
* **功能**：基础信息用于 UI/分页
* **失败模式**：无标题/无目录 → 允许空

### MIU-04：页面级文本提取（带位置）

* **输入**：`stored_path`
* **输出**：`TextBlock[]`（至少 page_index+text；最好带 bbox）
* **功能**：把 PDF 拆成可定位的块（段落/标题/caption）
* **失败模式**：扫描版无文本层 → 触发 OCR 分支（Demo 可先不做，但要留接口）

### MIU-05：段落/标题/Caption 分类

* **输入**：`TextBlock[]`
* **输出**：为每个 block 填充 `block_type`
* **功能**：区分正文段落 vs 标题 vs 图注，为后面 “intro/figure” 做准备
* **失败模式**：格式乱 → 回退为基于字体大小/位置的规则分类

### MIU-06：章节切分与 Introduction 定位

* **输入**：已分类 `TextBlock[]`
* **输出**：每个 block 填 `section_path`；并输出 `intro_blocks[]`
* **功能**：找到 Introduction 的正文段落集合
* **失败模式**：没有显式 “Introduction” → 用启发式：Abstract 后第一大节、或 “1. Introduction” 模式匹配

### MIU-07：引用与公式噪声清理（保守）

* **输入**：`intro_blocks[]`
* **输出**：`clean_intro_blocks[]`（保留原文 + 清洗版）
* **功能**：移除多余换行、参考文献编号噪声，但不能破坏证据定位
* **失败模式**：清洗导致定位偏移 → 保留 raw_text 并建立映射

### MIU-08：句子切分与统计特征提取

* **输入**：`clean_intro_blocks[]`
* **输出**：`StyleFeatures`（句长分布、连接词分布、we/this paper 比例…）
* **功能**：微观风格特征的“硬统计”，减少纯 LLM 幻觉
* **失败模式**：英文切分不准 → 简单规则+模型混合

### MIU-09：Rhetorical Move 标注（段落功能分类）

* **输入**：`clean_intro_blocks[]` + `field_hint(optional)`
* **输出**：`RhetoricalMove[]`
* **功能**：判断每段在做什么（Context/Gap/Contribution…）
* **实现策略**：LLM 分类 + 自一致投票（多次采样）+ 规则校正
* **失败模式**：模型摇摆 → 用投票+置信度；低置信度段落标黄提示人工确认（Demo 可先自动）

### MIU-10：Move 序列压缩（生成“段落骨架”）

* **输入**：`RhetoricalMove[]`
* **输出**：`MoveSequence`（例如：Context→Problem→Gap→Approach→Contrib→Roadmap）
* **功能**：把一篇 intro 的结构压缩成可比对序列
* **失败模式**：段落过多/过少 → 允许同类 move 合并

### MIU-11：跨文档对齐（3 篇 intro 的结构对齐）

* **输入**：三份 `MoveSequence`
* **输出**：`AlignedMovePlan`（共性段落数、每段的主标签/可选标签）
* **功能**：找出导师“稳定套路”（共同的段落功能排序）
* **失败模式**：3 篇差异大 → 输出“主套路 + 分支套路”，并提示稳定性分数

### MIU-12：Intro Blueprint 生成（可读说明 + 可执行清单）

* **输入**：`AlignedMovePlan` + `supporting blocks`
* **输出**：`intro_blueprint`（段落级说明 + Do/Don’t + checklist）
* **功能**：产出用户可读的“导师 intro 写法”
* **失败模式**：抽象过度 → 强制每条结论带 evidence，否则不输出为“规则”

### MIU-13：证据自动选择（每条规则找支撑片段）

* **输入**：`intro_blueprint items` + `TextBlock index`
* **输出**：每条规则绑定 `supporting_evidence[]`
* **功能**：把“规则”锚定到具体页码/段落/高亮框
* **失败模式**：找不到证据 → 降级为建议（非规则），或丢弃

### MIU-14：模板槽位设计（slot schema）

* **输入**：`AlignedMovePlan` + `StyleFeatures`
* **输出**：`slot_schema`（每个槽位填什么、长度范围、语气强度、引用密度建议）
* **功能**：让模板可复用、可控，不是一次性生成
* **失败模式**：槽位太抽象 → 用具体字段（PROBLEM/GAP/METHOD/CLAIM）

### MIU-15：3 个段落模板生成（Text with Slots）

* **输入**：`slot_schema` + `lexicon` + `do/dont rules`
* **输出**：`Template A/B/C`
* **功能**：直接给用户可复制的段落骨架，符合导师连接词与语气
* **失败模式**：模板像通用英语作文 → 必须注入导师特征（连接词、hedging、roadmap 句式）

### MIU-16：模板反抄袭检查（自检）

* **输入**：`Template text` + `source blocks`
* **输出**：`plagiarism_risk_report`（相似片段警报）
* **功能**：避免模板与原文过近（字符 n-gram/embedding 相似度）
* **失败模式**：误报 → 允许人工忽略，但默认要重写

### MIU-17：Figure Caption 抽取

* **输入**：全论文 `TextBlock[]`
* **输出**：`captions[]`（figure number + caption text + page + bbox）
* **功能**：为 storyboard 找证据与风格样本
* **失败模式**：caption 识别不准 → 用 “Figure/Fig.” 模式 + 字体/位置规则

### MIU-18：Intro 内 figure 引用关系抽取

* **输入**：`intro_blocks[]` + `captions[]`
* **输出**：`intro_figure_refs[]`（Intro 中引用了哪些 Fig/Tab）
* **功能**：判断导师是否喜欢在 intro 提前“预告图 1”
* **失败模式**：无引用 → 仍可做 storyboard，但证据弱

### MIU-19：Storyboard 角色归类（从 caption 推断图类型）

* **输入**：`captions[]` + `field_hint`
* **输出**：每个 caption 的 `figure_role`（overview/method/qualitative…）
* **功能**：形成“导师常见的图表叙事顺序”
* **失败模式**：caption 太短 → 降级为 role=unknown

### MIU-20：Figure Storyboard 生成

* **输入**：`figure_role distribution` + `intro blueprint` + `caption styles`
* **输出**：`StoryboardItem[]`
* **功能**：给出“建议画什么图、放哪里、caption 怎么写”
* **失败模式**：缺少方法细节 → 输出“通用 storyboard（按导师论文常见）”，并标注泛化程度

### MIU-21：SkillPack 结构化打包

* **输入**：`intro_blueprint + templates + storyboard + patterns + evidence`
* **输出**：`skillpack.json/yaml`
* **功能**：可版本化、可分享、可复用
* **失败模式**：字段缺失 → schema 校验不过就构建失败（返回错误报告）

### MIU-22：质量报告（Quality Report）

* **输入**：构建产物
* **输出**：`quality_report`
* **功能**：至少包含：

  * evidence 覆盖率（多少规则有证据）
  * pattern 强度（3 篇一致性）
  * 可执行性评分（模板槽位完整度）
* **失败模式**：分数低 → UI 给“弱结论”标识

### MIU-23：PDF Viewer 跳转（page anchor）

* **输入**：`pdf_id + page_index`
* **输出**：PDF 指定页渲染
* **功能**：点击证据 → 直接跳页
* **失败模式**：浏览器兼容 → fallback：下载并提示页码

### MIU-24：PDF 高亮框渲染（bbox overlay）

* **输入**：`pdf_id + page_index + bbox`
* **输出**：高亮显示文本位置
* **功能**：让“证据”可视化，用户一眼信服
* **失败模式**：bbox 不准 → fallback：只高亮该页并在侧边栏显示 excerpt

### MIU-25：报告页渲染（HTML/React）

* **输入**：`skillpack.json`
* **输出**：可交互报告页（蓝图/模板/分镜/证据）
* **功能**：用户看得懂、可复制、可点击证据
* **失败模式**：大文本渲染卡顿 → 分段加载

### MIU-26：构建任务编排（Job Orchestrator）

* **输入**：`build request`
* **输出**：`job_id + status + progress`
* **功能**：异步构建、进度条、失败可重试
* **失败模式**：API 速率限制（你提到的约束）→ 自动退避 + 队列限流

### MIU-27：缓存与复用（避免重复花 token）

* **输入**：`pdf_id`、中间产物 key
* **输出**：命中则直接返回
* **功能**：同一 PDF 多次 build 不重复提取/标注
* **失败模式**：缓存污染 → 版本号隔离（模型版本变更则失效）

### MIU-28：观测与日志（最小必需）

* **输入**：每次构建的关键事件
* **输出**：`trace_id`、错误栈、token 用量、耗时分布
* **功能**：定位失败点、优化体验
* **失败模式**：日志泄露隐私 → 脱敏策略（不记录原文全文）

### MIU-29：隐私与权限（Demo 也要有）

* **输入**：用户权限设置（private/public）
* **输出**：访问控制结果
* **功能**：默认私有；分享需显式开启
* **失败模式**：误公开 → 默认 private + 二次确认（Demo 可简化但要有默认）

---

## 5) 端到端构建流程（E2E Flow，逐步列出）

1. 用户上传 3 个 PDF（MIU-01,02）
2. 系统解析每个 PDF：

   * 元信息（MIU-03）
   * 文本块与定位（MIU-04）
   * 块类型分类（MIU-05）
   * 章节切分+Intro 定位（MIU-06）
   * Intro 清洗（MIU-07）
3. 对每篇 intro：

   * 统计风格特征（MIU-08）
   * 段落功能标注（MIU-09）
   * 得到 move 序列（MIU-10）
4. 跨 3 篇：

   * 对齐共性结构（MIU-11）
   * 生成 intro 蓝图（MIU-12）
   * 为蓝图每条结论找证据（MIU-13）
5. 生成模板：

   * 槽位 schema（MIU-14）
   * 3 个模板（MIU-15）
   * 反抄袭自检（MIU-16）
6. 生成 figure storyboard：

   * caption 抽取（MIU-17）
   * intro figure 引用关系（MIU-18）
   * role 归类（MIU-19）
   * storyboard（MIU-20）
7. 打包与报告：

   * skillpack.json（MIU-21）
   * 质量报告（MIU-22）
   * 报告页（MIU-25）
   * 证据跳转与高亮（MIU-23,24）
8. 全流程由任务编排与限流支持（MIU-26,27,28,29）

---

## 6) 每个“内层部分”的输入/输出/功能（按你要的“逐一对应”）

为了更符合你“内层每一部分分别对应 I/O”的要求，我把 Demo 的 3 个主输出拆成“子组件级 I/O”。

---

### 6.1 Intro Blueprint（导师 intro 蓝图）

**子组件 BP-1：Intro 段落结构图（段落功能与顺序）**

* 输入：`intro_blocks[] + rhetorical_moves[] + aligned_move_plan`
* 输出：`paragraph_plan[]`（P1 label=Context，P2 label=Gap…）
* 功能：告诉用户每段“该做什么”
* 证据：每段 plan 需附 1–3 个 EvidencePointer（对应样本段落）

**子组件 BP-2：故事推进规则（Story Rules）**

* 输入：`connector_profile + move transitions + exemplar sentences`
* 输出：`story_rules[]`（例如：Gap 段常以 However 起句）
* 功能：从“顺序”升级到“怎么转折、怎么承诺、怎么落地”
* 证据：每条 rule 绑定多个论文例子（不同 PDF）

**子组件 BP-3：Claim/Contrib 语气规则（Claim Strength & Hedging）**

* 输入：`hedging_profile + contribution paragraphs`
* 输出：`claim_rules[]`（例如：贡献列表用 assertive 但避免 absolute）
* 功能：对齐导师“说话力度”
* 证据：指向贡献段、结论段

**子组件 BP-4：Checklist（可执行检查清单）**

* 输入：`aligned_move_plan + do/dont rules`
* 输出：`intro_checklist[]`
* 功能：写完 intro 自查（是否有 gap、是否有 roadmap）
* 证据：清单条目引用对应证据（可选但推荐）

---

### 6.2 三个段落模板（Templates A/B/C）

**子组件 T-1：Slot Schema（槽位定义）**

* 输入：`aligned_move_plan + style_features`
* 输出：每个模板的槽位定义（字段、长度、语气、引用密度）
* 功能：模板可复用、可控

**子组件 T-2：Phrase Bank（短语库）**

* 输入：`connector_profile + frequent n-grams (filtered) + hedging profile`
* 输出：连接词/转折句/总结句的可选短语列表（非原句复制）
* 功能：让模板“像导师”，不只是结构像

**子组件 T-3：Template Text（带槽位的段落正文）**

* 输入：`slot_schema + phrase_bank + do/dont rules`
* 输出：模板文本（含 {PROBLEM} {GAP} {METHOD} {CLAIM}）
* 功能：直接复制到 Overleaf 或 Word 里填空

**子组件 T-4：Template Evidence（模板依据）**

* 输入：模板中每个关键句式的来源匹配
* 输出：模板证据列表（告诉用户：这类句式来自哪些段落/页码）
* 功能：增强信任与可解释性

---

### 6.3 Figure Storyboard（分镜）

**子组件 F-1：Figure Inventory（样本图表清单）**

* 输入：`captions[]`
* 输出：每篇论文有哪些图、各自 role、caption 风格
* 功能：建立“导师常用图谱”

**子组件 F-2：Narrative Placement（图与叙事位置关系）**

* 输入：`intro_figure_refs[] + blueprint paragraph plan`
* 输出：图应该在 intro 的哪段后出现/被提及
* 功能：把“图表”变成“故事的一部分”

**子组件 F-3：Storyboard Suggestions（建议图序列）**

* 输入：`figure role distribution + field_hint`
* 输出：图 1..N 的建议（职责、内容要点、caption 公式）
* 功能：为新论文提供可执行分镜，而不是只复述样本

**子组件 F-4：Evidence Links（示例链接）**

* 输入：每个 figure_role 的 exemplar captions
* 输出：每条分镜建议对应的样本 figure 证据（页码+高亮 caption）
* 功能：用户点一下就看到导师图注长什么样

---

## 7) 系统实现方案（架构、接口、存储、限流）

### 7.1 总体架构（最小可行）

* **前端**：上传 + 构建进度 + 报告页 + PDF viewer（证据高亮）
* **后端 API**：上传、构建 job、查询结果、拉取证据索引
* **Worker**：跑解析、标注、生成、打包（LLM 调用集中在 worker）
* **存储**：

  * 原始 PDF 对象存储
  * 解析产物（TextBlock、captions）存数据库/文档库
  * 向量索引（可选，用于相似段落检索和证据匹配）

### 7.2 最小 API 设计（Demo 够用）

1. `POST /api/pdfs/upload`

   * in：multipart pdf
   * out：`pdf_id`

2. `POST /api/skillpacks/build`

   * in：`pdf_ids[3] + pack_name + optional hints`
   * out：`job_id`

3. `GET /api/jobs/{job_id}`

   * out：`status + progress + error(optional) + pack_id(optional)`

4. `GET /api/skillpacks/{pack_id}`

   * out：`skillpack.json`

5. `GET /api/pdfs/{pdf_id}/page/{page_index}`

   * out：PDF page render / url

6. `GET /api/evidence/{pack_id}`

   * out：`evidence_index.json`（用于前端高亮）

### 7.3 速率限制与“几乎无限 API”的使用方式

你在 Microsoft Research Asia 这边 API 配额宽松，但有调用速率限制 → Worker 必须实现：

* **队列**（job queue）
* **并发上限**
* **指数退避重试**
* **分阶段缓存**（PDF 解析、Intro move 标注等中间结果都缓存）

---

## 8) 完整实现计划（分阶段、逐一 list、最小不可分任务）

> 你要的是“逐阶段逐一 list，不省略任何内容”。下面我把上面的 MIU 按交付阶段组织成一个可执行 plan。
> 我不写“需要几周几天”的时间估计（你明确不需要也不该靠这个），只给**顺序、依赖、验收标准（Exit Criteria）**。

---

### Phase 0：项目底座与规范（必须先做，否则后面不可控）

**P0-01** 建 repo / monorepo 结构（frontend/backend/worker/shared）

* 输出：可运行的空服务骨架

**P0-02** 定义 JSON Schema（SkillPack、EvidencePointer、TextBlock…）

* 输出：schema 文件 + 校验器

**P0-03** 选择并实现存储接口（对象存储 + DB）

* 输出：`store_pdf() / load_pdf()`；`save_blocks()/load_blocks()`

**P0-04** 实现 job 框架（创建 job、更新状态、失败记录）

* 输出：`create_job/build_job/update_job_status`

**P0-05** 日志与 trace_id 贯穿（前端请求→worker）

* 输出：可追踪一条构建链路

**Exit Criteria**

* 上传一个 PDF → 建 job → job 标记完成（即便无实际分析）→ 前端可看到状态

---

### Phase 1：PDF 解析与证据定位（Evidence 的地基）

**P1-01（MIU-01）** 上传与校验
**P1-02（MIU-02）** hash 去重
**P1-03（MIU-03）** 元信息抽取
**P1-04（MIU-04）** 文本块提取（page + text + bbox）
**P1-05（MIU-05）** block_type 分类
**P1-06（MIU-06）** section 切分 + Intro 定位
**P1-07（MIU-23）** PDF viewer 跳页
**P1-08（MIU-24）** bbox 高亮渲染（最少支持 caption/段落高亮）

**Exit Criteria**

* 任意一篇 PDF：能在网页打开并跳到某页；能对某段 TextBlock 做高亮；Intro 段落可被列出来（即便还没标注 move）

---

### Phase 2：Intro 结构标注（从文本到“导师套路”）

**P2-01（MIU-07）** Intro 清洗（保留 raw 映射）
**P2-02（MIU-08）** 统计风格特征（硬统计）
**P2-03（MIU-09）** Rhetorical Move 标注（段落功能分类）

* 关键：多次采样投票 + 置信度输出

**P2-04（MIU-10）** Move 序列压缩（段落骨架）

**Exit Criteria**

* 对单篇论文：能输出 Intro 段落列表，每段有 label + confidence；能展示 move sequence

---

### Phase 3：跨 3 篇对齐与 Blueprint 生成（Demo 核心）

**P3-01（MIU-11）** 三篇 move sequence 对齐（主套路+分支套路）
**P3-02（MIU-12）** 生成 Intro Blueprint（段落级说明 + rules + checklist）
**P3-03（MIU-13）** 为 blueprint 每条规则自动找证据（至少 1 条，多则更强）
**P3-04（MIU-22）** 质量报告（证据覆盖率、结构稳定性）

**Exit Criteria**

* 用户看到一个“P1~Pn 该怎么写”的蓝图；每条规则能点回 PDF 页码并高亮；质量报告能提示哪些结论稳定/不稳定

---

### Phase 4：3 个可直接用模板（把“建议”变“产出”）

**P4-01（MIU-14）** 槽位 schema（可执行约束）
**P4-02（MIU-15）** 模板 A/B/C 生成（带槽位、带 phrase bank）
**P4-03（MIU-16）** 反抄袭自检（相似度报警，触发重写）
**P4-04** 模板证据绑定（模板风格特征来源于哪些段落）

**Exit Criteria**

* 报告页里有 3 个可复制模板；每个模板都有“怎么填”的槽位说明；模板有证据链接；相似度过高会自动重写或标红

---

### Phase 5：Figure Storyboard（让 demo 有“第二个 wow”）

**P5-01（MIU-17）** caption 抽取（Fig./Table）
**P5-02（MIU-18）** intro 中 figure 引用关系抽取
**P5-03（MIU-19）** figure_role 归类（overview/method/qualitative…）
**P5-04（MIU-20）** storyboard 生成（图序列、放置建议、caption 公式）
**P5-05** storyboard 证据绑定（点击跳到对应图注）

**Exit Criteria**

* 报告页里有 Figure 1..N 的建议分镜；每条建议可点到样本图；caption 公式能反映导师风格

---

### Phase 6：SkillPack 打包与对外接口（从 demo 到产品资产）

**P6-01（MIU-21）** skillpack.json/yaml 打包
**P6-02（MIU-25）** 报告页渲染（蓝图/模板/分镜/证据）
**P6-03（MIU-26）** job 编排完善（进度条分阶段）
**P6-04（MIU-27）** 缓存复用（解析/标注结果可复用）
**P6-05（MIU-28）** 观测面板（token、错误率、构建成功率）
**P6-06（MIU-29）** 隐私权限（默认 private；分享可选）

**Exit Criteria**

* 用户可下载 skillpack.json；同一组 PDF 重建速度显著提升（缓存命中）；构建失败能定位；默认私有可控

---

## 9) 如何用你现有的 1.5 万条 skill（不是摆出来，而是做“抽取加速器”）

你已经从 GitHub、Zhihu、Stack Overflow、Google + Baidu 聚合了 125 topic / 1.5 万 skill。Demo 阶段它的最佳用法是：

* **在 MIU-09（Move 标注）里做 label prior**：给 LLM 一个“领域写作 move 标签体系 + 定义 + 例子”作为稳定器
* **在 MIU-12（Blueprint rules）里做 rule candidate pool**：先从 skill 库抽候选规则，再让 LLM 只做“选择 + 证据匹配 + 改写成导师风格”
* **在 MIU-13（证据匹配）里做 retrieval hint**：用 skill 库的关键词（However/Gap/Contribution 等）提升证据查找成功率
* **在质量报告里做 coverage**：哪些 topic/规则在这 3 篇里出现/缺失（帮助判断“导师风格强特征”）

一句话：skill 库是“引擎燃料”，不是用户界面上的“目录”。

---

## 10) Demo 成功判定（你要验证“爆火可能性”，必须量化）

### 10.1 必须达到的“硬指标”

1. **Evidence 覆盖率**：Intro Blueprint 的规则中 ≥ 80% 有可点击证据（page + excerpt；最好 bbox）
2. **模板可用性**：用户复制模板到论文里，只需填槽位即可形成自然段（不需要重写 70%）
3. **结构一致性强**：三篇论文对齐出的主套路能解释至少 2 篇（否则说明抽取不稳定）
4. **输出稳定**：同一输入重复 build，Blueprint 结构不乱跳（措辞可变但结构要稳定）

### 10.2 可选的“爆火指标”（增长视角）

* 报告页可生成一张“Style Fingerprint 卡片”（可分享截图）
* 一键生成 “导师 intro roadmap 段落”对比（前后对比更容易传播）

---

## 11) 你下一步真正要做的“最小闭环 Demo”清单（如果你只做 10 件事）

如果你要极致最小化，也能按下面顺序打穿：

1. 上传 PDF + 存储（MIU-01）
2. 文本块提取（MIU-04）
3. Intro 定位（MIU-06）
4. 段落 move 标注（MIU-09）
5. 三篇对齐（MIU-11）
6. Blueprint 生成（MIU-12）
7. 证据绑定（MIU-13）
8. PDF 跳页（MIU-23）
9. 报告页渲染（MIU-25）
10. SkillPack JSON 打包（MIU-21）

做到这 10 件事，你就已经能验证“用户看见证据 → 信服 → 觉得爽 → 分享”的爆火链路。模板与 storyboard 是第二阶段加速器。

---

如果你希望我把以上 proposal 再进一步“工程化到可以直接开工”的程度：我可以在同一套结构下，把 **skillpack.json 的具体字段示例（含 Blueprint/Template/Storyboard 的 JSON 样例）**也写出来，保证你前后端完全对齐、直接照 schema 开发。
