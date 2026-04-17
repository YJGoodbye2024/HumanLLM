# HumanLLM 数据说明

[English Version](README.md)

本说明文档只介绍数据本身的含义与结构，帮助读者按论文中的逻辑理解 `Dataset/` 目录。

HumanLLM 的数据可以按下面的顺序理解：

`pattern -> scenario -> conversation -> checklist -> SFT sample`

也就是说，作者先定义心理模式，再把多个模式组合成场景，再为场景生成多轮对话，并配套两层 checklist，最后把场景级数据转换成 ShareGPT 风格的监督微调样本。

## 目录结构

```text
Dataset/
├── patterns_data/
│   └── patterns_data.json
├── checklist/
│   └── pattern_level_checklist.json
├── split/
│   ├── train.json
│   ├── id_eval.json
│   ├── ood_eval.json
│   └── mixed_eval.json
└── sft_data/
    ├── sharegpt_train.json
    ├── sharegpt_id_eval.json
    ├── sharegpt_ood_eval.json
    └── sharegpt_mixed_eval.json
```

## 1. Pattern 层

对应文件：

- `Dataset/patterns_data/patterns_data.json`

这一层保存的是 244 个心理模式本身的知识描述。论文中这部分对应 `pattern data`。

### 包含什么

每个 pattern 是一个顶层条目，value 是该模式的结构化摘要。核心字段包括：

- `construct_name`
  - 模式名称。
- `description`
  - 这个模式是什么，接近论文中的 `Definition`。
- `core_mechanisms`
  - 这个模式在认知、情绪、行为层面的机制。
- `real_world_manifestation`
  - 这个模式在现实情境中的典型表现方式。

### 这一层表达的含义

这一层不是对话数据，而是“心理学知识层”。它定义了：

- 一个 pattern 的基本概念
- 这个 pattern 如何驱动人的想法、情绪和行为
- 这个 pattern 在真实世界中会如何表现

可以把它理解成后续所有场景构造的“心理学原材料”。

## 2. Scenario 层

对应文件：

- `Dataset/split/train.json`
- `Dataset/split/id_eval.json`
- `Dataset/split/ood_eval.json`
- `Dataset/split/mixed_eval.json`

这一层保存的是“多个 pattern 如何在一个具体情境中组合成角色与事件”。

论文中这部分对应：

- scenario synthesis
- scenario variation

### 每条 scenario 样本包含什么

常见字段如下：

- `pattern`
  - 该场景包含的目标 pattern 列表。
  - 一个场景通常包含 2 到 5 个 pattern。
- `situation`
  - 场景所属的情境类型，例如 adversity、decision 等。
- `analysis`
  - 对这个场景中 pattern 如何相互作用的文字分析。
- `scenario`
  - 场景主体，里面包含背景和角色画像。
- `protagonist`
  - 主角名称。
- `supporting_characters`
  - 配角名称列表。
- `char2pattern`
  - 角色与 pattern 的对应关系。

部分样本还包含 variation 相关字段：

- `factor_analysis`
- `new_factor_settings`
- `scenario_variation`
- `analysis_variation`
- `supporting_characters_variation`

这些字段表示作者不仅生成了一个场景，还分析了“如果改变某些情境因素，角色行为会怎样变化”。

### `scenario` 里面包含什么

`scenario` 通常是一个对象，核心结构包括：

- `storyBackground`
  - 叙事背景。
  - 描述时间、地点、事件氛围、冲突触发点等。
- `charactersProfiles`
  - 角色画像。

`charactersProfiles` 通常又分成：

- `protagonist`
  - 主角画像。
- `supportingCharacter`
  - 配角画像列表。

### 每个角色画像包含什么

角色画像通常包含：

- `name`
  - 角色名。
- `aboutSelf`
  - 角色对自己的描述。
  - 包括身份、经历、性格、动机等。
- `aboutOthers`
  - 角色对其他角色的认知和态度。

### 这一层表达的含义

这一层的任务是把抽象的心理模式落到一个具体社会情境里。它回答的问题是：

- 哪些 pattern 会同时出现
- 它们分配给了哪些角色
- 这些角色之间是什么关系
- 当前场景会激活哪些行为倾向

## 3. Conversation 层

对应文件：

- `Dataset/split/*.json` 中的 `conversation` 字段

这一层保存的是在某个 scenario 中生成出的多轮互动。

论文中这部分对应：

- conversation synthesis

### 每条 conversation 包含什么

标准情况下，`conversation` 是一个按轮次排列的列表。每一轮通常包含：

- `char`
  - 当前说话的角色。
- `content`
  - 该角色这一轮的完整表达。

### `content` 的内部结构

论文规定每一轮表达包含三种维度：

- 内心想法
- 外显动作
- 说出口的话

在文本中通常表现为：

- `[...]`
  - 内心想法
- `(...)`
  - 动作或表情
- 剩余文本
  - 台词

也就是说，一轮对话并不只是“说了什么”，而是同时编码了：

- 角色此刻在想什么
- 角色外在做了什么
- 角色口头上说了什么

### 这一层表达的含义

这一层是 pattern 在动态互动中的实际展开。它不是静态标签，而是让读者看到：

- pattern 如何驱动角色反应
- 多个 pattern 如何在多人互动中彼此强化、冲突或调节
- 同一个角色如何在思想、动作、语言三个层面共同体现心理模式

## 4. Checklist 层

HumanLLM 使用两层 checklist。

### 4.1 Pattern-Level Checklist

对应文件：

- `Dataset/checklist/pattern_level_checklist.json`

这一层是“跨场景通用”的 checklist，用来评估一个角色是否表现出某个 pattern。

### 每条 pattern-level checklist 包含什么

每个条目通常包含：

- `trait`
  - pattern 名称。
- `checklist`
  - 行为指标列表。
- `case_review`
  - 对清单项的案例支持情况说明。
- `notes`
  - 使用这个清单时的补充说明。

其中 `checklist` 里的每一项通常包含：

- `question`
  - 要观察的行为问题。
- `evidence_status`
  - 该问题在案例中的支持情况。

### 这一层表达的含义

这一层回答的是：

- “如果某个角色具有某个 pattern，一般应该看到哪些可观察行为？”

它关注的是 pattern 自身的定义是否被表达出来，而不依赖具体场景。

### 4.2 Scenario-Level Checklist

对应文件：

- `Dataset/split/*.json` 中的 `conversation_checklist`
- 部分样本还带有 `conversation_checklist_variation`

这一层是“场景相关”的 checklist，用来评估某个角色在当前 scenario 中，是否按预期表现出该 pattern 组合。

### 每条 scenario-level checklist 包含什么

`conversation_checklist` 通常是一个“角色名 -> 条目列表”的映射。

例如：

- 某个角色在当前场景中应该如何表达压力
- 某个角色是否会因为被评判而防御
- 某个角色是否会试图调和冲突

每个角色通常有 2 到 6 条情境化行为描述。

### variation checklist 表示什么

如果样本中存在：

- `scenario_variation`
- `conversation_checklist_variation`

那么它表示同一组人物和 pattern 被放进了一个“变化后的情境”中，作者同时给出了：

- 变化后的场景版本
- 变化后的行为预期

### 这一层表达的含义

这一层回答的是：

- “在这个具体场景里，这些角色应该怎样表现才算符合 pattern 组合？”

它关注的不是单个 pattern 的一般定义，而是多 pattern 在具体场景下的联合作用。

## 5. SFT Sample 层

对应文件：

- `Dataset/sft_data/sharegpt_train.json`
- `Dataset/sft_data/sharegpt_id_eval.json`
- `Dataset/sft_data/sharegpt_ood_eval.json`
- `Dataset/sft_data/sharegpt_mixed_eval.json`

这一层是从 scenario-level 数据导出的监督微调样本。

论文中这部分对应：

- Supervised Fine-Tuning

### 每条 SFT 样本包含什么

每条样本只有两个核心字段：

- `system`
- `conversations`

### `system` 包含什么

`system` 通常会把目标角色需要知道的信息压缩成一段角色设定，包括：

- 你是谁
- 你的自我画像
- 你如何看待其他角色
- 当前场景是什么
- 输出时应如何组织 thought / action / speech

### `conversations` 包含什么

`conversations` 是 ShareGPT 风格的轮次序列，通常交替出现：

- `human`
- `gpt`

这里的含义是：

- `human`
  - 其他角色或上下文输入
- `gpt`
  - 目标角色的回应

### 这一层表达的含义

这一层已经不是“完整场景档案”，而是“可以直接用来训练模型”的格式。它把 scenario、角色画像和对话窗口打包成标准监督样本，供模型学习如何扮演目标角色。

## 四类 split 的含义

`split/` 和 `sft_data/` 中都按同样的评测逻辑分成四类：

- `train`
  - 训练集
  - 对应论文中的训练场景
- `id_eval`
  - in-domain 评测集
  - pattern 仍来自训练域
- `ood_eval`
  - out-of-domain 评测集
  - 由论文指定的 OOD pattern 组合构成
- `mixed_eval`
  - 混合评测集
  - 同时包含 in-domain 和 OOD pattern

## 一句话理解每一层

- `patterns_data`：定义 pattern 是什么
- `split.scenario`：定义 pattern 被放进了什么社会情境
- `split.conversation`：展示角色在情境中如何互动
- `checklist + conversation_checklist`：定义什么样的表现算“符合 pattern”
- `sft_data`：把这些内容转换成可直接训练模型的样本
