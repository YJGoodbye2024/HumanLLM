# HumanLLM

[中文说明](README_zh.md)

This document describes the data itself: what each file means, how the dataset is organized, and how the main concepts connect.

The easiest way to understand `Dataset/` is to follow the same progression as the paper:

`pattern -> scenario -> conversation -> checklist -> SFT sample`

In other words, HumanLLM first defines psychological patterns, then combines multiple patterns into scenarios, then generates multi-turn conversations inside those scenarios, then provides checklist-based evaluation targets, and finally converts the scenario data into ShareGPT-style supervised fine-tuning samples.

## Directory Structure

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

## 1. Pattern Layer

File:

- `Dataset/patterns_data/patterns_data.json`

This layer stores the 244 psychological patterns themselves. In the paper, this is the `pattern data` layer.

### What it contains

Each top-level entry is one pattern. The value is a structured description with fields such as:

- `construct_name`
  - The pattern name.
- `description`
  - What the pattern is. This is close to the paper's `Definition`.
- `core_mechanisms`
  - The cognitive, emotional, and behavioral mechanisms behind the pattern.
- `real_world_manifestation`
  - Typical ways the pattern appears in real situations.

### What this layer means

This is not dialogue data. It is the psychological knowledge layer that defines:

- what a pattern is
- how it works
- how it tends to appear in human behavior

You can think of it as the conceptual foundation for everything downstream.

## 2. Scenario Layer

Files:

- `Dataset/split/train.json`
- `Dataset/split/id_eval.json`
- `Dataset/split/ood_eval.json`
- `Dataset/split/mixed_eval.json`

This layer stores how multiple patterns are combined into concrete social situations.

In the paper, this corresponds to:

- scenario synthesis
- scenario variation

### What each scenario sample contains

Common fields include:

- `pattern`
  - The list of target patterns in the scenario.
  - A scenario usually contains 2 to 5 patterns.
- `situation`
  - The situation type, such as adversity, decision, and so on.
- `analysis`
  - A textual explanation of how the selected patterns interact in this scenario.
- `scenario`
  - The main scenario object, containing the background and character profiles.
- `protagonist`
  - The name of the protagonist.
- `supporting_characters`
  - The list of supporting character names.
- `char2pattern`
  - The mapping from characters to assigned patterns.

Some samples also include variation-related fields:

- `factor_analysis`
- `new_factor_settings`
- `scenario_variation`
- `analysis_variation`
- `supporting_characters_variation`

These fields indicate that the dataset does not only provide one scenario, but also an alternative version where some situational factors are changed and the resulting behavioral tendencies are re-analyzed.

### What `scenario` contains

`scenario` is usually an object with two main parts:

- `storyBackground`
  - The narrative background.
  - This describes the time, place, atmosphere, and triggering tension in the scene.
- `charactersProfiles`
  - The character profiles.

`charactersProfiles` is usually split into:

- `protagonist`
  - The protagonist's profile.
- `supportingCharacter`
  - A list of supporting character profiles.

### What each character profile contains

A character profile usually includes:

- `name`
  - The character name.
- `aboutSelf`
  - How the character is described from their own perspective.
  - This includes identity, background, personality, and motivations.
- `aboutOthers`
  - How the character understands and evaluates the other characters.

### What this layer means

This layer grounds abstract patterns in concrete situations. It answers questions like:

- which patterns appear together
- which character receives which patterns
- what the relationships are between the characters
- what kind of pressure or context activates those patterns

## 3. Conversation Layer

Files:

- the `conversation` field inside `Dataset/split/*.json`

This layer stores the actual multi-turn interaction generated inside a scenario.

In the paper, this corresponds to:

- conversation synthesis

### What each conversation contains

In the standard format, `conversation` is a list of turns. Each turn usually contains:

- `char`
  - The speaking character.
- `content`
  - The full expression produced in that turn.

### Internal structure of `content`

The paper defines each turn as having three dimensions:

- inner thought
- physical action
- spoken dialogue

In the text, these usually appear as:

- `[...]`
  - inner thought
- `(...)`
  - physical action or visible expression
- the remaining text
  - spoken dialogue

So a turn is not only "what the character says." It also encodes:

- what the character is thinking
- what the character is doing
- what the character says out loud

### What this layer means

This is where patterns become dynamic behavior. Instead of static labels, the dataset shows:

- how a pattern shapes a character's reaction
- how multiple patterns reinforce, conflict with, or modulate one another
- how one character expresses psychology through thoughts, actions, and speech at the same time

## 4. Checklist Layer

HumanLLM uses two checklist layers.

### 4.1 Pattern-Level Checklist

File:

- `Dataset/checklist/pattern_level_checklist.json`

This is the cross-scenario checklist layer. It evaluates whether a character expresses a pattern in general.

### What each pattern-level checklist contains

Each entry usually includes:

- `trait`
  - The pattern name.
- `checklist`
  - A list of behavioral indicators.
- `case_review`
  - Notes about how well the checklist items are supported by example cases.
- `notes`
  - Additional instructions for using the checklist.

Each item in `checklist` usually includes:

- `question`
  - A behavior-focused evaluation question.
- `evidence_status`
  - Whether that question is supported in example cases.

### What this layer means

This layer answers:

- "If a character has this pattern, what behaviors should generally be observable?"

It is pattern-centric rather than scenario-centric.

### 4.2 Scenario-Level Checklist

Files:

- the `conversation_checklist` field inside `Dataset/split/*.json`
- and, for some samples, `conversation_checklist_variation`

This is the scenario-specific checklist layer. It evaluates whether a character behaves as expected under the current combination of patterns and the current situational setup.

### What each scenario-level checklist contains

`conversation_checklist` is usually a mapping from character name to a list of expected behaviors.

For example, it may specify:

- how a given character should react under criticism
- whether a character should become defensive under evaluation pressure
- whether a character should try to mediate or de-escalate

Each target character usually has 2 to 6 context-specific checklist items.

### What variation checklists mean

If a sample includes:

- `scenario_variation`
- `conversation_checklist_variation`

then the dataset is also providing:

- an alternative version of the scenario
- an alternative set of expected behaviors under the changed situational factors

### What this layer means

This layer answers:

- "In this specific scenario, what does correct pattern expression look like?"

It is not about the general definition of one pattern. It is about how several patterns jointly play out in one concrete interaction.

## 5. SFT Sample Layer

Files:

- `Dataset/sft_data/sharegpt_train.json`
- `Dataset/sft_data/sharegpt_id_eval.json`
- `Dataset/sft_data/sharegpt_ood_eval.json`
- `Dataset/sft_data/sharegpt_mixed_eval.json`

This layer stores the supervised fine-tuning samples derived from the scenario-level data.

In the paper, this corresponds to:

- Supervised Fine-Tuning

### What each SFT sample contains

Each sample has two main fields:

- `system`
- `conversations`

### What `system` contains

`system` compresses the target role setup into one prompt, typically including:

- who the target character is
- how the character sees themselves
- how the character sees others
- what the current scenario is
- how the output should express thought, action, and speech

### What `conversations` contains

`conversations` is a ShareGPT-style turn sequence, usually alternating between:

- `human`
- `gpt`

Here the meaning is:

- `human`
  - the surrounding dialogue context or the other characters' turns
- `gpt`
  - the target character's response

### What this layer means

This is no longer the full scenario archive. It is the model-ready training format. It packages scenario context, role information, and dialogue windows into standard supervised examples.

## Meaning of the Four Splits

Both `split/` and `sft_data/` follow the same split logic:

- `train`
  - the training set
- `id_eval`
  - the in-domain evaluation set
  - patterns remain within the training domain
- `ood_eval`
  - the out-of-domain evaluation set
  - built from the paper's designated OOD patterns
- `mixed_eval`
  - the mixed evaluation set
  - contains both in-domain and OOD patterns

## One-Line Summary of Each Layer

- `patterns_data`: defines what each pattern is
- `split.scenario`: defines the social situation where patterns are instantiated
- `split.conversation`: shows how characters interact inside that situation
- `checklist + conversation_checklist`: defines what correct pattern expression looks like
- `sft_data`: converts the above into model-ready supervised samples
