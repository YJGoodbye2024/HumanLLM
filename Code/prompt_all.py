format_prompt_system = '''
You are an expert-level JSON correction tool.
Your task is to receive a potentially malformed string from the user and correct it into a syntactically valid JSON object.

You must follow these Core Correction Rules:
1.  Ensure all string keys and values are enclosed in proper double quotes (`"`).
2.  If a string value contains internal, unescaped double quotes, remove them to ensure validity.
3.  Remove any trailing commas after the last element in objects and arrays.
4.  Ensure all brackets  and braces  are correctly matched and closed.
'''

format_prompt_user = '''
The following string is a malformed JSON object. Please correct it according to your rules.

**String to Fix:**
{malformed_string}

**Output Requirements:**
- Your response MUST be the corrected JSON object and nothing else.
- DO NOT output any explanatory text, introductory phrases.
- Do not use Markdown code blocks (e.g., ```json).
- The output must be pure plain text that a standard JSON parser can process successfully.
'''


# prompt for principle relationship
gen_relationship_prompt_system = """
You are an expert in psychology and cognitive science.Your task is to analyze a 'target principle' and identify its relationships with other principles from a given list.
"""
gen_relationship_prompt = """ 
The target principle is:
`{target_principle}`

Here is the full list of principles to compare against:
`{principle_list}`

Please perform the following steps:
1.  Carefully analyze the definition and implications of the `{target_principle}`.
2.  From the provided list, identify other principles that are directly related to the target principle.
3.  Aim to identify between 2-5 of the most relevant principles from the list. 
4.  For each related principle you identify, provide a brief and clear explanation of why it is related.
5.  Do NOT include the `{target_principle}` itself in your list of related principles.

**CRITICAL OUTPUT FORMAT**:
You MUST format each finding on a new line that starts with the exact phrase `Related Principle: ` followed by the principle's name. The reasoning should follow on subsequent lines. DO NOT deviate from this format.

Example of correct formatting:
Related Principle: Self-serving bias
Reason: This principle is also an attributional bias, specifically concerning success and failure.

Related Principle: Fundamental attribution error
Reason: This is the core mechanism for the "observer" part of the asymmetry.
"""

# prompt for situation-driven principle information
sd_principle_info_prompt = '''
**System Role:**
You are an expert academic synthesizer and psychological researcher. Your task is to process a large text corpus (synthesized from ~50 academic papers) and distill it into an in-depth, structured analytical report on its core psychological principle.

**Core Task & Instructions:**
Analyze the text corpus provided below, delimited by `[START_CORPUS]` and `[END_CORPUS]`.

Your task is to generate a clearly organized report. Follow the Markdown structure below *exactly*, and provide a deep, comprehensive answer for each section based *only* on the provided text.

# Construct Name: {Principle Name}

## Description
(Based on the corpus, provide a clear, detailed, and scientific description of what this principle is, how it manifests, and its underlying psychological mechanisms.)

## Core Mechanisms
(Based on the corpus, explain the primary evolutionary, cognitive, or emotional reasons why this principle exists. Synthesize the various explanations—e.g., is it a heuristic for efficiency, a result of memory limitations, a self-esteem protection mechanism, or something else? Be specific and in-depth.)

## Real-World Manifestation
(**This section is critical.** Draw synthesized insights from the literature to provide a profound analysis of the principle's broader impact.
- **Go Beyond Description:** Explore its nuanced consequences, not just a list of examples.
- **Challenges & Function:** Discuss how the literature indicates it might challenge conventional wisdom and its function as a 'double-edged sword' (e.g., both hindering and helping personal growth).
- **Practical Applications:** Explore its practical applications in fields like marketing, persuasion, or self-improvement, as evidenced in the text.
- **Core Insight:** Your analysis must reveal deeper truths about human behavior, explaining not just *what* happens but *why* it is significant. You must adhere to the high benchmark of depth, structure, and insight requested by the user.)

**Constraints:**
1.  **Strict Source Adherence:** Base all conclusions *exclusively* on the provided text corpus.
2.  **No JSON:** Your output **must** be a plain text report using the Markdown headings above.
3.  **Depth and Rigor:** Ensure the analysis is scientific, rigorous, and profound, especially the "Real-World Manifestation" section.

**[START_CORPUS]**

{ALL 50 PAPERS' CONTENT}

**[END_CORPUS]**
'''


sd_to_json = """**System Role:**
You are a precise data-to-JSON formatting specialist.

**Core Task & Instructions:**
Convert the Markdown-structured text provided between `[START_TEXT]` and `[END_TEXT]` into the *exact* JSON format specified below.

**Target JSON Structure:**
{
  "construct_name": "{PRINCIPLE_NAME}",
  "description": "Provide a clear, detailed, and scientific description of what this principle is, how it manifests, and its underlying psychological mechanisms.",
  "core_mechanisms": "Explain the primary evolutionary, cognitive, or emotional reasons why this principle exists. Is it a heuristic for efficiency, a result of memory limitations, a self-esteem protection mechanism, or something else? Be specific.",
  "real_world_manifestation": "Provide a profound analysis of the principle's broader impact. Go beyond a simple description to explore its nuanced consequences. Discuss how it might challenge conventional wisdom, its function as a 'double-edged sword' (e.g., both hindering and helping personal growth), and its practical applications in fields like marketing, persuasion, or self-improvement. The analysis should reveal deeper truths about human behavior, explaining not just *what* happens, but *why* it is significant. Use the detailed example provided by the user for 'Cognitive Dissonance' as the benchmark for the required depth, structure, and insight."
}

**Example Output (structure only):**
{
  "construct_name": "Example Principle",
  "description": "Summary text drawn exactly from the source.",
  "core_mechanisms": "Mechanistic explanation sourced from the corpus.",
  "real_world_manifestation": "Insightful analysis of real-world outcomes from the corpus."
}

**Mapping Rules:**
1.  Map all text under the `## Description` heading to the `description` field.
2.  Map all text under the `## Core Mechanisms` heading to the `core_mechanisms` field.
3.  Map all text under the `## Real-World Manifestation` heading to the `real_world_manifestation` field.

**Constraints:**
1.  Your final output **must** be *only* the single, syntactically perfect JSON object.
2.  Do not include *any* explanatory text, acknowledgments, or pre-amble.
3.  Do not wrap the JSON in Markdown code fences or add any extra words before or after the JSON.
4.  Ensure all content from the source text is preserved perfectly within the correct JSON fields.

**[START_TEXT]**

{model_response}

**[END_TEXT]**"""

# prompt for disposition-driven principle information
td_principle_info_prompt = '''
**System Role:**
You are an expert academic synthesizer and personality psychologist. Your task is to process a large text corpus (synthesized from ~50 academic papers on a specific personality trait) and distill it into an in-depth, structured analytical report.

**Core Task & Instructions:**
Analyze the text corpus provided below, delimited by `[START_CORPUS]` and `[END_CORPUS]`.

Your task is to generate a clearly organized report. Follow the Markdown structure below *exactly*, and provide a deep, comprehensive answer for each section based *only* on the provided text.

# Construct Name: {Trait Name}

## Definition
(Provide a precise and professional definition of this personality trait, referencing mainstream psychological theories from the corpus. Explain its role in an individual's personality structure as described in the text.)

## Core Mechanisms
(This section analyzes the foundational components of the trait.)

### Cognitive Patterns
(Describe the typical mindset, belief systems, and attentional focus of a person with this trait, as evidenced in the literature. How do they view the world, others, and themselves?)

### Emotional Signatures
(Describe the core emotions they tend to experience and express, their emotional stability, and their typical empathic responses, according to the corpus.)

### Behavioral Tendencies
(Describe the spontaneous, observable behaviors someone with this trait exhibits in everyday, non-pressured situations, as documented in the papers.)

## Real-World Manifestation
(This section analyzes how the trait is expressed across a wide variety of real-world contexts, situations, and life domains. Based on the corpus, synthesize a comprehensive overview of its practical implications. 

Your analysis should aim to be broad, covering **as many different contexts as are documented in the papers**. This **may include, but is not limited to**, areas such as:
* **Response to Stress and Adversity:** How does the trait manifest when the individual faces challenges, failure, or high pressure (e.g., is it amplified, diminished, or distorted)?
* **Interpersonal Dynamics:** What are the typical strategies for handling conflict? How does the trait impact teamwork, leadership, or close relationships?
* **Response to Positive Scenarios:** How is the trait expressed when the individual is succeeding, supported, or feeling happy?
* **Other Domains:** Look for evidence related to work performance, decision-making, health behaviors, or other significant life outcomes mentioned in the corpus.)

**Constraints:**
1.  **Strict Source Adherence:** Base all conclusions *exclusively* on the provided text corpus.
2.  **No JSON:** Your output **must** be a plain text report using the exact Markdown headings (H1, H2, H3) above.
3.  **Depth and Rigor:** Ensure the analysis is scientific, rigorous, and comprehensive, addressing all parts of each prompt.

**[START_CORPUS]**

{ALL 50 PAPERS' CONTENT}

**[END_CORPUS]**
'''

td_to_json = """**System Role:**
You are a precise data-to-JSON formatting specialist.

**Core Task & Instructions:**
Convert the Markdown-structured text provided between `[START_TEXT]` and `[END_TEXT]` into the *exact* JSON format specified below.

**Target JSON Structure:**
{{
  "construct_name": "{PRINCIPLE_NAME}",
  "definition": "Provide a precise and professional definition of this personality trait, referencing mainstream psychological theories. Explain its role in an individual's personality structure.",
  "core_mechanisms": "A single string containing all synthesized content from the '## Core Mechanisms' section, including cognitive patterns, emotional signatures, and behavioral tendencies.",
  "real_world_manifestation": "A single string containing all synthesized content from the '## Real-World Manifestation' section of the source text."
}}

**Mapping Rules:**
1.  Map the trait name from the `# Construct Name: {Trait Name}` line to `construct_name`.
2.  Map all text under `## Definition` to the single string field `definition`.
3.  Map **all** text under `## Core Mechanisms` (including all content from its sub-sections `### Cognitive Patterns`, `### Emotional Signatures`, and `### Behavioral Tendencies`) to the single string field `core_mechanisms`.
4.  Map **all** text under `## Real-World Manifestation` to the single string field `real_world_manifestation`.

**Constraints:**
1.  Your final output **must** be *only* the single, syntactically perfect JSON object.
2.  Do not include *any* explanatory text, acknowledgments, or pre-amble.
3.  Ensure all content from the source text is preserved perfectly within the correct JSON fields.

**[START_TEXT]**

{model_response}

**[END_TEXT]**"""

# prompt for scenario
scenario_sys_prompt = '''Role: You are a dual-specialist: an expert psychologist and creative screenwriter for scenario generation, and a rigorous narrative analyst for deconstruction. You excel at both creating vivid, human stories and then, in a separate step, precisely analyzing *why* they work.'''

gen_scenario_prompt = '''
Task: Your core mission is to take one human psychological or behavioral pattern I provide and first create a concise, analytical **Design Process**, **followed by** the detailed **scenario** that brings it to life and sets the necessary stage for the subsequent dialogue.
You will accomplish this by completing two distinct tasks in order: Task 1 (The Design Process) and Task 2 (The Scenario Execution).

Input Data:
1. Psychological/Behavioral Patterns: {pattern_information}
2. Situational Framework: {situation}
3. Candidate Names: {candidate_names} (5 Males, 5 Females)

[CRITICAL CONSTRAINT - NAMES]: You must select the Protagonist and all Supporting Characters（one or more) **STRICTLY** from the provided "Candidate Names" list. You determine which names to use based on the scenario needs, but you cannot invent new names.

You must follow the instructions below to complete the two tasks:

# Task 1: The Design Process (Analytical)

Adopt your role as the **"rigorous narrative analyst"**. You are planning the scenario.
Length: Extremely concise, **UNDER 500 TOKENS**.
Content:
1. **Design Rationale**: In a brief paragraph (1-2 sentences maximum), explain where each input pattern will be reflected in the scenario, pointing to the specific design choices (characters, setting, or events) that will embody it.

2. **Catalyst Details**: Using concise bullet points, identify the most critical details you will place in the scenario to act as 'catalysts'. For each bullet, briefly (1-2 sentences) explain its intended function.

3. **Expected Protagonist Tendencies**: List the protagonist's most likely cognitive or behavioral tendencies within this planned scenario, along with their key psychological conflicts. Provide a brief description for each tendency using the format: [Expectation Name] [Brief description].

# Task 2: The Scenario Execution (Creative)

Shift to your **"expert psychologist and creative screenwriter"** role. Execute the plan from Task 1.
Length: Detailed and comprehensive, **UNDER 1000 TOKENS**.
Purity: No self-analysis or author notes here.


## Requirement A: Story Background
Create a vivid setting description that acts as a complete "scene setup."
* **Core Elements (Setting)**: Depict time, place, setup, and atmosphere leading up to the core event.
* **Current Actions & Interaction Setup**: You must explicitly describe **what the characters are currently doing** (e.g., specific physical actions, work tasks, or non-verbal interactions) right before the conversation begins. Establish the immediate "status quo" and clarify the context so that all necessary information is known before the dialogue starts.
* **Absolute Constraint**: **Do not include any spoken dialogue in this section.** Keep the description purely narrative and observational to prepare the stage for the upcoming interaction.
* **Details**: Treat the {situation} framework flexibly. It can serve as the primary foundation or merely a minor element. You have full creative freedom to adapt the setting based on the Psychological/Behavioral Patterns to maximize the scenario's effectiveness, while adding specific cultural or environmental details to make it authentic and lived-in.

## Requirement B: Characters' Profiles (Multi-Dimensional)
For each selected character (Protagonist and Supporting Characters), you must structure their profile into two distinct parts:

1. **About Self (Objective/Full Profile)**:
   * Identity, social role, and core personality.
   * Key past experiences relevant to the current situation.
   * **Motivation**: Their specific goal or internal drive within this specific scenario.

2. **About Others (Subjective/Visible Profile)**:
    * For **EACH** other character present in the scene, provide a separate, specific description of the relationship from the current character's perspective.
    * **Format**: List each character by name (e.g., "**[Name]:** [Description]").
    * **Content**: Describe the current character's knowledge of the other character (e.g., their name, known identity, partial personality, or known past experiences). 
    * *Note: If they do not know a character, explicitly state that they are strangers.*

## Core Creative Mindset for Task 2
    * **Layer 1: Ensuring Compatibility**: The foundational goal is to create a context where the given pattern can emerge naturally. **The protagonist** should be "potentially susceptible" to the pattern, not "naturally immune." Your scenario should act as a "catalyst," not an "inhibitor." 
    * **Layer 2: Pursuing Richness (The Higher Goal)**: Beyond mere compatibility, the objective is to create three-dimensional characters, not simple archetypes. Characters serve the story, not the pattern. Be bold in giving them authentic human depth and complexity. 
    * **Key Concept ("Situational Susceptibility")**: A person with strong opinions might still conform under specific pressures; a decisive leader might rely on flawed heuristics when overwhelmed. This "situational susceptibility" is far more compelling and dramatically potent than a flat, one-note character. 
    * **Ultimate Goal**: Your aim is to create "the authentic reaction of a multi-dimensional person in a specific situation," not "the preset behavior of a one-dimensional character in a tailor-made scenario."
# Output Format

You must strictly follow the format below to structure your response. Do not alter the headers.

## Part 1
**Design Rationale**:
[Content here]

**Catalyst Details**:
* [Detail 1]: [Function]
* [Detail 2]: [Function]

**Expected Protagonist Tendencies**:
* [Expect1] [Description]
* [Expect2] [Description]

## Part 2
**Story Background**:
[Content here]

**Characters' Profiles**:

### Protagonist: [Name Selected from Input]
* **About Self**:
    [Full Profile + Past Experience + Motivation in this scenario]
* **About Others**:
    * **[Supporting Character 1 Name]**: [Identity, Relationship, known details, impressions...]
    * **[Supporting Character 2 Name]**: [Identity, Relationship, known details, impressions...] (if applicable)

### Supporting Character 1: [Name Selected from Input]
* **About Self**:
    [Full Profile + Past Experience + Motivation in this scenario]
* **About Others**:
    * **[Protagonist Name]**: [Identity, Relationship, known details, impressions...]
    * **[Supporting Character 2 Name]**: [Identity, Relationship, known details, impressions...] (if applicable)

### Supporting Character 2: [Name Selected from Input] (Optional, if applicable)
* **About Self**:
    [Full Profile + Past Experience + Motivation in this scenario]
* **About Others**:
    * **[Protagonist Name]**: [Identity, Relationship, known details, impressions...]
    * **[Supporting Character 1 Name]**: [Identity, Relationship, known details, impressions...] (if applicable)

(Continue for other characters if necessary)
'''.strip()

gen_scenario_prompt_no_situation = '''
Task: Your core mission is to take one human psychological or behavioral pattern I provide and first create a concise, analytical **Design Process**, **followed by** the detailed **scenario** that brings it to life and sets the necessary stage for the subsequent dialogue.
You will accomplish this by completing two distinct tasks in order: Task 1 (The Design Process) and Task 2 (The Scenario Execution).

Input Data:
1. Psychological/Behavioral Patterns: {pattern_information}
2. Candidate Names: {candidate_names} (5 Males, 5 Females)

[CRITICAL CONSTRAINT - NAMES]: You must select the Protagonist and all Supporting Characters（one or more) **STRICTLY** from the provided "Candidate Names" list. You determine which names to use based on the scenario needs, but you cannot invent new names.

You must follow the instructions below to complete the two tasks:

# Task 1: The Design Process (Analytical)

Adopt your role as the **"rigorous narrative analyst"**. You are planning the scenario.
Length: Extremely concise, **UNDER 500 TOKENS**.
Content:
1. **Design Rationale**: In a brief paragraph (1-2 sentences maximum), explain where each input pattern will be reflected in the scenario, pointing to the specific design choices (characters, setting, or events) that will embody it.

2. **Catalyst Details**: Using concise bullet points, identify the most critical details you will place in the scenario to act as 'catalysts'. For each bullet, briefly (1-2 sentences) explain its intended function.

3. **Expected Protagonist Tendencies**: List the protagonist's most likely cognitive or behavioral tendencies within this planned scenario, along with their key psychological conflicts. Provide a brief description for each tendency using the format: [Expectation Name] [Brief description].

# Task 2: The Scenario Execution (Creative)

Shift to your **"expert psychologist and creative screenwriter"** role. Execute the plan from Task 1.
Length: Detailed and comprehensive, **UNDER 1000 TOKENS**.
Purity: No self-analysis or author notes here.


## Requirement A: Story Background
Create a vivid setting description that acts as a complete "scene setup."
* **Core Elements (Setting)**: Depict time, place, setup, and atmosphere leading up to the core event.
* **Current Actions & Interaction Setup**: You must explicitly describe **what the characters are currently doing** (e.g., specific physical actions, work tasks, or non-verbal interactions) right before the conversation begins. Establish the immediate "status quo" and clarify the context so that all necessary information is known before the dialogue starts.
* **Absolute Constraint**: **Do not include any spoken dialogue in this section.** Keep the description purely narrative and observational to prepare the stage for the upcoming interaction.
* **Details**: Invent whichever situational framework best expresses the Psychological/Behavioral Patterns. You have full creative freedom to adapt or combine settings to maximize the scenario's effectiveness, while adding specific cultural or environmental details to make it authentic and lived-in.

## Requirement B: Characters' Profiles (Multi-Dimensional)
For each selected character (Protagonist and Supporting Characters), you must structure their profile into two distinct parts:

1. **About Self (Objective/Full Profile)**:
   * Identity, social role, and core personality.
   * Key past experiences relevant to the current situation.
   * **Motivation**: Their specific goal or internal drive within this specific scenario.

2. **About Others (Subjective/Visible Profile)**:
    * For **EACH** other character present in the scene, provide a separate, specific description of the relationship from the current character's perspective.
    * **Format**: List each character by name (e.g., "**[Name]:** [Description]").
    * **Content**: Describe the current character's knowledge of the other character (e.g., their name, known identity, partial personality, or known past experiences). 
    * *Note: If they do not know a character, explicitly state that they are strangers.*

## Core Creative Mindset for Task 2
    * **Layer 1: Ensuring Compatibility**: The foundational goal is to create a context where the given pattern can emerge naturally. **The protagonist** should be "potentially susceptible" to the pattern, not "naturally immune." Your scenario should act as a "catalyst," not an "inhibitor." 
    * **Layer 2: Pursuing Richness (The Higher Goal)**: Beyond mere compatibility, the objective is to create three-dimensional characters, not simple archetypes. Characters serve the story, not the pattern. Be bold in giving them authentic human depth and complexity. 
    * **Key Concept ("Situational Susceptibility")**: A person with strong opinions might still conform under specific pressures; a decisive leader might rely on flawed heuristics when overwhelmed. This "situational susceptibility" is far more compelling and dramatically potent than a flat, one-note character. 
    * **Ultimate Goal**: Your aim is to create "the authentic reaction of a multi-dimensional person in a specific situation," not "the preset behavior of a one-dimensional character in a tailor-made scenario."
# Output Format

You must strictly follow the format below to structure your response. Do not alter the headers.

## Part 1
**Design Rationale**:
[Content here]

**Catalyst Details**:
* [Detail 1]: [Function]
* [Detail 2]: [Function]

**Expected Protagonist Tendencies**:
* [Expect1] [Description]
* [Expect2] [Description]

## Part 2
**Story Background**:
[Content here]

**Characters' Profiles**:

### Protagonist: [Name Selected from Input]
* **About Self**:
    [Full Profile + Past Experience + Motivation in this scenario]
* **About Others**:
    * **[Supporting Character 1 Name]**: [Identity, Relationship, known details, impressions...]
    * **[Supporting Character 2 Name]**: [Identity, Relationship, known details, impressions...] (if applicable)

### Supporting Character 1: [Name Selected from Input]
* **About Self**:
    [Full Profile + Past Experience + Motivation in this scenario]
* **About Others**:
    * **[Protagonist Name]**: [Identity, Relationship, known details, impressions...]
    * **[Supporting Character 2 Name]**: [Identity, Relationship, known details, impressions...] (if applicable)

### Supporting Character 2: [Name Selected from Input] (Optional, if applicable)
* **About Self**:
    [Full Profile + Past Experience + Motivation in this scenario]
* **About Others**:
    * **[Protagonist Name]**: [Identity, Relationship, known details, impressions...]
    * **[Supporting Character 1 Name]**: [Identity, Relationship, known details, impressions...] (if applicable)

(Continue for other characters if necessary)
'''.strip()

gen_scenario_prompt_multi_patterns = '''
Task: Your core mission is to take 2-4 human psychological or behavioral patterns I provide and first create a concise, analytical **Design Process**, **followed by** the detailed **scenario** that brings it to life and sets the necessary stage for the subsequent dialogue.
You will accomplish this by completing two distinct tasks in order: Task 1 (The Design Process) and Task 2 (The Scenario Execution).

Input Data:
1. Psychological/Behavioral Patterns: {pattern_information}
2. Situational Framework: {situation}
3. Candidate Names: {candidate_names} (5 Males, 5 Females)

[CRITICAL CONSTRAINT - NAMES]: You must select the Protagonist and all Supporting Characters（one or more) **STRICTLY** from the provided "Candidate Names" list. You determine which names to use based on the scenario needs, but you cannot invent new names.

You must follow the instructions below to complete the two tasks:

# Task 1: The Design Process (Analytical)

Adopt your role as the **"rigorous narrative analyst"**. You are planning the scenario.
Length: Extremely concise, **UNDER 500 TOKENS**.
Content:
1. **Design Rationale**: In a brief paragraph (2-4 sentences maximum), explain where each input pattern will be reflected in the scenario, pointing to the specific design choices (characters, setting, or events) that will embody it.

2. **Catalyst Details**: Using concise bullet points, identify the most critical details you will place in the scenario to act as 'catalysts'. For each bullet, briefly (1-2 sentences) explain its intended function.

3. **Expected Character Tendencies**: For **ALL characters** (Protagonist and Supporting Characters), list their most likely cognitive or behavioral tendencies within this planned scenario, along with their key psychological conflicts.
   * **Format Requirement (STRICT)**: Use the following format for easy parsing:
```
     @ [Character Name]: 1. [Tendency1]; 2. [Tendency2]; 3. [Tendency3]
```
   * Each character must be on a **separate line**, starting with `@` followed by a space.
   * Character name must be enclosed in `[ ]`.
   * Tendencies must be **numbered** as `1. `, `2. `, `3. ` and separated by `; ` (semicolon + space).
   


# Task 2: The Scenario Execution (Creative)

Shift to your **"expert psychologist and creative screenwriter"** role. Execute the plan from Task 1.
Length: Detailed and comprehensive, **UNDER 1000 TOKENS**.
Purity: No self-analysis or author notes here.


## Requirement A: Story Background
Create a vivid setting description that acts as a complete "scene setup."
* **Core Elements (Setting)**: Depict time, place, setup, and atmosphere leading up to the core event.
* **Current Actions & Interaction Setup**: You must explicitly describe **what the characters are currently doing** (e.g., specific physical actions, work tasks, or non-verbal interactions) right before the conversation begins. Establish the immediate "status quo" and clarify the context so that all necessary information is known before the dialogue starts.
* **Absolute Constraint**: **Do not include any spoken dialogue in this section.** Keep the description purely narrative and observational to prepare the stage for the upcoming interaction.
* **Details**: Treat the {situation} framework flexibly. It can serve as the primary foundation or merely a minor element. 

## Requirement B: Characters' Profiles (Multi-Dimensional)
For each selected character (Protagonist and Supporting Characters), you must structure their profile into two distinct parts:

1. **About Self (Objective/Full Profile)**:
   * **Identity & Personality**: Social role and at least 4 distinct personality descriptors that together form a complete character.
   * **Relevant Background**: 1-2 sentences of past experience.
   * **Motivation**: Their specific goal or internal drive within this specific scenario.

2. **About Others (Subjective/Visible Profile)**:
    * For **EACH** other character present in the scene, provide a separate, specific description of the relationship from the current character's perspective.
    * **Format**: List each character by name (e.g., "**[Name]:** [Description]").
    * **Content**: Describe the current character's knowledge of the other character (e.g., their name, known identity, partial personality, or known past experiences). 
    * *Note: If they do not know a character, explicitly state that they are strangers.*

## Core Creative Mindset for Task 2
    * **Compatibility**: The foundational goal is to create a context where the given pattern can emerge naturally. 
    * **Situational Authenticity**: A person with strong opinions might still conform under specific pressures; a decisive leader might rely on flawed heuristics when overwhelmed. Design for authentic human reactions, not archetypal behaviors.
    * **Ultimate Goal**: Your aim is to create "the authentic reaction of a multi-dimensional person in a specific situation," not "the preset behavior of a one-dimensional character in a tailor-made scenario."
# Output Format

You must strictly follow the format below to structure your response. Do not alter the headers.

## Part 1
**Design Rationale**:
[Content here]

**Catalyst Details**:
* [Detail 1]: [Function]
* [Detail 2]: [Function]

**Expected Character Tendencies**: 
@ [Character Name 1]: 1. [Tendency1]; 2. [Tendency2]; 3. [Tendency3]
@ [Character Name 2]: 1. [Tendency1]; 2. [Tendency2]
@ [Character Name 3]: 1. [Tendency1]; 2. [Tendency2]; 3. [Tendency3]
(Continue for other characters if necessary)

## Part 2
**Story Background**:
[Content here]

**Characters' Profiles**:

### Protagonist: [Name Selected from Input]
* **About Self**:
    [Full Profile + Past Experience + Motivation in this scenario]
* **About Others**:
    * **[Supporting Character 1 Name]**: [Identity, Relationship, known details, impressions...]
    * **[Supporting Character 2 Name]**: [Identity, Relationship, known details, impressions...] (if applicable)

### Supporting Character 1: [Name Selected from Input]
* **About Self**:
    [Full Profile + Past Experience + Motivation in this scenario]
* **About Others**:
    * **[Protagonist Name]**: [Identity, Relationship, known details, impressions...]
    * **[Supporting Character 2 Name]**: [Identity, Relationship, known details, impressions...] (if applicable)

### Supporting Character 2: [Name Selected from Input] (Optional, if applicable)
* **About Self**:
    [Full Profile + Past Experience + Motivation in this scenario]
* **About Others**:
    * **[Protagonist Name]**: [Identity, Relationship, known details, impressions...]
    * **[Supporting Character 1 Name]**: [Identity, Relationship, known details, impressions...] (if applicable)

(Continue for other characters if necessary)
'''.strip()

gen_scenario_prompt_multi_patterns_no_situation = '''
Task: Your core mission is to take 2-4 human psychological or behavioral patterns I provide and first create a concise, analytical **Design Process**, **followed by** the detailed **scenario** that brings it to life and sets the necessary stage for the subsequent dialogue.
You will accomplish this by completing two distinct tasks in order: Task 1 (The Design Process) and Task 2 (The Scenario Execution).

Input Data:
1. Psychological/Behavioral Patterns: {pattern_information}
2. Candidate Names: {candidate_names} (5 Males, 5 Females)

[CRITICAL CONSTRAINT - NAMES]: You must select the Protagonist and all Supporting Characters（one or more) **STRICTLY** from the provided "Candidate Names" list. You determine which names to use based on the scenario needs, but you cannot invent new names.

You must follow the instructions below to complete the two tasks:

# Task 1: The Design Process (Analytical)

Adopt your role as the **"rigorous narrative analyst"**. You are planning the scenario.
Length: Extremely concise, **UNDER 500 TOKENS**.
Content:
1. **Design Rationale**: In a brief paragraph (2-4 sentences maximum), explain where each input pattern will be reflected in the scenario, pointing to the specific design choices (characters, setting, or events) that will embody it.

2. **Catalyst Details**: Using concise bullet points, identify the most critical details you will place in the scenario to act as 'catalysts'. For each bullet, briefly (1-2 sentences) explain its intended function.

3. **Expected Protagonist Tendencies**: List the protagonist's most likely cognitive or behavioral tendencies within this planned scenario, along with their key psychological conflicts. Provide a brief description for each tendency using the format: [Expectation Name] [Brief description].

# Task 2: The Scenario Execution (Creative)

Shift to your **"expert psychologist and creative screenwriter"** role. Execute the plan from Task 1.
Length: Detailed and comprehensive, **UNDER 1000 TOKENS**.
Purity: No self-analysis or author notes here.


## Requirement A: Story Background
Create a vivid setting description that acts as a complete "scene setup."
* **Core Elements (Setting)**: Depict time, place, setup, and atmosphere leading up to the core event.
* **Current Actions & Interaction Setup**: You must explicitly describe **what the characters are currently doing** (e.g., specific physical actions, work tasks, or non-verbal interactions) right before the conversation begins. Establish the immediate "status quo" and clarify the context so that all necessary information is known before the dialogue starts.
* **Absolute Constraint**: **Do not include any spoken dialogue in this section.** Keep the description purely narrative and observational to prepare the stage for the upcoming interaction.
* **Details**: Invent whichever situational framework best expresses the Psychological/Behavioral Patterns. You have full creative freedom to adapt or combine settings to maximize the scenario's effectiveness, while adding specific cultural or environmental details to make it authentic and lived-in.

## Requirement B: Characters' Profiles (Multi-Dimensional)
For each selected character (Protagonist and Supporting Characters), you must structure their profile into two distinct parts:

1. **About Self (Objective/Full Profile)**:
   * Identity, social role, and core personality.
   * Key past experiences relevant to the current situation.
   * **Motivation**: Their specific goal or internal drive within this specific scenario.

2. **About Others (Subjective/Visible Profile)**:
    * For **EACH** other character present in the scene, provide a separate, specific description of the relationship from the current character's perspective.
    * **Format**: List each character by name (e.g., "**[Name]:** [Description]").
    * **Content**: Describe the current character's knowledge of the other character (e.g., their name, known identity, partial personality, or known past experiences). 
    * *Note: If they do not know a character, explicitly state that they are strangers.*

## Core Creative Mindset for Task 2
    * **Layer 1: Ensuring Compatibility**: The foundational goal is to create a context where the given patterns can emerge naturally. **The protagonist** should be "potentially susceptible" to the pattern, not "naturally immune." Your scenario should act as a "catalyst," not an "inhibitor." 
    * **Layer 2: Pursuing Richness (The Higher Goal)**: Beyond mere compatibility, the objective is to create three-dimensional characters, not simple archetypes. Characters serve the story, not the pattern. Be bold in giving them authentic human depth and complexity. 
    * **Key Concept ("Situational Susceptibility")**: A person with strong opinions might still conform under specific pressures; a decisive leader might rely on flawed heuristics when overwhelmed. This "situational susceptibility" is far more compelling and dramatically potent than a flat, one-note character. 
    * **Ultimate Goal**: Your aim is to create "the authentic reaction of a multi-dimensional person in a specific situation," not "the preset behavior of a one-dimensional character in a tailor-made scenario."
# Output Format

You must strictly follow the format below to structure your response. Do not alter the headers.

## Part 1
**Design Rationale**:
[Content here]

**Catalyst Details**:
* [Detail 1]: [Function]
* [Detail 2]: [Function]

**Expected Protagonist Tendencies**:
* [Expect1] [Description]
* [Expect2] [Description]

## Part 2
**Story Background**:
[Content here]

**Characters' Profiles**:

### Protagonist: [Name Selected from Input]
* **About Self**:
    [Full Profile + Past Experience + Motivation in this scenario]
* **About Others**:
    * **[Supporting Character 1 Name]**: [Identity, Relationship, known details, impressions...]
    * **[Supporting Character 2 Name]**: [Identity, Relationship, known details, impressions...] (if applicable)

### Supporting Character 1: [Name Selected from Input]
* **About Self**:
    [Full Profile + Past Experience + Motivation in this scenario]
* **About Others**:
    * **[Protagonist Name]**: [Identity, Relationship, known details, impressions...]
    * **[Supporting Character 2 Name]**: [Identity, Relationship, known details, impressions...] (if applicable)

### Supporting Character 2: [Name Selected from Input] (Optional, if applicable)
* **About Self**:
    [Full Profile + Past Experience + Motivation in this scenario]
* **About Others**:
    * **[Protagonist Name]**: [Identity, Relationship, known details, impressions...]
    * **[Supporting Character 1 Name]**: [Identity, Relationship, known details, impressions...] (if applicable)

(Continue for other characters if necessary)
'''.strip()


# prompt for conversation
gen_conversationtion_sys_prompt = '''
**Role**: You are a master screenwriter and behavioral psychologist. Your expertise lies in bringing characters to life through nuanced dialogue and action, ensuring their **pivotal thoughts and resulting behaviors** in the dialogue are rooted in authentic psychological principles.
**Task**: Your mission is to take the provided psychological principles, a detailed scenario, and the accompanying design analysis (analysis), then write a multi-turn dialogue based on that scenario. This dialogue must, **at key moments**, vividly and concretely enact the specified principles through the characters' inner thoughts, spoken words, and physical actions.
'''
gen_conversationtion_prompt = '''

**You must follow this structure and these requirements:**

**Inputs:**

1.  **Principles**: `{pattern_information}`
2.  **Scenario**: `{scenario}`
3.  **Protagonist**: `{protagonist}`
4.  **Supporting Characters**: `{supporting_characters}`
5.  **Design Analysis**: `{analysis}`

**Output Requirements & Formatting:**

1.  **Content:** Create a multi-turn dialogue between the **Protagonist** and the **Supporting Characters**. **You must strictly limit the participants to the provided **Protagonist** and **Supporting Characters**; do not create or introduce any new characters.** The dialogue should contain **between 12 and 20 individual speaking turns** (each time a character speaks counts as one turn), ensuring there is sufficient developmental space to clearly showcase the principle in operation.
2.  **Mandatory Flow (Start & End)**:
    * **Opener**: The dialogue **must begin** with a turn from one of the **Supporting Characters** ({supporting_characters}). Do not let the Protagonist speak first.
    * **Closer**: The dialogue **must conclude** with a turn from the **Protagonist** ({protagonist}).
3.  Turn Structure (Important): The dialogue must strictly follow a turn-based format. One character must completely finish their turn (including any thoughts, dialogue, and actions) before the next character begins. Strictly prohibit any instances of characters interrupting each other or having overlapping speech. For example, a character's dialogue must not end with an em dash (—) indicating they were cut off by the next speaker.
4.  **Trinity of Expression**: Seamlessly integrate **inner thought, external action, and spoken dialogue** throughout the conversation. Ensure these three elements are fused naturally to form a consistent, non-contradictory behavioral whole.
5.  **Strict Formatting Rules**:
    * **Inner thoughts/psychology**: Use `[square brackets]`.
    * **Actions/expressions/behaviors**: Use `(parentheses)`.
    * **Spoken dialogue**: Use no brackets.
    * **Formatting Example**: `Hermione: [I have to devise a foolproof plan.] (She quickly draws her wand, pointing it at the door) Harry, use the flute, now!`
6.  **No Preamble**: Do not **begin** the response with any introductory text, preambles, or explanations (e.g., "Here is the dialogue...", "according to your detailed instructions").
**Core Creative Principles:**

1.  **Focus and Breathing Room**: This is the most crucial principle. You do **not** need to have every minor gesture or piece of small talk carry the weight of a psychological principle. Use the principles as a "**spotlight**" to illuminate and explain **the most critical turning points, the core conflicts, or the moments that best define the characters' arcs**. Other routine, functional dialogue and actions (like greetings or pouring water) should exist naturally, creating "breathing room" for these key moments and making the manifestation of the principles more prominent and powerful.
2.  **Show, Don't Tell**: Never allow characters to openly state or explain the psychological principles by name. Instead, you must **show** how the principles influence their judgment and choices through their concrete actions (the combination of thoughts, dialogue, and physical behavior).
3.  **Psychology Drives Action**: In the key moments illuminated by the "spotlight," the character's `[inner thought]`should be the origin of their behavior, directly reflecting the influence of a psychological principle. The subsequent dialogue and `(actions)` should be the logical, external expression of that internal state.
4.  **Seamless Integration**: Weave the principles into the natural flow of the story. The entire dialogue should feel like an authentic interaction, not a contrived demonstration for a psychology case study.

'''

format_scenario_prompt = '''
**Role**: You are a meticulous data structuring specialist. Your expertise is in parsing unstructured, natural language text and converting it into perfectly formatted, machine-readable JSON based on a strict schema.

**Task**: Your primary task is to read the provided `Scenario Text` and the associated `Principles`, then accurately map their contents to a predefined JSON structure. This conversion is crucial for data processing and for systematically feeding the scenario into subsequent creative tools. Your output must be nothing but a single, valid JSON object.

**Inputs:**

1.  **`Principles`**: A list of one or more psychological principles that the scenario is based on.
    `[{principle1}]`

2.  **`Scenario Text`**: The full, unstructured prose of the scenario, including the background and character descriptions.
    `[{scenario}]`

-----

**Instructions & Required JSON Schema:**

You must parse the `Scenario Text` and populate the following JSON structure. Use the exact key names and data types as specified below. The comments in the schema are your guide for mapping the text to the correct fields.

```json
{{
  "principles": [
    // An array of strings. Populate this with the list from the `Principles` input.
    "string"
  ],
  "story_background": "string", // Extract the complete, flowing description of the story background, including the setting, atmosphere, and the current situation the characters are facing, into a single paragraph.
  "character_profiles": {{
    "protagonist": {{
      "name": "string", // Extract the protagonist's name from the text.
      "profile": "string" // Extract the complete, flowing description of the protagonist, merging their identity, personality, past experiences, and other rich details into a single, cohesive paragraph.
    }},
    "other_characters": [
      // This MUST be an array of objects, one for each non-protagonist character mentioned.
      {{
        "name": "string", // Extract the character's name.
        "identity_and_personality": "string", // Extract the description of this character's role and multi-faceted personality.
        "relationship_to_protagonist": "string" // Extract the description of their relationship with the protagonist.
      }}
    ]
  }}
}}
```

-----

**Core Rules:**

1.  **Strict Schema Adherence**: You **must** use the exact key names (e.g., `story_background`, `protagonist`, `profile`) and the nested structure shown in the schema. Do not add, remove, or rename any keys. Pay close attention to data types (string, object, array).
2.  **Extract, Don't Invent**: Your output must **only** contain information extracted directly from the provided `Scenario Text`. Do not add any new creative content, summaries, or interpretations. Your role is to parse, not to write.
3.  **Handle Missing Information**: If the source text does not provide clear information for a specific field, use an empty string `""` as its value. **Do not omit the key.** This ensures structural consistency.
4.  **Valid JSON Output**: The final output must be a single, complete, and syntactically correct JSON object that can be immediately parsed by a computer. Ensure all brackets `{}` `[]` are closed, commas are placed correctly, and all strings are properly quoted and escaped.
'''


gen_conversationtion_sys_prompt_no_analysis = '''
**Role**: You are a master screenwriter and behavioral psychologist. Your expertise lies in bringing characters to life through nuanced dialogue and action, ensuring their **pivotal thoughts and resulting behaviors** in the dialogue are rooted in authentic psychological principles.
**Task**: Your mission is to take the provided psychological principles and a detailed scenario, then write a multi-turn dialogue. This dialogue must, **at key moments**, vividly and concretely enact the specified principles through the characters' inner thoughts, spoken words, and physical actions.
'''
gen_conversationtion_prompt_no_analysis = '''

**You must follow this structure and these requirements:**

**Inputs:**

1. **Principles**: `{pattern_information}`
2. **Scenario**: `{scenario}`

**Output Requirements & Formatting:**

1. **Content:** Create a multi-turn dialogue between the protagonist and other characters. The dialogue should contain at least 10 individual speaking turns (each time a character speaks counts as one turn), ensuring there is sufficient developmental space to clearly showcase the principle in operation.
2. Turn Structure (Important): The dialogue must strictly follow a turn-based format. One character must completely finish their turn (including any thoughts, dialogue, and actions) before the next character begins. Strictly prohibit any instances of characters interrupting each other or having overlapping speech.
3. **Trinity of Expression**: At the key moments that reveal the principles, you should integrate the three layers of **inner thought, spoken dialogue, and external action** to form a complete, cohesive moment of behavior.
4. **Strict Formatting Rules**:

- **Inner thoughts/psychology**: Use `[square brackets]`.
- **Actions/expressions/behaviors**: Use `(parentheses)`.
- **Spoken dialogue**: Use no brackets.
- **Formatting Example**: `Hermione: [I have to devise a foolproof plan.] (She quickly draws her wand, pointing it at the door) Harry, use the flute, now!`
5.  **No Preamble**: Do not **begin** the response with any introductory text, preambles, or explanations (e.g., "Here is the dialogue...", "according to your detailed instructions").

**Core Creative Principles:**

1. **Focus and Breathing Room**: This is the most crucial principle. You do **not** need to have every minor gesture or piece of small talk carry the weight of a psychological principle. Use the principles as a "**spotlight**" to illuminate and explain **the most critical turning points, the core conflicts, or the moments that best define the characters' arcs**. Other routine, functional dialogue and actions (like greetings or pouring water) should exist naturally, creating "breathing room" for these key moments and making the manifestation of the principles more prominent and powerful.
2. **Show, Don't Tell**: Never allow characters to openly state or explain the psychological principles by name. Instead, you must **show** how the principles influence their judgment and choices through their concrete actions (the combination of thoughts, dialogue, and physical behavior).
3. **Psychology Drives Action**: In the key moments illuminated by the "spotlight," the character's `[inner thought]`should be the origin of their behavior, directly reflecting the influence of a psychological principle. The subsequent dialogue and `(actions)` should be the logical, external expression of that internal state.
4. **Seamless Integration**: Weave the principles into the natural flow of the story. The entire dialogue should feel like an authentic interaction, not a contrived demonstration for a psychology case study.


'''
# prompt for protagonist name extraction
characters_prompt_sys = '''You are a rigorous text extraction assistant. Your task is to analyze a scenario description and extract character names into a structured JSON format.

### Extraction Rules
1. **Protagonist**: Identify the name of the *first* character who is fully introduced in the text.
2. **Supporting Characters**: Identify the names of *all other* characters introduced. There may be one or more supporting characters.
   - **CRITICAL REQUIREMENT**: Only include characters referred to by a **specific proper name** (e.g., "Sarah", "Mr. Jones").
   - **EXCLUDE**: Do not include unnamed characters referred to only by their role, relationship, title, or occupation (e.g., exclude "his wife", "the judges", "a stranger").

### Output Format
Return strictly a valid JSON object. Do not use Markdown code blocks (like ```json).

### Example
**Input Scenario:**
"Marcus Chen had been attending this festival since childhood. He waved at Sarah, the volunteer coordinator, saw David working on his pottery, and nodded to the judges."

**Expected Output:**
{
    "protagonist": "Marcus Chen",
    "supporting_characters": ["Sarah", "David"]
}
'''
characters_prompt = '''Please extract the protagonist and supporting characters from the following scenario:

{scenario}'''

# prompt for checlist
gen_checklist_sys_prompt = '''You are an AI assistant specialized in analyzing psychological principles in narratives. Your task is to generate a list of conversation evaluation items based on the analysis and scenario provided by the user.

# Task: Generate Conversation Evaluation Items

You will generate a list of evaluation items for a "Conversation" based on the provided 【Pattern Information】, 【Story Background】 and 【Expected Character Tendencies】.

# Generation Steps
1.  The evaluation items are intended to assess whether the characters' actions, statements, and psychology in the "Conversation" (which takes place under the 【Story Background】) align with the 【Pattern Information】.
2.  The specific generation logic is: Based on the 【Pattern Information】 and the specific details in the 【Story Background】, you must transform the content of the 【Expected Character Tendencies】. into evaluation items for the "Conversation".
3.  The source for the evaluation items must be **strictly based on** the  the user-provided 【Expected Character Tendencies】.

# Output Requirements
* Your output format must be a numbered list (1. ... 2. ... 3. ...).
* The evaluation items must be **brief and concise**.
* The items should first list all checkpoints based on "Catalyst Details", followed by all checkpoints based on "Expected Protagonist Tendencies".
* The items must use specific names and events from the 【Scenario】 (e.g., character names, specific actions, system names) to make the content concrete.

Wait for the user to provide the three inputs, then begin the task immediately.'''

gen_checklist_prompt = '''
# 【Pattern Information】
{pattern_information}

#【Story Background】
{story_background}

# 【Expected Character Tendencies】
{expected_character_tendencies}
'''
