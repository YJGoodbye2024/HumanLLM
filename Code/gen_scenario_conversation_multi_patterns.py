import argparse
import asyncio
import copy
import json
import os
import random
import re
import time
from collections import deque
from typing import Any, Dict, List, Tuple, Optional, Sequence

from names_dataset import NameDataset
from openai import AsyncOpenAI  # Use the asynchronous client
from tqdm.asyncio import tqdm_asyncio  # For asynchronous progress bars

# Make sure prompt_all.py and principle_situaton.py are in the same directory or accessible via PYTHONPATH
from principle_situaton import sd_pri_list, td_pri_list_100, Situation_list
from prompt_all import (
    scenario_sys_prompt,
    gen_scenario_prompt_multi_patterns,
    gen_scenario_prompt_multi_patterns_no_situation,
    gen_conversationtion_sys_prompt,
    gen_conversationtion_prompt,
    characters_prompt_sys,
    characters_prompt,
)
JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```",
                           re.IGNORECASE | re.DOTALL)
PROTAGONIST_HEADING_RE = re.compile(
    r"^###\s*Protagonist:\s*(.+)$", re.IGNORECASE)
SUPPORTING_HEADING_RE = re.compile(
    r"^###\s*Supporting Character[^:]*:\s*(.+)$", re.IGNORECASE)
SPEAKER_LINE_RE = re.compile(r"^\s*([^\n:]{1,100}?)\s*:\s*(.*)$", re.UNICODE)
NAME_CLEAN_RE = re.compile(r"[^a-z0-9]+", re.IGNORECASE)
NAME_STOPWORDS = {
    "dr", "mr", "mrs", "ms", "miss", "sir", "madam", "madame",
    "prof", "professor", "coach", "capt", "captain", "doctor",
}


NAME_DATASET: Optional[NameDataset] = None
NAME_POOLS_READY = False
MALE_NAME_POOL: List[str] = []
FEMALE_NAME_POOL: List[str] = []


# --- Pattern Compatibility Check Prompts ---
COMPATIBILITY_SYS_PROMPT = '''
Role: You are a behavioral analyst and expert psychologist specializing in personality theory. Your expertise lies in assessing the compatibility of different psychological traits and behavioral patterns.
'''.strip()

COMPATIBILITY_USER_PROMPT = '''
Task: Your sole task is to analyze the provided list of psychological or behavioral patterns for their internal compatibility. You must determine if these patterns can believably co-exist as core, defining traits within a single individual.

Input (JSON array of trait/behavior words):
{patterns_json}

Definition of "clear contradiction":
- If two or more patterns denote stable and directly opposing dispositional tendencies within the same general context (e.g., "quarrelsome" vs "placid"), such that they are unlikely to co-exist as core, persistent traits of one person at the same time (ignoring temporary state fluctuations), mark as contradictory.
- Otherwise (e.g., complementary, orthogonal, context-dependent traits), mark as not contradictory.

Example:
- ["quarrelsome","placid"] -> 是   (clear contradiction)
- ["perceptive","placid"] -> 否    (no clear contradiction)

Output:
Return ONLY one character:
- "是" if the set contains a clear contradiction,
- "否" if no clear contradiction exists.

Do not add any explanations or additional text.
'''.strip()


# --- Configuration ---
CLAUDE_MODEL = "claude-sonnet-4-5-20250929"
# CLAUDE_MODEL = "gemini-3-pro-preview"

GPT_MODEL = "gpt-5"
GEMINI_MODEL = "[G]gemini-2.5-pro"
GEMINI_REQUESTS_PER_MINUTE = 45

# --- File Paths ---
PATTERNS_INFO_FILE = 'Dataset/patterns_info/psy_patterns_info.json'
OUTPUT_FILE = 'Dataset/generated_data_multi_patterns.json'
FAILURES_FILE = 'Dataset/gen_scenario_conversation_multi_failures.json'

# --- Concurrency Control ---
# 控制同时发送的API请求数量，根据你的API速率限制调整
# A good starting point is between 10 and 50.
MAX_CONCURRENT_REQUESTS = 20

# --- Retry / Timeout ---
REQUEST_TIMEOUT_SECONDS = 300
MAX_RETRY_ATTEMPTS = 5
RETRY_BACKOFF_SECONDS = 2

# --- Sampling Weights ---
# 每次从 Big-5 里随机选 2-3 个维度；在每个维度内，正极/负极的选择概率如下：
TD_POSITIVE_PROB = 0.6
TD_NEGATIVE_PROB = 0.4
# 以该概率附加 1 个 SD pattern
SD_PROBABILITY = 0.7

TD_DIMENSION_BUCKETS = [
    td_pri_list_100[0:20],
    td_pri_list_100[20:40],
    td_pri_list_100[40:60],
    td_pri_list_100[60:80],
    td_pri_list_100[80:100],
]


class PatternCoverageTracker:
    """Tracks how often each pattern has been used to encourage full coverage."""

    def __init__(self, initial_patterns: Optional[Sequence[str]] = None) -> None:
        self._counts: Dict[str, int] = {}
        if initial_patterns:
            for pattern in initial_patterns:
                self._counts[pattern] = 0

    def _ensure(self, pattern: str) -> None:
        if pattern not in self._counts:
            self._counts[pattern] = 0

    def pick_least_used(self, candidates: Sequence[str], rng: random.Random) -> str:
        if not candidates:
            raise ValueError("No candidates provided for coverage selection.")
        for pattern in candidates:
            self._ensure(pattern)
        min_usage = min(self._counts[p] for p in candidates)
        pool = [p for p in candidates if self._counts[p] == min_usage]
        choice = rng.choice(pool)
        self._counts[choice] += 1
        return choice


TD_PATTERN_COVERAGE = PatternCoverageTracker(td_pri_list_100)
SD_PATTERN_COVERAGE = PatternCoverageTracker(sd_pri_list)

# --- Functions (modified for async) ---


class ModelRunner:
    def __init__(
        self,
        name: str,
        client: AsyncOpenAI,
        model_name: str,
        rate_limit_per_minute: Optional[int] = None,
    ) -> None:
        self.name = name
        self.client = client
        self.model_name = model_name
        self.rate_limit_per_minute = rate_limit_per_minute or 0
        self._rate_lock = asyncio.Lock() if self.rate_limit_per_minute else None
        self._request_timestamps: deque[float] = deque()


async def _respect_rate_limit(runner: ModelRunner) -> None:
    if not runner.rate_limit_per_minute or not runner._rate_lock:
        return

    window_seconds = 60.0
    while True:
        async with runner._rate_lock:
            now = time.monotonic()
            timestamps = runner._request_timestamps
            while timestamps and now - timestamps[0] >= window_seconds:
                timestamps.popleft()

            if len(timestamps) < runner.rate_limit_per_minute:
                timestamps.append(now)
                return

            wait_for = window_seconds - (now - timestamps[0])
        await asyncio.sleep(max(wait_for, 0.01))


async def get_model_answer_async(runner: ModelRunner, sys_prompt: str = "", user_prompt: str = ""):
    """Asynchronous function to get a response from a chat model."""
    for attempt in range(1, MAX_RETRY_ATTEMPTS + 1):
        try:
            await _respect_rate_limit(runner)
            response = await asyncio.wait_for(
                runner.client.chat.completions.create(
                    model=runner.model_name,
                    stream=False,
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": user_prompt},
                    ]
                ),
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            return response.choices[0].message.content
        except asyncio.TimeoutError:
            print(
                f"[{runner.name}] Request timed out (attempt {attempt}/{MAX_RETRY_ATTEMPTS}).")
        except Exception as e:
            print(
                f"[{runner.name}] API call failed on attempt {attempt}/{MAX_RETRY_ATTEMPTS}: {e}")

        if attempt < MAX_RETRY_ATTEMPTS:
            backoff = RETRY_BACKOFF_SECONDS * attempt
            await asyncio.sleep(backoff)

    return None


def load_patterns_info(file_path: str) -> Dict[str, Dict]:
    """Loads principle details from the consolidated patterns info file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return {}
    except json.JSONDecodeError:
        print(f"Error: The file {file_path} contains invalid JSON.")
        return {}


def _format_patterns_information(patterns_map: Dict[str, Dict], pattern_names: List[str]) -> str:
    info_list = []
    for name in pattern_names:
        data = patterns_map.get(name)
        if not data:
            raise KeyError(f"Pattern '{name}' not found in patterns info.")
        entry = dict(data)
        entry.setdefault("construct_name", name)
        info_list.append(entry)
    return json.dumps(info_list, ensure_ascii=False, indent=2)


def _sample_td_bundle(rng: random.Random) -> List[str]:
    return [rng.choice(bucket) for bucket in TD_DIMENSION_BUCKETS]


def _sample_td_patterns_weighted(rng: random.Random) -> List[str]:
    """
    Sample 3-4 TD patterns by:
    - Randomly choosing 3 or 4 distinct Big-5 dimensions,
    - For each chosen dimension: choose positive (p=0.6) or negative (p=0.4),
      then randomly pick 1 pattern from the corresponding half of the bucket.
    """
    # Choose how many dimensions to sample: 3 or 4
    k = rng.choice([3, 4])
    dims = rng.sample(range(len(TD_DIMENSION_BUCKETS)), k)
    selected: List[str] = []
    for d in dims:
        bucket = TD_DIMENSION_BUCKETS[d]
        # First 10 are positive, last 10 are negative by construction
        if rng.random() < TD_POSITIVE_PROB:
            # positive
            choice_candidates = bucket[:10]
        else:
            # negative
            choice_candidates = bucket[10:20]
        choice = TD_PATTERN_COVERAGE.pick_least_used(choice_candidates, rng)
        selected.append(choice)
    return selected


def build_pattern_bundle(
    patterns_map: Dict[str, Dict],
    rng: random.Random,
) -> Dict[str, Any]:
    # Sample only TD patterns (3-4) with positive/negative weighting
    td_selected = _sample_td_patterns_weighted(rng)
    include_sd = rng.random() < SD_PROBABILITY
    selected = list(td_selected)
    if include_sd:
        selected.append(SD_PATTERN_COVERAGE.pick_least_used(sd_pri_list, rng))
    info_text = _format_patterns_information(patterns_map, selected)
    construct_name = " + ".join(selected)
    td_count = len(td_selected)
    return {
        "construct_name": construct_name,
        "patterns": selected,
        "principles_information": info_text,
        "bundle_type": f"td_{td_count}_sd" if include_sd else f"td_{td_count}",
    }


async def _vet_bundle_compatibility(
    bundle: Dict[str, Any],
    runner: ModelRunner,
    semaphore: asyncio.Semaphore,
) -> Optional[Dict[str, Any]]:
    """
    Call Gemini to decide whether this pattern bundle contains clear contradictions.
    Returns the bundle if compatible (否), otherwise None.
    """
    patterns = bundle.get("patterns") or []
    if not patterns:
        return None
    patterns_json = json.dumps(patterns, ensure_ascii=False)
    user_prompt = COMPATIBILITY_USER_PROMPT.format(patterns_json=patterns_json)
    # print(f"prompt为：{user_prompt}")
    async with semaphore:
        reply = await get_model_answer_async(runner, COMPATIBILITY_SYS_PROMPT, user_prompt)
    if not reply:
        return None
    ans = reply.strip().strip('"').strip()
    # Compatible if model says "否" (no contradiction)
    if ans in {"否", "No", "NO", "no"}:
        return bundle
    return None


def ensure_principles_information(
    entry: Dict[str, Any],
    patterns_map: Dict[str, Dict],
) -> str:
    info = entry.get("principles_information")
    if info:
        return info
    pattern_names = entry.get("patterns") or []
    if not pattern_names:
        raise ValueError(
            "Existing entry lacks both 'principles_information' and 'patterns'.")
    return _format_patterns_information(patterns_map, pattern_names)


def gather_gemini_credentials() -> List[Tuple[str, str]]:
    """Collect a single BASE_URL/API key pair for Gemini (API_KEY_LIMIT)."""
    base_url = os.getenv("BASE_URL_LIMIT")
    api_key = os.getenv("API_KEY_LIMIT")
    if not base_url:
        raise RuntimeError("BASE_URL_LIMIT environment variable is not set.")
    if not api_key:
        raise RuntimeError("API_KEY_LIMIT environment variable is not set.")
    return [(base_url.rstrip("/"), api_key)]


def load_failure_entries(file_path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(file_path):
        return []
    try:
        with open(file_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[warn] 无法读取失败样本文件 {file_path}: {exc}")
        return []
    if not isinstance(data, list):
        print(f"[warn] 失败样本文件 {file_path} 格式无效，忽略。")
        return []
    entries: List[Dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        principle = item.get("principle")
        situation = item.get("situation")
        if isinstance(principle, dict) and isinstance(situation, str) and situation.strip():
            entries.append({
                "principle": principle,
                "situation": situation.strip(),
            })
    return entries


def save_failure_entries(entries: List[Dict[str, Any]], file_path: str) -> None:
    if not entries:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except OSError as exc:
            print(f"[warn] 无法删除失败样本文件 {file_path}: {exc}")
        return
    try:
        with open(file_path, "w", encoding="utf-8") as handle:
            json.dump(entries, handle, ensure_ascii=False, indent=2)
    except OSError as exc:
        print(f"[warn] 无法写入失败样本文件 {file_path}: {exc}")


def _prepare_name_pools() -> None:
    global NAME_DATASET, NAME_POOLS_READY, MALE_NAME_POOL, FEMALE_NAME_POOL
    if NAME_POOLS_READY:
        return
    NAME_DATASET = NAME_DATASET or NameDataset()
    male_pool: List[str] = []
    female_pool: List[str] = []
    for country_code in ["US", "GB", "CA", "IE"]:
        male_data = NAME_DATASET.get_top_names(
            n=50000, gender="Male", country_alpha2=country_code)
        male_pool.extend(male_data.get(country_code, {}).get("M", []))
        female_data = NAME_DATASET.get_top_names(
            n=50000, gender="Female", country_alpha2=country_code)
        female_pool.extend(female_data.get(country_code, {}).get("F", []))
    male_pool = list(dict.fromkeys(male_pool))
    female_pool = list(dict.fromkeys(female_pool))
    if len(male_pool) < 5 or len(female_pool) < 5:
        raise RuntimeError(
            "Insufficient English names from NameDataset; ensure dataset is available.")
    MALE_NAME_POOL = male_pool
    FEMALE_NAME_POOL = female_pool
    NAME_POOLS_READY = True


def generate_candidate_names() -> str:
    _prepare_name_pools()
    males = random.sample(MALE_NAME_POOL, 5)
    females = random.sample(FEMALE_NAME_POOL, 5)
    payload = {"Male": males, "Female": females}
    return json.dumps(payload, ensure_ascii=False)


def extract_characters_from_scenario(text: str) -> Tuple[Optional[str], List[str]]:
    protagonist: Optional[str] = None
    supporting: List[str] = []

    def _clean(name: str) -> str:
        cleaned = re.sub(r"[\*\[\]_`]+", "", name).strip()
        cleaned = re.sub(r"\(.*?\)$", "", cleaned).strip()
        return cleaned

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        match_pro = PROTAGONIST_HEADING_RE.match(stripped)
        if match_pro and not protagonist:
            protagonist = _clean(match_pro.group(1))
            continue
        match_sup = SUPPORTING_HEADING_RE.match(stripped)
        if match_sup:
            candidate = _clean(match_sup.group(1))
            if candidate:
                supporting.append(candidate)
    return protagonist, supporting


def _normalize_name(value: str) -> str:
    return NAME_CLEAN_RE.sub(" ", value or "").strip().lower()


def _name_tokens(value: str) -> List[str]:
    tokens = [token for token in _normalize_name(value).split() if token]
    return [token for token in tokens if token not in NAME_STOPWORDS]


def _build_aliases(value: str) -> List[str]:
    tokens = _name_tokens(value)
    aliases: List[str] = []
    if tokens:
        aliases.append(" ".join(tokens))
        aliases.extend(tokens)
    return aliases


def _speaker_matches(speaker: str, aliases: List[str]) -> bool:
    if not aliases:
        return False
    speaker_tokens = _name_tokens(speaker)
    if not speaker_tokens:
        return False
    speaker_joined = " ".join(speaker_tokens)
    for alias in aliases:
        if not alias:
            continue
        if alias == speaker_joined or alias in speaker_tokens:
            return True
    return False


def _resolve_speaker(speaker: str, alias_map: List[Tuple[str, List[str]]]) -> Optional[str]:
    for canonical, aliases in alias_map:
        if _speaker_matches(speaker, aliases):
            return canonical
    return None


def _collect_character_names(protagonist: str, supporting_characters: List[str]) -> List[str]:
    names: List[str] = []
    protagonist_clean = (protagonist or "").strip()
    if protagonist_clean:
        names.append(protagonist_clean)
    for name in supporting_characters or []:
        clean = (name or "").strip()
        if clean:
            names.append(clean)
    seen = set()
    unique: List[str] = []
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        unique.append(name)
    return unique


def _build_alias_map(names: List[str]) -> List[Tuple[str, List[str]]]:
    return [(name, _build_aliases(name)) for name in names]


def parse_conversation(text: str, alias_map: List[Tuple[str, List[str]]]) -> List[Tuple[str, str]]:
    """
    Parse raw conversation text into (canonical_speaker, utterance) tuples.
    A new turn starts only when the line's speaker matches known names/aliases.
    """
    lines = text.splitlines()
    turns: List[Tuple[str, str]] = []
    current_speaker: Optional[str] = None
    buffer: List[str] = []
    preamble: List[str] = []

    def flush() -> None:
        nonlocal buffer, current_speaker
        if current_speaker is None:
            return
        utterance = "\n".join(buffer).strip()
        turns.append((current_speaker, utterance))
        buffer = []
        current_speaker = None

    for line in lines:
        match = SPEAKER_LINE_RE.match(line)
        if match:
            candidate = match.group(1).strip()
            canonical = _resolve_speaker(candidate, alias_map)
            if canonical:
                flush()
                current_speaker = canonical
                combined_buffer: List[str] = []
                if preamble:
                    combined_buffer.extend(preamble)
                    preamble = []
                combined_buffer.append(match.group(2))
                buffer = combined_buffer
                continue
        if current_speaker is not None:
            buffer.append(line)
        else:
            preamble.append(line)

    flush()
    return turns


def convert_conversation_to_list(
    conversation: Any,
    protagonist: str,
    supporting_characters: List[str],
) -> List[Dict[str, str]]:
    """
    Convert raw conversation text into a list of {"char","content"} dicts.
    Falls back to an empty list when parsing yields no segments.
    """
    if isinstance(conversation, list):
        return conversation
    conversation_text = conversation or ""
    if not isinstance(conversation_text, str):
        conversation_text = str(conversation_text)

    names = _collect_character_names(protagonist, supporting_characters)
    alias_map = _build_alias_map(names) if names else []
    segments: List[Dict[str, str]] = []
    for speaker, content in parse_conversation(conversation_text, alias_map):
        speaker_clean = speaker.strip()
        content_clean = content.strip()
        if not speaker_clean and not content_clean:
            continue
        segments.append({"char": speaker_clean, "content": content_clean})
    return segments


def _sanitize_json_blob(raw: str) -> str:
    if not isinstance(raw, str):
        return raw
    text = raw.strip()
    match = JSON_FENCE_RE.search(text)
    if match:
        return match.group(1).strip()
    if text.startswith("```") and text.endswith("```"):
        return text.strip("`").strip()
    return text


def _clean_section_text(text: str) -> str:
    """Trim whitespace and drop common markdown separators from a section."""
    separator_lines = {"---", "***", "___"}
    lines = text.splitlines()
    start = 0
    end = len(lines)

    while start < end and not lines[start].strip():
        start += 1
    while end > start and not lines[end - 1].strip():
        end -= 1

    while start < end and lines[start].strip() in separator_lines:
        start += 1
        while start < end and not lines[start].strip():
            start += 1

    while end > start and lines[end - 1].strip() in separator_lines:
        end -= 1
        while end > start and not lines[end - 1].strip():
            end -= 1

    return "\n".join(lines[start:end]).strip()


def structure_scenario_text(text: str) -> Dict[str, Any]:
    story_match = re.search(
        r"\*\*\s*Story Background\s*\*\*\s*:?\s*(.*?)(?=\*\*\s*Characters' Profiles\s*\*\*\s*:?\s*|\Z)",
        text,
        re.S | re.IGNORECASE,
    )
    characters_section_match = re.search(
        r"\*\*\s*Characters' Profiles\s*\*\*\s*:?\s*(.*)", text, re.S | re.IGNORECASE)

    if story_match:
        story_background_raw = story_match.group(1)
    elif characters_section_match:
        story_background_raw = text[:characters_section_match.start()]
    else:
        story_background_raw = text

    story_background = _clean_section_text(story_background_raw)
    characters_section = characters_section_match.group(
        1) if characters_section_match else ""

    protagonist_profile = {"name": "", "aboutSelf": "", "aboutOthers": ""}
    supporting_profiles: List[Dict[str, str]] = []

    block_pattern = re.compile(
        r"###\s*(Protagonist|Supporting Character[^:]*):\s*(.+?)(?=\n###\s*(?:Protagonist|Supporting Character)|\Z)",
        re.S,
    )
    for block in block_pattern.finditer(characters_section):
        role = block.group(1).strip()
        body = block.group(2)
        name_match = re.match(r"([^\n]+)\n?(.*)", body, re.S)
        name = name_match.group(1).strip() if name_match else ""
        remainder = name_match.group(2) if name_match else ""
        about_self_match = re.search(
            r"\*\*\s*About Self\s*\*\*:\s*(.+?)(?=\*\*\s*About Others\s*\*\*|\Z)",
            remainder,
            re.S,
        )
        about_others_match = re.search(
            r"\*\*\s*About Others\s*\*\*:\s*(.+)", remainder, re.S)
        about_self = _clean_section_text(
            about_self_match.group(1)) if about_self_match else ""
        about_others = _clean_section_text(
            about_others_match.group(1)) if about_others_match else ""

        profile = {
            "name": name,
            "aboutSelf": about_self,
            "aboutOthers": about_others,
        }
        if role.lower().startswith("protagonist") and not protagonist_profile["name"]:
            protagonist_profile = profile
        else:
            supporting_profiles.append(profile)

    return {
        "storyBackground": story_background,
        "charactersProfiles": {
            "protagonist": protagonist_profile,
            "supportingCharacter": supporting_profiles,
        },
    }


def format_structured_scenario(data: Dict[str, Any]) -> str:
    lines: List[str] = []
    story_background = data.get("storyBackground", "")
    characters_profiles = data.get("charactersProfiles", {}) or {}
    protagonist = characters_profiles.get("protagonist", {}) or {}
    supporting = characters_profiles.get("supportingCharacter", []) or []

    lines.append("**Story Background**")
    lines.append(story_background.strip())
    lines.append("")
    lines.append("**Characters' Profiles**")

    if protagonist.get("name"):
        lines.append(f"### Protagonist: {protagonist.get('name', '').strip()}")
        lines.append("* **About Self**:")
        lines.append(protagonist.get("aboutSelf", "").strip())
        lines.append("* **About Others**:")
        lines.append(protagonist.get("aboutOthers", "").strip())
        lines.append("")

    for idx, profile in enumerate(supporting, start=1):
        name = profile.get("name", "").strip()
        heading = f"### Supporting Character {idx}: {name}" if name else f"### Supporting Character {idx}:"
        lines.append(heading)
        lines.append("* **About Self**:")
        lines.append(profile.get("aboutSelf", "").strip())
        lines.append("* **About Others**:")
        lines.append(profile.get("aboutOthers", "").strip())
        lines.append("")

    return "\n".join(lines).strip()


def split_part_sections(content: str) -> Tuple[str, str]:
    """
    Split model output into two parts and return (scenario, analysis).
    Assumes current prompt order: Part 1 = Analysis, Part 2 = Scenario.
    Falls back to returning the full content as the scenario if split fails.
    """
    heading_regex = re.compile(
        r'(?im)^\s*(?:[#>*-]+\s*)?(?:\*\*|__)?\s*Part\s*(\d)\s*[:\-–—.]*.*?$',
        re.MULTILINE,
    )
    matches = list(heading_regex.finditer(content))

    part1_match = next((m for m in matches if m.group(1) == "1"), None)
    part2_match = next((m for m in matches if m.group(1) == "2"), None)

    if not part1_match or not part2_match or part2_match.start() <= part1_match.end():
        print("[warn] Unable to detect distinct Part 1/Part 2 sections; storing full output under 'scenario'.")
        return content.strip(), ""

    analysis_raw = content[part1_match.end():part2_match.start()]
    scenario_raw = content[part2_match.end():]

    analysis = _clean_section_text(analysis_raw)
    scenario = _clean_section_text(scenario_raw)

    # Fallbacks if a section is unexpectedly empty
    if not analysis:
        analysis = _clean_section_text(
            content[part1_match.start():part2_match.start()])
    if not scenario:
        scenario = _clean_section_text(content[part2_match.start():])

    return scenario, analysis


async def process_single_combination(
        principle,
        situation,
        scenario_runner: ModelRunner,
        conversation_runner: Optional[ModelRunner],
        protagonist_runner: Optional[ModelRunner],
        semaphore,
        scenario_override: Optional[str] = None,
        analysis_override: Optional[str] = None,
        conversation_sys_prompt_override: Optional[str] = None,
        conversation_prompt_override: Optional[str] = None,
        scenario_prompt_override: Optional[str] = None):
    """
    Processes a single principle-situation pair asynchronously.
    This function combines scenario and conversation generation.
    """
    async with semaphore:  # Acquire a semaphore slot
        convo_runner = conversation_runner or scenario_runner
        if scenario_override is None:
            # Step 1: Generate Scenario
            candidate_names_text = generate_candidate_names()
            scenario_template = scenario_prompt_override or gen_scenario_prompt_multi_patterns
            scenario_prompt = scenario_template.format(
                pattern_information=principle["principles_information"],
                situation=situation,
                candidate_names=candidate_names_text,
            )
            scenario_content_raw = await get_model_answer_async(scenario_runner, scenario_sys_prompt, scenario_prompt)
            if not scenario_content_raw:
                print(
                    f"[{scenario_runner.name}] Failed to generate scenario for '{principle['construct_name']}' with '{situation}'.")
                return None

            scenario_content, analysis_content = split_part_sections(
                scenario_content_raw)
            scenario_struct = structure_scenario_text(scenario_content)
            scenario_text_for_prompt = scenario_content
        else:
            scenario_struct = scenario_override if isinstance(
                scenario_override, dict) else structure_scenario_text(scenario_override)
            scenario_text_for_prompt = format_structured_scenario(
                scenario_struct) if isinstance(
                scenario_override, dict) else scenario_override
            analysis_content = analysis_override or ""
            if not scenario_text_for_prompt:
                print(
                    f"[warn] Empty scenario provided for '{principle['construct_name']}' with '{situation}'. Skipping conversation regeneration.")
                return None

        scenario_content = scenario_text_for_prompt
        name_runner = protagonist_runner or scenario_runner
        protagonist_name, supporting_characters = extract_characters_from_scenario(
            scenario_content)
        role_success = bool(protagonist_name)

        if not role_success:
            protagonist_prompt = characters_prompt.format(
                scenario=scenario_content)
            protagonist_name = ""
            supporting_characters = []
            for attempt in range(1, MAX_RETRY_ATTEMPTS + 1):
                roles_raw = await get_model_answer_async(
                    name_runner, characters_prompt_sys, protagonist_prompt)
                if roles_raw:
                    try:
                        cleaned_roles_raw = _sanitize_json_blob(roles_raw)
                        roles_data = json.loads(cleaned_roles_raw)
                    except json.JSONDecodeError:
                        roles_data = {}
                    protagonist_name = (roles_data.get(
                        "protagonist") or "").strip()
                    supp = roles_data.get("supporting_characters")
                    supporting_characters = []
                    if isinstance(supp, list):
                        supporting_characters = [
                            str(item).strip()
                            for item in supp
                            if isinstance(item, str) and item.strip()
                        ]
                    if protagonist_name:
                        role_success = True
                        break
                await asyncio.sleep(RETRY_BACKOFF_SECONDS * attempt)

            if not role_success:
                print(
                    f"[warn] Failed to extract protagonist/supporting characters for '{principle['construct_name']}' with '{situation}'.")
                return None

        protagonist_name = protagonist_name.splitlines()[0].strip()

        conversation_sys = conversation_sys_prompt_override or gen_conversationtion_sys_prompt
        conversation_template = conversation_prompt_override or gen_conversationtion_prompt

        prompt_kwargs = {
            "pattern_information": principle["principles_information"],
            "scenario": scenario_content,
            "protagonist": protagonist_name or "Unknown",
            "supporting_characters": ", ".join(supporting_characters) if supporting_characters else "None",
        }
        prompt_kwargs["analysis"] = analysis_content
        conversation_prompt = conversation_template.format(**prompt_kwargs)
        conversation_content = await get_model_answer_async(
            convo_runner, conversation_sys, conversation_prompt)
        if not conversation_content:
            print(
                f"[{convo_runner.name}] Failed to generate conversation for '{principle['construct_name']}'.")
            return None

        # Step 3: Structure and return data
        result_pattern = principle.get("patterns")
        if result_pattern and len(result_pattern) == 1:
            result_pattern = result_pattern[0]
        elif not result_pattern:
            result_pattern = principle['construct_name']

        conversation_segments = convert_conversation_to_list(
            conversation_content, protagonist_name, supporting_characters)

        return {
            "pattern": result_pattern,
            "situation": situation,
            "analysis": analysis_content,  # ensure analysis before scenario
            "scenario": scenario_struct,
            "protagonist": protagonist_name,
            "supporting_characters": supporting_characters,
            "conversation": conversation_segments
        }


def save_data_to_json(data, file_path):
    """Saves the generated data list to a JSON file."""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"\nData successfully saved to {file_path}")
    except IOError as e:
        print(f"An error occurred while writing to the file: {e}")


async def main(
    limit: Optional[int] = None,
    situation_limit: Optional[int] = None,
    only_conversation: bool = False,
    conversation_limit: Optional[int] = None,
    sd_probability: float = 0.7,
    random_seed: Optional[int] = None,
    retry_failures: bool = False,
    no_situation: bool = False,
    only_claude: bool = False,
) -> None:
    start_time = time.time()

    # Load principle data from consolidated patterns info file
    patterns_info = load_patterns_info(PATTERNS_INFO_FILE)
    if not patterns_info:
        print("No principle information loaded. Exiting.")
        return

    if retry_failures and only_conversation:
        print("[error] --retry-failures 仅适用于完整生成流程，不能与 --only-conversation 同时使用。")
        return

    sd_probability = min(max(sd_probability, 0.0), 1.0)
    rng = random.Random(random_seed)

    base_url_full = os.getenv("BASE_URL_FULL")
    api_key_full = os.getenv("API_KEY_FULL")
    if not base_url_full:
        raise RuntimeError("BASE_URL_FULL environment variable is not set.")
    if not api_key_full:
        raise RuntimeError("API_KEY_FULL environment variable is not set.")

    claude_runner = ModelRunner(
        "claude",
        AsyncOpenAI(api_key=api_key_full, base_url=base_url_full.rstrip("/")),
        CLAUDE_MODEL,
    )
    gpt_runner = ModelRunner(
        "gpt",
        AsyncOpenAI(api_key=api_key_full, base_url=base_url_full.rstrip("/")),
        GPT_MODEL,
    )

    protagonist_runner = gpt_runner

    # Create a semaphore to limit concurrency
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    # Create a list of all tasks to be run
    tasks = []
    claude_assignments = 0
    gpt_assignments = 0
    success_count = 0
    task_entries: List[Tuple[int, Dict]] = []
    existing_data: List[Dict] = []
    final_dataset: List[Dict]
    successful_results: List[Dict] = []

    if only_conversation:
        if not os.path.exists(OUTPUT_FILE):
            print(
                f"[error] Dataset file '{OUTPUT_FILE}' 不存在，无法重新生成对话。")
            return
        try:
            with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)
                if not isinstance(loaded_data, list):
                    raise ValueError("Existing dataset is not a list.")
                existing_data = loaded_data
        except (json.JSONDecodeError, ValueError) as exc:
            print(f"[error] 无法读取已有数据集：{exc}")
            return

        final_dataset = list(existing_data)
        total_entries = len(existing_data)
        if total_entries == 0:
            print("[info] 数据集为空，无需重新生成对话。")
            return

        process_count = total_entries
        if conversation_limit is not None and conversation_limit > 0:
            process_count = min(conversation_limit, total_entries)
        elif limit is not None and limit > 0:
            process_count = min(limit, total_entries)
        if process_count < total_entries:
            print(
                f"[info] 仅重新生成前 {process_count} 条样本的对话（共 {total_entries} 条）。")

        convo_sys_prompt = gen_conversationtion_sys_prompt
        convo_prompt_template = gen_conversationtion_prompt

        for idx in range(process_count):
            entry = existing_data[idx]
            patterns_list = entry.get("patterns") or []
            try:
                principles_information = ensure_principles_information(
                    entry, patterns_info)
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[warn] 无法重建 pattern 信息（{exc}），保留原对话。")
                continue

            construct_name = entry.get("pattern") or " + ".join(patterns_list)
            principle_entry = {
                "construct_name": construct_name,
                "patterns": patterns_list,
                "bundle_type": entry.get("bundle_type"),
                "principles_information": principles_information,
            }

            situation = entry.get("situation", "")
            scenario_text = entry.get("scenario", "")
            analysis_text = entry.get("analysis", "")

            task_index = len(tasks)
            if only_claude or task_index % 2 == 0:
                scenario_runner = claude_runner
                claude_assignments += 1
            else:
                scenario_runner = gpt_runner
                gpt_assignments += 1
            conversation_runner = claude_runner

            tasks.append(process_single_combination(
                principle_entry,
                situation,
                scenario_runner,
                conversation_runner,
                protagonist_runner,
                semaphore,
                scenario_override=scenario_text,
                analysis_override=analysis_text,
                conversation_sys_prompt_override=convo_sys_prompt,
                conversation_prompt_override=convo_prompt_template,
            ))
            task_entries.append((idx, entry))

        if not tasks:
            print("[warn] 没有任何样本进入重新生成流程。")
            return

        print(f"Created {len(tasks)} conversation regeneration tasks. Starting concurrent processing with a limit of {MAX_CONCURRENT_REQUESTS} requests...")
        print(
            f"Task allocation -> Claude: {claude_assignments}, GPT: {gpt_assignments}")

        results = await tqdm_asyncio.gather(*tasks)

        for (idx, original_entry), res in zip(task_entries, results):
            if res is None:
                continue
            # Rebuild entry to enforce key order with analysis before scenario
            updated_entry = {
                "pattern": res["pattern"],
                "situation": res["situation"],
                "analysis": res["analysis"],
                "scenario": res["scenario"],
                "protagonist": res["protagonist"],
                "supporting_characters": res.get("supporting_characters", []),
                "conversation": res["conversation"],
            }
            final_dataset[idx] = updated_entry
            success_count += 1

        successful_results = final_dataset
        total_tasks = len(tasks)
    else:
        combination_specs: List[Dict[str, Any]] = []
        if retry_failures:
            failure_specs = load_failure_entries(FAILURES_FILE)
            if not failure_specs:
                print("[info] 没有需要重试的失败样本。")
                return
            for spec in failure_specs:
                principle = spec.get("principle")
                situation = spec.get("situation")
                if isinstance(principle, dict) and isinstance(situation, str):
                    combination_specs.append({
                        "principle": copy.deepcopy(principle),
                        "situation": situation,
                        "scenario_template": gen_scenario_prompt_multi_patterns_no_situation if not situation else gen_scenario_prompt_multi_patterns,
                    })
            if not combination_specs:
                print("[info] 失败样本文件中没有有效条目。")
                return
            print(f"[info] 将重试 {len(combination_specs)} 条失败样本。")
        else:
            bundle_count = len(
                sd_pri_list) if not limit or limit <= 0 else limit
            pattern_bundles: List[Dict[str, Any]] = []
            for _ in range(bundle_count):
                try:
                    bundle = build_pattern_bundle(
                        patterns_info, rng)
                except KeyError as exc:
                    print(f"[warn] {exc}; skipping this bundle.")
                    continue
                pattern_bundles.append(bundle)

            if not pattern_bundles:
                print("[error] 无法生成任何 pattern 组合，退出。")
                return
            print(
                f"[info] Prepared {len(pattern_bundles)} multi-pattern bundles (td_count=3–4, pos:neg={TD_POSITIVE_PROB}:{TD_NEGATIVE_PROB}, sd_p={SD_PROBABILITY}).")

            # Vet pattern compatibility using GPT; only keep non-contradictory bundles
            vet_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
            vet_tasks = []
            for b in pattern_bundles:
                vet_tasks.append(_vet_bundle_compatibility(
                    b, gpt_runner, vet_semaphore))
            vet_results = await tqdm_asyncio.gather(*vet_tasks)
            compatible_bundles = [b for b in vet_results if b is not None]
            print(
                f"[info] {len(compatible_bundles)}/{len(pattern_bundles)} bundles passed compatibility vetting.")
            if not compatible_bundles:
                print("[error] 所有模式组合均判定为存在明显矛盾，退出。")
                return

            selected_situations = list(Situation_list)
            if situation_limit is not None:
                if situation_limit > 0:
                    k = min(situation_limit, len(selected_situations))
                    selected_situations = rng.sample(selected_situations, k)
                    print(
                        f"[info] Randomly sampled {len(selected_situations)} situations for testing.")
                else:
                    selected_situations = []

            for bundle in compatible_bundles:
                for situation in selected_situations:
                    combination_specs.append({
                        "principle": copy.deepcopy(bundle),
                        "situation": situation,
                        "scenario_template": gen_scenario_prompt_multi_patterns,
                    })
                if no_situation:
                    combination_specs.append({
                        "principle": copy.deepcopy(bundle),
                        "situation": "",
                        "scenario_template": gen_scenario_prompt_multi_patterns_no_situation,
                    })
            if no_situation:
                print(
                    "[info] Added one no-situation scenario per pattern bundle in addition to situation-based samples.")

        if not combination_specs:
            print("[warn] 没有可处理的样本组合。")
            return

        for idx, spec in enumerate(combination_specs):
            principle = spec["principle"]
            situation = spec["situation"]
            scenario_template = spec.get(
                "scenario_template", gen_scenario_prompt_multi_patterns)
            if only_claude or idx % 2 == 0:
                scenario_runner = claude_runner
                claude_assignments += 1
            else:
                scenario_runner = gpt_runner
                gpt_assignments += 1
            conversation_runner = claude_runner
            tasks.append(process_single_combination(
                principle,
                situation,
                scenario_runner,
                conversation_runner,
                protagonist_runner,
                semaphore,
                scenario_prompt_override=scenario_template,
            ))

        print(
            f"Created {len(tasks)} tasks. Starting concurrent processing with a limit of {MAX_CONCURRENT_REQUESTS} requests...")
        print(
            f"Task allocation -> Claude: {claude_assignments}, GPT: {gpt_assignments}")

        results = await tqdm_asyncio.gather(*tasks)

        failed_specs: List[Dict[str, Any]] = []
        for spec, res in zip(combination_specs, results):
            if res is None:
                failed_specs.append({
                    "principle": copy.deepcopy(spec["principle"]),
                    "situation": spec["situation"],
                })
            else:
                successful_results.append(res)

        success_count = len(successful_results)
        total_tasks = len(combination_specs)
        save_failure_entries(failed_specs, FAILURES_FILE)

    # Save the successful results
    if successful_results:
        if (not only_conversation) and retry_failures:
            existing_records: List[Dict[str, Any]] = []
            if os.path.exists(OUTPUT_FILE):
                try:
                    with open(OUTPUT_FILE, "r", encoding="utf-8") as handle:
                        loaded = json.load(handle)
                        if isinstance(loaded, list):
                            existing_records = loaded
                        else:
                            print(
                                f"[warn] Existing dataset {OUTPUT_FILE} 不是数组，重写为仅包含追加结果。")
                except (OSError, json.JSONDecodeError) as exc:
                    print(
                        f"[warn] 无法读取现有数据集 {OUTPUT_FILE}，将只写入新结果：{exc}")
            merged = existing_records + successful_results
            save_data_to_json(merged, OUTPUT_FILE)
        else:
            save_data_to_json(successful_results, OUTPUT_FILE)
    else:
        print("No data was generated successfully.")

    end_time = time.time()
    print(f"\nScript finished in {end_time - start_time:.2f} seconds.")
    print(
        f"Successfully generated {success_count} out of {total_tasks} total items.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate scenarios and conversations based on principle summaries."
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Only sample the first N multi-pattern bundles (testing helper).",
    )
    parser.add_argument(
        "--situation-limit",
        type=int,
        help="Randomly sample M situations from Situation_list (testing helper).",
    )
    parser.add_argument(
        "--only-conversation",
        action="store_true",
        help="只重新生成已有数据集中的对话，不重新生成场景。",
    )
    parser.add_argument(
        "--conversation-limit",
        type=int,
        help="在 --only-conversation 模式下，仅重写前 N 条样本的对话。",
    )
    parser.add_argument(
        "--retry-failures",
        action="store_true",
        help="仅对上一次记录的失败样本重新生成，并将成功结果追加到已有数据中。",
    )
    parser.add_argument(
        "--sd-probability",
        type=float,
        default=0.5,
        help="Probability of adding one sd_pri_list pattern (0-1, default 0.5).",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        help="Optional random seed to make pattern sampling reproducible.",
    )
    parser.add_argument(
        "--no_situation",
        action="store_true",
        help="使用不含 Situation 描述的 Prompt 来生成所有场景。",
    )
    parser.add_argument(
        "--only-claude",
        action="store_true",
        help="仅使用 Claude 生成场景（对话始终使用 Claude）。",
    )
    args = parser.parse_args()

    try:
        asyncio.run(
            main(
                limit=args.limit,
                situation_limit=args.situation_limit,
                only_conversation=args.only_conversation,
                conversation_limit=args.conversation_limit,
                sd_probability=args.sd_probability,
                random_seed=args.random_seed,
                retry_failures=args.retry_failures,
                no_situation=args.no_situation,
                only_claude=args.only_claude,
            ))
    except KeyboardInterrupt:
        print("\nScript interrupted by user. Exiting.")
