#!/usr/bin/env python3
"""
Convert generated conversation data into ShareGPT-style records without using an LLM.

The script relies on the existing `protagonist` and `supporting_characters`
fields to parse the `conversation` text with regex rules, then structures the
result so that:
  * The ShareGPT conversation alternates strictly between "human" and "gpt".
  * The first turn is always "human".
  * The final turn is always "gpt".

Example usage:
    python gen_sharegpt_dataset.py \
        --input-files Dataset/generated_data_with_factors.json \
        Dataset/generated_data_multi_patterns_with_factors.json \
        --output Dataset/sft_data/sharegpt_dataset.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

DEFAULT_INPUT_FILES = [
    "Dataset/generated_data_with_factors.json",
    "Dataset/generated_data_multi_patterns_with_factors.json",
]
DEFAULT_OUTPUT_PATH = "Dataset/sft_data/sharegpt_dataset.json"

SPEAKER_LINE_RE = re.compile(
    r"^\s*([^\n:]{1,100}?)\s*:\s*(.*)$", re.UNICODE)
NAME_CLEAN_RE = re.compile(r"[^a-z0-9]+", re.IGNORECASE)
NAME_STOPWORDS = {
    "dr", "mr", "mrs", "ms", "miss", "sir", "madam", "madame",
    "prof", "professor", "coach", "capt", "captain", "doctor",
}


def load_dataset(path: Path) -> List[Dict]:
    if not path.exists():
        print(f"[warn] Input file missing: {path}")
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"[warn] Failed to parse {path}: {exc}")
        return []
    if not isinstance(data, list):
        print(f"[warn] Dataset {path} is not a list; skipped.")
        return []
    return data


def parse_conversation(raw: Any) -> List[Tuple[str, str]]:
    """Return a list of (speaker, utterance) tuples parsed from raw text or list."""
    # If already structured as a list of dicts, normalize to tuples.
    if isinstance(raw, list):
        turns: List[Tuple[str, str]] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            speaker = (item.get("char") or item.get("speaker") or "").strip()
            content = item.get("content") or item.get("text") or ""
            if not isinstance(content, str):
                content = str(content)
            if speaker or content:
                turns.append((speaker, content.strip()))
        return turns

    text = (raw or "").strip()
    if not text:
        return []

    lines = text.splitlines()
    turns: List[Tuple[str, str]] = []
    current_speaker: Optional[str] = None
    buffer: List[str] = []

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
            flush()
            current_speaker = match.group(1).strip()
            buffer = [match.group(2)]
        else:
            if current_speaker is not None:
                buffer.append(line)

    flush()
    return turns


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


def _speaker_matches(speaker: str, aliases: Sequence[str]) -> bool:
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


def _strip_inner_thoughts(text: str) -> str:
    return re.sub(r"\[[^\[\]]*\]", "", text)


def build_sharegpt_turns(
        turns: Sequence[Tuple[str, str]],
        protagonist_name: str,
        supporting_names: Sequence[str]) -> List[Dict[str, str]]:
    """
    Group supporting-character speech between protagonist turns.
    Ensures the conversation starts with "human" and ends with "gpt".
    """
    protagonist_aliases = _build_aliases(protagonist_name)
    supporting_aliases = [_build_aliases(name)
                          for name in supporting_names if name]

    def classify_speaker(name: str) -> str:
        if _speaker_matches(name, protagonist_aliases):
            return "protagonist"
        for aliases in supporting_aliases:
            if _speaker_matches(name, aliases):
                return "support"
        return "support"

    share_turns: List[Dict[str, str]] = []
    support_buffer: List[str] = []

    def flush_support() -> None:
        nonlocal support_buffer
        if not support_buffer:
            return
        content = "\n\n".join(
            line for line in support_buffer if line.strip()).strip()
        support_buffer = []
        if not content:
            return
        if share_turns and share_turns[-1]["from"] == "human":
            joiner = "\n\n" if share_turns[-1]["value"] else ""
            share_turns[-1]["value"] = f"{share_turns[-1]['value']}{joiner}{content}"
        else:
            share_turns.append({"from": "human", "value": content})

    for speaker, content in turns:
        message = f"{speaker}: {content}".strip(
        ) if content else speaker.strip()
        if not message:
            continue
        speaker_type = classify_speaker(speaker)
        if speaker_type == "protagonist":
            flush_support()
            if not share_turns:
                # ensure the conversation starts with a human turn
                share_turns.append({"from": "human", "value": ""})
            if share_turns and share_turns[-1]["from"] == "gpt":
                joiner = "\n\n" if share_turns[-1]["value"] else ""
                share_turns[-1]["value"] = f"{share_turns[-1]['value']}{joiner}{message}"
            else:
                share_turns.append({"from": "gpt", "value": message})
        else:
            cleaned = _strip_inner_thoughts(message).strip()
            if cleaned:
                support_buffer.append(cleaned)

    # discard trailing supporting speech (should not exist if protagonist closes)
    support_buffer = []

    if not share_turns:
        return []
    if share_turns[0]["from"] != "human":
        share_turns.insert(0, {"from": "human", "value": ""})
    # drop trailing non-gpt turns to satisfy ending constraint
    while share_turns and share_turns[-1]["from"] != "gpt":
        share_turns.pop()

    cleaned: List[Dict[str, str]] = []
    for turn in share_turns:
        if cleaned and cleaned[-1]["from"] == turn["from"]:
            joiner = "\n\n" if cleaned[-1]["value"] else ""
            cleaned[-1]["value"] = f"{cleaned[-1]['value']}{joiner}{turn['value']}"
        else:
            cleaned.append(turn)

    if len(cleaned) < 2 or cleaned[0]["from"] != "human" or cleaned[-1]["from"] != "gpt":
        return []
    return cleaned


def iter_input_entries(files: Iterable[Path]) -> Iterable[Tuple[str, Dict]]:
    for path in files:
        dataset = load_dataset(path)
        if not dataset:
            continue
        for entry in dataset:
            yield str(path), entry


def _build_support_profile_map(profiles: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    mapping: Dict[str, Dict[str, Any]] = {}
    for profile in profiles:
        name = profile.get("name")
        if not name:
            continue
        normalized = _normalize_name(name)
        if not normalized:
            continue
        mapping.setdefault(normalized, profile)
    return mapping


def build_character_infos(entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    scenario = entry.get("scenario") or {}
    characters = scenario.get("charactersProfiles") or {}
    protagonist_profile = characters.get("protagonist") or {}
    protagonist_name = (
        protagonist_profile.get("name")
        or entry.get("protagonist")
        or ""
    )

    infos: List[Dict[str, Any]] = []
    if protagonist_name:
        infos.append(
            {
                "name": protagonist_name,
                "profile": protagonist_profile,
                "role": "protagonist",
            }
        )

    supporting_profiles = characters.get("supportingCharacter") or []
    profile_map = _build_support_profile_map(supporting_profiles)
    supporting_names = entry.get("supporting_characters")
    if not supporting_names:
        supporting_names = [
            profile.get("name")
            for profile in supporting_profiles
            if profile.get("name")
        ]
    for name in supporting_names or []:
        clean_name = (name or "").strip()
        if not clean_name:
            continue
        profile = profile_map.get(_normalize_name(clean_name), {})
        infos.append(
            {
                "name": clean_name,
                "profile": profile,
                "role": "supporting",
            }
        )
    return infos


def build_system_prompt(entry: Dict[str, Any], character_info: Dict[str, Any]) -> str:
    scenario = entry.get("scenario") or {}
    profile = character_info.get("profile") or {}
    protagonist_name = character_info.get("name") or "Protagonist"
    about_self = profile.get("aboutSelf") or ""
    about_others = profile.get("aboutOthers") or ""
    story_background = scenario.get("storyBackground") or ""

    prompt = (
        f"You are {protagonist_name}.\n"
        f"==About {protagonist_name}==\n"
        f"{about_self}\n"
        f"=={protagonist_name}'s Perception of Others==\n"
        f"{about_others}\n"
        f"==Current Scenario==\n"
        f"{story_background}\n\n"
        "==Requirements==\n"
        "Your output should include **thought**, **speech**, and **action**.\n"
        "- Use [...] for inner thoughts, which others can't see.\n"
        "- Use (...) for physical actions or expressions, which others can see.\n"
        "- Write speech directly without special markers.\n\n"
        f"Think, act and speak as {protagonist_name}. Stay in character and respond\n"
        "naturally based on your personality and the situation."
    )
    return prompt.strip()


def build_sharegpt_record(
    entry: Dict[str, Any],
    character_info: Dict[str, Any],
    supporting_names: Sequence[str],
    conversations: List[Dict[str, str]],
) -> Dict[str, Any]:
    return {
        "system": build_system_prompt(entry, character_info),
        "conversations": conversations,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert generated dataset into ShareGPT format via regex parsing."
    )
    parser.add_argument(
        "--input-files",
        nargs="+",
        default=DEFAULT_INPUT_FILES,
        help="List of dataset JSON files to convert.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_PATH,
        help="Output JSON path for the ShareGPT dataset.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N valid samples across all inputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_paths = [Path(p) for p in args.input_files]
    output_path = Path(args.output)
    os.makedirs(output_path.parent, exist_ok=True)

    sharegpt_records: List[Dict[str, Any]] = []
    processed = 0

    for source_path, entry in iter_input_entries(input_paths):
        conversation_raw = entry.get("conversation")
        turns = parse_conversation(conversation_raw)
        if not turns:
            continue

        character_infos = build_character_infos(entry)
        if not character_infos:
            continue

        for character_info in character_infos:
            lead_name = (character_info.get("name") or "").strip()
            if not lead_name:
                continue
            supporting_names = [
                (info.get("name") or "").strip()
                for info in character_infos
                if info is not character_info and (info.get("name") or "").strip()
            ]
            share_turns = build_sharegpt_turns(
                turns, lead_name, supporting_names)
            if not share_turns:
                continue

            record = build_sharegpt_record(
                entry, character_info, supporting_names, share_turns
            )
            sharegpt_records.append(record)
            processed += 1

            if args.limit is not None and processed >= args.limit:
                break

        if args.limit is not None and processed >= args.limit:
            break

    output_path.write_text(
        json.dumps(sharegpt_records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"[done] Saved {len(sharegpt_records)} ShareGPT items to {output_path} "
        f"(from {len(input_paths)} input file(s))."
    )


if __name__ == "__main__":
    main()
