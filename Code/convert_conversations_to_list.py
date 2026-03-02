#!/usr/bin/env python3
"""
Convert the `conversation` field in `generated_data_with_factors.json` from a
string into a list of `{char, content}` dictionaries. The parsing logic mirrors
`gen_sharegpt_dataset.py`: each line that looks like "<speaker>: ..." starts a
new turn, and subsequent lines belong to that speaker until the next match.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

DEFAULT_INPUT_PATH = "Dataset/generated_data_with_factors.json"
DEFAULT_OUTPUT_PATH = "Dataset/generated_data_with_factors_list.json"

SPEAKER_LINE_RE = re.compile(
    r"^\s*([^\n:]{1,100}?)\s*:\s*(.*)$", re.UNICODE
)
NAME_CLEAN_RE = re.compile(r"[^a-z0-9]+", re.IGNORECASE)
NAME_STOPWORDS = {
    "dr", "mr", "mrs", "ms", "miss", "sir", "madam", "madame",
    "prof", "professor", "coach", "capt", "captain", "doctor",
}


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


def _resolve_speaker(
    speaker: str,
    alias_map: List[Tuple[str, List[str]]],
) -> str | None:
    for canonical, aliases in alias_map:
        if _speaker_matches(speaker, aliases):
            return canonical
    return None


def _collect_character_names(entry: Dict[str, Any]) -> List[str]:
    names: List[str] = []
    protagonist = (entry.get("protagonist") or "").strip()
    if protagonist:
        names.append(protagonist)
    for name in entry.get("supporting_characters") or []:
        clean = (name or "").strip()
        if clean:
            names.append(clean)
    # keep order but deduplicate
    seen = set()
    unique: List[str] = []
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        unique.append(name)
    return unique


def _build_alias_map(names: Sequence[str]) -> List[Tuple[str, List[str]]]:
    alias_map: List[Tuple[str, List[str]]] = []
    for name in names:
        alias_map.append((name, _build_aliases(name)))
    return alias_map


def parse_conversation(
    text: str,
    alias_map: List[Tuple[str, List[str]]],
) -> List[Tuple[str, str]]:
    """
    Parse raw conversation text into (canonical_speaker, utterance) tuples.
    A new turn starts only when the line's speaker matches known names/aliases.
    """
    lines = text.splitlines()
    turns: List[Tuple[str, str]] = []
    current_speaker: str | None = None
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
                combined_buffer = []
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


def convert_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of the entry with a structured `conversation` list."""
    if isinstance(entry.get("conversation"), list):
        return dict(entry)

    names = _collect_character_names(entry)
    if not names:
        return dict(entry)

    conversation_text = entry.get("conversation") or ""
    if not isinstance(conversation_text, str):
        conversation_text = str(conversation_text)

    alias_map = _build_alias_map(names)

    segments = []
    for speaker, content in parse_conversation(conversation_text, alias_map):
        speaker_clean = speaker.strip()
        content_clean = content.strip()
        if not speaker_clean and not content_clean:
            continue
        segments.append({"char": speaker_clean, "content": content_clean})

    updated = dict(entry)
    updated["conversation"] = segments
    return updated


def convert_dataset(entries: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    converted: List[Dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        converted.append(convert_entry(entry))
    return converted


def load_dataset(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse {path}: {exc}") from exc
    if not isinstance(data, list):
        raise ValueError(f"Expected a list at top-level in {path}")
    return data


def save_dataset(data: List[Dict[str, Any]], path: Path) -> None:
    os.makedirs(path.parent, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert conversation strings to list-of-dict segments."
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT_PATH,
        help="Input dataset path (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_PATH,
        help="Output path for the converted dataset (default: %(default)s).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    entries = load_dataset(input_path)
    converted = convert_dataset(entries)
    save_dataset(converted, output_path)

    print(
        f"[done] Converted {len(converted)} items from {input_path} "
        f"to {output_path}"
    )


if __name__ == "__main__":
    main()
