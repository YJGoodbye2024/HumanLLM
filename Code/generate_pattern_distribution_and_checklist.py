import argparse
import asyncio
import json
import os
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from openai import AsyncOpenAI
from tqdm import tqdm

DEFAULT_INPUT_FILE = "Dataset/generated_data_with_factors.json"
DEFAULT_OUTPUT_FILE = "Dataset/generated_data_with_artifacts.json"
DEFAULT_MODEL = "gpt-5"
DEFAULT_FAILURES_FILE = "Dataset/pattern_checklist_failures.json"

PATTERN_DIST_SYS_PROMPT = """
You act as a precise analyst who maps psychological patterns to characters.
Return strictly formatted JSON.
""".strip()

PATTERN_DIST_PROMPT = """
Design Basis (use this to determine pattern alignment):
{design_basis}

Characters:
{character_list}

Allowed Pattern Names (only choose from this list):
{pattern_list}

Task:
- For each character, select the relevant patterns from the allowed list.
- Do not introduce new pattern names.
- If a pattern is unclear for a character, leave their list empty.

Return JSON using exactly this schema:
{{
  "char2pattern": {{
    "Character A": ["pattern1", "pattern2"],
    "Character B": []
  }}
}}
""".strip()

EXPECTED_HEADER = "**Expected Character Tendencies**"


class RateLimiter:
    def __init__(self, max_calls: int, period: float) -> None:
        self.max_calls = max_calls
        self.period = period
        self._timestamps: List[float] = []
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        while True:
            async with self._lock:
                now = asyncio.get_running_loop().time()
                self._timestamps = [
                    ts for ts in self._timestamps if now - ts < self.period
                ]
                if len(self._timestamps) < self.max_calls:
                    self._timestamps.append(now)
                    return
                wait_for = self.period - (now - self._timestamps[0])
            await asyncio.sleep(max(wait_for, 0.01))


@dataclass
class GPTClient:
    client: AsyncOpenAI
    label: str
    limiter: RateLimiter
    min_interval: float = 0.5
    _interval_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _next_time: float = 0.0


async def throttle_client(credential: GPTClient) -> None:
    await credential.limiter.acquire()
    async with credential._interval_lock:
        now = asyncio.get_running_loop().time()
        wait_for = credential._next_time - now
        if wait_for > 0:
            await asyncio.sleep(wait_for)
            now = asyncio.get_running_loop().time()
        credential._next_time = now + credential.min_interval


def build_client(max_calls_per_minute: int, model_label: str) -> GPTClient:
    base_url = os.getenv("BASE_URL_FULL")
    api_key = os.getenv("API_KEY_FULL")
    if not base_url or not api_key:
        raise RuntimeError("BASE_URL_FULL and API_KEY_FULL must be set.")

    configured = max(max_calls_per_minute, 1)
    effective = min(configured, 150)
    if effective < configured:
        print(
            f"[warn] Limiting request rate to {effective}/min (requested {configured}/min).")

    client = AsyncOpenAI(api_key=api_key, base_url=base_url.rstrip("/"))
    return GPTClient(
        client=client,
        label=model_label,
        limiter=RateLimiter(max_calls=effective, period=60.0),
    )


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False,
                   indent=2), encoding="utf-8")
    tmp.replace(path)


def _normalize_name_key(name: str) -> str:
    base = re.sub(r"\(.*?\)", "", name).strip()
    base = re.sub(r"\s+", " ", base)
    normalized = unicodedata.normalize("NFKD", base)
    without_diacritics = "".join(
        ch for ch in normalized if not unicodedata.combining(ch)
    )
    return without_diacritics.lower()


def collect_character_names(entry: Dict[str, Any]) -> List[str]:
    names: List[str] = []
    seen: set[str] = set()

    def try_add(raw: Any) -> None:
        if not isinstance(raw, str):
            return
        candidate = raw.strip()
        if not candidate:
            return
        key = _normalize_name_key(candidate)
        if key and key not in seen:
            seen.add(key)
            names.append(candidate)

    scenario = entry.get("scenario") or {}
    if isinstance(scenario, dict):
        chars = scenario.get("charactersProfiles") or {}
        protagonist = chars.get("protagonist") or {}
        try_add(protagonist.get("name"))
        supporting = chars.get("supportingCharacter") or []
        if isinstance(supporting, list):
            for profile in supporting:
                if not isinstance(profile, dict):
                    continue
                try_add(profile.get("name"))

    try_add(entry.get("protagonist"))
    fallback_support = entry.get("supporting_characters") or []
    if isinstance(fallback_support, list):
        for raw in fallback_support:
            try_add(raw)
    return names


def extract_story_background(entry: Dict[str, Any]) -> str:
    scenario = entry.get("scenario") or {}
    if isinstance(scenario, dict):
        story = scenario.get("storyBackground")
        if isinstance(story, str):
            return story.strip()
    return ""


def extract_pattern_names(pattern_field: Any) -> List[str]:
    names: List[str] = []

    def add(value: Any) -> None:
        if value is None:
            return
        text = str(value).strip()
        if text and text not in names:
            names.append(text)

    if isinstance(pattern_field, Sequence) and not isinstance(pattern_field, str):
        for item in pattern_field:
            add(item)
    else:
        add(pattern_field)
    return names


def extract_design_basis(analysis: str) -> str:
    marker = "**Catalyst Details**"
    idx = analysis.find(marker)
    if idx == -1:
        return analysis.strip()
    return analysis[:idx].strip()


def extract_expected_tendencies(analysis: str) -> str:
    idx = analysis.find(EXPECTED_HEADER)
    if idx == -1:
        return ""
    return analysis[idx:].strip()


def extract_protagonist_tendencies(analysis: str) -> str:
    marker = "**Expected Protagonist Tendencies**"
    idx = analysis.find(marker)
    if idx == -1:
        return ""
    return analysis[idx:].strip()


def needs_processing(entry: Dict[str, Any], mode: int) -> bool:
    if mode == 1:
        return (
            "char2pattern" not in entry
            or "conversation_checklist" not in entry
        )
    if mode == 2:
        return "conversation_checklist_variation" not in entry
    return (
        "char2pattern" not in entry
        or "conversation_checklist" not in entry
        or "conversation_checklist_variation" not in entry
    )


async def process_entry_v2(
    idx: int,
    entry: Dict[str, Any],
    credential: GPTClient,
    args: argparse.Namespace,
    semaphore: asyncio.Semaphore,
    entries_lock: asyncio.Lock,
) -> Tuple[int, Optional[str]]:
    async with semaphore:
        analysis = entry.get("analysis")
        if not isinstance(analysis, str) or not analysis.strip():
            return idx, "Missing analysis text."

        character_names = collect_character_names(entry)
        if not character_names:
            return idx, "No character names found."

        needs_pattern = args.mode != 2 and "char2pattern" not in entry
        needs_checklist = args.mode != 2 and "conversation_checklist" not in entry
        needs_variation = args.mode != 1 and "conversation_checklist_variation" not in entry

        if not (needs_pattern or needs_checklist or needs_variation):
            return idx, None

    pattern_field = entry.get("pattern")
    single_pattern_mode = isinstance(pattern_field, str)
    pattern_names: List[str] = []
    if needs_pattern:
        pattern_names = extract_pattern_names(pattern_field)
        if not pattern_names:
            return idx, "Pattern list is empty."

    existing_pd = entry.get("char2pattern")
    pattern_distribution: Optional[Dict[str, List[str]]] = None
    checklist: Optional[Dict[str, List[str]]] = None
    checklist_variation: Optional[Dict[str, List[str]]] = None

    async def commit_updates() -> None:
        if pattern_distribution is None and checklist is None and checklist_variation is None:
            return
        async with entries_lock:
            if pattern_distribution is not None:
                entry["char2pattern"] = pattern_distribution
            if checklist is not None:
                entry["conversation_checklist"] = checklist
            if checklist_variation is not None:
                entry["conversation_checklist_variation"] = checklist_variation

    if single_pattern_mode:
        protagonist_name = get_primary_protagonist(entry, character_names)
        if not protagonist_name:
            return idx, "Unable to determine protagonist name."
        if needs_pattern:
            pattern_distribution = {protagonist_name: pattern_names}
        if needs_checklist:
            tendency_section = extract_protagonist_tendencies(analysis)
            try:
                checklist = build_protagonist_checklist(
                    tendency_section,
                    protagonist_name,
                )
            except RuntimeError as exc:
                await commit_updates()
                return idx, str(exc)
        if needs_variation:
            analysis_var = entry.get("analysis_variation")
            if isinstance(analysis_var, str) and analysis_var.strip():
                tendency_var_section = extract_protagonist_tendencies(
                    analysis_var)
                try:
                    checklist_variation = build_protagonist_checklist(
                        tendency_var_section,
                        protagonist_name,
                    )
                except RuntimeError as exc:
                    await commit_updates()
                    return idx, f"Variation checklist error: {exc}"
        await commit_updates()
        return idx, None
    else:
        expected_section: Optional[str] = None
        expected_var: Optional[str] = None

        if needs_checklist:
            expected_section = extract_expected_tendencies(analysis)
            if not expected_section.startswith(EXPECTED_HEADER):
                return idx, "Analysis missing '**Expected Character Tendencies**'."
            # checklist built after pattern assignment is known

        if needs_variation:
            analysis_var = entry.get("analysis_variation")
            if isinstance(analysis_var, str) and analysis_var.strip():
                expected_var = extract_expected_tendencies(analysis_var)
                if not expected_var.startswith(EXPECTED_HEADER):
                    return idx, "analysis_variation missing '**Expected Character Tendencies**'."
                # checklist_variation built after pattern assignment is known

        if needs_pattern:
            design_basis = extract_design_basis(analysis)
            if not design_basis:
                return idx, "Unable to extract design basis portion."
            pattern_list_text = ", ".join(pattern_names)
            allowed_lookup = {name.lower(): name for name in pattern_names}
            pattern_messages = [
                {"role": "system", "content": PATTERN_DIST_SYS_PROMPT},
                {"role": "user", "content": PATTERN_DIST_PROMPT.format(
                    design_basis=design_basis,
                    character_list=", ".join(character_names),
                    pattern_list=pattern_list_text,
                )},
            ]

            parsed_distribution = None
            max_parse_attempts = max(args.retries, 1)
            for attempt in range(1, max_parse_attempts + 1):
                try:
                    pattern_raw = await fetch_chat_completion(
                        credential,
                        args.model,
                        pattern_messages,
                        args.timeout,
                        args.retries,
                        args.backoff,
                    )
                except Exception as exc:  # noqa: BLE001
                    return idx, f"char2pattern API error: {exc}"

                try:
                    parsed_distribution = parse_pattern_distribution(pattern_raw)
                    filtered: Dict[str, List[str]] = {}
                    for name, patterns in parsed_distribution.items():
                        cleaned: List[str] = []
                        for pat in patterns:
                            key = pat.lower()
                            if key in allowed_lookup:
                                cleaned.append(allowed_lookup[key])
                        filtered[name] = cleaned
                    if not filtered or all(len(pats) == 0 for pats in filtered.values()):
                        raise ValueError(
                            "char2pattern returned empty assignments for all characters."
                        )
                    pattern_distribution = filtered
                    break
                except ValueError as exc:
                    print(f"[warn] sample {idx} char2pattern parse error (attempt {attempt}/{max_parse_attempts}): {exc}")
                    print(f"[warn] raw char2pattern response: {pattern_raw}")
                    if attempt >= max_parse_attempts:
                        return idx, f"char2pattern parse error: {exc}"
                    await asyncio.sleep(args.backoff * attempt)

        pd_source = pattern_distribution if pattern_distribution is not None else (
            existing_pd if isinstance(existing_pd, dict) else None
        )
        assigned_names = [
            name for name, pats in pd_source.items() if pats] if pd_source else []
        target_names = assigned_names if assigned_names else character_names
        required_names = assigned_names if assigned_names else None

        if needs_checklist and expected_section:
            try:
                checklist = build_checklist_from_expected(
                    expected_section,
                    target_names,
                    required_names,
                )
            except RuntimeError as exc:
                await commit_updates()
                return idx, str(exc)

        if needs_variation and expected_var:
            try:
                checklist_variation = build_checklist_from_expected(
                    expected_var,
                    target_names,
                    required_names,
                )
            except RuntimeError as exc:
                await commit_updates()
                return idx, f"Variation checklist error: {exc}"

        await commit_updates()
        return idx, None


def strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        inner = stripped[3:]
        newline = inner.find("\n")
        stripped = inner[newline + 1:] if newline != -1 else ""
        if stripped.endswith("```"):
            stripped = stripped[:-3]
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
    return stripped.strip()


def parse_pattern_distribution(payload: str) -> Dict[str, List[str]]:
    cleaned = strip_code_fences(payload)
    cleaned = cleaned.strip()
    if cleaned.lower().startswith("json"):
        cleaned = cleaned[4:].strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            data = json.loads(cleaned[start : end + 1])
        else:
            raise
    if not isinstance(data, dict) or "char2pattern" not in data:
        raise ValueError("Response missing 'char2pattern'.")
    pd = data["char2pattern"]
    if not isinstance(pd, dict):
        raise ValueError("'char2pattern' must be an object.")
    result: Dict[str, List[str]] = {}
    for name, patterns in pd.items():
        if isinstance(patterns, list):
            result[name] = [str(item).strip()
                            for item in patterns if str(item).strip()]
        else:
            result[name] = []
    return result


def load_failure_indices(path: Path, total_entries: int) -> List[int]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[warn] Unable to load failures file {path}: {exc}")
        return []
    if not isinstance(data, list):
        print(f"[warn] Failures file {path} is not a list; ignoring.")
        return []
    seen = set()
    indices: List[int] = []
    for item in data:
        idx = item.get("index") if isinstance(item, dict) else item
        if isinstance(idx, int) and 0 <= idx < total_entries and idx not in seen:
            seen.add(idx)
            indices.append(idx)
    return sorted(indices)


def save_failure_records(path: Path, errors: Dict[int, str]) -> None:
    if not errors:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        return
    records = [{"index": idx, "error": message}
               for idx, message in sorted(errors.items())]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2),
                    encoding="utf-8")


async def fetch_chat_completion(
    credential: GPTClient,
    model: str,
    messages: List[Dict[str, str]],
    timeout: float,
    retries: int,
    backoff: float,
) -> str:
    last_error: Optional[Exception] = None
    for attempt in range(1, max(retries, 1) + 1):
        try:
            await throttle_client(credential)
            response = await asyncio.wait_for(
                credential.client.chat.completions.create(
                    model=model,
                    messages=messages,
                ),
                timeout=timeout,
            )
            choices = response.choices or []
            if choices:
                content = choices[0].message.content
                if isinstance(content, str) and content.strip():
                    return content
                raise ValueError("Empty response content from API.")
            raise ValueError("Empty choices from API.")
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < retries:
                await asyncio.sleep(backoff * attempt)
                continue
            raise
    assert last_error is not None
    raise last_error


def build_checklist_from_expected(
    expected_section: str,
    character_names: List[str],
    required_names: Optional[List[str]] = None,
) -> Dict[str, List[str]]:
    lines = [
        line.strip()
        for line in expected_section.splitlines()
        if line.strip()
    ]
    checklist: Dict[str, List[str]] = {}
    target = required_names if required_names is not None else character_names
    name_lookup = {_normalize_name_key(name): name for name in target}

    for line in lines:
        if line.lower().startswith(EXPECTED_HEADER.lower()):
            continue
        if not line.startswith("@"):
            continue
        parts = line.split(":", 1)
        if len(parts) != 2:
            continue
        name_segment, tendencies_segment = parts
        name_match = re.search(r"\[(.+?)\]", name_segment)
        if name_match:
            raw_name = name_match.group(1)
        else:
            raw_name = re.sub(r"^@\s*", "", name_segment).strip()
        raw_name = re.sub(r"\(.*?\)", "", raw_name).strip()
        if not raw_name:
            continue
        normalized = _normalize_name_key(raw_name)
        canonical = name_lookup.get(normalized)
        if not canonical:
            continue
        tendencies = [
            item.strip()
            for item in tendencies_segment.split(";")
            if item.strip()
        ]
        checklist[canonical] = tendencies

    return checklist


def build_protagonist_checklist(
    tendencies_section: str,
    protagonist_name: str,
) -> Dict[str, List[str]]:
    if not tendencies_section:
        raise RuntimeError("Protagonist tendencies section missing.")

    lines = [
        line.strip()
        for line in tendencies_section.splitlines()
        if line.strip()
    ]
    collected: List[str] = []
    for line in lines:
        if line.lower().startswith("**expected protagonist tendencies"):
            continue
        if not line.startswith("*"):
            continue
        content = line.lstrip("*").strip()
        match = re.match(r"\*\*(.+?)\*\*:\s*(.+)", content)
        if match:
            title = match.group(1).strip()
            desc = match.group(2).strip()
            combined = f"{title}: {desc}" if desc else title
        else:
            combined = content
        if combined:
            collected.append(combined)
    if not collected:
        raise RuntimeError("Unable to parse protagonist tendencies.")
    return {protagonist_name: collected}


def get_primary_protagonist(entry: Dict[str, Any], candidates: List[str]) -> Optional[str]:
    scenario = entry.get("scenario") or {}
    if isinstance(scenario, dict):
        chars = scenario.get("charactersProfiles") or {}
        protagonist = chars.get("protagonist") or {}
        name = protagonist.get("name")
        if isinstance(name, str) and name.strip():
            return name.strip()
    fallback = entry.get("protagonist")
    if isinstance(fallback, str) and fallback.strip():
        return fallback.strip()
    return candidates[0] if candidates else None


async def process_entry(
    idx: int,
    entry: Dict[str, Any],
    credential: GPTClient,
    args: argparse.Namespace,
    semaphore: asyncio.Semaphore,
    entries_lock: asyncio.Lock,
) -> Tuple[int, Optional[str]]:
    async with semaphore:
        analysis = entry.get("analysis")
        if not isinstance(analysis, str) or not analysis.strip():
            return idx, "Missing analysis text."

        character_names = collect_character_names(entry)
        if not character_names:
            return idx, "No character names found."

        pattern_field = entry.get("pattern")
        single_pattern_mode = isinstance(pattern_field, str)

        pattern_names = extract_pattern_names(pattern_field)
        if not pattern_names:
            return idx, "Pattern list is empty."

        if single_pattern_mode:
            protagonist_name = get_primary_protagonist(entry, character_names)
            if not protagonist_name:
                return idx, "Unable to determine protagonist name."
            tendency_section = extract_protagonist_tendencies(analysis)
            try:
                checklist = build_protagonist_checklist(
                    tendency_section,
                    protagonist_name,
                )
            except RuntimeError as exc:
                return idx, str(exc)
            pattern_distribution = {protagonist_name: pattern_names}
            async with entries_lock:
                entry["char2pattern"] = pattern_distribution
                entry["conversation_checklist"] = checklist
            return idx, None

        story_background = extract_story_background(entry)
        if not story_background:
            return idx, "Missing story background."

        expected_section = extract_expected_tendencies(analysis)
        if not expected_section.startswith(EXPECTED_HEADER):
            return idx, "Analysis missing '**Expected Character Tendencies**'."

        design_basis = extract_design_basis(analysis)
        if not design_basis:
            return idx, "Unable to extract design basis portion."

        pattern_list_text = ", ".join(pattern_names)
        allowed_lookup = {name.lower(): name for name in pattern_names}

        pattern_messages = [
            {"role": "system", "content": PATTERN_DIST_SYS_PROMPT},
            {"role": "user", "content": PATTERN_DIST_PROMPT.format(
                design_basis=design_basis,
                character_list=", ".join(character_names),
                pattern_list=pattern_list_text,
            )},
        ]
        try:
            pattern_raw = await fetch_chat_completion(
                credential,
                args.model,
                pattern_messages,
                args.timeout,
                args.retries,
                args.backoff,
            )
        except Exception as exc:  # noqa: BLE001
            return idx, f"char2pattern API error: {exc}"

        try:
            parsed_distribution = parse_pattern_distribution(pattern_raw)
        except ValueError as exc:
            return idx, f"char2pattern parse error: {exc}"

        filtered: Dict[str, List[str]] = {}
        for name, patterns in parsed_distribution.items():
            cleaned: List[str] = []
            for pat in patterns:
                key = pat.lower()
                if key in allowed_lookup:
                    cleaned.append(allowed_lookup[key])
            filtered[name] = cleaned

        try:
            checklist = build_checklist_from_expected(
                expected_section,
                character_names,
            )
        except RuntimeError as exc:
            return idx, str(exc)

        async with entries_lock:
            entry["char2pattern"] = filtered
            entry["conversation_checklist"] = checklist
        return idx, None


async def run(args: argparse.Namespace) -> None:
    input_path = Path(args.input_file)
    output_path = Path(args.output_file)
    failures_path = Path(args.failures_file)

    dataset_path = output_path if output_path.exists() else input_path
    if dataset_path != input_path:
        print(f"[info] Resuming from existing output dataset: {dataset_path}")

    entries = load_json(dataset_path)
    if not isinstance(entries, list):
        raise RuntimeError(f"{dataset_path} must contain a list of entries.")

    # Backward compatibility: migrate old key to new schema.
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if "pattern_distribution" in entry:
            if "char2pattern" not in entry:
                entry["char2pattern"] = entry["pattern_distribution"]
            # Always drop the legacy key to keep outputs consistent.
            entry.pop("pattern_distribution", None)

    credential = build_client(args.max_calls_per_minute, args.model)
    semaphore = asyncio.Semaphore(max(1, args.concurrency))
    entries_lock = asyncio.Lock()

    total_entries = len(entries)
    failure_indices = load_failure_indices(failures_path, total_entries)
    if failure_indices:
        if args.limit:
            print("[info] Failure list detected; ignoring --limit to retry failures.")
        pending_indices = failure_indices
        print(
            f"[info] Retrying {len(pending_indices)} sample(s) from {failures_path}.")
    else:
        pending_indices = [
            idx for idx, entry in enumerate(entries)
            if needs_processing(entry, args.mode)
        ]
        if args.limit is not None and args.limit > 0:
            pending_indices = pending_indices[: min(
                args.limit, len(pending_indices))]

    if not pending_indices:
        print("[info] No samples to process.")
        save_failure_records(failures_path, {})
        write_json(output_path, entries)
        return

    progress = tqdm(total=len(pending_indices),
                    desc="Generating pattern/checklist", unit="sample")

    async def wrapped(idx: int) -> Tuple[int, Optional[str]]:
        result = await process_entry_v2(
            idx,
            entries[idx],
            credential,
            args,
            semaphore,
            entries_lock,
        )
        progress.update(1)
        return result

    tasks = [asyncio.create_task(wrapped(idx)) for idx in pending_indices]
    errors: Dict[int, str] = {}
    try:
        results = await asyncio.gather(*tasks)
        for idx, error in results:
            if error:
                errors[idx] = error
    finally:
        progress.close()
        close_method = getattr(credential.client, "close", None)
        if close_method:
            maybe_coro = close_method()
            if asyncio.iscoroutine(maybe_coro):
                await maybe_coro

    if errors:
        print(f"[warn] {len(errors)} entries failed.")
        for idx, message in errors.items():
            print(f"  - sample {idx}: {message}")
    else:
        print("[info] All targeted samples processed successfully.")

    failed_count = len(errors)
    dropped_count = 0
    if args.drop_failures:
        if errors:
            # Remove failed entries in reverse order so indices stay valid.
            for idx in sorted(errors.keys(), reverse=True):
                if 0 <= idx < len(entries):
                    entries.pop(idx)
                    dropped_count += 1
            print(f"[info] Dropped {dropped_count} failed entr{'y' if dropped_count == 1 else 'ies'} due to --drop-failures.")
        # Always delete the failures file when dropping.
        save_failure_records(failures_path, {})
    else:
        save_failure_records(failures_path, errors)

    write_json(output_path, entries)
    patterns_count = sum(1 for e in entries if "char2pattern" in e)
    checklist_count = sum(1 for e in entries if "conversation_checklist" in e)
    checklist_var_count = sum(1 for e in entries if "conversation_checklist_variation" in e)
    print(
        f"[info] Processed {len(pending_indices) - failed_count} / {len(pending_indices)} entries. Output -> {output_path}")
    if args.drop_failures and dropped_count:
        print(f"[info] Dataset size after dropping failures: {len(entries)}")
    print(
        f"[info] Totals: char2pattern={patterns_count}, conversation_checklist={checklist_count}, conversation_checklist_variation={checklist_var_count}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate pattern distributions and conversation checklists.",
    )
    parser.add_argument("--input-file", default=DEFAULT_INPUT_FILE,
                        help="Input dataset JSON path.")
    parser.add_argument("--output-file", default=DEFAULT_OUTPUT_FILE,
                        help="Output dataset JSON path.")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="Model name for char2pattern generation.")
    parser.add_argument("--limit", type=int,
                        help="Process only the first N samples.")
    parser.add_argument("--timeout", type=float, default=180.0,
                        help="Per-request timeout in seconds.")
    parser.add_argument("--retries", type=int, default=3,
                        help="Maximum retries for API errors.")
    parser.add_argument("--backoff", type=float, default=5.0,
                        help="Base seconds for retry backoff.")
    parser.add_argument("--max-calls-per-minute", type=int, default=120,
                        help="API rate limit per minute.")
    parser.add_argument("--concurrency", type=int, default=20,
                        help="Number of concurrent samples.")
    parser.add_argument("--failures-file", default=DEFAULT_FAILURES_FILE,
                        help="Path to JSON file storing failed sample indices.")
    parser.add_argument(
        "--drop-failures",
        action="store_true",
        help="If set, remove failed samples from the dataset and delete the failures file.",
    )
    parser.add_argument(
        "--mode",
        type=int,
        choices=[0, 1, 2],
        default=0,
        help="0: generate all artifacts (default); 1: only char2pattern and conversation_checklist; 2: only conversation_checklist_variation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        raise
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"[error] {exc}") from exc


if __name__ == "__main__":
    main()
