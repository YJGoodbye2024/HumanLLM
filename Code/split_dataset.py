#!/usr/bin/env python3
"""
Split Dataset/final_data.json into train/ood_eval/id_eval/mixed_eval.

- Randomly choose 8 OOD patterns (4 from sd_pri_list + 4 from td_pri_list_100
  across Big-5 buckets), prioritizing patterns missing from the dataset and then
  the least frequent ones.
- Samples whose pattern list is fully inside the OOD set feed ood_eval (50 items).
- Samples whose pattern list has no OOD pattern feed id_eval (50 items).
- Samples mixing OOD and non-OOD patterns feed mixed_eval (all such entries).
- Remaining in-domain samples become train.
- If not enough OOD-only samples exist, interactively generate new samples using
  gen_scenario_conversation_multi_patterns_use_situation2.py and enrich them with
  char2pattern + conversation_checklist via generate_pattern_distribution_and_checklist.py.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

from principle_situaton import sd_pri_list, td_pri_list_100, Situation_list_2
from tqdm import tqdm

DEFAULT_INPUT = Path("Dataset/final_data.json")
DEFAULT_OUTPUT_DIR = Path("Dataset/split")
OOD_TARGET = 50
ID_TARGET = 50
DEFAULT_SEED = 42

TD_DIMENSION_BUCKETS = [
    td_pri_list_100[0:20],
    td_pri_list_100[20:40],
    td_pri_list_100[40:60],
    td_pri_list_100[60:80],
    td_pri_list_100[80:100],
]


def load_dataset(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Input dataset must be a JSON array.")
    migrated = 0
    for entry in data:
        if not isinstance(entry, dict):
            continue
        if "pattern_distribution" in entry:
            if "char2pattern" not in entry:
                entry["char2pattern"] = entry["pattern_distribution"]
            entry.pop("pattern_distribution", None)
            migrated += 1
    if migrated:
        print(f"[info] Migrated {migrated} entr{'y' if migrated == 1 else 'ies'} from pattern_distribution to char2pattern.")
    return data


def normalize_patterns(pattern_field: Any) -> List[str]:
    if isinstance(pattern_field, str):
        value = pattern_field.strip()
        return [value] if value else []
    if isinstance(pattern_field, Sequence):
        patterns: List[str] = []
        for item in pattern_field:
            if isinstance(item, str):
                token = item.strip()
                if token:
                    patterns.append(token)
        return patterns
    return []


def compute_pattern_counts(entries: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for entry in entries:
        for pattern in normalize_patterns(entry.get("pattern")):
            counts[pattern] = counts.get(pattern, 0) + 1
    return counts


def pick_patterns_for_group(
    source_list: Sequence[str],
    needed: int,
    counts: Dict[str, int],
    rng: random.Random,
    used: Set[str],
) -> List[str]:
    available = [p for p in source_list if p not in used]
    if len(available) < needed:
        raise ValueError(
            f"Not enough unique patterns in source list (need {needed}, have {len(available)})."
        )

    missing = [p for p in available if counts.get(p, 0) == 0]
    rng.shuffle(missing)
    selected: List[str] = missing[:needed]

    if len(selected) < needed:
        remaining = [p for p in available if p not in selected]
        rng.shuffle(remaining)
        remaining.sort(key=lambda p: counts.get(p, 0))
        for candidate in remaining:
            if len(selected) == needed:
                break
            selected.append(candidate)

    used.update(selected)
    return selected


def select_sd_patterns(
    counts: Dict[str, int], rng: random.Random, sd_count: int = 4
) -> List[str]:
    used: Set[str] = set()
    return pick_patterns_for_group(sd_pri_list, sd_count, counts, rng, used)


def select_td_patterns(
    counts: Dict[str, int], rng: random.Random, td_count: int = 4
) -> List[str]:
    if len(TD_DIMENSION_BUCKETS) < td_count:
        raise ValueError("Not enough Big-5 dimensions to satisfy TD selection.")
    per_dimension: List[Tuple[str, int]] = []
    for bucket in TD_DIMENSION_BUCKETS:
        if not bucket:
            continue
        min_count = min(counts.get(pattern, 0) for pattern in bucket)
        candidates = [p for p in bucket if counts.get(p, 0) == min_count]
        choice = rng.choice(candidates)
        per_dimension.append((choice, min_count))
    if len(per_dimension) < td_count:
        raise ValueError(
            f"Unable to gather {td_count} TD patterns across dimensions."
        )
    rng.shuffle(per_dimension)
    per_dimension.sort(key=lambda item: item[1])
    return [pattern for pattern, _ in per_dimension[:td_count]]


def select_ood_patterns(
    counts: Dict[str, int],
    rng: random.Random,
    sd_count: int = 4,
    td_count: int = 4,
) -> Tuple[List[str], List[str]]:
    sd_selected = select_sd_patterns(counts, rng, sd_count)
    td_selected = select_td_patterns(counts, rng, td_count)
    return sd_selected, td_selected


def categorize_entries(
    data: Sequence[Dict[str, Any]],
    ood_set: Set[str],
) -> Tuple[
    List[Tuple[int, Dict[str, Any]]],
    List[Tuple[int, Dict[str, Any]]],
    List[Tuple[int, Dict[str, Any]]],
]:
    ood_only: List[Tuple[int, Dict[str, Any]]] = []
    in_domain: List[Tuple[int, Dict[str, Any]]] = []
    mixed: List[Tuple[int, Dict[str, Any]]] = []
    for idx, entry in enumerate(data):
        patterns = normalize_patterns(entry.get("pattern"))
        if not patterns:
            continue
        pattern_set = set(patterns)
        if pattern_set <= ood_set:
            ood_only.append((idx, entry))
        elif pattern_set.isdisjoint(ood_set):
            in_domain.append((idx, entry))
        else:
            mixed.append((idx, entry))
    return ood_only, in_domain, mixed


def contains_ood_pattern(entry: Dict[str, Any], ood_set: Set[str]) -> bool:
    patterns = normalize_patterns(entry.get("pattern"))
    return any(pattern in ood_set for pattern in patterns)


def sample_entries(
    entries: Sequence[Tuple[int, Dict[str, Any]]],
    count: int,
    rng: random.Random,
    label: str,
) -> List[Tuple[int, Dict[str, Any]]]:
    if len(entries) < count:
        raise RuntimeError(
            f"Not enough entries for {label}: need {count}, only {len(entries)} available."
        )
    shuffled = list(entries)
    rng.shuffle(shuffled)
    return shuffled[:count]


def propose_combos(
    sd_patterns: Sequence[str],
    td_patterns: Sequence[str],
    needed: int,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    if len(td_patterns) < 2:
        raise ValueError("Need at least two TD patterns to form combos.")
    td_range = range(2, min(4, len(td_patterns)) + 1)
    unique_pattern_sets: List[List[str]] = []
    seen_keys: Set[Tuple[str, ...]] = set()

    for td_count in td_range:
        for td_subset in combinations(td_patterns, td_count):
            td_list = list(td_subset)
            key = tuple(sorted(td_list))
            if key not in seen_keys:
                seen_keys.add(key)
                unique_pattern_sets.append(td_list)
            for sd in sd_patterns:
                combo = td_list + [sd]
                combo_key = tuple(sorted(combo))
                if combo_key not in seen_keys:
                    seen_keys.add(combo_key)
                    unique_pattern_sets.append(combo)

    if not unique_pattern_sets:
        raise RuntimeError("No valid pattern combinations available.")

    rng.shuffle(unique_pattern_sets)
    combos: List[Dict[str, Any]] = []
    idx = 0
    while len(combos) < needed:
        if idx < len(unique_pattern_sets):
            patterns = unique_pattern_sets[idx]
        else:
            patterns = rng.choice(unique_pattern_sets)
        idx += 1
        situation = rng.choice(Situation_list_2)
        combos.append({"patterns": patterns, "situation": situation})
    return combos


async def _generate_new_entries_async(
    combos: Sequence[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not combos:
        return [], []
    import gen_scenario_conversation_multi_patterns_use_situation2 as generator

    base_url = os.getenv("BASE_URL_FULL")
    api_key = os.getenv("API_KEY_FULL")
    if not base_url or not api_key:
        raise RuntimeError(
            "BASE_URL_FULL and API_KEY_FULL must be configured to generate new samples."
        )
    patterns_info = generator.load_patterns_info(generator.PATTERNS_INFO_FILE)
    if not patterns_info:
        raise RuntimeError("Failed to load pattern information for generation.")

    client = generator.AsyncOpenAI(api_key=api_key, base_url=base_url.rstrip("/"))
    runner = generator.ModelRunner("gpt", client, generator.GPT_MODEL)
    semaphore = asyncio.Semaphore(
        max(1, min(generator.MAX_CONCURRENT_REQUESTS, len(combos), 5))
    )

    tasks: List[asyncio.Task] = []

    async def run_combo(combo: Dict[str, Any], principle: Dict[str, Any]):
        try:
            result = await generator.process_single_combination(
                principle,
                combo["situation"],
                runner,
                runner,
                runner,
                semaphore,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] Generation task failed for combo {combo}: {exc}")
            return combo, None
        return combo, result

    for combo in combos:
        try:
            info_text = generator._format_patterns_information(
                patterns_info, combo["patterns"]
            )
        except KeyError as exc:  # noqa: PERF203
            raise RuntimeError(f"Unknown pattern during generation: {exc}") from exc
        principle = {
            "construct_name": " + ".join(combo["patterns"]),
            "patterns": combo["patterns"],
            "principles_information": info_text,
            "bundle_type": f"custom_{len(combo['patterns'])}",
        }
        task = asyncio.create_task(run_combo(combo, principle))
        tasks.append(task)

    successes: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    progress = tqdm(total=len(tasks), desc="Generating samples", unit="sample")
    try:
        for future in asyncio.as_completed(tasks):
            combo, result = await future
            if result is None:
                failures.append(combo)
            else:
                successes.append(result)
            progress.update(1)
    finally:
        progress.close()

    close = getattr(client, "close", None)
    if close:
        maybe = close()
        if asyncio.iscoroutine(maybe):
            await maybe
    return successes, failures


async def _enrich_entries_async(entries: Sequence[Dict[str, Any]]) -> None:
    if not entries:
        return
    import generate_pattern_distribution_and_checklist as artifacts

    args = argparse.Namespace(
        mode=1,
        retries=3,
        backoff=5.0,
        timeout=180.0,
        model=artifacts.DEFAULT_MODEL,
    )
    credential = artifacts.build_client(60, args.model)
    semaphore = asyncio.Semaphore(max(1, min(5, len(entries))))
    entries_lock = asyncio.Lock()

    tasks: List[asyncio.Task] = []
    for idx, entry in enumerate(entries):
        task = asyncio.create_task(
            artifacts.process_entry_v2(
                idx,
                entry,
                credential,
                args,
                semaphore,
                entries_lock,
            )
        )
        tasks.append(task)
    errors: List[str] = []
    progress = tqdm(total=len(tasks), desc="Enriching samples", unit="sample")
    try:
        for task in asyncio.as_completed(tasks):
            try:
                idx, message = await task
            except Exception as exc:  # noqa: BLE001
                message = f"Unexpected exception: {exc}"
                idx = -1
            if message:
                errors.append(f"sample {idx}: {message}")
            progress.update(1)
    finally:
        progress.close()

    close = getattr(credential.client, "close", None)
    if close:
        maybe = close()
        if asyncio.iscoroutine(maybe):
            await maybe

    if errors:
        raise RuntimeError(
            "Failed to enrich new samples: " + "; ".join(errors)
        )


def generate_additional_samples(
    combos: Sequence[Dict[str, Any]],
    max_attempts: int = 3,
) -> List[Dict[str, Any]]:
    if not combos:
        return []
    remaining = list(combos)
    all_new_entries: List[Dict[str, Any]] = []
    for attempt in range(1, max_attempts + 1):
        if not remaining:
            break
        print(f"[info] Generation attempt {attempt}: {len(remaining)} combos")
        generated, failed = asyncio.run(_generate_new_entries_async(remaining))
        if generated:
            all_new_entries.extend(generated)
            print(
                f"[info] Attempt {attempt} succeeded for {len(generated)} combos."
            )
        if not failed:
            remaining = []
            break
        remaining = failed
        if attempt < max_attempts:
            print(
                f"[warn] {len(remaining)} combos failed in attempt {attempt}, retrying..."
            )
    if remaining:
        if all_new_entries:
            print(
                f"[error] Unable to generate {len(remaining)} combos after {max_attempts} attempts."
            )
        else:
            print(
                f"[error] All {len(remaining)} combos failed after {max_attempts} attempts."
            )
    if all_new_entries:
        print("[info] Enriching supplemental samples with char2pattern/checklist...")
        asyncio.run(_enrich_entries_async(all_new_entries))
    if remaining:
        raise RuntimeError(
            f"Failed to generate {len(remaining)} combos after {max_attempts} attempts."
        )
    return all_new_entries


def write_json(path: Path, data: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(list(data), ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split Dataset/final_data.json into train/ood/id/mixed subsets."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--ood-target", type=int, default=OOD_TARGET)
    parser.add_argument("--id-target", type=int, default=ID_TARGET)
    parser.add_argument(
        "--generation-retries",
        type=int,
        default=3,
        help="Maximum attempts for supplemental generation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    data = load_dataset(args.input)

    pattern_counts = compute_pattern_counts(data)
    sd_patterns, td_patterns = select_ood_patterns(pattern_counts, rng)
    ood_patterns = sd_patterns + td_patterns
    ood_set = set(ood_patterns)

    ood_entries, id_entries, mixed_entries = categorize_entries(data, ood_set)
    generated_batches: List[Dict[str, Any]] = []
    generation_combos: List[Dict[str, Any]] = []

    if len(ood_entries) < args.ood_target:
        deficit = args.ood_target - len(ood_entries)
        print(
            f"[warn] Only {len(ood_entries)} OOD-only samples available, need {args.ood_target}."
        )
        print(
            f"[info] Current category counts -> OOD-only: {len(ood_entries)}, "
            f"ID-only: {len(id_entries)}, Mixed: {len(mixed_entries)}"
        )
        combos = propose_combos(sd_patterns, td_patterns, deficit, rng)
        generation_combos = combos
        print("\nProposed new combinations (patterns + situation):")
        for idx, combo in enumerate(combos, 1):
            print(
                f"  {idx:>2}. patterns={', '.join(combo['patterns'])} | "
                f"situation={combo['situation']}"
            )
        answer = input(
            "\nType 'yes' to generate these samples automatically (anything else aborts): "
        ).strip().lower()
        if answer != "yes":
            raise SystemExit("Aborted due to insufficient OOD samples.")
        new_entries = generate_additional_samples(
            combos, max_attempts=args.generation_retries
        )
        data.extend(new_entries)
        generated_batches = new_entries
        pattern_counts = compute_pattern_counts(data)
        ood_entries, id_entries, mixed_entries = categorize_entries(data, ood_set)
        if len(ood_entries) < args.ood_target:
            raise RuntimeError(
                f"Still only {len(ood_entries)} OOD-only samples after generation."
            )

    if len(id_entries) < args.id_target:
        raise RuntimeError(
            f"Not enough ID-only samples: need {args.id_target}, only {len(id_entries)}."
        )

    ood_eval = sample_entries(ood_entries, args.ood_target, rng, "ood_eval")
    id_eval = sample_entries(id_entries, args.id_target, rng, "id_eval")
    mixed_eval = list(mixed_entries)

    selected_indices: Set[int] = (
        {idx for idx, _ in ood_eval}
        | {idx for idx, _ in id_eval}
        | {idx for idx, _ in mixed_eval}
    )
    train_entries = [
        entry
        for idx, entry in enumerate(data)
        if idx not in selected_indices and not contains_ood_pattern(entry, ood_set)
    ]

    output_dir: Path = args.output_dir
    write_json(output_dir / "ood_eval.json", (entry for _, entry in ood_eval))
    write_json(output_dir / "id_eval.json", (entry for _, entry in id_eval))
    write_json(output_dir / "mixed_eval.json", (entry for _, entry in mixed_eval))
    write_json(output_dir / "train.json", train_entries)

    metadata = {
        "input": str(args.input),
        "seed": args.seed,
        "ood_patterns": {
            "sd": sd_patterns,
            "td": td_patterns,
        },
        "counts": {
            "train": len(train_entries),
            "ood_eval": len(ood_eval),
            "id_eval": len(id_eval),
            "mixed_eval": len(mixed_eval),
        },
        "generated_samples": len(generated_batches),
        "generation_combos": generation_combos,
    }
    write_json(output_dir / "metadata.json", [metadata])

    print(
        f"Done. Train={len(train_entries)}, "
        f"OOD Eval={len(ood_eval)}, ID Eval={len(id_eval)}, Mixed Eval={len(mixed_eval)}"
    )
    print(f"OOD patterns: {', '.join(ood_patterns)}")


if __name__ == "__main__":
    main()
