import argparse
import json
import os
import queue
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Set

import requests

try:
    from pdfminer.high_level import extract_text  # type: ignore
except ImportError as exc:  # pragma: no cover - highlighted at runtime
    raise RuntimeError(
        "pdfminer.six is required for PDF text extraction. Install with "
        "'pip install pdfminer.six'."
    ) from exc

from principle_situaton import sd_pri_list, td_pri_list_100
from prompt_all import (
    sd_principle_info_prompt,
    td_principle_info_prompt,
    sd_to_json,
    td_to_json,
)


BASE_DIR = Path("Dataset/papers_info")
PRINCIPLES_PATH = BASE_DIR / "principles.json"
PDF_DIR = BASE_DIR / "pdfs"
TEXT_DIR = BASE_DIR / "texts"
OUTPUT_DIR = Path("Dataset/principle_summaries")
RAW_OUTPUT_PATH = Path("Dataset/patterns_info/psy_patterns_markdown.json")
JSON_OUTPUT_PATH = Path("Dataset/patterns_info/psy_patterns_info.json")
DEBUG_JSON_DIR = Path("Dataset/patterns_info/debug_failed_json")

DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-pro-cli")
DEFAULT_SLEEP_SECONDS = float(os.getenv("DEEP_REQUEST_INTERVAL", "15"))
MAX_DOC_CHARS = int(os.getenv("SUMMARY_MAX_DOC_CHARS", "30000"))

SOURCE_SECTION_PATTERN = re.compile(
    r"(?:^|\n)##?\s*3\.\s*Sources\b.*", re.IGNORECASE | re.DOTALL)


def slugify(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return cleaned or "unknown"


def load_principles() -> Dict[str, List[Dict]]:
    if not PRINCIPLES_PATH.exists():
        raise FileNotFoundError(
            f"Principles file not found: {PRINCIPLES_PATH}")
    with PRINCIPLES_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def remove_sources_section(content: str) -> str:
    return SOURCE_SECTION_PATTERN.sub("", content).strip()


def read_text_file(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    try:
        return remove_sources_section(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        # fallback to binary read + decode ignore errors
        data = path.read_bytes()
        return remove_sources_section(data.decode("utf-8", errors="ignore"))


def extract_text_from_pdf(pdf_path: Path) -> Optional[str]:
    if not pdf_path.exists():
        return None
    try:
        text = extract_text(pdf_path)
        return text.strip() if text else None
    except Exception as exc:
        print(f"[warn] Failed to extract text from PDF {pdf_path.name}: {exc}")
        return None


def gather_document_content(paper_key: str) -> Optional[str]:
    text_path = TEXT_DIR / f"{paper_key}.txt"
    pdf_path = PDF_DIR / f"{paper_key}.pdf"

    text_content = read_text_file(text_path)
    if text_content:
        return text_content

    pdf_content = extract_text_from_pdf(pdf_path)
    return pdf_content


def collect_corpus_for_principle(
    principle: str,
    entries: List[Dict],
    max_documents: Optional[int] = None,
) -> List[str]:
    corpus: List[str] = []
    for entry in entries[: max_documents or len(entries)]:
        paper_key = entry.get("paper_key")
        if not paper_key:
            continue
        content = gather_document_content(paper_key)
        if not content:
            print(f"[warn] No content found for {paper_key} ({principle})")
            continue
        if MAX_DOC_CHARS and len(content) > MAX_DOC_CHARS:
            content = content[:MAX_DOC_CHARS]
        corpus.append(content)
    return corpus


def build_corpus_payload(corpus: Iterable[str]) -> str:
    return "\n\n".join(doc.strip() for doc in corpus)


def choose_prompt(principle: str) -> Tuple[str, str]:
    if principle in sd_pri_list:
        return sd_principle_info_prompt, "sd"
    if principle in td_pri_list_100:
        return td_principle_info_prompt, "td"
    return sd_principle_info_prompt, "sd"


def build_request_payload(
    principle: str,
    corpus_text: str,
    prompt_template: str,
    prompt_type: str,
    model: str,
) -> Dict[str, Any]:
    prompt_text = prompt_template
    if prompt_type == "td":
        prompt_text = prompt_text.replace("{Trait Name}", principle)
    else:
        prompt_text = prompt_text.replace("{Principle Name}", principle)
    prompt_text = prompt_text.replace("{ALL 50 PAPERS' CONTENT}", corpus_text)
    return {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an expert academic synthesizer. "
                    "Follow the user's instructions exactly and return the markdown report."
                ),
            },
            {"role": "user", "content": prompt_text.strip()},
        ],
        "temperature": 0.2,
    }


def send_completion(
    base_url: str,
    api_key: str,
    payload: Dict[str, Any],
    timeout: int = 180,
    max_retries: int = 5,
    initial_backoff: float = 2.0,
) -> str:
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    retryable_statuses = {429}
    backoff = max(initial_backoff, 1.0)

    for attempt in range(max_retries + 1):
        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=timeout,
            )
        except requests.RequestException as exc:
            if attempt < max_retries:
                print(
                    "[warn] Network error calling model API. "
                    f"Retrying in {backoff:.1f}s..."
                )
                time.sleep(backoff)
                backoff *= 2
                continue
            raise

        status = response.status_code
        if status in retryable_statuses or 500 <= status < 600:
            if attempt < max_retries:
                retry_after = response.headers.get("Retry-After")
                wait_time = backoff
                if retry_after:
                    try:
                        wait_time = float(retry_after)
                    except (TypeError, ValueError):
                        wait_time = backoff
                print(
                    f"[warn] HTTP {status} from model API. Retrying in {wait_time:.1f}s..."
                )
                time.sleep(wait_time)
                backoff = max(backoff * 2, 1.0)
                continue

        response.raise_for_status()
        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError("Model response missing choices.")
        message = choices[0].get("message") or {}
        content = message.get("content")
        if not content:
            raise RuntimeError("Model response missing content.")
        return content

    raise RuntimeError("Exceeded maximum retries when calling model API.")


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_summary_markdown(principle: str, data: str, output_path: Path) -> None:
    try:
        parsed = json.loads(output_path.read_text(
            encoding="utf-8")) if output_path.exists() else {}
    except json.JSONDecodeError:
        parsed = {}
    parsed[principle] = data
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(
        parsed, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[info] Cached markdown summary for '{principle}'.")


def sanitize_model_json(raw: str) -> str:
    """Remove common Markdown code fences and trim whitespace."""
    cleaned = raw.strip()
    if cleaned.lower().startswith("```json"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else ""
    elif cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else ""
    if cleaned.endswith("```"):
        cleaned = cleaned.rsplit("```", 1)[0]
    return cleaned.strip()


def build_json_conversion_payload(
    principle: str,
    markdown_text: str,
    prompt_template: str,
    model: str,
) -> Dict:
    prompt_text = prompt_template.replace("{PRINCIPLE_NAME}", principle).replace(
        "{model_response}", markdown_text
    )
    return {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a precise formatter that converts structured text into valid JSON."
                ),
            },
            {"role": "user", "content": prompt_text.strip()},
        ],
        "temperature": 0,
    }


def write_summary_json(principle: str, data: str, output_path: Path) -> None:
    cleaned_data = sanitize_model_json(data)
    try:
        parsed = json.loads(cleaned_data)
    except json.JSONDecodeError as exc:
        DEBUG_JSON_DIR.mkdir(parents=True, exist_ok=True)
        debug_path = DEBUG_JSON_DIR / f"{slugify(principle)}.txt"
        debug_payload = (
            "=== raw ===\n"
            f"{data}\n\n"
            "=== cleaned ===\n"
            f"{cleaned_data}\n"
        )
        debug_path.write_text(debug_payload, encoding="utf-8")
        raise RuntimeError(
            f"Model returned invalid JSON for '{principle}': {exc}"
        ) from exc

    existing: Dict[str, Dict] = {}
    if output_path.exists():
        try:
            existing = json.loads(output_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(
                f"[warn] Existing JSON at {output_path} is invalid; ignoring cache."
            )
            existing = {}

    existing[principle] = parsed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(
        existing, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[info] Updated JSON summary for '{principle}' in {output_path}")


def gather_api_credentials() -> List[Tuple[str, str]]:
    credentials: List[Tuple[str, str]] = []
    base_default = os.getenv("BASE_URL_LIMIT")
    for idx in range(10):
        suffix = "" if idx == 0 else str(idx)
        api_key = os.getenv(f"API_KEY_LIMIT{suffix}")
        if not api_key:
            continue
        base_env = os.getenv(f"BASE_URL_LIMIT{suffix}") or base_default
        if not base_env:
            print(
                f"[warn] Missing BASE_URL_LIMIT{suffix or ''} for available API key; skipping.")
            continue
        credentials.append((base_env.rstrip("/"), api_key))
    return credentials


class PrincipleWorker(threading.Thread):
    def __init__(
        self,
        name: str,
        task_queue: "queue.Queue[str]",
        credentials: Tuple[str, str],
        principles_data: Dict[str, List[Dict]],
        output_dir: Path,
        max_docs: Optional[int],
        model: str,
        sleep_seconds: float,
        overwrite: bool,
        raw_output_path: Path,
        json_output_path: Path,
        skip_principles: Optional[Set[str]] = None,
    ) -> None:
        super().__init__(name=name)
        self.task_queue = task_queue
        self.base_url, self.api_key = credentials
        self.principles_data = principles_data
        self.output_dir = output_dir
        self.max_docs = max_docs
        self.model = model
        self.sleep_seconds = sleep_seconds
        self.overwrite = overwrite
        self.raw_output_path = raw_output_path
        self.json_output_path = json_output_path
        self.skip_principles = skip_principles or set()
        self.daemon = True

    def run(self) -> None:
        while True:
            try:
                principle = self.task_queue.get_nowait()
            except queue.Empty:
                return

            try:
                self.process_principle(principle)
            except Exception as exc:
                print(
                    f"[error] Worker {self.name} failed for '{principle}': {exc}")
            finally:
                self.task_queue.task_done()
                time.sleep(self.sleep_seconds)

    def process_principle(self, principle: str) -> None:
        if not self.overwrite and principle in self.skip_principles:
            print(f"[info] Skipping '{principle}' (already completed).")
            return

        entries = self.principles_data.get(principle, [])
        if not entries:
            print(f"[warn] No entries for principle '{principle}'. Skipping.")
            return

        corpus = collect_corpus_for_principle(
            principle, entries, self.max_docs)
        if not corpus:
            print(f"[warn] No corpus gathered for '{principle}'. Skipping.")
            return

        corpus_payload = build_corpus_payload(corpus)
        prompt_template, prompt_type = choose_prompt(principle)
        payload = build_request_payload(
            principle, corpus_payload, prompt_template, prompt_type, self.model)
        summary_markdown = send_completion(
            self.base_url, self.api_key, payload)
        write_summary_markdown(
            principle, summary_markdown, self.raw_output_path)

        json_prompt = sd_to_json if prompt_type == "sd" else td_to_json
        json_payload = build_json_conversion_payload(
            principle, summary_markdown, json_prompt, self.model)
        summary_json = send_completion(
            self.base_url, self.api_key, json_payload)
        write_summary_json(principle, summary_json, self.json_output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize principle literature corpus using multiple API keys."
    )
    parser.add_argument(
        "--principles",
        nargs="*",
        help="Optional list of principles to process. Defaults to all principles found.",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=50,
        help="Maximum number of documents to load per principle (default: 50).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Model name to request (default from GEMINI_MODEL env).",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=DEFAULT_SLEEP_SECONDS,
        help="Seconds to sleep between requests per worker (default 15).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs.",
    )
    args = parser.parse_args()

    credentials = gather_api_credentials()
    if not credentials:
        raise RuntimeError(
            "No API credentials found in API_KEY_LIMIT…API_KEY_LIMIT9.")

    principles_data = load_principles()
    all_principles = list(args.principles) if args.principles else list(
        principles_data.keys())
    if not all_principles:
        print("No principles to process.")
        return

    ensure_output_dir(OUTPUT_DIR)

    existing_raw: Dict[str, Any] = {}
    existing_json: Dict[str, Any] = {}
    already_done: set[str] = set()
    if RAW_OUTPUT_PATH.exists() and not args.overwrite:
        try:
            existing_raw = json.loads(
                RAW_OUTPUT_PATH.read_text(encoding="utf-8"))
            already_done.update(existing_raw.keys())
        except json.JSONDecodeError:
            existing_raw = {}
    if JSON_OUTPUT_PATH.exists() and not args.overwrite:
        try:
            existing_json = json.loads(
                JSON_OUTPUT_PATH.read_text(encoding="utf-8"))
            already_done.update(existing_json.keys())
        except json.JSONDecodeError:
            existing_json = {}

    task_queue: "queue.Queue[str]" = queue.Queue()
    for principle in all_principles:
        if args.overwrite or principle not in already_done:
            task_queue.put(principle)

    if task_queue.empty():
        print("[info] No principles to process (everything already completed).")
        return

    workers: List[PrincipleWorker] = []
    worker_paths: List[Tuple[Path, Path]] = []
    for idx, cred in enumerate(credentials):
        raw_output_path = RAW_OUTPUT_PATH.with_name(
            f"{RAW_OUTPUT_PATH.stem}_{idx+1}.json"
        )
        json_output_path = JSON_OUTPUT_PATH.with_name(
            f"{JSON_OUTPUT_PATH.stem}_{idx+1}.json"
        )
        worker_paths.append((raw_output_path, json_output_path))
        worker = PrincipleWorker(
            name=f"worker-{idx+1}",
            task_queue=task_queue,
            credentials=cred,
            principles_data=principles_data,
            output_dir=OUTPUT_DIR,
            max_docs=args.max_docs,
            model=args.model,
            sleep_seconds=args.sleep_seconds,
            overwrite=args.overwrite,
            raw_output_path=raw_output_path,
            json_output_path=json_output_path,
            skip_principles=already_done if not args.overwrite else set(),
        )
        worker.start()
        workers.append(worker)

    task_queue.join()
    for worker in workers:
        worker.join()
    print("[done] All principle summaries generated.")

    combined_raw = {} if args.overwrite else existing_raw.copy()
    combined_json = {} if args.overwrite else existing_json.copy()

    for raw_path, json_path in worker_paths:
        if raw_path.exists():
            try:
                data = json.loads(raw_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                data = {}
            combined_raw.update(data)
        if json_path.exists():
            try:
                data = json.loads(json_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                data = {}
            combined_json.update(data)

    if combined_raw:
        RAW_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        RAW_OUTPUT_PATH.write_text(json.dumps(
            combined_raw, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[info] Aggregated raw summaries written to {RAW_OUTPUT_PATH}")
    if combined_json:
        JSON_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        JSON_OUTPUT_PATH.write_text(json.dumps(
            combined_json, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            f"[info] Aggregated JSON summaries written to {JSON_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
