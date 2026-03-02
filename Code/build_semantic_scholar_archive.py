import argparse
import asyncio
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from difflib import SequenceMatcher
from hashlib import sha1
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urljoin, quote

import requests
from xml.etree import ElementTree as ET
from collections import Counter


RESPONSES_PATH = Path("Dataset/gemini_references/responses_parsed.json")
BASE_DIR = Path("Dataset/papers_info")
PAPERS_DIR = BASE_DIR / "papers"
PDF_DIR = BASE_DIR / "pdfs"
TEXT_DIR = BASE_DIR / "texts"
LOG_DIR = BASE_DIR / "logs"
PRINCIPLES_INDEX_PATH = BASE_DIR / "principles.json"
NOT_FOUND_LOG = LOG_DIR / "not_found.jsonl"
ERROR_LOG = LOG_DIR / "errors.jsonl"
PIPELINE_STATE_PATH = LOG_DIR / "pipeline_state.json"

# PDF 下载参数（超时时间、最大重试次数、退避因子、重试状态码和默认 UA）
PDF_DOWNLOAD_TIMEOUT = (10, 120)  # (connect timeout, read timeout) in seconds
PDF_DOWNLOAD_MAX_ATTEMPTS = 3
PDF_DOWNLOAD_BACKOFF_FACTOR = 2.0
PDF_DOWNLOAD_RETRY_STATUS = {429, 500, 502, 503, 504}
PDF_DOWNLOAD_USER_AGENT = os.getenv(
    "PDF_DOWNLOAD_USER_AGENT",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36",
)

# 在两类场景会“直接读取 PDF 并提取文本”
# 下载失败兜底：如果 openAccessPdf 有链接，但多次下载原始 PDF 仍失败（或用户设置了 --skip-pdf），为了至少保留文本内容，会调用 fetch_pdf_text_content 直接抓取数据并用 pdfminer.six 提取文字。
# 用户主动跳过 PDF：当命令行加了 --skip-pdf（或把 --pdf-workers 设为 0），程序不会保存原文件，但如果能获取到 PDF，就尝试只取文本，避免完全没有内容。
PDF_TEXT_MAX_BYTES = int(
    os.getenv("PDF_TEXT_MAX_BYTES", str(25 * 1024 * 1024)))  # 直接读取 PDF 内容并提取文本时最多拉取多少字节的数据。默认值是 25 MB
PIPELINE_SENTINEL = object()
PROGRESS_INTERVAL_SECONDS = float(
    os.getenv("PIPELINE_PROGRESS_INTERVAL", "120"))  # 每隔多少秒打印一次进度

# Regular expressions for citation parsing
NUMBERED_PREFIX_RE = re.compile(r"^\d+[\).\s]+")
MARKDOWN_MARKERS_RE = re.compile(r"[\*\u2013\u2014]")
TRAILING_NOTE_RE = re.compile(r"\s*\([^)]*[\u4e00-\u9fff][^)]*\)\s*$")
DOI_RE = re.compile(r"(10\.\d{4,9}/[^\s\)]+)", re.IGNORECASE)

# Default configurations
SEARCH_FIELDS_DEFAULT = "paperId,title,abstract,authors,venue,year,citationCount,openAccessPdf"
DEEP_SEARCH_MODEL_DEFAULT = "gemini-2.5-pro-preview-06-05-search"

# Cache for Semantic Scholar base URL
_S2_BASE_URL_CACHE: Optional[str] = None

# Supported academic databases for fallback search:
# - ArXiv: preprints in physics, mathematics, computer science
# - PubMed: biomedical and life sciences literature
# - OpenAlex: comprehensive scholarly works database
# - Crossref: DOI registration and metadata
# - PsyArXiv: psychology preprints via OSF
# - DOAJ: open access journals


def get_s2_base_url() -> str:
    global _S2_BASE_URL_CACHE
    if _S2_BASE_URL_CACHE is None:
        base = os.getenv("S2_BASE_URL")
        if not base:
            raise RuntimeError("S2_BASE_URL environment variable is not set.")
        base = base.rstrip("/") + "/"
        if "graph/v1/" not in base:
            base = base + "graph/v1/"
        _S2_BASE_URL_CACHE = base
    return _S2_BASE_URL_CACHE


@dataclass
class CitationInfo:
    principle: str
    citation_raw: str
    index: int
    doi: Optional[str]
    title_hint: Optional[str]
    year_hint: Optional[int]


@dataclass
class PipelineTask:
    info: CitationInfo
    paper_key: Optional[str]
    metadata: Optional[Dict]
    pdf_url: Optional[str]
    lookup_sources: List[str]
    needs_pdf_download: bool
    needs_deep_search: bool

    @property
    def key(self) -> Optional[str]:
        return self.paper_key


class PipelineState:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data: Dict[str, Dict[str, Any]] = {}
        if path.exists():
            try:
                self._data = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                self._data = {}
        self._lock = asyncio.Lock()

    def get(self, key: str) -> Dict[str, Any]:
        return self._data.get(key, {})

    async def update(self, key: str, update: Dict[str, Any]) -> None:
        async with self._lock:
            record = dict(self._data.get(key, {}))
            record.update(update)
            record["updated_at"] = time.time()
            self._data[key] = record
            tmp_path = self.path.with_suffix(".tmp")
            tmp_path.write_text(json.dumps(
                self._data, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp_path.replace(self.path)

    def items(self):
        return self._data.items()

    def snapshot(self) -> Dict[str, Dict[str, Any]]:
        return {key: dict(value) for key, value in self._data.items()}


def collect_deep_search_credentials(max_keys: int = 10) -> List[Tuple[str, str]]:
    credentials: List[Tuple[str, str]] = []
    base_default = os.getenv("BASE_URL_LIMIT")
    for idx in range(max_keys):
        suffix = "" if idx == 0 else str(idx)
        api_key = os.getenv(f"API_KEY_LIMIT{suffix}")
        if not api_key:
            continue
        base_env = os.getenv(f"BASE_URL_LIMIT{suffix}") or base_default
        if not base_env:
            print(
                f"[warn] BASE_URL_LIMIT{suffix or ''} not set for available API key; skipping."
            )
            continue
        credentials.append((base_env.rstrip("/"), api_key))
    return credentials


def normalize_title(title: str) -> str:
    cleaned = re.sub(r"\s+", " ", title).strip().lower()
    cleaned = re.sub(r"[^a-z0-9\s]", "", cleaned)
    return re.sub(r"\s+", " ", cleaned)


def slugify(value: str, length: int = 80) -> str:
    value = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return value[:length] or sha1(value.encode("utf-8")).hexdigest()[:length]


class PaperRegistry:
    def __init__(self) -> None:
        self.papers_dir = PAPERS_DIR
        self.pdf_dir = PDF_DIR
        self.text_dir = TEXT_DIR
        self.papers_dir.mkdir(parents=True, exist_ok=True)
        self.pdf_dir.mkdir(parents=True, exist_ok=True)
        self.text_dir.mkdir(parents=True, exist_ok=True)
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.by_key: Dict[str, Dict] = {}
        self.by_doi: Dict[str, str] = {}
        self.by_paper_id: Dict[str, str] = {}
        self.by_title: Dict[str, str] = {}
        self._load_existing()

    def _load_existing(self) -> None:
        for path in self.papers_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            paper_key = path.stem
            self.by_key[paper_key] = data
            doi = data.get("doi")
            if doi:
                self.by_doi[doi.lower()] = paper_key
            paper_id = data.get("paperId")
            if paper_id:
                self.by_paper_id[paper_id] = paper_key
            title = data.get("title")
            if title:
                self.by_title[normalize_title(title)] = paper_key

    def get_by_identifiers(self, doi: Optional[str], title: Optional[str]) -> Optional[Tuple[str, Dict]]:
        if doi:
            key = self.by_doi.get(doi.lower())
            if key:
                return key, self.by_key[key]
        if title:
            norm = normalize_title(title)
            key = self.by_title.get(norm)
            if key:
                return key, self.by_key[key]
        return None

    def get_by_key(self, paper_key: str) -> Optional[Dict]:
        return self.by_key.get(paper_key)

    def build_key(self, metadata: Dict, default_title: str) -> str:
        if metadata.get("paperId"):
            # Sanitize paperId to ensure it's a valid filename
            paper_id = metadata['paperId']
            # Remove or replace problematic characters
            sanitized_id = paper_id.replace(
                'https://', '').replace('http://', '').replace('/', '_').replace(':', '_')
            return f"s2_{sanitized_id}"
        doi = metadata.get("doi")
        if doi:
            return f"doi_{slugify(doi)}"
        base = metadata.get("title") or default_title or "unknown"
        norm_title = slugify(base)
        year = metadata.get("year")
        suffix = f"_{year}" if year else ""
        raw = f"{norm_title}{suffix}"
        if raw not in self.by_key:
            return raw
        digest = sha1(base.encode("utf-8")).hexdigest()[:8]
        return f"{raw}_{digest}"

    def save_metadata(self, paper_key: str, metadata: Dict) -> None:
        metadata = dict(metadata)
        metadata["paper_key"] = paper_key
        path = self.papers_dir / f"{paper_key}.json"
        path.write_text(json.dumps(
            metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        self.by_key[paper_key] = metadata
        doi = metadata.get("doi")
        if doi:
            self.by_doi[doi.lower()] = paper_key
        paper_id = metadata.get("paperId")
        if paper_id:
            self.by_paper_id[paper_id] = paper_key
        title = metadata.get("title")
        if title:
            self.by_title[normalize_title(title)] = paper_key

    def pdf_path(self, paper_key: str) -> Path:
        return self.pdf_dir / f"{paper_key}.pdf"

    def pdf_exists(self, paper_key: str) -> bool:
        return self.pdf_path(paper_key).exists()

    def text_path(self, paper_key: str) -> Path:
        return self.text_dir / f"{paper_key}.txt"

    def text_exists(self, paper_key: str) -> bool:
        return self.text_path(paper_key).exists()

    def save_text(self, paper_key: str, content: str) -> None:
        path = self.text_path(paper_key)
        path.write_text(content, encoding="utf-8")


def parse_citation(principle: str, citation: str, idx: int) -> CitationInfo:
    cleaned = citation.replace("**", "")
    cleaned = MARKDOWN_MARKERS_RE.sub("", cleaned).strip()

    cleaned = re.split(r"\s(?:#{2,}|###)\s", cleaned)[0].strip()

    while True:
        updated = TRAILING_NOTE_RE.sub("", cleaned)
        if updated == cleaned:
            break
        cleaned = updated.strip()

    doi_match = DOI_RE.search(cleaned)
    doi = doi_match.group(1).rstrip(".") if doi_match else None

    title = None
    year_value: Optional[int] = None
    year_match = re.search(r"\(\d{4}\)", cleaned)
    if year_match:
        try:
            year_value = int(year_match.group(0).strip("()"))
        except ValueError:
            year_value = None
        after_year = cleaned[year_match.end():].lstrip(". ").strip()
        if after_year:
            title_candidate = re.split(
                r"\.\s+", after_year, maxsplit=1)[0].strip(" ,;:")
            if title_candidate and not title_candidate.lower().startswith("in "):
                title = title_candidate

    if not title:
        segments = [seg.strip()
                    for seg in re.split(r"\.\s+", cleaned) if seg.strip()]
        for seg in segments:
            if re.search(r"\(\d{4}\)", seg):
                continue
            if seg.lower().startswith("in "):
                continue
            title = seg.strip(" ,;:")
            break

    if title:
        title = re.sub(r"\s+", " ", title).strip()

    return CitationInfo(
        principle=principle,
        citation_raw=citation,
        index=idx,
        doi=doi,
        title_hint=title or None,
        year_hint=year_value,
    )


def _normalize_name(name: str) -> str:
    return re.sub(r"[^a-z]", "", name.lower())


def extract_citation_last_names(citation: str) -> List[str]:
    header = citation.split("(", 1)[0]
    matches = re.findall(
        r"([A-Z][\w'\-]*(?:\s+[A-Z][\w'\-]*)*)\s*,", header)
    return [_normalize_name(match) for match in matches if match]


def extract_metadata_last_names(metadata: Dict) -> List[str]:
    authors = metadata.get("authors") or []
    last_names: List[str] = []
    for author in authors:
        if not isinstance(author, dict):
            continue
        name = author.get("name")
        if not name:
            continue
        parts = name.split()
        if not parts:
            continue
        last_names.append(_normalize_name(parts[-1]))
    return last_names


def titles_similar(meta_title: str, citation_title: str, min_ratio: float = 0.6) -> bool:
    norm_meta = normalize_title(meta_title)
    norm_citation = normalize_title(citation_title)
    if not norm_meta or not norm_citation:
        return False
    if norm_meta == norm_citation:
        return True
    if norm_meta in norm_citation or norm_citation in norm_meta:
        return True
    ratio = SequenceMatcher(None, norm_meta, norm_citation).ratio()
    return ratio >= min_ratio


def metadata_matches_citation(metadata: Dict, info: CitationInfo) -> bool:
    meta_doi = (metadata.get("doi") or "").lower()
    info_doi = (info.doi or "").lower()
    if info_doi and meta_doi and info_doi == meta_doi:
        return True

    if info.title_hint:
        meta_title = metadata.get("title")
        if not meta_title:
            return False
        if not titles_similar(meta_title, info.title_hint):
            return False

    if info.year_hint is not None:
        meta_year = metadata.get("year")
        try:
            meta_year_int = int(meta_year) if meta_year is not None else None
        except (TypeError, ValueError):
            meta_year_int = None
        if meta_year_int is not None and abs(meta_year_int - info.year_hint) > 1:
            return False

    citation_last_names = extract_citation_last_names(info.citation_raw)
    metadata_last_names = extract_metadata_last_names(metadata)
    if citation_last_names and metadata_last_names:
        if not any(name in metadata_last_names for name in citation_last_names):
            return False

    return True


class PdfLinkExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.pdf_url: Optional[str] = None

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        if self.pdf_url:
            return
        attrs_dict = {key.lower(): value for key, value in attrs if key}
        if tag == "meta":
            name = (attrs_dict.get("name") or attrs_dict.get(
                "property") or "").lower()
            if name == "citation_pdf_url":
                content = attrs_dict.get("content")
                if content:
                    self.pdf_url = content.strip()
        elif tag in {"a", "link"}:
            href = attrs_dict.get("href")
            if href and href.lower().endswith(".pdf"):
                self.pdf_url = href.strip()
        elif tag in {"iframe", "embed"}:
            src = attrs_dict.get("src")
            if src and src.lower().endswith(".pdf"):
                self.pdf_url = src.strip()


def extract_pdf_url(html_text: str, base_url: str) -> Optional[str]:
    parser = PdfLinkExtractor()
    parser.feed(html_text)
    parser.close()
    candidate = parser.pdf_url
    if candidate:
        return urljoin(base_url, unescape(candidate))
    match = re.search(
        r'href=["\']([^"\']+\.pdf(?:\?[^"\']*)?)["\']', html_text, re.IGNORECASE)
    if match:
        return urljoin(base_url, unescape(match.group(1)))
    return None


def _format_authors(metadata: Dict) -> str:
    authors = metadata.get("authors") or []
    names = [author.get("name")
             for author in authors if isinstance(author, dict)]
    filtered = [name for name in names if name]
    return ", ".join(filtered) if filtered else "Unknown"


def deep_search_paper(
    metadata: Dict,
    info: CitationInfo,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> Optional[str]:
    base_url = base_url or os.getenv("BASE_URL_LIMIT")
    api_key = api_key or os.getenv("API_KEY_LIMIT")
    if not base_url or not api_key:
        print("[warn] Deep search skipped: BASE_URL_LIMIT or API_KEY_LIMIT not set.")
        return None

    model = os.getenv("DEEP_SEARCH_MODEL", DEEP_SEARCH_MODEL_DEFAULT)
    title = metadata.get("title") or info.title_hint or info.citation_raw
    authors = _format_authors(metadata)
    year = metadata.get("year")
    doi = metadata.get("doi") or info.doi or "Unknown"
    venue = metadata.get("venue") or ""

    system_prompt = (
        "You are an expert-level academic research assistant with privileged Deep Search capabilities. "
        "Your primary mission is to locate and deeply analyze the academic paper specified by the user. "
        "You must strictly follow this workflow hierarchy:\n"

        "1.  **Primary Objective: Retrieve Official Abstract and Full-Text Synopsis.**"
        "    - Prioritize searching official channels (e.g., the publisher's website, arXiv, Google Scholar, DBLP, author's personal homepage, university repositories) to ensure authoritative information."
        "    - **Goal 1 (Abstract):** Locate and **quote verbatim** the official abstract."
        "    - **Goal 2 (Full Text):** If the full text (HTML or PDF) is accessible and parsable by you, provide a comprehensive synopsis structured **according to the paper's original sections** (e.g., Introduction, Methods, Results, Discussion/Conclusion). This synopsis must quote key text where possible and clearly summarize the core arguments, findings, and conclusions of each section."

        "2.  **Fallback Objective: Retrieve Detailed Interpretation.**"
        "    - **Trigger:** This objective is triggered **if and only if** the full text is inaccessible or its content cannot be parsed. You must explicitly inform the user 'Full text was not accessible' and then automatically proceed with this task."
        "    - **Goal:** Retrieve and summarize a **Detailed Interpretation** or **Critical Analysis** of the paper from reliable third-party sources (e.g., authoritative academic blogs, in-depth technical forum analyses, or citations and commentary in related review papers)."

        "3.  **Citation Mandate:**"
        "    - All information you provide—whether it is the abstract, synopsis, or interpretation—must be accompanied by clear, verifiable source URLs."
        "    - You must strictly adhere to the user's requested output structure."
    )

    user_prompt_lines = [
        "### Target Paper",
        f"**Title:** {title}",
        f"**Authors:** {authors}",
        f"**Year:** {year or 'Unknown'}",
        f"**Venue:** {venue or 'Unknown'}",
        f"**DOI:** {doi}",

        "\n### Task Directive",
        "Execute a deep search immediately and return the results in the following strict Markdown format:",

        "\n## 1. Official Abstract",
        "(**Requirement:** Quote the official abstract verbatim. **Requirement:** Cite the source URL.)",

        "\n## 2. Core Insights & Analysis",
        "(**Primary Task:** If the full text is accessible, provide a detailed synopsis organized by the paper's original sections (Introduction, Methods, Results, etc.). Cite sources for any quotes.)",
        "(**Fallback Task:** If the full text is inaccessible, state that clearly here, and provide a **Detailed Interpretation** or **Critical Analysis** from third-party sources instead. Cite all sources.)",

        "\n## 3. Sources",
        "(**Requirement:** List all verifiable URLs used to complete Task 1 and Task 2.)"
    ]

    user_prompt = "\n".join(user_prompt_lines)

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
    }

    url = base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    try:
        print(f"[info] Initiating deep search for '{title}'...")
        response = requests.post(url, headers=headers,
                                 json=payload, timeout=120)  # Reduced from 180s
        response.raise_for_status()
    except requests.Timeout:
        print(f"[error] Deep search timed out (120s) for '{title}'")
        return None
    except requests.RequestException as exc:
        print(f"[error] Deep search request failed for '{title}': {exc}")
        return None

    try:
        data = response.json()
    except ValueError:
        print(f"[error] Deep search returned non-JSON payload for '{title}'.")
        return None

    choices = data.get("choices") or []
    if not choices:
        print(f"[warn] Deep search returned empty choices for '{title}'.")
        return None

    message = choices[0].get("message") or {}
    content = message.get("content")
    if content:
        return content.strip()
    print(f"[warn] Deep search response missing content for '{title}'.")
    return None


def load_responses(principles: Optional[Iterable[str]] = None, limit: Optional[int] = None) -> List[CitationInfo]:
    data = json.loads(RESPONSES_PATH.read_text(encoding="utf-8"))
    selected = set(principles) if principles else None
    tasks: List[CitationInfo] = []
    for principle, citations in data.items():
        if selected and principle not in selected:
            continue
        upper = limit if limit is not None else len(citations)
        for idx, citation in enumerate(citations[:upper], start=1):
            tasks.append(parse_citation(principle, citation, idx))
    return tasks


def semantic_scholar_request(
    url: str,
    params: Dict,
    session: requests.Session,
    retries: int = 3,
    backoff: float = 2.0,
) -> Optional[Dict]:
    api_key = os.getenv("S2_API_KEY")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    for attempt in range(1, retries + 1):
        try:
            print(f"[request] attempt={attempt} URL={url} params={params}")
            resp = session.get(url, params=params, headers=headers, timeout=30)
            print(f"[response] status={resp.status_code} url={resp.url}")
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            if hasattr(exc, "response") and exc.response is not None:
                try:
                    body = exc.response.text
                except Exception:
                    body = "<unavailable>"
                print(
                    f"[error] status={exc.response.status_code} url={exc.response.url} body={body[:200]}")
            else:
                print(f"[error] request failed: {exc}")
            if attempt == retries:
                raise exc
            time.sleep(backoff * attempt)
    return None


def search_arxiv(info: CitationInfo, session: requests.Session) -> Optional[Dict]:
    """Search ArXiv API for paper metadata"""
    try:
        base_url = "http://export.arxiv.org/api/query?"
        if info.title_hint:
            query = f"ti:{quote(info.title_hint)}"
        elif info.doi:
            return None  # ArXiv doesn't support DOI search directly
        else:
            return None

        params = {
            "search_query": query,
            "start": 0,
            "max_results": 1
        }
        # ArXiv can be slow, add retry logic
        for attempt in range(2):
            try:
                response = session.get(base_url, params=params, timeout=30)
                response.raise_for_status()
                break
            except (requests.Timeout, requests.HTTPError) as e:
                if attempt == 0 and (isinstance(e, requests.Timeout) or (hasattr(e, 'response') and e.response.status_code == 503)):
                    print(
                        f"[warn] ArXiv attempt {attempt + 1} failed, retrying...")
                    time.sleep(2)
                    continue
                raise

        root = ET.fromstring(response.content)
        entries = root.findall("{http://www.w3.org/2005/Atom}entry")
        if not entries:
            return None

        entry = entries[0]
        title = entry.find("{http://www.w3.org/2005/Atom}title").text.strip()
        summary = entry.find(
            "{http://www.w3.org/2005/Atom}summary").text.strip()
        published = entry.find(
            "{http://www.w3.org/2005/Atom}published").text[:4]

        authors = []
        for author in entry.findall("{http://www.w3.org/2005/Atom}author"):
            name = author.find("{http://www.w3.org/2005/Atom}name").text
            authors.append({"name": name})

        pdf_link = None
        for link in entry.findall("{http://www.w3.org/2005/Atom}link"):
            if link.get("title") == "pdf":
                pdf_link = link.get("href")
                break

        print(f"[info] ArXiv search succeeded for '{info.title_hint}'")
        return {
            "paperId": None,
            "title": title,
            "abstract": summary,
            "authors": authors,
            "year": int(published),
            "venue": "arXiv",
            "citationCount": 0,
            "doi": info.doi,
            "openAccessPdf": {"url": pdf_link} if pdf_link else None,
            "source": "arxiv"
        }
    except Exception as e:
        print(f"[warn] ArXiv search failed: {e}")
        return None


def search_pubmed(info: CitationInfo, session: requests.Session) -> Optional[Dict]:
    """Search PubMed API for paper metadata"""
    try:
        email = os.getenv("PUBMED_EMAIL", "your.email@example.com")

        # First, search for the paper
        search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        search_params = {
            "db": "pubmed",
            "term": info.doi if info.doi else info.title_hint,
            "retmode": "json",
            "retmax": 1,
            "email": email
        }

        response = session.get(search_url, params=search_params, timeout=30)
        response.raise_for_status()
        search_result = response.json()

        id_list = search_result.get("esearchresult", {}).get("idlist", [])
        if not id_list:
            return None

        # Fetch details
        fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        fetch_params = {
            "db": "pubmed",
            "id": id_list[0],
            "retmode": "xml",
            "email": email
        }

        response = session.get(fetch_url, params=fetch_params, timeout=30)
        response.raise_for_status()

        root = ET.fromstring(response.content)
        article = root.find(".//Article")
        if article is None:
            return None

        title_elem = article.find(".//ArticleTitle")
        title = title_elem.text if title_elem is not None else None

        abstract_elem = article.find(".//AbstractText")
        abstract = abstract_elem.text if abstract_elem is not None else None

        year_elem = article.find(".//PubDate/Year")
        year = int(year_elem.text) if year_elem is not None else None

        authors = []
        for author in article.findall(".//Author"):
            lastname = author.find("LastName")
            forename = author.find("ForeName")
            if lastname is not None and forename is not None:
                authors.append({"name": f"{forename.text} {lastname.text}"})

        journal = article.find(".//Journal/Title")
        venue = journal.text if journal is not None else "PubMed"

        # Try to get DOI
        doi = info.doi
        for article_id in root.findall(".//ArticleId"):
            if article_id.get("IdType") == "doi":
                doi = article_id.text
                break

        # PubMed Central link for PDF
        pmc_id = None
        for article_id in root.findall(".//ArticleId"):
            if article_id.get("IdType") == "pmc":
                pmc_id = article_id.text
                break

        pdf_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmc_id}/pdf/" if pmc_id else None

        print(
            f"[info] PubMed search succeeded for '{info.title_hint or info.doi}'")
        return {
            "paperId": f"PMID:{id_list[0]}",
            "title": title,
            "abstract": abstract,
            "authors": authors,
            "year": year,
            "venue": venue,
            "citationCount": 0,
            "doi": doi,
            "openAccessPdf": {"url": pdf_url} if pdf_url else None,
            "source": "pubmed"
        }
    except Exception as e:
        print(f"[warn] PubMed search failed: {e}")
        return None


def search_openalex(info: CitationInfo, session: requests.Session) -> Optional[Dict]:
    """Search OpenAlex API for paper metadata"""
    try:
        base_url = "https://api.openalex.org/works"
        email = os.getenv("OPENALEX_EMAIL", "your.email@example.com")

        if info.doi:
            doi_url = f"https://doi.org/{info.doi}"
            search_url = f"https://api.openalex.org/works/{doi_url}"
            params = {"mailto": email}
        elif info.title_hint:
            params = {
                "filter": f'title.search:"{info.title_hint}"',
                "mailto": email,
                "per_page": 1
            }
            search_url = base_url
        else:
            return None

        response = session.get(search_url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        # Handle different response formats
        if "results" in data:
            if not data["results"]:
                return None
            work = data["results"][0]
        else:
            work = data

        title = work.get("title")
        abstract = work.get("abstract_inverted_index")
        if abstract:
            # Convert inverted index to text
            words = [""] * max(max(positions)
                               for positions in abstract.values())
            for word, positions in abstract.items():
                for pos in positions:
                    if pos < len(words):
                        words[pos] = word
            abstract = " ".join(words)

        authors = []
        for authorship in work.get("authorships", []):
            author = authorship.get("author", {})
            name = author.get("display_name")
            if name:
                authors.append({"name": name})

        year = work.get("publication_year")
        venue = work.get("primary_location", {}).get(
            "source", {}).get("display_name", "")

        pdf_url = None
        if work.get("open_access", {}).get("is_oa"):
            pdf_url = work.get("open_access", {}).get("oa_url")

        # Extract clean paper ID from OpenAlex URL
        openalex_id = work.get("id", "")
        if openalex_id.startswith("https://openalex.org/"):
            openalex_id = openalex_id.replace("https://openalex.org/", "")
        paper_id = f"openalex:{openalex_id}" if openalex_id else None

        print(
            f"[info] OpenAlex search succeeded for '{info.title_hint or info.doi}'")
        return {
            "paperId": paper_id,
            "title": title,
            "abstract": abstract,
            "authors": authors,
            "year": year,
            "venue": venue,
            "citationCount": work.get("cited_by_count", 0),
            "doi": work.get("doi", "").replace("https://doi.org/", "") if work.get("doi") else info.doi,
            "openAccessPdf": {"url": pdf_url} if pdf_url else None,
            "source": "openalex"
        }
    except Exception as e:
        print(f"[warn] OpenAlex search failed: {e}")
        return None


def search_crossref(info: CitationInfo, session: requests.Session) -> Optional[Dict]:
    """Search Crossref API for paper metadata"""
    try:
        email = os.getenv("CROSSREF_EMAIL", "your.email@example.com")

        if info.doi:
            url = f"https://api.crossref.org/works/{info.doi}"
            params = {"mailto": email}
        elif info.title_hint:
            url = "https://api.crossref.org/works"
            params = {
                "query.title": info.title_hint,
                "rows": 1,
                "mailto": email
            }
        else:
            return None

        response = session.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        if "message" not in data:
            return None

        # Handle different response formats
        if "items" in data["message"]:
            if not data["message"]["items"]:
                return None
            work = data["message"]["items"][0]
        else:
            work = data["message"]

        title_list = work.get("title", [])
        title = title_list[0] if title_list else None

        abstract = work.get("abstract")

        authors = []
        for author in work.get("author", []):
            given = author.get("given", "")
            family = author.get("family", "")
            name = f"{given} {family}".strip()
            if name:
                authors.append({"name": name})

        pub_date = work.get(
            "published-print") or work.get("published-online") or work.get("created")
        year = None
        if pub_date and "date-parts" in pub_date:
            date_parts = pub_date["date-parts"][0]
            if date_parts:
                year = date_parts[0]

        venue = ""
        if "container-title" in work:
            venue_list = work["container-title"]
            venue = venue_list[0] if venue_list else ""

        # Crossref doesn't directly provide PDF links, but we can check for open access
        pdf_url = None
        for link in work.get("link", []):
            if link.get("content-type") == "application/pdf":
                pdf_url = link.get("URL")
                break

        print(
            f"[info] Crossref search succeeded for '{info.title_hint or info.doi}'")

        # Use DOI as paperId, but ensure it's properly formatted
        crossref_doi = work.get("DOI")
        paper_id = f"crossref:{crossref_doi}" if crossref_doi else None

        return {
            "paperId": paper_id,
            "title": title,
            "abstract": abstract,
            "authors": authors,
            "year": year,
            "venue": venue,
            "citationCount": work.get("is-referenced-by-count", 0),
            "doi": crossref_doi or info.doi,
            "openAccessPdf": {"url": pdf_url} if pdf_url else None,
            "source": "crossref"
        }
    except Exception as e:
        print(f"[warn] Crossref search failed: {e}")
        return None


def search_psyarxiv(info: CitationInfo, session: requests.Session) -> Optional[Dict]:
    """Search PsyArXiv (OSF) API for paper metadata"""
    try:
        base_url = "https://api.osf.io/v2/preprints"

        if info.title_hint:
            params = {
                "filter[provider]": "psyarxiv",
                "filter[title]": info.title_hint,
                "page[size]": 1
            }
        elif info.doi:
            # Try to search by DOI
            params = {
                "filter[provider]": "psyarxiv",
                "filter[article_doi]": info.doi,
                "page[size]": 1
            }
        else:
            return None

        response = session.get(base_url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        if not data.get("data"):
            return None

        preprint = data["data"][0]
        attrs = preprint.get("attributes", {})

        title = attrs.get("title")
        abstract = attrs.get("description")
        date_published = attrs.get("date_published", "")
        year = int(date_published[:4]) if date_published else None

        # Get authors
        authors = []
        contributors = preprint.get("relationships", {}).get(
            "contributors", {}).get("links", {}).get("related", {}).get("href")
        if contributors:
            try:
                author_response = session.get(contributors, timeout=30)
                author_response.raise_for_status()
                author_data = author_response.json()
                for contributor in author_data.get("data", []):
                    embeds = contributor.get("embeds", {})
                    users = embeds.get("users", {})
                    if users and "data" in users:
                        full_name = users["data"].get(
                            "attributes", {}).get("full_name")
                        if full_name:
                            authors.append({"name": full_name})
            except Exception:
                pass

        # Get PDF link
        pdf_url = None
        links = preprint.get("links", {})
        preprint_doi = links.get("preprint_doi")
        if preprint_doi:
            pdf_url = preprint_doi.replace(
                "/preprints/", "/preprints/download/")

        print(f"[info] PsyArXiv search succeeded for '{info.title_hint}'")

        # Format paperId with source prefix
        psyarxiv_id = preprint.get("id")
        paper_id = f"psyarxiv:{psyarxiv_id}" if psyarxiv_id else None

        return {
            "paperId": paper_id,
            "title": title,
            "abstract": abstract,
            "authors": authors,
            "year": year,
            "venue": "PsyArXiv",
            "citationCount": 0,
            "doi": attrs.get("article_doi", info.doi),
            "openAccessPdf": {"url": pdf_url} if pdf_url else None,
            "source": "psyarxiv"
        }
    except Exception as e:
        print(f"[warn] PsyArXiv search failed: {e}")
        return None


def search_doaj(info: CitationInfo, session: requests.Session) -> Optional[Dict]:
    """Search DOAJ API for paper metadata"""
    try:
        base_url = "https://doaj.org/api/search/articles/"

        if info.doi:
            query = f'doi:"{info.doi}"'
        elif info.title_hint:
            query = f'title:"{info.title_hint}"'
        else:
            return None

        params = {
            "q": query,
            "pageSize": 1
        }

        response = session.get(base_url + query, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        results = data.get("results", [])
        if not results:
            return None

        article = results[0]
        bibjson = article.get("bibjson", {})

        title = bibjson.get("title")
        abstract = bibjson.get("abstract")
        year = bibjson.get("year")

        authors = []
        for author in bibjson.get("author", []):
            name = author.get("name")
            if name:
                authors.append({"name": name})

        journal = bibjson.get("journal", {})
        venue = journal.get("title", "")

        # Get PDF link
        pdf_url = None
        for link in bibjson.get("link", []):
            if link.get("type") == "fulltext":
                pdf_url = link.get("url")
                break

        # Get DOI
        doi = info.doi
        for identifier in bibjson.get("identifier", []):
            if identifier.get("type") == "doi":
                doi = identifier.get("id")
                break

        print(
            f"[info] DOAJ search succeeded for '{info.title_hint or info.doi}'")

        # Format paperId with source prefix
        doaj_id = article.get("id")
        paper_id = f"doaj:{doaj_id}" if doaj_id else None

        return {
            "paperId": paper_id,
            "title": title,
            "abstract": abstract,
            "authors": authors,
            "year": int(year) if year else None,
            "venue": venue,
            "citationCount": 0,
            "doi": doi,
            "openAccessPdf": {"url": pdf_url} if pdf_url else None,
            "source": "doaj"
        }
    except Exception as e:
        print(f"[warn] DOAJ search failed: {e}")
        return None


def fetch_metadata_from_multiple_sources(info: CitationInfo, session: requests.Session) -> Optional[Dict]:
    """
    Fetch metadata from multiple academic databases in parallel.
    Returns the first successful result with PDF, or the first successful result without PDF.
    """
    print(
        f"[info] Searching multiple databases for: {info.title_hint or info.doi}")

    # Define all search functions
    search_functions = [
        ("ArXiv", search_arxiv),
        ("PubMed", search_pubmed),
        ("OpenAlex", search_openalex),
        ("Crossref", search_crossref),
        ("PsyArXiv", search_psyarxiv),
        ("DOAJ", search_doaj),
    ]

    results_with_pdf = []
    results_without_pdf = []
    failed_sources = []

    # Execute searches in parallel
    with ThreadPoolExecutor(max_workers=6) as executor:
        future_to_source = {
            executor.submit(search_func, info, session): source_name
            for source_name, search_func in search_functions
        }

        for future in as_completed(future_to_source):
            source_name = future_to_source[future]
            try:
                result = future.result()
                if result:
                    if not metadata_matches_citation(result, info):
                        print(
                            f"[warn] {source_name} result rejected due to mismatch with citation")
                        continue
                    has_pdf = result.get(
                        "openAccessPdf") and result["openAccessPdf"].get("url")
                    if has_pdf:
                        results_with_pdf.append((source_name, result))
                        print(f"[success] {source_name} found paper WITH PDF")
                    else:
                        results_without_pdf.append((source_name, result))
                        print(
                            f"[success] {source_name} found paper WITHOUT PDF")
                else:
                    print(f"[info] {source_name} returned no results")
            except Exception as e:
                print(f"[error] {source_name} search raised exception: {e}")
                failed_sources.append(source_name)

    # Summary
    print(
        f"[summary] Results: {len(results_with_pdf)} with PDF, {len(results_without_pdf)} without PDF, {len(failed_sources)} failed")

    # Prefer results with PDF
    if results_with_pdf:
        source_name, result = results_with_pdf[0]
        print(f"[final] Using {source_name} result (has PDF)")
        return result

    # Fallback to results without PDF
    if results_without_pdf:
        source_name, result = results_without_pdf[0]
        print(f"[final] Using {source_name} result (no PDF available)")
        return result

    print("[final] No results found from any database")
    return None


def fetch_metadata(info: CitationInfo, session: requests.Session) -> Tuple[Optional[Dict], List[str]]:
    """
    Fetch metadata with fallback strategy:
    1. Try Semantic Scholar first
    2. If not found or no PDF, try other databases in parallel
    3. If still not found, return (None, []) to indicate no sources matched.

    Returns:
        A tuple of (metadata, sources), where `metadata` is the selected metadata
        dictionary (or None) and `sources` is a list of source identifiers that
        successfully returned a match.
    """
    fields = os.getenv("S2_SEARCH_FIELDS", SEARCH_FIELDS_DEFAULT)
    base_url = get_s2_base_url()
    s2_result = None
    s2_has_pdf = False
    sources_found: List[str] = []

    # Try Semantic Scholar first
    if info.doi:
        try:
            result = semantic_scholar_request(
                base_url + f"paper/DOI:{info.doi}",
                {"fields": fields},
                session,
            )
            if result:
                if metadata_matches_citation(result, info):
                    print(
                        f"[info] Semantic Scholar DOI lookup succeeded for {info.doi}")
                    s2_result = result
                    s2_result["source"] = "semantic_scholar"
                    s2_has_pdf = bool(result.get("openAccessPdf")
                                      and result["openAccessPdf"].get("url"))
                    if "semantic_scholar" not in sources_found:
                        sources_found.append("semantic_scholar")
                else:
                    print(
                        f"[warn] Semantic Scholar DOI result rejected due to mismatch with citation")
        except requests.RequestException:
            pass

    if not s2_result and info.title_hint:
        query_variants = [info.title_hint]
        precise = f'title:"{info.title_hint}"'
        if precise not in query_variants:
            query_variants.append(precise)
        for query in query_variants:
            try:
                result = semantic_scholar_request(
                    base_url + "paper/search",
                    {"query": query, "limit": 1, "fields": fields},
                    session,
                    retries=1,
                )
            except requests.RequestException:
                continue
            if result:
                data = result.get("data")
                if data:
                    print(f"[info] Semantic Scholar title search succeeded")
                    candidate = data[0]
                    if metadata_matches_citation(candidate, info):
                        s2_result = candidate
                        s2_result["source"] = "semantic_scholar"
                        s2_has_pdf = bool(s2_result.get(
                            "openAccessPdf") and s2_result["openAccessPdf"].get("url"))
                        if "semantic_scholar" not in sources_found:
                            sources_found.append("semantic_scholar")
                        break
                    print(
                        "[warn] Semantic Scholar title result rejected due to mismatch with citation")
                print(f"[info] Query '{query}' returned no matches.")

    # If Semantic Scholar found result with PDF, return it
    if s2_result and s2_has_pdf:
        print("[success] Using Semantic Scholar result with PDF")
        s2_result["_lookup_sources"] = list(dict.fromkeys(sources_found))
        return s2_result, s2_result["_lookup_sources"]

    # If Semantic Scholar didn't find anything or no PDF, try other databases
    print("[info] Trying alternative databases...")
    alternative_result = fetch_metadata_from_multiple_sources(info, session)
    alternative_sources: List[str] = []
    if alternative_result:
        existing_sources = alternative_result.get("_lookup_sources")
        if isinstance(existing_sources, list):
            alternative_sources.extend(existing_sources)
        alt_source = alternative_result.get("source")
        if alt_source and alt_source not in alternative_sources:
            alternative_sources.append(alt_source)

    # If alternative databases found result with PDF, use it
    if alternative_result:
        alt_has_pdf = bool(alternative_result.get(
            "openAccessPdf") and alternative_result["openAccessPdf"].get("url"))
        combined_sources = list(dict.fromkeys(
            sources_found + alternative_sources))
        alternative_result["_lookup_sources"] = combined_sources
        if alt_has_pdf:
            print("[success] Using alternative database result with PDF")
            return alternative_result, combined_sources
        # If alternative has result but no PDF, and S2 has result, prefer S2
        elif s2_result:
            print(
                "[success] Using Semantic Scholar result (no PDF found in any source)")
            s2_result["_lookup_sources"] = combined_sources or [
                "semantic_scholar"]
            return s2_result, s2_result["_lookup_sources"]
        else:
            print("[success] Using alternative database result (no PDF)")
            return alternative_result, combined_sources

    # If we have S2 result but no PDF anywhere, return it
    if s2_result:
        print("[success] Using Semantic Scholar result (no alternatives found)")
        sources = list(dict.fromkeys(sources_found)) or ["semantic_scholar"]
        s2_result["_lookup_sources"] = sources
        return s2_result, sources

    return None, []


def download_pdf(url: str, path: Path, session: requests.Session) -> bool:
    tmp_path = path.with_suffix(".tmp")
    backoff = 1.0

    headers = {"User-Agent": PDF_DOWNLOAD_USER_AGENT}

    for attempt in range(1, PDF_DOWNLOAD_MAX_ATTEMPTS + 1):
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass

        try:
            with session.get(
                url,
                stream=True,
                timeout=PDF_DOWNLOAD_TIMEOUT,
                allow_redirects=True,
                headers=headers,
            ) as resp:
                resp.raise_for_status()
                content_type = resp.headers.get("content-type", "").lower()
                if "pdf" not in content_type:
                    if "text/html" in content_type:
                        html = resp.text
                        alt_url = extract_pdf_url(html, resp.url)
                        if alt_url and alt_url != url:
                            return download_pdf(alt_url, path, session)
                    return False
                with open(tmp_path, "wb") as fh:
                    for chunk in resp.iter_content(chunk_size=8192):
                        if chunk:
                            fh.write(chunk)
                tmp_path.replace(path)
                return True
        except requests.RequestException as exc:
            status = getattr(getattr(exc, "response", None),
                             "status_code", None)
            retryable_status = status in PDF_DOWNLOAD_RETRY_STATUS if status is not None else False
            retryable_type = isinstance(
                exc, (requests.Timeout, requests.ConnectionError))
            if attempt < PDF_DOWNLOAD_MAX_ATTEMPTS and (retryable_type or retryable_status):
                time.sleep(backoff)
                backoff *= PDF_DOWNLOAD_BACKOFF_FACTOR
                continue
            break
        except OSError:
            break

    if tmp_path.exists():
        try:
            tmp_path.unlink()
        except OSError:
            pass
    return False


def append_log(path: Path, record: Dict) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False))
        fh.write("\n")


def determine_status(
    paper_key: Optional[str],
    meta: Optional[Dict],
    pdf_saved: bool,
    pdf_available: bool,
    fetch_failed: bool,
    pdf_attempted: bool,
    text_saved: bool,
) -> str:
    if fetch_failed:
        return "fetch_failed"
    if paper_key is None:
        return "not_found"
    if pdf_saved:
        return "pdf_saved"
    if text_saved:
        return "text_saved"
    if pdf_available:
        if pdf_attempted:
            return "pdf_download_failed"
        return "open_access_available"
    return "metadata_only" if meta else "not_found"


def build_state_key(info: CitationInfo, paper_key: Optional[str]) -> str:
    if paper_key:
        return paper_key
    principle_slug = slugify(info.principle, length=40)
    return f"missing_{principle_slug}_{info.index}"


def prepare_pipeline_task(
    info: CitationInfo,
    registry: "PaperRegistry",
    session: requests.Session,
    download_pdfs: bool,
    existing_record: Optional[Dict] = None,
) -> Tuple[Optional[PipelineTask], Dict[str, Any]]:
    lookup_sources: List[str] = []
    metadata: Optional[Dict] = None
    paper_key: Optional[str] = None
    fetch_failed = False

    if existing_record:
        existing_key = existing_record.get("paper_key")
        if existing_key:
            cached_metadata = registry.get_by_key(existing_key)
            if cached_metadata and metadata_matches_citation(cached_metadata, info):
                paper_key = existing_key
                metadata = cached_metadata
                stored_sources = cached_metadata.get("_lookup_sources")
                if isinstance(stored_sources, list) and stored_sources:
                    lookup_sources = list(dict.fromkeys(stored_sources))
                else:
                    source_name = cached_metadata.get("source")
                    if source_name:
                        lookup_sources = [source_name]

    if metadata is None:
        existing = registry.get_by_identifiers(info.doi, info.title_hint)
        if existing:
            paper_key, metadata = existing
            if metadata and not metadata_matches_citation(metadata, info):
                print(
                    f"[warn] Stored metadata for '{paper_key}' failed validation; refetching.")
                metadata = None
                paper_key = None
            elif metadata:
                stored_sources = metadata.get("_lookup_sources")
                if isinstance(stored_sources, list) and stored_sources:
                    lookup_sources = list(dict.fromkeys(stored_sources))
                else:
                    source_name = metadata.get("source")
                    if source_name:
                        lookup_sources = [source_name]

    if metadata is None:
        try:
            metadata, lookup_sources = fetch_metadata(info, session)
            lookup_sources = list(dict.fromkeys(lookup_sources))
        except requests.RequestException as exc:
            fetch_failed = True
            append_log(ERROR_LOG, {
                "principle": info.principle,
                "index": info.index,
                "citation": info.citation_raw,
                "error": str(exc),
            })
            metadata = None

        if metadata:
            paper_key = registry.build_key(
                metadata, info.title_hint or info.citation_raw)
            registry.save_metadata(paper_key, metadata)
        else:
            paper_key = None

    if not metadata:
        state_key = build_state_key(info, paper_key)
        state_payload = {
            "paper_key": paper_key,
            "principle": info.principle,
            "citation_index": info.index,
            "citation": info.citation_raw,
            "status": "fetch_failed" if fetch_failed else "not_found",
            "lookup_sources": lookup_sources,
        }
        append_log(NOT_FOUND_LOG, {
            "principle": info.principle,
            "citation": info.citation_raw,
            "index": info.index,
            "doi": info.doi,
            "title_hint": info.title_hint,
        })
        return None, {**state_payload, "state_key": state_key}

    pdf_info = metadata.get("openAccessPdf") or {}
    pdf_url = pdf_info.get("url")
    pdf_available = bool(pdf_url)
    pdf_saved = bool(paper_key and registry.pdf_exists(paper_key))
    text_saved = bool(paper_key and registry.text_exists(paper_key))
    pdf_attempted = False  # Stage 1 does not perform download

    status = determine_status(
        paper_key,
        metadata,
        pdf_saved,
        pdf_available,
        fetch_failed,
        pdf_attempted,
        text_saved,
    )

    needs_pdf_download = bool(pdf_available and not pdf_saved)
    has_abstract = bool(metadata.get("abstract")
                        or metadata.get("abstractText"))
    needs_deep_search = bool(
        not text_saved and not has_abstract and (
            not pdf_available or not download_pdfs)
    )

    task = PipelineTask(
        info=info,
        paper_key=paper_key,
        metadata=metadata,
        pdf_url=pdf_url if pdf_available else None,
        lookup_sources=lookup_sources,
        needs_pdf_download=needs_pdf_download,
        needs_deep_search=needs_deep_search,
    )

    state_payload = {
        "paper_key": paper_key,
        "principle": info.principle,
        "citation_index": info.index,
        "citation": info.citation_raw,
        "status": status,
        "pdf_available": pdf_available,
        "pdf_saved": pdf_saved,
        "text_saved": text_saved,
        "lookup_sources": lookup_sources,
        "needs_pdf_download": needs_pdf_download,
        "needs_deep_search": needs_deep_search,
        "pdf_url": pdf_url,
    }
    state_payload["state_key"] = build_state_key(info, paper_key)
    return task, state_payload


async def stage1_producer(
    tasks: List[CitationInfo],
    registry: "PaperRegistry",
    session: requests.Session,
    pdf_queue: "asyncio.Queue[PipelineTask]",
    deep_queue: "asyncio.Queue[PipelineTask]",
    state: PipelineState,
    existing_records: Dict[Tuple[str, int, str], Dict],
    download_pdfs: bool,
    pdf_worker_count: int,
    deep_worker_count: int,
    progress: Dict[str, Any],
    progress_lock: asyncio.Lock,
) -> None:
    total = len(tasks)
    for idx, info in enumerate(tasks, start=1):
        print(
            f"[{idx}/{total}] Processing principle='{info.principle}' citation #{info.index}"
        )
        existing_record = existing_records.get(
            (info.principle, info.index, info.citation_raw)
        )
        task, payload = await asyncio.to_thread(
            prepare_pipeline_task, info, registry, session, download_pdfs, existing_record
        )
        state_key = payload.pop("state_key")
        await state.update(state_key, payload)

        async with progress_lock:
            progress["stage1_processed"] = idx

        if task:
            if task.needs_pdf_download and pdf_worker_count > 0:
                await pdf_queue.put(task)
            if task.needs_deep_search and deep_worker_count > 0:
                await deep_queue.put(task)

        await asyncio.sleep(0.2)

    async with progress_lock:
        progress["stage1_processed"] = total
        progress["stage1_done"] = True


async def progress_monitor(
    state: PipelineState,
    pdf_queue: "asyncio.Queue[PipelineTask]",
    deep_queue: "asyncio.Queue[PipelineTask]",
    progress: Dict[str, Any],
    progress_lock: asyncio.Lock,
    stop_event: asyncio.Event,
) -> None:
    while True:
        async with progress_lock:
            processed = progress.get("stage1_processed", 0)
            total = progress.get("stage1_total", 0)
        snapshot = state.snapshot()
        total_entries = len(snapshot)
        status_counts = Counter(
            record.get("status", "unknown") for record in snapshot.values()
        )
        needs_deep = sum(
            1 for record in snapshot.values() if record.get("needs_deep_search")
        )
        needs_pdf = sum(
            1 for record in snapshot.values() if record.get("needs_pdf_download")
        )
        total_denominator = total if total else 1
        percent = (processed / total_denominator) * 100 if total else 0.0
        print(
            "\n\n[progress] Stage1 "
            f"{processed}/{total_denominator} ({percent:.1f}%) "
            f"| pdf_queue={pdf_queue.qsize()} | deep_queue={deep_queue.qsize()} "
            f"| recorded={total_entries} | status={dict(status_counts)} "
            f"| pending_pdf={needs_pdf} | pending_deep={needs_deep}\n\n"
        )

        if stop_event.is_set():
            break
        try:
            await asyncio.wait_for(
                stop_event.wait(), timeout=PROGRESS_INTERVAL_SECONDS
            )
        except asyncio.TimeoutError:
            continue
        if stop_event.is_set():
            break


async def pdf_download_worker(
    name: str,
    queue: "asyncio.Queue[PipelineTask]",
    registry: "PaperRegistry",
    state: PipelineState,
    deep_queue: Optional["asyncio.Queue[PipelineTask]"] = None,
    deep_worker_count: int = 0,
) -> None:
    session = requests.Session()
    try:
        while True:
            task = await queue.get()
            if task is PIPELINE_SENTINEL:
                queue.task_done()
                break
            if not isinstance(task, PipelineTask):
                queue.task_done()
                continue
            if not task.paper_key or not task.pdf_url:
                queue.task_done()
                continue
            pdf_path = registry.pdf_path(task.paper_key)
            if registry.pdf_exists(task.paper_key):
                await state.update(task.paper_key, {
                    "status": "pdf_saved",
                    "pdf_saved": True,
                    "needs_pdf_download": False,
                })
                queue.task_done()
                continue

            success = await asyncio.to_thread(
                download_pdf, task.pdf_url, pdf_path, session
            )
            if success:
                await state.update(task.paper_key, {
                    "status": "pdf_saved",
                    "pdf_saved": True,
                    "needs_pdf_download": False,
                    "needs_deep_search": False,
                })
                print(f"[worker:{name}] PDF downloaded for {task.paper_key}")
            else:
                needs_deep = False
                if (
                    deep_queue is not None
                    and deep_worker_count > 0
                    and task.metadata
                    and not registry.text_exists(task.paper_key)
                ):
                    has_abstract = bool(
                        task.metadata.get("abstract")
                        or task.metadata.get("abstractText")
                    )
                    if not has_abstract:
                        task.needs_deep_search = True
                        await deep_queue.put(task)
                        needs_deep = True

                await state.update(task.paper_key, {
                    "status": "pdf_download_failed",
                    "pdf_saved": False,
                    "needs_deep_search": needs_deep,
                    "needs_pdf_download": False,
                })
                append_log(ERROR_LOG, {
                    "principle": task.info.principle,
                    "paper_key": task.paper_key,
                    "citation": task.info.citation_raw,
                    "error": "Failed to download PDF",
                    "pdf_url": task.pdf_url,
                    "worker": name,
                })
            queue.task_done()
    finally:
        session.close()


async def deep_search_worker(
    name: str,
    queue: "asyncio.Queue[PipelineTask]",
    registry: "PaperRegistry",
    state: PipelineState,
    credential: Optional[Tuple[str, str]] = None,
) -> None:
    if credential:
        base_url, api_key = credential
    else:
        base_url = None
        api_key = None
    while True:
        task = await queue.get()
        if task is PIPELINE_SENTINEL:
            queue.task_done()
            break
        if not isinstance(task, PipelineTask):
            queue.task_done()
            continue
        if not task.paper_key or not task.metadata:
            queue.task_done()
            continue
        if registry.text_exists(task.paper_key):
            await state.update(task.paper_key, {
                "status": "text_saved",
                "text_saved": True,
                "needs_deep_search": False,
            })
            queue.task_done()
            continue

        content = await asyncio.to_thread(
            deep_search_paper, task.metadata, task.info, api_key, base_url
        )
        if content:
            registry.save_text(task.paper_key, content)
            await state.update(task.paper_key, {
                "status": "text_saved",
                "text_saved": True,
                "deep_search_attempted": True,
                "needs_deep_search": False,
            })
            print(
                f"[worker:{name}] Deep search saved text for {task.paper_key}")
        else:
            await state.update(task.paper_key, {
                "status": "deep_search_failed",
                "deep_search_attempted": True,
                "needs_deep_search": True,
            })
            append_log(ERROR_LOG, {
                "principle": task.info.principle,
                "paper_key": task.paper_key,
                "citation": task.info.citation_raw,
                "error": "Deep search failed",
                "worker": name,
            })
        queue.task_done()


def update_principles_index(entries: Dict[str, List[Dict[str, Any]]]) -> None:
    index: Dict[str, List[Dict[str, Any]]] = {}
    if PRINCIPLES_INDEX_PATH.exists():
        try:
            index = json.loads(
                PRINCIPLES_INDEX_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            index = {}
    index.update(entries)
    PRINCIPLES_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    PRINCIPLES_INDEX_PATH.write_text(json.dumps(
        index, ensure_ascii=False, indent=2), encoding="utf-8")


async def run_pipeline_async(
    principles: Optional[List[str]],
    limit: Optional[int],
    download_pdfs: bool,
    pdf_workers: int,
    deep_workers: int,
) -> Dict[str, Dict[str, Any]]:
    base_url = get_s2_base_url()
    print(f"[info] Using Semantic Scholar base URL: {base_url}")
    tasks = load_responses(principles, limit)
    if not tasks:
        print("No citations found for the specified principles.")
        return {}

    registry = PaperRegistry()
    state = PipelineState(PIPELINE_STATE_PATH)
    pdf_queue: "asyncio.Queue[PipelineTask]" = asyncio.Queue()
    deep_queue: "asyncio.Queue[PipelineTask]" = asyncio.Queue()

    progress: Dict[str, Any] = {
        "stage1_total": len(tasks),
        "stage1_processed": 0,
        "stage1_done": False,
    }
    progress_lock = asyncio.Lock()
    stop_event = asyncio.Event()

    credentials = collect_deep_search_credentials()
    if deep_workers > 0 and not credentials:
        print(
            "[warn] Deep search workers requested but no API_KEY_LIMIT* credentials found; disabling deep search workers."
        )
    active_deep_workers = min(deep_workers, len(
        credentials)) if credentials else 0
    if active_deep_workers > 0:
        print(
            f"[info] Deep search workers active: {active_deep_workers} (available credentials: {len(credentials)})"
        )

    pdf_worker_tasks = [
        asyncio.create_task(
            pdf_download_worker(
                f"pdf-{idx+1}",
                pdf_queue,
                registry,
                state,
                deep_queue if active_deep_workers > 0 else None,
                active_deep_workers,
            )
        )
        for idx in range(pdf_workers)
    ]
    deep_worker_tasks = [
        asyncio.create_task(
            deep_search_worker(
                f"deep-{idx+1}",
                deep_queue,
                registry,
                state,
                credentials[idx],
            )
        )
        for idx in range(active_deep_workers)
    ]

    existing_records_map: Dict[Tuple[str, int, str], Dict] = {}
    initial_snapshot = state.snapshot()
    for record in initial_snapshot.values():
        principle = record.get("principle")
        citation_index = record.get("citation_index")
        citation = record.get("citation")
        if principle and citation_index is not None and citation:
            existing_records_map[(principle, int(citation_index), citation)] = record

    session = requests.Session()
    monitor_task = asyncio.create_task(
        progress_monitor(
            state,
            pdf_queue,
            deep_queue,
            progress,
            progress_lock,
            stop_event,
        )
    )
    try:
        try:
            await stage1_producer(
                tasks,
                registry,
                session,
                pdf_queue,
                deep_queue,
                state,
                existing_records_map,
                download_pdfs,
                pdf_workers,
                active_deep_workers,
                progress,
                progress_lock,
            )
        finally:
            session.close()

        if pdf_worker_tasks:
            await pdf_queue.join()
            for _ in pdf_worker_tasks:
                await pdf_queue.put(PIPELINE_SENTINEL)
            await asyncio.gather(*pdf_worker_tasks)
        if deep_worker_tasks:
            await deep_queue.join()
            for _ in deep_worker_tasks:
                await deep_queue.put(PIPELINE_SENTINEL)
            await asyncio.gather(*deep_worker_tasks)
    finally:
        stop_event.set()
        try:
            await monitor_task
        except asyncio.CancelledError:
            pass

    return state.snapshot()


def build_principle_entries_from_state(
    state_data: Dict[str, Dict[str, Any]]
) -> Dict[str, List[Dict[str, Any]]]:
    aggregated: Dict[str, List[Dict[str, Any]]] = {}
    for record in state_data.values():
        principle = record.get("principle")
        if not principle:
            continue
        entry = {
            "paper_key": record.get("paper_key"),
            "status": record.get("status"),
            "pdf_saved": record.get("pdf_saved"),
            "needs_pdf_download": record.get("needs_pdf_download"),
            "needs_deep_search": record.get("needs_deep_search"),
            "text_saved": record.get("text_saved"),
            "deep_search_attempted": record.get("deep_search_attempted"),
            "lookup_sources": record.get("lookup_sources"),
        }
        aggregated.setdefault(principle, []).append(entry)
    return aggregated


def process_citations(
    principles: Optional[List[str]],
    limit: Optional[int],
    download_pdfs: bool,
    pdf_workers: int,
    deep_workers: int,
) -> None:
    state_snapshot = asyncio.run(
        run_pipeline_async(principles, limit, download_pdfs,
                           pdf_workers, deep_workers)
    )
    if state_snapshot:
        entries = build_principle_entries_from_state(state_snapshot)
        update_principles_index(entries)
        print(f"[done] Updated principles index at {PRINCIPLES_INDEX_PATH}")
    else:
        print("[done] No updates to principles index")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert responses_parsed.json into Semantic Scholar metadata and PDF archive."
    )
    parser.add_argument(
        "--principles",
        nargs="*",
        help="Optional list of principle names to process. Defaults to all in responses_parsed.json.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional limit on number of citations per principle.",
    )
    parser.add_argument(
        "--skip-pdf",
        action="store_true",
        help="Skip downloading PDFs even if open access links are available.",
    )
    parser.add_argument(
        "--pdf-workers",
        type=int,
        default=int(os.getenv("PDF_WORKERS", "3")),
        help="Number of concurrent PDF download workers (default: 3).",
    )
    parser.add_argument(
        "--deep-workers",
        type=int,
        default=int(os.getenv("DEEP_WORKERS", "10")),
        help="Number of concurrent deep search workers (default: 10).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    download_pdfs = not args.skip_pdf
    pdf_workers = max(0, args.pdf_workers if download_pdfs else 0)
    deep_workers = max(0, args.deep_workers)
    process_citations(args.principles, args.limit,
                      download_pdfs, pdf_workers, deep_workers)
