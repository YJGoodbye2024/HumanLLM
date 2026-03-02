import argparse
import asyncio
import json
import os
import re
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests
from requests import exceptions as req_exc

from principle_situaton import sd_pri_list, td_pri_list_100

TD_PRINCIPLE_LOOKUP = {name.lower() for name in td_pri_list_100}

MAX_CALL = 4  # 每分钟最大请求次数
NUMBERED_PREFIX_RE = re.compile(r'^\d+[\).\s]+')
BULLET_PREFIX_RE = re.compile(r'^[\-\*\u2022]\s+')
MARKDOWN_MARKERS_RE = re.compile(r'[\*\u2013\u2014]')
TRAILING_NOTE_RE = re.compile(r'\s*\([^)]*[\u4e00-\u9fff][^)]*\)\s*$')
HEADING_SPLIT_RE = re.compile(r'\s#{2,}.*')

SECTION_DEFINITIONS: List[Tuple[str, re.Pattern[str]]] = [
    (
        "Foundational Definition & Description",
        re.compile(r"foundational definition|基础定义|定义与描述", re.IGNORECASE),
    ),
    (
        "Core Mechanisms & Theoretical Explanations",
        re.compile(r"core mechanisms?|核心机制|理论解释", re.IGNORECASE),
    ),
    (
        "Real-World Impact & Applications",
        re.compile(r"real[-\s]*world impact|现实世界的影响|现实世界的应用", re.IGNORECASE),
    ),
]

SD_PROMPT_TEMPLATE = """# **角色:** 您是一位顶尖的学术研究员和心理学文献专家，擅长为特定研究课题系统性地搜集和组织最关键的学术资源。

# **任务目标:** 您的核心任务是针对心理学原则 **`{PRINCIPLE_NAME}`**，进行一次深度、广泛且目标明确的文献检索。您需要找出 **50篇** 最具相关性和影响力的学术文献（包括开创性论文、重要的综述、以及关键的实证研究）。文献的选择必须紧密围绕以下三个核心主题。

# ---

# ### **文献检索的三个核心主题：**

# 您需要确保最终的文献列表能够全面覆盖以下三个分析维度。请将这些维度作为您搜索和筛选文献的指导框架：

# **1. 基础定义与描述 (Foundational Definition & Description)**
# * 寻找那些为 **`{PRINCIPLE_NAME}`** 提供了最权威、最清晰定义的基础性文献。
# * 这些文献应该专注于阐述该原则的核心现象、构成要素以及其作为一种基本心理过程的运作方式。

# **2. 核心机制与理论解释 (Core Mechanisms & Theoretical Explanations)**
# * 寻找深入探讨该原则存在原因的文献，包括其背后的演化、认知或情感驱动力。
# * 重点关注那些解释该现象为何会发生的理论或模型研究，例如，它是一种**认知启发式（mental shortcut）**、**自尊保护机制**、**记忆或感知的局限性**，还是一种**适应性的演化特征**等。

# **3. 现实世界的影响与应用 (Real-World Impact & Application)**
# * 寻找研究该原则在现实世界中具体表现、影响和实际应用的文献。
# * 这个类别的文献应包含两个方面：
#     * **双刃剑效应：** 既有探讨其积极/适应性功能（如简化决策、维持动机）的文献，也要有揭示其消极/不适应后果（如阻碍个人成长、加剧社会偏见）的研究。
#     * **实际应用：** 寻找将该原则应用于管理、市场营销、临床治疗、冲突解决或个人发展等领域的实证研究或案例分析。

# ---

# ### **输出要求：**

# 1.  **文献列表:**
#     * 请提供一个包含 **50篇** 相关文献的最终列表。
#     * 所有文献请使用 **APA引文格式**。

# 2.  **组织结构:**
#     * 请将这50篇文献明确地归入上述三个核心主题类别中（**基础定义与描述**、**核心机制与理论解释**、**现实世界的影响与应用**）。
#     * 如果某一篇文献可以归入多个类别，请根据其最主要的贡献将其放入最合适的类别。力求每个类别下的文献数量分布相对均衡，以确保研究视角的全面性。"""


TD_PROMPT_TEMPLATE = """角色: 您是一位顶尖的学术研究员和心理学文献专家，擅长为特定研究课题系统性地搜集和组织最关键的学术资源。
任务目标: 您的核心任务是针对心理学原则 {PRINCIPLE_NAME}，进行一次深度、广泛且目标明确的文献检索。您需要找出 50篇 最具相关性和影响力的学术文献（包括开创性论文、重要的综述、以及关键的实证研究）。文献的选择必须紧密围绕以下三个核心主题。
---
### 文献检索的三个核心主题：
您需要确保最终的文献列表能够全面覆盖以下三个分析维度。请将这些维度作为您搜索和筛选文献的指导框架：

1. 基础定义与描述 (Foundational Definition & Description)
寻找为 {PRINCIPLE_NAME} 提供精确、专业的定义，并引用主流心理学理论的文献。
重点关注阐释该特质在个体人格结构中所扮演角色的研究。

2. 核心机制与理论解释 (Core Mechanisms & Theoretical Explanations)
认知模式 (Cognitive Patterns): 寻找描述具有此特质个体典型思维模式、信念系统和注意焦点的文献。他们如何看待世界、他人和自己？
情感特征 (Emotional Signatures): 寻找描述他们倾向于体验和表达的核心情绪、情绪稳定性以及典型共情反应的文献。
行为倾向 (Behavioral Tendencies): 寻找描述在日常、非压力情境下，该特质个体所表现出的自发性、可观察行为的文献。

3. 现实世界的影响与应用 (Real-World Impact & Application)
寻找探讨该特质在个体实际生活情境中如何表现和应用的文献。这可以包含但不限于多个角度，**比如**：
* **压力下的表现**：个体面临挑战、失败或高压时，该特质如何表现（是被放大、减弱还是扭曲？）。
* **冲突中的应对**：个体在处理人际冲突时的典型策略。
* **积极情境中的表现**：个体取得成功、获得支持或感到快乐时，该特质如何表达。
* **以及其他**在人际关系、工作绩效或决策制定中的具体影响**等等**。

---
输出要求：

文献列表:
请提供一个包含 50篇 相关文献的最终列表。
所有文献请使用 APA引文格式。

组织结构:
请将这50篇文献明确地归入上述三个核心主题类别中（基础定义与描述、核心机制与理论解释、现实世界的影响与应用）。
如果某一篇文献可以归入多个类别，请根据其最主要的贡献将其放入最合适的类别。力求每个类别下的文献数量分布相对均衡，以确保研究视角的全面性。"""


class RateLimiter:
    """异步速率限制器，确保在任意连续的时间窗口内请求次数不会超过设定值。"""

    def __init__(self, max_calls: int, period: float) -> None:
        """
        初始化速率限制器。

        参数:
            max_calls: 在一个时间窗口内允许的最大请求次数。
            period: 时间窗口的长度（秒）。
        """
        # 记录限制配置，max_calls 表示每段时间允许的最大调用次数
        self.max_calls = max_calls
        self.period = period
        # 使用队列保存最近一次段时间内的调用时间戳
        self._timestamps: deque[float] = deque()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """
        在发起请求前调用，若达到限制则等待至允许的新时间点。
        """
        while True:
            async with self._lock:
                now = time.monotonic()
                # 清理超出周期的旧时间戳
                while self._timestamps and now - self._timestamps[0] >= self.period:
                    self._timestamps.popleft()

                if len(self._timestamps) < self.max_calls:
                    # 剩余配额时立即放行
                    self._timestamps.append(now)
                    return

                wait_time = self.period - (now - self._timestamps[0])

            await asyncio.sleep(wait_time)


def _build_messages(principle_name: str) -> List[Dict[str, str]]:
    """
    生成聊天补全接口所需的消息列表。

    参数:
        principle_name: 目标心理学原则名称。

    返回:
        所有原则对应的prompt组成的列表。
    """
    normalized_name = principle_name.strip().lower()
    prompt_template = (
        TD_PROMPT_TEMPLATE
        if normalized_name in TD_PRINCIPLE_LOOKUP
        else SD_PROMPT_TEMPLATE
    )
    developer_prompt = prompt_template.format(PRINCIPLE_NAME=principle_name)
    user_prompt = f"请基于上述指令，为心理学原则「{principle_name}」提供结果。"
    return [
        {"role": "system", "content": developer_prompt},
        {"role": "user", "content": user_prompt},
    ]


def fetch_references(principle_name: str, retries: int = 3, backoff_seconds: float = 5.0) -> str:
    """
    调用聊天补全接口，获取指定心理学原则的文献推荐。

    参数:
        principle_name: 需要查询的心理学原则名称。
        retries: 出现可恢复错误时的最大重试次数。
        backoff_seconds: 每次重试前的等待时间（秒），会按尝试次数线性递增。

    返回:
        模型返回的正文内容字符串。
    """
    base_url = os.getenv("BASE_URL_LIMIT")
    api_key = os.getenv("API_KEY_LIMIT")

    if not base_url:
        raise RuntimeError("BASE_URL_LIMIT environment variable is not set.")
    if not api_key:
        raise RuntimeError("API_KEY_LIMIT environment variable is not set.")

    # 构造请求体与 HTTP 头
    url = base_url.rstrip("/") + "/chat/completions"
    payload: Dict[str, Any] = {
        "model": "gemini-2.5-pro-preview-06-05-search",
        "messages": _build_messages(principle_name),
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    last_error: Optional[Exception] = None

    for attempt in range(1, max(retries, 1) + 1):
        try:
            response = requests.post(
                url, headers=headers, json=payload, timeout=120)
            response.raise_for_status()
        except req_exc.HTTPError as exc:
            status_code = exc.response.status_code if exc.response is not None else None
            last_error = exc
            if status_code and 500 <= status_code < 600 and attempt < retries:
                time.sleep(backoff_seconds * attempt)
                continue
            raise
        except req_exc.RequestException as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(backoff_seconds * attempt)
                continue
            raise

        data = response.json()
        choices = data.get("choices") or []

        if choices:
            try:
                return choices[0]["message"]["content"]
            except (KeyError, IndexError) as exc:
                last_error = ValueError(
                    f"Unexpected response payload structure: {data}")
                last_error.__cause__ = exc
        else:
            # 将无内容的返回视为可恢复错误，准备重试
            error_detail = data.get(
                "error") or "Empty choices returned by API."
            last_error = ValueError(
                f"Unexpected response payload: {error_detail}")

        if attempt < retries:
            wait_time = backoff_seconds * attempt
            time.sleep(wait_time)

    assert last_error is not None  # for type checkers;逻辑上一定有错误
    raise last_error


def _clean_reference_text(text: str) -> str:
    """
    清理引用文本中的 Markdown 标记、中文注释和标题说明。
    """
    if not text:
        return ""
    cleaned = text.replace("**", "")
    cleaned = MARKDOWN_MARKERS_RE.sub("", cleaned).strip()

    heading_match = HEADING_SPLIT_RE.search(cleaned)
    if heading_match:
        cleaned = cleaned[:heading_match.start()].rstrip()

    while True:
        updated = TRAILING_NOTE_RE.sub("", cleaned)
        if updated == cleaned:
            break
        cleaned = updated.strip()

    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


async def _worker(
    principle_name: str,
    index: int,
    total: int,
    limiter: RateLimiter,
    output_dir: Path,
) -> Dict[str, Any]:
    """
    处理单个心理学原则：受速率限制后发起请求并保存响应。

    参数:
        principle_name: 当前要查询的原则。
        index: 当前原则的序号（从 1 开始）。
        total: 总原则数。
        limiter: 速率限制器实例。
        output_dir: 输出文件所在目录。

    返回:
        包含原则名称与响应信息的字典，失败时含错误描述。
    """
    await limiter.acquire()
    try:
        content: str = await asyncio.to_thread(fetch_references, principle_name)
    except Exception as exc:  # noqa: BLE001 - propagate error info
        return {"principle": principle_name, "error": str(exc)}

    if not content.strip():
        return {"principle": principle_name, "error": "Empty response content"}

    print(f"[response] {principle_name}已经成功获取内容")
    # print(f"[response] {principle_name}:\n{content}\n")
    return {"principle": principle_name, "content": content}


def _extract_references(content: str) -> List[str]:
    """
    使用正则表达式将模型输出解析为文献引用列表。

    参数:
        content: 模型生成的原始文本。

    返回:
        提取出的 APA 引文列表，长度上限为 50。
    """
    return _extract_references_from_lines(content.splitlines())


def _extract_references_from_lines(lines: Iterable[str]) -> List[str]:
    """Helper to parse references from a sequence of lines."""
    references: List[str] = []
    current: List[str] = []

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue

        if NUMBERED_PREFIX_RE.match(line):
            if current:
                raw_candidate = " ".join(current).strip()
                candidate = _clean_reference_text(raw_candidate)
                if candidate:
                    references.append(candidate)
                current = []
            line = NUMBERED_PREFIX_RE.sub("", line).strip()
            if line:
                current.append(line)
            continue

        if BULLET_PREFIX_RE.match(line):
            # 忽略包含“摘要”等说明的项目符号行
            continue

        current.append(line)

    if current:
        raw_candidate = " ".join(current).strip()
        candidate = _clean_reference_text(raw_candidate)
        if candidate:
            references.append(candidate)

    cleaned = [ref for ref in references if re.search(r'\(\d{4}\)', ref)]
    return cleaned[:50]


def _split_content_by_sections(content: str) -> Dict[str, List[str]]:
    """
    将模型输出按三个主题部分拆分成行列表。
    """
    section_lines = {name: [] for name, _ in SECTION_DEFINITIONS}
    current_section: Optional[str] = None

    for raw_line in content.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue

        matched_section = None
        for name, pattern in SECTION_DEFINITIONS:
            if pattern.search(stripped):
                matched_section = name
                break

        if matched_section:
            current_section = matched_section
            continue

        if current_section:
            section_lines[current_section].append(raw_line)

    return section_lines


def _extract_references_by_section(content: str) -> Dict[str, List[str]]:
    """
    提取按照三个主题分类的引用列表。
    """
    section_lines = _split_content_by_sections(content)
    categorized: Dict[str, List[str]] = {}
    for name, _ in SECTION_DEFINITIONS:
        categorized[name] = _extract_references_from_lines(
            section_lines.get(name, []))
    return categorized


def _write_outputs(aggregated: Dict[str, str], errors: Dict[str, str], output_dir: Path) -> None:
    """
    将成功与失败结果写入磁盘，并打印简要汇总信息。

    参数:
        aggregated: 成功结果映射，键为原则名称，值为模型原始输出。
        errors: 失败的原则及其错误信息。
        output_dir: 输出目录。
    """
    raw_path = output_dir / "responses_raw.json"
    raw_content = json.dumps(aggregated, ensure_ascii=False, indent=2)
    raw_path.write_text(raw_content, encoding="utf-8")
    print(f"[done] Raw responses saved to {raw_path}")

    parsed: Dict[str, List[str]] = {}
    parsed_by_section: Dict[str, Dict[str, List[str]]] = {}
    for principle, text in aggregated.items():
        parsed[principle] = _extract_references(text)
        parsed_by_section[principle] = _extract_references_by_section(text)

    parsed_path = output_dir / "responses_parsed.json"
    parsed_content = json.dumps(parsed, ensure_ascii=False, indent=2)
    parsed_path.write_text(parsed_content, encoding="utf-8")
    print(f"[done] Parsed references saved to {parsed_path}")

    parsed_sections_path = output_dir / "responses_parsed_by_section.json"
    parsed_sections_content = json.dumps(
        parsed_by_section, ensure_ascii=False, indent=2)
    parsed_sections_path.write_text(parsed_sections_content, encoding="utf-8")
    print(
        f"[done] Sectioned references saved to {parsed_sections_path}")

    if errors:
        errors_path = output_dir / "errors.json"
        errors_path.write_text(json.dumps(
            errors, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            f"[warn] {len(errors)} principles failed. Details in {errors_path}")


async def process_principles(principles: Iterable[str], output_dir: Path) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    批量处理心理学原则并生成汇总文件。

    参数:
        principles: 需要查询的原则列表。
        output_dir: 保存响应文本与汇总索引的目录。

    返回:
        二元组 (aggregated, errors)。aggregated 为成功原则的 {名称: 模型输出}，
        errors 为失败原则的 {名称: 错误信息}。
    """
    principles = list(principles)
    if not principles:
        print("No principles provided for processing.")
        return {}, {}

    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[info] Processing {len(principles)} principles. Output directory: {output_dir}")

    limiter = RateLimiter(max_calls=MAX_CALL, period=60.0)
    # 为每个原则启动一个协程任务，利用速率限制控制实际发送频率
    total = len(principles)
    tasks = [
        asyncio.create_task(
            _worker(principle, idx, total, limiter, output_dir))
        for idx, principle in enumerate(principles, start=1)
    ]

    aggregated: Dict[str, str] = {}
    errors: Dict[str, str] = {}
    stored_count = 0
    stored_count = 0

    for task in asyncio.as_completed(tasks):
        result = await task
        principle = result["principle"]

        if "error" in result:
            errors[principle] = result["error"]
            print(f"[error] {principle}: {result['error']}")
            continue

        aggregated[principle] = result["content"]
        stored_count += 1
        print(f"[saved] {principle}")
        print(f"已存储 {stored_count}/{total} 条结果\n")

    return aggregated, errors


def process_principles_sync(principles: Iterable[str], output_dir: Path) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    同步处理心理学原则，逐个调用 API。

    参数:
        principles: 需要查询的原则列表。
        output_dir: 保存响应文本与汇总索引的目录。

    返回:
        二元组 (aggregated, errors)。aggregated 为成功原则的 {名称: 模型输出}，
        errors 为失败原则的 {名称: 错误信息}。
    """
    principles = list(principles)
    if not principles:
        print("No principles provided for processing.")
        return {}, {}

    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[info] Sequential mode: processing {len(principles)} principles. Output directory: {output_dir}")

    aggregated: Dict[str, str] = {}
    errors: Dict[str, str] = {}
    stored_count = 0

    total = len(principles)
    for idx, principle in enumerate(principles, start=1):
        print(f"[info] ({idx}/{total}) Requesting {principle}...")
        try:
            content = fetch_references(principle)
        except Exception as exc:  # noqa: BLE001
            errors[principle] = str(exc)
            print(f"[error] {principle}: {exc}")
            continue

        if not content.strip():
            errors[principle] = "Empty response content"
            print(f"[error] {principle}: Empty response content")
            continue

        print(f"[response] {principle}:\n{content}\n")
        aggregated[principle] = content
        stored_count += 1
        print(f"[saved] {principle}")
        print(f"已存储 {stored_count}/{total} 条结果")

    return aggregated, errors


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数，支持自定义原则集合、数量限制与输出目录。

    返回:
        argparse.Namespace，包含用户输入的所有参数。
    """
    parser = argparse.ArgumentParser(
        description=(
            "Query the Gemini chat completions API for every psychological principle "
            "listed in sd_pri_list and td_pri_list_100 with a rate limit of 5 calls per minute."
        )
    )
    parser.add_argument(
        "--principles",
        nargs="*",
        help="Optional subset of principles to query. Defaults to all items in both principle lists.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit the number of principles to process (after applying --principles).",
    )
    parser.add_argument(
        "--output-dir",
        default="Dataset/gemini_references",
        help="Directory to store individual responses and the aggregated index.",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Process requests sequentially (debug mode) instead of using async workers.",
    )
    return parser.parse_args()


def main() -> None:
    """
    入口函数：读取参数、准备原则列表并触发处理流程。
    """
    args = parse_args()

    if args.principles:
        principles: List[str] = list(args.principles)
    else:
        combined = list(sd_pri_list)
        for item in td_pri_list_100:
            if item not in combined:
                combined.append(item)
        principles = combined

    if args.limit is not None:
        # limit 仅截取前 N 个元素，便于测试或部分运行
        principles = principles[: max(args.limit, 0)]

    output_dir = Path(args.output_dir)

    if args.sequential:
        aggregated_round1, errors_round1 = process_principles_sync(
            principles, output_dir)
    else:
        aggregated_round1, errors_round1 = asyncio.run(
            process_principles(principles, output_dir))

    aggregated_total: Dict[str, str] = dict(aggregated_round1)
    final_errors: Dict[str, str] = errors_round1.copy()

    retry_principles = list(errors_round1.keys())
    failure_path_round3 = output_dir / "failed_principles_round3.txt"

    if retry_principles:
        print(
            f"[retry] Starting second round for {len(retry_principles)} principles.")
        if args.sequential:
            aggregated_round2, errors_round2 = process_principles_sync(
                retry_principles, output_dir)
        else:
            aggregated_round2, errors_round2 = asyncio.run(
                process_principles(retry_principles, output_dir))

        aggregated_total.update(aggregated_round2)
        final_errors = errors_round2.copy()

        retry_principles_round3 = list(final_errors.keys())
        if retry_principles_round3:
            print(
                f"[retry] Starting third round (sequential) for {len(retry_principles_round3)} principles.")
            aggregated_round3, errors_round3 = process_principles_sync(
                retry_principles_round3, output_dir)
            aggregated_total.update(aggregated_round3)
            final_errors = errors_round3.copy()

            if final_errors:
                failure_path_round3.write_text(
                    "\n".join(final_errors.keys()),
                    encoding="utf-8"
                )
                print(
                    f"[warn] {len(final_errors)} principles still failed after third round. See {failure_path_round3}")
            else:
                if failure_path_round3.exists():
                    failure_path_round3.unlink()
                print("[info] All retry principles succeeded in the third round.")
        else:
            if failure_path_round3.exists():
                failure_path_round3.unlink()
            print("[info] All retry principles succeeded in the second round.")
    else:
        if failure_path_round3.exists():
            failure_path_round3.unlink()
        print("[info] No retry needed; all principles succeeded on the first pass.")

    _write_outputs(aggregated_total, final_errors, output_dir)


if __name__ == "__main__":
    main()
