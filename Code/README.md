整个流程可以拆成 5 个阶段，依次运行下列脚本即可把文献信息转化为最终的场景与对话数据。

- 准备环境：确认 BASE_URL_LIMIT/API_KEY_LIMIT(0~9)、BASE_URL_FULL/API_KEY_FULL、S2_API_KEY 等 API 凭据已写入终端环境（若走代理，也要在 shell 里提前 export http_proxy=…）。

- 文献检索：运行 python google_search.py --output-dir Dataset/gemini_references --sequential（或并行模式），为每个原则生成 50 条引用，产出 responses_raw.json、responses_parsed.json 等文件。

- 语料构建：执行 python build*semantic_scholar_archive.py --skip-pdf（或根据需要下载 PDF），把 responses_parsed.json 转成 Dataset/papers_info 目录下的 principles.json、papers/*.json、texts/\_.txt 等语料。若要分批处理，可先用 split_responses.py，若需要清理错误项，可用 identify_metadata_mismatches.py/clean_mismatched_entries.py。

- 模式总结：基于前一步的文本语料，运行 python summarize_psy_patterns.py （可加 --principles / --overwrite 控制范围），生成 Dataset/patterns_info/psy_patterns_markdown.json 和结构化的 psy_patterns_info.json；如需补充旧数据或贴上引用，可再运行 backfill_patterns_info.py 来补充旧数据、add_sources_to_patterns.py 来生成包含 source 的 psy_patterns_info_with_source.json 文件。

- 场景生成：最后执行 python gen_scenario_conversation.py（默认读取 Dataset/patterns_info/psy_patterns_info.json），它会遍历原则 × 情境组合，分别调用大模型生成场景与对话并写入 Dataset/generated_data.json。

## 多主机并行构建语料

### 整体思路

- 先把 Dataset/gemini_references/responses_parsed.json 按原则拆分成多个“任务包”，每台机器各拿一个；
- 各台机器独立运行 build_semantic_scholar_archive.py（可选择下载 PDF）；
- 最后把所有机器产生的输出目录合并回主机，再统一执行后续步骤（汇总 principles.json、合并 papers/ 与 texts/、整合日志等）。

### 具体步骤

1. 拆分任务

在主机上运行
python split_responses.py --responses Dataset/gemini_references/responses_parsed.json --output-dir Dataset/task_splits --splits N --archive
其中 N 为机器数。该脚本会生成 Dataset/task_splits/split_i 目录和 .tar.gz 压缩包，每个目录内包含：
responses_parsed.json：该分片需处理的引用列表；
tasks.jsonl、pipeline_state.json：供 Stage1/Stage2 使用的状态文件；
principles.txt：本分片包含的原则清单。

2. 分发任务包

将各自的 split_i.tar.gz 分发到不同机器；在目标机器解压，确保在该目录中运行后续命令。

3. 每台机器运行构建脚本

- 切换到分片目录，执行：
  python build_semantic_scholar_archive.py --responses responses_parsed.json --pipeline-state pipeline_state.json --principles-file principles.txt --output-base Dataset/papers_info_split_i ...
- 可根据需要添加 --skip-pdf、--pdf-workers、--deep-workers 等参数；
- 生成的成果位于 Dataset/papers_info_split_i/，结构和主机上最终的 Dataset/papers_info 相同（包含 papers/、pdfs/、texts/、logs/、principles.json 等）。

4. 收集结果

- 每台机器把 Dataset/papers_info_split_i/ 目录打包带回主机；
- 在主机上，将各子目录的内容按文件夹名合并/去重：
  papers/、pdfs/、texts/ 直接追加并确保 paper_key 唯一；
  principles.json 可读取后合并为一个大字典；
  logs/（如 not_found.jsonl、errors.jsonl）按行追加；
  pipeline_state.json 可转为字典后合并（注意去重）。

5. 后续流程
   合并完成的目录即成为新的 Dataset/papers_info，随后继续执行 summarize_psy_patterns.py 等步骤。
   若要检查遗漏或错误，可运行 identify_metadata_mismatches.py、clean_mismatched_entries.py 再次校验。
   借助 split_responses.py 和 build_semantic_scholar_archive.py 的参数化设计，可以把工作均匀分发到多台机器，显著加快语料构建阶段的整体速度。

principle_situation.py 文件里有列表

## google_search.py

--limit <N>
限制只搜索前 N 条 principle，用于测试

--sequential
限制是否异步 request

--output-dir
输出文件夹路径，默认为 Dataset/gemini_references

--principles
后面跟要处理的原则名（可多条），名称之间用空格分隔即可。用于测试

## build_semantic_scholar_archive.py

实现从 responses_parsed.json 到结构化索引与 PDF 的完整流水线：

- 逐条解析引用，提取 DOI/标题提示，构建待查询任务。
- 访问 Semantic Scholar Graph API（自动重试、速率限制间隔 0.2s），复用已有元数据并按 paperId/DOI/标题生成去重 paper_key。
- 将新元数据写入
  Dataset/semantic_scholar/papers/<paper_key>.json，可选下载开放获取 PDF 到 Dataset/semantic_scholar/pdfs/<paper_key>.pdf，失败会记录日志。
- 生成/更新
  Dataset/semantic_scholar/principles.json，每个原则对应 50 个 {paper_key, status} 索引。
- 将未命中、错误信息追加到 Dataset/semantic_scholar/logs/not_found.jsonl 与 errors.jsonl。

运行前请设定 S2_API_KEY，可用 --principles、--limit、--skip-pdf 控制范围与是否抓取 PDF。

--principles <名称 1> <名称 2> ...
只处理指定的原则，名称要与 responses_parsed.json 里的键完全一致。
例：python build_semantic_scholar_archive.py --principles "actor observer asymmetry" "defensive attribution hypothesis"

--limit <N>
限制每个原则最多处理前 N 条引用（用于小样本调试）。
例：python build_semantic_scholar_archive.py --limit 10

--skip-pdf
添加该开关后，即使抓到了开放获取链接也不会下载 PDF，只写元数据。
例：python build_semantic_scholar_archive.py --skip-pdf

--pdf-workers
启动 N 个异步下载 worker。队列里只要有待下载的 PDF，就会被最多 N 个 worker 并行处理，所以下载任务最多同时进行 N 条。

--deep-workers
Number of concurrent deep search workers (default: 10)
启动 N 个深度检索 worker。每个 worker 独立消费 deep search 队列；因此当 --deep-workers=10 时，最多会有 10 个 deep search 请求在同一时间执行。

## gen_scenario_conversation.py/gen_scenario_conversation_multi_patterns.py

生成单/多 pattern 的 scenario 和 conversation

## generate_pattern_distribution_and_checklist.py

python generate_pattern_distribution_and_checklist.py --mode 1 --input-file Dataset/generated_data.json --output-file Dataset/generated_data_with_p_c.json

python generate_pattern_distribution_and_checklist.py --mode 1 --input-file Dataset/generated_data_multi_patterns.json --output-file Dataset/generated_data_multi_patterns_with_p_c.json

可以使用--drop-failures来丢弃失败样本
检查没有问题就拷贝到原文件 generated_data.json 和 generated_data_multi_patterns.json 中

为生成的生成单/多 pattern 的 scenario 和 conversation 添加"char2pattern"和"conversation_checklist"字段（旧字段 pattern_distribution 已统一替换为 char2pattern）

## factor_new_scenario.py

python factor_new_scenario.py

生成 factor 以及新的 scenario 和 analysis

## generate_pattern_distribution_and_checklist.py

python generate_pattern_distribution_and_checklist.py --mode 2

## split_dataset.py
把Dataset/final_data.json的数据分为Train, OOD Eval, ID Eval, Mixed Eval
python split_dataset.py

## gen_sharegpt_dataset.py
