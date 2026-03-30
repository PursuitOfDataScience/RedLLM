#!/usr/bin/env python3
"""
Upload 红楼梦续写 dataset to HuggingFace Hub.

Usage:
    pip install huggingface_hub
    huggingface-cli login
    python hlm_upload.py <jsonl_path>

Example:
    python hlm_upload.py ./data/hlm_continuations.jsonl
"""
import sys, os, json, tempfile

if len(sys.argv) < 2:
    print("Usage: python hlm_upload.py <jsonl_path>")
    sys.exit(1)

jsonl_path = sys.argv[1]
if not os.path.exists(jsonl_path):
    print(f"❌ {jsonl_path} not found")
    sys.exit(1)

USERNAME = "PursuitOfDataScience"
DATASET_NAME = "dream-of-the-red-chamber-continuations"
REPO_ID = f"{USERNAME}/{DATASET_NAME}"

# ========== Compute stats by streaming JSONL (low memory) ==========
print("📊 Computing dataset statistics...")
num_versions = 0
total_chars_all = 0
total_chapters_all = 0
all_chapter_lens = []
total_input_tokens = 0
total_output_tokens = 0

with open(jsonl_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        v = json.loads(line)
        num_versions += 1
        total_chars_all += v.get("total_chars", 0)
        total_input_tokens += v.get("total_input_tokens", 0)
        total_output_tokens += v.get("total_output_tokens", 0)
        for ch in v.get("chapters", []):
            total_chapters_all += 1
            all_chapter_lens.append(ch["char_count"])
        del v  # free memory immediately

all_chapter_lens.sort()
avg_chars_per_version = total_chars_all // num_versions if num_versions else 0
avg_chars_per_chapter = total_chars_all // total_chapters_all if total_chapters_all else 0
min_ch = all_chapter_lens[0] if all_chapter_lens else 0
max_ch = all_chapter_lens[-1] if all_chapter_lens else 0
median_ch = all_chapter_lens[len(all_chapter_lens) // 2] if all_chapter_lens else 0
file_size_mb = os.path.getsize(jsonl_path) / (1024 * 1024)

print(f"   {num_versions} versions, {total_chars_all:,} total chars, {file_size_mb:.1f} MB")

# ========== Build README ==========
readme = f"""---
language:
  - zh
license: cc-by-4.0
tags:
  - chinese-literature
  - creative-writing
  - hongloumeng
  - dream-of-the-red-chamber
  - red-chamber
  - cao-xueqin
  - 红楼梦
  - 曹雪芹
  - llm-generated
  - fiction
  - classical-chinese
  - chapter-novel
size_categories:
  - 10M<n<100M
task_categories:
  - text-generation
pretty_name: 红楼梦续写 · Dream of the Red Chamber Continuations
---

# 红楼梦续写 · Dream of the Red Chamber: 100 AI Continuations

---

## 项目简介

本数据集包含 **{num_versions} 个独立的AI续写版本**，续写中国古典文学巅峰之作《红楼梦》的第八十一回至第一百零八回（共28回）。所有续写严格遵循曹雪芹前八十回中埋下的伏笔、谶语和人物命运，**完全拒绝高鹗续书**。

### 为什么做这个数据集

《红楼梦》的结局是世界文学史上最大的悬案之一。曹雪芹约于1763年去世前未能完成全书，仅留下前八十回。1791年左右，高鹗发表了一百二十回本，补写了后四十回，但红学研究日益表明高鹗续书严重违背了曹雪芹在前八十回中精心布置的伏笔。

### 曹雪芹原意 vs 高鹗续书

| 情节 | 曹雪芹原意 | 高鹗续书 |
|---|---|---|
| **黛玉之死** | 泪尽而亡，呼应"绛珠还泪"神话 | 焚稿断痴情 |
| **宝玉宝钗婚姻** | "纵然是齐眉举案，到底意难平" | 掉包计骗婚 |
| **贾府败落** | 政治牵连，锦衣军抄家，"忽喇喇似大厦倾" | 败而复兴，"兰桂齐芳" |
| **结局** | "落了片白茫茫大地真干净"——彻底幻灭 | 家道复兴，皆大欢喜 |

### 生成方式

- **模型**：MiniMax-M2.7（通过Anthropic兼容API调用）
- **策略**：每回独立调用一次API（每版本28次，共{num_versions * 28:,}次调用）
- **上下文管理**：最近3回完整传入，更早章回压缩为回目+摘要
- **独立性**：{num_versions}个版本之间完全独立，不共享任何状态
- **容错**：无限重试，指数退避，不丢弃任何章回或版本
- **断点续跑**：每完成一回保存进度，中断后可从断点恢复

### 核心约束（写入Prompt）

1. 依据前八十回判词、红楼梦曲推进情节，每个人物命运必须与判词吻合
2. 黛玉泪尽而亡（绛珠还泪）
3. 宝钗宝玉婚姻体现"到底意难平"
4. 贾府因政治牵连被抄家
5. 人物性格与前八十回一脉相承
6. 体现"假作真时真亦假，无为有处有还无"的哲学
7. 诗词贴近原著水准
8. **绝对禁止使用高鹗续书的任何回目或情节**
9. 第108回大结局：宝玉随一僧一道飘然而去，白茫茫大地真干净

### 数据统计

| 指标 | 数值 |
|---|---|
| 版本数 | {num_versions} |
| 每版本章回数 | 28（第81-108回） |
| 总章回数 | {total_chapters_all:,} |
| 总字数 | {total_chars_all:,} |
| 每版本平均字数 | ~{avg_chars_per_version:,} |
| 每回平均字数 | ~{avg_chars_per_chapter:,} |
| 每回中位字数 | {median_ch:,} |
| 每回最短/最长 | {min_ch:,} / {max_ch:,} |
| 文件大小 | {file_size_mb:.1f} MB |

原著前八十回每回约6,000-8,000字。

### 使用方法

```python
from datasets import load_dataset

# 流式加载（省内存）
ds = load_dataset("{REPO_ID}", data_files="hlm_continuations.jsonl", split="train", streaming=True)
for version in ds:
    for ch in version["chapters"]:
        print(ch["chapter_number"], ch["char_count"])
```

### 潜在研究用途

- **计算文学分析**：对比AI生成的古典中文与曹雪芹原著的语言风格（虚词指纹、句长分布、对话比例、词汇丰富度）
- **叙事分歧研究**：分析{num_versions}个独立版本如何处理同一组伏笔——哪些判词被一致实现，哪些被不同解读？
- **人物一致性评估**：检验AI生成的人物行为是否与前八十回的性格塑造吻合
- **古典中文生成基准**：评估大语言模型生成长篇古典中文叙事的能力
- **数字人文**：探索世界文学史上最受关注的未完成作品的多种可能结局

---

## Overview (English)

This dataset contains **{num_versions} independent AI-generated continuations** of the classical Chinese novel *Dream of the Red Chamber* (红楼梦 / *Hónglóumèng*), widely regarded as the greatest work of Chinese fiction. Each continuation covers **chapters 81–108** (28 chapters), picking up where Cao Xueqin's (曹雪芹) authenticated first 80 chapters end.

All {num_versions} versions strictly follow Cao Xueqin's original foreshadowing, prophecies, and character arcs as established in the first 80 chapters. **None of them follow the widely-read Gao E (高鹗) continuation** (chapters 81–120), which most scholars agree deviates significantly from Cao Xueqin's intended plot.

## Why This Dataset Exists

The ending of *Dream of the Red Chamber* is one of the great unsolved puzzles in world literature. Cao Xueqin died before completing the novel (~1763), leaving only 80 chapters. Around 1791, Gao E published a 120-chapter version with 40 additional chapters, but scholarship increasingly shows Gao E's continuation contradicts the extensive foreshadowing Cao Xueqin embedded in the first 80 chapters.

Key divergences between Cao Xueqin's intent (as evidenced by prophecy poems, Zhi Yanzhai's (脂砚斋) commentary, and textual clues) and Gao E's continuation include:

| Plot Point | Cao Xueqin's Intent | Gao E's Version |
|---|---|---|
| **Lin Daiyu's death** | Dies when her tears run out (绛珠还泪), fulfilling the Crimson Pearl Grass mythology | Burns her manuscripts in despair ("焚稿断痴情") |
| **Baoyu-Baochai marriage** | Happens but is deeply melancholic — "even with perfect marital harmony, the heart remains restless" (到底意难平) | Engineered through a cruel "swap trick" (掉包计) |
| **Fall of the Jia family** | Political catastrophe, house raided by imperial guards — "like a great mansion collapsing" (忽喇喇似大厦倾) | Softened; family eventually recovers |
| **Ending** | "A vast white expanse of earth, truly clean" (白茫茫大地真干净) — total desolation | "Orchid and osmanthus flourish together" (兰桂齐芳) — restoration and hope |

This dataset provides {num_versions} different explorations of what the novel's ending might have looked like had Cao Xueqin completed it, all generated by a frontier LLM guided by the original text's internal evidence.

## Dataset Statistics

| Metric | Value |
|---|---|
| Number of versions | {num_versions} |
| Chapters per version | 28 (chapters 81–108) |
| Total chapters | {total_chapters_all:,} |
| Total characters (Chinese) | {total_chars_all:,} |
| Average characters per version | ~{avg_chars_per_version:,} |
| Average characters per chapter | ~{avg_chars_per_chapter:,} |
| Median characters per chapter | {median_ch:,} |
| Min/Max chapter length | {min_ch:,} / {max_ch:,} |
| File size | {file_size_mb:.1f} MB |
| Total input tokens consumed | {total_input_tokens:,} |
| Total output tokens generated | {total_output_tokens:,} |

For reference, Cao Xueqin's original chapters average ~6,000–8,000 characters each.

## Generation Method

- **Model**: MiniMax-M2.7 via Anthropic-compatible API
- **Approach**: One API call per chapter (28 calls per version, {num_versions * 28:,} total calls)
- **Context management**: The 3 most recent chapters are passed in full; earlier chapters are compressed into title + 600-character summaries
- **Independence**: Each of the {num_versions} versions is generated from scratch with no shared state between versions
- **Retry policy**: Infinite retry with exponential backoff (30s, 60s, ... up to 5 min) — no chapter or version is ever skipped
- **Progress saving**: Checkpoint saved after every chapter; fully resumable on interruption

### Guiding Principles in the Prompt

The model was given the following hard constraints:

1. Follow the prophecy poems (判词) and *Dream of the Red Chamber* song cycle (红楼梦曲) — each character's fate must match these predictions
2. Lin Daiyu dies when her tears are exhausted (泪尽而亡), echoing the Crimson Pearl repaying-tears mythology (绛珠还泪)
3. Baoyu and Baochai's marriage must convey "even with perfect marital harmony, the heart remains restless" (到底意难平)
4. The Jia family falls due to political entanglement, not mere domestic mismanagement
5. Character personalities must be fully consistent with the first 80 chapters
6. Embody Cao Xueqin's philosophy: "When the false is taken for the true, the true becomes false" (假作真时真亦假)
7. Poetry quality must approach the original
8. **Absolutely no Gao E continuation content** — all chapter titles and plots must be original
9. Chapter 108 is the finale: Baoyu departs with a monk and a Taoist into the snow, returning to the Great Barren Mountain (大荒山); the story ends with "a vast white expanse of earth, truly clean" (白茫茫大地真干净)

## Data Format

JSONL (one JSON object per line, one line per version):

```json
{{
  "version_id": 1,
  "model": "MiniMax-M2.7",
  "started_at": "2026-03-21T...",
  "completed_at": "2026-03-21T...",
  "num_chapters": 28,
  "total_chars": 240223,
  "avg_chars_per_chapter": 8579,
  "total_input_tokens": 502341,
  "total_output_tokens": 168432,
  "chapters": [
    {{
      "chapter_number": 81,
      "text": "第八十一回 ...(full chapter text)...",
      "char_count": 6229,
      "input_tokens": 360,
      "output_tokens": 4643
    }},
    ...
  ]
}}
```

## Usage

### Streaming (memory-efficient)

```python
from datasets import load_dataset

ds = load_dataset(
    "{REPO_ID}",
    data_files="hlm_continuations.jsonl",
    split="train",
    streaming=True
)

for version in ds:
    print(f"Version {{version['version_id']}}: {{version['total_chars']}} chars")
    for ch in version["chapters"]:
        print(f"  Chapter {{ch['chapter_number']}}: {{ch['char_count']}} chars")
```

### Load fully

```python
from datasets import load_dataset

ds = load_dataset("{REPO_ID}", data_files="hlm_continuations.jsonl", split="train")
print(f"{{len(ds)}} versions loaded")

# Access a specific version and chapter
version_0 = ds[0]
chapter_81 = version_0["chapters"][0]
print(chapter_81["text"][:200])
```

### Raw Python (no dependencies)

```python
import json

with open("hlm_continuations.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        version = json.loads(line)
        print(f"Version {{version['version_id']}}: {{version['total_chars']}} chars")
```

## Potential Research Uses

- **Computational literary analysis**: Compare AI-generated classical Chinese prose against Cao Xueqin's authenticated text across stylistic dimensions (function word fingerprints, sentence length distributions, dialogue ratios, vocabulary richness)
- **Narrative divergence studies**: Analyze how {num_versions} independent continuations handle the same set of foreshadowed plot points — which prophecies are universally fulfilled vs. interpreted differently?
- **Character consistency evaluation**: Measure whether AI-generated character behavior aligns with personalities established in the first 80 chapters
- **Classical Chinese generation benchmarking**: Evaluate LLM capability in producing extended classical Chinese prose with complex narrative structure
- **Digital humanities**: Explore alternative endings to one of world literature's most studied unfinished works

## Citation

If you use this dataset, please cite:

```bibtex
@dataset{{hlm_continuations_2026,
  title={{红楼梦续写: 100 AI Continuations of Dream of the Red Chamber}},
  author={{PursuitOfDataScience}},
  year={{2026}},
  url={{https://huggingface.co/datasets/{REPO_ID}}},
  note={{Generated using MiniMax-M2.7, following Cao Xueqin's original foreshadowing}}
}}
```

## License

CC-BY-4.0

## Disclaimer

This dataset is AI-generated creative fiction intended for research purposes. It does not claim to represent Cao Xueqin's actual lost manuscript. The continuations are interpretive explorations based on textual evidence from the first 80 chapters.
"""

# ========== Upload ==========
from huggingface_hub import HfApi, create_repo

api = HfApi()

print(f"📦 Creating repo: {REPO_ID}")
create_repo(REPO_ID, repo_type="dataset", exist_ok=True)

print(f"📤 Uploading JSONL ({file_size_mb:.1f} MB)...")
api.upload_file(
    path_or_fileobj=jsonl_path,
    path_in_repo="hlm_continuations.jsonl",
    repo_id=REPO_ID,
    repo_type="dataset",
)

print("📤 Uploading README...")
with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False, encoding="utf-8") as f:
    f.write(readme)
    readme_path = f.name

api.upload_file(
    path_or_fileobj=readme_path,
    path_in_repo="README.md",
    repo_id=REPO_ID,
    repo_type="dataset",
)
os.remove(readme_path)

print(f"\n✅ Done! https://huggingface.co/datasets/{REPO_ID}")