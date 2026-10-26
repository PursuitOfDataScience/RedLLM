#!/usr/bin/env python3
"""
Upload 红楼梦续写 dataset to HuggingFace Hub.

Usage:
    pip install huggingface_hub
    huggingface-cli login
    python hlm_upload.py <jsonl_path> [--dry-run]

Example:
    python hlm_upload.py /project/rcc/youzhi/data/hlm_continuations.jsonl
    python hlm_upload.py /project/rcc/youzhi/data/hlm_continuations.jsonl --dry-run  # print the card only
"""
import sys, os, json, tempfile
from collections import defaultdict

if len(sys.argv) < 2:
    print("Usage: python hlm_upload.py <jsonl_path> [--dry-run]")
    sys.exit(1)

jsonl_path = sys.argv[1]
DRY_RUN = "--dry-run" in sys.argv[2:]
if not os.path.exists(jsonl_path):
    print(f"❌ {jsonl_path} not found")
    sys.exit(1)

USERNAME = "PursuitOfDataScience"
DATASET_NAME = "dream-of-the-red-chamber-continuations"
REPO_ID = f"{USERNAME}/{DATASET_NAME}"

# Display names for the dataset card
MODEL_NAMES = {
    "MiniMax-M2.7": "MiniMax-M2.7",
    "claude-opus-5-5": "Claude Opus 5.5",
    "claude-sonnet-5-5": "Claude Sonnet 5.5",
}

# ========== Compute stats by streaming JSONL (low memory) ==========
print("📊 Computing dataset statistics...")
per_model = defaultdict(lambda: {"ids": [], "chars": 0, "chapters": 0, "efforts": set()})
all_ids = []
total_chars_all = 0
total_chapters_all = 0

with open(jsonl_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        v = json.loads(line)
        pm = per_model[v.get("model", "MiniMax-M2.7")]
        pm["ids"].append(v["version_id"])
        pm["chars"] += v.get("total_chars", 0)
        pm["chapters"] += len(v.get("chapters", []))
        if v.get("effort"):
            pm["efforts"].add(v["effort"])
        all_ids.append(v["version_id"])
        total_chars_all += v.get("total_chars", 0)
        total_chapters_all += len(v.get("chapters", []))
        del v  # free memory immediately

if len(set(all_ids)) != len(all_ids):
    dupes = sorted({i for i in all_ids if all_ids.count(i) > 1})
    print(f"❌ duplicate version_id values: {dupes}")
    sys.exit(1)

def id_span(ids):
    ids = sorted(ids)
    if len(ids) > 1 and ids == list(range(ids[0], ids[-1] + 1)):
        return f"{ids[0]}-{ids[-1]}"
    return ", ".join(map(str, ids))

num_versions = len(all_ids)
avg_chars_per_chapter = total_chars_all // total_chapters_all if total_chapters_all else 0
file_size_mb = os.path.getsize(jsonl_path) / (1024 * 1024)

model_rows = []
for m, pm in sorted(per_model.items(), key=lambda kv: min(kv[1]["ids"])):
    if m.startswith("claude"):
        how = f"Claude Code `claude -p`, effort {', '.join(sorted(pm['efforts']))}"
    else:
        how = "MiniMax API, thinking budget 1,024"
    avg = pm["chars"] // pm["chapters"] if pm["chapters"] else 0
    model_rows.append(f"| {MODEL_NAMES.get(m, m)} | {len(pm['ids'])} | {id_span(pm['ids'])} | {avg:,} | {pm['chars']:,} | {how} |")
model_rows = "\n".join(model_rows)
models_note = " and ".join(MODEL_NAMES.get(m, m) for m in per_model)

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
  - n<1K
task_categories:
  - text-generation
pretty_name: 红楼梦续写 · Dream of the Red Chamber Continuations
---

<h1 align="center">红楼梦续写 · Dream of the Red Chamber Continuations</h1>

<p align="center"><b>{num_versions} complete AI endings to China's greatest unfinished novel: chapters 81-108, built only from Cao Xueqin's own foreshadowing, never from Gao E's.</b></p>

<p align="center">{num_versions}个独立续写版本，续第八十一回至第一百零八回，只依前八十回的伏笔与判词，完全抛弃高鹗续书。</p>

## 📚 What's inside 数据概览

| 模型 Model | 版本 Versions | version_id | 每回平均字数 Avg chars/chapter | 总字数 Total chars | 调用方式 How |
|---|---|---|---|---|---|
{model_rows}
| **合计 Total** | **{num_versions}** | | **{avg_chars_per_chapter:,}** | **{total_chars_all:,}** | |

Each version is 28 chapters (第81-108回): {total_chapters_all:,} chapters, {file_size_mb:.1f} MB. Cao Xueqin's own chapters run about 6,000-8,000 characters.

## 🚀 Quick start 快速上手

```python
from datasets import load_dataset

ds = load_dataset("{REPO_ID}", data_files="hlm_continuations.jsonl", split="train")
opus = ds.filter(lambda v: v["model"] == "claude-opus-5-5")
print(opus[0]["chapters"][0]["text"][:200])
```

## 🧭 Why 为什么续写

Cao Xueqin died around 1763 with 80 chapters written. The 120-chapter edition of about 1791 added Gao E's 40, which contradict much of what the first 80 foreshadow:

| 情节 Plot point | 曹雪芹原意 Cao's intent | 高鹗续书 Gao E |
|---|---|---|
| 黛玉之死 Daiyu's death | 泪尽而亡，呼应"绛珠还泪" | 焚稿断痴情 |
| 宝玉宝钗婚姻 The marriage | "到底意难平" | 掉包计 |
| 贾府 The Jia family | 政治牵连被抄家，"忽喇喇似大厦倾" | 败而复兴，"兰桂齐芳" |
| 结局 Ending | "白茫茫大地真干净" | 家道复兴 |

Every version is told to follow the 判词 and 红楼梦曲, keep each character true to the first 80 chapters, use no Gao E chapter title or plot, and close at chapter 108 with 宝玉出家 and 白茫茫大地真干净.

## ⚙️ How it was made 生成方式

- One model call per chapter, 28 per version, and versions share no state.
- Context: the last 3 chapters in full, earlier ones cut to title plus 600 characters.
- Every model gets the same prompts: [红楼梦续写.py](https://github.com/PursuitOfDataScience/RedLLM/blob/transformers/红楼梦续写.py).

## 🗂️ Format 数据格式

JSONL, one version per line. `output_tokens` includes thinking.

```json
{{"version_id": 93, "model": "claude-opus-5-5", "effort": "max", "num_chapters": 28, "total_chars": 140000,
 "avg_chars_per_chapter": 5000, "total_input_tokens": 900000, "total_output_tokens": 600000,
 "chapters": [{{"chapter_number": 81, "text": "第81回 ...", "char_count": 5000,
               "input_tokens": 600, "output_tokens": 20000}}]}}
```

## ⚠️ Good to know 注意

- This is AI-written fiction for research, not Cao Xueqin's lost manuscript.
- Chapter length differs a lot by model, so compare per model before pooling.
- Chapter headings in `text` mix 第八十一回 and 第81回 styles; use `chapter_number` instead.

## 📝 Cite 引用

CC-BY-4.0.

```bibtex
@dataset{{hlm_continuations_2026,
  title  = {{红楼梦续写: AI Continuations of Dream of the Red Chamber}},
  author = {{PursuitOfDataScience}},
  year   = {{2026}},
  url    = {{https://huggingface.co/datasets/{REPO_ID}}},
  note   = {{Generated with {models_note}, following Cao Xueqin's original foreshadowing}}
}}
```
"""

if DRY_RUN:
    print(readme)
    sys.exit(0)

# ========== Upload ==========
from huggingface_hub import HfApi, CommitOperationAdd, create_repo

api = HfApi()

print(f"📦 Creating repo: {REPO_ID}")
create_repo(REPO_ID, repo_type="dataset", exist_ok=True)

print(f"📤 Cleaning & uploading JSONL ({file_size_mb:.1f} MB) and README...")
DROP_KEYS = {"started_at", "completed_at"}
clean_path = jsonl_path + ".clean"
with open(jsonl_path, "r", encoding="utf-8") as fin, \
     open(clean_path, "w", encoding="utf-8") as fout:
    for line in fin:
        line = line.strip()
        if not line:
            continue
        v = json.loads(line)
        for k in DROP_KEYS:
            v.pop(k, None)
        v.setdefault("effort", None)  # same keys on every row; MiniMax has no effort level
        fout.write(json.dumps(v, ensure_ascii=False) + "\n")
        del v

with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False, encoding="utf-8") as f:
    f.write(readme)
    readme_path = f.name

# One commit, so the data and its card never disagree on the Hub
api.create_commit(
    repo_id=REPO_ID,
    repo_type="dataset",
    operations=[
        CommitOperationAdd(path_in_repo="hlm_continuations.jsonl", path_or_fileobj=clean_path),
        CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=readme_path),
    ],
    commit_message=f"{num_versions} versions: {models_note}",
)
os.remove(clean_path)
os.remove(readme_path)

print(f"\n✅ Done! https://huggingface.co/datasets/{REPO_ID}")
