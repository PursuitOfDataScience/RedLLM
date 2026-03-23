#!/usr/bin/env python3
"""
红楼梦续写 × 100版本 (Dream of the Red Chamber: 100 AI Continuations)

Usage:
    python hlm_generate.py <output_dir>
    python hlm_generate.py ./output

Requires: ANTHROPIC_API_KEY env var (MiniMax M2.7 via Anthropic-compatible API)
Output:   <output_dir>/hlm_continuations.jsonl (one version per line, resumable)
"""

import subprocess, json, os, sys, time, re
from datetime import datetime

if len(sys.argv) < 2:
    print("Usage: python hlm_generate.py <output_dir>")
    sys.exit(1)

OUTPUT_DIR = sys.argv[1]
os.makedirs(OUTPUT_DIR, exist_ok=True)
JSONL_FILE = os.path.join(OUTPUT_DIR, "hlm_continuations.jsonl")

api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    print("❌ ANTHROPIC_API_KEY not set")
    sys.exit(1)

# ========== Config ==========
TOTAL_VERSIONS = 100
TOTAL_CHAPTERS = 28
START_CHAPTER = 81
MAX_OUTPUT = 131072
THINKING_BUDGET = 1024
MAX_RETRIES = 3

# ========== Prompts ==========
CORE_PRINCIPLES = """【最高优先级指令——禁止使用高鹗续书】
高鹗所续后四十回（第81-120回）与曹雪芹原意严重不符。你必须完全抛弃高鹗的版本，包括：
- 禁止使用高鹗的任何回目名称（如"占旺相四美钓游鱼""奉严词两番入家塾""美香菱屈受贪夫棒"等）
- 禁止使用"掉包计"（把宝钗换黛玉骗宝玉成亲）的情节
- 禁止使用"焚稿断痴情"的情节
- 禁止使用宝玉中举后出家的情节
- 禁止使用贾府"兰桂齐芳"复兴的情节
你必须自创全新的回目和情节，完全基于曹雪芹前八十回的伏笔。

【核心原则——严格遵守曹雪芹原笔原意】
1. 依据前八十回伏笔、谶语、判词、红楼梦曲推进情节，不可违背。
2. 黛玉泪尽而亡，呼应"绛珠还泪"。
3. 宝钗宝玉婚姻体现"到底意难平"的悲凉。
4. 贾府因政治牵连被抄家，"忽喇喇似大厦倾"。
5. 人物性格与前八十回一脉相承。
6. 体现"假作真时真亦假"的哲学和末世批判。
7. 诗词贴近原著水准。
8. 全书108回，第108回大结局：宝玉出家，白茫茫大地真干净。"""

SYSTEM_MSG = """你是曹雪芹，正在写《红楼梦》。你现在只需要写一回（一个章回）。

硬性要求：
- 这一回必须写满4000-5000字。不到4000字绝对不可以停笔。
- 写完整的小说正文：场景、对话、心理、诗词，一个都不能少。
- 每回至少3-4个完整场景。
- 古典章回体，语言风格与前八十回无异。
- 直接输出正文，不要任何解释、评论、前言。
- 回目必须是你原创的七言对偶，禁止使用高鹗续书的任何回目。"""


# ========== Helpers ==========
def estimate_tokens(text):
    return int(len(text) * 1.7)

def compress_chapter(text, max_chars=600):
    lines = text.strip().split('\n')
    title = lines[0] if lines else ""
    body = '\n'.join(lines[1:]) if len(lines) > 1 else text
    if len(body) > max_chars:
        body = body[:max_chars] + "…"
    return f"{title}\n{body}"

def split_into_chapters(texts):
    combined = "\n\n".join(texts)
    parts = re.split(r'(?=第[零一二三四五六七八九十百千\d]+回[\s　])', combined)
    return [p.strip() for p in parts
            if p.strip() and re.match(r'第[零一二三四五六七八九十百千\d]+回', p.strip())]

def build_context(all_texts):
    if not all_texts:
        return ""
    chapters = split_into_chapters(all_texts)
    if not chapters:
        full = "\n\n".join(all_texts)
        return f"【已写内容】\n{full[-8000:]}" if len(full) > 8000 else f"【已写内容】\n{full}"

    available = 204800 - estimate_tokens(CORE_PRINCIPLES + SYSTEM_MSG) - 16000 - 500
    recent_count = min(3, len(chapters))
    recent = chapters[-recent_count:]
    earlier = chapters[:-recent_count] if len(chapters) > recent_count else []
    recent_text = "\n\n".join(recent)

    if not earlier:
        if estimate_tokens(recent_text) <= available:
            return f"【已写内容】\n{recent_text}"
        return f"【上一回】\n{chapters[-1]}"

    earlier_text = "\n\n".join(compress_chapter(ch) for ch in earlier)
    if estimate_tokens(recent_text + earlier_text) <= available:
        return f"【前文摘要】\n{earlier_text}\n\n【最近几回】\n{recent_text}"

    remaining = available - estimate_tokens(recent_text)
    if remaining > 3000:
        return f"【前文摘要】\n{earlier_text[:int(remaining*0.6)]}…\n\n【最近几回】\n{recent_text}"
    return f"【上一回】\n{chapters[-1]}"

def call_api(system, user_msg, tmp_label):
    req_file = os.path.join(OUTPUT_DIR, f".tmp_{tmp_label}.json")
    body = {
        "model": "MiniMax-M2.7",
        "max_tokens": MAX_OUTPUT,
        "thinking": {"type": "enabled", "budget_tokens": THINKING_BUDGET},
        "system": system,
        "messages": [{"role": "user", "content": user_msg}]
    }
    with open(req_file, "w", encoding="utf-8") as f:
        json.dump(body, f, ensure_ascii=False)
    try:
        proc = subprocess.Popen(
            ["curl", "-s", "--max-time", "1800",
             "-X", "POST", "https://api.minimax.io/anthropic/v1/messages",
             "-H", "Content-Type: application/json",
             "-H", f"Authorization: Bearer {api_key}",
             "-d", f"@{req_file}"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, _ = proc.communicate()
        resp = json.loads(stdout.decode('utf-8'))
        usage = resp.get("usage", {})
        for item in resp.get("content", []):
            if isinstance(item, dict) and item.get("type") == "text":
                return item["text"], usage
        for item in resp.get("content", []):
            if isinstance(item, dict) and "text" in item and item.get("type") != "thinking":
                return item["text"], usage
        return None, usage
    except:
        return None, {}
    finally:
        try: os.remove(req_file)
        except: pass

def generate_chapter(ch_num, prev_texts):
    ctx = build_context(prev_texts)
    if ctx:
        user = f"""{CORE_PRINCIPLES}

{ctx}

紧接前文，写第{ch_num}回。只写这一回，写满4000-5000字。

格式：
第{ch_num}回 [你原创的七言对偶回目，禁止使用高鹗续书回目]
话说...（正文4000-5000字）
...
{"此为全书大结局。结尾不用'且听下回分解'，以收束全书的方式结尾，呼应'白茫茫大地真干净'。" if ch_num == 108 else "不知后事如何，且听下回分解。"}

开始："""
    else:
        user = f"""{CORE_PRINCIPLES}

写第{ch_num}回。这是续写的第一回（紧接原著第八十回）。只写这一回，写满4000-5000字。

格式：
第{ch_num}回 [你原创的七言对偶回目，禁止使用高鹗续书回目]
话说...（正文4000-5000字）
...
不知后事如何，且听下回分解。

开始："""

    for attempt in range(1, MAX_RETRIES + 1):
        text, usage = call_api(SYSTEM_MSG, user, f"v{ch_num}_a{attempt}")
        if text:
            return text, usage
        if attempt < MAX_RETRIES:
            print(f"      retry {attempt+1}/{MAX_RETRIES} in {30*attempt}s...")
            time.sleep(30 * attempt)
    return None, {}

def count_completed():
    if not os.path.exists(JSONL_FILE):
        return 0
    n = 0
    with open(JSONL_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip(): n += 1
    return n

def load_progress():
    """Load in-progress version from progress file, or return None."""
    pf = os.path.join(OUTPUT_DIR, ".progress.json")
    if os.path.exists(pf):
        try:
            with open(pf, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            pass
    return None

def save_progress(data):
    """Save in-progress version to progress file."""
    pf = os.path.join(OUTPUT_DIR, ".progress.json")
    with open(pf, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

def clear_progress():
    pf = os.path.join(OUTPUT_DIR, ".progress.json")
    try: os.remove(pf)
    except: pass

def finalize_version(data):
    """Write completed version to JSONL and clear progress."""
    total = sum(c["char_count"] for c in data["chapters"])
    data["num_chapters"] = len(data["chapters"])
    data["total_chars"] = total
    data["avg_chars_per_chapter"] = total // len(data["chapters"]) if data["chapters"] else 0
    data["completed_at"] = datetime.utcnow().isoformat() + "Z"

    with open(JSONL_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")
    clear_progress()
    return data

def generate_version(vid, resume_data=None):
    """Generate one version, saving progress after each chapter."""
    if resume_data:
        data = resume_data
        chapters = data["chapters"]
        texts = [c["text"] for c in chapters]
        done_ch = len(chapters)
        print(f"  ↻ Resuming version {vid} from chapter {START_CHAPTER + done_ch} ({done_ch}/{TOTAL_CHAPTERS} done)")
    else:
        data = {
            "version_id": vid,
            "model": "MiniMax-M2.7",
            "started_at": datetime.utcnow().isoformat() + "Z",
            "chapters": [],
            "total_input_tokens": 0,
            "total_output_tokens": 0,
        }
        chapters = []
        texts = []
        done_ch = 0

    for i in range(done_ch, TOTAL_CHAPTERS):
        ch = START_CHAPTER + i
        t0 = time.time()
        text, usage = generate_chapter(ch, texts)
        elapsed = time.time() - t0

        if text:
            n = len(text)
            data["chapters"].append({
                "chapter_number": ch,
                "text": text,
                "char_count": n,
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
            })
            texts.append(text)
            data["total_input_tokens"] += usage.get("input_tokens", 0)
            data["total_output_tokens"] += usage.get("output_tokens", 0)

            # Save progress after every chapter
            save_progress(data)

            mark = "✓" if n >= 3000 else "~"
            print(f"    {mark} 第{ch}回 {n}字 ({elapsed:.0f}s)", flush=True)
        else:
            print(f"    ✗ 第{ch}回 failed after {MAX_RETRIES} retries, aborting version", flush=True)
            return None

        if i < TOTAL_CHAPTERS - 1:
            time.sleep(10)

    return finalize_version(data)


# ========== Main ==========
done = count_completed()
progress = load_progress()

print(f"{'='*60}")
print(f"红楼梦续写 × {TOTAL_VERSIONS} versions")
print(f"Completed: {done}/{TOTAL_VERSIONS}")
if progress:
    print(f"In-progress: version {progress['version_id']}, {len(progress['chapters'])}/{TOTAL_CHAPTERS} chapters")
print(f"Output: {JSONL_FILE}")
print(f"{'='*60}")

if done >= TOTAL_VERSIONS:
    print("✅ All done!")
    sys.exit(0)

# If there's an in-progress version, resume it first
start_v = done + 1
if progress:
    vid = progress["version_id"]
    print(f"\n📖 Version {vid}/{TOTAL_VERSIONS} (resuming)  [{datetime.now().strftime('%Y-%m-%d %H:%M')}]")
    result = generate_version(vid, resume_data=progress)
    if result:
        print(f"  ✅ {result['total_chars']:,}字 avg={result['avg_chars_per_chapter']}字/回")
        done += 1
    else:
        print(f"  ❌ Version {vid} failed, skipping")
        clear_progress()
    start_v = vid + 1
    if start_v <= TOTAL_VERSIONS:
        time.sleep(30)

for v in range(start_v, TOTAL_VERSIONS + 1):
    if count_completed() >= TOTAL_VERSIONS:
        break

    print(f"\n📖 Version {v}/{TOTAL_VERSIONS}  [{datetime.now().strftime('%Y-%m-%d %H:%M')}]")
    result = generate_version(v)

    if result:
        print(f"  ✅ {result['total_chars']:,}字 avg={result['avg_chars_per_chapter']}字/回")
    else:
        print(f"  ❌ Version {v} failed, skipping")

    if v < TOTAL_VERSIONS:
        time.sleep(30)

print(f"\n{'='*60}")
print(f"✅ Done. {count_completed()} versions in {JSONL_FILE}")
print(f"{'='*60}")