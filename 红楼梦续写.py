#!/usr/bin/env python3
"""
红楼梦续写 × 100版本 (Dream of the Red Chamber: 100 AI Continuations)

Usage:
    python 红楼梦续写.py <output_dir> [--model MODEL] [--effort LEVEL] [--target N]
    python 红楼梦续写.py ./output                        # Claude Opus 5.5, effort max
    python 红楼梦续写.py ./output --model MiniMax-M2.7   # how versions 1-92 were made

Requires: claude-* models: a logged-in Claude Code CLI (`claude auth status`)
          MiniMax-M2.7:    ANTHROPIC_API_KEY env var (MiniMax Anthropic-compatible API)
Output:   <output_dir>/hlm_continuations.jsonl (one version per line, resumable).
          Versions are appended until the file holds --target of them, any model.
"""

import argparse, subprocess, json, os, sys, time, re
from datetime import datetime

ap = argparse.ArgumentParser(description="红楼梦续写: AI continuations of chapters 81-108")
ap.add_argument("output_dir")
ap.add_argument("--model", default="claude-opus-5-5",
                help="claude-* (through the Claude Code CLI) or MiniMax-M2.7")
ap.add_argument("--effort", default="max",
                choices=["low", "medium", "high", "xhigh", "max"], help="Claude effort level")
ap.add_argument("--target", type=int, default=100,
                help="stop once the JSONL holds this many versions")
ap.add_argument("--chapters", type=int, default=28,
                help="chapters per version; lower it for a quick test")
args = ap.parse_args()

OUTPUT_DIR = args.output_dir
os.makedirs(OUTPUT_DIR, exist_ok=True)
JSONL_FILE = os.path.join(OUTPUT_DIR, "hlm_continuations.jsonl")
MODEL = args.model
IS_CLAUDE = MODEL.startswith("claude")
# MiniMax keeps the original name so its unfinished version still resumes
PROGRESS_FILE = os.path.join(
    OUTPUT_DIR, f".progress.{MODEL}.json" if IS_CLAUDE else ".progress.json")

api_key = os.getenv("ANTHROPIC_API_KEY")
if not IS_CLAUDE and not api_key:
    print("❌ ANTHROPIC_API_KEY not set")
    sys.exit(1)

# ========== Config ==========
TOTAL_VERSIONS = args.target
TOTAL_CHAPTERS = args.chapters
START_CHAPTER = 81
MAX_OUTPUT = 128000 if IS_CLAUDE else 131072  # Opus 5.5 caps output at 128K
THINKING_BUDGET = 1024  # MiniMax only; Claude thinking depth is set by --effort

# ========== Prompts ==========
CORE_PRINCIPLES = """【最高优先级指令：禁止使用高鹗续书】
高鹗所续后四十回（第81-120回）与曹雪芹原意严重不符。你必须完全抛弃高鹗的版本，包括：
- 禁止使用高鹗的任何回目名称（如"占旺相四美钓游鱼""奉严词两番入家塾""美香菱屈受贪夫棒"等）
- 禁止使用"掉包计"（把宝钗换黛玉骗宝玉成亲）的情节
- 禁止使用"焚稿断痴情"的情节
- 禁止使用宝玉中举后出家的情节
- 禁止使用贾府"兰桂齐芳"复兴的情节
你必须自创全新的回目和情节，完全基于曹雪芹前八十回的伏笔。

【核心原则：严格遵守曹雪芹原笔原意】
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
- 对话一律用中文引号“”。
- 构思与正文都不要整段照录前八十回原文（判词、曲词、诗词），需要时只化用一两句。
- 回目必须是你原创的八言对偶（与前八十回回目同式），禁止使用高鹗续书的任何回目。"""


# ========== Helpers ==========
def cn_num(n):
    """81 -> 八十一, 108 -> 一百零八, the way the original numbers its chapters."""
    d = "零一二三四五六七八九"
    h, t, u = n // 100, n // 10 % 10, n % 10
    s = d[h] + "百" if h else ""
    if t:
        s += ("" if t == 1 and not h else d[t]) + "十"
    elif h and u:
        s += "零"
    return s + (d[u] if u else "")

def clean_chapter(text, ch_num):
    """Cut anything written before this chapter's 回目 (low effort can leak its plan
    there, even on the same line), number the heading in Chinese, and turn ASCII
    quotes into “” and ‘’. Quotes alternate across the whole text because quoted
    poems span lines. None means no 回目 was found, so the caller retries."""
    m = re.search(rf"第(?:{cn_num(ch_num)}|{ch_num})回[ \t　]+(?=\S)", text)
    if not m:
        return None
    text = re.sub(rf"^第{ch_num}回", f"第{cn_num(ch_num)}回", text[m.start():].strip())
    for q, pair in (('"', "“”"), ("'", "‘’")):
        parts = text.split(q)
        text = "".join(s + pair[i % 2] for i, s in enumerate(parts[:-1])) + parts[-1]
    return text

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

def call_claude(system, user_msg):
    """One headless Claude Code call per chapter. --setting-sources "" keeps the user's
    CLAUDE.md, hooks and plugins out of the request, and the prompt goes over stdin
    because it outgrows argv. ANTHROPIC_* and the parent session's CLAUDE* variables
    are dropped so a MiniMax key is never sent to Anthropic and nesting is allowed."""
    cmd = ["claude", "-p", "--model", MODEL, "--effort", args.effort,
           "--system-prompt", system, "--tools", "", "--setting-sources", "",
           "--strict-mcp-config", "--disable-slash-commands",
           "--no-session-persistence", "--output-format", "json"]
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("ANTHROPIC_", "CLAUDE")) or k == "CLAUDE_CONFIG_DIR"}
    env.update(ENABLE_CLAUDEAI_MCP_SERVERS="false", CLAUDE_CODE_MAX_OUTPUT_TOKENS=str(MAX_OUTPUT))
    try:
        proc = subprocess.run(cmd, input=user_msg, capture_output=True, text=True,
                              encoding="utf-8", env=env, timeout=3600)
    except subprocess.TimeoutExpired:
        print("      claude call timed out after 1h", flush=True)
        return None, {}
    try:
        resp = json.loads(proc.stdout)
    except json.JSONDecodeError:
        print(f"      claude exit {proc.returncode}: {(proc.stderr or proc.stdout)[-300:].strip()}", flush=True)
        return None, {}
    if resp.get("is_error") or resp.get("stop_reason") != "end_turn" or not resp.get("result"):
        print(f"      claude: {resp.get('stop_reason')} {str(resp.get('result'))[:200]}", flush=True)
        return None, {}
    u = resp.get("usage", {})
    usage = {
        "input_tokens": u.get("input_tokens", 0) + u.get("cache_creation_input_tokens", 0)
                        + u.get("cache_read_input_tokens", 0),
        "output_tokens": u.get("output_tokens", 0),  # includes thinking
    }
    return resp["result"], usage

def call_minimax(system, user_msg, tmp_label):
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
    cn = cn_num(ch_num)
    if ctx:
        user = f"""{CORE_PRINCIPLES}

{ctx}

紧接前文，写第{cn}回。只写这一回，写满4000-5000字。

格式：
第{cn}回 [你原创的八言对偶回目，禁止使用高鹗续书回目]
话说...（正文4000-5000字）
...
{"此为全书大结局。结尾不用'且听下回分解'，以收束全书的方式结尾，呼应'白茫茫大地真干净'。" if ch_num == 108 else "不知后事如何，且听下回分解。"}

开始："""
    else:
        user = f"""{CORE_PRINCIPLES}

写第{cn}回。这是续写的第一回（紧接原著第八十回）。只写这一回，写满4000-5000字。

格式：
第{cn}回 [你原创的八言对偶回目，禁止使用高鹗续书回目]
话说...（正文4000-5000字）
...
不知后事如何，且听下回分解。

开始："""

    attempt = 0
    while True:
        attempt += 1
        if IS_CLAUDE:
            text, usage = call_claude(SYSTEM_MSG, user)
        else:
            text, usage = call_minimax(SYSTEM_MSG, user, f"v{ch_num}_a{attempt}")
        if text:
            text = clean_chapter(text, ch_num)
            if text:
                return text, usage
            print(f"      no 第{cn}回 heading in the output", flush=True)
        wait = min(30 * attempt, 300)  # cap at 5 min
        print(f"      retry #{attempt} in {wait}s...", flush=True)
        time.sleep(wait)

def completed_ids():
    if not os.path.exists(JSONL_FILE):
        return []
    with open(JSONL_FILE, "r", encoding="utf-8") as f:
        return [json.loads(line)["version_id"] for line in f if line.strip()]

def count_completed():
    return len(completed_ids())

def load_progress():
    """Load this model's in-progress version from its progress file, or return None."""
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            pass
    return None

def save_progress(data):
    """Save in-progress version to progress file."""
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

def clear_progress():
    try: os.remove(PROGRESS_FILE)
    except: pass

def finalize_version(data):
    """Write completed version to JSONL and clear progress."""
    taken = completed_ids()
    if data["version_id"] in taken:  # another model's run used this id meanwhile
        data["version_id"] = max(taken) + 1
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
            "model": MODEL,
            "effort": args.effort if IS_CLAUDE else None,
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

        save_progress(data)

        mark = "✓" if n >= 3000 else "~"
        print(f"    {mark} 第{ch}回 {n}字 ({elapsed:.0f}s)", flush=True)

        if i < TOTAL_CHAPTERS - 1:
            time.sleep(10)

    return finalize_version(data)


# ========== Main ==========
done = count_completed()
progress = load_progress()

print(f"{'='*60}")
print(f"红楼梦续写 × {TOTAL_VERSIONS} versions  [{MODEL}{', effort ' + args.effort if IS_CLAUDE else ''}]")
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
    print(f"  ✅ {result['total_chars']:,}字 avg={result['avg_chars_per_chapter']}字/回")
    start_v = vid + 1
    if start_v <= TOTAL_VERSIONS:
        time.sleep(30)

for v in range(start_v, TOTAL_VERSIONS + 1):
    if count_completed() >= TOTAL_VERSIONS:
        break

    print(f"\n📖 Version {v}/{TOTAL_VERSIONS}  [{datetime.now().strftime('%Y-%m-%d %H:%M')}]")
    result = generate_version(v)
    print(f"  ✅ {result['total_chars']:,}字 avg={result['avg_chars_per_chapter']}字/回")

    if v < TOTAL_VERSIONS:
        time.sleep(30)

print(f"\n{'='*60}")
print(f"✅ Done. {count_completed()} versions in {JSONL_FILE}")
print(f"{'='*60}")