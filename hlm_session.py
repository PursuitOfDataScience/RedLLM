#!/usr/bin/env python3
"""Tools for writing one 红楼梦续写 version in an interactive Claude Code session.

The session follows hlm_session.md, a brief kept next to this script but out of git.

    hlm_session.py prepare <dir> --model M --effort E   set up a session folder; hlm_session.sh runs it
    hlm_session.py check <n> [--dir D]                  check chapter n: prints OK, or FIX lines and exits 1
    hlm_session.py finish [--dir D]                     check all 28, pack them into D/hlm_continuations.jsonl
    hlm_session.py publish [--dir D] [--main M] [--dry-run]
                                                        append the packed version to the dataset and upload

D defaults to the current directory, which is the session's own folder. check holds a chapter
to the dataset's hard checks (the ones the worker's post-merge check stops on) and to the
brief's own rules: the right 回目 number, an 8+8 回目, at least 4,000 characters, the ending
line and “” quotes. Its messages are in Chinese because the writing session reads them.

finish takes token counts from the session's transcripts. Each chapter gets the usage since
the previous chapter file was written, and input counts cache reads and writes, as the
script's numbers do. publish runs the same hlm_merge.py and hlm_upload.py as the worker;
with --dry-run it prints the card the upload would carry and changes nothing.
"""
import argparse, glob, hashlib, json, os, re, shutil, subprocess, sys, tempfile
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
ORIG = os.path.join(REPO, "data", "Hongloumeng.txt")
BRIEF = os.path.join(REPO, "hlm_session.md")
MAIN = "/project/rcc/youzhi/data/hlm_continuations.jsonl"
FIRST, LAST = 81, 108
BANNED = ("掉包", "焚稿", "中乡魁", "兰桂齐芳")
HAN = re.compile(r"[一-鿿]")
HEAD = re.compile(r"^第[一二三四五六七八九十百零〇]+回")
TITLE = re.compile(r"第[零一二三四五六七八九十百千\d]+回[\s　:：]*(.*)")
ENDING = ("且听下回分解。", "且听下回分解")


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


def now():
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat() + "Z"


def original():
    """The 120 table-of-contents titles and the 80 body chapters, each a list of lines."""
    lines = open(ORIG, encoding="utf-8-sig").read().split("\n")
    heads = [i for i, l in enumerate(lines) if HEAD.match(l.strip())]
    body_start = [i for i in heads if lines[i].strip().startswith("第一回")][1]
    toc = [lines[i].strip() for i in heads if i < body_start]
    seen, body = set(), []
    for i in heads:
        label = HEAD.match(lines[i].strip()).group()
        if i >= body_start and label not in seen:
            seen.add(label)
            body.append(i)
    assert len(toc) == 120 and len(body) == 80, (len(toc), len(body))
    return toc, [lines[a:b] for a, b in zip(body, body[1:] + [len(lines)])]


def gao_halves():
    """Each half of Gao E's 回目 for chapters 81-120, from the original's table of contents."""
    toc, _ = original()
    return {p for t in toc[80:] for p in re.split(r"[\s　]+", t)[1:3] if p}


def wrap(lines, width=300):
    """The original keeps a paragraph on one line, often past 2,000 characters, which a file
    viewer may cut, so long paragraphs are split at sentence ends into lines of about width."""
    out = []
    for line in lines:
        line = line.rstrip()
        if not line.strip():
            continue
        cur = ""
        for s in re.findall(r"[^。！？]*[。！？]+[”’」』]?|[^。！？]+$", line):
            if cur and len(cur) + len(s) > width:
                out.append(cur)
                cur = ""
            cur += s
        out.append(cur)
    return out


def problems(n, t, gao):
    """What chapter n's text breaks, in Chinese for the writing session, and its first line."""
    cn, out = cn_num(n), []
    first = t.split("\n")[0].strip()
    m = TITLE.match(first)
    if not first.startswith(f"第{cn}回"):
        out.append(f"第一行须以“第{cn}回”开头，后接回目")
    halves = [p for p in re.split(r"[\s　，,、；;]+", m.group(1)) if p] if m else []
    shape = "+".join(str(len(HAN.findall(p))) for p in halves) or "无"
    if shape != "8+8":
        out.append(f"回目须为八言对偶：上下联各八字，用全角空格隔开（现为 {shape}）")
    if any(p in gao for p in halves):
        out.append("回目与高鹗续书的回目相同，须另拟")
    heads = len(re.findall(rf"第(?:{cn}|{n})回[ \t　]+(?=\S)", t))
    if heads > 1:
        out.append(f"文中有{heads}处“第{cn}回”回目，只留开头一处")
    if len(t) < 4000:
        out.append(f"全回只有{len(t)}字，须写满4000-5000字")
    ends = t.rstrip().endswith(ENDING)
    if n < LAST and not ends:
        out.append("结尾须是“不知后事如何，且听下回分解。”")
    if n == LAST and ends:
        out.append("第一百零八回是全书结局，结尾不用“且听下回分解”")
    latin = list(dict.fromkeys(re.findall(r"[A-Za-z#*`]+", t)))
    if latin:
        out.append("文中有西文字母或 Markdown 符号：" + "、".join(latin[:8]))
    if "�" in t:
        out.append("文中有乱码字符 U+FFFD")
    hit = [k for k in BANNED if k in t]
    if hit:
        out.append("文中有禁用词：" + "、".join(hit))
    if re.search("[\"']", t):
        out.append("对话须用中文引号“”，文中有英文引号")
    return out, first


def read_chapter(d, n):
    p = os.path.join(d, f"{n:03d}.txt")
    return open(p, encoding="utf-8").read().strip() if os.path.exists(p) else None


def load_meta(d):
    p = os.path.join(d, "meta.json")
    if not os.path.exists(p):
        sys.exit(f"{d} is not a session folder: no meta.json")
    return json.load(open(p, encoding="utf-8"))


def merged(d):
    p = os.path.join(d, ".merged_count")
    return os.path.exists(p) and int(open(p).read()) >= 1


def chapter_tokens(d):
    """Chapter number -> (input, output) tokens from the session's transcripts, and whether
    any were found. A response spans several transcript lines and a resumed session can copy
    earlier ones, so responses count once by id and file writes once by tool-use id."""
    root = os.environ.get("CLAUDE_CONFIG_DIR") or os.path.expanduser("~/.claude")
    folder = os.path.join(root, "projects", re.sub(r"[^A-Za-z0-9]", "-", d))
    msgs, tools = {}, set()
    for path in glob.glob(os.path.join(folder, "*.jsonl")):
        for line in open(path, encoding="utf-8"):
            try:
                e = json.loads(line)
            except ValueError:
                continue
            m = e.get("message") or {}
            if e.get("type") != "assistant" or not isinstance(m, dict) or not m.get("id"):
                continue
            r = msgs.setdefault(m["id"], {"ts": e.get("timestamp", ""), "in": 0, "out": 0, "writes": []})
            u = m.get("usage") or {}
            r["in"] = max(r["in"], u.get("input_tokens", 0) + u.get("cache_creation_input_tokens", 0)
                          + u.get("cache_read_input_tokens", 0))
            r["out"] = max(r["out"], u.get("output_tokens", 0))
            for b in m.get("content") or []:
                if not (isinstance(b, dict) and b.get("type") == "tool_use"
                        and b.get("name") in ("Write", "Edit") and b.get("id") not in tools):
                    continue
                tools.add(b.get("id"))
                p = os.path.abspath(os.path.join(d, (b.get("input") or {}).get("file_path", "")))
                k = re.fullmatch(r"(\d{3})\.txt", os.path.basename(p))
                if k and os.path.dirname(p) == d and FIRST <= int(k.group(1)) <= LAST:
                    r["writes"].append(int(k.group(1)))
    tokens, acc, last = {}, [0, 0], None
    for r in sorted(msgs.values(), key=lambda r: r["ts"]):
        acc = [acc[0] + r["in"], acc[1] + r["out"]]
        for n in r["writes"]:
            a, b = tokens.get(n, (0, 0))
            tokens[n], acc, last = (a + acc[0], b + acc[1]), [0, 0], n
    if last is not None:
        a, b = tokens[last]
        tokens[last] = (a + acc[0], b + acc[1])
    return tokens, bool(msgs)


def cmd_prepare(a):
    d = os.path.abspath(a.dir)
    if os.path.exists(os.path.join(d, "meta.json")):
        sys.exit(f"{d} is already a session folder")
    if not os.path.exists(BRIEF):
        sys.exit(f"no brief at {BRIEF}: it is gitignored, so a fresh clone needs its own copy")
    os.makedirs(os.path.join(d, "原著"), exist_ok=True)
    shutil.copyfile(BRIEF, os.path.join(d, "BRIEF.md"))
    for k, lines in enumerate(original()[1], 1):
        with open(os.path.join(d, "原著", f"{k:03d}.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(wrap(lines)) + "\n")
    with open(os.path.join(d, "progress.md"), "w", encoding="utf-8") as f:
        f.write("# 进度\n\n每写完一回，在下面加一行：回数 | 回目 | 字数 | 一句话要点\n\n")
    meta = {"model": a.model, "effort": a.effort, "created_at": now(),
            "brief_sha256": hashlib.sha256(open(BRIEF, "rb").read()).hexdigest()}
    with open(os.path.join(d, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    print(f"prepared {d}")


def cmd_check(a):
    n, d = a.n, os.path.abspath(a.dir)
    if not FIRST <= n <= LAST:
        sys.exit(f"回数须在 {FIRST}-{LAST} 之间")
    t = read_chapter(d, n)
    if t is None:
        print(f"FIX 第{cn_num(n)}回：找不到 {n:03d}.txt")
        sys.exit(1)
    probs, first = problems(n, t, gao_halves())
    if probs:
        print(f"FIX 第{cn_num(n)}回（{len(t)}字）：")
        for p in probs:
            print(f"- {p}")
        sys.exit(1)
    print(f"OK 第{cn_num(n)}回：{len(t)}字。{first}")


def cmd_finish(a):
    d = os.path.abspath(a.dir)
    meta = load_meta(d)
    if merged(d):
        sys.exit("这一版已经发布，不再打包")
    gao, chapters, bad = gao_halves(), [], []
    for n in range(FIRST, LAST + 1):
        t = read_chapter(d, n)
        if t is None:
            bad.append(f"第{cn_num(n)}回：找不到 {n:03d}.txt")
            continue
        bad += [f"第{cn_num(n)}回：{p}" for p in problems(n, t, gao)[0]]
        chapters.append({"chapter_number": n, "text": t, "char_count": len(t)})
    if bad:
        print("FIX")
        for b in bad:
            print(f"- {b}")
        sys.exit(1)
    tokens, found = chapter_tokens(d)
    for c in chapters:
        c["input_tokens"], c["output_tokens"] = tokens.get(c["chapter_number"], (0, 0))
    total = sum(c["char_count"] for c in chapters)
    rec = {"version_id": 0, "model": meta["model"], "effort": meta["effort"], "method": "session",
           "started_at": meta["created_at"], "chapters": chapters,
           "total_input_tokens": sum(c["input_tokens"] for c in chapters),
           "total_output_tokens": sum(c["output_tokens"] for c in chapters),
           "num_chapters": len(chapters), "total_chars": total,
           "avg_chars_per_chapter": total // len(chapters), "completed_at": now()}
    out = os.path.join(d, "hlm_continuations.jsonl")
    with open(out + ".tmp", "w", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    os.replace(out + ".tmp", out)
    print(f"OK 二十八回已打包：{total:,}字，平均每回{rec['avg_chars_per_chapter']:,}字，"
          f"输出 {rec['total_output_tokens']:,} tokens。")
    if not found:
        print("没找到会话记录，tokens 记为 0。")
    print("发布须由用户明确要求：hlm_session.py publish")


def cmd_publish(a):
    d, main = os.path.abspath(a.dir), os.path.abspath(a.main)
    packed = os.path.join(d, "hlm_continuations.jsonl")
    if not os.path.exists(packed):
        sys.exit("还没有打包：先运行 finish")
    if merged(d):
        sys.exit("这一版已经发布过了")
    merge = [sys.executable, "-I", os.path.join(REPO, "hlm_merge.py")]
    upload = [sys.executable, "-u", os.path.join(REPO, "hlm_upload.py")]
    if a.dry_run:
        with tempfile.TemporaryDirectory() as tmp:
            work = os.path.join(tmp, "session")
            os.makedirs(work)
            shutil.copyfile(packed, os.path.join(work, "hlm_continuations.jsonl"))
            shutil.copyfile(main, os.path.join(tmp, "main.jsonl"))
            for cmd in (merge + [os.path.join(tmp, "main.jsonl"), work],
                        upload + [os.path.join(tmp, "main.jsonl"), "--dry-run"]):
                if subprocess.run(cmd, cwd=REPO).returncode:
                    sys.exit(f"dry run failed at {os.path.basename(cmd[2])}")
        return
    lock = ["flock", main + ".lock"]
    if subprocess.run(lock + merge + [main, d], cwd=REPO).returncode:
        sys.exit("并入数据集失败，什么也没上传")
    if subprocess.run(lock + upload + [main], cwd=REPO).returncode:
        sys.exit("上传失败：这一版已并入数据集，下次上传会带上它")


ap = argparse.ArgumentParser(description="Write one 红楼梦续写 version in a Claude Code session.")
sub = ap.add_subparsers(dest="cmd", required=True)
p = sub.add_parser("prepare", help="set up a session folder")
p.add_argument("dir")
p.add_argument("--model", required=True)
p.add_argument("--effort", required=True)
p = sub.add_parser("check", help="check one chapter")
p.add_argument("n", type=int)
p.add_argument("--dir", default=".")
p = sub.add_parser("finish", help="check all 28 chapters and pack the version")
p.add_argument("--dir", default=".")
p = sub.add_parser("publish", help="add the packed version to the dataset and upload it")
p.add_argument("--dir", default=".")
p.add_argument("--main", default=MAIN)
p.add_argument("--dry-run", action="store_true")
args = ap.parse_args()
{"prepare": cmd_prepare, "check": cmd_check, "finish": cmd_finish, "publish": cmd_publish}[args.cmd](args)
