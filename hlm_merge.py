"""Append a worker's finished but unmerged versions to the main JSONL as the next free ids.

Usage: python -I hlm_merge.py <main.jsonl> <worker_dir>   (run under the main file's flock)
"""
import json, sys

main, d = sys.argv[1], sys.argv[2]
local = [json.loads(l) for l in open(f"{d}/hlm_continuations.jsonl", encoding="utf-8") if l.strip()]
try:
    done = int(open(f"{d}/.merged_count").read())
except FileNotFoundError:
    done = 0
ids = [json.loads(l)["version_id"] for l in open(main, encoding="utf-8") if l.strip()]
for k in range(done, len(local)):
    v = local[k]
    assert len(v["chapters"]) == 28, (k, len(v["chapters"]))
    v["version_id"] = max(ids) + 1
    with open(main, "a", encoding="utf-8") as f:
        f.write(json.dumps(v, ensure_ascii=False) + "\n")
    ids.append(v["version_id"])
    open(f"{d}/.merged_count", "w").write(f"{k + 1}\n")
    print(f"merged {d} local version {k + 1} as version {v['version_id']}", flush=True)
