#!/bin/bash
# Usage: hlm_session.sh [name] [-- claude options...]
#   Starts an interactive Claude Code session that writes one 红楼梦续写 version (第81-108回)
#   by following hlm_session.md, in $HLM_SESSIONS/<name>/. With no name it makes the next
#   numbered folder (001, 002, ...); with the name of an existing one it resumes that session.
#   HLM_MODEL (default claude-sonnet-5-5) and HLM_EFFORT (default high) pick the writer of a
#   new folder; a resumed one keeps its own. HLM_SESSIONS moves the root and HLM_PROMPT
#   replaces the opening message (for tests). Options after -- go to claude, e.g. -- --bg.
#
# The session is isolated the way the generator's claude -p calls are: its own system prompt,
# no settings sources (so no CLAUDE.md, hooks or plugins), no MCP servers, and only Read,
# Write, Edit and Bash limited to hlm_session.py. ANTHROPIC_* and the CLAUDE* variables of a
# parent session are dropped, so a MiniMax key is never used and it can start from inside one.
set -eu
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PY=/software/python-miniforge-25.3.0-el8-x86_64/envs/AI/bin/python
ROOT=${HLM_SESSIONS:-/project/rcc/youzhi/data/hlm_sessions}
name=
if [ $# -gt 0 ] && [ "$1" != "--" ]; then name=$1; shift; fi
if [ "${1:-}" = "--" ]; then shift; fi
if [ -z "$name" ]; then
    n=1
    while [ -e "$ROOT/$(printf %03d "$n")" ]; do n=$((n + 1)); done
    name=$(printf %03d "$n")
fi
DIR=$ROOT/$name
if [ -f "$DIR/meta.json" ]; then
    resume=(--continue)
    prompt=${HLM_PROMPT:-继续。先读 progress.md，从下一回接着写。}
    MODEL=$("$PY" -c 'import json, sys; print(json.load(open(sys.argv[1]))["model"])' "$DIR/meta.json")
    EFFORT=$("$PY" -c 'import json, sys; print(json.load(open(sys.argv[1]))["effort"])' "$DIR/meta.json")
else
    MODEL=${HLM_MODEL:-claude-sonnet-5-5}
    EFFORT=${HLM_EFFORT:-high}
    mkdir -p "$ROOT"
    "$PY" "$REPO/hlm_session.py" prepare "$DIR" --model "$MODEL" --effort "$EFFORT"
    resume=()
    prompt=${HLM_PROMPT:-请读 BRIEF.md，然后照着做。}
fi
SYSTEM="你是曹雪芹，正在续写《红楼梦》第八十一回至第一百零八回。本目录里的 BRIEF.md 是这次写作的全部要求和步骤：先通读，再严格照做。"
for v in $(compgen -e); do
    case $v in CLAUDE_CONFIG_DIR) ;; CLAUDE*|ANTHROPIC_*) unset "$v" ;; esac
done
export ENABLE_CLAUDEAI_MCP_SERVERS=false CLAUDE_CODE_MAX_OUTPUT_TOKENS=128000
cd "$DIR"
echo "session folder $DIR, $MODEL, effort $EFFORT"
exec claude ${resume[@]+"${resume[@]}"} --model "$MODEL" --effort "$EFFORT" --system-prompt "$SYSTEM" \
    --setting-sources "" --strict-mcp-config --tools Read,Write,Edit,Bash \
    --permission-mode acceptEdits --allowedTools "Bash($PY $REPO/hlm_session.py *)" \
    -n "hlm-$name" "$@" "$prompt"
