#!/bin/bash
# Usage: hlm_worker.sh <label> <name>...   (HLM_DATA overrides the data dir)
#
# Finishes one Opus 5.5 (effort high) version in each hlm_runs/<name>/ in turn, resuming any
# saved progress, appends it to the main JSONL as the next free version_id and publishes the
# dataset to the Hub. On the Claude seat's usage limit it sleeps until the reset the message
# names and resumes. Exits after the last name, on a generator crash, or at the next check
# once hlm_runs/STOP exists. Output goes to hlm_runs/<label>.log. Killing it is safe:
# progress is saved per chapter, and rerunning with the same names skips merged ones.
LABEL=${1:?usage: hlm_worker.sh <label> <name>...}
shift
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DATA=${HLM_DATA:-/project/rcc/youzhi/data}
RUNS=$DATA/hlm_runs
MAIN=$DATA/hlm_continuations.jsonl
LOG=$RUNS/$LABEL.log

eval "$(/software/python-miniforge-25.3.0-el8-x86_64/bin/conda shell.bash hook)"
conda activate /software/python-miniforge-25.3.0-el8-x86_64/envs/AI
# Only after activation: the env's binutils activate script reads unset variables
set -u
cd "$REPO"

say() { echo "[$LABEL $(date '+%F %T')] $*" >> "$LOG"; }
finished() { if [ -f "$1/hlm_continuations.jsonl" ]; then grep -c . "$1/hlm_continuations.jsonl"; else echo 0; fi; }
merged() { cat "$1/.merged_count" 2>/dev/null || echo 0; }

# Seconds until the reset named in the last limit message, plus 2 minutes; 30 minutes when
# the message has no parsable "resets <time> (<zone>)". A reset time under an hour in the
# past means it has just passed, not that it is tomorrow's.
limit_wait() {
    local when t tz target now
    when=$(grep 'claude usage limit' "$LOG" | tail -n 1 | sed -n 's/.*resets \(.*\) (\([A-Za-z_/]*\)).*/\1|\2/p')
    t=${when%%|*}
    tz=${when#*|}
    now=$(date +%s)
    if [ -n "$when" ] && target=$(TZ="$tz" date -d "$t" +%s 2>/dev/null); then
        if [ "$target" -le "$now" ]; then
            if [ $((now - target)) -lt 3600 ]; then target=$now; else target=$((target + 86400)); fi
        fi
        echo $((target - now + 120))
    else
        echo 1800
    fi
}

say "started: $*"
for NAME in "$@"; do
    DIR=$RUNS/$NAME
    mkdir -p "$DIR"
    while [ "$(merged "$DIR")" -lt 1 ]; do
        [ -e "$RUNS/STOP" ] && { say "STOP file found, exiting"; exit 0; }
        if [ "$(finished "$DIR")" -lt 1 ]; then
            say "$NAME: generating"
            python -u 红楼梦续写.py "$DIR" --effort high --target 1 >> "$LOG" 2>&1
            rc=$?
            if [ $rc -eq 3 ]; then
                w=$(limit_wait)
                say "usage limit, sleeping ${w}s until $(date -d "@$(( $(date +%s) + w ))" '+%F %T')"
                for ((i = 0; i < w; i += 60)); do
                    [ -e "$RUNS/STOP" ] && break
                    sleep 60
                done
                continue
            fi
            if [ $rc -ne 0 ] || [ "$(finished "$DIR")" -lt 1 ]; then
                say "$NAME: generator exited $rc without a finished version, stopping"
                exit 1
            fi
        fi
        if ! flock "$MAIN.lock" python3 -I "$REPO/hlm_merge.py" "$MAIN" "$DIR" >> "$LOG" 2>&1; then
            say "$NAME: merge failed, stopping"
            exit 1
        fi
        flock "$MAIN.lock" python -u hlm_upload.py "$MAIN" >> "$LOG" 2>&1 || say "$NAME: upload failed, the next one will include it"
    done
    say "$NAME: done"
done
say "all done"
