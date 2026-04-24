#!/usr/bin/env bash
# opencode-ask.sh — delegate a task to OpenCode as a sub-agent
#
# Usage: opencode-ask.sh [OPTIONS] "prompt"
#
# Model options (mutually exclusive; --model takes precedence):
#   --model MODEL       Explicit model ID in provider/model format (e.g. openai/gpt-5.4).
#   --minimax-free      opencode/minimax-m2.5-free (default)
#   --nemotron-super-free opencode/nemotron-3-super-free
#   --gpt5.4            openai/gpt-5.4 (requires OPENAI_API_KEY)
#   --gpt5.3-codex      openai/gpt-5.3-codex (requires OPENAI_API_KEY)
#
# Other options:
#   --silence-timeout N Seconds of silence before declaring a hang (default 300)
#
# Exit codes:
#   0 = success
#   2 = quota / rate limit exceeded
#   3 = hung (no output for N seconds)
#   1 = other error

set -uo pipefail

SILENCE_TIMEOUT=300
OPENCODE_RUN_MODEL="opencode/minimax-m2.5-free"

while [[ $# -gt 0 ]]; do
    case $1 in
        --silence-timeout)
            SILENCE_TIMEOUT="$2"
            shift 2
            ;;
        --minimax-free)
            OPENCODE_RUN_MODEL="opencode/minimax-m2.5-free"
            shift
            ;;
        --nemotron-super-free)
            OPENCODE_RUN_MODEL="opencode/nemotron-3-super-free"
            shift
            ;;
        --gpt5.4)
            OPENCODE_RUN_MODEL="openai/gpt-5.4"
            shift
            ;;
        --gpt5.3-codex)
            OPENCODE_RUN_MODEL="openai/gpt-5.3-codex"
            shift
            ;;
        --model)
            OPENCODE_RUN_MODEL="$2"
            shift 2
            ;;
        --) shift; break ;;
        -*) echo "Unknown option: $1" >&2; exit 1 ;;
        *) break ;;
    esac
done

PROMPT="${1:?Usage: opencode-ask.sh [--silence-timeout N] \"prompt\"}"

# Sub-agent disclaimer prepended to every prompt so OpenCode knows its role.
# This is only injected here (not in project config) so it does not affect
# manual interactive opencode sessions.
DISCLAIMER="[SYSTEM NOTE — read before responding]: You are being invoked as a
sub-agent by a Claude orchestrator via opencode-ask.sh. Rules:
- Do NOT call opencode or invoke yourself recursively.
- Your output goes directly to the orchestrator, not a human user.
- Do not ask clarifying questions — complete the task as best you can.
- Project context is already loaded from CLAUDE.md."

FULL_PROMPT="${DISCLAIMER}

TASK:
${PROMPT}"

TMPOUT=$(mktemp /tmp/opencode-ask-XXXXXX.txt)
trap 'rm -f "$TMPOUT"' EXIT

# Ensure nvm-managed node is on PATH
# shellcheck disable=SC1090
if [[ -f "$HOME/.nvm/nvm.sh" ]]; then
    source "$HOME/.nvm/nvm.sh"
fi

# Launch opencode in background, capture output to temp file
opencode run "$FULL_PROMPT" \
    --model "$OPENCODE_RUN_MODEL" \
    --dangerously-skip-permissions \
    > "$TMPOUT" 2>&1 &
OPENCODE_PID=$!

# Hang detection: two-phase approach to avoid false positives from startup latency.
#
# Phase 1 (STARTUP_GRACE seconds): skip silence counting entirely.
#   Covers API connection latency and model generation startup.
# Phase 2 (after grace): count silence normally. Silence here means OpenCode
#   produced some output then stopped — a genuine hang.
STARTUP_GRACE=60  # seconds before silence counting begins
LAST_SIZE=-1
SILENT_COUNT=0
STATUS=0
ELAPSED=0
HALF_TIMEOUT=$(( SILENCE_TIMEOUT > 1 ? SILENCE_TIMEOUT / 2 : 1 ))
while kill -0 "$OPENCODE_PID" 2>/dev/null; do
    sleep 1
    ELAPSED=$((ELAPSED + 1))
    CURRENT_SIZE=$(stat -c%s "$TMPOUT" 2>/dev/null || echo 0)

    # Phase 1: grace period — keep LAST_SIZE current so counter starts fresh
    if [[ "$ELAPSED" -lt "$STARTUP_GRACE" ]]; then
        LAST_SIZE=$CURRENT_SIZE
        continue
    fi

    # Phase 2: silence detection
    if [[ "$CURRENT_SIZE" -eq "$LAST_SIZE" && "$CURRENT_SIZE" -gt 0 ]]; then
        SILENT_COUNT=$((SILENT_COUNT + 1))
        if [[ "$SILENT_COUNT" -eq "$HALF_TIMEOUT" ]]; then
            echo "[opencode-ask] warning: no new output for ${SILENT_COUNT}s (timeout at ${SILENCE_TIMEOUT}s)" >&2
        fi
        if [[ "$SILENT_COUNT" -ge "$SILENCE_TIMEOUT" ]]; then
            echo "[opencode-ask] killing OpenCode after ${SILENT_COUNT}s of silence" >&2
            kill "$OPENCODE_PID" 2>/dev/null
            STATUS=3
            break
        fi
    else
        SILENT_COUNT=0
    fi
    LAST_SIZE=$CURRENT_SIZE
done

wait "$OPENCODE_PID" 2>/dev/null; OPENCODE_EXIT=$?

# Check for quota / rate limit (exit code 2 from opencode, or keywords in output)
if [[ "$OPENCODE_EXIT" -eq 2 ]]; then
    STATUS=2
elif grep -qiE "quota|rate.?limit|resource.?exhausted|too.?many.?requests" "$TMPOUT" 2>/dev/null; then
    STATUS=2
elif [[ "$STATUS" -eq 0 && "$OPENCODE_EXIT" -ne 0 ]]; then
    STATUS=1
fi

# Strip ANSI escape codes so the orchestrator receives plain text
RESPONSE_TEXT=$(sed 's/\x1b\[[0-9;]*[mK]//g' "$TMPOUT")

# Emit structured summary for the orchestrator
case $STATUS in
    0)
        echo "STATUS: success"
        echo "MODEL: ${OPENCODE_RUN_MODEL}"
        echo "---"
        echo "$RESPONSE_TEXT"
        ;;
    2)
        echo "STATUS: quota_exceeded"
        echo "OpenCode hit a quota/rate limit. Partial output below (may be empty)."
        echo "---"
        echo "$RESPONSE_TEXT"
        ;;
    1)
        echo "STATUS: error (opencode exited $OPENCODE_EXIT)"
        echo "---"
        echo "$RESPONSE_TEXT"
        ;;
    3)
        echo "STATUS: hung (no output for ${SILENCE_TIMEOUT}s, process killed)"
        echo "Partial output collected so far:"
        echo "---"
        echo "$RESPONSE_TEXT"
        ;;
esac

exit $STATUS
