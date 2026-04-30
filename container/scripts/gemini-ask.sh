#!/usr/bin/env bash
# gemini-ask.sh — delegate a task to Gemini as a sub-agent
#
# Usage: gemini-ask.sh [--silence-timeout N] [--pro|--auto|--flash|--flash-lite] "prompt"
#
# Options:
#   --pro               Use gemini-3.1-pro-preview (explicit; overrides Flash default).
#   --auto              Use the Gemini CLI default model (omits -m flag entirely).
#                       Prefer this for lightweight tasks (short lookups,
#                       summaries, simple Q&A) where Pro is overkill.
#   --flash             Use "gemini-3-flash-preview". Prefer this for log
#                       parsing, metric extraction, and other fast structured
#                       reads where Pro is overkill.
#   --flash-lite       Use "gemini-3.1-flash-lite-preview". Fast, low-cost
#                       model for simple, high-volume tasks.
#   --silence-timeout N Seconds of silence before declaring a hang (default 300)
#
# Exit codes:
#   0 = success
#   2 = quota / rate limit exceeded
#   3 = hung (no output for N seconds)
#   1 = other error

set -uo pipefail

SILENCE_TIMEOUT=300
GEMINI_MODEL="gemini-3-flash-preview"
GEMINI_AUTO=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --silence-timeout)
            SILENCE_TIMEOUT="$2"
            shift 2
            ;;
        --pro)
            GEMINI_MODEL="gemini-3.1-pro-preview"
            shift
            ;;
        --auto)
            GEMINI_AUTO=1
            shift
            ;;
        --flash)
            GEMINI_MODEL="gemini-3-flash-preview"
            shift
            ;;
        --flash-lite)
            GEMINI_MODEL="gemini-3.1-flash-lite-preview"
            shift
            ;;
        --) shift; break ;;
        -*) echo "Unknown option: $1" >&2; exit 1 ;;
        *) break ;;
    esac
done

PROMPT="${1:?Usage: gemini-ask.sh [--silence-timeout N] \"prompt\"}"

# Sub-agent disclaimer prepended to every prompt so Gemini knows its role.
# This is only injected here (not in GEMINI.md) so it does not affect
# manual interactive gemini sessions.
DISCLAIMER="[SYSTEM NOTE — read before responding]: You are being invoked as a
sub-agent by an Orchestrator Agent via gemini-ask.sh. Rules:
- Do NOT call gemini-cli or invoke yourself recursively.
- Your output goes directly to the orchestrator, not a human user.
- Do not ask clarifying questions — complete the task as best you can.
- Project context is already loaded from AGENTS.md."

FULL_PROMPT="${DISCLAIMER}

TASK:
${PROMPT}"

TMPEVENTS=$(mktemp /tmp/gemini-ask-XXXXXX.jsonl)
trap 'rm -f "$TMPEVENTS"' EXIT

# Launch gemini in background, stream-json events to temp file
# --auto omits -m to let the CLI use its own default model
if [[ "$GEMINI_AUTO" -eq 1 ]]; then
    gemini --yolo \
        --output-format stream-json \
        -p "$FULL_PROMPT" \
        > "$TMPEVENTS" 2>&1 &
else
    gemini --yolo \
        -m "$GEMINI_MODEL" \
        --output-format stream-json \
        -p "$FULL_PROMPT" \
        > "$TMPEVENTS" 2>&1 &
fi
GEMINI_PID=$!

# Hang detection: two-phase approach to avoid false positives from buffering.
#
# stream-json output is bursty: gemini emits init/user events in one burst
# (~t=6s), then is silent for several seconds while the API processes the
# request, then emits assistant/result events in a second burst. Counting
# silence from the very first output causes false positives during that
# inter-burst gap.
#
# Phase 1 (STARTUP_GRACE seconds): skip silence counting entirely.
#   Covers API connection latency + init event + model generation startup.
# Phase 2 (after grace): count silence normally. Silence here means Gemini
#   produced some output (e.g. tool calls) then stopped — a genuine hang.
STARTUP_GRACE=60  # seconds before silence counting begins
LAST_SIZE=-1
SILENT_COUNT=0
STATUS=0
ELAPSED=0
HALF_TIMEOUT=$(( SILENCE_TIMEOUT > 1 ? SILENCE_TIMEOUT / 2 : 1 ))
while kill -0 "$GEMINI_PID" 2>/dev/null; do
    sleep 1
    ELAPSED=$((ELAPSED + 1))
    CURRENT_SIZE=$(stat -c%s "$TMPEVENTS" 2>/dev/null || echo 0)

    # Phase 1: grace period — keep LAST_SIZE current so counter starts fresh
    if [[ "$ELAPSED" -lt "$STARTUP_GRACE" ]]; then
        LAST_SIZE=$CURRENT_SIZE
        continue
    fi

    # Phase 2: silence detection
    if [[ "$CURRENT_SIZE" -eq "$LAST_SIZE" && "$CURRENT_SIZE" -gt 0 ]]; then
        SILENT_COUNT=$((SILENT_COUNT + 1))
        if [[ "$SILENT_COUNT" -eq "$HALF_TIMEOUT" ]]; then
            echo "[gemini-ask] warning: no new output for ${SILENT_COUNT}s (timeout at ${SILENCE_TIMEOUT}s)" >&2
        fi
        if [[ "$SILENT_COUNT" -ge "$SILENCE_TIMEOUT" ]]; then
            echo "[gemini-ask] killing Gemini after ${SILENT_COUNT}s of silence" >&2
            kill "$GEMINI_PID" 2>/dev/null
            STATUS=3
            break
        fi
    else
        SILENT_COUNT=0
    fi
    LAST_SIZE=$CURRENT_SIZE
done

wait "$GEMINI_PID" 2>/dev/null || true

# Check for quota / rate limit — only inspect the result event to avoid
# false positives from CLAUDE.md content echoed in the user message event
if python3 - <<'EOF' "$TMPEVENTS"
import sys, json
with open(sys.argv[1]) as f:
    for line in f:
        try:
            obj = json.loads(line)
            if obj.get("type") == "result" and obj.get("status") != "success":
                err = str(obj).lower()
                if any(k in err for k in ("resource_exhausted", "ratelimit", "quota", "exhausted")):
                    sys.exit(0)  # quota detected
        except Exception:
            pass
sys.exit(1)  # no quota error
EOF
then
    STATUS=2
fi

# Extract assistant response text from stream-json events
RESPONSE_TEXT=$(python3 - <<'EOF' "$TMPEVENTS"
import sys, json
parts = []
with open(sys.argv[1]) as f:
    for line in f:
        try:
            obj = json.loads(line)
            if obj.get("type") == "message" and obj.get("role") == "assistant":
                parts.append(obj.get("content", ""))
        except Exception:
            pass
print("".join(parts))
EOF
)

# Extract token usage from result event
USAGE=$(python3 - <<'EOF' "$TMPEVENTS"
import sys, json
with open(sys.argv[1]) as f:
    for line in f:
        try:
            obj = json.loads(line)
            if obj.get("type") == "result":
                s = obj.get("stats", {})
                print(f"input={s.get('input_tokens','?')} output={s.get('output_tokens','?')} total={s.get('total_tokens','?')}")
        except Exception:
            pass
EOF
)

# Emit structured summary for Haiku to relay verbatim to the orchestrator
case $STATUS in
    0)
        echo "STATUS: success"
        echo "USAGE: ${USAGE:-unknown}"
        echo "---"
        echo "$RESPONSE_TEXT"
        ;;
    2)
        echo "STATUS: quota_exceeded"
        echo "Gemini hit a quota/rate limit. Partial output below (may be empty)."
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
