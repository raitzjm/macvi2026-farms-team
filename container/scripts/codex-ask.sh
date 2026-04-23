#!/usr/bin/env bash
# codex-ask.sh — delegate a task to OpenAI Codex CLI as a sub-agent
#
# Usage: codex-ask.sh [--silence-timeout N] [--gpt5|--mini|--codex|--gpt52] "prompt"
#
# Options:
#   --gpt5              Use gpt-5.4 (default)
#   --mini              Use gpt-5.4-mini. Prefer for lightweight tasks (short
#                       lookups, summaries, simple Q&A).
#   --codex             Use gpt-5.3-codex. The coding-specialist model; prefer
#                       for code generation, refactoring, and structured edits.
#   --gpt52             Use gpt-5.2. Legacy fallback.
#   --silence-timeout N Seconds of silence before declaring a hang (default 300)
#
# Exit codes:
#   0 = success
#   2 = quota / rate limit exceeded
#   3 = hung (no output for N seconds)
#   1 = other error

set -uo pipefail

SILENCE_TIMEOUT=300
CODEX_MODEL="gpt-5.4"

while [[ $# -gt 0 ]]; do
    case $1 in
        --silence-timeout)
            SILENCE_TIMEOUT="$2"
            shift 2
            ;;
        --gpt5)
            CODEX_MODEL="gpt-5.4"
            shift
            ;;
        --mini)
            CODEX_MODEL="gpt-5.4-mini"
            shift
            ;;
        --codex)
            CODEX_MODEL="gpt-5.3-codex"
            shift
            ;;
        --gpt52)
            CODEX_MODEL="gpt-5.2"
            shift
            ;;
        --) shift; break ;;
        -*) echo "Unknown option: $1" >&2; exit 1 ;;
        *) break ;;
    esac
done

PROMPT="${1:?Usage: codex-ask.sh [--silence-timeout N] \"prompt\"}"

# Sub-agent disclaimer prepended to every prompt so Codex knows its role.
# This is only injected here (not in AGENTS.md) so it does not affect
# manual interactive codex sessions.
DISCLAIMER="[SYSTEM NOTE — read before responding]: You are being invoked as a
sub-agent by a Claude orchestrator via codex-ask.sh. Rules:
- Do NOT invoke codex-ask.sh or call the Codex CLI recursively.
- Your output goes directly to the orchestrator, not a human user.
- Do not ask clarifying questions — complete the task as best you can.
- Project context is already loaded from AGENTS.md."

FULL_PROMPT="${DISCLAIMER}

TASK:
${PROMPT}"

TMPEVENTS=$(mktemp /tmp/codex-ask-XXXXXX.jsonl)
trap 'rm -f "$TMPEVENTS"' EXIT

# Launch codex exec in background, stream JSONL events to temp file.
# Pipe an empty line to close stdin defensively — some environments observed
# codex waiting on stdin after the prompt arg; the harmless
# "Reading additional input from stdin..." notice on stderr is filtered by
# the JSON parsers below.
echo "" | codex exec \
    --dangerously-bypass-approvals-and-sandbox \
    -m "$CODEX_MODEL" \
    --json \
    "$FULL_PROMPT" \
    > "$TMPEVENTS" 2>&1 &
CODEX_PID=$!

# Hang detection: two-phase approach to avoid false positives from buffering.
#
# codex exec output is bursty: it emits a thread.started/turn.started burst
# quickly, then may be silent while the API processes, then emits
# item.completed/turn.completed. Counting silence from the very first output
# causes false positives during the inter-burst gap.
#
# Phase 1 (STARTUP_GRACE seconds): skip silence counting entirely.
#   Covers API connection latency + init events + model generation startup.
# Phase 2 (after grace): count silence normally. Silence here means Codex
#   produced some output (e.g. tool calls) then stopped — a genuine hang.
STARTUP_GRACE=60  # seconds before silence counting begins
LAST_SIZE=-1
SILENT_COUNT=0
STATUS=0
ELAPSED=0
HALF_TIMEOUT=$(( SILENCE_TIMEOUT > 1 ? SILENCE_TIMEOUT / 2 : 1 ))
while kill -0 "$CODEX_PID" 2>/dev/null; do
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
            echo "[codex-ask] warning: no new output for ${SILENT_COUNT}s (timeout at ${SILENCE_TIMEOUT}s)" >&2
        fi
        if [[ "$SILENT_COUNT" -ge "$SILENCE_TIMEOUT" ]]; then
            echo "[codex-ask] killing Codex after ${SILENT_COUNT}s of silence" >&2
            kill "$CODEX_PID" 2>/dev/null
            STATUS=3
            break
        fi
    else
        SILENT_COUNT=0
    fi
    LAST_SIZE=$CURRENT_SIZE
done

wait "$CODEX_PID" 2>/dev/null || true

# Check for quota / rate limit errors — skip if we already classified as hung,
# otherwise a partial event stream containing the word "quota" would clobber
# STATUS=3 with a misleading STATUS=2.
# Scope the search to structured error fields (not full JSON dump) to avoid
# false positives from prompt content that happens to contain these keywords.
if [[ "$STATUS" -ne 3 ]] && python3 - <<'EOF' "$TMPEVENTS"
import sys, json
QUOTA_KEYS = ("rate_limit", "ratelimit", "quota", "resource_exhausted", "429")
with open(sys.argv[1]) as f:
    for line in f:
        try:
            obj = json.loads(line)
            if obj.get("type") == "error":
                # Codex error events carry a `message` or `error` field
                msg = str(obj.get("message", "") or obj.get("error", "")).lower()
                if any(k in msg for k in QUOTA_KEYS):
                    sys.exit(0)
        except Exception:
            pass
sys.exit(1)
EOF
then
    STATUS=2
fi

# Extract assistant response text from item.completed agent_message events.
# Multiple agent_message items are concatenated (Codex may split long responses).
RESPONSE_TEXT=$(python3 - <<'EOF' "$TMPEVENTS"
import sys, json
parts = []
with open(sys.argv[1]) as f:
    for line in f:
        try:
            obj = json.loads(line)
            if (obj.get("type") == "item.completed"
                    and obj.get("item", {}).get("type") == "agent_message"):
                parts.append(obj["item"].get("text", ""))
        except Exception:
            pass
print("\n".join(parts))
EOF
)

# Extract token usage from the LAST turn.completed event. Multi-turn sessions
# emit multiple turn.completed events; we only want one USAGE line in the
# wrapper envelope.
USAGE=$(python3 - <<'EOF' "$TMPEVENTS"
import sys, json
last = None
with open(sys.argv[1]) as f:
    for line in f:
        try:
            obj = json.loads(line)
            if obj.get("type") == "turn.completed":
                last = obj.get("usage", {})
        except Exception:
            pass
if last is not None:
    inp = last.get("input_tokens", "?")
    cached = last.get("cached_input_tokens", "?")
    out = last.get("output_tokens", "?")
    total = (inp if isinstance(inp, int) else 0) + (out if isinstance(out, int) else 0)
    print(f"input={inp} cached={cached} output={out} total={total}")
EOF
)

# Emit structured summary matching gemini-ask.sh envelope format
case $STATUS in
    0)
        echo "STATUS: success"
        echo "USAGE: ${USAGE:-unknown}"
        echo "---"
        echo "$RESPONSE_TEXT"
        ;;
    2)
        echo "STATUS: quota_exceeded"
        echo "Codex hit a quota/rate limit. Partial output below (may be empty)."
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
