#!/usr/bin/env bash
# Ensure every AGENTS.md has a sibling CLAUDE.md import file.
#
# Policy:
# - repo-root CLAUDE.md may contain Claude-specific instructions, but it must
#   include an `@AGENTS.md` import line
# - child-directory CLAUDE.md files must be the canonical one-line stub
#   containing only `@AGENTS.md`
#
# Claude Code only auto-discovers CLAUDE.md / CLAUDE.local.md — it never walks
# AGENTS.md. Without a sibling CLAUDE.md, Claude silently works without that
# subtree context while Codex loads AGENTS.md normally. This check keeps the two
# tools aligned while still allowing Claude-specific root instructions.
#
# Usage:
#   sync-agent-stubs.sh [--check | --fix]
#     --check  (default) exit 1 if any missing stubs, no file changes
#     --fix    create missing stubs
set -euo pipefail

MODE="${1:---check}"
case "$MODE" in
    --check|--fix) ;;
    *) echo "Usage: $0 [--check | --fix]" >&2; exit 2 ;;
esac

cd "$(git rev-parse --show-toplevel)"

is_canonical_stub() {
    printf '@AGENTS.md\n' | cmp -s - "$1"
}

root_imports_agents() {
    grep -qx '@AGENTS.md' "$1"
}

missing=()
non_stub=()
root_missing_import=()
orphan_stub=()

while IFS= read -r -d '' agents; do
    dir=$(dirname "$agents")
    stub="$dir/CLAUDE.md"
    if [[ ! -f "$stub" ]]; then
        missing+=("$stub")
    elif [[ "$agents" == "AGENTS.md" ]]; then
        if ! root_imports_agents "$stub"; then
            root_missing_import+=("$stub")
        fi
    elif ! is_canonical_stub "$stub"; then
        non_stub+=("$stub")
    fi
done < <(git ls-files -z -- '*AGENTS.md' 'AGENTS.md')

# Detect orphan stubs: canonical CLAUDE.md stub with no sibling AGENTS.md
while IFS= read -r -d '' stub; do
    dir=$(dirname "$stub")
    if is_canonical_stub "$stub" && [[ ! -f "$dir/AGENTS.md" ]]; then
        orphan_stub+=("$stub")
    fi
done < <(git ls-files -z -- '*CLAUDE.md' 'CLAUDE.md')

if [[ "$MODE" == "--fix" ]]; then
    for stub in "${missing[@]}"; do
        printf '@AGENTS.md\n' > "$stub"
        echo "[sync-agent-stubs] created: $stub"
    done
    missing=()
fi

status=0

if (( ${#missing[@]} > 0 )); then
    echo "[sync-agent-stubs] ERROR: missing CLAUDE.md stub for AGENTS.md at:" >&2
    printf '  %s\n' "${missing[@]}" >&2
    echo "  Run: bash container/scripts/sync-agent-stubs.sh --fix" >&2
    status=1
fi

if (( ${#root_missing_import[@]} > 0 )); then
    echo "[sync-agent-stubs] ERROR: repo-root CLAUDE.md must include an '@AGENTS.md' import line:" >&2
    printf '  %s\n' "${root_missing_import[@]}" >&2
    status=1
fi

if (( ${#non_stub[@]} > 0 )); then
    echo "[sync-agent-stubs] ERROR: child CLAUDE.md beside AGENTS.md is not the canonical stub (expected exactly '@AGENTS.md' + newline):" >&2
    printf '  %s\n' "${non_stub[@]}" >&2
    echo "  Child directories must keep CLAUDE.md as the canonical stub so AGENTS.md remains the single source of truth there." >&2
    status=1
fi

if (( ${#orphan_stub[@]} > 0 )); then
    echo "[sync-agent-stubs] WARNING: orphan CLAUDE.md stub (no sibling AGENTS.md) at:" >&2
    printf '  %s\n' "${orphan_stub[@]}" >&2
fi

exit "$status"
