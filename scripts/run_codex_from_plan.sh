#!/usr/bin/env bash
# Run Codex to implement the task described in ai/plan.md.
# Expects Codex CLI to be installed and authenticated.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

PLAN_FILE="$REPO_ROOT/ai/plan.md"
LOG_DIR="$REPO_ROOT/ai/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
JSONL_LOG="$LOG_DIR/codex_${TIMESTAMP}.jsonl"
STDERR_LOG="$LOG_DIR/codex_${TIMESTAMP}.stderr"
LAST_MSG="$REPO_ROOT/ai/codex_last_message.txt"
STATUS_FILE="$REPO_ROOT/ai/status.md"

mkdir -p "$LOG_DIR"

log() { echo "[$(date +%H:%M:%S)] $*"; }

# Pre-flight checks
if [ ! -f "$PLAN_FILE" ]; then
    log "ERROR: ai/plan.md not found. Write a plan first."
    exit 1
fi

PLAN_GOAL=$(head -5 "$PLAN_FILE" | grep -v '^#' | grep -v '^$' | head -1)
if [ -z "$PLAN_GOAL" ] || echo "$PLAN_GOAL" | grep -q '<!--'; then
    log "WARNING: ai/plan.md appears to be a template (no real goal set)."
    log "Write a concrete plan before running Codex."
    exit 1
fi

log "=== Running Codex from plan ==="
log "Plan goal: $PLAN_GOAL"
log "Logs: $JSONL_LOG"

# Build the prompt
CODEX_PROMPT="You are implementing a scoped task for the Omega PBPK repository.

INSTRUCTIONS:
1. Read AGENTS.md for project context and operating rules.
2. Read ai/plan.md for the specific task, allowed files, and verification commands.
3. If a repo skill 'implement-from-plan' is available, follow its instructions.
4. Edit ONLY files listed in ai/plan.md under 'Allowed Files'.
5. Keep changes minimal — smallest diff that meets completion criteria.
6. Run verification commands listed in ai/plan.md when possible.
7. Write ai/codex_report.md with these sections: Summary, Files Changed, Verification Commands Run, Results, Risks, Open Questions.
8. Stop after the scoped task. Do not widen scope."

# Run Codex
log "Starting Codex exec..."
set +e
codex exec \
    --full-auto \
    "$CODEX_PROMPT" \
    > "$LAST_MSG" 2>"$STDERR_LOG"
CODEX_EXIT=$?
set -e

log "Codex exited with code $CODEX_EXIT"

if [ $CODEX_EXIT -ne 0 ]; then
    log "ERROR: Codex failed. Check $STDERR_LOG"
    cat "$STDERR_LOG" | tail -20
    echo "" >> "$STATUS_FILE"
    echo "## Last Codex Run ($TIMESTAMP)" >> "$STATUS_FILE"
    echo "- Exit code: $CODEX_EXIT" >> "$STATUS_FILE"
    echo "- Stderr: $STDERR_LOG" >> "$STATUS_FILE"
    exit $CODEX_EXIT
fi

# Report success
log "Codex completed successfully."
log "Last message saved to: $LAST_MSG"

if [ -f "$REPO_ROOT/ai/codex_report.md" ]; then
    log "Codex report written: ai/codex_report.md"
else
    log "WARNING: ai/codex_report.md was not written by Codex."
fi

log "=== Done ==="
