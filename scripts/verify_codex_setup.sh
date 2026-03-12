#!/usr/bin/env bash
# Verify Codex CLI setup: git, node, npm, codex, auth.
# Exit codes: 0=all ok, 1=missing tool, 2=auth issue, 3=other error
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$REPO_ROOT/ai/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/verify_$(date +%Y%m%d_%H%M%S).log"
STATUS_FILE="$REPO_ROOT/ai/status.md"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG_FILE"; }

log "=== Codex Setup Verification ==="
log "Repo: $REPO_ROOT"

# Check git
if command -v git &>/dev/null; then
    GIT_VER=$(git --version)
    log "OK: $GIT_VER"
else
    log "FAIL: git not found"
    exit 1
fi

# Check node
if command -v node &>/dev/null; then
    NODE_VER=$(node --version)
    log "OK: node $NODE_VER"
else
    log "FAIL: node not found. Install Node.js first."
    exit 1
fi

# Check npm
if command -v npm &>/dev/null; then
    NPM_VER=$(npm --version)
    log "OK: npm $NPM_VER"
else
    log "FAIL: npm not found"
    exit 1
fi

# Check codex
if command -v codex &>/dev/null; then
    CODEX_VER=$(codex --version 2>&1)
    log "OK: $CODEX_VER"
else
    log "FAIL: codex not found. Install with: npm i -g @openai/codex@latest"
    exit 1
fi

# Auth check — try a minimal non-destructive codex call
log "Testing Codex auth (non-destructive)..."
AUTH_LOG="$LOG_DIR/auth_test_$(date +%Y%m%d_%H%M%S).log"
if codex exec --full-auto "echo 'hello'" > "$AUTH_LOG" 2>&1; then
    log "OK: Codex auth verified"
    AUTH_STATUS="verified"
else
    AUTH_EXIT=$?
    log "WARN: Codex auth test returned exit code $AUTH_EXIT"
    log "This may mean OPENAI_API_KEY is not set or login is required."
    log "Run: codex auth login"
    log "Or set: export OPENAI_API_KEY=sk-..."
    AUTH_STATUS="PENDING — run 'codex auth login' or set OPENAI_API_KEY"
    # Don't exit — report the status
fi

log "=== Verification Complete ==="
log "Auth status: $AUTH_STATUS"

# Update status file if it exists
if [ -f "$STATUS_FILE" ]; then
    log "Results logged to $LOG_FILE"
fi

if [ "$AUTH_STATUS" != "verified" ]; then
    exit 2
fi
exit 0
