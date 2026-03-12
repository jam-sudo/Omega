# Verify Codex CLI setup: git, node, npm, codex, auth.
# Exit codes: 0=all ok, 1=missing tool, 2=auth issue
$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$LogDir = Join-Path $RepoRoot "ai/logs"
New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
$LogFile = Join-Path $LogDir "verify_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"

function Log($msg) {
    $line = "[$(Get-Date -Format 'HH:mm:ss')] $msg"
    Write-Host $line
    Add-Content -Path $LogFile -Value $line
}

Log "=== Codex Setup Verification ==="
Log "Repo: $RepoRoot"

# Check git
try {
    $gitVer = & git --version 2>&1
    Log "OK: $gitVer"
} catch {
    Log "FAIL: git not found"
    exit 1
}

# Check node
try {
    $nodeVer = & node --version 2>&1
    Log "OK: node $nodeVer"
} catch {
    Log "FAIL: node not found"
    exit 1
}

# Check npm
try {
    $npmVer = & npm --version 2>&1
    Log "OK: npm $npmVer"
} catch {
    Log "FAIL: npm not found"
    exit 1
}

# Check codex
try {
    $codexVer = & codex --version 2>&1
    Log "OK: $codexVer"
} catch {
    Log "FAIL: codex not found. Install: npm i -g @openai/codex@latest"
    exit 1
}

# Auth check
Log "Testing Codex auth (non-destructive)..."
try {
    $authResult = & codex exec --quiet --approval-mode full-auto "echo 'hello'" 2>&1
    Log "OK: Codex auth verified"
    $authStatus = "verified"
} catch {
    Log "WARN: Codex auth test failed"
    Log "Run: codex auth login"
    Log "Or set: `$env:OPENAI_API_KEY = 'sk-...'"
    $authStatus = "PENDING"
}

Log "=== Verification Complete ==="
Log "Auth status: $authStatus"

if ($authStatus -ne "verified") { exit 2 }
exit 0
