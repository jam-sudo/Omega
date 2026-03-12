# Run Codex to implement the task described in ai/plan.md.
$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$PlanFile = Join-Path $RepoRoot "ai/plan.md"
$LogDir = Join-Path $RepoRoot "ai/logs"
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$StderrLog = Join-Path $LogDir "codex_${Timestamp}.stderr"
$LastMsg = Join-Path $RepoRoot "ai/codex_last_message.txt"

New-Item -ItemType Directory -Path $LogDir -Force | Out-Null

function Log($msg) { Write-Host "[$(Get-Date -Format 'HH:mm:ss')] $msg" }

# Pre-flight
if (-not (Test-Path $PlanFile)) {
    Log "ERROR: ai/plan.md not found. Write a plan first."
    exit 1
}

$planContent = Get-Content $PlanFile -Raw
if ($planContent -match '<!--') {
    $firstLine = ($planContent -split "`n" | Where-Object { $_ -notmatch '^#' -and $_ -notmatch '^\s*$' -and $_ -notmatch '<!--' } | Select-Object -First 1)
    if (-not $firstLine) {
        Log "WARNING: ai/plan.md appears to be a template."
        exit 1
    }
}

Log "=== Running Codex from plan ==="

$CodexPrompt = @"
You are implementing a scoped task for the Omega PBPK repository.

INSTRUCTIONS:
1. Read AGENTS.md for project context and operating rules.
2. Read ai/plan.md for the specific task, allowed files, and verification commands.
3. If a repo skill 'implement-from-plan' is available, follow its instructions.
4. Edit ONLY files listed in ai/plan.md under 'Allowed Files'.
5. Keep changes minimal — smallest diff that meets completion criteria.
6. Run verification commands listed in ai/plan.md when possible.
7. Write ai/codex_report.md with these sections: Summary, Files Changed, Verification Commands Run, Results, Risks, Open Questions.
8. Stop after the scoped task. Do not widen scope.
"@

Log "Starting Codex exec..."
try {
    & codex exec `
        --approval-mode auto-edit `
        --quiet `
        $CodexPrompt `
        > $LastMsg 2> $StderrLog
    $codexExit = $LASTEXITCODE
} catch {
    Log "ERROR: Codex failed: $_"
    exit 1
}

if ($codexExit -ne 0) {
    Log "ERROR: Codex exited with code $codexExit"
    Get-Content $StderrLog | Select-Object -Last 20
    exit $codexExit
}

Log "Codex completed successfully."
Log "Last message: $LastMsg"

if (Test-Path (Join-Path $RepoRoot "ai/codex_report.md")) {
    Log "Codex report written: ai/codex_report.md"
} else {
    Log "WARNING: ai/codex_report.md was not written."
}

Log "=== Done ==="
