---
name: codex-loop
description: Use when the user wants a scoped repo task implemented through the Claude-as-architect / Codex-as-implementer workflow. Good for bug fixes, narrow refactors, test additions, and bounded feature work.
---

# /codex-loop Workflow

When this skill is invoked with a task description:

## Step 1 — Plan

1. Inspect the current repo state and understand the task.
2. Write a narrow, scoped implementation plan into `ai/plan.md`:
   - Clear goal (one sentence)
   - Background context
   - Explicit list of allowed files
   - Forbidden changes
   - Verification commands to run
   - Testable completion criteria
3. Keep the scope small. If the task is too large, break it into parts and handle only the first part.

## Step 2 — Execute Codex

1. Determine the correct execution script for the current OS:
   - Linux/macOS/WSL: `scripts/run_codex_from_plan.sh`
   - Windows: `scripts/run_codex_from_plan.ps1`
2. Run the script using the Bash tool.
3. If the script fails due to auth, stop and tell the user exactly what to run.

## Step 3 — Review

1. Read `ai/codex_report.md` to understand what Codex did.
2. Run `git diff` to see actual changes.
3. Write `ai/claude_review.md` with assessment.
4. Check:
   - Are changes within scope of `ai/plan.md`?
   - Do verification commands pass?
   - Any scientific/business logic errors?
   - Any regression risks?

## Step 4 — Optional Second Pass

If the result is close but incomplete (minor issues only):
1. Update `ai/plan.md` with a narrower follow-up scope.
2. Re-run the Codex script ONE more time.
3. Review again.
4. Do NOT iterate more than twice total.

## Step 5 — Report

Return a concise summary to the user:
- What was planned
- What Codex changed
- Review verdict (APPROVE / NEEDS CHANGES / BLOCKED)
- Any remaining follow-ups

## Rules

- Do NOT use /codex-loop for huge architectural rewrites.
- Do NOT let Codex modify files outside the ai/plan.md scope.
- Prefer the smallest diff that satisfies completion criteria.
- Stop and surface blockers when auth/admin rights are required.
- Preserve logs under `ai/logs/`.
- All plans and reviews are saved for auditability.
