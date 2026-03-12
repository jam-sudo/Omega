---
name: implement-from-plan
description: Use ONLY when ai/plan.md exists and contains a scoped implementation task for this repository. Read AGENTS.md and ai/plan.md first, modify only allowed files, run listed verification when possible, and write ai/codex_report.md before finishing.
---

# Implement From Plan

## Before editing any code:

1. Read `AGENTS.md` for project context and operating rules.
2. Read `ai/plan.md` for the scoped task.
3. Internally restate the goal in one sentence before proceeding.

## While implementing:

4. Modify ONLY files listed in the "Allowed Files" section of `ai/plan.md`.
5. Do NOT make unrelated refactors, add unrelated features, or touch files outside scope.
6. Do NOT modify `.env`, secrets, credentials, or CI configuration unless explicitly allowed.
7. Keep changes minimal — prefer the smallest diff that meets the completion criteria.
8. Follow existing code style (Python: ruff format, 100 char lines, double quotes).

## After implementing:

9. Run the verification commands listed in `ai/plan.md` when feasible.
10. Write `ai/codex_report.md` with these exact sections:
    - Summary (2-3 sentences)
    - Files Changed (every file touched)
    - Verification Commands Run (commands + exit codes)
    - Results (what worked, what didn't)
    - Risks (potential regressions)
    - Open Questions (anything unclear)
11. Stop. Do not widen scope or start additional tasks.
