# Team Blockers — Escalation to Orchestrator

> When stuck, team members write blockers here.
> Orchestrator resolves and updates status.

---

<!-- Entries below. Format:
## Blocker: [Title]
- Member: [role]
- Impact: [what's blocked]
- Question: [what needs answering]
- Status: OPEN / RESOLVED
- Resolution: [how it was resolved]
-->

## Blocker: PK-DB Authentication Required for C(t) Data
- Member: data-engineer
- Impact: Cannot download actual concentration-time curves from PK-DB. 4,700+ timecourse records exist for our 49 queried drugs (including all 25 benchmark drugs) but `outputs` and `timecourses` API endpoints return 0 records without authentication. This blocks the full clinical data pipeline for L3 training.
- Question: Can someone register for a PK-DB account at pk-db.com and obtain an API token or session credentials? Account registration appears to require human sign-up at the website.
- Status: OPEN
