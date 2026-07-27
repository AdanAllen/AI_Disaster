# StayReady incident and rollback runbook

## Severity and immediate action

- **SEV-1:** incorrect jurisdiction, misleading safety language, exposed address/coordinates, lost result state, or critical accessibility failure. Stop rollout and roll back immediately.
- **SEV-2:** elevated source failure, summary latency above gate, map-layer failures, or partial translation defect. Hold rollout, visibly degrade the affected source, and assign an owner.
- **SEV-3:** cosmetic or non-critical content issue. Record and schedule without changing safety claims automatically.

## Response

1. Confirm the release ID, environment, first failing time, and affected aggregate result types. Do not copy resident addresses or coordinates into tickets or logs.
2. Disable or mark only the affected data source unavailable. An external GIS outage must not take down the summary.
3. For any SEV-1, restore the previously retained immutable Render release.
4. Verify `/api/health`, `/api/ready`, address confirmation, expired-result behavior, one English journey, and one approved Spanish journey.
5. Record the decision, owner, reviewer, affected source edition, and time. Reopen traffic only after the original rollback trigger is resolved.

## Rollout gates

At 5%, 25%, and 100%, observe for 48 hours: jurisdiction outcomes, p50/p95 location latency, source-unavailable rate, incomplete summaries, JavaScript errors, map failures, state-expiration behavior, and accessibility reports.
