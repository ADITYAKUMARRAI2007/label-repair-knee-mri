"""Account for still-running training budgets before committing more GPU time."""
from datetime import datetime, timezone

# Covers observed input setup and artifact finalization outside our parent timer.
STARTUP_FINALIZATION_MARGIN_HOURS = .5
TERMINAL = {"COMPLETE", "ERROR", "CANCELLED", "CANCELED"}


def remaining_live_commitment(receipt, status, observed_utc):
    if status in TERMINAL or receipt["status"] == "PUSH_REJECTED":
        return 0.0
    started = datetime.fromisoformat(receipt["observed_utc"])
    now = datetime.fromisoformat(observed_utc)
    if started.tzinfo is None or now.tzinfo is None:
        raise RuntimeError("GPU budget timestamps must identify their UTC offset")
    elapsed = max(0., (now - started).total_seconds() / 3600)
    bound = receipt["job_budget_hours"] + STARTUP_FINALIZATION_MARGIN_HOURS
    if elapsed >= bound:
        raise RuntimeError("a live job exceeded its planned budget; inspect it before launching more work")
    return bound - elapsed


def proposed_job_fits(allocated_free_hours, live_commitments, proposed_hours):
    remaining_live = sum(live_commitments)
    proposed = proposed_hours + STARTUP_FINALIZATION_MARGIN_HOURS
    return allocated_free_hours >= remaining_live + proposed
