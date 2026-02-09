from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class StageStatus(str, Enum):
    SUCCESS = "success"
    FAIL = "fail"
    PARTIAL_SUCCESS = "partial_success"
    RETRY = "retry"
    SKIPPED = "skipped"


@dataclass
class Outcome:
    status: StageStatus
    preferred_label: str = ""
    suggested_next_ids: list[str] = field(default_factory=list)
    context_updates: dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    failure_reason: str = ""

    def is_success(self) -> bool:
        return self.status in (StageStatus.SUCCESS, StageStatus.PARTIAL_SUCCESS)

    def as_json(self) -> dict[str, Any]:
        return {
            "outcome": self.status.value,
            "preferred_next_label": self.preferred_label,
            "suggested_next_ids": list(self.suggested_next_ids),
            "context_updates": dict(self.context_updates),
            "notes": self.notes,
            "failure_reason": self.failure_reason,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Outcome:
        status = StageStatus(data.get("outcome", StageStatus.FAIL.value))
        return cls(
            status=status,
            preferred_label=str(data.get("preferred_next_label") or ""),
            suggested_next_ids=list(data.get("suggested_next_ids") or []),
            context_updates=dict(data.get("context_updates") or {}),
            notes=str(data.get("notes") or ""),
            failure_reason=str(data.get("failure_reason") or ""),
        )
