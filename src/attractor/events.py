from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class Event:
    type: str
    payload: dict[str, Any]


class EventEmitter:
    def __init__(self, callback: Callable[[Event], None] | None = None) -> None:
        self._callback = callback

    def emit(self, event_type: str, **payload: Any) -> None:
        if self._callback is None:
            return
        self._callback(Event(type=event_type, payload=payload))
