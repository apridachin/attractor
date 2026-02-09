from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock
from typing import Any


@dataclass
class Context:
    values: dict[str, Any] = field(default_factory=dict)
    logs: list[str] = field(default_factory=list)
    _lock: RLock = field(default_factory=RLock, repr=False)

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self.values[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self.values.get(key, default)

    def get_string(self, key: str, default: str = "") -> str:
        value = self.get(key, default)
        if value is None:
            return default
        return str(value)

    def append_log(self, entry: str) -> None:
        with self._lock:
            self.logs.append(entry)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return dict(self.values)

    def clone(self) -> Context:
        with self._lock:
            return Context(values=dict(self.values), logs=list(self.logs))

    def apply_updates(self, updates: dict[str, Any]) -> None:
        with self._lock:
            for key, value in updates.items():
                self.values[key] = value
