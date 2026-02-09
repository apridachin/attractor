from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .utils import now_iso, read_json, write_json


@dataclass
class Checkpoint:
    timestamp: str
    current_node: str
    completed_nodes: list[str] = field(default_factory=list)
    node_retries: dict[str, int] = field(default_factory=dict)
    context_values: dict[str, Any] = field(default_factory=dict)
    logs: list[str] = field(default_factory=list)

    def save(self, path: Path) -> None:
        data = {
            "timestamp": self.timestamp,
            "current_node": self.current_node,
            "completed_nodes": list(self.completed_nodes),
            "node_retries": dict(self.node_retries),
            "context": self.context_values,
            "logs": list(self.logs),
        }
        write_json(path, data)

    @classmethod
    def load(cls, path: Path) -> Checkpoint:
        data = read_json(path)
        return cls(
            timestamp=str(data.get("timestamp") or ""),
            current_node=str(data.get("current_node") or ""),
            completed_nodes=list(data.get("completed_nodes") or []),
            node_retries=dict(data.get("node_retries") or {}),
            context_values=dict(data.get("context") or {}),
            logs=list(data.get("logs") or []),
        )

    @classmethod
    def fresh(cls, current_node: str, context_values: dict[str, Any], logs: list[str]) -> Checkpoint:
        return cls(
            timestamp=now_iso(),
            current_node=current_node,
            completed_nodes=[],
            node_retries={},
            context_values=context_values,
            logs=logs,
        )
