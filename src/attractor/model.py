from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Node:
    id: str
    attrs: dict[str, Any] = field(default_factory=dict)
    explicit_attrs: set[str] = field(default_factory=set)

    def attr(self, key: str, default: Any = None) -> Any:
        return self.attrs.get(key, default)

    @property
    def shape(self) -> str:
        return str(self.attrs.get("shape", "box"))

    @property
    def label(self) -> str:
        return str(self.attrs.get("label", self.id))

    @property
    def type(self) -> str:
        return str(self.attrs.get("type", ""))

    def classes(self) -> list[str]:
        raw = str(self.attrs.get("class", ""))
        if not raw:
            return []
        return [c.strip() for c in raw.split(",") if c.strip()]

    def add_class(self, cls: str) -> None:
        classes = self.classes()
        if cls in classes:
            return
        classes.append(cls)
        self.attrs["class"] = ",".join(classes)


@dataclass
class Edge:
    from_node: str
    to_node: str
    attrs: dict[str, Any] = field(default_factory=dict)

    def attr(self, key: str, default: Any = None) -> Any:
        return self.attrs.get(key, default)


@dataclass
class Graph:
    id: str
    attrs: dict[str, Any] = field(default_factory=dict)
    nodes: dict[str, Node] = field(default_factory=dict)
    edges: list[Edge] = field(default_factory=list)

    def node(self, node_id: str) -> Node | None:
        return self.nodes.get(node_id)

    def add_node(self, node: Node) -> None:
        self.nodes[node.id] = node

    def add_edge(self, edge: Edge) -> None:
        self.edges.append(edge)

    def outgoing_edges(self, node_id: str) -> list[Edge]:
        return [e for e in self.edges if e.from_node == node_id]

    def incoming_edges(self, node_id: str) -> list[Edge]:
        return [e for e in self.edges if e.to_node == node_id]

    def iter_nodes(self) -> Iterable[Node]:
        return self.nodes.values()

    @property
    def goal(self) -> str:
        return str(self.attrs.get("goal", ""))

    @property
    def default_max_retry(self) -> int:
        value = self.attrs.get("default_max_retry")
        try:
            return int(value)
        except (TypeError, ValueError):
            return 50

    @property
    def default_fidelity(self) -> str:
        return str(self.attrs.get("default_fidelity", ""))

    @property
    def retry_target(self) -> str:
        return str(self.attrs.get("retry_target", ""))

    @property
    def fallback_retry_target(self) -> str:
        return str(self.attrs.get("fallback_retry_target", ""))
