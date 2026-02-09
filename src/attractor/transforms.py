from __future__ import annotations

from dataclasses import dataclass

from .model import Graph
from .stylesheet import StylesheetError, apply_stylesheet, parse_stylesheet


class Transform:
    def apply(self, graph: Graph) -> Graph:  # noqa: D401
        """Apply a transform to the graph."""
        return graph


@dataclass
class VariableExpansionTransform(Transform):
    def apply(self, graph: Graph) -> Graph:
        goal = graph.goal
        for node in graph.nodes.values():
            prompt = node.attrs.get("prompt")
            if isinstance(prompt, str) and "$goal" in prompt:
                node.attrs["prompt"] = prompt.replace("$goal", goal)
        return graph


@dataclass
class StylesheetTransform(Transform):
    def apply(self, graph: Graph) -> Graph:
        stylesheet = str(graph.attrs.get("model_stylesheet", ""))
        if not stylesheet:
            return graph
        try:
            rules = parse_stylesheet(stylesheet)
        except StylesheetError as exc:
            graph.attrs["_stylesheet_error"] = str(exc)
            return graph
        apply_stylesheet(rules, list(graph.nodes.values()))
        graph.attrs["_stylesheet_rules"] = rules
        return graph


DEFAULT_TRANSFORMS: list[Transform] = [
    StylesheetTransform(),
    VariableExpansionTransform(),
]
