from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum

from .conditions import validate_condition
from .handlers import SHAPE_TO_TYPE
from .model import Graph, Node
from .stylesheet import StylesheetError, parse_stylesheet


class Severity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class Diagnostic:
    rule: str
    severity: Severity
    message: str
    node_id: str | None = None
    edge: tuple[str, str] | None = None
    fix: str | None = None


def validate(graph: Graph, known_types: Iterable[str] | None = None) -> list[Diagnostic]:
    diagnostics: list[Diagnostic] = []
    known_types_set = set(known_types or SHAPE_TO_TYPE.values())

    start_nodes = _find_start_nodes(graph)
    if len(start_nodes) != 1:
        diagnostics.append(
            Diagnostic(
                rule="start_node",
                severity=Severity.ERROR,
                message="Pipeline must have exactly one start node",
            )
        )
    exit_nodes = _find_exit_nodes(graph)
    if len(exit_nodes) != 1:
        diagnostics.append(
            Diagnostic(
                rule="terminal_node",
                severity=Severity.ERROR,
                message="Pipeline must have exactly one exit node",
            )
        )
    if start_nodes:
        start_id = start_nodes[0].id
        if graph.incoming_edges(start_id):
            diagnostics.append(
                Diagnostic(
                    rule="start_no_incoming",
                    severity=Severity.ERROR,
                    message="Start node must have no incoming edges",
                    node_id=start_id,
                )
            )
    if exit_nodes:
        exit_id = exit_nodes[0].id
        if graph.outgoing_edges(exit_id):
            diagnostics.append(
                Diagnostic(
                    rule="exit_no_outgoing",
                    severity=Severity.ERROR,
                    message="Exit node must have no outgoing edges",
                    node_id=exit_id,
                )
            )

    for edge in graph.edges:
        if edge.from_node not in graph.nodes or edge.to_node not in graph.nodes:
            diagnostics.append(
                Diagnostic(
                    rule="edge_target_exists",
                    severity=Severity.ERROR,
                    message=f"Edge references missing node {edge.from_node}->{edge.to_node}",
                    edge=(edge.from_node, edge.to_node),
                )
            )

    if start_nodes:
        reachable = _reachable_nodes(graph, start_nodes[0].id)
        for node in graph.nodes.values():
            if node.id not in reachable:
                diagnostics.append(
                    Diagnostic(
                        rule="reachability",
                        severity=Severity.ERROR,
                        message="Node is unreachable from start",
                        node_id=node.id,
                    )
                )

    for edge in graph.edges:
        condition = str(edge.attr("condition", ""))
        error = validate_condition(condition)
        if error:
            diagnostics.append(
                Diagnostic(
                    rule="condition_syntax",
                    severity=Severity.ERROR,
                    message=error,
                    edge=(edge.from_node, edge.to_node),
                )
            )

    stylesheet = str(graph.attrs.get("model_stylesheet", ""))
    if stylesheet:
        if "_stylesheet_error" in graph.attrs:
            diagnostics.append(
                Diagnostic(
                    rule="stylesheet_syntax",
                    severity=Severity.ERROR,
                    message=str(graph.attrs["_stylesheet_error"]),
                )
            )
        else:
            try:
                parse_stylesheet(stylesheet)
            except StylesheetError as exc:
                diagnostics.append(
                    Diagnostic(
                        rule="stylesheet_syntax",
                        severity=Severity.ERROR,
                        message=str(exc),
                    )
                )

    for node in graph.nodes.values():
        if node.type and node.type not in known_types_set:
            diagnostics.append(
                Diagnostic(
                    rule="type_known",
                    severity=Severity.WARNING,
                    message=f"Unknown handler type: {node.type}",
                    node_id=node.id,
                )
            )

        fidelity = str(node.attr("fidelity", ""))
        if fidelity and fidelity not in _valid_fidelities():
            diagnostics.append(
                Diagnostic(
                    rule="fidelity_valid",
                    severity=Severity.WARNING,
                    message=f"Invalid fidelity: {fidelity}",
                    node_id=node.id,
                )
            )

        retry_target = str(node.attr("retry_target", ""))
        fallback_retry = str(node.attr("fallback_retry_target", ""))
        if retry_target and retry_target not in graph.nodes:
            diagnostics.append(
                Diagnostic(
                    rule="retry_target_exists",
                    severity=Severity.WARNING,
                    message=f"retry_target does not exist: {retry_target}",
                    node_id=node.id,
                )
            )
        if fallback_retry and fallback_retry not in graph.nodes:
            diagnostics.append(
                Diagnostic(
                    rule="retry_target_exists",
                    severity=Severity.WARNING,
                    message=f"fallback_retry_target does not exist: {fallback_retry}",
                    node_id=node.id,
                )
            )

        if bool(node.attr("goal_gate", False)):
            if not retry_target and not fallback_retry and not graph.retry_target:
                diagnostics.append(
                    Diagnostic(
                        rule="goal_gate_has_retry",
                        severity=Severity.WARNING,
                        message="Goal gate node lacks retry_target",
                        node_id=node.id,
                    )
                )

        if _resolves_to_codergen(node):
            if "prompt" not in node.explicit_attrs and "label" not in node.explicit_attrs:
                diagnostics.append(
                    Diagnostic(
                        rule="prompt_on_llm_nodes",
                        severity=Severity.WARNING,
                        message="Codergen node missing prompt or label",
                        node_id=node.id,
                    )
                )

    graph_fidelity = str(graph.default_fidelity or "")
    if graph_fidelity and graph_fidelity not in _valid_fidelities():
        diagnostics.append(
            Diagnostic(
                rule="fidelity_valid",
                severity=Severity.WARNING,
                message=f"Invalid graph default fidelity: {graph_fidelity}",
            )
        )

    for edge in graph.edges:
        fidelity = str(edge.attr("fidelity", ""))
        if fidelity and fidelity not in _valid_fidelities():
            diagnostics.append(
                Diagnostic(
                    rule="fidelity_valid",
                    severity=Severity.WARNING,
                    message=f"Invalid edge fidelity: {fidelity}",
                    edge=(edge.from_node, edge.to_node),
                )
            )

    graph_retry = str(graph.retry_target)
    graph_fallback = str(graph.fallback_retry_target)
    if graph_retry and graph_retry not in graph.nodes:
        diagnostics.append(
            Diagnostic(
                rule="retry_target_exists",
                severity=Severity.WARNING,
                message=f"Graph retry_target does not exist: {graph_retry}",
            )
        )
    if graph_fallback and graph_fallback not in graph.nodes:
        diagnostics.append(
            Diagnostic(
                rule="retry_target_exists",
                severity=Severity.WARNING,
                message=f"Graph fallback_retry_target does not exist: {graph_fallback}",
            )
        )

    return diagnostics


def validate_or_raise(graph: Graph, known_types: Iterable[str] | None = None) -> list[Diagnostic]:
    diagnostics = validate(graph, known_types)
    errors = [d for d in diagnostics if d.severity == Severity.ERROR]
    if errors:
        messages = "\n".join(f"{d.rule}: {d.message}" for d in errors)
        raise ValueError(messages)
    return diagnostics


def _find_start_nodes(graph: Graph) -> list[Node]:
    nodes = [node for node in graph.nodes.values() if node.shape == "Mdiamond"]
    if nodes:
        return nodes
    return [
        node
        for node in graph.nodes.values()
        if node.id.lower() == "start"
    ]


def _find_exit_nodes(graph: Graph) -> list[Node]:
    nodes = [node for node in graph.nodes.values() if node.shape == "Msquare"]
    if nodes:
        return nodes
    return [
        node
        for node in graph.nodes.values()
        if node.id.lower() in {"exit", "end"}
    ]


def _reachable_nodes(graph: Graph, start_id: str) -> set[str]:
    visited = set()
    stack = [start_id]
    while stack:
        node_id = stack.pop()
        if node_id in visited:
            continue
        visited.add(node_id)
        for edge in graph.outgoing_edges(node_id):
            stack.append(edge.to_node)
    return visited


def _valid_fidelities() -> set[str]:
    return {
        "full",
        "truncate",
        "compact",
        "summary:low",
        "summary:medium",
        "summary:high",
    }


def _resolves_to_codergen(node: Node) -> bool:
    if node.type:
        return node.type == "codergen"
    return node.shape == "box" or node.shape == ""
