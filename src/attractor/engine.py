from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .checkpoint import Checkpoint
from .conditions import evaluate_condition
from .context import Context
from .events import EventEmitter
from .handlers import (
    CodergenBackend,
    CodergenHandler,
    ConditionalHandler,
    ExitHandler,
    FanInHandler,
    HandlerRegistry,
    ManagerLoopHandler,
    ParallelHandler,
    StartHandler,
    ToolHandler,
    WaitForHumanHandler,
)
from .interviewer import AutoApproveInterviewer, Interviewer
from .model import Edge, Graph, Node
from .outcome import Outcome, StageStatus
from .transforms import DEFAULT_TRANSFORMS, Transform
from .utils import ensure_dir, jitter_delay, normalize_label, now_iso, write_json
from .validator import validate_or_raise


@dataclass
class BackoffConfig:
    initial_delay_ms: int = 200
    backoff_factor: float = 2.0
    max_delay_ms: int = 60_000
    jitter: bool = True

    def delay_for_attempt(self, attempt: int) -> float:
        delay = self.initial_delay_ms * (self.backoff_factor ** (attempt - 1))
        delay = min(delay, self.max_delay_ms)
        if self.jitter:
            delay = jitter_delay(delay)
        return delay / 1000.0


@dataclass
class RetryPolicy:
    max_attempts: int
    backoff: BackoffConfig = field(default_factory=BackoffConfig)

    def should_retry_exception(self, exc: Exception) -> bool:
        status = getattr(exc, "status_code", None)
        if status in {400, 401, 403}:
            return False
        if status and 500 <= status < 600:
            return True
        message = str(exc).lower()
        if "rate limit" in message or "timeout" in message:
            return True
        return True


@dataclass
class RunConfig:
    logs_root: Path | None = None
    resume: bool = False
    start_at: str | None = None


class PipelineRunner:
    def __init__(
        self,
        backend: CodergenBackend | None = None,
        interviewer: Interviewer | None = None,
        transforms: list[Transform] | None = None,
        on_event: Callable[[object], None] | None = None,
    ) -> None:
        self.backend = backend
        self.interviewer = interviewer or AutoApproveInterviewer()
        self.transforms = transforms or list(DEFAULT_TRANSFORMS)
        self.events = EventEmitter(on_event)
        self.registry = HandlerRegistry()
        self.registry.register("start", StartHandler())
        self.registry.register("exit", ExitHandler())
        self.registry.register("codergen", CodergenHandler(backend=backend))
        self.registry.register("wait.human", WaitForHumanHandler(self.interviewer))
        self.registry.register("conditional", ConditionalHandler())
        self.registry.register("parallel", ParallelHandler(self._execute_subgraph))
        self.registry.register("parallel.fan_in", FanInHandler())
        self.registry.register("tool", ToolHandler())
        self.registry.register("stack.manager_loop", ManagerLoopHandler())
        self.registry.default_handler = self.registry.handlers["codergen"]

    def prepare(self, graph: Graph) -> Graph:
        for transform in self.transforms:
            graph = transform.apply(graph)
        validate_or_raise(graph, known_types=self.registry.handlers.keys())
        return graph

    def run(self, graph: Graph, config: RunConfig | None = None, context: Context | None = None) -> Outcome:
        graph = self.prepare(graph)
        cfg = config or RunConfig()
        ctx = context or Context()
        logs_root = cfg.logs_root or self._default_logs_root(graph)
        ensure_dir(logs_root)
        self._write_manifest(graph, logs_root)
        ctx.set("graph.goal", graph.goal)
        self.events.emit("PipelineStarted", name=graph.id, goal=graph.goal)

        completed_nodes: list[str] = []
        node_outcomes: dict[str, Outcome] = {}
        retry_counts: dict[str, int] = {}

        degrade_full_fidelity_once = False
        if cfg.resume:
            checkpoint_path = logs_root / "checkpoint.json"
            if checkpoint_path.exists():
                checkpoint = Checkpoint.load(checkpoint_path)
                ctx.values = dict(checkpoint.context_values)
                ctx.logs = list(checkpoint.logs)
                completed_nodes = list(checkpoint.completed_nodes)
                retry_counts = dict(checkpoint.node_retries)
                last_node_id = checkpoint.current_node
                last_node = graph.node(last_node_id)
                if last_node is None:
                    current_node_id = self._find_start_node(graph).id
                else:
                    last_status = ctx.get("outcome", StageStatus.SUCCESS.value)
                    try:
                        status = StageStatus(str(last_status))
                    except ValueError:
                        status = StageStatus.SUCCESS
                    last_outcome = Outcome(
                        status=status,
                        preferred_label=ctx.get_string("preferred_label"),
                        suggested_next_ids=list(ctx.get("suggested_next_ids", [])),
                    )
                    next_edge = self._select_edge(last_node, last_outcome, ctx, graph)
                    current_node_id = next_edge.to_node if next_edge else last_node_id
                    last_fidelity = ctx.get("internal.last_fidelity")
                    if str(last_fidelity) == "full":
                        degrade_full_fidelity_once = True
            else:
                current_node_id = self._find_start_node(graph).id
        else:
            current_node_id = self._find_start_node(graph).id

        if cfg.start_at:
            current_node_id = cfg.start_at

        incoming_edge: Edge | None = None
        previous_node_id: str | None = None

        while True:
            node = graph.node(current_node_id)
            if node is None:
                raise ValueError(f"Unknown node: {current_node_id}")

            ctx.set("current_node", node.id)
            if self._is_terminal(node):
                gate_ok, failed_gate = self._check_goal_gates(graph, node_outcomes)
                if not gate_ok and failed_gate:
                    retry_target = self._resolve_retry_target(failed_gate, graph)
                    if retry_target:
                        current_node_id = retry_target
                        incoming_edge = None
                        continue
                    raise RuntimeError("Goal gate unsatisfied and no retry target")
                break

            resolved_fidelity, resolved_thread_id = self._resolve_fidelity(
                node, incoming_edge, graph, previous_node_id
            )
            if degrade_full_fidelity_once and resolved_fidelity == "full":
                resolved_fidelity = "summary:high"
                degrade_full_fidelity_once = False
            node.attrs["resolved_fidelity"] = resolved_fidelity
            if resolved_thread_id:
                node.attrs["resolved_thread_id"] = resolved_thread_id
            ctx.set("internal.last_fidelity", resolved_fidelity)

            resolved_prompt = self._build_prompt(node, ctx, graph, completed_nodes)
            node.attrs["_resolved_prompt"] = resolved_prompt

            retry_policy = self._build_retry_policy(node, graph)
            self.events.emit("StageStarted", name=node.id, index=len(completed_nodes) + 1)
            outcome = self._execute_with_retry(node, ctx, graph, logs_root, retry_policy, retry_counts)
            if outcome.status == StageStatus.FAIL:
                self.events.emit(
                    "StageFailed",
                    name=node.id,
                    index=len(completed_nodes) + 1,
                    error=outcome.failure_reason,
                )
            else:
                self.events.emit(
                    "StageCompleted",
                    name=node.id,
                    index=len(completed_nodes) + 1,
                    status=outcome.status.value,
                )

            completed_nodes.append(node.id)
            node_outcomes[node.id] = outcome

            for key, value in outcome.context_updates.items():
                ctx.set(key, value)
            ctx.set("outcome", outcome.status.value)
            ctx.set("preferred_label", outcome.preferred_label)
            if outcome.suggested_next_ids:
                ctx.set("suggested_next_ids", list(outcome.suggested_next_ids))

            checkpoint = Checkpoint(
                timestamp=now_iso(),
                current_node=node.id,
                completed_nodes=completed_nodes,
                node_retries=retry_counts,
                context_values=ctx.snapshot(),
                logs=list(ctx.logs),
            )
            checkpoint.save(logs_root / "checkpoint.json")
            self.events.emit("CheckpointSaved", node_id=node.id)

            next_edge = self._select_edge(node, outcome, ctx, graph)
            if next_edge is None:
                if outcome.status == StageStatus.FAIL:
                    target = self._resolve_retry_target(node, graph)
                    if target:
                        current_node_id = target
                        incoming_edge = None
                        continue
                    raise RuntimeError("Stage failed with no outgoing fail edge")
                break

            if bool(next_edge.attr("loop_restart", False)):
                self.run(graph, RunConfig(logs_root=None, resume=False, start_at=next_edge.to_node), ctx)
                return outcome

            previous_node_id = node.id
            current_node_id = next_edge.to_node
            incoming_edge = next_edge

        result = node_outcomes.get(current_node_id, Outcome(status=StageStatus.SUCCESS))
        if result.status == StageStatus.SUCCESS:
            self.events.emit("PipelineCompleted", duration=None, artifact_count=None)
        else:
            self.events.emit("PipelineFailed", error=result.failure_reason, duration=None)
        return result

    def _execute_subgraph(self, start_node_id: str, context: Context, graph: Graph, logs_root: Path) -> Outcome:
        runner = PipelineRunner(backend=self.backend, interviewer=self.interviewer, transforms=self.transforms)
        runner.registry = self.registry
        return runner.run(graph, RunConfig(logs_root=logs_root, start_at=start_node_id), context=context)

    def _write_manifest(self, graph: Graph, logs_root: Path) -> None:
        manifest = {
            "graph": graph.id,
            "goal": graph.goal,
            "started_at": now_iso(),
        }
        write_json(logs_root / "manifest.json", manifest)

    def _default_logs_root(self, graph: Graph) -> Path:
        base = Path(os.getcwd()) / "runs"
        ensure_dir(base)
        return base / f"{graph.id}-{int(time.time())}"

    def _find_start_node(self, graph: Graph) -> Node:
        candidates = [node for node in graph.nodes.values() if node.shape == "Mdiamond"]
        if candidates:
            return candidates[0]
        for node in graph.nodes.values():
            if node.id.lower() == "start":
                return node
        raise ValueError("Start node not found")

    def _is_terminal(self, node: Node) -> bool:
        return node.shape == "Msquare" or node.type == "exit"

    def _build_retry_policy(self, node: Node, graph: Graph) -> RetryPolicy:
        max_retries = node.attrs.get("max_retries")
        if max_retries is None or max_retries == "":
            max_retries = graph.default_max_retry
        try:
            retries = int(max_retries)
        except (TypeError, ValueError):
            retries = 0
        return RetryPolicy(max_attempts=max(1, retries + 1))

    def _execute_with_retry(
        self,
        node: Node,
        context: Context,
        graph: Graph,
        logs_root: Path,
        retry_policy: RetryPolicy,
        retry_counts: dict[str, int],
    ) -> Outcome:
        handler = self.registry.resolve(node)
        for attempt in range(1, retry_policy.max_attempts + 1):
            try:
                outcome = handler.execute(node, context, graph, logs_root)
            except Exception as exc:  # noqa: BLE001
                if attempt < retry_policy.max_attempts and retry_policy.should_retry_exception(exc):
                    delay = retry_policy.backoff.delay_for_attempt(attempt)
                    retry_counts[node.id] = retry_counts.get(node.id, 0) + 1
                    context.set(f"internal.retry_count.{node.id}", retry_counts[node.id])
                    self.events.emit(
                        "StageRetrying",
                        name=node.id,
                        attempt=attempt,
                        delay=delay,
                    )
                    time.sleep(delay)
                    continue
                return Outcome(status=StageStatus.FAIL, failure_reason=str(exc))

            if outcome.status in {StageStatus.SUCCESS, StageStatus.PARTIAL_SUCCESS}:
                retry_counts[node.id] = 0
                context.set(f"internal.retry_count.{node.id}", 0)
                self._write_status_file(node, logs_root, outcome)
                return outcome

            if outcome.status in {StageStatus.FAIL, StageStatus.RETRY}:
                if attempt < retry_policy.max_attempts:
                    retry_counts[node.id] = retry_counts.get(node.id, 0) + 1
                    context.set(f"internal.retry_count.{node.id}", retry_counts[node.id])
                    delay = retry_policy.backoff.delay_for_attempt(attempt)
                    self.events.emit(
                        "StageRetrying",
                        name=node.id,
                        attempt=attempt,
                        delay=delay,
                    )
                    time.sleep(delay)
                    continue
                if bool(node.attrs.get("allow_partial", False)):
                    outcome = Outcome(
                        status=StageStatus.PARTIAL_SUCCESS,
                        notes="retries exhausted, partial accepted",
                    )
                    self._write_status_file(node, logs_root, outcome)
                    return outcome
                outcome = Outcome(status=StageStatus.FAIL, failure_reason="max retries exceeded")
                self._write_status_file(node, logs_root, outcome)
                return outcome

        outcome = Outcome(status=StageStatus.FAIL, failure_reason="max retries exceeded")
        self._write_status_file(node, logs_root, outcome)
        return outcome

    def _select_edge(
        self, node: Node, outcome: Outcome, context: Context, graph: Graph
    ) -> Edge | None:
        edges = graph.outgoing_edges(node.id)
        if not edges:
            return None
        if outcome.status == StageStatus.FAIL:
            fail_edges = []
            for edge in edges:
                condition = str(edge.attr("condition", ""))
                if condition and evaluate_condition(condition, outcome, context):
                    fail_edges.append(edge)
            if fail_edges:
                return self._best_by_weight_then_lexical(fail_edges)
            return None
        condition_matched: list[Edge] = []
        for edge in edges:
            condition = str(edge.attr("condition", ""))
            if condition:
                if evaluate_condition(condition, outcome, context):
                    condition_matched.append(edge)
        if condition_matched:
            return self._best_by_weight_then_lexical(condition_matched)

        if outcome.preferred_label:
            target_label = normalize_label(outcome.preferred_label)
            for edge in edges:
                condition = str(edge.attr("condition", ""))
                if condition and not evaluate_condition(condition, outcome, context):
                    continue
                if normalize_label(str(edge.attr("label", ""))) == target_label:
                    return edge

        if outcome.suggested_next_ids:
            for suggested in outcome.suggested_next_ids:
                for edge in edges:
                    condition = str(edge.attr("condition", ""))
                    if condition and not evaluate_condition(condition, outcome, context):
                        continue
                    if edge.to_node == suggested:
                        return edge

        unconditional = [edge for edge in edges if not str(edge.attr("condition", ""))]
        if unconditional:
            return self._best_by_weight_then_lexical(unconditional)
        return self._best_by_weight_then_lexical(edges)

    def _best_by_weight_then_lexical(self, edges: list[Edge]) -> Edge:
        def sort_key(edge: Edge) -> tuple[int, str]:
            try:
                weight = int(edge.attr("weight", 0))
            except (TypeError, ValueError):
                weight = 0
            return (-weight, edge.to_node)

        return sorted(edges, key=sort_key)[0]

    def _check_goal_gates(self, graph: Graph, node_outcomes: dict[str, Outcome]) -> tuple[bool, Node | None]:
        for node_id, outcome in node_outcomes.items():
            node = graph.node(node_id)
            if node and bool(node.attr("goal_gate", False)):
                if outcome.status not in {StageStatus.SUCCESS, StageStatus.PARTIAL_SUCCESS}:
                    return False, node
        return True, None

    def _resolve_retry_target(self, node: Node, graph: Graph) -> str | None:
        retry_target = str(node.attr("retry_target", ""))
        if retry_target and retry_target in graph.nodes:
            return retry_target
        fallback = str(node.attr("fallback_retry_target", ""))
        if fallback and fallback in graph.nodes:
            return fallback
        if graph.retry_target and graph.retry_target in graph.nodes:
            return graph.retry_target
        if graph.fallback_retry_target and graph.fallback_retry_target in graph.nodes:
            return graph.fallback_retry_target
        return None

    def _resolve_fidelity(
        self,
        node: Node,
        incoming_edge: Edge | None,
        graph: Graph,
        previous_node_id: str | None,
    ) -> tuple[str, str]:
        fidelity = ""
        if incoming_edge is not None:
            fidelity = str(incoming_edge.attr("fidelity", ""))
        if not fidelity:
            fidelity = str(node.attr("fidelity", ""))
        if not fidelity:
            fidelity = str(graph.default_fidelity or "")
        if not fidelity:
            fidelity = "compact"

        thread_id = ""
        if fidelity == "full":
            thread_id = str(node.attr("thread_id", ""))
            if not thread_id and incoming_edge is not None:
                thread_id = str(incoming_edge.attr("thread_id", ""))
            if not thread_id:
                thread_id = str(graph.attrs.get("thread_id", ""))
            if not thread_id:
                classes = node.classes()
                if classes:
                    thread_id = classes[0]
            if not thread_id and previous_node_id:
                thread_id = previous_node_id
        return fidelity, thread_id

    def _build_prompt(
        self, node: Node, context: Context, graph: Graph, completed_nodes: list[str]
    ) -> str:
        prompt = str(node.attrs.get("prompt") or node.label)
        fidelity = str(node.attrs.get("resolved_fidelity") or node.attrs.get("fidelity") or "compact")
        if fidelity == "full":
            return prompt
        preamble = self._preamble(fidelity, context, graph, completed_nodes)
        if not preamble:
            return prompt
        return f"{preamble}\n\n{prompt}"

    def _preamble(self, fidelity: str, context: Context, graph: Graph, completed_nodes: list[str]) -> str:
        goal = graph.goal
        summary_lines = []
        if goal:
            summary_lines.append(f"Goal: {goal}")
        summary_lines.append(f"Completed stages: {', '.join(completed_nodes) if completed_nodes else 'none'}")
        if fidelity in {"summary:medium", "summary:high", "compact"}:
            context_items = {k: v for k, v in context.snapshot().items() if not str(k).startswith("internal.")}
            if context_items:
                summary_lines.append("Context snapshot:")
                for key, value in list(context_items.items())[:10]:
                    summary_lines.append(f"- {key}: {value}")
        if fidelity == "summary:high":
            summary_lines.append(f"Context keys: {', '.join(context.snapshot().keys())}")
        if fidelity == "truncate":
            summary_lines = summary_lines[:1]
        return "\n".join(summary_lines)

    def _write_status_file(self, node: Node, logs_root: Path, outcome: Outcome) -> None:
        stage_dir = logs_root / node.id
        ensure_dir(stage_dir)
        write_json(stage_dir / "status.json", outcome.as_json())
