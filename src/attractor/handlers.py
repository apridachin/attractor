from __future__ import annotations

import asyncio
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .context import Context
from .interviewer import AnswerValue, Interviewer, Option, Question, QuestionType
from .model import Edge, Graph, Node
from .outcome import Outcome, StageStatus
from .utils import ensure_dir, parse_duration, write_json


class Handler:
    def execute(self, node: Node, context: Context, graph: Graph, logs_root: Path) -> Outcome:
        raise NotImplementedError


class CodergenBackend:
    def run(self, node: Node, prompt: str, context: Context) -> str | Outcome:
        raise NotImplementedError


class HandlerRegistry:
    def __init__(self) -> None:
        self.handlers: dict[str, Handler] = {}
        self.default_handler: Handler | None = None

    def register(self, type_string: str, handler: Handler) -> None:
        self.handlers[type_string] = handler

    def resolve(self, node: Node) -> Handler:
        if node.type and node.type in self.handlers:
            return self.handlers[node.type]
        handler_type = SHAPE_TO_TYPE.get(node.shape, "codergen")
        if handler_type in self.handlers:
            return self.handlers[handler_type]
        if self.default_handler is None:
            raise ValueError("No default handler registered")
        return self.default_handler


SHAPE_TO_TYPE = {
    "Mdiamond": "start",
    "Msquare": "exit",
    "box": "codergen",
    "hexagon": "wait.human",
    "diamond": "conditional",
    "component": "parallel",
    "tripleoctagon": "parallel.fan_in",
    "parallelogram": "tool",
    "house": "stack.manager_loop",
}


class StartHandler(Handler):
    def execute(self, node: Node, context: Context, graph: Graph, logs_root: Path) -> Outcome:
        return Outcome(status=StageStatus.SUCCESS)


class ExitHandler(Handler):
    def execute(self, node: Node, context: Context, graph: Graph, logs_root: Path) -> Outcome:
        return Outcome(status=StageStatus.SUCCESS)


class ConditionalHandler(Handler):
    def execute(self, node: Node, context: Context, graph: Graph, logs_root: Path) -> Outcome:
        return Outcome(status=StageStatus.SUCCESS, notes=f"Conditional node evaluated: {node.id}")


@dataclass
class CodergenHandler(Handler):
    backend: CodergenBackend | None = None

    def execute(self, node: Node, context: Context, graph: Graph, logs_root: Path) -> Outcome:
        prompt = str(node.attrs.get("_resolved_prompt") or node.attrs.get("prompt") or node.label)
        stage_dir = logs_root / node.id
        ensure_dir(stage_dir)
        (stage_dir / "prompt.md").write_text(prompt)

        response_text = ""
        if self.backend is None:
            response_text = f"[Simulated] Response for stage: {node.id}"
        else:
            try:
                result = self.backend.run(node, prompt, context)
                if isinstance(result, Outcome):
                    write_json(stage_dir / "status.json", result.as_json())
                    return result
                response_text = str(result)
            except Exception as exc:  # noqa: BLE001
                return Outcome(status=StageStatus.FAIL, failure_reason=str(exc))

        (stage_dir / "response.md").write_text(response_text)
        outcome = Outcome(
            status=StageStatus.SUCCESS,
            notes=f"Stage completed: {node.id}",
            context_updates={
                "last_stage": node.id,
                "last_response": response_text[:200],
            },
        )
        write_json(stage_dir / "status.json", outcome.as_json())
        return outcome


@dataclass
class WaitForHumanHandler(Handler):
    interviewer: Interviewer

    def execute(self, node: Node, context: Context, graph: Graph, logs_root: Path) -> Outcome:
        edges = graph.outgoing_edges(node.id)
        choices = []
        for edge in edges:
            label = str(edge.attr("label") or edge.to_node)
            key = _parse_accelerator_key(label)
            choices.append({"key": key, "label": label, "to": edge.to_node})
        if not choices:
            return Outcome(
                status=StageStatus.FAIL, failure_reason="No outgoing edges for human gate"
            )

        options = [Option(key=choice["key"], label=choice["label"]) for choice in choices]
        question = Question(
            text=node.label or "Select an option:",
            type=QuestionType.MULTIPLE_CHOICE,
            options=options,
            stage=node.id,
        )
        answer = self.interviewer.ask(question)
        if answer.value == AnswerValue.TIMEOUT.value:
            default_choice = node.attrs.get("human.default_choice")
            if default_choice:
                selected = next(
                    (choice for choice in choices if choice["to"] == default_choice), None
                )
                if selected is None:
                    selected = choices[0]
            else:
                return Outcome(
                    status=StageStatus.RETRY,
                    failure_reason="human gate timeout, no default",
                )
        elif answer.value == AnswerValue.SKIPPED.value:
            return Outcome(status=StageStatus.FAIL, failure_reason="human skipped interaction")
        else:
            selected = _match_choice(answer, choices)

        return Outcome(
            status=StageStatus.SUCCESS,
            suggested_next_ids=[selected["to"]],
            context_updates={
                "human.gate.selected": selected["key"],
                "human.gate.label": selected["label"],
            },
        )


@dataclass
class ParallelHandler(Handler):
    execute_branch: Callable[[str, Context, Graph, Path], Outcome]

    def execute(self, node: Node, context: Context, graph: Graph, logs_root: Path) -> Outcome:
        branches = graph.outgoing_edges(node.id)
        if not branches:
            return Outcome(status=StageStatus.FAIL, failure_reason="No branches to execute")
        join_policy = str(node.attrs.get("join_policy", "wait_all"))
        error_policy = str(node.attrs.get("error_policy", "continue"))
        max_parallel = int(node.attrs.get("max_parallel", 4))
        results: list[dict[str, Any]] = []

        def run_branch(edge: Edge) -> dict[str, Any]:
            branch_context = context.clone()
            branch_dir = logs_root / "parallel" / node.id / edge.to_node
            ensure_dir(branch_dir)
            outcome = self.execute_branch(edge.to_node, branch_context, graph, branch_dir)
            return {"id": edge.to_node, "outcome": outcome, "context": branch_context.snapshot()}

        if max_parallel < 1:
            max_parallel = 1
        loop = asyncio.new_event_loop()
        try:
            old_loop = asyncio.get_event_loop()
        except RuntimeError:
            old_loop = None
        asyncio.set_event_loop(loop)
        try:
            results = loop.run_until_complete(
                _execute_concurrent(branches, run_branch, max_parallel, error_policy)
            )
        finally:
            asyncio.set_event_loop(old_loop)
            loop.close()

        success_count = sum(1 for r in results if r["outcome"].status == StageStatus.SUCCESS)
        fail_count = sum(1 for r in results if r["outcome"].status == StageStatus.FAIL)

        context.set("parallel.results", _serialize_results(results))

        if join_policy == "first_success":
            return Outcome(status=StageStatus.SUCCESS if success_count > 0 else StageStatus.FAIL)
        if join_policy == "k_of_n":
            required = int(node.attrs.get("k", 1))
            status = StageStatus.SUCCESS if success_count >= required else StageStatus.FAIL
            return Outcome(status=status)
        if join_policy == "quorum":
            quorum = float(node.attrs.get("quorum", 0.5))
            status = StageStatus.SUCCESS if success_count / len(results) >= quorum else StageStatus.FAIL
            return Outcome(status=status)
        if join_policy == "wait_all":
            if fail_count == 0:
                return Outcome(status=StageStatus.SUCCESS)
            return Outcome(status=StageStatus.PARTIAL_SUCCESS)
        return Outcome(status=StageStatus.SUCCESS)


@dataclass
class FanInHandler(Handler):
    def execute(self, node: Node, context: Context, graph: Graph, logs_root: Path) -> Outcome:
        results = context.get("parallel.results")
        if not results:
            return Outcome(status=StageStatus.FAIL, failure_reason="No parallel results to evaluate")
        candidates = _normalize_candidates(results)
        if node.attrs.get("prompt"):
            best = _heuristic_select(candidates)
        else:
            best = _heuristic_select(candidates)
        context_updates = {
            "parallel.fan_in.best_id": best["id"],
            "parallel.fan_in.best_outcome": best["outcome"].status.value,
        }
        return Outcome(
            status=StageStatus.SUCCESS,
            context_updates=context_updates,
            notes=f"Selected best candidate: {best['id']}",
        )


class ToolHandler(Handler):
    def execute(self, node: Node, context: Context, graph: Graph, logs_root: Path) -> Outcome:
        command = str(node.attrs.get("tool_command", ""))
        if not command:
            return Outcome(status=StageStatus.FAIL, failure_reason="No tool_command specified")
        timeout = None
        if node.attrs.get("timeout"):
            duration = parse_duration(str(node.attrs.get("timeout")))
            if duration:
                timeout = duration.milliseconds / 1000.0
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if result.returncode != 0:
                return Outcome(
                    status=StageStatus.FAIL,
                    failure_reason=result.stderr.strip() or "tool failed",
                )
            return Outcome(
                status=StageStatus.SUCCESS,
                context_updates={"tool.output": result.stdout},
                notes=f"Tool completed: {command}",
            )
        except Exception as exc:  # noqa: BLE001
            return Outcome(status=StageStatus.FAIL, failure_reason=str(exc))


class ManagerLoopHandler(Handler):
    def execute(self, node: Node, context: Context, graph: Graph, logs_root: Path) -> Outcome:
        return Outcome(status=StageStatus.FAIL, failure_reason="Manager loop not implemented")


def _parse_accelerator_key(label: str) -> str:
    label = label.strip()
    if label.startswith("[") and "]" in label:
        return label[1 : label.index("]")].strip()
    if ")" in label and len(label) >= 2 and label[1] == ")":
        return label[0]
    if "-" in label:
        parts = label.split("-", 1)
        if parts[0].strip():
            return parts[0].strip()[0]
    return label[0] if label else ""


def _match_choice(answer, choices):
    if answer.selected_option:
        for choice in choices:
            if choice["key"].lower() == answer.selected_option.key.lower():
                return choice
    if answer.value:
        for choice in choices:
            if choice["key"].lower() == str(answer.value).lower():
                return choice
            if choice["label"].lower() == str(answer.value).lower():
                return choice
    return choices[0]


async def _execute_concurrent(
    branches: list[Edge],
    run_branch: Callable[[Edge], dict[str, Any]],
    max_parallel: int,
    error_policy: str,
) -> list[dict[str, Any]]:
    loop = asyncio.get_running_loop()
    results: list[dict[str, Any]] = []
    tasks: list[asyncio.Future[dict[str, Any]]] = []

    sem = asyncio.Semaphore(max_parallel)

    async def wrapped(edge: Edge) -> dict[str, Any]:
        async with sem:
            return await asyncio.to_thread(run_branch, edge)

    for edge in branches:
        tasks.append(loop.create_task(wrapped(edge)))

    for task in asyncio.as_completed(tasks):
        result = await task
        results.append(result)
        if error_policy == "fail_fast" and result["outcome"].status == StageStatus.FAIL:
            for other in tasks:
                if not other.done():
                    other.cancel()
            break

    if error_policy == "ignore":
        results = [r for r in results if r["outcome"].status == StageStatus.SUCCESS]
    return results


def _serialize_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    serialized = []
    for item in results:
        serialized.append(
            {
                "id": item["id"],
                "outcome": item["outcome"].status.value,
            }
        )
    return serialized


def _normalize_candidates(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for item in results:
        outcome = item.get("outcome")
        if isinstance(outcome, Outcome):
            normalized.append(item)
            continue
        if isinstance(outcome, str):
            try:
                status = StageStatus(outcome)
            except ValueError:
                status = StageStatus.FAIL
            normalized.append({**item, "outcome": Outcome(status=status)})
            continue
        normalized.append({**item, "outcome": Outcome(status=StageStatus.FAIL)})
    return normalized


def _heuristic_select(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    rank = {
        StageStatus.SUCCESS: 0,
        StageStatus.PARTIAL_SUCCESS: 1,
        StageStatus.RETRY: 2,
        StageStatus.FAIL: 3,
    }
    sorted_candidates = sorted(
        candidates,
        key=lambda c: (
            rank.get(c["outcome"].status, 99),
            c["id"],
        ),
    )
    return sorted_candidates[0]
