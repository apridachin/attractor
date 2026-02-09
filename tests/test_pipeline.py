from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from attractor.dot_parser import parse_dot
from attractor.engine import PipelineRunner, RunConfig
from attractor.handlers import Handler
from attractor.outcome import Outcome, StageStatus
from attractor.validator import Severity, validate


def test_parse_linear_pipeline():
    dot = """
    digraph Simple {
        graph [goal="Run tests"]
        start [shape=Mdiamond]
        exit [shape=Msquare]
        run_tests [label="Run Tests", prompt="Run"]
        start -> run_tests -> exit
    }
    """
    graph = parse_dot(dot)
    assert graph.goal == "Run tests"
    assert len(graph.nodes) == 3
    assert len(graph.edges) == 2


def test_chained_edges():
    dot = """
    digraph Chain {
        start [shape=Mdiamond]
        exit [shape=Msquare]
        A -> B -> C [label="next"]
    }
    """
    graph = parse_dot(dot)
    assert len(graph.edges) == 2
    assert graph.edges[0].from_node == "A"
    assert graph.edges[0].to_node == "B"
    assert graph.edges[1].from_node == "B"
    assert graph.edges[1].to_node == "C"


def test_validation_missing_start():
    dot = """
    digraph MissingStart {
        exit [shape=Msquare]
    }
    """
    graph = parse_dot(dot)
    diagnostics = validate(graph)
    assert any(d.rule == "start_node" and d.severity == Severity.ERROR for d in diagnostics)


def test_validation_missing_exit():
    dot = """
    digraph MissingExit {
        start [shape=Mdiamond]
    }
    """
    graph = parse_dot(dot)
    diagnostics = validate(graph)
    assert any(d.rule == "terminal_node" and d.severity == Severity.ERROR for d in diagnostics)


def test_execute_linear_pipeline():
    dot = """
    digraph Linear {
        start [shape=Mdiamond]
        exit [shape=Msquare]
        A [label="Do A", prompt="Do A"]
        start -> A -> exit
    }
    """
    graph = parse_dot(dot)
    runner = PipelineRunner()
    with TemporaryDirectory() as tmp:
        outcome = runner.run(graph, RunConfig(logs_root=Path(tmp)))
    assert outcome.status in {StageStatus.SUCCESS, StageStatus.PARTIAL_SUCCESS}


def test_retry_logic():
    dot = """
    digraph Retry {
        start [shape=Mdiamond]
        exit [shape=Msquare]
        work [type="flaky", max_retries=1]
        start -> work -> exit
    }
    """
    graph = parse_dot(dot)

    class FlakyHandler(Handler):
        def __init__(self):
            self.calls = 0

        def execute(self, node, context, graph, logs_root):
            self.calls += 1
            if self.calls == 1:
                return Outcome(status=StageStatus.RETRY)
            return Outcome(status=StageStatus.SUCCESS)

    runner = PipelineRunner()
    runner.registry.register("flaky", FlakyHandler())
    with TemporaryDirectory() as tmp:
        outcome = runner.run(graph, RunConfig(logs_root=Path(tmp)))
    assert outcome.status == StageStatus.SUCCESS


def test_condition_edge_priority():
    dot = """
    digraph Branch {
        start [shape=Mdiamond]
        exit [shape=Msquare]
        A [label="A", prompt="A"]
        B [label="B", prompt="B"]
        C [label="C", prompt="C"]
        start -> A
        A -> B [condition="outcome=success", weight=0]
        A -> C [weight=10]
        B -> exit
        C -> exit
    }
    """
    graph = parse_dot(dot)
    runner = PipelineRunner()
    with TemporaryDirectory() as tmp:
        logs_root = Path(tmp)
        runner.run(graph, RunConfig(logs_root=logs_root))
        checkpoint = (logs_root / "checkpoint.json").read_text()
    assert "\"B\"" in checkpoint
