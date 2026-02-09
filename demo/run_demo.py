from __future__ import annotations

import argparse
from pathlib import Path

from attractor import CodingAgentBackend, PipelineRunner, parse_dot
from attractor.events import Event
from attractor.engine import RunConfig
from attractor.interviewer import ConsoleInterviewer


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Attractor demo pipeline")
    parser.add_argument("--dot", type=Path, default=Path(__file__).with_name("pipeline.dot"))
    parser.add_argument("--logs-root", type=Path, default=None)
    parser.add_argument("--live", action="store_true", help="Use CodingAgentBackend (real LLM calls)")
    parser.add_argument(
        "--no-live-log",
        action="store_false",
        dest="live_log",
        help="Disable live agent logging during execution",
    )
    parser.set_defaults(live_log=True)
    parser.add_argument(
        "--workdir",
        type=Path,
        default=Path(__file__).with_name("output"),
        help="Working directory for the coding agent",
    )
    args = parser.parse_args()

    graph = parse_dot(args.dot.read_text())

    args.workdir.mkdir(parents=True, exist_ok=True)

    backend = None
    if args.live:
        backend = CodingAgentBackend(working_dir=args.workdir, live_logging=args.live_log)

    def _log_event(event: Event) -> None:
        if event.type == "StageStarted":
            print(f"[pipeline] stage start: {event.payload.get('name')}")
        if event.type == "StageCompleted":
            print(f"[pipeline] stage done: {event.payload.get('name')}")
        if event.type == "StageFailed":
            print(f"[pipeline] stage failed: {event.payload.get('name')}")

    runner = PipelineRunner(backend=backend, interviewer=ConsoleInterviewer(), on_event=_log_event)
    runner.run(graph, RunConfig(logs_root=args.logs_root))
    print(f"Demo output directory: {args.workdir}")


if __name__ == "__main__":
    main()
