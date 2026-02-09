from __future__ import annotations

import argparse
from pathlib import Path

from .dot_parser import parse_dot
from .engine import PipelineRunner, RunConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an Attractor pipeline")
    parser.add_argument("dotfile", type=Path, help="Path to DOT file")
    parser.add_argument("--logs-root", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--start-at", type=str, default=None)
    args = parser.parse_args()

    source = args.dotfile.read_text()
    graph = parse_dot(source)
    runner = PipelineRunner()
    runner.run(graph, RunConfig(logs_root=args.logs_root, resume=args.resume, start_at=args.start_at))


if __name__ == "__main__":
    main()
