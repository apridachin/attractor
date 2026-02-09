# Attractor

[factory.strongdm.ai](https://factory.strongdm.ai)

DOT-based pipeline runner for AI workflows. See the spec in the task description.

## Quick Start

```bash
uv sync
```

```python
from attractor import CodingAgentBackend, PipelineRunner, parse_dot

DOT = """
digraph Simple {
    graph [goal="Run tests and report"]
    start [shape=Mdiamond]
    exit [shape=Msquare]
    run_tests [label="Run Tests", prompt="Run tests"]
    report [label="Report", prompt="Summarize results"]
    start -> run_tests -> report -> exit
}
"""

graph = parse_dot(DOT)
backend = CodingAgentBackend(working_dir="/path/to/repo")
runner = PipelineRunner(backend=backend)
runner.run(graph)
```
