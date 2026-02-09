# Attractor Demo

This demo runs a simple pipeline with a human gate.

## Run (simulated)

```bash
uv run python demo/run_demo.py
```

## Run with real LLM calls

```bash
export OPENAI_API_KEY=...
uv run python demo/run_demo.py --live
```

You can also set `ANTHROPIC_API_KEY` or `GEMINI_API_KEY`.
Live agent logging is enabled by default; pass `--no-live-log` to disable.

## Logs

By default, logs are written under `./runs/<graph-id>-<timestamp>/`.
You can set a fixed location:

```bash
uv run python demo/run_demo.py --logs-root /tmp/attractor-demo
```

## Output files

The live run writes `hello.py` into `demo/output/` by default.
