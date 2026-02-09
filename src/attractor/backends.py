from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from coding_agent import (
    AnthropicProfile,
    GeminiProfile,
    LocalExecutionEnvironment,
    OpenAIProfile,
    Session,
    SessionConfig,
)
from coding_agent.events import EventKind, SessionEvent
from coding_agent.turns import AssistantTurn
from unified_llm.client import Client

from .context import Context
from .handlers import CodergenBackend
from .model import Node
from .outcome import Outcome, StageStatus


@dataclass
class CodingAgentBackend(CodergenBackend):
    working_dir: Path
    default_provider: str = "openai"
    default_model: str = "gpt-5.2-codex"
    live_logging: bool = False
    event_callback: Callable[[SessionEvent], None] | None = None
    sessions: dict[str, Session] = field(default_factory=dict)
    session_meta: dict[str, tuple[str, str]] = field(default_factory=dict)
    _client: Client | None = None

    def run(self, node: Node, prompt: str, context: Context) -> str:
        provider = str(node.attrs.get("llm_provider") or "").strip()
        model = str(node.attrs.get("llm_model") or self.default_model).strip()
        reasoning_effort = str(node.attrs.get("reasoning_effort") or "").strip() or None
        fidelity = str(node.attrs.get("resolved_fidelity") or node.attrs.get("fidelity") or "")
        thread_id = str(node.attrs.get("resolved_thread_id") or node.attrs.get("thread_id") or "")

        if not provider:
            provider = _infer_provider(model) or self.default_provider

        profile = _build_profile(provider, model)
        config = SessionConfig()
        config.reasoning_effort = reasoning_effort
        config.use_streaming = self.live_logging

        env = LocalExecutionEnvironment(working_dir=str(self.working_dir))
        client = self._client or Client.from_env()
        self._client = client
        if not client.providers:
            return Outcome(
                status=StageStatus.FAIL,
                failure_reason="No LLM providers configured. Set OPENAI_API_KEY, ANTHROPIC_API_KEY, or GEMINI_API_KEY.",
            )
        if provider and provider not in client.providers:
            return Outcome(
                status=StageStatus.FAIL,
                failure_reason=f"LLM provider '{provider}' not configured. Available: {', '.join(client.providers.keys())}.",
            )

        session: Session | None = None
        if fidelity == "full" and thread_id:
            session = self.sessions.get(thread_id)
            meta = self.session_meta.get(thread_id)
            if session is None or meta != (provider, model):
                session = None
        if session is None:
            session = Session(provider_profile=profile, execution_env=env, llm_client=client, config=config)
            if fidelity == "full" and thread_id:
                self.sessions[thread_id] = session
                self.session_meta[thread_id] = (provider, model)

        if self.live_logging:
            callback = self.event_callback or _default_event_printer
            session.event_emitter.subscribe(callback)

        _run_coroutine(session.submit(prompt))

        last_text = ""
        for turn in reversed(session.history):
            if isinstance(turn, AssistantTurn):
                last_text = turn.content
                break
        return last_text


def _build_profile(provider: str, model: str):
    if provider == "anthropic":
        return AnthropicProfile(model=model)
    if provider == "gemini":
        return GeminiProfile(model=model)
    return OpenAIProfile(model=model)


def _infer_provider(model: str) -> str | None:
    lowered = model.lower()
    if "claude" in lowered:
        return "anthropic"
    if "gemini" in lowered:
        return "gemini"
    if "gpt" in lowered or lowered.startswith("o"):
        return "openai"
    return None


def _run_coroutine(coro):
    try:
        _ = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result: dict[str, object] = {}

    def _runner():
        result["value"] = asyncio.run(coro)

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    thread.join()
    return result.get("value")


def _start_event_consumer(session: Session, callback: Callable[[SessionEvent], None]) -> threading.Thread:
    def _run() -> None:
        async def _consume() -> None:
            async for event in session.event_emitter.events():
                callback(event)
                if event.kind == EventKind.SESSION_END:
                    break

        asyncio.run(_consume())

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return thread


_SEEN_DELTAS: set[str] = set()


def _truncate_text(text: str, limit: int = 2000) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    return text[:half] + "\n... (truncated) ...\n" + text[-half:]


def _default_event_printer(event: SessionEvent) -> None:
    kind = event.kind
    if kind == EventKind.ASSISTANT_TEXT_DELTA:
        delta = event.data.get("delta", "")
        _SEEN_DELTAS.add(event.session_id)
        print(delta, end="", flush=True)
        return
    if kind == EventKind.ASSISTANT_TEXT_END:
        text = event.data.get("text", "")
        if text and event.session_id not in _SEEN_DELTAS:
            print(text)
        else:
            print()
        reasoning = event.data.get("reasoning", "")
        if reasoning:
            print("[reasoning]")
            print(reasoning)
        return
    if kind == EventKind.TOOL_CALL_START:
        name = event.data.get("tool_name", "tool")
        print(f"[tool] start {name}")
        args = event.data.get("args")
        if args is None:
            args = event.data.get("args_raw")
        if args is not None:
            try:
                args_text = json.dumps(args, indent=2)
            except TypeError:
                args_text = str(args)
            print(f"[tool] args:\n{args_text}")
        return
    if kind == EventKind.TOOL_CALL_END:
        name = event.data.get("tool_name", "tool")
        error = event.data.get("error")
        if error:
            print(f"[tool] end {name} error={error}")
        else:
            print(f"[tool] end {name}")
        output = event.data.get("output")
        if output:
            print(f"[tool] output:\n{_truncate_text(str(output))}")
        return
    if kind == EventKind.SESSION_END:
        _SEEN_DELTAS.discard(event.session_id)
        return
    if kind == EventKind.ERROR:
        print(f"[agent] error: {event.data.get('message', '')}")
        return
    if kind == EventKind.WARNING:
        print(f"[agent] warning: {event.data.get('message', '')}")
        return
