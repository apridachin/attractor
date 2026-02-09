from __future__ import annotations

import json
import random
import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DURATION_RE = re.compile(r"^(?P<value>-?\d+)(?P<unit>ms|s|m|h|d)$")


@dataclass
class Duration:
    raw: str
    milliseconds: int


def parse_duration(raw: str) -> Duration | None:
    if not raw:
        return None
    match = DURATION_RE.match(raw.strip())
    if not match:
        return None
    value = int(match.group("value"))
    unit = match.group("unit")
    multiplier = {
        "ms": 1,
        "s": 1000,
        "m": 60 * 1000,
        "h": 60 * 60 * 1000,
        "d": 24 * 60 * 60 * 1000,
    }[unit]
    return Duration(raw=raw, milliseconds=value * multiplier)


def now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def normalize_label(label: str) -> str:
    value = label.strip().lower()
    if not value:
        return value
    patterns = [
        r"^\[[A-Za-z0-9]\]\s*",
        r"^[A-Za-z0-9]\)\s*",
        r"^[A-Za-z0-9]\s*-\s*",
    ]
    for pattern in patterns:
        value = re.sub(pattern, "", value)
    return value.strip()


def derive_class(label: str) -> str:
    lowered = label.strip().lower()
    lowered = lowered.replace(" ", "-")
    lowered = re.sub(r"[^a-z0-9-]", "", lowered)
    return lowered


def jitter_delay(delay_ms: float) -> float:
    return delay_ms * random.uniform(0.5, 1.5)


def chunked(iterable: Iterable[Any], size: int) -> list[list[Any]]:
    batch: list[Any] = []
    batches: list[list[Any]] = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= size:
            batches.append(batch)
            batch = []
    if batch:
        batches.append(batch)
    return batches
