from __future__ import annotations

import builtins
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any


@dataclass
class ArtifactInfo:
    id: str
    name: str
    size_bytes: int
    stored_at: str
    is_file_backed: bool


class ArtifactStore:
    def __init__(self, base_dir: Path | None = None, threshold_bytes: int = 100_000) -> None:
        self._lock = RLock()
        self._artifacts: dict[str, tuple[ArtifactInfo, Any]] = {}
        self.base_dir = base_dir
        self.threshold_bytes = threshold_bytes

    def store(self, artifact_id: str, name: str, data: Any) -> ArtifactInfo:
        raw = json.dumps(data).encode("utf-8") if not isinstance(data, (bytes, bytearray)) else data
        size = len(raw)
        is_file_backed = size > self.threshold_bytes and self.base_dir is not None
        stored_data: Any = data
        if is_file_backed and self.base_dir is not None:
            artifacts_dir = self.base_dir / "artifacts"
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            path = artifacts_dir / f"{artifact_id}.json"
            path.write_bytes(raw)
            stored_data = str(path)
        info = ArtifactInfo(
            id=artifact_id,
            name=name,
            size_bytes=size,
            stored_at=datetime.now(tz=timezone.utc).isoformat(),
            is_file_backed=is_file_backed,
        )
        with self._lock:
            self._artifacts[artifact_id] = (info, stored_data)
        return info

    def retrieve(self, artifact_id: str) -> Any:
        with self._lock:
            if artifact_id not in self._artifacts:
                raise KeyError("Artifact not found")
            info, data = self._artifacts[artifact_id]
        if info.is_file_backed:
            return json.loads(Path(str(data)).read_text())
        return data

    def has(self, artifact_id: str) -> bool:
        with self._lock:
            return artifact_id in self._artifacts

    def list(self) -> builtins.list[ArtifactInfo]:
        with self._lock:
            return [info for info, _ in self._artifacts.values()]

    def remove(self, artifact_id: str) -> None:
        with self._lock:
            if artifact_id in self._artifacts:
                del self._artifacts[artifact_id]

    def clear(self) -> None:
        with self._lock:
            self._artifacts.clear()
