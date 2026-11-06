"""Disk cache for model outputs: identical requests are never generated twice."""
import hashlib
import json
from pathlib import Path


class DiskCache:
    """Stores one small JSON file per request, named by a hash of its inputs."""

    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, key: dict) -> Path:
        digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()
        return self.directory / f"{digest}.json"

    def get(self, key: dict) -> str | None:
        path = self._path(key)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))["output"]

    def set(self, key: dict, output: str) -> None:
        self._path(key).write_text(
            json.dumps({"key": key, "output": output}, ensure_ascii=False),
            encoding="utf-8",
        )
