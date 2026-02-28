from __future__ import annotations

import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path


@dataclass(frozen=True)
class ModelMetadata:
    package_version: str
    git_commit: str | None
    timestamp_utc: str


def _read_git_commit(repo_root: Path) -> str | None:
    if not (repo_root / ".git").exists():
        return None
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = output.strip()
    return commit or None


def get_model_metadata(
    package_name: str = "physio_sim", repo_root: Path | None = None
) -> dict[str, str | None]:
    root = repo_root or Path.cwd()
    try:
        package_version = metadata.version(package_name)
    except metadata.PackageNotFoundError:
        package_version = "0+unknown"

    payload = ModelMetadata(
        package_version=package_version,
        git_commit=_read_git_commit(root),
        timestamp_utc=datetime.now(timezone.utc).isoformat(),  # noqa: UP017
    )
    return asdict(payload)
