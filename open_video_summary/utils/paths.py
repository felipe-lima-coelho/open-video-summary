"""Resolve runtime paths and keep saved video metadata portable."""

from dataclasses import asdict, is_dataclass
from pathlib import Path

from open_video_summary.utils.config import PROJECT_DIR


def project_path(path: str | Path) -> Path:
    """Interpret relative paths from the repository root, regardless of cwd."""
    path = Path(path)
    return path if path.is_absolute() else PROJECT_DIR / path


def portable_path(path: str | Path) -> str:
    """Store paths inside the repository relative to its root."""
    resolved = project_path(path).resolve()
    try:
        return resolved.relative_to(PROJECT_DIR.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def video_paths(data, *, resolve: bool = False):
    """Convert path fields in videos, segments and summary logs recursively."""
    if is_dataclass(data):
        data = asdict(data)
    if isinstance(data, dict):
        return {
            key: (
                (project_path(value).as_posix() if resolve else portable_path(value))
                if key in {"path", "video_path"} and isinstance(value, str) and value
                else video_paths(value, resolve=resolve)
            )
            for key, value in data.items()
        }
    if isinstance(data, (list, tuple, set)):
        return [video_paths(item, resolve=resolve) for item in data]
    return data
