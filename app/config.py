"""Environment-driven application configuration."""

from dataclasses import dataclass
from pathlib import Path
import os


def _positive_float(name: str, default: float) -> float:
    value = float(os.environ.get(name, default))
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


def _boolean(name: str, default: bool) -> bool:
    value = os.environ.get(name, str(default)).strip().lower()
    if value not in {"true", "false"}:
        raise ValueError(f"{name} must be true or false")
    return value == "true"


@dataclass(frozen=True)
class Settings:
    camera_device: str | None
    recordings_dir: Path
    images_dir: Path
    videos_dir: Path
    database_path: Path
    segment_duration_seconds: float
    camera_fps: float
    video_codec: str
    show_preview: bool

    @classmethod
    def from_environment(cls) -> "Settings":
        recordings_dir = Path(os.environ.get("RECORDINGS_DIR", "recordings"))
        return cls(
            camera_device=os.environ.get("CAMERA_DEVICE") or None,
            recordings_dir=recordings_dir,
            images_dir=recordings_dir / "images",
            videos_dir=recordings_dir / "videos",
            database_path=Path(os.environ.get("DATABASE_PATH", "recordings.db")),
            segment_duration_seconds=_positive_float(
                "SEGMENT_DURATION_SECONDS", 300.0
            ),
            camera_fps=_positive_float("CAMERA_FPS", 30.0),
            video_codec=os.environ.get("VIDEO_CODEC", "mp4v"),
            # Disable this explicitly for headless deployments.
            show_preview=_boolean("SHOW_PREVIEW", True),
        )
