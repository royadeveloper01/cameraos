"""FastAPI application for continuous camera recording."""

from contextlib import asynccontextmanager
from pathlib import Path

import cv2
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response

from .camera import CameraRecorder
from .config import Settings
from .database import RecordingRepository


settings = Settings.from_environment()
repository = RecordingRepository(settings.database_path)
recorder = CameraRecorder(settings, repository)


@asynccontextmanager
async def lifespan(_: FastAPI):
    recorder.start()
    yield
    recorder.stop()


app = FastAPI(title="Camera Recording Service", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/status")
def status() -> dict[str, object]:
    return {
        "running": recorder.status.running,
        "recording": recorder.status.recording,
        "last_error": recorder.status.last_error,
        "current_filename": recorder.status.current_filename,
        "fps": recorder.status.fps,
    }


@app.get("/recordings")
def recordings(limit: int = Query(default=100, ge=1, le=1000)) -> list[dict[str, object]]:
    return repository.list(limit)


def _recording_path(recording: dict[str, object]) -> Path:
    path = Path(str(recording["datapath"])).resolve()
    recordings_root = settings.recordings_dir.resolve()
    if path != recordings_root and recordings_root not in path.parents:
        raise HTTPException(status_code=500, detail="Recording path is outside storage")
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Recording file was not found")
    return path


def _get_recording(recording_id: int) -> dict[str, object]:
    recording = repository.get(recording_id)
    if recording is None:
        raise HTTPException(status_code=404, detail="Recording was not found")
    return recording


@app.get("/recordings/{recording_id}")
def recording(recording_id: int) -> dict[str, object]:
    return _get_recording(recording_id)


@app.get("/recordings/{recording_id}/media")
def recording_media(recording_id: int) -> FileResponse:
    recording = _get_recording(recording_id)
    path = _recording_path(recording)
    media_type = "image/jpeg" if path.suffix.lower() in {".jpg", ".jpeg"} else "video/mp4"
    return FileResponse(path, media_type=media_type, filename=path.name)


@app.get("/recordings/{recording_id}/frame")
def recording_frame(
    recording_id: int,
    seconds: float = Query(default=0.0, ge=0.0),
) -> Response:
    recording = _get_recording(recording_id)
    path = _recording_path(recording)
    if path.suffix.lower() not in {".mp4", ".avi", ".mov", ".mkv"}:
        raise HTTPException(status_code=400, detail="Recording is not a video")

    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise HTTPException(status_code=422, detail="Unable to open video")
        duration = capture.get(cv2.CAP_PROP_FRAME_COUNT) / max(
            capture.get(cv2.CAP_PROP_FPS), 1.0
        )
        if seconds > duration:
            raise HTTPException(
                status_code=416,
                detail=f"Requested timestamp exceeds video duration ({duration:.3f}s)",
            )
        capture.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
        ok, frame = capture.read()
    finally:
        capture.release()

    if not ok or frame is None:
        raise HTTPException(status_code=422, detail="Unable to read video frame")
    ok, encoded = cv2.imencode(".jpg", frame)
    if not ok:
        raise HTTPException(status_code=422, detail="Unable to encode video frame")
    return Response(content=encoded.tobytes(), media_type="image/jpeg")
