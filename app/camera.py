"""OpenCV camera discovery and recording worker."""

from dataclasses import dataclass
from datetime import datetime, timezone
import glob
from pathlib import Path
import threading
from typing import Callable

import cv2

from .config import Settings
from .database import Recording, RecordingRepository


@dataclass
class RecorderStatus:
    running: bool = False
    recording: bool = False
    last_error: str | None = None
    current_filename: str | None = None
    fps: float = 30.0


def _camera_candidates(camera_device: str | None) -> list[str | int]:
    if camera_device:
        if camera_device.isdigit():
            return [int(camera_device)]
        return [camera_device]

    candidates = glob.glob("/dev/v4l/by-id/*-video-index0")
    candidates.extend(sorted(glob.glob("/dev/video*")))
    unique_candidates: list[str | int] = []
    seen_targets: set[str] = set()
    for candidate in candidates:
        target = str(Path(candidate).resolve())
        if target not in seen_targets:
            seen_targets.add(target)
            unique_candidates.append(candidate)
    return unique_candidates or list(range(10))


def _open_camera(camera_device: str | int) -> cv2.VideoCapture:
    if isinstance(camera_device, str) and camera_device.startswith("/dev/"):
        return cv2.VideoCapture(camera_device, cv2.CAP_V4L2)
    return cv2.VideoCapture(camera_device)


def find_external_camera(camera_device: str | None = None) -> str | int:
    candidates = _camera_candidates(camera_device)
    for candidate in candidates:
        capture = _open_camera(candidate)
        try:
            if capture.isOpened():
                ok, frame = capture.read()
                if ok and frame is not None:
                    return candidate
        finally:
            capture.release()

    if camera_device:
        raise RuntimeError(
            f"Unable to open camera device {camera_device!r}. "
            "Check CAMERA_DEVICE and camera permissions."
        )
    raise RuntimeError(
        "No usable camera was found. Connect a camera or set CAMERA_DEVICE "
        "to a device path such as /dev/video0."
    )


class CameraRecorder:
    def __init__(self, settings: Settings, repository: RecordingRepository) -> None:
        self.settings = settings
        self.repository = repository
        self.status = RecorderStatus()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._record_requested = False
        self._capture_requested = False
        self._state_lock = threading.Lock()
        self._shutdown_callback: Callable[[], None] | None = None

    def set_shutdown_callback(self, callback: Callable[[], None]) -> None:
        """Register a callback invoked when the user quits the camera preview."""
        self._shutdown_callback = callback

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self.status.last_error = None
        self._thread = threading.Thread(target=self._run, name="camera-recorder")
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=10)
        self.status.running = False

    def _mouse_callback(
        self, event: int, x: int, y: int, _flags: int, _param: object
    ) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if 20 <= x <= 180 and 80 <= y <= 130:
            with self._state_lock:
                self._capture_requested = True
        elif 200 <= x <= 360 and 80 <= y <= 130:
            with self._state_lock:
                self._record_requested = not self._record_requested

    def _run(self) -> None:
        self.status.running = True
        camera = None
        writer = None
        recording_started = None
        recording_path = None
        width = height = 0
        frame_count = 0
        shutdown_requested = False
        preview_closed = False
        try:
            camera_path = find_external_camera(self.settings.camera_device)
            camera = _open_camera(camera_path)
            if not camera.isOpened():
                raise RuntimeError(f"Unable to open external camera: {camera_path}")

            self.settings.images_dir.mkdir(parents=True, exist_ok=True)
            self.settings.videos_dir.mkdir(parents=True, exist_ok=True)
            if self.settings.show_preview:
                cv2.namedWindow("External Camera")
                cv2.setMouseCallback("External Camera", self._mouse_callback)

            while not self._stop_event.is_set():
                ok, frame = camera.read()
                if not ok or frame is None:
                    raise RuntimeError("Frame read failed from external camera")

                now = datetime.now(timezone.utc)
                with self._state_lock:
                    capture_requested = self._capture_requested
                    self._capture_requested = False
                    record_requested = self._record_requested

                if capture_requested:
                    capture_name = (
                        f"capture_{now.strftime('%Y%m%dT%H%M%S%fZ')}.jpg"
                    )
                    capture_path = self.settings.images_dir / capture_name
                    if not cv2.imwrite(str(capture_path), frame):
                        raise RuntimeError(f"Unable to save captured image: {capture_path}")
                    self.repository.add(
                        Recording(
                            time=now.isoformat(),
                            datapath=str(capture_path),
                            size=capture_path.stat().st_size,
                            fps=None,
                        )
                    )

                if record_requested and writer is None:
                    height, width = frame.shape[:2]
                    recording_started = now
                    filename = f"recording_{now.strftime('%Y%m%dT%H%M%S%fZ')}.mp4"
                    recording_path = self.settings.videos_dir / filename
                    writer = cv2.VideoWriter(
                        str(recording_path),
                        cv2.VideoWriter_fourcc(*self.settings.video_codec),
                        self.settings.camera_fps,
                        (width, height),
                    )
                    if not writer.isOpened():
                        raise RuntimeError(f"Unable to open video writer: {recording_path}")
                    frame_count = 0
                    self.status.current_filename = filename
                    self.status.recording = True

                if writer is not None:
                    writer.write(frame)
                    frame_count += 1
                    self.status.fps = frame_count / max(
                        (now - recording_started).total_seconds(), 0.001
                    )
                if self.settings.show_preview:
                    preview = frame.copy()
                    cv2.putText(
                        preview,
                        f"FPS: {self.status.fps:.1f}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        preview,
                        "CAPTURE",
                        (35, 112),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.rectangle(preview, (20, 80), (180, 130), (0, 255, 0), 2)
                    cv2.rectangle(preview, (200, 80), (360, 130), (0, 0, 255), 2)
                    cv2.putText(
                        preview,
                        "STOP" if record_requested else "RECORD",
                        (220, 112),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.imshow("External Camera", preview)
                    key = cv2.waitKey(1) & 0xFF
                    window_closed = cv2.getWindowProperty(
                        "External Camera", cv2.WND_PROP_VISIBLE
                    ) < 1
                    if key == ord("q") or window_closed:
                        self._stop_event.set()
                        shutdown_requested = True
                        preview_closed = window_closed
                        break
                    if key == ord("c"):
                        with self._state_lock:
                            self._capture_requested = True
                    elif key == ord("r"):
                        with self._state_lock:
                            self._record_requested = True
                    elif key == ord("s"):
                        with self._state_lock:
                            self._record_requested = False
                if writer is not None and (
                    not record_requested
                    or (
                    recording_started
                    and (now - recording_started).total_seconds()
                    >= self.settings.segment_duration_seconds
                    )
                ):
                    writer.release()
                    writer = None
                    self._save_recording(
                        recording_path,
                        recording_started,
                    )
                    recording_started = None
                    recording_path = None
                    self.status.current_filename = None
                    self.status.recording = False
        except Exception as error:
            self.status.last_error = str(error)
        finally:
            if writer is not None:
                writer.release()
                if recording_started and recording_path:
                    self._save_recording(
                        recording_path,
                        recording_started,
                    )
            if camera is not None:
                camera.release()
            if self.settings.show_preview and not preview_closed:
                cv2.destroyAllWindows()
            self.status.running = False
            self.status.recording = False
            if shutdown_requested and self._shutdown_callback:
                self._shutdown_callback()

    def _save_recording(
        self,
        recording_path: Path,
        recording_started: datetime,
    ) -> None:
        self.repository.add(
            Recording(
                time=recording_started.isoformat(),
                datapath=str(recording_path),
                size=recording_path.stat().st_size,
                fps=self.status.fps,
            )
        )
