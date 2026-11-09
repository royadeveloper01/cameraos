"""SQLite persistence for image and video metadata."""

from dataclasses import asdict, dataclass
import sqlite3
from pathlib import Path


@dataclass(frozen=True)
class Recording:
    time: str
    datapath: str
    size: int
    fps: float | None


class RecordingRepository:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            columns = {
                row[1]
                for row in connection.execute("PRAGMA table_info(recordings)")
            }
            if columns and "datapath" not in columns:
                self._migrate_legacy_table(connection)
            else:
                self._create_table(connection)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.database_path)

    def add(self, recording: Recording) -> None:
        values = asdict(recording)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO recordings (
                    time, datapath, size, fps
                ) VALUES (
                    :time, :datapath, :size, :fps
                )
                """,
                values,
            )

    def get(self, recording_id: int) -> dict[str, object] | None:
        with self._connect() as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                """
                SELECT id, time, datapath, size, fps
                FROM recordings
                WHERE id = ?
                """,
                (recording_id,),
            ).fetchone()
        return dict(row) if row is not None else None

    def list(self, limit: int = 100) -> list[dict[str, object]]:
        with self._connect() as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                """
                SELECT id, time, datapath, size, fps
                FROM recordings
                ORDER BY time DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    @staticmethod
    def _create_table(connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS recordings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                time TEXT NOT NULL,
                datapath TEXT NOT NULL UNIQUE,
                size INTEGER NOT NULL,
                fps REAL
            )
            """
        )

    def _migrate_legacy_table(self, connection: sqlite3.Connection) -> None:
        connection.execute("ALTER TABLE recordings RENAME TO recordings_legacy")
        self._create_table(connection)
        connection.execute(
            """
            INSERT INTO recordings (id, time, datapath, size, fps)
            SELECT id, started_at, filename, 0, fps
            FROM recordings_legacy
            """
        )
        connection.execute("DROP TABLE recordings_legacy")
