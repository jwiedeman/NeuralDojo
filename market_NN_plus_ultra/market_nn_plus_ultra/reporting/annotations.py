"""Analyst annotation utilities for Market NN Plus Ultra."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

DECISION_CHOICES: tuple[str, ...] = (
    "approve",
    "escalate",
    "flag",
    "hold",
    "reject",
)


def _normalise_tags(tags: Iterable[str] | None) -> tuple[str, ...]:
    if not tags:
        return ()
    return tuple(sorted({tag.strip() for tag in tags if tag and tag.strip()}))


def _ensure_timezone(value: datetime | pd.Timestamp | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        ts = value
    else:
        ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.to_pydatetime()


@dataclass(slots=True)
class TradeAnnotation:
    """Structured annotation captured by an analyst."""

    trade_id: int
    decision: str
    rationale: str
    author: str
    symbol: str | None = None
    trade_timestamp: datetime | None = None
    confidence: float | None = None
    tags: tuple[str, ...] = field(default_factory=tuple)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Mapping[str, Any] | None = None
    context_window_start: datetime | None = None
    context_window_end: datetime | None = None

    def __post_init__(self) -> None:
        if self.decision not in DECISION_CHOICES:
            raise ValueError(
                f"decision '{self.decision}' is not supported; choose from {', '.join(DECISION_CHOICES)}"
            )
        self.tags = _normalise_tags(self.tags)
        if self.confidence is not None and not (0.0 <= self.confidence <= 1.0):
            raise ValueError("confidence must be in the range [0, 1]")
        self.trade_timestamp = _ensure_timezone(self.trade_timestamp)
        self.context_window_start = _ensure_timezone(self.context_window_start)
        self.context_window_end = _ensure_timezone(self.context_window_end)
        created = _ensure_timezone(self.created_at)
        if created is not None:
            self.created_at = created

    def to_record(self) -> dict[str, Any]:
        """Return a SQLite-ready record dictionary."""

        def _iso(dt: datetime | None) -> str | None:
            if dt is None:
                return None
            ts = pd.Timestamp(dt)
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            else:
                ts = ts.tz_convert("UTC")
            return ts.isoformat()

        payload = {
            "trade_id": int(self.trade_id),
            "decision": self.decision,
            "rationale": self.rationale,
            "author": self.author,
            "symbol": self.symbol,
            "trade_timestamp": _iso(self.trade_timestamp),
            "confidence": float(self.confidence) if self.confidence is not None else None,
            "tags": ",".join(self.tags) if self.tags else None,
            "created_at": _iso(self.created_at),
            "metadata": json.dumps(self.metadata) if self.metadata is not None else None,
            "context_window_start": _iso(self.context_window_start),
            "context_window_end": _iso(self.context_window_end),
        }
        return payload


ANNOTATION_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS trade_annotations (
    annotation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id INTEGER NOT NULL,
    symbol TEXT,
    trade_timestamp DATETIME,
    decision TEXT NOT NULL,
    rationale TEXT NOT NULL,
    confidence REAL,
    tags TEXT,
    author TEXT NOT NULL,
    created_at DATETIME NOT NULL,
    metadata TEXT,
    context_window_start DATETIME,
    context_window_end DATETIME,
    FOREIGN KEY(trade_id) REFERENCES trades(trade_id)
);
"""

ANNOTATION_INDEXES = (
    "CREATE INDEX IF NOT EXISTS idx_trade_annotations_trade_id ON trade_annotations(trade_id)",
    "CREATE INDEX IF NOT EXISTS idx_trade_annotations_created_at ON trade_annotations(created_at)",
)


def ensure_annotation_schema(conn: sqlite3.Connection) -> None:
    """Create the annotation table and indexes if they do not exist."""

    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute(ANNOTATION_TABLE_SQL)
    for statement in ANNOTATION_INDEXES:
        conn.execute(statement)


def insert_trade_annotation(conn: sqlite3.Connection, annotation: TradeAnnotation) -> dict[str, Any]:
    """Insert an annotation and return the stored record."""

    record = annotation.to_record()
    placeholders = ", ".join(f":{key}" for key in record)
    columns = ", ".join(record)
    sql = f"INSERT INTO trade_annotations ({columns}) VALUES ({placeholders})"
    conn.execute(sql, record)
    conn.commit()
    cursor = conn.execute(
        "SELECT * FROM trade_annotations WHERE rowid = last_insert_rowid()"
    )
    row = cursor.fetchone()
    if row is None:
        raise RuntimeError("failed to fetch stored annotation")
    columns = [col[0] for col in cursor.description]
    return dict(zip(columns, row))


def load_trade_annotations(
    conn: sqlite3.Connection,
    *,
    trade_ids: Sequence[int] | None = None,
    symbol: str | None = None,
    decision: str | None = None,
) -> pd.DataFrame:
    """Return annotations filtered by the optional parameters."""

    ensure_annotation_schema(conn)
    query = "SELECT * FROM trade_annotations"
    clauses: list[str] = []
    params: list[Any] = []

    if trade_ids:
        placeholders = ",".join(["?"] * len(trade_ids))
        clauses.append(f"trade_id IN ({placeholders})")
        params.extend(int(tid) for tid in trade_ids)
    if symbol:
        clauses.append("symbol = ?")
        params.append(symbol)
    if decision:
        clauses.append("decision = ?")
        params.append(decision)

    if clauses:
        query = f"{query} WHERE {' AND '.join(clauses)}"

    query += " ORDER BY created_at DESC"
    frame = pd.read_sql_query(query, conn, params=params, parse_dates=[
        "trade_timestamp",
        "created_at",
        "context_window_start",
        "context_window_end",
    ])
    if "tags" in frame.columns:
        frame["tags"] = frame["tags"].fillna("").apply(
            lambda value: tuple(tag for tag in value.split(",") if tag)
        )
    if "metadata" in frame.columns:
        frame["metadata"] = frame["metadata"].apply(
            lambda payload: json.loads(payload) if isinstance(payload, str) and payload else None
        )
    return frame


def list_trade_annotations(
    conn: sqlite3.Connection,
    *,
    limit: int | None = None,
    **filters: Any,
) -> list[dict[str, Any]]:
    """Return annotation records as dictionaries for display/serialization."""

    frame = load_trade_annotations(conn, **filters)
    if limit is not None:
        frame = frame.head(limit)
    return frame.to_dict(orient="records")


def export_annotations(database: Path) -> pd.DataFrame:
    """Load annotations from a SQLite database path."""

    with sqlite3.connect(database) as conn:
        return load_trade_annotations(conn)

