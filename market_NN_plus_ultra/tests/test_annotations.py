from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from market_nn_plus_ultra.reporting.annotations import (
    DECISION_CHOICES,
    TradeAnnotation,
    ensure_annotation_schema,
    insert_trade_annotation,
    load_trade_annotations,
)
from market_nn_plus_ultra.data.validation import validate_trade_annotation_frame


def _setup_database(tmp_path: Path) -> sqlite3.Connection:
    db_path = tmp_path / "market.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE trades (trade_id INTEGER PRIMARY KEY, timestamp TEXT, symbol TEXT)"
    )
    conn.execute(
        "INSERT INTO trades (trade_id, timestamp, symbol) VALUES (?, ?, ?)",
        (1, "2025-10-01T00:00:00+00:00", "AAA"),
    )
    ensure_annotation_schema(conn)
    return conn


def test_insert_and_load_annotation(tmp_path: Path) -> None:
    with _setup_database(tmp_path) as conn:
        annotation = TradeAnnotation(
            trade_id=1,
            decision=DECISION_CHOICES[0],
            rationale="Risk desk approved",
            author="analyst",
            confidence=0.9,
            tags=("risk", "review"),
            created_at=datetime(2025, 10, 5, tzinfo=timezone.utc),
            metadata={"ticket": "ABC-123"},
        )
        stored = insert_trade_annotation(conn, annotation)
        assert stored["trade_id"] == 1
        assert stored["decision"] == DECISION_CHOICES[0]

        frame = load_trade_annotations(conn)
        assert len(frame) == 1
        row = frame.iloc[0]
        assert row["trade_id"] == 1
        assert set(row["tags"]) == {"review", "risk"}
        assert row["metadata"]["ticket"] == "ABC-123"


def test_validate_trade_annotation_frame(tmp_path: Path) -> None:
    with _setup_database(tmp_path) as conn:
        payload = {
            "trade_id": [1, 1],
            "decision": [DECISION_CHOICES[0], DECISION_CHOICES[1]],
            "rationale": ["ok", "needs review"],
            "author": ["alice", "bob"],
            "created_at": pd.to_datetime(
                ["2025-10-01T00:00:00Z", "2025-10-02T00:00:00Z"]
            ),
        }
        frame = pd.DataFrame(payload)
        validated = validate_trade_annotation_frame(frame, trades=pd.DataFrame({"trade_id": [1]}))
        assert len(validated) == 2
        assert list(validated["trade_id"]) == [1, 1]

        with pytest.raises(Exception):
            bad = frame.copy()
            bad.loc[0, "decision"] = "invalid"
            validate_trade_annotation_frame(bad)

