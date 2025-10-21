from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from scripts import annotate_trades


def _prepare_database(tmp_path: Path) -> Path:
    db_path = tmp_path / "trades.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE trades (trade_id INTEGER PRIMARY KEY, timestamp TEXT, symbol TEXT)"
        )
        conn.execute(
            "INSERT INTO trades (trade_id, timestamp, symbol) VALUES (?, ?, ?)",
            (42, "2025-10-01T00:00:00+00:00", "AAA"),
        )
    return db_path


def test_record_annotation_cli(tmp_path: Path, capsys) -> None:
    db_path = _prepare_database(tmp_path)
    exit_code = annotate_trades.main(
        [
            "--database",
            str(db_path),
            "record",
            "--trade-id",
            "42",
            "--decision",
            "approve",
            "--rationale",
            "Risk approved",
            "--author",
            "analyst",
            "--tags",
            "risk,manual",
            "--confidence",
            "0.8",
            "--indent",
            "0",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["trade_id"] == 42
    assert payload["decision"] == "approve"

    list_exit = annotate_trades.main(
        [
            "--database",
            str(db_path),
            "list",
            "--format",
            "json",
            "--indent",
            "0",
        ]
    )

    assert list_exit == 0
    listed = json.loads(capsys.readouterr().out.strip())
    assert len(listed) == 1
    assert listed[0]["decision"] == "approve"

