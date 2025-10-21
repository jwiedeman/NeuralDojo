"""CLI for recording and inspecting analyst trade annotations."""

from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from market_nn_plus_ultra.reporting import (
    DECISION_CHOICES,
    TradeAnnotation,
    ensure_annotation_schema,
    insert_trade_annotation,
    load_trade_annotations,
)


def _parse_datetime(value: str | None) -> datetime | None:
    if value is None:
        return None
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.to_pydatetime()


def _parse_tags(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(sorted({tag.strip() for tag in value.split(",") if tag.strip()}))


def _load_trade_metadata(conn: sqlite3.Connection, trade_id: int) -> dict[str, Any] | None:
    cursor = conn.execute(
        "SELECT trade_id, timestamp, symbol FROM trades WHERE trade_id = ?",
        (trade_id,),
    )
    row = cursor.fetchone()
    if row is None:
        return None
    return {
        "trade_id": row[0],
        "timestamp": pd.Timestamp(row[1]) if row[1] is not None else None,
        "symbol": row[2],
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, (datetime, pd.Timestamp)):
        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return ts.isoformat()
    if isinstance(value, (set, tuple)):
        return list(value)
    return value


def _print_json(payload: Any, *, indent: int | None) -> None:
    print(json.dumps(payload, indent=indent, default=_json_default))


def _command_record(args: argparse.Namespace) -> int:
    db_path = Path(args.database)
    with sqlite3.connect(db_path) as conn:
        ensure_annotation_schema(conn)
        trade_info = None
        trade_id = args.trade_id
        if trade_id is not None:
            trade_info = _load_trade_metadata(conn, trade_id)
            if trade_info is None:
                print(f"error: trade_id {trade_id} not found", flush=True)
                return 1
        else:
            if not args.symbol or not args.trade_timestamp:
                print(
                    "error: provide either --trade-id or both --symbol and --trade-timestamp",
                    flush=True,
                )
                return 1
            ts = _parse_datetime(args.trade_timestamp)
            query = (
                "SELECT trade_id, timestamp, symbol FROM trades WHERE symbol = ? AND timestamp = ?"
            )
            trade_info = None
            if ts is not None:
                cursor = conn.execute(query, (args.symbol, pd.Timestamp(ts, tz="UTC").isoformat()))
                row = cursor.fetchone()
                if row is not None:
                    trade_id = int(row[0])
                    trade_info = {
                        "trade_id": row[0],
                        "timestamp": pd.Timestamp(row[1]) if row[1] is not None else ts,
                        "symbol": row[2],
                    }
            if trade_id is None:
                print("error: could not locate trade for provided symbol/timestamp", flush=True)
                return 1

        symbol = args.symbol or (trade_info["symbol"] if trade_info else None)
        trade_timestamp = args.trade_timestamp or (trade_info["timestamp"] if trade_info else None)
        annotation = TradeAnnotation(
            trade_id=trade_id,
            decision=args.decision,
            rationale=args.rationale,
            author=args.author,
            symbol=symbol,
            trade_timestamp=_parse_datetime(trade_timestamp) if trade_timestamp is not None else None,
            confidence=args.confidence,
            tags=_parse_tags(args.tags),
            created_at=_parse_datetime(args.created_at) if args.created_at else datetime.now(timezone.utc),
            metadata=json.loads(args.metadata) if args.metadata else None,
            context_window_start=_parse_datetime(args.context_window_start),
            context_window_end=_parse_datetime(args.context_window_end),
        )
        stored = insert_trade_annotation(conn, annotation)
        _print_json(stored, indent=args.indent)
    return 0


def _command_list(args: argparse.Namespace) -> int:
    db_path = Path(args.database)
    filters: dict[str, Any] = {}
    if args.trade_id:
        filters["trade_ids"] = [args.trade_id]
    if args.symbol:
        filters["symbol"] = args.symbol
    if args.decision:
        filters["decision"] = args.decision

    with sqlite3.connect(db_path) as conn:
        frame = load_trade_annotations(conn, **filters)
    if args.limit is not None:
        frame = frame.head(args.limit)
    if args.format == "json":
        _print_json(frame.to_dict(orient="records"), indent=args.indent)
    else:
        if frame.empty:
            print("No annotations found.")
        else:
            print(frame.to_markdown(index=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Record or view trade annotations.")
    parser.add_argument("--database", required=True, help="Path to the SQLite database.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    record = subparsers.add_parser("record", help="Record an analyst annotation.")
    record.add_argument("--trade-id", type=int, required=False, help="Identifier of the trade to annotate.")
    record.add_argument("--decision", choices=DECISION_CHOICES, required=True, help="Decision outcome.")
    record.add_argument("--rationale", required=True, help="Human-readable rationale for the decision.")
    record.add_argument("--author", required=True, help="Name or handle of the analyst.")
    record.add_argument("--symbol", help="Optional symbol override if missing from trades.")
    record.add_argument("--trade-timestamp", dest="trade_timestamp", help="Trade timestamp if not present in trades table.")
    record.add_argument("--confidence", type=float, help="Optional analyst confidence in [0,1].")
    record.add_argument("--tags", help="Comma-separated tags for downstream search.")
    record.add_argument("--created-at", dest="created_at", help="Override creation timestamp (ISO 8601).")
    record.add_argument("--metadata", help="Optional JSON payload with structured metadata.")
    record.add_argument("--context-window-start", help="Start timestamp for the reviewed window.")
    record.add_argument("--context-window-end", help="End timestamp for the reviewed window.")
    record.add_argument("--indent", type=int, default=2, help="Indentation for JSON output.")
    record.set_defaults(func=_command_record)

    list_cmd = subparsers.add_parser("list", help="List stored annotations.")
    list_cmd.add_argument("--trade-id", type=int, help="Filter by trade identifier.")
    list_cmd.add_argument("--symbol", help="Filter by symbol.")
    list_cmd.add_argument("--decision", choices=DECISION_CHOICES, help="Filter by decision outcome.")
    list_cmd.add_argument("--limit", type=int, help="Optional row limit.")
    list_cmd.add_argument("--format", choices=("json", "table"), default="json", help="Output format.")
    list_cmd.add_argument("--indent", type=int, default=2, help="Indentation for JSON output.")
    list_cmd.set_defaults(func=_command_list)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())

