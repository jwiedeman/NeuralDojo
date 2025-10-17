"""Utilities for fusing alternative data tables into the training panel."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Iterable, Mapping, Sequence

import pandas as pd

logger = logging.getLogger(__name__)


def _default_parse_dates(columns: Iterable[str]) -> tuple[str, ...]:
    """Return timestamp-like columns that should be parsed as datetimes."""

    parsed: list[str] = []
    for column in columns:
        lowered = column.lower()
        if "time" in lowered or "date" in lowered:
            parsed.append(column)
    return tuple(parsed)


@dataclass(slots=True)
class AlternativeDataSpec:
    """Configuration describing how to ingest an alternative data table."""

    name: str
    table: str
    join_columns: tuple[str, ...] = ("timestamp", "symbol")
    columns: tuple[str, ...] | None = None
    prefix: str | None = None
    fill_forward: bool = True
    fill_backward: bool = False
    filters: Mapping[str, Sequence[str]] = field(default_factory=dict)
    parse_dates: tuple[str, ...] | None = None

    def build(self) -> "AlternativeDataConnector":
        """Instantiate a runtime connector from this specification."""

        return AlternativeDataConnector(
            name=self.name,
            table=self.table,
            join_columns=self.join_columns,
            columns=self.columns,
            prefix=self.prefix,
            fill_forward=self.fill_forward,
            fill_backward=self.fill_backward,
            filters={key: tuple(values) for key, values in self.filters.items()},
            parse_dates=self.parse_dates,
        )


@dataclass(slots=True)
class AlternativeDataConnector:
    """Load and merge alternative data features from SQLite tables."""

    name: str
    table: str
    join_columns: tuple[str, ...]
    columns: tuple[str, ...] | None
    prefix: str | None
    fill_forward: bool
    fill_backward: bool
    filters: Mapping[str, tuple[str, ...]]
    parse_dates: tuple[str, ...] | None

    def fetch(self, conn, *, symbols: Sequence[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
        """Return the alternative data frame and the produced feature columns."""

        query = self.table if " " in self.table.lower() or "\n" in self.table else f"SELECT * FROM {self.table}"
        parse_dates = self.parse_dates or _default_parse_dates(self.join_columns)
        df = pd.read_sql_query(query, conn, parse_dates=list(parse_dates))

        if not set(self.join_columns).issubset(df.columns):
            missing = sorted(set(self.join_columns) - set(df.columns))
            raise KeyError(
                f"Alternative data '{self.name}' is missing required join columns: {missing}"
            )

        if self.columns is None:
            value_columns = [column for column in df.columns if column not in self.join_columns]
        else:
            value_columns = list(self.columns)

        missing_values = [column for column in value_columns if column not in df.columns]
        if missing_values:
            raise KeyError(
                f"Alternative data '{self.name}' is missing requested columns: {missing_values}"
            )

        if symbols is not None and "symbol" in self.join_columns and "symbol" in df.columns:
            df = df[df["symbol"].isin(symbols)]

        if self.filters:
            for column, values in self.filters.items():
                if column not in df.columns:
                    raise KeyError(
                        f"Alternative data '{self.name}' cannot filter on missing column '{column}'"
                    )
                df = df[df[column].isin(values)]

        if df.empty:
            logger.debug("Alternative data '%s' produced an empty frame", self.name)
            empty_columns = list(self.join_columns) + [self._rename_column(col) for col in value_columns]
            return pd.DataFrame(columns=empty_columns), [self._rename_column(col) for col in value_columns]

        keep_columns = list(self.join_columns) + value_columns
        df = df[keep_columns]
        df = df.sort_values(list(self.join_columns))
        df = df.drop_duplicates(subset=list(self.join_columns), keep="last")

        renamed_columns: list[str] = []
        prefix = self.prefix if self.prefix is not None else self.name
        if prefix:
            rename_map = {}
            for column in value_columns:
                new_name = f"{prefix}__{column}" if prefix else column
                rename_map[column] = new_name
                renamed_columns.append(new_name)
            df = df.rename(columns=rename_map)
        else:
            renamed_columns = list(value_columns)

        return df, renamed_columns

    def enrich(self, frame: pd.DataFrame, alt_df: pd.DataFrame, value_columns: list[str]) -> pd.DataFrame:
        """Merge alternative data columns into ``frame`` and apply fills."""

        if alt_df.empty:
            # Merge keeps column placeholders even for empty frames.
            alt_df = alt_df.copy()

        join_columns = list(self.join_columns)
        merged = frame.merge(alt_df, on=join_columns, how="left", sort=False)

        if not value_columns:
            return merged

        sort_columns: list[str] = []
        if "timestamp" in merged.columns:
            sort_columns.append("timestamp")
        for column in join_columns:
            if column != "timestamp" and column in merged.columns and column not in sort_columns:
                sort_columns.append(column)

        if sort_columns:
            merged = merged.sort_values(sort_columns)

        group_columns = [column for column in join_columns if column != "timestamp" and column in merged.columns]
        if self.fill_forward:
            if group_columns:
                merged[value_columns] = merged.groupby(group_columns, sort=False)[value_columns].ffill()
            else:
                merged[value_columns] = merged[value_columns].ffill()
        if self.fill_backward:
            if group_columns:
                merged[value_columns] = merged.groupby(group_columns, sort=False)[value_columns].bfill()
            else:
                merged[value_columns] = merged[value_columns].bfill()

        return merged

    def _rename_column(self, column: str) -> str:
        prefix = self.prefix if self.prefix is not None else self.name
        return f"{prefix}__{column}" if prefix else column


__all__ = ["AlternativeDataSpec", "AlternativeDataConnector"]

