"""Series namespace for Map operations."""

from __future__ import annotations

import functools
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, overload

import polars as pl

from ._utils import expr_eval, tag, validate


@pl.api.register_series_namespace("map")
@dataclass(frozen=True)
class MapSeries:
    """Series namespace for Map operations on List(Struct({key, value})) columns."""

    _series: pl.Series

    def from_entries(
        self,
        *,
        validate_fields: bool = True,
        deduplicate: bool = True,
        parallel: bool = False,
    ) -> pl.Series:
        """Wrap a List(Struct({key, value})) Series as a Map extension type.

        Parameters
        ----------
        deduplicate
            If True, deduplicate by key, keeping the first occurrence.
        parallel
            Run list evaluations in parallel.
        """
        inner = validate(
            pl.element(),
            validate_fields=validate_fields,
            deduplicate=deduplicate,
        )
        return tag(self._series.list.eval(inner, parallel=parallel))

    @functools.cached_property
    def _entries(self) -> pl.Series:
        return self._series.ext.storage()

    def entries(self) -> pl.Series:
        """Strip the Map extension type, returning raw List(Struct({key, value}))."""
        return self._entries

    def keys(self) -> pl.Series:
        """Extract all keys as a List column."""
        return self._entries.list.eval(pl.element().struct["key"])

    def values(self) -> pl.Series:
        """Extract all values as a List column."""
        return self._entries.list.eval(pl.element().struct["value"])

    def len(self) -> pl.Series:
        """Return the number of entries in the map."""
        return self._entries.list.len()

    def _get(self, key: object) -> pl.Series:
        """Look up a single value by key. Returns scalar per row."""
        return (
            self._entries.list.eval(
                pl.element()
                .filter(pl.element().struct["key"] == key)
                .struct["value"]
                .first()
            )
            .list.first()
            .alias(str(key))
        )

    @overload
    def get(self, key: object, /) -> pl.Series: ...  # pyright: ignore[reportOverlappingOverload]
    @overload
    def get(self, key: object, *keys: object) -> pl.DataFrame: ...

    def get(self, key: object, *keys: object) -> pl.Series | pl.DataFrame:
        """Look up a value by key. Returns scalar per row."""
        if keys:
            return pl.DataFrame([self._get(key), *(self._get(k) for k in keys)])
        else:
            return self._get(key)

    def contains_key(self, key: object) -> pl.Series:
        """Check if a key exists in the map."""
        return self._entries.list.eval(pl.element().struct["key"] == key).list.any()

    def eval(
        self,
        expr: pl.Expr,
        *,
        validate_fields: bool = True,
        deduplicate: bool = True,
        parallel: bool = False,
    ) -> pl.Series:
        """Evaluate an expression on entries, returning a Map."""
        inner = validate(expr, validate_fields=validate_fields, deduplicate=deduplicate)
        return tag(self._entries.list.eval(inner, parallel=parallel))

    def eval_keys(
        self, expr: pl.Expr, *, deduplicate: bool = True, parallel: bool = False
    ) -> pl.Series:
        """Transform keys, returning a Map with new key type."""
        inner: pl.Expr = pl.element().struct.with_fields(  # pyright: ignore[reportUnknownMemberType]
            key=expr_eval(pl.element().struct["key"], expr)
        )
        if deduplicate:
            inner = inner.filter(pl.element().struct["key"].is_first_distinct())
        return tag(self._entries.list.eval(inner, parallel=parallel))

    def eval_values(self, expr: pl.Expr, *, parallel: bool = False) -> pl.Series:
        """Transform values, returning a Map with new value type."""
        inner = pl.element().struct.with_fields(  # pyright: ignore[reportUnknownMemberType]
            value=expr_eval(pl.element().struct["value"], expr)
        )
        return tag(self._entries.list.eval(inner, parallel=parallel))

    def filter(self, predicate: pl.Expr, *, parallel: bool = False) -> pl.Series:
        """Filter entries by a predicate on the struct entry."""
        return tag(
            self._entries.list.eval(pl.element().filter(predicate), parallel=parallel)
        )

    def filter_keys(self, predicate: pl.Expr, *, parallel: bool = False) -> pl.Series:
        """Filter entries where the key satisfies the predicate."""
        inner = pl.element().filter(
            expr_eval(pl.element().struct["key"], predicate)  # pyright: ignore[reportUnknownMemberType]
        )
        return tag(self._entries.list.eval(inner, parallel=parallel))

    def filter_values(self, predicate: pl.Expr, *, parallel: bool = False) -> pl.Series:
        """Filter entries where the value satisfies the predicate."""
        return tag(
            self._entries.list.eval(
                pl.element().filter(
                    expr_eval(pl.element().struct["value"], predicate)  # pyright: ignore[reportUnknownMemberType]
                ),
                parallel=parallel,
            )
        )

    def merge(self, other: pl.Series, *, parallel: bool = False) -> pl.Series:
        """Merge two maps. Right-side values win on key conflict."""
        combined = self._entries.list.concat(other.map.entries())  # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType,reportUnknownArgumentType]
        return tag(
            combined.list.eval(
                pl.element().filter(pl.element().struct["key"].is_last_distinct()),
                parallel=parallel,
            )
        )

    def intersection(self, other: pl.Series, *, parallel: bool = False) -> pl.Series:
        """Keep entries from self where the key also exists in other."""
        combined = self._entries.list.concat(other.map.entries())  # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType,reportUnknownArgumentType]
        inner = pl.element().filter(
            pl.element().struct["key"].is_duplicated()
            & pl.element().struct["key"].is_first_distinct()
        )
        return tag(combined.list.eval(inner, parallel=parallel))

    def difference(self, other: pl.Series, *, parallel: bool = False) -> pl.Series:
        """Keep entries from self where the key does NOT exist in other."""
        other_entries: pl.Series = other.map.entries()  # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType,reportUnknownVariableType]
        combined = self._entries.list.concat(other_entries).list.concat(other_entries)  # pyright: ignore[reportUnknownArgumentType]
        inner = pl.element().filter(~pl.element().struct["key"].is_duplicated())
        return tag(combined.list.eval(inner, parallel=parallel))

    def __iter__(self) -> Iterator[dict[Any, Any] | None]:
        """Iterate over rows, yielding Python dicts."""
        for row in self._entries:
            if row is None:
                yield None
            else:
                yield {entry["key"]: entry["value"] for entry in row}

    def to_list(self) -> list[dict[Any, Any] | None]:
        """Convert into a list of python dictionaries."""
        return [*self]
