"""Expr namespace for Map operations."""

from __future__ import annotations

import functools
from dataclasses import dataclass

import polars as pl

from ._utils import expr_eval, infer_map, validate


@pl.api.register_expr_namespace("map")
@dataclass(frozen=True)
class MapExpr:
    """Expression namespace for Map operations on List(Struct({key, value})) columns."""

    _expr: pl.Expr

    def _as_self(self, expr: pl.Expr) -> pl.Expr:
        """Wrap a List(Struct) result back as Map, preserving the original dtype."""
        return expr.ext.to(pl.dtype_of(self._expr))

    def from_entries(
        self,
        *,
        validate_fields: bool = True,
        deduplicate: bool = True,
        parallel: bool = False,
    ) -> pl.Expr:
        """Wrap a List(Struct({key, value})) expression as a Map extension type.

        Parameters
        ----------
        deduplicate
            If True, deduplicate by key, keeping the first occurrence.
        parallel
            Run list evaluations in parallel.
        """
        return infer_map(
            self._expr.list.eval(
                validate(
                    pl.element(),
                    validate_fields=validate_fields,
                    deduplicate=deduplicate,
                ),
                parallel=parallel,
            )
        )

    @functools.cached_property
    def _entries(self) -> pl.Expr:
        return self._expr.ext.storage()

    def entries(self) -> pl.Expr:
        """Strip the Map extension type, returning raw List(Struct({key, value}))."""
        return self._entries

    def keys(self) -> pl.Expr:
        """Extract all keys as a List column."""
        return self._entries.list.eval(pl.element().struct["key"])

    def values(self) -> pl.Expr:
        """Extract all values as a List column."""
        return self._entries.list.eval(pl.element().struct["value"])

    def len(self) -> pl.Expr:
        """Return the number of entries in the map."""
        return self._entries.list.len()

    def _get(self, key: object) -> pl.Expr:
        """Look up a value by key. Returns scalar per row."""
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

    def get(self, key: object, *keys: object) -> pl.Expr:
        """Look up a value by key. Returns scalar per row."""
        if keys:
            return pl.struct(  # pyright: ignore[reportUnknownMemberType]
                self._get(key), *(self._get(k) for k in keys)
            ).struct.unnest()
        else:
            return self._get(key)

    def contains_key(self, key: object) -> pl.Expr:
        """Check if a key exists in the map."""
        return self._entries.list.eval(pl.element().struct["key"] == key).list.any()

    def eval(
        self,
        expr: pl.Expr,
        *,
        validate_fields: bool = True,
        deduplicate: bool = True,
        parallel: bool = False,
    ) -> pl.Expr:
        """Evaluate an expression on entries, returning a Map.

        The expression operates on the struct elements via ``pl.element()``.

        Example
        -------
        >>> col.map.eval(pl.element().struct.with_fields(pl.element().struct["value"] * 2))
        """
        inner = validate(expr, validate_fields=validate_fields, deduplicate=deduplicate)
        return infer_map(self._entries.list.eval(inner, parallel=parallel))

    def eval_keys(
        self, expr: pl.Expr, *, deduplicate: bool = True, parallel: bool = False
    ) -> pl.Expr:
        """Transform keys, returning a Map with new key type.

        The expression operates on each key via ``pl.element()``.

        Example
        -------
        >>> col.map.eval_keys(pl.element().str.to_uppercase())
        """
        inner: pl.Expr = pl.element().struct.with_fields(  # pyright: ignore[reportUnknownMemberType]
            key=expr_eval(pl.element().struct["key"], expr)
        )
        if deduplicate:
            inner = inner.filter(pl.element().struct["key"].is_first_distinct())
        return infer_map(self._entries.list.eval(inner, parallel=parallel))

    def eval_values(self, expr: pl.Expr, *, parallel: bool = False) -> pl.Expr:
        """Transform values, returning a Map with new value type.

        The expression operates on each value via ``pl.element()``.

        Example
        -------
        >>> col.map.eval_values(pl.element() * 2)
        """
        inner = pl.element().struct.with_fields(  # pyright: ignore[reportUnknownMemberType]
            value=expr_eval(pl.element().struct["value"], expr)
        )
        return infer_map(self._entries.list.eval(inner, parallel=parallel))

    def filter(self, predicate: pl.Expr, *, parallel: bool = False) -> pl.Expr:
        """Filter entries by a predicate on the struct entry.

        Example
        -------
        >>> col.map.filter(pl.element().struct["key"] > "b")
        """
        return self._as_self(
            self._entries.list.eval(pl.element().filter(predicate), parallel=parallel)
        )

    def filter_keys(self, predicate: pl.Expr, *, parallel: bool = False) -> pl.Expr:
        """Filter entries where the key satisfies the predicate.

        Example
        -------
        >>> col.map.filter_keys(pl.element() > "b")
        """
        inner = pl.element().filter(
            expr_eval(pl.element().struct["key"], predicate)  # pyright: ignore[reportUnknownMemberType]
        )
        return self._as_self(self._entries.list.eval(inner, parallel=parallel))

    def filter_values(self, predicate: pl.Expr, *, parallel: bool = False) -> pl.Expr:
        """Filter entries where the value satisfies the predicate.

        Example
        -------
        >>> col.map.filter_values(pl.element() > 5)
        """
        inner = pl.element().filter(
            expr_eval(pl.element().struct["value"], predicate)  # pyright: ignore[reportUnknownMemberType]
        )
        return self._as_self(self._entries.list.eval(inner, parallel=parallel))

    def merge(self, other: pl.Expr, *, parallel: bool = False) -> pl.Expr:
        """Merge two maps. Right-side values win on key conflict."""
        combined = pl.concat_list([self._entries, other.map.entries()])  # pyright: ignore[reportUnknownMemberType,reportAttributeAccessIssue,reportUnknownVariableType]
        return self._as_self(
            combined.list.eval(
                pl.element().filter(pl.element().struct["key"].is_last_distinct()),
                parallel=parallel,
            )
        )

    def intersection(self, other: pl.Expr, *, parallel: bool = False) -> pl.Expr:
        """Keep entries from self where the key also exists in other."""
        combined = pl.concat_list([self._entries, other.map.entries()])  # pyright: ignore[reportUnknownMemberType,reportAttributeAccessIssue,reportUnknownVariableType]
        return self._as_self(
            combined.list.eval(
                pl.element().filter(
                    pl.element().struct["key"].is_duplicated()
                    & pl.element().struct["key"].is_first_distinct()
                ),
                parallel=parallel,
            )
        )

    def difference(self, other: pl.Expr, *, parallel: bool = False) -> pl.Expr:
        """Keep entries from self where the key does NOT exist in other."""
        other_entries = other.map.entries()  # pyright: ignore[reportUnknownMemberType,reportAttributeAccessIssue,reportUnknownVariableType]
        combined = pl.concat_list([self._entries, other_entries, other_entries])  # pyright: ignore[reportUnknownMemberType]
        return self._as_self(
            combined.list.eval(
                pl.element().filter(~pl.element().struct["key"].is_duplicated()),
                parallel=parallel,
            )
        )
