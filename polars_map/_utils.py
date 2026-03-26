"""Shared helpers for Map operations."""

from __future__ import annotations

import polars as pl

from ._dtype import Map


def tag(ser: pl.Series) -> pl.Series:
    """Tag a List(Struct({key, value})) series as a Map extension type."""
    [key, value] = ser.dtype.inner.fields  # type: ignore[union-attr]
    return ser.ext.to(Map(key.dtype, value.dtype))  # type: ignore[arg-type]


def infer_map(expr: pl.Expr) -> pl.Expr:
    """Wrap a List(Struct({key, value})) expr as Map, inferring the dtype at runtime."""
    return expr.map_batches(tag, is_elementwise=True)


def validate(expr: pl.Expr, *, validate_fields: bool, deduplicate: bool) -> pl.Expr:
    """Validate and deduplicate a list.eval interior expression."""
    if validate_fields:
        expr = pl.struct(  # pyright: ignore[reportUnknownMemberType]
            expr.struct["key"].alias("key"),
            expr.struct["value"].alias("value"),
        )
    if deduplicate:
        expr = expr.filter(pl.element().struct["key"].is_first_distinct())
    return expr


def expr_eval(expr: pl.Expr, evaled: pl.Expr) -> pl.Expr:
    """Evaluate one expression in the context of another.

    Wraps *expr* in a single-element array, evaluates *evaled* on it
    (where ``pl.element()`` is rebound to the value), then unwraps.
    """
    return (
        pl.concat_arr(expr)  # pyright: ignore[reportUnknownMemberType]
        .arr.eval(evaled)
        .arr.first()
    )
