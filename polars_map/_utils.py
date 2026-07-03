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


def validate(expr: pl.Expr) -> pl.Expr:
    """Rebuild each entry as a clean ``{key, value}`` struct, preserving null entries.

    A null element stays null rather than becoming a non-null struct with null
    key and value fields.
    """
    rebuilt = pl.struct(  # pyright: ignore[reportUnknownMemberType]
        expr.struct["key"].alias("key"),
        expr.struct["value"].alias("value"),
    )
    # preserve nulls
    return pl.when(expr.is_null()).then(None).otherwise(rebuilt)  # pyright: ignore[reportUnknownMemberType]


def dedup() -> pl.Expr:
    """Interior ``list.eval`` expr keeping the first entry for each distinct key.

    Applied as a second pass over already-transformed entries so ``pl.element()``
    refers to the transformed keys, not the pre-transform ones.
    """
    return pl.element().filter(pl.element().struct["key"].is_first_distinct())


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
