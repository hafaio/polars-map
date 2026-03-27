"""Tests for polars_map.col typed helper."""

from __future__ import annotations

import polars as pl

import polars_map
from polars_map import col


def test_col_is_pl_col() -> None:
    """Verify col is pl.col at runtime."""
    assert col is pl.col


def test_col_returns_expr() -> None:
    """Verify col returns a regular Expr at runtime."""
    assert isinstance(col("a"), pl.Expr)


def test_col_map_namespace(map_frame: pl.DataFrame) -> None:
    """Verify col("map").map.keys() works end-to-end."""
    expr = col("map").map.keys()  # this should type-check!
    [ser] = map_frame.select(expr)  # pyright: ignore[reportUnknownMemberType]
    assert ser.to_list() == [["a", "b", "c"], ["x"], None, []]


def test_col_map_get(map_frame: pl.DataFrame) -> None:
    """Verify col("map").map.get() works end-to-end."""
    expr = col("map").map.get("a")  # this should type-check!
    [ser] = map_frame.select(expr)  # pyright: ignore[reportUnknownMemberType]
    assert ser.to_list() == [1, None, None, None]


def test_col_expr_type_annotation() -> None:
    """Verify Expr type can be used in annotations."""
    _e: polars_map.Expr = col("a")
