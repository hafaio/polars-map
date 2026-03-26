"""Shared test fixtures."""

from __future__ import annotations

import polars as pl
import pytest

from polars_map import Map, MapExpr, MapSeries


def emap(expr: pl.Expr) -> MapExpr:
    """Access the map namespace on an Expr with proper typing."""
    return expr.map  # type: ignore[attr-defined]


def smap(ser: pl.Series) -> MapSeries:
    """Access the map namespace on a Series with proper typing."""
    return ser.map  # type: ignore[attr-defined]


@pytest.fixture()
def map_series() -> pl.Series:
    """Create a test Series with a map column."""
    return pl.Series(
        "map",
        [
            [
                {"key": "a", "value": 1},
                {"key": "b", "value": 2},
                {"key": "c", "value": 3},
            ],
            [{"key": "x", "value": 10}],
            None,
            [],
        ],
        dtype=Map(pl.String(), pl.Int64()),
    )


@pytest.fixture()
def map_frame(map_series: pl.Series) -> pl.DataFrame:
    """Create a test DataFrame with a map column."""
    return pl.DataFrame([map_series])
