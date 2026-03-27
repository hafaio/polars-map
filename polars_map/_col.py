"""Typed ``col`` helper that exposes the Map namespace to type checkers."""

from __future__ import annotations

from typing import Protocol, cast

import polars as pl

from polars_map._expr import MapExpr


class Expr(pl.Expr):
    """Expr with a typed ``map`` namespace (for type-checking only)."""

    map: MapExpr  # type: ignore[assignment]


class _Col(Protocol):
    def __call__(
        self,
        name: str,
        *more_names: str,
    ) -> Expr: ...

    def __getattr__(self, name: str) -> Expr: ...

    @property
    def map(self) -> MapExpr: ...


col = cast(_Col, pl.col)
