"""Tests for Map extension type."""

from __future__ import annotations

import polars as pl

from polars_map import Map

from .conftest import smap


def test_key_value_types() -> None:
    """Verify key and value dtype accessors."""
    m = Map(pl.String(), pl.Int64())
    assert m.key == pl.String()
    assert m.value == pl.Int64()

    m2 = Map(pl.Int32(), pl.Float64())
    assert m2.key == pl.Int32()
    assert m2.value == pl.Float64()


def test_isinstance_check() -> None:
    """Verify isinstance works for Map type."""
    assert isinstance(Map(pl.String(), pl.Int64()), Map)
    assert not isinstance(pl.String(), Map)
    assert not isinstance(pl.List(pl.Int64), Map)


def test_string_repr() -> None:
    """Verify internal string representation."""
    m = Map(pl.String(), pl.Int64())
    assert m._string_repr() == "map[str,i64]"  # pyright: ignore[reportPrivateUsage]


def test_extension_name() -> None:
    """Verify extension name used for registration."""
    assert Map(pl.String(), pl.Int64()).ext_name() == "polars_map.map"


def test_create_series() -> None:
    """Verify Series creation with Map dtype and display format."""
    ser = pl.Series(
        "map", [[{"key": "a", "value": 1}]], dtype=Map(pl.String(), pl.Int64())
    )
    assert isinstance(ser.dtype, Map)
    assert "ext[map[str,i64]]" in str(ser)


def test_series_rows(map_series: pl.Series) -> None:
    """Verify multiple rows, null rows, and empty maps."""
    assert map_series.len() == 4  # noqa: PLR2004
    assert map_series.is_null()[2]
    [length, *_] = map_series.ext.storage().list.len()
    assert length == 3  # noqa: PLR2004


def test_ext_storage_roundtrip() -> None:
    """Verify ext.storage strips Map type to plain List."""
    ser = pl.Series(
        "map", [[{"key": "a", "value": 1}]], dtype=Map(pl.String(), pl.Int64())
    )
    storage = ser.ext.storage()
    assert not isinstance(storage.dtype, Map)
    assert isinstance(storage.dtype, pl.List)


def test_iter(map_series: pl.Series) -> None:
    """Verify iterating over map series yields Python dicts."""
    result = [*smap(map_series)]
    assert result == [{"a": 1, "b": 2, "c": 3}, {"x": 10}, None, {}]
