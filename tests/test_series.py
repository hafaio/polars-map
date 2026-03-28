"""Tests for MapSeries namespace."""

from __future__ import annotations

import polars as pl

from polars_map import Map

from .conftest import smap


def test_keys(map_series: pl.Series) -> None:
    """Verify keys extracts all keys per row."""
    ser = smap(map_series).keys()
    assert ser.to_list() == [["a", "b", "c"], ["x"], None, []]


def test_values(map_series: pl.Series) -> None:
    """Verify values extracts all values per row."""
    ser = smap(map_series).values()
    assert ser.to_list() == [[1, 2, 3], [10], None, []]


def test_len(map_series: pl.Series) -> None:
    """Verify len returns entry count per row."""
    assert smap(map_series).len().to_list() == [3, 1, None, 0]


def test_get(map_series: pl.Series) -> None:
    """Verify get returns value for existing key, None for missing/null/empty."""
    ser = smap(map_series).get("a")
    assert ser.to_list() == [1, None, None, None]


def test_getitem(map_series: pl.Series) -> None:
    """Verify __getitem__ forwards to get for a single key."""
    ser = smap(map_series)["a"]
    assert ser.to_list() == [1, None, None, None]


def test_get_multi(map_series: pl.Series) -> None:
    """Verify get with multiple keys returns multiple columns."""
    [a_ser, z_ser] = smap(map_series).get("a", "z")
    assert a_ser.to_list() == [1, None, None, None]
    assert z_ser.to_list() == [None, None, None, None]


def test_contains_key(map_series: pl.Series) -> None:
    """Verify contains_key for present and absent keys."""
    ser = smap(map_series).contains_key("a")
    assert ser.to_list() == [True, False, None, False]
    ser = smap(map_series).contains_key("z")
    assert ser.to_list() == [False, False, None, False]


def test_filter(map_series: pl.Series) -> None:
    """Verify filter on struct field keeps matching entries and retains Map dtype."""
    ser = smap(map_series).filter(pl.element().struct["value"] > 1)
    assert isinstance(ser.dtype, Map)
    vals = ser.to_list()[0]
    keys = [e["key"] for e in vals]
    assert keys == ["b", "c"]


def test_filter_keys(map_series: pl.Series) -> None:
    """Verify filter_keys keeps entries with matching keys."""
    ser = smap(map_series).filter_keys(pl.element() > "a")
    vals = ser.to_list()[0]
    keys = [e["key"] for e in vals]
    assert keys == ["b", "c"]


def test_filter_values(map_series: pl.Series) -> None:
    """Verify filter_values keeps entries with matching values."""
    ser = smap(map_series).filter_values(pl.element() >= 2)  # noqa: PLR2004
    vals = ser.to_list()[0]
    keys = [e["key"] for e in vals]
    assert keys == ["b", "c"]


def test_eval_values(map_series: pl.Series) -> None:
    """Verify eval_values transforms values and preserves Map type."""
    ser = smap(map_series).eval_values(pl.element() * 2)
    assert isinstance(ser.dtype, Map)
    vals = ser.ext.storage().to_list()[0]
    val_dict = {e["key"]: e["value"] for e in vals}
    assert val_dict == {"a": 2, "b": 4, "c": 6}


def test_eval_keys(map_series: pl.Series) -> None:
    """Verify eval_keys transforms keys and preserves Map type."""
    ser = smap(map_series).eval_keys(pl.element().str.to_uppercase())
    assert isinstance(ser.dtype, Map)
    vals = ser.ext.storage().to_list()[0]
    keys = [e["key"] for e in vals]
    assert keys == ["A", "B", "C"]


def test_entries_strips_extension() -> None:
    """Verify entries strips Map type back to plain List."""
    ser = pl.Series(
        "map", [[{"key": "a", "value": 1}]], dtype=Map(pl.String(), pl.Int64())
    )
    result = smap(ser).entries()
    assert not isinstance(result.dtype, Map)
    assert isinstance(result.dtype, pl.List)


def test_from_entries() -> None:
    """Verify from_entries wraps plain list-of-structs as Map."""
    ser = pl.Series(
        "map",
        [[{"key": "a", "value": 1}, {"key": "b", "value": 2}]],
        dtype=pl.List(pl.Struct({"key": pl.String, "value": pl.Int64})),
    )
    result = smap(ser).from_entries()
    assert isinstance(result.dtype, Map)


def test_from_entries_deduplicates_by_default() -> None:
    """Verify from_entries deduplicates keys, keeping first."""
    ser = pl.Series(
        "map",
        [[{"key": "a", "value": 1}, {"key": "a", "value": 99}]],
        dtype=pl.List(pl.Struct({"key": pl.String, "value": pl.Int64})),
    )
    result = smap(ser).from_entries()
    vals = result.ext.storage().to_list()[0]
    assert len(vals) == 1
    assert vals[0]["value"] == 1


def test_from_entries_no_deduplicate() -> None:
    """Verify from_entries keeps duplicates when disabled."""
    ser = pl.Series(
        "map",
        [[{"key": "a", "value": 1}, {"key": "a", "value": 99}]],
        dtype=pl.List(pl.Struct({"key": pl.String, "value": pl.Int64})),
    )
    result = smap(ser).from_entries(deduplicate=False)
    vals = result.ext.storage().to_list()[0]
    assert len(vals) == 2  # noqa: PLR2004


def test_merge_no_overlap() -> None:
    """Verify merge combines non-overlapping maps and retains Map dtype."""
    left = pl.Series(
        "l", [[{"key": "a", "value": 1}]], dtype=Map(pl.String(), pl.Int64())
    )
    right = pl.Series(
        "r", [[{"key": "b", "value": 2}]], dtype=Map(pl.String(), pl.Int64())
    )
    result = smap(left).merge(right)
    assert isinstance(result.dtype, Map)
    vals = result.to_list()[0]
    keys = sorted(e["key"] for e in vals)
    assert keys == ["a", "b"]


def test_merge_overlap_right_wins() -> None:
    """Verify merge uses right-side value on key conflict."""
    left = pl.Series(
        "l",
        [[{"key": "a", "value": 1}, {"key": "b", "value": 2}]],
        dtype=Map(pl.String(), pl.Int64()),
    )
    right = pl.Series(
        "r", [[{"key": "a", "value": 99}]], dtype=Map(pl.String(), pl.Int64())
    )
    result = smap(left).merge(right)
    vals = result.to_list()[0]
    val_dict = {e["key"]: e["value"] for e in vals}
    assert val_dict["a"] == 99  # noqa: PLR2004
    assert val_dict["b"] == 2  # noqa: PLR2004


def test_intersection() -> None:
    """Verify intersection keeps entries with shared keys and retains Map dtype."""
    left = pl.Series(
        "l",
        [
            [
                {"key": "a", "value": 1},
                {"key": "b", "value": 2},
                {"key": "c", "value": 3},
            ]
        ],
        dtype=Map(pl.String(), pl.Int64()),
    )
    right = pl.Series(
        "r",
        [[{"key": "a", "value": 10}, {"key": "c", "value": 30}]],
        dtype=Map(pl.String(), pl.Int64()),
    )
    result = smap(left).intersection(right)
    assert isinstance(result.dtype, Map)
    vals = result.to_list()[0]
    keys = sorted(e["key"] for e in vals)
    assert keys == ["a", "c"]
    val_dict = {e["key"]: e["value"] for e in vals}
    assert val_dict["a"] == 1
    assert val_dict["c"] == 3  # noqa: PLR2004


def test_difference() -> None:
    """Verify difference keeps entries with keys not in other and retains Map dtype."""
    left = pl.Series(
        "l",
        [
            [
                {"key": "a", "value": 1},
                {"key": "b", "value": 2},
                {"key": "c", "value": 3},
            ]
        ],
        dtype=Map(pl.String(), pl.Int64()),
    )
    right = pl.Series(
        "r",
        [[{"key": "a", "value": 10}, {"key": "c", "value": 30}]],
        dtype=Map(pl.String(), pl.Int64()),
    )
    result = smap(left).difference(right)
    assert isinstance(result.dtype, Map)
    vals = result.to_list()[0]
    keys = [e["key"] for e in vals]
    assert keys == ["b"]
    assert vals[0]["value"] == 2  # noqa: PLR2004
