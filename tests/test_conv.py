"""Tests for arrow conversion functions."""

# pyright: reportUnknownMemberType=false

from __future__ import annotations

from collections.abc import Callable

import polars as pl
import pyarrow as pa
import pytest

from polars_map import (
    Map,
    from_arrow,
    from_arrow_array,
    scan_arrow,
    to_arrow,
    to_arrow_array,
)
from polars_map._conv import Array, ChunkedArray, Field


def _from_arrow_via_table(arr: Array) -> pl.Series:
    return from_arrow(pa.table({"col": arr}))["col"]


@pytest.mark.parametrize(
    "convert", [_from_arrow_via_table, from_arrow_array], ids=["table", "array"]
)
def test_from_arrow_simple_map(convert: Callable[[Array], pl.Series]) -> None:
    """Convert a simple map<str, i64> column."""
    arr: Array = pa.array(
        [[("a", 1), ("b", 2)], [("x", 10)]],
        type=pa.map_(pa.string(), pa.int64()),
    )
    ser = convert(arr)
    assert isinstance(ser.dtype, Map)
    assert ser.dtype.key == pl.String()
    assert ser.dtype.value == pl.Int64()


@pytest.mark.parametrize(
    "convert", [_from_arrow_via_table, from_arrow_array], ids=["table", "array"]
)
def test_from_arrow_map_of_maps(convert: Callable[[Array], pl.Series]) -> None:
    """Convert map<str, map<str, i64>> — two levels of map nesting."""
    inner_type = pa.map_(pa.string(), pa.int64())
    outer_type = pa.map_(pa.string(), inner_type)
    arr: Array = pa.array([[("a", [("x", 1), ("y", 2)])]], type=outer_type)
    ser = convert(arr)
    assert isinstance(ser.dtype, Map)
    assert ser.dtype.key == pl.String()
    assert isinstance(ser.dtype.value, Map)
    assert ser.dtype.value.key == pl.String()
    assert ser.dtype.value.value == pl.Int64()


def test_from_arrow_map_key_is_map() -> None:
    """Convert map<map<str, i64>, str> — map keys are themselves maps."""
    offsets: Array = pa.array([0, 1])
    key_maps: Array = pa.MapArray.from_arrays(offsets, pa.array(["a"]), pa.array([1]))  # pyright: ignore[reportUnknownVariableType]
    outer: Array = pa.MapArray.from_arrays(offsets, key_maps, pa.array(["val"]))  # pyright: ignore[reportUnknownVariableType]
    table = pa.table({"map": outer})  # pyright: ignore[reportUnknownArgumentType]
    frame = from_arrow(table)
    dtype = frame["map"].dtype
    assert isinstance(dtype, Map)
    assert isinstance(dtype.key, Map)
    assert dtype.key.key == pl.String()
    assert dtype.key.value == pl.Int64()
    assert dtype.value == pl.String()


def test_from_arrow_map_in_struct() -> None:
    """Convert struct<x: i64, m: map<str, i64>> column."""
    struct_type = pa.struct(
        [
            pa.field("x", pa.int64()),
            pa.field("m", pa.map_(pa.string(), pa.int64())),
        ]
    )
    arr: Array = pa.array([{"x": 1, "m": [("a", 10)]}], type=struct_type)
    table = pa.table({"s": arr})
    frame = from_arrow(table)
    dtype = frame["s"].dtype
    assert isinstance(dtype, pl.Struct)
    fields = {f.name: f.dtype for f in dtype.fields}
    assert isinstance(fields["m"], Map)
    assert fields["m"].key == pl.String()  # type: ignore[union-attr]
    assert not isinstance(fields["x"], Map)


def test_from_arrow_map_in_list() -> None:
    """Convert list<map<str, i64>> column."""
    list_type = pa.list_(pa.map_(pa.string(), pa.int64()))
    arr: Array = pa.array([[[("a", 1)], [("b", 2)]]], type=list_type)
    table = pa.table({"l": arr})
    frame = from_arrow(table)
    dtype = frame["l"].dtype
    assert isinstance(dtype, pl.List)
    assert isinstance(dtype.inner, Map)
    assert dtype.inner.key == pl.String()  # type: ignore[union-attr]
    assert dtype.inner.value == pl.Int64()  # type: ignore[union-attr]


def test_from_arrow_map_in_array() -> None:
    """Convert fixed_size_list<map<str, i64>, 2> column."""
    array_type = pa.list_(pa.map_(pa.string(), pa.int64()), 2)
    arr: Array = pa.array([[[("a", 1)], [("b", 2)]]], type=array_type)
    table = pa.table({"a": arr})
    frame = from_arrow(table)
    dtype = frame["a"].dtype
    assert isinstance(dtype, pl.Array)
    assert isinstance(dtype.inner, Map)


def test_from_arrow_deeply_nested() -> None:
    """Convert list<struct<m: map<str, map<str, i64>>>> — maps two deep inside containers."""
    inner_map = pa.map_(pa.string(), pa.int64())
    outer_map = pa.map_(pa.string(), inner_map)
    struct_type = pa.struct([pa.field("m", outer_map)])
    list_type = pa.list_(struct_type)
    arr: Array = pa.array([[{"m": [("a", [("x", 1)])]}]], type=list_type)
    table = pa.table({"l": arr})
    frame = from_arrow(table)
    list_dtype = frame["l"].dtype
    assert isinstance(list_dtype, pl.List)
    struct_dtype = list_dtype.inner
    assert isinstance(struct_dtype, pl.Struct)
    [m_field] = struct_dtype.fields
    assert isinstance(m_field.dtype, Map)
    assert isinstance(m_field.dtype.value, Map)  # type: ignore[union-attr]


def test_from_arrow_no_maps() -> None:
    """Table with no map columns passes through unchanged."""
    x: Array = pa.array([1, 2])
    y: Array = pa.array(["a", "b"])
    table = pa.table({"x": x, "y": y})
    frame = from_arrow(table)
    assert frame["x"].to_list() == [1, 2]
    assert frame["y"].to_list() == ["a", "b"]


def test_from_arrow_nulls_and_empties() -> None:
    """Map column with null rows and empty maps."""
    arr: Array = pa.array(
        [[("a", 1)], None, []],
        type=pa.map_(pa.string(), pa.int64()),
    )
    table = pa.table({"map": arr})
    frame = from_arrow(table)
    assert isinstance(frame["map"].dtype, Map)
    storage = frame["map"].ext.storage().to_list()
    assert storage[0] == [{"key": "a", "value": 1}]
    assert storage[1] is None
    assert storage[2] == []


def test_from_arrow_mixed_columns() -> None:
    """Table with both map and non-map columns."""
    map_arr: Array = pa.array([[("a", 1)]], type=pa.map_(pa.string(), pa.int64()))
    x_arr: Array = pa.array([42])
    table = pa.table({"map": map_arr, "x": x_arr})
    frame = from_arrow(table)
    assert isinstance(frame["map"].dtype, Map)
    assert frame["x"].dtype == pl.Int64()
    assert frame["x"].to_list() == [42]


def _to_arrow_via_table(ser: pl.Series) -> ChunkedArray:
    return to_arrow(pl.DataFrame([ser])).column(ser.name)


@pytest.mark.parametrize(
    "convert", [_to_arrow_via_table, to_arrow_array], ids=["table", "array"]
)
def test_to_arrow_simple_map(convert: Callable[[pl.Series], Array]) -> None:
    """Convert Map(Str, I64) to arrow map type."""
    ser = pl.Series(
        "map",
        [[{"key": "a", "value": 1}, {"key": "b", "value": 2}]],
        dtype=Map(pl.String(), pl.Int64()),
    )
    result = convert(ser)
    assert pa.types.is_map(result.type)
    assert result.type.key_type == pa.large_string()
    assert result.type.item_type == pa.int64()


@pytest.mark.parametrize(
    "convert", [_to_arrow_via_table, to_arrow_array], ids=["table", "array"]
)
def test_to_arrow_map_of_maps(convert: Callable[[pl.Series], Array]) -> None:
    """Convert Map(Str, Map(Str, I64)) to nested arrow map types."""
    ser = pl.Series(
        "map",
        [[{"key": "a", "value": [{"key": "x", "value": 1}]}]],
        dtype=Map(pl.String(), Map(pl.String(), pl.Int64())),
    )
    result = convert(ser)
    assert pa.types.is_map(result.type)
    assert pa.types.is_map(result.type.item_type)
    assert result.type.item_type.key_type == pa.large_string()
    assert result.type.item_type.item_type == pa.int64()


def test_to_arrow_map_key_is_map() -> None:
    """Convert Map(Map(Str, I64), Str) — map keys are themselves maps."""
    ser = pl.Series(
        "map",
        [[{"key": [{"key": "a", "value": 1}], "value": "val"}]],
        dtype=Map(Map(pl.String(), pl.Int64()), pl.String()),
    )
    table = to_arrow(pl.DataFrame([ser]))
    col = table.column("map")
    assert pa.types.is_map(col.type)
    assert pa.types.is_map(col.type.key_type)
    assert pa.types.is_large_string(col.type.item_type)


def test_to_arrow_map_in_struct() -> None:
    """Convert a struct column containing a Map field to arrow struct with map field."""
    inner = pl.Series(
        "m",
        [[{"key": "a", "value": 1}]],
        dtype=Map(pl.String(), pl.Int64()),
    )
    frame = pl.DataFrame({"x": [1], "m": inner})
    struct_frame = frame.select(pl.struct("x", "m").alias("s"))
    table = to_arrow(struct_frame)
    struct_type = table.column("s").type
    assert pa.types.is_struct(struct_type)
    m_field: Field = struct_type.field("m")  # pyright: ignore[reportUnknownVariableType]
    assert pa.types.is_map(m_field.type)  # pyright: ignore[reportUnknownArgumentType]


def test_to_arrow_map_in_list() -> None:
    """Convert a list-of-Map column to arrow list of maps."""
    inner = pl.Series(
        "map",
        [[{"key": "a", "value": 1}], [{"key": "b", "value": 2}]],
        dtype=Map(pl.String(), pl.Int64()),
    )
    frame = pl.DataFrame({"map": inner}).select(pl.col("map").implode().alias("l"))
    table = to_arrow(frame)
    col = table.column("l")
    assert pa.types.is_list(col.type) or pa.types.is_large_list(col.type)
    assert pa.types.is_map(col.type.value_type)


def test_to_arrow_no_maps() -> None:
    """DataFrame with no Map columns passes through unchanged."""
    frame = pl.DataFrame({"x": [1, 2], "y": ["a", "b"]})
    table = to_arrow(frame)
    assert table.column("x").to_pylist() == [1, 2]
    assert table.column("y").to_pylist() == ["a", "b"]


@pytest.mark.parametrize(
    "convert", [_to_arrow_via_table, to_arrow_array], ids=["table", "array"]
)
def test_to_arrow_nulls_and_empties(convert: Callable[[pl.Series], Array]) -> None:
    """Map column with nulls and empty maps converts correctly."""
    ser = pl.Series(
        "map",
        [[{"key": "a", "value": 1}], None, []],
        dtype=Map(pl.String(), pl.Int64()),
    )
    result = convert(ser)
    assert pa.types.is_map(result.type)
    assert result[0].as_py() == [("a", 1)]  # pyright: ignore[reportUnknownMemberType]
    assert result[1].as_py() is None  # pyright: ignore[reportUnknownMemberType]
    assert result[2].as_py() == []  # pyright: ignore[reportUnknownMemberType]


def test_to_arrow_mixed_columns() -> None:
    """DataFrame with Map and non-Map columns."""
    ser = pl.Series(
        "map",
        [[{"key": "a", "value": 1}]],
        dtype=Map(pl.String(), pl.Int64()),
    )
    frame = pl.DataFrame({"map": ser, "x": [42]})
    table = to_arrow(frame)
    assert pa.types.is_map(table.column("map").type)
    assert table.column("x").type == pa.int64()


def test_from_arrow_record_batch() -> None:
    """Convert a RecordBatch with a map column."""
    arr: Array = pa.array([[("a", 1)]], type=pa.map_(pa.string(), pa.int64()))
    batch = pa.record_batch({"map": arr})
    frame = from_arrow(batch)
    assert isinstance(frame["map"].dtype, Map)
    storage = frame["map"].ext.storage().to_list()
    assert storage == [[{"key": "a", "value": 1}]]


def test_from_arrow_array_chunked() -> None:
    """Convert a chunked array with maps to a Series."""
    chunk: Array = pa.array([[("a", 1)]], type=pa.map_(pa.string(), pa.int64()))
    chunked: ChunkedArray = pa.chunked_array([chunk])
    ser = from_arrow_array(chunked)
    assert isinstance(ser.dtype, Map)


def test_from_arrow_array_no_map() -> None:
    """Non-map array passes through without conversion."""
    arr: Array = pa.array([1, 2, 3], type=pa.int64())
    ser = from_arrow_array(arr)
    assert ser.name == ""
    assert ser.dtype == pl.Int64()
    assert ser.to_list() == [1, 2, 3]


def test_scan_arrow_simple() -> None:
    """Scan an iterable of Arrow tables preserving map types."""
    a1: Array = pa.array([[("a", 1)]], type=pa.map_(pa.string(), pa.int64()))
    a2: Array = pa.array([[("b", 2)]], type=pa.map_(pa.string(), pa.int64()))
    tables = [pa.table({"map": a1}), pa.table({"map": a2})]
    lf = scan_arrow(lambda: tables)
    frame = lf.collect()
    assert isinstance(frame["map"].dtype, Map)
    assert len(frame) == 2  # noqa: PLR2004


def test_scan_arrow_record_batches() -> None:
    """Scan an iterable of RecordBatches."""
    arr: Array = pa.array([[("a", 1)]], type=pa.map_(pa.string(), pa.int64()))
    batches = [pa.record_batch({"map": arr})]
    lf = scan_arrow(lambda: batches)
    frame = lf.collect()
    assert isinstance(frame["map"].dtype, Map)


def test_scan_arrow_n_rows() -> None:
    """Scan with n_rows limit."""
    a1: Array = pa.array([1, 2, 3])
    a2: Array = pa.array([4, 5, 6])
    tables = [pa.table({"x": a1}), pa.table({"x": a2})]
    lf = scan_arrow(lambda: tables)
    frame = lf.head(2).collect()
    assert len(frame) == 2  # noqa: PLR2004


_roundtrip_frames = [
    pytest.param(
        pl.DataFrame(
            {
                "map": pl.Series(
                    "map",
                    [
                        [{"key": "a", "value": 1}, {"key": "b", "value": 2}],
                        [{"key": "x", "value": 10}],
                        None,
                        [],
                    ],
                    dtype=Map(pl.String(), pl.Int64()),
                ),
                "x": [1, 2, 3, 4],
            }
        ),
        id="simple_map_with_nulls",
    ),
    pytest.param(
        pl.DataFrame(
            {
                "map": pl.Series(
                    "map",
                    [[{"key": "a", "value": [{"key": "x", "value": 1}]}]],
                    dtype=Map(pl.String(), Map(pl.String(), pl.Int64())),
                ),
            }
        ),
        id="nested_map",
    ),
    pytest.param(
        pl.DataFrame({"x": [1, 2], "y": ["a", "b"]}),
        id="no_maps",
    ),
]


@pytest.mark.parametrize("frame", _roundtrip_frames)
def test_roundtrip_polars_arrow_polars(frame: pl.DataFrame) -> None:
    """Polars DataFrame survives to_arrow then from_arrow."""
    result = from_arrow(to_arrow(frame))
    assert result.schema == frame.schema
    assert result.equals(frame)


@pytest.mark.parametrize("frame", _roundtrip_frames)
def test_roundtrip_arrow_polars_arrow(frame: pl.DataFrame) -> None:
    """Arrow table survives from_arrow then to_arrow."""
    table = to_arrow(frame)
    result = to_arrow(from_arrow(table))
    assert result.schema.equals(table.schema)
    assert result.equals(table)


@pytest.mark.parametrize("frame", _roundtrip_frames)
def test_roundtrip_series(frame: pl.DataFrame) -> None:
    """Each series survives to_arrow_array then from_arrow_array."""
    for name in frame.columns:
        ser = frame[name]
        result = from_arrow_array(to_arrow_array(ser))
        assert result.dtype == ser.dtype
        assert result.to_list() == ser.to_list()


def test_to_arrow_array_no_map() -> None:
    """Non-Map series passes through unchanged."""
    ser = pl.Series("x", [1, 2, 3], dtype=pl.Int64())
    result_arr = to_arrow_array(ser)
    assert result_arr.type == pa.int64()
    result = pl.from_arrow(result_arr)
    assert isinstance(result, pl.Series)
    assert result.to_list() == [1, 2, 3]


def test_from_arrow_large_list_of_maps() -> None:
    """Convert large_list<map<str, i64>> column."""
    inner: Array = pa.array(
        [[("a", 1)], [("b", 2)]], type=pa.map_(pa.string(), pa.int64())
    )
    offsets: Array = pa.array([0, 2], type=pa.int64())
    col: Array = pa.LargeListArray.from_arrays(offsets, inner)  # pyright: ignore[reportUnknownVariableType]
    table = pa.table({"l": col})  # pyright: ignore[reportUnknownArgumentType]
    frame = from_arrow(table)
    dtype = frame["l"].dtype
    assert isinstance(dtype, pl.List)
    assert isinstance(dtype.inner, Map)


def test_from_arrow_map_in_array_roundtrip() -> None:
    """Convert fixed_size_list<map<str, i64>, 2> through to_arrow and back."""
    array_type = pa.list_(pa.map_(pa.string(), pa.int64()), 2)
    arr: Array = pa.array([[[("a", 1)], [("b", 2)]]], type=array_type)
    table = pa.table({"a": arr})
    frame = from_arrow(table)
    result = to_arrow(frame)
    assert pa.types.is_map(result.column("a").type.value_type)


def test_scan_arrow_with_columns() -> None:
    """Scan with column projection."""
    x: Array = pa.array([1])
    y: Array = pa.array(["a"])
    tables = [pa.table({"x": x, "y": y})]
    lf = scan_arrow(lambda: tables)
    frame = lf.select("x").collect()
    assert frame.columns == ["x"]
    assert frame["x"].to_list() == [1]


def test_scan_arrow_with_predicate() -> None:
    """Scan with predicate filter."""
    arr: Array = pa.array([1, 2, 3])
    tables = [pa.table({"x": arr})]
    lf = scan_arrow(lambda: tables)
    frame = lf.filter(pl.col("x") > 1).collect()
    assert frame["x"].to_list() == [2, 3]


def test_scan_arrow_collect_twice() -> None:
    """Collecting a scanned LazyFrame twice reuses cached schema."""
    call_count = 0

    def source() -> list[pa.Table]:
        nonlocal call_count
        call_count += 1
        arr: Array = pa.array([1, 2])
        return [pa.table({"x": arr})]

    lf = scan_arrow(source)
    frame1 = lf.collect()
    frame2 = lf.collect()
    assert frame1.equals(frame2)
    assert call_count == 2  # noqa: PLR2004
