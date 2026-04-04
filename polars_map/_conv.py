"""Convert between Arrow tables and Polars DataFrames preserving Map extension types."""

# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownParameterType=false

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterable, Iterator, Mapping
from functools import lru_cache, partial
from typing import TypeAlias

import polars as pl
import pyarrow as pa
from polars.io.plugins import register_io_source

from ._dtype import Map

# NOTE These aliases are strings because the pyarrow types are not
# subscriptable, but the stubs are
Array: TypeAlias = "pa.Array[pa.Scalar[pa.DataType]]"
ChunkedArray: TypeAlias = "pa.ChunkedArray[pa.Scalar[pa.DataType]]"
Field: TypeAlias = "pa.Field[pa.DataType]"
FromConv: TypeAlias = "Callable[[pl.Expr], pl.Expr]"
ToConv: TypeAlias = "Callable[[Array], Array]"


def _arrow_leaf_dtype(arrow_type: pa.DataType) -> pl.DataType:
    """Convert a leaf arrow type to a polars dtype."""
    (conv,) = pl.Schema([pa.field("", arrow_type)]).values()
    return conv


def _apply_map_nested(
    struct_expr: pl.Expr, map_dtype: pl.DataType, expr: pl.Expr
) -> pl.Expr:
    return expr.list.eval(struct_expr).ext.to(map_dtype)


def _apply_map_simple(map_dtype: pl.DataType, expr: pl.Expr) -> pl.Expr:
    return expr.ext.to(map_dtype)


def _apply_struct(field_convs: dict[str, FromConv], expr: pl.Expr) -> pl.Expr:
    return expr.struct.with_fields(  # pyright: ignore[reportUnknownMemberType]
        **{name: conv(expr.struct[name]) for name, conv in field_convs.items()}
    )


def _apply_list(inner_expr: pl.Expr, expr: pl.Expr) -> pl.Expr:
    return expr.list.eval(inner_expr)


def _apply_array(inner_expr: pl.Expr, width: int, expr: pl.Expr) -> pl.Expr:
    # arr.eval panics with extension types in children, so roundtrip through
    # list instead, which is not _free_. #27193
    return expr.arr.to_list().list.eval(inner_expr).list.to_array(width)


@lru_cache
def _from_arrow_walk(  # noqa: PLR0911, PLR0912
    arrow_type: pa.DataType,
) -> tuple[pa.DataType, pl.DataType, FromConv] | None:
    """Walk an arrow type, returning a cast target, polars dtype, and expression converter.

    The cast target replaces ``map<>`` with ``list<struct<key, value>>``, preserving
    nulls that polars would otherwise drop. The polars dtype includes :class:`Map`
    extension types. The converter tags converted columns with Map. Returns None if
    no map types are present.
    """
    # If we don't need to convert, we fall through to returning None
    if pa.types.is_map(arrow_type):
        key_walk = _from_arrow_walk(arrow_type.key_type)
        val_walk = _from_arrow_walk(arrow_type.item_type)

        key_type, key_dtype, key_conv = key_walk or (
            arrow_type.key_type,
            _arrow_leaf_dtype(arrow_type.key_type),
            None,
        )
        val_type, val_dtype, val_conv = val_walk or (
            arrow_type.item_type,
            _arrow_leaf_dtype(arrow_type.item_type),
            None,
        )
        demap = pa.list_(
            pa.struct([pa.field("key", key_type), pa.field("value", val_type)])
        )

        map_dtype = Map(key_dtype, val_dtype)
        if key_conv or val_conv:
            key_expr = pl.element().struct["key"]
            val_expr = pl.element().struct["value"]
            if key_conv:
                key_expr = key_conv(key_expr)
            if val_conv:
                val_expr = val_conv(val_expr)
            return (
                demap,
                map_dtype,
                partial(
                    _apply_map_nested,
                    pl.struct(key=key_expr, value=val_expr),
                    map_dtype,
                ),
            )
        else:
            return demap, map_dtype, partial(_apply_map_simple, map_dtype)

    elif pa.types.is_struct(arrow_type):
        fields: list[Field] = []
        field_convs: dict[str, FromConv] = {}
        polars_fields: dict[str, pl.DataType] = {}
        for field in arrow_type.fields:
            if walk := _from_arrow_walk(field.type):
                walk_type, walk_dtype, walk_conv = walk
                fields.append(pa.field(field.name, walk_type))
                field_convs[field.name] = walk_conv
                polars_fields[field.name] = walk_dtype
            else:
                fields.append(field)
                polars_fields[field.name] = _arrow_leaf_dtype(field.type)
        if field_convs:
            return (
                pa.struct(fields),
                pl.Struct(polars_fields),
                partial(_apply_struct, field_convs),
            )

    elif pa.types.is_list(arrow_type):
        if inner := _from_arrow_walk(arrow_type.value_type):
            inner_type, inner_dtype, inner_conv = inner
            return (
                pa.list_(inner_type),
                pl.List(inner_dtype),
                partial(_apply_list, inner_conv(pl.element())),
            )

    elif pa.types.is_large_list(arrow_type):
        if inner := _from_arrow_walk(arrow_type.value_type):
            inner_type, inner_dtype, inner_conv = inner
            return (
                pa.large_list(inner_type),
                pl.List(inner_dtype),
                partial(_apply_list, inner_conv(pl.element())),
            )

    elif pa.types.is_fixed_size_list(arrow_type):
        if inner := _from_arrow_walk(arrow_type.value_type):
            inner_type, inner_dtype, inner_conv = inner
            return (
                pa.list_(inner_type, arrow_type.list_size),
                pl.Array(inner_dtype, arrow_type.list_size),
                partial(_apply_array, inner_conv(pl.element()), arrow_type.list_size),
            )


def from_arrow(data: pa.Table | pa.RecordBatch) -> pl.DataFrame:
    """Convert an Arrow table or record batch to a Polars DataFrame, preserving map types.

    Arrow ``map<K, V>`` columns are tagged with the :class:`Map` extension type.
    Nested maps (inside struct, list, array, or other maps) are handled recursively.
    """
    if isinstance(data, pa.RecordBatch):
        data = pa.Table.from_batches([data])
    conversions: list[pl.Expr] = []
    for i, (col, field) in enumerate(zip(data.columns, data.schema, strict=True)):
        if walk := _from_arrow_walk(field.type):
            cast_type, _, conv = walk
            # Cast map columns to list<struct> in arrow to preserve nulls,
            # since polars from_arrow drops nulls when converting map types.
            data = data.set_column(i, field.name, col.cast(cast_type))
            conversions.append(conv(pl.col(field.name)).alias(field.name))

    frame = pl.from_arrow(data)
    assert isinstance(frame, pl.DataFrame)
    if conversions:
        return frame.with_columns(conversions)
    else:
        return frame


def from_arrow_array(data: Array | ChunkedArray) -> pl.Series:
    """Convert an Arrow array to a Polars Series, preserving map types.

    Arrow ``map<K, V>`` arrays are tagged with the :class:`Map` extension type.
    Nested maps (inside struct, list, array, or other maps) are handled recursively.
    """
    if walk := _from_arrow_walk(data.type):
        cast_type, _, conv = walk
        ser = pl.from_arrow(data.cast(cast_type))
        assert isinstance(ser, pl.Series)
        # NOTE typed for expression, but works for series
        return conv(ser)  # pyright: ignore[reportArgumentType, reportReturnType]
    else:
        ser = pl.from_arrow(data)
        assert isinstance(ser, pl.Series)
        return ser


def _null_mask(arr: Array) -> Array | None:
    """Return null mask if array has nulls, else None."""
    return arr.is_null() if arr.null_count > 0 else None


def _to_map(key_conv: ToConv | None, val_conv: ToConv | None, arr: Array) -> Array:
    """Convert a list<struct<key, value>> array to a MapArray."""
    # Arrow maps only use int32 offsets, drop if/when LargeMap is added
    if isinstance(arr, pa.LargeListArray):
        arr = arr.cast(pa.list_(arr.type.value_type))
    structs = arr.values  # pyright: ignore[reportAttributeAccessIssue]
    keys = structs.field("key")
    values = structs.field("value")
    if key_conv:
        keys = key_conv(keys)
    if val_conv:
        values = val_conv(values)
    mask = _null_mask(arr)
    return pa.MapArray.from_arrays(arr.offsets, keys, values, mask=mask)  # pyright: ignore[reportAttributeAccessIssue]


def _to_struct(field_convs: Mapping[str, ToConv], arr: Array) -> Array:
    """Convert struct fields containing Maps to arrow map types."""
    field_arrays: list[Array] = []
    field_names: list[str] = []
    for i in range(arr.type.num_fields):
        name = arr.type.field(i).name
        child = arr.field(i)  # pyright: ignore[reportAttributeAccessIssue]
        if conv := field_convs.get(name):
            child = conv(child)
        field_arrays.append(child)
        field_names.append(name)
    mask = _null_mask(arr)
    return pa.StructArray.from_arrays(field_arrays, names=field_names, mask=mask)


def _to_list(inner_conv: ToConv, arr: Array) -> Array:
    """Convert list or large_list elements containing Maps."""
    converted = inner_conv(arr.values)  # pyright: ignore[reportAttributeAccessIssue]
    mask = _null_mask(arr)
    match arr:
        case pa.LargeListArray():
            return pa.LargeListArray.from_arrays(arr.offsets, converted, mask=mask)  # pyright: ignore[reportAttributeAccessIssue]
        case pa.ListArray():  # pragma: no cover
            return pa.ListArray.from_arrays(arr.offsets, converted, mask=mask)  # pyright: ignore[reportAttributeAccessIssue]
        case _:  # pragma: no cover
            raise TypeError(f"Expected list or large_list array, got {arr.type}")


def _to_array(inner_conv: ToConv, list_size: int, arr: Array) -> Array:
    """Convert fixed_size_list elements containing Maps."""
    converted = inner_conv(arr.values)  # pyright: ignore[reportAttributeAccessIssue]
    mask = _null_mask(arr)
    return pa.FixedSizeListArray.from_arrays(converted, list_size, mask=mask)


@lru_cache
def _to_conv(dtype: pl.DataType | type[pl.DataType]) -> ToConv | None:  # noqa: PLR0911
    """Build a function that converts an arrow array's list<struct> to map<>.

    Returns None if no Map types are present.
    """
    match dtype:
        case Map():
            key_conv = _to_conv(dtype.key)
            val_conv = _to_conv(dtype.value)
            return partial(_to_map, key_conv, val_conv)
        case pl.Struct():
            field_convs: dict[str, ToConv] = {}
            for field in dtype.fields:
                if conv := _to_conv(field.dtype):  # pyright: ignore[reportArgumentType]
                    field_convs[field.name] = conv
            if field_convs:
                return partial(_to_struct, field_convs)
        case pl.Array():
            if inner_conv := _to_conv(dtype.inner):  # type: ignore[arg-type]
                return partial(_to_array, inner_conv, dtype.size)
        case pl.List():
            if inner_conv := _to_conv(dtype.inner):  # type: ignore[arg-type]
                return partial(_to_list, inner_conv)
        case _:
            pass


def to_arrow(frame: pl.DataFrame) -> pa.Table:
    """Convert a Polars DataFrame to an Arrow table, converting Map columns to arrow maps.

    :class:`Map` extension type columns are converted to arrow ``map<K, V>`` arrays.
    Nested maps (inside struct, list, or other maps) are handled recursively.
    """
    table: pa.Table = frame.to_arrow()
    for i, (col, (name, dtype)) in enumerate(
        zip(table.columns, frame.schema.items(), strict=True)
    ):
        if conv := _to_conv(dtype):
            chunks = [conv(chunk) for chunk in col.chunks]
            table = table.set_column(i, name, pa.chunked_array(chunks))
    return table


def to_arrow_array(ser: pl.Series) -> Array:
    """Convert a Polars Series to an Arrow array, converting Map columns to arrow maps.

    :class:`Map` extension type columns are converted to arrow ``map<K, V>`` arrays.
    Nested maps (inside struct, list, or other maps) are handled recursively.
    """
    arr: Array = ser.to_arrow()
    if conv := _to_conv(ser.dtype):
        return conv(arr)
    else:
        return arr


def scan_arrow(
    source: Callable[[], Iterable[pa.Table | pa.RecordBatch]],
) -> pl.LazyFrame:
    """Create a LazyFrame from an Arrow source, preserving map types."""
    schema: pl.Schema | None = None
    lazy: Iterator[pl.DataFrame] | None = None

    def polars_source() -> Iterator[pl.DataFrame]:
        return (from_arrow(batch) for batch in source())

    def schema_fn() -> pl.Schema:
        nonlocal schema, lazy
        if schema is not None:
            return schema

        it = polars_source()
        first = next(it)
        lazy = itertools.chain([first], it)
        schema = first.schema
        return schema

    def io_source(
        with_columns: list[str] | None,
        predicate: pl.Expr | None,
        n_rows: int | None,
        batch_size: int | None,
    ) -> Iterator[pl.DataFrame]:
        nonlocal lazy
        if lazy is None:
            it = polars_source()
        else:
            it = lazy
            lazy = None

        for batch in it:
            frame = batch
            if with_columns is not None:
                frame = frame.select(with_columns)
            if predicate is not None:
                frame = frame.filter(predicate)
            if n_rows is not None:
                frame = frame.head(n_rows)
                n_rows -= len(frame)
            yield frame
            if n_rows is not None and n_rows <= 0:
                break

    return register_io_source(io_source, schema=schema_fn)
