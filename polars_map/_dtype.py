"""Map extension data type for Polars."""

from __future__ import annotations

import polars as pl


def _ensure_instance(dt: pl.DataType | type[pl.DataType]) -> pl.DataType:
    return dt() if isinstance(dt, type) else dt


class Map(pl.BaseExtension):
    """Map extension type backed by List(Struct({key, value})).

    Usage as a dtype for Series construction::

        dtype = Map(pl.String(), pl.Int64())
    """

    def __init__(self, key: pl.DataType, value: pl.DataType) -> None:
        storage = pl.List(pl.Struct({"key": key, "value": value}))
        super().__init__("polars_map.map", storage)

    @property
    def key(self) -> pl.DataType:
        """Key data type."""
        [key, _] = self.ext_storage().inner.fields  # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType,reportUnknownVariableType]
        return _ensure_instance(key.dtype)  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType,reportUnknownArgumentType]

    @property
    def value(self) -> pl.DataType:
        """Value data type."""
        [_, value] = self.ext_storage().inner.fields  # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType,reportUnknownVariableType]
        return _ensure_instance(value.dtype)  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType,reportUnknownArgumentType]

    def _string_repr(self) -> str:
        return f"map[{self.key._string_repr()},{self.value._string_repr()}]"  # pyright: ignore[reportUnknownMemberType]


pl.register_extension_type("polars_map.map", Map)
