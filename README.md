# polars-map

[![build](https://github.com/hafaio/polars-map/actions/workflows/build.yml/badge.svg)](https://github.com/hafaio/polars-map/actions/workflows/build.yml)
[![pypi](https://img.shields.io/pypi/v/polars-map)](https://pypi.org/project/polars-map/)

Polars plugin providing a Map extension type and functions.
Maps represent a mapping from unique keys of any type to values, and are stored as `List(Struct({key, value}))` columns.
All function in the `.map` namespace can be used on the extension type or on the
underlying list.

## Installation

```bash
pip install polars-map
```

## Supported operations (`.map.*`)

| Category   | Methods                                                    |
| ---------- | ---------------------------------------------------------- |
| Accessors  | `entries`, `keys`, `values`, `len`, `get`, `contains_key`  |
| Filtering  | `filter`, `filter_keys`, `filter_values`                   |
| Transform  | `eval`, `eval_keys`, `eval_values`                         |
| Set ops    | `merge`, `intersection`, `difference`                      |
| Conversion | `from_entries`                                             |
| Iteration  | `__iter__`, `to_list` (Series only)                        |

## Arrow conversion

| Function                  | Description                                                              |
| ------------------------- | ------------------------------------------------------------------------ |
| `from_arrow(table)`       | Arrow Table/RecordBatch to Polars DataFrame, preserving `map<>` as `Map` |
| `from_arrow_array(array)` | Arrow Array to Polars Series, preserving `map<>` as `Map`                |
| `to_arrow(frame)`         | Polars DataFrame to Arrow Table, converting `Map` back to `map<>`        |
| `to_arrow_array(series)`  | Polars Series to Arrow Array, converting `Map` back to `map<>`           |
| `scan_arrow(source)`      | Lazy scan from an Arrow source with `Map` preservation                   |

## Usage

```python
import polars as pl
import pyarrow as pa
from polars_map import Map, from_arrow, to_arrow, scan_arrow

# Construction
ser = pl.Series(
    "m",
    [
        [{"key": "a", "value": 1}, {"key": "b", "value": 2}],
        [{"key": "x", "value": 10}],
    ],
    dtype=Map(pl.String(), pl.Int64()),
)
df = pl.DataFrame([ser])

# Accessors
df.select(pl.col("m").map.keys())    # [["a", "b"], ["x"]]
df.select(pl.col("m").map.values())  # [[1, 2], [10]]
df.select(pl.col("m").map.len())     # [2, 1]

# Lookup
df.select(pl.col("m").map.get("a"))           # [1, None]
df.select(pl.col("m").map.contains_key("a"))  # [True, False]

# Filtering
df.select(pl.col("m").map.filter(pl.element().struct["value"] > 1))
df.select(pl.col("m").map.filter_keys(pl.element() > "a"))
df.select(pl.col("m").map.filter_values(pl.element() >= 2))

# Transform keys or values
df.select(pl.col("m").map.eval_keys(pl.element().str.to_uppercase()))
df.select(pl.col("m").map.eval_values(pl.element() * 2))

# Merge (right-side wins on key conflict)
left = pl.Series("l", [[{"key": "a", "value": 1}, {"key": "b", "value": 2}]], dtype=Map(pl.String(), pl.Int64()))
right = pl.Series("r", [[{"key": "a", "value": 99}, {"key": "c", "value": 3}]], dtype=Map(pl.String(), pl.Int64()))
pl.DataFrame([left, right]).select(pl.col("l").map.merge(pl.col("r")))
# [{"a": 99, "b": 2, "c": 3}]

# Set operations
pl.DataFrame([left, right]).select(pl.col("l").map.intersection(pl.col("r")))  # keys in both
pl.DataFrame([left, right]).select(pl.col("l").map.difference(pl.col("r")))    # keys only in left

# Convert to/from plain List(Struct)
df.select(pl.col("m").map.entries())        # strip Map type
df.select(pl.col("m").map.from_entries())   # wrap as Map (with deduplication)

# Series iteration yields Python dicts
for d in ser.map:
    print(d)  # {"a": 1, "b": 2}, {"x": 10}

# Arrow table with map column → Polars DataFrame
table = pa.table({"m": pa.array([[("a", 1)]], type=pa.map_(pa.string(), pa.int64()))})
df = from_arrow(table)          # Map(String, Int64) dtype preserved
table2 = to_arrow(df)           # roundtrips back to arrow map<>

# Lazy scanning from an Arrow source
lf = scan_arrow(lambda: [table])
result = lf.collect()
```

## Caveats

- **Extension types** — used to wrap the underlying `List(Struct)` storage with a semantic `Map` dtype, are not yet stabilized and may change across Polars releases.
- **`pl.dtype_of`** — used to efficiently cast to the extension type after _some_ operations is also unstable.
- **GIL** - is required to automatically wrap an expression as the extension type, and so operations which could change the underlying key or value types will briefly lock the GIL to do the cast. This may also prevent the polars engine from reasoning about the type.
- **LongMap** - arrow currently only support Map, not LongMap. Polars generlly uses LongList, but if a frame is every converted to arrow with offsets that don't fit in a u32, this will exproting will error.
