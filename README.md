# polars-map

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

## Usage

```python
import polars as pl
from polars_map import Map

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
df.select(pl.col("m").map.keys())        # [["a", "b"], ["x"]]
df.select(pl.col("m").map.values())       # [[1, 2], [10]]
df.select(pl.col("m").map.len())          # [2, 1]

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
```

## Caveats

- **Extension types** — used to wrap the underlying `List(Struct)` storage with a semantic `Map` dtype is not yet stabilized and may change across Polars releases.
- **`pl.dtype_of`** — used to efficiently cast to the extension type after _some_ operations is also unstable.
- **GIL** - is required to automatically wrap an expression as the extension type, and so operations which could change the underlying key or value types will briefly lock the GIL to do the cast. This may also prevent the polars engine from reasoning about the type.
