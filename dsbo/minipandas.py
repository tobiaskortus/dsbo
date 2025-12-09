# minipandas.py
"""
MiniFrame: a minimal, NumPy-backed, pandas-like table.

Each column is a NumPy array sharing the same row length (axis 0).
Columns may be:
  • scalar columns: shape (N,)
  • vector columns: shape (N, M) with fixed M across the frame

Supports familiar pandas patterns:
  • len(mf), mf.shape, mf.keys(), "col" in mf
  • column access: mf["col"] -> ndarray
  • row selection: mf[idx], mf[:N], mf[mask] -> MiniFrame (views where possible)
  • head(), copy(), describe(), to_pandas(), to_pickle()/read_pickle()

Parameters
----------
data : Mapping[str, np.ndarray or scalar]
    Column data. Scalars are broadcast to length N.
length : int, optional (keyword-only)
    Row count N to use when all provided values are scalars. If omitted,
    N is inferred from the first non-scalar column.

Notes
-----
- Only 1D (N,) and 2D (N, M) columns are supported.
- Vector columns are expanded to wide columns (<name>_0..M-1) by to_pandas().
- Properties:
    scalar_cols -> list[str]   # columns with ndim == 1
    vector_cols -> list[str]   # columns with ndim == 2
    vector_length(col) -> int or None

Examples
--------
>>> mf = MiniFrame({"run": 7, "event": np.arange(5), "energy": np.random.rand(5, 4)})
>>> mf[:3]                  # first 3 rows
>>> mf["energy"]            # raw ndarray (5, 4)
>>> mf.describe()           # quick stats
>>> df = mf.to_pandas()     # convert to pandas (energy -> energy_0.._3)
>>> mf.to_pickle("out.pkl")
>>> mf2 = MiniFrame.read_pickle("out.pkl")
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Iterable, Mapping, Optional, Union

ArrayLike = np.ndarray

class MiniFrame:
    """
    Minimal NumPy-backed dataframe-like container.
    Holds {column_name: np.ndarray}; all columns share length on axis 0.
    Supports:
      - len(mf), mf.shape, mf.keys()
      - column access: mf["col"] -> ndarray
      - row slicing/indexing: mf[:N], mf[idx], mf[mask] -> MiniFrame (views where possible)
      - head(), copy(), to_pandas()
    """
    def __init__(self, data: Mapping[str, np.ndarray], *, length: int | None = None):
        if not data:
            raise ValueError("MiniFrame: empty data dict.")
    
        norm: dict[str, np.ndarray] = {}
        col_lengths: set[int] = set()
    
        # First pass: determine N if not provided
        N = length
        if N is None:
            for v in data.values():
                if np.isscalar(v):
                    continue
                arr = np.asarray(v)
                if arr.ndim >= 1:
                    N = len(arr)
                    break
            if N is None:
                raise ValueError("MiniFrame: cannot infer length from only scalars; pass length=...")
    
        # Second pass: normalize and validate
        for k, v in data.items():
            if np.isscalar(v):
                arr = np.full(N, v)  # broadcast scalar -> (N,)
            else:
                arr = np.asarray(v)
                if arr.ndim == 0:
                    arr = np.full(N, arr.item())
                elif arr.ndim not in (1, 2):
                    raise ValueError(f"MiniFrame: column '{k}' has unsupported ndim={arr.ndim} (only 1D or 2D).")
                if len(arr) != N:
                    raise ValueError(f"MiniFrame: column '{k}' length {len(arr)} != expected {N}.")
    
            norm[k] = arr
            col_lengths.add(len(arr))
    
        if len(col_lengths) != 1:
            raise ValueError(f"MiniFrame: inconsistent column lengths: {col_lengths}")
    
        self._data = norm
        self._length = N

    def __len__(self) -> int:
        return self._length
    
    @property
    def scalar_cols(self) -> list[str]:
        return [k for k, v in self._data.items() if v.ndim == 1]
    
    @property
    def vector_cols(self) -> list[str]:
        return [k for k, v in self._data.items() if v.ndim == 2]
    
    def vector_length(self, col: str) -> int | None:
        v = self._data[col]
        return v.shape[1] if v.ndim == 2 else None

    @property
    def shape(self) -> tuple[int, int]:
        return (self._length, len(self._data))

    def keys(self) -> List[str]:
        return list(self._data.keys())

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __repr__(self) -> str:
        cols = ", ".join(self._data.keys())
        return f"<MiniFrame n={self._length}, cols=[{cols}]>"

    # --- access ---
    def __getitem__(self, key: Union[str, int, slice, np.ndarray, List[int]]):
        # column access
        if isinstance(key, str):
            return self._data[key]
        # row selection (returns a new MiniFrame of views where possible)
        idx = key
        view = {k: v[idx] for k, v in self._data.items()}
        return MiniFrame(view)

    def head(self, n: int = 5) -> "MiniFrame":
        return self[: min(n, self._length)]

    def copy(self) -> "MiniFrame":
        return MiniFrame({k: v.copy() for k, v in self._data.items()})

    def to_pandas(self, indiv_cols: bool = True):
        import pandas as pd
        out = {}
        for k, v in self._data.items():
            if v.ndim == 1:
                out[k] = v  # 1D numeric column
            elif v.ndim == 2 and indiv_cols:
                # expand into wide columns
                for i in range(v.shape[1]):
                    out[f"{k}_{i}"] = v[:, i]
            elif v.ndim == 2 and not indiv_cols:
                # legacy nested: one Python list per row (object dtype)
                # NOTE: v.tolist() -> List[List[...]] which most legacy code expects
                out[k] = pd.Series(v.tolist(), dtype=object)
            else:
                raise ValueError(f"Column '{k}' has ndim={v.ndim}, not supported.")
        return pd.DataFrame(out)

    # convenience constructor
    @staticmethod
    def from_scalars_and_arrays(
        scalars: Mapping[str, ArrayLike],
        arrays: Mapping[str, ArrayLike],
    ) -> "MiniFrame":
        return MiniFrame({**scalars, **arrays})
    
    def __repr__(self) -> str:
        # keep a compact technical summary
        cols = ", ".join(self._data.keys())
        return f"<MiniFrame n={self._length}, cols=[{cols}]>"
    
    def __str__(self) -> str:
        """Pretty print like a lightweight pandas.DataFrame."""
        # Number of rows/cols to display
        n_show = min(6, self._length)
        out_lines = []
        header = " | ".join(f"{k}" for k in self._data.keys())
        out_lines.append(header)
        out_lines.append("-" * len(header))
    
        for i in range(n_show):
            row_vals = []
            for v in self._data.values():
                arr = v[i]
                # scalar column
                if v.ndim == 1:
                    row_vals.append(f"{arr!r}")
                # vector column: show as short array summary
                elif v.ndim == 2:
                    # show first 3 elements
                    if v.shape[1] > 3:
                        short = ", ".join(f"{x:.3g}" for x in arr[:3])
                        row_vals.append(f"[{short}, …]")
                    else:
                        short = ", ".join(f"{x:.3g}" for x in arr)
                        row_vals.append(f"[{short}]")
            out_lines.append(" | ".join(row_vals))
    
        if self._length > n_show:
            out_lines.append(f"... ({self._length - n_show} more rows)")
        out_lines.append(f"[{self._length} rows x {len(self._data)} columns]")
        return "\n".join(out_lines)
    
    def describe(self):
        stats = {}
        for k, v in self._data.items():
            if v.ndim == 1:
                stats[k] = {
                    "mean": float(np.mean(v)),
                    "std":  float(np.std(v)),
                    "min":  float(np.min(v)),
                    "max":  float(np.max(v)),
                }
            elif v.ndim == 2:
                stats[k] = {
                    "mean": float(np.mean(v)),
                    "std":  float(np.std(v)),
                    "min":  float(np.min(v)),
                    "max":  float(np.max(v)),
                    "shape": v.shape[1],
                }
        import pandas as pd
        return pd.DataFrame(stats).T
    
    def to_pickle(self, path: str, protocol: int = 4):
        """
        Save the MiniFrame to a pickle file.
        Compatible in spirit with pandas.DataFrame.to_pickle().
        """
        import pickle
        with open(path, "wb") as f:
            pickle.dump(self._data, f, protocol=protocol)
    
    
    @staticmethod
    def read_pickle(path: str) -> "MiniFrame":
        """
        Load a MiniFrame from a pickle file created by to_pickle().
        """
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)
        return MiniFrame(data)
    

# --- pandas-like top-level concat ---

def concat(
    objs: Iterable[MiniFrame],
    axis: int = 0,
    ignore_index: bool = False,
    copy: bool = False,
) -> MiniFrame:
    """
Concatenate MiniFrames along rows (axis=0), pandas-style.

This is a lightweight analogue of pandas.concat for MiniFrame objects.
All frames must have identical column sets and per-column shapes
(except for the row dimension).

Parameters
----------
objs : Iterable[MiniFrame]
    Frames to concatenate.
axis : int, default 0
    Only axis=0 (row-wise) is supported.
ignore_index : bool, default False
    Accepted for API similarity; MiniFrame has no explicit index (no-op).
copy : bool, default False
    If True, return a deep copy of the result.

Returns
-------
MiniFrame
    The row-wise concatenation of the input frames.

Examples
--------
>>> mf = mpd.concat([mf1, mf2])
>>> mf.shape
"""
    objs = list(objs)
    if not objs:
        raise ValueError("minipandas.concat: empty iterable.")

    if axis != 0:
        raise NotImplementedError("minipandas.concat: only axis=0 is supported.")

    if len(objs) == 1:
        return objs[0].copy() if copy else objs[0]

    # check column names and per-column shapes (except first dimension)
    cols = objs[0].keys()
    for f in objs[1:]:
        if f.keys() != cols:
            raise ValueError("minipandas.concat: column mismatch among frames.")
        for c in cols:
            a0, a1 = objs[0]._data[c], f._data[c]
            if a0.ndim != a1.ndim or (a0.ndim == 2 and a0.shape[1] != a1.shape[1]):
                raise ValueError(f"minipandas.concat: shape mismatch in column '{c}'.")

    # concatenate each column along axis 0
    new_data: Dict[str, ArrayLike] = {}
    for c in cols:
        stacks = [mf._data[c] for mf in objs]
        new_data[c] = np.concatenate(stacks, axis=0)

    out = MiniFrame(new_data)

    # ignore_index matches pandas semantics for RangeIndex; MiniFrame has no index,
    # so the flag is accepted for API similarity but is a no-op.
    _ = ignore_index  # no-op

    if copy:
        return out.copy()
    return out