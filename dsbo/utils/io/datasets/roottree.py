from __future__ import annotations

from bisect import bisect_right
from typing import Callable

import numpy as np
import uproot


class PartitionedROOTTTreeReader:
    def __init__(self, fpath: str):
        from dsbo.utils.io.storage.root import ROOTFile

        ROOTFile.validate(fpath)
        self.metadata, sources = ROOTFile._load_container(fpath)
        self._sources = [fpath] if sources is None else ROOTFile._resolved_sources(fpath, sources)
        self._files = {}

    def partition_sizes(
        self,
        dataset_name: str,
        stored: list[int] | tuple[int, ...] | None,
        total_size: int,
        row_axis: int = 0,
    ) -> tuple[int, ...]:
        from dsbo.utils.io.storage.root import ROOTFile

        if stored is not None:
            return tuple(stored)
        if len(self._sources) == 1:
            return (total_size,)
        sizes = []
        for source in self._sources:
            shape = ROOTFile._load_container(source)[0]["datasets"][dataset_name]["shape"]
            sizes.append(1 if len(shape) <= row_axis else shape[row_axis])
        return tuple(sizes)

    def read_values(
        self,
        partitions: tuple[int, ...],
        tree_name: str,
        branch: str,
        index: int | slice | list[int] | np.ndarray,
        row_shape: tuple[int, ...],
        dtype: np.dtype,
    ) -> np.ndarray:
        return self.read(
            partitions, tree_name, branch, index, row_shape, dtype,
            lambda tree, start, stop: self._decode(tree, branch, start, stop),
        )

    def read_columns(
        self,
        partitions: tuple[int, ...],
        tree_name: str,
        branches: list[str],
        index: int | slice | list[int] | np.ndarray,
        row_shape: tuple[int, ...],
        dtype: np.dtype,
    ) -> np.ndarray:
        def decode(tree, start: int, stop: int) -> np.ndarray:
            return np.stack(
                [self._decode(tree, branch, start, stop) for branch in branches],
                axis=-1,
            )

        return self.read(partitions, tree_name, branches[0], index, row_shape, dtype, decode)

    def read_keyed(
        self,
        partitions: tuple[int, ...],
        tree_name: str,
        branches: list[str],
        value_branch: str,
        single_branch: bool,
        column_indices: list[int],
        n_selected: int,
        index: int | slice | list[int] | np.ndarray,
        dtype: np.dtype,
    ) -> np.ndarray:
        def decode(tree, start: int, stop: int) -> np.ndarray:
            if n_selected == 0:
                return np.empty((stop - start, 0), dtype=dtype)
            if single_branch:
                values = self._decode(tree, value_branch, start, stop)
                return values if column_indices is None else values[..., column_indices]
            values = [self._decode(tree, branch, start, stop) for branch in branches]
            return np.column_stack(values)

        representative = (
            value_branch
            if single_branch or not branches
            else branches[0]
        )
        return self.read(
            partitions, tree_name, representative, index,
            (n_selected,), dtype, decode,
        )

    def _decode(self, tree, branch: str, start: int, stop: int) -> np.ndarray:
        return np.asarray(tree[branch].array(entry_start=start, entry_stop=stop, library="np"))

    def read(
        self,
        partitions: tuple[int, ...],
        tree_name: str,
        basket_branch: str,
        index: int | slice | list[int] | np.ndarray,
        row_shape: tuple[int, ...],
        dtype: np.dtype,
        decode: Callable[[object, int, int], np.ndarray],
    ) -> np.ndarray:
        offsets = np.cumsum((0, *partitions), dtype=np.int64)
        size = int(offsets[-1])
        scalar = isinstance(index, (int, np.integer))
        if scalar:
            requested = self._normalize([index], size)
        elif isinstance(index, slice):
            start, stop, step = index.indices(size)
            if step == 1:
                chunks = [
                    self._decode_range(decode, tree, begin, end, row_shape, dtype)
                    for tree, begin, end in self._ranges(offsets, tree_name, start, stop)
                ]
                if not chunks:
                    return np.empty((0, *row_shape), dtype=dtype)
                return chunks[0] if len(chunks) == 1 else np.concatenate(chunks)
            requested = np.arange(start, stop, step, dtype=np.int64)
        else:
            requested = self._normalize(index, size)

        result = np.empty((requested.size, *row_shape), dtype=dtype)
        source_indices = np.searchsorted(offsets[1:], requested, side="right")
        for source_index in np.unique(source_indices):
            positions = np.flatnonzero(source_indices == source_index)
            local = requested[positions] - offsets[source_index]
            tree = self._tree(int(source_index), tree_name)
            basket_offsets = np.asarray(tree[basket_branch].entry_offsets)
            baskets = np.searchsorted(basket_offsets[1:], local, side="right")
            for basket in np.unique(baskets):
                basket_positions = np.flatnonzero(baskets == basket)
                selected = local[basket_positions]
                start, stop = int(selected.min()), int(selected.max()) + 1
                values = self._decode_range(decode, tree, start, stop, row_shape, dtype)
                result[positions[basket_positions]] = values[selected - start]
        return result[0] if scalar else result

    @staticmethod
    def _normalize(indices, size: int) -> np.ndarray:
        requested = np.asarray(indices, dtype=np.int64).reshape(-1)
        requested = np.where(requested < 0, requested + size, requested)
        if np.any((requested < 0) | (requested >= size)):
            raise IndexError(f"ROOT dataset index is out of range for size {size}.")
        return requested

    def _ranges(self, offsets, tree_name: str, start: int, stop: int):
        if start == stop:
            return
        first = max(0, bisect_right(offsets, start) - 1)
        for source_index in range(first, len(self._sources)):
            offset = int(offsets[source_index])
            local_start = max(0, start - offset)
            local_stop = min(int(offsets[source_index + 1]) - offset, stop - offset)
            if local_start < local_stop:
                yield self._tree(source_index, tree_name), local_start, local_stop
            if offsets[source_index + 1] >= stop:
                break

    @staticmethod
    def _decode_range(decode, tree, start, stop, row_shape, dtype):
        return np.asarray(decode(tree, start, stop), dtype=dtype).reshape(
            (stop - start, *row_shape)
        )

    def _tree(self, source_index: int, tree_name: str):
        source = self._sources[source_index]
        if source not in self._files:
            self._files[source] = uproot.open(source, array_cache=None)
        return self._files[source][tree_name]

    def close(self) -> None:
        for file in self._files.values():
            file.close()
        self._files.clear()
