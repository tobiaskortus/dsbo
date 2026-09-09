from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import (
    BatchSampler,
    DataLoader,
    Dataset,
    DistributedSampler,
    RandomSampler,
)

from dsbo.surrogates.easyscale.random import (
    VirtualRankContext,
    initial_context,
    seed_everything,
    seed_scope,
)


@dataclass(frozen=True)
class TaggedSample:
    """One dataset sample tagged with the boundary it belongs to."""

    rank: int
    epoch: int
    batch_position: int
    index: int
    sample: Any


@dataclass(frozen=True)
class VirtualBatch:
    ranks: tuple[int, ...]
    indices: tuple[tuple[int, ...], ...]
    batches: tuple[Any, ...]

    def items(self):
        return zip(self.ranks, self.batches)


class VirtualDataset(Dataset):
    def __init__(self, dataset: Dataset, seed: int):
        self.dataset = dataset
        self.seed = seed

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, key: tuple[int, int, int, int, int]):
        rank, epoch, batch_position, sample_offset, index = key
        seed_sequence = np.random.SeedSequence([self.seed, *key])
        sample_seed = int(seed_sequence.generate_state(1, dtype=np.uint32)[0])

        with seed_scope(sample_seed, torch.device("cpu")):
            sample = self.dataset[index]

        return TaggedSample(rank, epoch, batch_position, index, sample)


class SeedWorker:
    def __init__(self, worker_init_fn: Callable[[int], None] | None):
        self.worker_init_fn = worker_init_fn

    def __call__(self, worker_id: int) -> None:
        worker_seed = torch.initial_seed() % 2**32
        seed_everything(worker_seed)
        if self.worker_init_fn is not None:
            self.worker_init_fn(worker_id)


class CollateTagged:
    def __init__(self, collate_fn: Callable):
        self.collate_fn = collate_fn

    def __call__(self, samples: Sequence[TaggedSample]):
        first = samples[0]
        return (
            first.rank,
            first.epoch,
            first.batch_position,
            tuple(sample.index for sample in samples),
            self.collate_fn([sample.sample for sample in samples]),
        )


class VirtualBatchSampler:
    def __init__(
        self,
        dataset: Dataset,
        virtual_world_size: int,
        assigned_ranks: Sequence[int],
        batch_size: int,
        seed: int,
        drop_last: bool,
        shuffle: bool,
        contexts: Mapping[int, VirtualRankContext],
    ) -> None:
        self.seed = seed
        self.shuffle = shuffle
        self.dataset = dataset
        self.drop_last = drop_last
        self.batch_size = batch_size
        self.contexts = dict(contexts)
        self.virtual_world_size = virtual_world_size
        self.assigned_ranks = tuple(sorted(assigned_ranks))
        self.distributed_samplers = {
            rank: DistributedSampler(
                self.dataset,
                num_replicas=self.virtual_world_size,
                rank=rank,
                shuffle=self.shuffle,
                seed=self.seed,
                drop_last=self.drop_last,
            )
            for rank in self.assigned_ranks
        }

    def reconfigure(
        self,
        assigned_ranks: Sequence[int],
        contexts: Mapping[int, VirtualRankContext],
    ) -> None:
        self.contexts = dict(contexts)
        self.assigned_ranks = tuple(sorted(assigned_ranks))
        self.distributed_samplers = {
            rank: DistributedSampler(
                self.dataset,
                num_replicas=self.virtual_world_size,
                rank=rank,
                shuffle=self.shuffle,
                seed=self.seed,
                drop_last=self.drop_last,
            )
            for rank in self.assigned_ranks
        }

    def _rank_batches(self, rank: int) -> BatchSampler:
        context = self.contexts[rank]
        sampler = self.distributed_samplers[rank]
        sampler.set_epoch(context.sampler_epoch)
        return BatchSampler(sampler, self.batch_size, self.drop_last)

    def __iter__(self):
        positions = {self.contexts[rank].sampler_position for rank in self.assigned_ranks}
        epochs = {self.contexts[rank].sampler_epoch for rank in self.assigned_ranks}

        if len(positions) != 1 or len(epochs) != 1:
            raise RuntimeError("assigned EST sampler contexts are not at one boundary.")
        
        iterators = []
        epoch = epochs.pop()
        position = positions.pop()

        for rank in self.assigned_ranks:
            iterator = iter(self._rank_batches(rank))
            for _ in range(position):
                try:
                    next(iterator)
                except StopIteration as exc:
                    raise RuntimeError("EST sampler position exceeds the epoch.") from exc
                
            iterators.append(iterator)
            
        
        for batch_position, rank_batches in enumerate(zip(*iterators), start=position):
            for rank, indices in zip(self.assigned_ranks, rank_batches):
                yield [
                    (rank, epoch, batch_position, sample_offset, index)
                    for sample_offset, index in enumerate(indices)
                ]

    def __len__(self) -> int:
        return self.logical_length() * len(self.assigned_ranks)

    def logical_length(self) -> int:
        if not self.assigned_ranks:
            return 0
        total = len(self._rank_batches(self.assigned_ranks[0]))
        return max(0, total - self.contexts[self.assigned_ranks[0]].sampler_position)

    def commit(self, epoch: int, batch_position: int) -> None:
        total = len(self._rank_batches(self.assigned_ranks[0]))
        expected = (epoch, batch_position)
        if any(
            (context.sampler_epoch, context.sampler_position) != expected
            for context in self.contexts.values()
        ):
            raise RuntimeError("EST sampler contexts do not match the consumed batch.")

        for context in self.contexts.values():
            context.sampler_position += 1
            if context.sampler_position == total:
                context.sampler_epoch += 1
                context.sampler_position = 0


class VirtualDataLoader(DataLoader):
    def __init__(self, *args, assigned_ranks: Sequence[int], **kwargs) -> None:
        self.assigned_ranks = tuple(sorted(assigned_ranks))
        self._parked_iterator = None
        super().__init__(*args, **kwargs)

    def __len__(self) -> int:
        return self.batch_sampler.logical_length()

    def __iter__(self):
        prepared = super().__iter__()
        while True:
            group = []
            for _ in self.assigned_ranks:
                try:
                    group.append(next(prepared))
                except StopIteration:
                    if group:
                        raise RuntimeError("data workers returned an incomplete EST group.")
                    return

            ranks = tuple(item[0] for item in group)
            boundaries = {(item[1], item[2]) for item in group}
            if ranks != self.assigned_ranks or len(boundaries) != 1:
                raise RuntimeError("data workers returned EST batches out of order.")

            epoch, batch_position = boundaries.pop()
            self.batch_sampler.commit(epoch, batch_position)
            yield VirtualBatch(
                ranks=ranks,
                indices=tuple(item[3] for item in group),
                batches=tuple(item[4] for item in group),
            )

    def reconfigure(
        self,
        assigned_ranks: Sequence[int],
        contexts: Mapping[int, VirtualRankContext],
    ) -> None:
        if self._iterator is not None:
            raise RuntimeError("cannot reconfigure an active virtual dataloader.")
        self.assigned_ranks = tuple(sorted(assigned_ranks))
        self.batch_sampler.reconfigure(self.assigned_ranks, contexts)

    def park_workers(self) -> bool:
        if not self.persistent_workers or self.num_workers == 0:
            return False
        if self._iterator is not None:
            self._parked_iterator = self._iterator
            self._iterator = None
        return self._parked_iterator is not None

    def restore_workers(self) -> None:
        if self._parked_iterator is not None:
            self._iterator = self._parked_iterator
            self._parked_iterator = None

    def close_workers(self) -> None:
        iterators = (self._iterator, self._parked_iterator)
        self._iterator = None
        self._parked_iterator = None
        for iterator in iterators:
            if iterator is not None and hasattr(iterator, "_shutdown_workers"):
                iterator._shutdown_workers()


def build_virtual_dataloader(
    loader: DataLoader,
    virtual_world_size: int,
    seed: int,
    assigned_ranks: Sequence[int] | None = None,
    contexts: Sequence[VirtualRankContext] | Mapping[int, VirtualRankContext] | None = None,
) -> DataLoader:
    
    assigned = tuple(range(virtual_world_size) if assigned_ranks is None else assigned_ranks)
    if contexts is None:
        context_by_rank = {
            rank: initial_context(rank, seed) for rank in range(virtual_world_size)
        }
    elif isinstance(contexts, Mapping):
        context_by_rank = dict(contexts)
    else:
        context_by_rank = {context.rank: context for context in contexts}

    batch_sampler = VirtualBatchSampler(
        dataset=loader.dataset,
        virtual_world_size=virtual_world_size,
        assigned_ranks=assigned,
        batch_size=loader.batch_size,
        seed=seed,
        drop_last=loader.drop_last,
        shuffle=isinstance(loader.sampler, RandomSampler),
        contexts={rank: context_by_rank[rank] for rank in assigned},
    )

    loader_options = {
        "batch_sampler": batch_sampler,
        "collate_fn": CollateTagged(loader.collate_fn),
        "num_workers": loader.num_workers,
        "worker_init_fn": SeedWorker(loader.worker_init_fn),
        "generator": torch.Generator().manual_seed(seed),
        "persistent_workers": loader.persistent_workers,
        "pin_memory": loader.pin_memory,
        "pin_memory_device": getattr(loader, "pin_memory_device", ""),
        "timeout": loader.timeout,
    }

    if loader.num_workers:
        loader_options["prefetch_factor"] = loader.prefetch_factor
        loader_options["multiprocessing_context"] = loader.multiprocessing_context
    
    if hasattr(loader, "in_order"):
        loader_options["in_order"] = True

    return VirtualDataLoader(
        VirtualDataset(loader.dataset, seed),
        assigned_ranks=assigned,
        **loader_options,
    )
