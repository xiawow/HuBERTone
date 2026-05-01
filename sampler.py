from __future__ import annotations

import random
from collections import defaultdict
from typing import Iterator

from torch.utils.data import Sampler


class PKBatchSampler(Sampler[list[int]]):
    """PK Sampling: P classes (音色版本), K samples per class per batch.

    Ensures each batch contains multiple versions with multiple samples each,
    enabling effective contrastive learning.
    """

    def __init__(
        self,
        version_labels: list[str],
        P: int = 8,
        K: int = 4,
        seed: int = 42,
    ) -> None:
        self.P = P
        self.K = K
        self.seed = seed

        self.version_to_indices: dict[str, list[int]] = defaultdict(list)
        for idx, version in enumerate(version_labels):
            self.version_to_indices[version].append(idx)

        self.valid_versions = [
            v for v, indices in self.version_to_indices.items() if len(indices) >= K
        ]
        if len(self.valid_versions) < P:
            raise ValueError(
                f"Not enough versions with >= {K} samples. "
                f"Got {len(self.valid_versions)} valid versions, need >= {P}."
            )

        self._num_batches = len(self.valid_versions) // P
        if len(self.valid_versions) % P != 0:
            self._num_batches += 1

    def __len__(self) -> int:
        return self._num_batches

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed)
        all_versions = list(self.valid_versions)
        rng.shuffle(all_versions)

        batch: list[int] = []
        for version in all_versions:
            indices = self.version_to_indices[version]
            sampled = rng.sample(indices, min(self.K, len(indices)))
            batch.extend(sampled)

            if len(batch) >= self.P * self.K:
                yield batch[:self.P * self.K]
                batch = batch[self.P * self.K:]

        if batch:
            yield batch

    def set_epoch(self, epoch: int) -> None:
        self.seed = self.seed + epoch
