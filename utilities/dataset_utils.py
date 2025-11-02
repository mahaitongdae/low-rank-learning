import minari

import torch
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import DataLoader
import numpy as np
from functools import partial
from typing import List
from torch.utils.data import Sampler
import random


class ConcatMinariDataset:
    """Concatenate multiple Minari datasets as a single indexable dataset.

    This provides a light wrapper exposing __len__/__getitem__ so it can be
    consumed by PyTorch DataLoader and by our batching utilities unchanged.
    Assumes all provided datasets come from the same environment spec.
    """

    def __init__(self, datasets: List):
        if not datasets:
            raise ValueError("ConcatMinariDataset requires at least one dataset")
        self.datasets: List = list(datasets)
        # Precompute prefix sums of lengths for O(log N) indexing
        lengths = [len(d) for d in self.datasets]
        self._cum_lengths = np.cumsum([0] + lengths)

    def __len__(self) -> int:
        return int(self._cum_lengths[-1])

    def __getitem__(self, index: int):
        # Support negative indices
        if index < 0:
            index = len(self) + index
        if index < 0 or index >= len(self):
            raise IndexError("index out of range")
        # Find which dataset this index falls into
        # searchsorted over [len0, len0+len1, ...]
        ds_idx = int(np.searchsorted(self._cum_lengths[1:], index, side='right'))
        base = int(self._cum_lengths[ds_idx])
        local_idx = index - base
        return self.datasets[ds_idx][local_idx]


class RandomInterleavedConcatSampler(Sampler):
    """Randomly interleave indices from each sub-dataset without replacement.

    Ensures samples are well-mixed across datasets in a single epoch instead of
    consuming one dataset then the next.
    """

    def __init__(self, concat_dataset: ConcatMinariDataset, seed: int | None = None):
        self.concat_dataset = concat_dataset
        self._lengths = [len(d) for d in concat_dataset.datasets]
        self._bases = list(concat_dataset._cum_lengths[:-1])
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return sum(self._lengths)

    def __iter__(self):
        # Build per-dataset shuffled local indices
        per_ds_indices = []
        for n in self._lengths:
            idxs = list(range(n))
            self._rng.shuffle(idxs)
            per_ds_indices.append(idxs)

        # Track current positions
        positions = [0] * len(self._lengths)
        active = [i for i, n in enumerate(self._lengths) if n > 0]

        while active:
            # Pick a dataset uniformly at random among those with remaining items
            ds_idx = self._rng.choice(active)
            pos = positions[ds_idx]
            local_idx = per_ds_indices[ds_idx][pos]
            positions[ds_idx] += 1
            if positions[ds_idx] >= self._lengths[ds_idx]:
                # Remove exhausted dataset
                active.remove(ds_idx)
            yield self._bases[ds_idx] + local_idx
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple


class Normalizer:
    """Applies per-dimension standardization using provided mean/std.

    Works with numpy arrays and torch tensors. Broadcasts across all leading
    dimensions and normalizes the last dimension.
    """

    def __init__(self, mean: np.ndarray, std: np.ndarray, eps: float = 1e-6):
        mean = np.asarray(mean)
        std = np.asarray(std)
        self.mean = mean.astype(np.float32)
        self.std = (std + eps).astype(np.float32)
        self.eps = eps

    def __call__(self, x):
        if isinstance(x, torch.Tensor):
            mean_t = torch.as_tensor(self.mean, dtype=x.dtype, device=x.device)
            std_t = torch.as_tensor(self.std, dtype=x.dtype, device=x.device)
            return (x - mean_t) / std_t
        x_np = np.asarray(x)
        return (x_np - self.mean) / self.std

    def shift_scale(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (shift, scale) such that (x + shift) * scale == standardized x.

        Useful for env wrappers expecting shift/scale.
        """
        shift = -self.mean
        scale = 1.0 / self.std
        return shift, scale


def _accumulate_sums(arr: np.ndarray,
                      running_sum: np.ndarray,
                      running_sumsq: np.ndarray,
                      running_count: int) -> Tuple[np.ndarray, np.ndarray, int]:
    """Accumulate sum, sumsq, and count along the first dimension."""
    if arr.ndim == 1:
        arr = arr[:, None]
    running_sum += arr.sum(axis=0)
    running_sumsq += np.square(arr, dtype=np.float64).sum(axis=0)
    running_count += arr.shape[0]
    return running_sum, running_sumsq, running_count


def compute_stats_over_episodes(dataset,
                                keys: Sequence[str] = ("observations",),
                                num_episodes: Optional[int] = None,
                                ) -> Dict[str, Dict[str, np.ndarray]]:
    """Compute mean/std per-dimension over up to num_episodes for given keys.

    Args:
      dataset: A Minari dataset object (iterable of episodes with numpy fields).
      keys: Sequence of attribute names on each episode to aggregate.
      num_episodes: If provided, limit to the first N episodes.

    Returns:
      Dict mapping key -> { 'mean': np.ndarray, 'std': np.ndarray }.
    """
    # Peek first episode to infer dimensionality for each key
    first_ep = dataset[0]
    dims = {}
    for k in keys:
        v = getattr(first_ep, k)
        dim = v.shape[-1] if v.ndim > 1 else 1
        dims[k] = dim

    sums: Dict[str, np.ndarray] = {k: np.zeros((dims[k],), dtype=np.float64) for k in keys}
    sumsqs: Dict[str, np.ndarray] = {k: np.zeros((dims[k],), dtype=np.float64) for k in keys}
    counts: Dict[str, int] = {k: 0 for k in keys}

    total_eps = len(dataset) if num_episodes is None else min(num_episodes, len(dataset))
    for ep_idx in range(total_eps):
        ep = dataset[ep_idx]
        for k in keys:
            arr = getattr(ep, k)
            sums[k], sumsqs[k], counts[k] = _accumulate_sums(arr, sums[k], sumsqs[k], counts[k])

    stats: Dict[str, Dict[str, np.ndarray]] = {}
    for k in keys:
        count = max(counts[k], 1)
        mean = (sums[k] / count).astype(np.float32)
        var = (sumsqs[k] / count) - np.square(mean, dtype=np.float32)
        var = np.maximum(var, 0.0)
        std = np.sqrt(var, dtype=np.float32)
        stats[k] = {"mean": mean, "std": std}
    return stats


def create_normalizers_from_stats(stats: Mapping[str, Mapping[str, np.ndarray]]) -> Dict[str, Normalizer]:
    """Create Normalizer objects per key from stats dict produced above."""
    normalizers: Dict[str, Normalizer] = {}
    for k, v in stats.items():
        normalizers[k] = Normalizer(mean=v["mean"], std=v["std"])
    return normalizers


def normalize_batch_dict(batch: Mapping[str, torch.Tensor],
                         normalizers: Mapping[str, Normalizer],
                         keys: Sequence[str] = ("observations", "next_observations")) -> Dict[str, torch.Tensor]:
    """Apply normalizers to selected keys of a collated batch in-place-friendly.

    Assumes last dimension is the feature dimension to be standardized. Works for
    tensors of any leading rank.
    """
    out = dict(batch)
    for k in keys:
        if k in out and k in normalizers:
            out[k] = normalizers[k](out[k])
    return out


def collate_fn(batch,
               shuffle_trajectories=False,
               trajectory_length=10,
               stride=1,
               batch_size: int | str = 'auto'):
    T = trajectory_length
    S = stride
    B = batch_size

    def map_fn(x):
        if len(x.shape) == 1:
            x = x[:, None]
        # if shuffle_trajectories:
        #     return torch.as_tensor(np.random.permutation(x))
        # else: # TODO: implement shuffle trajectory
        data_dim = x.shape[-1]
        x = torch.as_tensor(x)
        x = x.unfold(dimension=0, size=T, step=S)
        x = x.permute(0, 2, 1)  # (num_windows, T, dim)
        num_windows = x.shape[0]
        if shuffle_trajectories:
            x = x[torch.randperm(num_windows)]
        x = x.permute(1, 0, 2)  # (T, B, dim)
        if isinstance(batch_size, int) and batch_size < num_windows:
            return x[:, :batch_size, :]
        else:
            return x


    return {
        "id":
        torch.Tensor([x.id for x in batch]),
        "observations":
        # torch.nn.utils.rnn.pad_sequence(
        #     [map_fn(x.observations[:-1]) for x in batch], batch_first=True),
        map_fn(batch[0].observations[:-1]
               ),  # [0] since we only have one Episode in the batch
        "actions":
        # torch.nn.utils.rnn.pad_sequence([map_fn(x.actions) for x in batch],
        #                                 batch_first=True),
        map_fn(batch[0].actions),
        "next_observations":
        # torch.nn.utils.rnn.pad_sequence(
        #     [map_fn(x.observations[1:]) for x in batch], batch_first=True),
        map_fn(batch[0].observations[1:]),
        "rewards":
        # torch.nn.utils.rnn.pad_sequence([map_fn(x.rewards) for x in batch],
        #                                 batch_first=True),
        map_fn(batch[0].rewards),
        # map_fn(batch[0].rewards),
        "terminations":
        # torch.nn.utils.rnn.pad_sequence(
        #     [map_fn(x.terminations) for x in batch], batch_first=True),
        map_fn(batch[0].terminations),
        # map_fn(batch[0].terminations),
        "truncations":
        # torch.nn.utils.rnn.pad_sequence([map_fn(x.truncations) for x in batch],
        #                                 batch_first=True),
        map_fn(batch[0].truncations),
        # map_fn(batch[0].truncations),
    }

def process_dataset_name(dataset_name: str) -> str:
    if dataset_name.startswith('mujoco_'):
        return dataset_name.replace('_', '/') + '-v0'
    else:
        return dataset_name

def test_get_dataset():
    dataset = minari.load_dataset('mujoco/halfcheetah/simple-v0',
                                  download=True)
    dataset_medium = minari.load_dataset('mujoco/halfcheetah/medium-v0',
                                         download=True)
    dataset_expert = minari.load_dataset('mujoco/halfcheetah/expert-v0',
                                         download=True)
    dataset_list = [dataset, dataset_medium, dataset_expert]
    dataset = ConcatMinariDataset(dataset_list)
    print(len(dataset))
    sampler = RandomInterleavedConcatSampler(dataset, seed=0)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=sampler,
        collate_fn=partial(collate_fn, shuffle_trajectories=False, trajectory_length=10, stride=5),
        num_workers=4,
        pin_memory=True
    )

    print(next(iter(dataloader))['observations'].shape)
    print(next(iter(dataloader))['next_observations'].shape)
    print(next(iter(dataloader))['actions'].shape)
    print(next(iter(dataloader))['rewards'].shape)
    print(next(iter(dataloader))['terminations'].shape)
    print(next(iter(dataloader))['truncations'].shape)

if __name__ == '__main__':
    test_get_dataset()
