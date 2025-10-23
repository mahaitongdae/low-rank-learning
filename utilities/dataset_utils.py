import minari

import torch
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import DataLoader
import numpy as np
from functools import partial


def collate_fn(batch, shuffle_trajectories=False, truncate_trajectories=-1):
    def map_fn(x):
        # if shuffle_trajectories:
        #     return torch.as_tensor(np.random.permutation(x))
        # else: # TODO: implement shuffle trajectory
        return torch.as_tensor(x)
    return {
        "id": torch.Tensor([x.id for x in batch]),
        "observations": torch.nn.utils.rnn.pad_sequence(
            [map_fn(x.observations[:-1]) for x in batch],
            batch_first=True
        ),
        "actions": torch.nn.utils.rnn.pad_sequence(
            [map_fn(x.actions) for x in batch],
            batch_first=True
        ),
        "next_observations": torch.nn.utils.rnn.pad_sequence(
            [map_fn(x.observations[1:]) for x in batch],
            batch_first=True
        ),
        "rewards": torch.nn.utils.rnn.pad_sequence(
            [map_fn(x.rewards) for x in batch],
            batch_first=True
        ),
        "terminations": torch.nn.utils.rnn.pad_sequence(
            [map_fn(x.terminations) for x in batch],
            batch_first=True
        ),
        "truncations": torch.nn.utils.rnn.pad_sequence(
            [map_fn(x.truncations) for x in batch],
            batch_first=True
        )
    }


def test_get_dataset():
    dataset = minari.load_dataset('mujoco/halfcheetah/simple-v0',
                                  download=True)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=partial(collate_fn, shuffle_trajectories=False),
        num_workers=4,
        pin_memory=True  # Set to True if you are training on a CUDA GPU
    )

    print(next(iter(dataloader))['observations'].shape)
    print(next(iter(dataloader))['next_observations'].shape)

if __name__ == '__main__':
    test_get_dataset()
