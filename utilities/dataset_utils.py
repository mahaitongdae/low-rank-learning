import minari

import torch
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import DataLoader
import numpy as np
from functools import partial


def collate_fn(batch,
               shuffle_trajectories=False,
               trajectory_length=10,
               stride=1,
               batch_size=32):

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
        num_batches = num_windows // B
        N_usable = num_batches * B

        x = x[:N_usable]
        x = x.view(num_batches, B, T, data_dim)
        x = x.permute(0, 2, 1, 3)
        return x

    return {
        "id":
        torch.Tensor([x.id for x in batch]),
        "observations":
        torch.nn.utils.rnn.pad_sequence(
            [map_fn(x.observations[:-1]) for x in batch], batch_first=True),
        "actions":
        torch.nn.utils.rnn.pad_sequence([map_fn(x.actions) for x in batch],
                                        batch_first=True),
        "next_observations":
        torch.nn.utils.rnn.pad_sequence(
            [map_fn(x.observations[1:]) for x in batch], batch_first=True),
        "rewards":
        torch.nn.utils.rnn.pad_sequence([map_fn(x.rewards) for x in batch],
                                        batch_first=True),
        "terminations":
        torch.nn.utils.rnn.pad_sequence(
            [map_fn(x.terminations) for x in batch], batch_first=True),
        "truncations":
        torch.nn.utils.rnn.pad_sequence([map_fn(x.truncations) for x in batch],
                                        batch_first=True)
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

if __name__ == '__main__':
    test_get_dataset()
