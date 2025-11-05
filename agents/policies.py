import os
import re
import torch
import math
from typing import Optional
from torch import nn
from torch import distributions as pyd


def build_mlp(input_dim: int, hidden_dim: int, output_dim: int, hidden_depth: int) -> nn.Sequential:
    if hidden_depth == 0:
        return nn.Sequential(nn.Linear(input_dim, output_dim))
    layers = [nn.Linear(input_dim, hidden_dim), nn.ELU(inplace=True)]
    for _ in range(hidden_depth - 1):
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.ELU(inplace=True)]
    layers += [nn.Linear(hidden_dim, output_dim)]
    return nn.Sequential(*layers)



# Lightweight squashed Gaussian policy (self-contained)
class TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        return 2.0 * (math.log(2.0) - x - nn.functional.softplus(-2.0 * x))

class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale
        base_dist = pyd.Normal(loc, scale)
        transforms = [TanhTransform()]
        super().__init__(base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu


class DiagGaussianPolicy(nn.Module):

    def __init__(self,
                 obs_dim: int,
                 action_dim: int,
                 hidden_dim: int,
                 hidden_depth: int,
                 log_std_bounds: tuple[float, float] = (-5.0, 2.0)):
        super().__init__()
        self.log_std_bounds = log_std_bounds
        self.trunk = build_mlp(obs_dim, hidden_dim, 2 * action_dim,
                               hidden_depth)

    def forward(self,
                obs: torch.Tensor) -> SquashedNormal:
        mu, log_std = self.trunk(obs).chunk(2, dim=-1)
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std +
                                                                     1.0)
        std = log_std.exp()
        return SquashedNormal(mu, std)

    def load(self, path: str, epoch: Optional[int] = None):
        """Load the policy from a checkpoint named as policy_{epoch}.pth. if 
        epoch is not specified, load the latest checkpoint.
        """
        if epoch is not None:
            assert os.path.exists(
                os.path.join(path, f'policy_{epoch}.pth')
            ), f"Policy checkpoint not found: {os.path.join(path, f'policy_{epoch}.pth')}"
            path = os.path.join(path, f'policy_{epoch}.pth')
        else:
            # 2. Define the regex pattern
            #    r'...'  -> Raw string, treats backslashes as literal
            #    ^        -> Asserts position at the start of the string
            #    policy_  -> Matches the literal text "policy_"
            #    \d+      -> Matches one or more digits (0-9)
            #    \.       -> Matches a literal dot (must be escaped)
            #    pth      -> Matches the literal text "pth"
            #    $        -> Asserts position at the end of the string
            pattern = r'^policy_\d+\.pth$'
            all_files = os.listdir(path)
            matched_files = [f for f in all_files if re.fullmatch(pattern, f)]
            if len(matched_files) == 0:
                raise FileNotFoundError(
                    f"No policy checkpoint found in {path}")
            path = os.path.join(path, sorted(matched_files)[-1])
        print(f"Loading policy from {path}")
        self.load_state_dict(torch.load(path))
        self.eval()
