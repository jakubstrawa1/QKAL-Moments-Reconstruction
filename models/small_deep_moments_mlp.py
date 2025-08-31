import torch
import torch.nn as nn

from utils.config import ReconstructionConfig


class SmallDeepMomentsMLP(nn.Module):
    """
    Small but deeper MLP for Legendre moment prediction.
    3 hidden layers with ReLU activations.
    """
    def __init__(self, config: ReconstructionConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.degree + 1)  # output moments
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
