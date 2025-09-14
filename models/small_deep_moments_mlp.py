import torch
import torch.nn as nn
from utils.config import ReconstructionConfig

class SmallDeepMomentsMLP(nn.Module):

    def __init__(self, config: ReconstructionConfig):
        super().__init__()
        self.ln0 = nn.LayerNorm(config.input_dim, elementwise_affine=True)
        self.net = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.degree + 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln0(x)
        return self.net(x)
