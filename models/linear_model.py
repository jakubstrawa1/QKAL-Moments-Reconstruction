import torch
import torch.nn as nn
from utils.config import ReconstructionConfig

class LinearMomentsTorch(nn.Module):
    def __init__(self, config: ReconstructionConfig):
        super().__init__()
        self.ln0 = nn.LayerNorm(config.input_dim, elementwise_affine=True)
        self.fc  = nn.Linear(config.input_dim, config.degree + 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln0(x)
        return self.fc(x)
