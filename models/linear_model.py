import torch

from utils.config import ReconstructionConfig


class LinearMomentsTorch(torch.nn.Module):
    def __init__(self, config: ReconstructionConfig):
        super().__init__()
        self.fc = torch.nn.Linear(config.input_dim, config.degree + 1, bias=True)  # +1 because degree includes P0
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)  # (B, degree+1)
