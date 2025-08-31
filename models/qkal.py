import torch, torch.nn as nn
from qkal.layer import QKAL_Layer
from utils.config import ReconstructionConfig


class QKAL(nn.Module):
    """
    Same model architecture as proposed in QKAL paper
    https://arxiv.org/html/2507.13393v1
    but now returning K moments K = degree + 1 (degrees are 1-indexed)
    instead of n classification outputs like in MNIST.
    """
    def __init__(
        self,
        config: ReconstructionConfig,
    ):
        super().__init__()
        self.config = config
        self.degree = self.config.degree
        self.K = self.config.degree + 1  #number of legendre basis

        self.ln0 = nn.LayerNorm(self.config.input_dim, elementwise_affine=self.config.elementwise_affine)

        self.l1 = QKAL_Layer(self.config.input_dim, self.config.hidden_dim, self.config.degree)
        self.ln1 = nn.LayerNorm(self.config.hidden_dim, elementwise_affine=self.config.elementwise_affine)

        self.l2 = QKAL_Layer(self.config.hidden_dim, self.config.hidden_dim, self.config.degree)
        self.ln2 = nn.LayerNorm(self.config.hidden_dim, elementwise_affine=self.config.elementwise_affine)

        self.l3 = QKAL_Layer(self.config.hidden_dim, self.K, self.config.degree)

        #calibration parameters
        self.cal_a = nn.Parameter(torch.tensor(1.0))
        self.cal_b = nn.Parameter(torch.tensor(1e-3))
        self.cal_c = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns [E[f_i(Y)|x]]_{i=0..K-1}
        which  predicts Legendre moments of target variable distribution
        """
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        x = self.ln0(x)
        x = self.l1(x); x = self.ln1(x)
        x = self.l2(x); x = self.ln2(x)
        m = self.l3(x)  # (B, K)
        return m