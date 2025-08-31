from dataclasses import dataclass
from enum import Enum
from typing import Literal


class CalibrationMode(Enum):
    SOFTPLUS = "softplus"
    CALIBRATED_SOFTPLUS = "calibrated_softplus"
    CLAMP = "clamp"

@dataclass
class ReconstructionConfig:
    degree: int = 0          # or ignore and set output=1
    input_dim: int = 13
    hidden_dim: int = 128    # only if you add an MLP
    batch_size: int = 32


    elementwise_affine: bool = True
    calibration_mode: CalibrationMode = CalibrationMode.CALIBRATED_SOFTPLUS
    grid_size = 512

    seed=42
    reduction: Literal["mean", "sum", "none"] = "none"
    learning_rate: float = 3e-3
    weight_decay: float = 1e-3
    epochs: int = 100

    kde_bandwidth = None
    normalize_density : bool  = True

    n_of_reconstructions: int = 10