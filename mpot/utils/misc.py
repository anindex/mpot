import torch
from typing import Optional, List

# --------------------------
# Min-Max Scaler (learnable)
# --------------------------
@torch.jit.script
class MinMaxScaler:
    min_t: Optional[torch.Tensor]
    max_t: Optional[torch.Tensor]

    def __init__(self, min_t: Optional[torch.Tensor] = None, max_t: Optional[torch.Tensor] = None):
        self.min_t = min_t
        self.max_t = max_t

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        if self.min_t is None:
            self.min_t = torch.min(X)
        if self.max_t is None:
            self.max_t = torch.max(X)
        min_val = self.min_t
        max_val = self.max_t
        assert min_val is not None
        assert max_val is not None
        return (X - min_val) / (max_val - min_val)

    def inverse(self, X: torch.Tensor) -> torch.Tensor:
        min_val = self.min_t
        max_val = self.max_t
        assert min_val is not None
        assert max_val is not None
        return X * (max_val - min_val) + min_val


# ----------------------------------
# Min-Max Center Scaler to [-1, 1]
# Accepts tensor min_v/max_v (0-D or 1-D matching slice length)
# In-place over a slice of dimensions
# ----------------------------------
@torch.jit.script
class MinMaxCenterScaler:
    dim_range: List[int]
    dim: int
    min_v: torch.Tensor
    max_v: torch.Tensor

    def __init__(self, dim_range: List[int], min_v: torch.Tensor, max_v: torch.Tensor):
        self.dim_range = dim_range
        self.dim = int(dim_range[1] - dim_range[0])
        self.min_v = min_v
        self.max_v = max_v

    def __call__(self, X: torch.Tensor) -> None:
        s = int(self.dim_range[0])
        e = int(self.dim_range[1])
        # Broadcast works for min_v/max_v shapes: () or (dim,)
        denom = (self.max_v - self.min_v)
        X[..., s:e] = 2.0 * (X[..., s:e] - self.min_v) / denom - 1.0

    def inverse(self, X: torch.Tensor) -> None:
        s = int(self.dim_range[0])
        e = int(self.dim_range[1])
        denom = (self.max_v - self.min_v)
        X[..., s:e] = (X[..., s:e] + 1.0) * denom / 2.0 + self.min_v


# ------------------------------
# Min-Max-Mean Scaler (in-place)
# mean is learned from data slice
# Accepts tensor min_v/max_v
# ------------------------------
@torch.jit.script
class MinMaxMeanScaler:
    dim_range: List[int]
    dim: int
    min_v: torch.Tensor
    max_v: torch.Tensor
    mean_t: Optional[torch.Tensor]

    def __init__(self, dim_range: List[int], min_v: torch.Tensor, max_v: torch.Tensor, mean_t: Optional[torch.Tensor] = None):
        self.dim_range = dim_range
        self.dim = int(dim_range[1] - dim_range[0])
        self.min_v = min_v
        self.max_v = max_v
        self.mean_t = mean_t

    def __call__(self, X: torch.Tensor) -> None:
        s = int(self.dim_range[0])
        e = int(self.dim_range[1])
        if self.mean_t is None:
            self.mean_t = X[..., s:e].reshape(-1, self.dim).mean(0)
        mean_val = self.mean_t
        assert mean_val is not None
        denom = (self.max_v - self.min_v)
        X[..., s:e] = (X[..., s:e] - mean_val) / denom

    def inverse(self, X: torch.Tensor) -> None:
        s = int(self.dim_range[0])
        e = int(self.dim_range[1])
        mean_val = self.mean_t
        assert mean_val is not None
        denom = (self.max_v - self.min_v)
        X[..., s:e] = X[..., s:e] * denom + mean_val


# -------------------
# Standard Scaler
# -------------------
@torch.jit.script
class StandardScaler:
    mean: torch.Tensor
    std: torch.Tensor

    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        self.mean = mean
        self.std = std

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        return (X - self.mean) / self.std

    def inverse(self, X: torch.Tensor) -> torch.Tensor:
        return X * self.std + self.mean
