import torch
from typing import Optional, List


class MinMaxScaler:
    """Min-max normalization to [0, 1]."""

    def __init__(self, min_t: Optional[torch.Tensor] = None, max_t: Optional[torch.Tensor] = None):
        self.min_t = min_t
        self.max_t = max_t

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        if self.min_t is None:
            self.min_t = torch.min(X)
        if self.max_t is None:
            self.max_t = torch.max(X)
        return (X - self.min_t) / (self.max_t - self.min_t)

    def inverse(self, X: torch.Tensor) -> torch.Tensor:
        assert self.min_t is not None and self.max_t is not None
        return X * (self.max_t - self.min_t) + self.min_t


class MinMaxCenterScaler:
    """In-place min-max normalization to [-1, 1] over a slice of dimensions."""

    def __init__(self, dim_range: List[int], min_v: torch.Tensor, max_v: torch.Tensor):
        self.dim_range = dim_range
        self.dim = dim_range[1] - dim_range[0]
        self.min_v = min_v
        self.max_v = max_v

    def __call__(self, X: torch.Tensor) -> None:
        s, e = self.dim_range[0], self.dim_range[1]
        denom = self.max_v - self.min_v
        X[..., s:e] = 2.0 * (X[..., s:e] - self.min_v) / denom - 1.0

    def inverse(self, X: torch.Tensor) -> None:
        s, e = self.dim_range[0], self.dim_range[1]
        denom = self.max_v - self.min_v
        X[..., s:e] = (X[..., s:e] + 1.0) * denom / 2.0 + self.min_v


class MinMaxMeanScaler:
    """In-place min-max-mean normalization over a slice of dimensions."""

    def __init__(self, dim_range: List[int], min_v: torch.Tensor, max_v: torch.Tensor,
                 mean_t: Optional[torch.Tensor] = None):
        self.dim_range = dim_range
        self.dim = dim_range[1] - dim_range[0]
        self.min_v = min_v
        self.max_v = max_v
        self.mean_t = mean_t

    def __call__(self, X: torch.Tensor) -> None:
        s, e = self.dim_range[0], self.dim_range[1]
        if self.mean_t is None:
            self.mean_t = X[..., s:e].reshape(-1, self.dim).mean(0)
        denom = self.max_v - self.min_v
        X[..., s:e] = (X[..., s:e] - self.mean_t) / denom

    def inverse(self, X: torch.Tensor) -> None:
        s, e = self.dim_range[0], self.dim_range[1]
        assert self.mean_t is not None
        denom = self.max_v - self.min_v
        X[..., s:e] = X[..., s:e] * denom + self.mean_t


class StandardScaler:
    """Standard (z-score) normalization."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        self.mean = mean
        self.std = std

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        return (X - self.mean) / self.std

    def inverse(self, X: torch.Tensor) -> torch.Tensor:
        return X * self.std + self.mean
