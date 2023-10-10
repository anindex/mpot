import torch
from typing import Any, Optional, List, Tuple


# min max scaler
class MinMaxScaler():
    def __init__(self, min: float = None, max: float = None):
        self.min = min
        self.max = max

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        if self.min is None:
            self.min = X.min()
        if self.max is None:
            self.max = X.max()
        return (X - self.min) / (self.max - self.min)

    def inverse(self, X: torch.Tensor) -> torch.Tensor:
        return X * (self.max - self.min) + self.min
    

# scale to [-1, 1]
class MinMaxCenterScaler():
    def __init__(self, dim_range: List[float], min: float = -1, max: float = 1):
        self.dim_range = dim_range
        self.dim = dim_range[1] - dim_range[0]
        self.min = min
        self.max = max

    def __call__(self, X: torch.Tensor):
        X[..., self.dim_range[0]:self.dim_range[1]] = 2 * (X[..., self.dim_range[0]:self.dim_range[1]] - self.min) / (self.max - self.min) - 1

    def inverse(self, X: torch.Tensor):
        X[..., self.dim_range[0]:self.dim_range[1]] = (X[..., self.dim_range[0]:self.dim_range[1]] + 1) * (self.max - self.min) / 2 + self.min


# min max mean scaler
class MinMaxMeanScaler():
    '''For torch tensors, NOTE: all are in-place operations'''
    def __init__(self, dim_range: List[float], min: float = -1, max: float = 1, mean: torch.Tensor = None):
        self.min = min
        self.max = max
        self.mean = mean
        self.dim_range = dim_range
        self.dim = dim_range[1] - dim_range[0]

    def __call__(self, X: torch.Tensor):
        if self.mean is None:
            self.mean = X[..., self.dim_range[0]:self.dim_range[1]].view((-1, self.dim)).mean(0)
        X[..., self.dim_range[0]:self.dim_range[1]] = (X[..., self.dim_range[0]:self.dim_range[1]] - self.mean) / (self.max - self.min)

    def inverse(self, X: torch.Tensor):
        # clamp min max of input
        # X[..., self.dim_range[0]:self.dim_range[1]] = torch.clamp(X[..., self.dim_range[0]:self.dim_range[1]], -1., 1.)
        X[..., self.dim_range[0]:self.dim_range[1]] = X[..., self.dim_range[0]:self.dim_range[1]] * (self.max - self.min) + self.mean


# STANDARD SCALER
class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        return (X - self.mean) / self.std

    def inverse(self, X: torch.Tensor) -> torch.Tensor:
        return X * self.std + self.mean
