import torch
from typing import Tuple, List


def interpolate_trajectory(trajs: torch.Tensor, num_interpolation: int = 3) -> torch.Tensor:
    """Linearly interpolate between trajectory waypoints.

    Args:
        trajs: Trajectory tensor of shape (..., T, D)
        num_interpolation: number of interpolation points per segment

    Returns:
        Interpolated trajectory of shape (..., (T-1)*num_interpolation, D)
    """
    if num_interpolation <= 0 or trajs.dim() < 2:
        return trajs

    T = trajs.size(-2)
    D = trajs.size(-1)
    if T <= 1:
        return trajs

    k = num_interpolation
    denom = float(k + 1)
    alpha_line = torch.arange(1, k + 1, dtype=trajs.dtype, device=trajs.device) / denom

    # Shape alpha to (..., 1, k, 1) to broadcast with (..., T-1, 1, D)
    nd = trajs.dim()
    shape_alpha: List[int] = []
    for _ in range(nd - 2):
        shape_alpha.append(1)
    shape_alpha.append(1)        # for T-1 (broadcast over segments)
    shape_alpha.append(k)        # interpolation factor dimension
    shape_alpha.append(1)        # for D
    alpha = alpha_line.view(shape_alpha)

    starts = trajs[..., 0:T-1, :].unsqueeze(-2)
    ends = trajs[..., 1:T, :].unsqueeze(-2)

    interpolated = starts * alpha + ends * (1.0 - alpha)

    # Reshape to (..., (T-1)*k, D)
    prefix_sizes: List[int] = []
    for d in range(nd - 2):
        prefix_sizes.append(trajs.size(d))
    new_time = (T - 1) * k
    new_shape = prefix_sizes + [new_time, D]
    return interpolated.reshape(new_shape)
