import torch
from typing import Tuple, List


@torch.jit.script
def interpolate_trajectory(trajs: torch.Tensor, num_interpolation: int = 3) -> torch.Tensor:
    # trajs: (..., T, D)
    if num_interpolation <= 0 or trajs.dim() < 2:
        return trajs

    T = int(trajs.size(-2))
    D = int(trajs.size(-1))
    if T <= 1:
        return trajs

    # alpha = [1/(k+1), 2/(k+1), ..., k/(k+1)]
    k = num_interpolation
    denom = float(k + 1)
    alpha_line = torch.arange(1, k + 1, dtype=trajs.dtype, device=trajs.device) / denom  # (k,)

    # Shape alpha to (..., 1, k, 1) to broadcast with (..., T-1, 1, D)
    nd = trajs.dim()
    shape_alpha: List[int] = []
    # ones for all leading dims before the last two (T, D)
    for _ in range(nd - 2):
        shape_alpha.append(1)
    shape_alpha.append(1)        # for T-1 (broadcast over segments)
    shape_alpha.append(k)        # interpolation factor dimension
    shape_alpha.append(1)        # for D
    alpha = alpha_line.view(shape_alpha)  # (..., 1, k, 1)

    starts = trajs[..., 0:T-1, :].unsqueeze(-2)  # (..., T-1, 1, D)
    ends   = trajs[..., 1:T,   :].unsqueeze(-2)  # (..., T-1, 1, D)

    interpolated = starts * alpha + ends * (1.0 - alpha)  # (..., T-1, k, D)

    # Reshape to (..., (T-1)*k, D) without using list(trajs.shape)
    prefix_sizes: List[int] = []
    for d in range(nd - 2):
        prefix_sizes.append(int(trajs.size(d)))
    new_time = (T - 1) * k
    new_shape = prefix_sizes + [new_time, D]
    return interpolated.reshape(new_shape)
