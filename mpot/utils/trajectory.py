import torch


def interpolate_trajectory(trajs: torch.Tensor, num_interpolation: int = 3) -> torch.Tensor:
    # Interpolates a trajectory linearly between waypoints
    dim = trajs.shape[-1]
    if num_interpolation > 0:
        assert trajs.ndim > 1
        traj_dim = trajs.shape
        alpha = torch.linspace(0, 1, num_interpolation + 2).type_as(trajs)[1:num_interpolation + 1]
        alpha = alpha.view((1,) * len(traj_dim[:-1]) + (-1, 1))
        interpolated_trajs = trajs[..., 0:traj_dim[-2] - 1, None, :] * alpha + \
                             trajs[..., 1:traj_dim[-2], None, :] * (1 - alpha)
        interpolated_trajs = interpolated_trajs.view(traj_dim[:-2] + (-1, dim))
    else:
        interpolated_trajs = trajs
    return interpolated_trajs
