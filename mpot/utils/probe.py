import torch


def get_random_probe_points(
    origin: torch.Tensor,
    points: torch.Tensor,
    probe_radius: float = 2.0,
    num_probe: int = 5,
) -> torch.Tensor:
    """Sample random probe points along polytope directions.

    Args:
        origin: (batch, dim) or (dim,)
        points: (batch, num_points, dim)
        probe_radius: maximum probe distance
        num_probe: number of probes per vertex

    Returns:
        Probe points of shape (batch, num_points, num_probe, dim)
    """
    if origin.dim() == 1:
        origin = origin.unsqueeze(0)

    batch, num_points, dim = points.shape

    alpha = torch.rand(
        (batch, num_points, num_probe, 1),
        dtype=points.dtype,
        device=points.device,
    )

    probe_points = points * probe_radius
    probe_points = probe_points.unsqueeze(-2) * alpha + origin.unsqueeze(1).unsqueeze(1)
    return probe_points


def get_probe_points(
    origin: torch.Tensor,
    points: torch.Tensor,
    probe_radius: float = 2.0,
    num_probe: int = 5,
) -> torch.Tensor:
    """Deterministic probes at fractions i/(num_probe+1), i=1..num_probe.

    Args:
        origin: (batch, dim) or (dim,)
        points: (batch, num_points, dim)
        probe_radius: maximum probe distance
        num_probe: number of probes per vertex

    Returns:
        Probe points of shape (batch, num_points, num_probe, dim)
    """
    if origin.dim() == 1:
        origin = origin.unsqueeze(0)

    denom = float(num_probe + 1)
    alpha_line = torch.arange(1, num_probe + 1, device=points.device, dtype=points.dtype) / denom
    alpha = alpha_line.view(1, 1, -1, 1)

    probe_points = points * probe_radius
    probe_points = probe_points.unsqueeze(-2) * alpha + origin.unsqueeze(1).unsqueeze(1)
    return probe_points


def get_shifted_points(new_origins: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Shift points to new origins.

    Args:
        new_origins: (no, dim)
        points: (nb, dim), assumed centered at origin

    Returns:
        Shifted points of shape (no, nb, dim)
    """
    return points.unsqueeze(0) + new_origins.unsqueeze(1)


def get_projecting_points(
    X1: torch.Tensor,
    X2: torch.Tensor,
    probe_step_size: float,
    num_probe: int = 5,
) -> torch.Tensor:
    """Compute probe points along directions from X1 to X2.

    Args:
        X1: (nb1, dim)
        X2: (nb2, dim) or (nb1, nb2, dim)
        probe_step_size: step size for probes
        num_probe: number of probes

    Returns:
        Probe points of shape (nb1, nb2, num_probe, dim)
    """
    if X2.dim() == 2:
        X1e = X1.unsqueeze(1).unsqueeze(-2)
        X2e = X2.unsqueeze(0).unsqueeze(-2)
    else:
        X1e = X1.unsqueeze(1).unsqueeze(-2)
        X2e = X2.unsqueeze(-2)

    alpha_line = torch.arange(1, num_probe + 1, device=X1.device, dtype=X1.dtype) * probe_step_size
    alpha = alpha_line.view(1, 1, -1, 1)

    points = X1e + (X2e - X1e) * alpha
    return points
