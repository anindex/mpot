import torch


@torch.jit.script
def get_random_probe_points(
    origin: torch.Tensor,
    points: torch.Tensor,
    probe_radius: float = 2.0,
    num_probe: int = 5
) -> torch.Tensor:
    """
    origin: (batch, dim) or (dim,)
    points: (batch, num_points, dim)
    returns: (batch, num_points, num_probe, dim)
    """
    if origin.dim() == 1:
        origin = origin.unsqueeze(0)
    batch = int(points.shape[0])
    num_points = int(points.shape[1])
    dim = int(points.shape[2])

    alpha = torch.rand(
        (batch, num_points, num_probe, 1),
        dtype=points.dtype,
        device=points.device
    )

    probe_points = points * probe_radius
    probe_points = probe_points.unsqueeze(-2) * alpha + origin.unsqueeze(1).unsqueeze(1)
    return probe_points


@torch.jit.script
def get_probe_points(
    origin: torch.Tensor,
    points: torch.Tensor,
    probe_radius: float = 2.0,
    num_probe: int = 5
) -> torch.Tensor:
    """
    Deterministic probes at fractions i/(num_probe+1), i=1..num_probe
    origin: (batch, dim) or (dim,)
    points: (batch, num_points, dim)
    returns: (batch, num_points, num_probe, dim)
    """
    if origin.dim() == 1:
        origin = origin.unsqueeze(0)

    # alpha in (0,1): 1/(n+1), 2/(n+1), ..., n/(n+1)
    denom = float(num_probe + 1)
    alpha_line = torch.arange(1, num_probe + 1, device=points.device, dtype=points.dtype) / denom
    alpha = alpha_line.view(1, 1, -1, 1)

    probe_points = points * probe_radius
    probe_points = probe_points.unsqueeze(-2) * alpha + origin.unsqueeze(1).unsqueeze(1)
    return probe_points


@torch.jit.script
def get_shifted_points(new_origins: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """
    new_origins: (no, dim)
    points: (nb, dim)
    returns: (no, nb, dim)
    Assumes points are centered at the origin.
    """
    return points.unsqueeze(0) + new_origins.unsqueeze(1)


@torch.jit.script
def get_projecting_points(
    X1: torch.Tensor,
    X2: torch.Tensor,
    probe_step_size: float,
    num_probe: int = 5
) -> torch.Tensor:
    """
    X1: (nb1, dim)
    X2: (nb2, dim) or (nb1, nb2, dim)
    returns: (nb1, nb2, num_probe, dim)
    """
    if X2.dim() == 2:
        # broadcast X1 over nb2
        X1e = X1.unsqueeze(1).unsqueeze(-2)  # (nb1, 1, 1, dim)
        X2e = X2.unsqueeze(0).unsqueeze(-2)  # (1, nb2, 1, dim)
    else:
        # X2: (nb1, nb2, dim)
        # assert X2.shape[0] == X1.shape[0]  # TorchScript avoids Python assert with message
        X1e = X1.unsqueeze(1).unsqueeze(-2)  # (nb1, 1, 1, dim)
        X2e = X2.unsqueeze(-2)               # (nb1, nb2, 1, dim)

    # alpha = [1,2,...,num_probe] * step
    alpha_line = torch.arange(1, num_probe + 1, device=X1.device, dtype=X1.dtype) * probe_step_size
    alpha = alpha_line.view(1, 1, -1, 1)  # (1,1,P,1)

    points = X1e + (X2e - X1e) * alpha
    return points


# Optional quick scripting checks:
# torch.jit.script(get_random_probe_points)
# torch.jit.script(get_probe_points)
# torch.jit.script(get_shifted_points)
# torch.jit.script(get_projecting_points)
