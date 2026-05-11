import torch
from typing import Tuple

from mpot.utils.probe import get_random_probe_points, get_probe_points
from mpot.utils.rotation import get_random_maximal_torus_matrix


def get_cube_vertices(origin: torch.Tensor, radius: float = 1.0) -> torch.Tensor:
    """Vertices of a hypercube centered at origin, normalized to L2 norm = radius.

    Args:
        origin: (dim,) center point
        radius: vertex distance from origin

    Returns:
        Vertices of shape (2^dim, dim)
    """
    dim = origin.shape[-1]
    n = 1 << dim
    out = torch.empty((n, dim), dtype=origin.dtype, device=origin.device)
    for i in range(n):
        for j in range(dim):
            s = 1.0 if ((i >> j) & 1) == 0 else -1.0
            out[i, j] = s
    norm_factor = torch.sqrt(torch.tensor(float(dim), dtype=origin.dtype, device=origin.device))
    points = out / norm_factor
    points = points * radius + origin
    return points


def get_orthoplex_vertices(origin: torch.Tensor, radius: float = 1.0) -> torch.Tensor:
    """Vertices of a cross-polytope (orthoplex) centered at origin.

    Args:
        origin: (dim,) center point
        radius: vertex distance from origin

    Returns:
        Vertices of shape (2*dim, dim)
    """
    dim = origin.shape[-1]
    out = torch.zeros((2 * dim, dim), dtype=origin.dtype, device=origin.device)
    first = torch.arange(0, dim, device=origin.device)
    second = torch.arange(dim, 2 * dim, device=origin.device)
    out[first, first] = radius
    out[second, first] = -radius
    return out + origin


def get_simplex_vertices(origin: torch.Tensor, radius: float = 1.0) -> torch.Tensor:
    """Regular simplex vertices in R^dim.

    Args:
        origin: (dim,) center point
        radius: vertex distance from origin

    Returns:
        Vertices of shape (dim+1, dim)
    """
    dim_i = origin.shape[-1]
    dim_f = float(dim_i)

    a = torch.sqrt(torch.tensor(1.0 + 1.0 / dim_f, dtype=origin.dtype, device=origin.device))
    b = (torch.sqrt(torch.tensor(dim_f + 1.0, dtype=origin.dtype, device=origin.device)) + 1.0) / torch.sqrt(
        torch.tensor(dim_f ** 3, dtype=origin.dtype, device=origin.device)
    )

    eye = torch.eye(dim_i, dtype=origin.dtype, device=origin.device)
    ones_dd = torch.ones((dim_i, dim_i), dtype=origin.dtype, device=origin.device)
    pts = a * eye - b * ones_dd

    one_row = (1.0 / torch.sqrt(torch.tensor(dim_f, dtype=origin.dtype, device=origin.device))) * torch.ones(
        (1, dim_i), dtype=origin.dtype, device=origin.device
    )
    points = torch.cat([pts, one_row], dim=0)
    points = points * radius + origin
    return points


def get_sampled_polytope_vertices(
    origin: torch.Tensor,
    polytope_vertices: torch.Tensor,
    step_radius: float = 1.0,
    probe_radius: float = 2.0,
    num_probe: int = 5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample polytope vertices with random rotation and compute probe points.

    Args:
        origin: (dim,) or (batch, dim)
        polytope_vertices: (num_vertices, dim) base vertices before rotation
        step_radius: step size for vertices
        probe_radius: probe distance
        num_probe: number of probe points per vertex

    Returns:
        step_points: (batch, num_vertices, dim)
        probe_points: (batch, num_vertices, num_probe, dim)
        rotated_verts: (batch, num_vertices, dim)
    """
    if origin.ndim == 1:
        origin = origin.unsqueeze(0)

    batch, dim = origin.shape
    verts = polytope_vertices.unsqueeze(0).expand(batch, polytope_vertices.shape[0], dim)

    rot = get_random_maximal_torus_matrix(origin)
    rotated = torch.matmul(verts, rot)

    step_points = rotated * step_radius + origin.unsqueeze(1)

    probe_points = get_probe_points(origin, rotated, probe_radius, num_probe)
    return step_points, probe_points, rotated


def get_sampled_points_on_sphere(
    origin: torch.Tensor,
    step_radius: float = 1.0,
    probe_radius: float = 2.0,
    num_probe: int = 5,
    num_sphere_point: int = 50,
    random_probe: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample random points on a unit sphere and compute probe points.

    Args:
        origin: (batch, dim) or (dim,)
        step_radius: step size
        probe_radius: probe distance
        num_probe: number of probes per point
        num_sphere_point: number of random sphere points
        random_probe: if True, use random probes; otherwise deterministic

    Returns:
        step_points: (batch, num_sphere_point, dim)
        probe_points: (batch, num_sphere_point, num_probe, dim)
        normalized_points: (batch, num_sphere_point, dim)
    """
    if origin.dim() == 1:
        origin = origin.unsqueeze(0)
    batch, dim = origin.shape

    points = torch.randn((batch, num_sphere_point, dim), dtype=origin.dtype, device=origin.device)
    norms = torch.linalg.norm(points, dim=-1, keepdim=True).clamp(min=1e-12)
    points = points / norms

    step_points = points * step_radius + origin.unsqueeze(1)

    if random_probe:
        probe_points = get_random_probe_points(origin, points, probe_radius, num_probe)
    else:
        probe_points = get_probe_points(origin, points, probe_radius, num_probe)

    return step_points, probe_points, points


POLYTOPE_MAP = {
    'cube': get_cube_vertices,
    'orthoplex': get_orthoplex_vertices,
    'simplex': get_simplex_vertices,
}

POLYTOPE_NUM_VERTICES_MAP = {
    'cube': lambda dim: 2 ** dim,
    'orthoplex': lambda dim: 2 * dim,
    'simplex': lambda dim: dim + 1,
}

SAMPLE_POLYTOPE_MAP = {
    'polytope': get_sampled_polytope_vertices,
    'random': get_sampled_points_on_sphere,
}
