import torch
from typing import Tuple

# If these utils are TorchScript-able in your codebase, imports are fine.
# Otherwise you can pass precomputed tensors to the functions below or
# make TorchScript-compatible versions of them as well.
from mpot.utils.probe import get_random_probe_points, get_probe_points
from mpot.utils.rotation import get_random_maximal_torus_matrix


@torch.jit.script
def _cube_sign_matrix(dim: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    # Builds all 2^dim sign vectors in {+1, -1}^dim without numpy/itertools.
    n: int = 1 << dim
    out = torch.empty((n, dim), dtype=dtype, device=device)
    for i in range(n):
        for j in range(dim):
            # Bit j of i: 0 -> +1, 1 -> -1 (any consistent mapping is fine)
            s = 1.0 if ((i >> j) & 1) == 0 else -1.0
            out[i, j] = s
    return out


@torch.jit.script
def get_cube_vertices(origin: torch.Tensor, radius: float = 1.0) -> torch.Tensor:
    """
    Returns (2^dim, dim) vertices of a hypercube centered at origin,
    normalized so each vertex has L2 norm = radius (same as original behavior).
    """
    dim = int(origin.shape[-1])
    signs = _cube_sign_matrix(dim, origin.dtype, origin.device)
    # normalize by sqrt(dim) to match original
    norm_factor = torch.sqrt(torch.tensor(float(dim), dtype=origin.dtype, device=origin.device))
    points = signs / norm_factor
    points = points * radius + origin
    return points


@torch.jit.script
def get_orthoplex_vertices(origin: torch.Tensor, radius: float = 1.0) -> torch.Tensor:
    """
    Returns (2*dim, dim) vertices of a cross-polytope (orthoplex) centered at origin.
    """
    dim = int(origin.shape[-1])
    out = torch.zeros((2 * dim, dim), dtype=origin.dtype, device=origin.device)
    first = torch.arange(0, dim, device=origin.device)
    second = torch.arange(dim, 2 * dim, device=origin.device)
    out[first, first] = radius
    out[second, first] = -radius
    return out + origin


@torch.jit.script
def get_simplex_vertices(origin: torch.Tensor, radius: float = 1.0) -> torch.Tensor:
    """
    Regular simplex coordinates in R^n (n = dim).
    Formula follows Wikipedia but in TorchScript-friendly ops.
    Output shape: (dim+1, dim)
    """
    dim_i = int(origin.shape[-1])
    dim_f = float(dim_i)

    a = torch.sqrt(torch.tensor(1.0 + 1.0 / dim_f, dtype=origin.dtype, device=origin.device))
    b = (torch.sqrt(torch.tensor(dim_f + 1.0, dtype=origin.dtype, device=origin.device)) + 1.0) / torch.sqrt(
        torch.tensor(dim_f ** 3, dtype=origin.dtype, device=origin.device)
    )

    eye = torch.eye(dim_i, dtype=origin.dtype, device=origin.device)
    ones_dd = torch.ones((dim_i, dim_i), dtype=origin.dtype, device=origin.device)
    pts = a * eye - b * ones_dd  # (dim, dim)

    one_row = (1.0 / torch.sqrt(torch.tensor(dim_f, dtype=origin.dtype, device=origin.device))) * torch.ones(
        (1, dim_i), dtype=origin.dtype, device=origin.device
    )
    points = torch.cat([pts, one_row], dim=0)  # (dim+1, dim)
    points = points * radius + origin
    return points


@torch.jit.script
def get_sampled_polytope_vertices(
    origin: torch.Tensor,
    polytope_vertices: torch.Tensor,
    step_radius: float = 1.0,
    probe_radius: float = 2.0,
    num_probe: int = 5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    origin: (dim,) or (batch, dim)
    polytope_vertices: (num_vertices, dim) — base vertices before rotation
    Returns:
      step_points:   (batch, num_vertices, dim)
      probe_points:  (batch, num_vertices, num_probe, dim)
      rotated_verts: (batch, num_vertices, dim)
    """
    if origin.ndim == 1:
        origin = origin.unsqueeze(0)  # (1, dim)

    batch, dim = origin.shape[0], origin.shape[1]
    # Tile vertices for batch
    verts = polytope_vertices.unsqueeze(0).expand(batch, polytope_vertices.shape[0], dim)  # (batch, V, dim)

    # Random maximal torus rotations (assumed TorchScript-able)
    # get_random_maximal_torus_matrix should return shape (batch, dim, dim) or (dim, dim).
    rot = get_random_maximal_torus_matrix(origin)  # ensure your implementation supports scripting
    # Support both (batch, dim, dim) and (dim, dim)
    if rot.ndim == 2:
        rotated = torch.matmul(verts, rot)  # (batch, V, dim)
    else:
        # (batch, V, dim) @ (batch, dim, dim) -> (batch, V, dim)
        rotated = torch.matmul(verts, rot)

    step_points = rotated * step_radius + origin.unsqueeze(1)

    # Probes (assumed TorchScript-able)
    probe_points = get_probe_points(origin, rotated, probe_radius, num_probe)
    return step_points, probe_points, rotated


@torch.jit.script
def get_sampled_points_on_sphere(
    origin: torch.Tensor,
    step_radius: float = 1.0,
    probe_radius: float = 2.0,
    num_probe: int = 5,
    num_sphere_point: int = 50,
    random_probe: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if origin.dim() == 1:
        origin = origin.unsqueeze(0)
    batch = int(origin.shape[0])
    dim = int(origin.shape[1])

    points = torch.randn((batch, num_sphere_point, dim), dtype=origin.dtype, device=origin.device)
    # TorchScript-safe L2 normalization (avoid .norm)
    eps = torch.tensor(1e-12, dtype=origin.dtype, device=origin.device)
    norms_sq = torch.sum(points * points, -1, True)         # (B, P, 1)
    norms = torch.sqrt(norms_sq + eps)                      # (B, P, 1)
    points = points / norms

    step_points = points * step_radius + origin.unsqueeze(1)

    if random_probe:
        probe_points = get_random_probe_points(origin, points, probe_radius, num_probe)
    else:
        probe_points = get_probe_points(origin, points, probe_radius, num_probe)

    return step_points, probe_points, points



# --- TorchScript-safe replacements for the original registries ---
@torch.jit.script
def polytope_vertices_by_name(name: str, origin: torch.Tensor, radius: float = 1.0) -> torch.Tensor:
    if name == "cube":
        return get_cube_vertices(origin, radius)
    elif name == "orthoplex":
        return get_orthoplex_vertices(origin, radius)
    elif name == "simplex":
        return get_simplex_vertices(origin, radius)
    else:
        # Empty tensor to keep TorchScript happy; you can raise if you prefer
        return torch.empty((0, int(origin.shape[-1])), dtype=origin.dtype, device=origin.device)


@torch.jit.script
def polytope_num_vertices(name: str, dim: int) -> int:
    if name == "cube":
        return 1 << dim
    elif name == "orthoplex":
        return 2 * dim
    elif name == "simplex":
        return dim + 1
    else:
        return 0


@torch.jit.script
def sample_strategy_by_name(
    name: str,
    origin: torch.Tensor,
    polytope_vertices: torch.Tensor,
    step_radius: float,
    probe_radius: float,
    num_probe: int,
    num_sphere_point: int,
    random_probe: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Unifies 'polytope' and 'random' (sphere) sampling with a TorchScript-friendly switch.
    - For 'polytope', polytope_vertices must be provided (V, dim).
    - For 'random', polytope_vertices is ignored.
    """
    if name == "polytope":
        return get_sampled_polytope_vertices(
            origin=origin,
            polytope_vertices=polytope_vertices,
            step_radius=step_radius,
            probe_radius=probe_radius,
            num_probe=num_probe,
        )
    else:
        # default to random-on-sphere
        return get_sampled_points_on_sphere(
            origin=origin,
            step_radius=step_radius,
            probe_radius=probe_radius,
            num_probe=num_probe,
            num_sphere_point=num_sphere_point,
            random_probe=random_probe,
        )


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

# ---------------------------
# Example: scripting check
# ---------------------------
# torch.jit.script(get_cube_vertices)  # ok
# torch.jit.script(get_orthoplex_vertices)  # ok
# torch.jit.script(get_simplex_vertices)  # ok
# torch.jit.script(get_sampled_points_on_sphere)  # ok (assuming probe utilities are TorchScript-able)
# torch.jit.script(get_sampled_polytope_vertices)  # ok (assuming rotation/probe utilities are TorchScript-able)
# torch.jit.script(polytope_vertices_by_name)
# torch.jit.script(polytope_num_vertices)
# torch.jit.script(sample_strategy_by_name)
