import torch
import math


def rotation_matrix(theta: torch.Tensor) -> torch.Tensor:
    """Build 2x2 rotation matrices from angles.

    Args:
        theta: (...,) angles in radians

    Returns:
        Rotation matrices of shape (..., 2, 2)
    """
    if theta.ndim == 0:
        theta = theta.unsqueeze(0)
    flat = theta.reshape(-1)
    c = torch.cos(flat)
    s = torch.sin(flat)

    mats = torch.zeros((flat.shape[0], 2, 2), dtype=theta.dtype, device=theta.device)
    mats[:, 0, 0] = c
    mats[:, 0, 1] = -s
    mats[:, 1, 0] = s
    mats[:, 1, 1] = c
    return mats.reshape(theta.shape + (2, 2,))


def get_random_maximal_torus_matrix(
    origin: torch.Tensor,
    angle_min: float = 0.0,
    angle_max: float = 2.0 * math.pi,
) -> torch.Tensor:
    """Generate random block-diagonal rotation matrices via maximal torus.

    Each matrix consists of independent 2x2 rotation blocks along the diagonal.
    Only supports even-dimensional inputs.

    Args:
        origin: (batch, dim) or (dim,)
        angle_min: minimum rotation angle
        angle_max: maximum rotation angle

    Returns:
        Rotation matrices of shape (batch, dim, dim)
    """
    if origin.ndim == 1:
        origin = origin.unsqueeze(0)
    batch, dim = origin.shape
    assert dim % 2 == 0, "Only even dimensions are supported."

    k = dim // 2
    theta = torch.rand((batch, k), dtype=origin.dtype, device=origin.device) * (angle_max - angle_min) + angle_min

    M = torch.zeros((batch, dim, dim), dtype=origin.dtype, device=origin.device)
    for j in range(k):
        c = torch.cos(theta[:, j])
        s = torch.sin(theta[:, j])
        ej = 2 * j
        oj = ej + 1
        M[:, ej, ej] = c
        M[:, ej, oj] = -s
        M[:, oj, ej] = s
        M[:, oj, oj] = c
    return M


def get_random_uniform_rot_matrix(origin: torch.Tensor) -> torch.Tensor:
    """Generate Haar-uniform random rotation matrices via Householder products.

    Implements Stewart (1980) for generating SO(n) matrices.

    Args:
        origin: (batch, dim) or (dim,)

    Returns:
        Rotation matrices of shape (batch, dim, dim) with det = 1
    """
    if origin.ndim == 1:
        origin = origin.unsqueeze(0)
    batch, dim = origin.shape

    H = torch.eye(dim, dtype=origin.dtype, device=origin.device).unsqueeze(0).repeat(batch, 1, 1)
    D = torch.ones((batch, dim), dtype=origin.dtype, device=origin.device)

    eps = 1e-12

    for i in range(1, dim):
        m = dim - i + 1
        v = torch.randn((batch, m), dtype=origin.dtype, device=origin.device)

        D[:, i - 1] = torch.sign(v[:, 0])

        v_sqsum = (v * v).sum(1)
        v_norm = torch.sqrt(v_sqsum + eps)
        v[:, 0] = v[:, 0] - D[:, i - 1] * v_norm

        denom = (v * v).sum(1) + eps
        beta = 2.0 / denom
        outer = v.unsqueeze(2) * v.unsqueeze(1)
        I_m = torch.eye(m, dtype=origin.dtype, device=origin.device).unsqueeze(0).repeat(batch, 1, 1)
        Hx = I_m - beta.view(batch, 1, 1) * outer

        T = torch.eye(dim, dtype=origin.dtype, device=origin.device).unsqueeze(0).repeat(batch, 1, 1)
        T[:, i - 1:, i - 1:] = Hx
        H = torch.matmul(H, T)

    factor = -1.0 if (dim % 2) == 0 else 1.0
    D[:, -1] = factor * torch.prod(D[:, :-1], 1)

    R = torch.matmul(H, torch.diag_embed(D))
    return R
