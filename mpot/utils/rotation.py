import torch

# ---------------------------------------------
# TorchScript-friendly 2D rotation block maker
# ---------------------------------------------
@torch.jit.script
def rotation_matrix(theta: torch.Tensor) -> torch.Tensor:
    """
    theta: (...,) angles (radians)
    returns: (..., 2, 2) rotation matrices with blocks [[c, -s],[s, c]]
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


# -------------------------------------------------------
# Maximal torus (block-diagonal 2x2 rotations) generator
# -------------------------------------------------------
@torch.jit.script
def get_random_maximal_torus_matrix(
    origin: torch.Tensor,
    angle_min: float = 0.0,
    angle_max: float = 6.283185307179586  # 2*pi as a literal for TorchScript defaults
) -> torch.Tensor:
    """
    origin: (batch, dim) or (dim,)
    returns: (batch, dim, dim) block-diagonal rotation matrices with 2x2 blocks
             using independent random angles per block.
    """
    if origin.ndim == 1:
        origin = origin.unsqueeze(0)
    batch = int(origin.shape[0])
    dim = int(origin.shape[1])
    assert dim % 2 == 0, "Only even dimensions are supported."

    k = dim // 2
    # Sample angles per (batch, block)
    theta = torch.rand((batch, k), dtype=origin.dtype, device=origin.device) * (angle_max - angle_min) + angle_min

    # Build block-diagonal matrix
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


# -------------------------------------------------------
# Uniform (Haar) random rotation via Householder products
# (Stewart 1980). Produces SO(n) matrices.
# -------------------------------------------------------
@torch.jit.script
def get_random_uniform_rot_matrix(origin: torch.Tensor) -> torch.Tensor:
    """
    origin: (batch, dim) or (dim,)
    returns: (batch, dim, dim) random rotations ~ Haar(SO(dim))
    """
    if origin.ndim == 1:
        origin = origin.unsqueeze(0)
    batch = int(origin.shape[0])
    dim = int(origin.shape[1])

    H = torch.eye(dim, dtype=origin.dtype, device=origin.device).unsqueeze(0).repeat(batch, 1, 1)
    D = torch.ones((batch, dim), dtype=origin.dtype, device=origin.device)

    eps = torch.tensor(1e-12, dtype=origin.dtype, device=origin.device)

    for i in range(1, dim):
        m = dim - i + 1
        v = torch.randn((batch, m), dtype=origin.dtype, device=origin.device)

        # sign of first component
        D[:, i - 1] = torch.sign(v[:, 0])

        # v := v - sign(v0)*||v||*e1  (avoid .norm())
        v_sqsum = (v * v).sum(1)                  # (batch,)
        v_norm = torch.sqrt(v_sqsum + eps)        # (batch,)
        v[:, 0] = v[:, 0] - D[:, i - 1] * v_norm  # broadcast-safe

        # Householder: I - 2 * v v^T / (v^T v)
        denom = (v * v).sum(1) + eps              # (batch,)
        beta = 2.0 / denom                        # (batch,)
        outer = v.unsqueeze(2) * v.unsqueeze(1)   # (batch, m, m)
        I_m = torch.eye(m, dtype=origin.dtype, device=origin.device).unsqueeze(0).repeat(batch, 1, 1)
        Hx = I_m - beta.view(batch, 1, 1) * outer # (batch, m, m)

        # Embed into full-sized T
        T = torch.eye(dim, dtype=origin.dtype, device=origin.device).unsqueeze(0).repeat(batch, 1, 1)
        T[:, i - 1:, i - 1:] = Hx
        H = torch.matmul(H, T)

    # Set final sign in D so det = 1  (factor = -1 if dim even, +1 if dim odd)
    factor = -1.0 if (dim % 2) == 0 else 1.0
    D[:, -1] = factor * torch.prod(D[:, :-1], 1)

    # R = H * diag(D)
    R = torch.matmul(H, torch.diag_embed(D))
    return R


# -------------------------
# Quick manual smoke test
# (not part of scripting)
# -------------------------
if __name__ == "__main__":
    B, D = 4, 8
    x = torch.zeros(B, D)

    # Script checks
    torch.jit.script(rotation_matrix)
    torch.jit.script(get_random_maximal_torus_matrix)
    torch.jit.script(get_random_uniform_rot_matrix)

    M = get_random_maximal_torus_matrix(x)
    R = get_random_uniform_rot_matrix(x)
    print(M.shape, R.shape)
    # Determinants close to 1
    print(torch.det(R))
