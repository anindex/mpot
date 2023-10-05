import torch


def rotation_matrix(theta: torch.Tensor) -> torch.Tensor:
    theta = theta.unsqueeze(1).unsqueeze(1)
    dim1 = torch.cat([torch.cos(theta), -torch.sin(theta)], dim=2)
    dim2 = torch.cat([torch.sin(theta), torch.cos(theta)], dim=2)
    mat = torch.cat([dim1, dim2], dim=1)
    return mat


def get_random_maximal_torus_matrix(origin: torch.Tensor,
                                    angle_range=[0, 2 * torch.pi], **kwargs) -> torch.Tensor:
    batch, dim = origin.shape
    assert dim % 2 == 0, 'Only work with even dim for random rotation for now.'
    theta = torch.rand(dim // 2, batch).type_as(origin) * (angle_range[1] - angle_range[0]) + angle_range[0]  # [batch, dim // 2]
    rot_mat = torch.vmap(rotation_matrix)(theta).transpose(0, 1)
    # make batch block diag
    max_torus_mat = torch.diag_embed(rot_mat[:, :, [0, 1], [0, 1]].flatten(-2, -1), offset=0)
    even, odd = torch.arange(0, dim, 2), torch.arange(1, dim, 2)
    max_torus_mat[:, even, odd] = rot_mat[:, :, 0, 1]
    max_torus_mat[:, odd, even] = rot_mat[:, :, 1, 0]
    return max_torus_mat


def get_random_uniform_rot_matrix(origin: torch.Tensor,
                                  **kwargs) -> torch.Tensor:
    """Compute a uniformly random rotation matrix drawn from the Haar distribution
    (the only uniform distribution on SO(n)). This is less efficient than maximal torus.
    See: Stewart, G.W., "The efficient generation of random orthogonal
    matrices with an application to condition estimators", SIAM Journal
    on Numerical Analysis, 17(3), pp. 403-409, 1980.
    For more information see
    http://en.wikipedia.org/wiki/Orthogonal_matrix#Randomization"""

    batch, dim = origin.shape
    H = torch.eye(dim).repeat(batch, 1, 1).type_as(origin)
    D = torch.ones((batch, dim)).type_as(origin)
    for i in range(1, dim):
        v = torch.normal(0, 1., size=(batch, dim - i + 1)).type_as(origin)
        D[:, i - 1] = torch.sign(v[:, 0])
        v[:, 0] -= D[:, i - 1] * torch.norm(v, dim=-1)
        # Householder transformation
        outer = v.unsqueeze(-2) * v.unsqueeze(-1)
        Hx = torch.eye(dim - i + 1).repeat(batch, 1, 1).type_as(origin) - 2 * outer / torch.square(v).sum(dim=-1).unsqueeze(-1).unsqueeze(-1)
        T = torch.eye(dim).repeat(batch, 1, 1).type_as(origin)
        T[:, i - 1:, i - 1:] = Hx
        H = torch.matmul(H, T)
    # Fix the last sign such that the determinant is 1
    D[:, -1] = (-1)**(1 - dim % 2) * D.prod(dim=-1)
    R = (D.unsqueeze(-1) * H.mT).mT
    return R


if __name__ == '__main__':
    from torch_robotics.torch_utils.torch_timer import TimerCUDA
    origin = torch.zeros(100, 8).cuda()
    with TimerCUDA() as t:
        rot_mat = get_random_maximal_torus_matrix(origin)
    print(t.elapsed)
    print(rot_mat.shape)
    with TimerCUDA() as t:
        rot_mat = get_random_uniform_rot_matrix(origin)
    print(t.elapsed)
    print(rot_mat.shape)
    print(torch.det(rot_mat))
