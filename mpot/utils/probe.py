import torch


def get_random_probe_points(origin: torch.Tensor, 
                            points: torch.Tensor, 
                            probe_radius: float = 2., 
                            num_probe: int = 5) -> torch.Tensor:
    batch, num_points, dim = points.shape
    alpha = torch.rand(batch, num_points, num_probe, 1).type_as(points)
    probe_points = points * probe_radius
    probe_points = probe_points.unsqueeze(-2) * alpha  + origin.unsqueeze(1).unsqueeze(1)  # [batch, num_points, num_probe, dim]
    return probe_points


def get_probe_points(origin: torch.Tensor, 
                     points: torch.Tensor, 
                     probe_radius: float = 2., 
                     num_probe: int = 5) -> torch.Tensor:
    alpha = torch.linspace(0, 1, num_probe + 2).type_as(points)[1:num_probe + 1].view(1, 1, -1, 1)
    probe_points = points * probe_radius
    probe_points = probe_points.unsqueeze(-2) * alpha  + origin.unsqueeze(1).unsqueeze(1)  # [batch, num_points, num_probe, dim]
    return probe_points


def get_shifted_points(new_origins: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    '''
    Args:
        new_origins: [no, dim]
        points: [nb, dim]
    Returns:
        shifted_points: [no, nb, dim]
    '''
    # asumming points has centroid at origin
    shifted_points = points + new_origins.unsqueeze(1)
    return shifted_points


def get_projecting_points(X1: torch.Tensor, X2: torch.Tensor, probe_step_size: float, num_probe: int = 5) -> torch.Tensor:
    '''
    X1: [nb1 x dim]
    X2: [nb2 x dim] or [nb1 x nb2 x dim]
    return [nb1 x nb2 x num_probe x dim]
    '''
    if X2.ndim == 2:
        X1 = X1.unsqueeze(1).unsqueeze(-2)
        X2 = X2.unsqueeze(0).unsqueeze(-2)
    elif X2.ndim == 3:
        assert X2.shape[0] == X1.shape[0]
        X1 = X1.unsqueeze(1).unsqueeze(-2)
        X2 = X2.unsqueeze(-2)
    alpha = torch.arange(1, num_probe + 1).type_as(X1) * probe_step_size
    alpha = alpha.view(1, 1, -1, 1)
    points = X1 + (X2 - X1) * alpha
    return points


if __name__ == '__main__':
    X1 = torch.tensor([
        [0, 0],
        [2, 2]
    ], dtype=torch.float32)
    X2 = torch.tensor([
        [0, 2],
        [2, 4]
    ], dtype=torch.float32)
    print(get_projecting_points(X1, X2, 0.5, 1))
