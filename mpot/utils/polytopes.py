import torch
from typing import Any, Optional, Tuple
import numpy as np
from itertools import product
import math
from mpot.utils.probe import get_random_probe_points, get_probe_points
from mpot.utils.rotation import get_random_maximal_torus_matrix


def get_cube_vertices(origin: torch.Tensor, radius: float = 1., **kwargs) -> torch.Tensor:
    dim = origin.shape[-1]
    points = torch.tensor(list(product([1, -1], repeat=dim))) / np.sqrt(dim)
    points = points.type_as(origin) * radius + origin
    return points


def get_orthoplex_vertices(origin: torch.Tensor, radius: float = 1., **kwargs) -> torch.Tensor:
    dim = origin.shape[-1]
    points = torch.zeros((2 * dim, dim)).type_as(origin)
    first = torch.arange(0, dim)
    second = torch.arange(dim, 2 * dim)
    points[first, first] = radius
    points[second, first] = -radius
    points = points + origin
    return points


def get_simplex_vertices(origin: torch.Tensor, radius: float = 1., **kwargs) -> torch.Tensor:
    '''
    Simplex coordinates: https://en.wikipedia.org/wiki/Simplex#Cartesian_coordinates_for_a_regular_n-dimensional_simplex_in_Rn
    '''
    dim = origin.shape[-1]
    points = math.sqrt(1 + 1/dim) * torch.eye(dim) - ((math.sqrt(dim + 1) + 1) / math.sqrt(dim ** 3)) * torch.ones((dim, dim))
    points = torch.concatenate([points, (1 / math.sqrt(dim)) * torch.ones((1, dim))], dim=0)
    points = points.type_as(origin) * radius + origin
    return points


def get_sampled_polytope_vertices(origin: torch.Tensor,
                                  polytope_vertices: torch.Tensor, 
                                  step_radius: float = 1., 
                                  probe_radius: float = 2.,
                                  num_probe: int = 5,
                                  **kwargs) -> Tuple[torch.Tensor]:
    if origin.ndim == 1:
        origin = origin.unsqueeze(0)
    batch, dim = origin.shape
    polytope_vertices = polytope_vertices.repeat(batch, 1, 1)  # [batch, num_vertices, dim]

    # batch random polytope
    maximal_torus_mat = get_random_maximal_torus_matrix(origin)
    polytope_vertices = polytope_vertices @ maximal_torus_mat
    step_points = polytope_vertices * step_radius + origin.unsqueeze(1)  # [batch, num_vertices, dim]
    probe_points = get_probe_points(origin, polytope_vertices, probe_radius, num_probe)  # [batch, num_vertices, num_probe, dim]
    return step_points, probe_points, polytope_vertices


def get_sampled_points_on_sphere(origin: torch.Tensor, 
                                 step_radius: float = 1., 
                                 probe_radius: float = 2., 
                                 num_probe: int = 5, 
                                 num_sphere_point: int = 50, 
                                 random_probe: bool = False, **kwargs) -> Tuple[torch.Tensor]:
    if origin.ndim == 1:
        origin = origin.unsqueeze(0)
    batch, dim = origin.shape
    # marsaglia method
    points = torch.randn(batch, num_sphere_point, dim).type_as(origin)  # [batch, num_points, dim]
    points = points / points.norm(dim=-1, keepdim=True)
    step_points = points * step_radius + origin.unsqueeze(1)  # [batch, num_points, dim]
    if random_probe:
        probe_points = get_random_probe_points(origin, points, probe_radius, num_probe)
    else:
        probe_points = get_probe_points(origin, points, probe_radius, num_probe)  # [batch, 2 * dim, num_probe, dim]
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
