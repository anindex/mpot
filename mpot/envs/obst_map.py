from typing import Any
import numpy as np
import torch
import matplotlib.pyplot as plt
from math import ceil
from abc import ABC, abstractmethod
import os.path as osp
from copy import deepcopy


class Obstacle(ABC):
    """
    Base 2D Obstacle class
    """

    def __init__(self,center_x,center_y):
        self.center_x = center_x
        self.center_y = center_y
        self.origin = np.array([self.center_x, self.center_y])

    def _obstacle_collision_check(self, obst_map):
        valid = True
        obst_map_test = self._add_to_map(deepcopy(obst_map))
        if np.any(obst_map_test.map > 1):
            valid = False
        return valid

    def _point_collision_check(self, obst_map, pts):
        valid = True
        if pts is not None:
            obst_map_test = self._add_to_map(np.copy(obst_map))
            for pt in pts:
                if obst_map_test[ceil(pt[0]), ceil(pt[1])] >= 1:
                    valid = False
                    break
        return valid

    @abstractmethod
    def _add_to_map(self, obst_map):
        pass


class ObstacleRectangle(Obstacle):
    """
    Derived 2D rectangular Obstacle class
    """

    def __init__(
            self,
            center_x=0,
            center_y=0,
            width=None,
            height=None,
    ):
        super().__init__(center_x, center_y)
        self.width = width
        self.height = height

    def _add_to_map(self, obst_map):
        # Convert dims to cell indices
        w = ceil(self.width / obst_map.cell_size)
        h = ceil(self.height / obst_map.cell_size)
        c_x = ceil(self.center_x / obst_map.cell_size)
        c_y = ceil(self.center_y / obst_map.cell_size)

        obst_map.map[
            c_y - ceil(h/2.) + obst_map.origin_yi:
            c_y + ceil(h/2.) + obst_map.origin_yi,
            c_x - ceil(w/2.) + obst_map.origin_xi:
            c_x + ceil(w/2.) + obst_map.origin_xi,
            ] += 1
        return obst_map


class ObstacleCircle(Obstacle):
    """
    Derived 2D circle Obstacle class
    """

    def __init__(
            self,
            center_x=0,
            center_y=0,
            radius=1.
    ):
        super().__init__(center_x, center_y)
        self.radius = radius

    def is_inside(self, p):
        # Check if point p is inside of the discretized circle
        return np.linalg.norm(p - self.origin) <= self.radius

    def _add_to_map(self, obst_map):
        # Convert dims to cell indices
        c_r = ceil(self.radius / obst_map.cell_size)
        c_x = ceil(self.center_x / obst_map.cell_size)
        c_y = ceil(self.center_y / obst_map.cell_size)

        for i in range(c_y - 2 * c_r + obst_map.origin_yi, c_y + 2 * c_r + obst_map.origin_yi):
            for j in range(c_x - 2 * c_r + obst_map.origin_xi, c_x + 2 * c_r + obst_map.origin_xi):
                p = np.array([(j - obst_map.origin_xi) * obst_map.cell_size,
                              (i - obst_map.origin_yi) * obst_map.cell_size])
                if self.is_inside(p):
                    obst_map.map[i, j] += 1
        return obst_map


class ObstacleMap:
    """
    Generates an occupancy grid.
    """
    def __init__(self, map_dim, cell_size, tensor_args=None):

        assert map_dim[0] % 2 == 0
        assert map_dim[1] % 2 == 0

        if tensor_args is None:
            tensor_args = {'device': torch.device('cpu'), 'dtype': torch.float32}
        self.tensor_args = tensor_args

        cmap_dim = [0, 0]
        cmap_dim[0] = ceil(map_dim[0]/cell_size)
        cmap_dim[1] = ceil(map_dim[1]/cell_size)

        self.map = np.zeros(cmap_dim)
        self.cell_size = cell_size

        # Map center (in cells)
        self.origin_xi = int(cmap_dim[0]/2)
        self.origin_yi = int(cmap_dim[1]/2)

        # self.xlim = map_dim[0]

        self.x_dim, self.y_dim = self.map.shape
        x_range = self.cell_size * self.x_dim
        y_range = self.cell_size * self.y_dim
        self.xlim = [-x_range/2, x_range/2]
        self.ylim = [-y_range/2, y_range/2]

        self.c_offset = torch.tensor([self.origin_xi, self.origin_yi], **self.tensor_args)

    def __call__(self, X, **kwargs):
        return self.compute_cost(X, **kwargs)

    def convert_map(self):
        self.map_torch = torch.Tensor(self.map).to(**self.tensor_args)
        return self.map_torch

    def plot(self, save_dir=None, filename="obst_map.png"):
        fig = plt.figure()
        plt.imshow(self.map)
        plt.gca().invert_yaxis()
        plt.show()
        if save_dir is not None:
            plt.savefig(osp.join(save_dir, filename))
        return fig

    def get_xy_grid(self, device):
        xv, yv = torch.meshgrid([torch.linspace(self.xlim[0], self.xlim[1], self.x_dim),
                                 torch.linspace(self.ylim[0], self.ylim[1], self.y_dim)])
        xy_grid = torch.stack((xv, yv), dim=2)
        return xy_grid.to(device)

    def get_collisions(self, X, **kwargs):
        """
        Checks for collision in a batch of trajectories using the generated occupancy grid (i.e. obstacle map), and
        returns sum of collision costs for the entire batch.

        :param weight: weight on obstacle cost, float tensor.
        :param X: Tensor of trajectories, of shape (batch_size, traj_length, position_dim)
        :return: collision cost on the trajectories
        """
        X_occ = X * (1/self.cell_size) + self.c_offset
        X_occ = X_occ.floor().int()

        # Project out-of-bounds locations to axis
        X_occ[...,0] = X_occ[..., 0].clamp(0, self.map.shape[0]-1)
        X_occ[...,1] = X_occ[..., 1].clamp(0, self.map.shape[1]-1)

        # Collisions
        collision_vals = self.map_torch[X_occ[..., 1], X_occ[..., 0]]
        return collision_vals

    def compute_cost(self, X, **kwargs):
        return self.get_collisions(X, **kwargs)

    def zero_grad(self):
        pass
