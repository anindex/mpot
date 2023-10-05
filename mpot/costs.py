from typing import Any, Optional, Tuple, List, Callable
from abc import ABC, abstractmethod
import torch


class Cost(ABC):
    def __init__(self, dim: int, tensor_args=None):
        if tensor_args is None:
            tensor_args = {'device': torch.device('cpu'), 'dtype': torch.float32}
        self.dim = dim
        self.state_dim = dim * 2
        self.tensor_args = tensor_args

    @abstractmethod
    def eval(self, points: int, observation=None):
        pass


class CostComposite(Cost):

    def __init__(
        self,
        dim: int,
        cost_list: List[Cost],
        FK: Callable = None,
        tensor_args=None
    ):
        super().__init__(dim, tensor_args=tensor_args)
        self.cost_list = cost_list
        self.FK = FK

    def cost(self, points: torch.Tensor, x_points: torch.Tensor = None, **observation) -> torch.Tensor:
        traj_dim = observation.get('traj_dim', None)
        x_points = None
        if self.FK is not None:  # NOTE: only works with SE(3) FK for now
            x_points = self.FK(points.view(-1, self.state_dim)[:, :self.dim]).reshape(traj_dim[:-1] + (-1, 4, 4))
        costs = 0
        for cost in self.cost_list:
            costs += cost.cost(points, x_points=x_points, **observation)
        return costs

    def eval(self, points: torch.Tensor, **observation) -> torch.Tensor:
        num1, num2, num_probe, dim = points.shape
        traj_dim = observation.get('traj_dim', None)
        assert dim == self.state_dim
        x_points = None
        if self.FK is not None:  # NOTE: only works with SE(3) FK for now
            x_points = self.FK(points.view(-1, self.state_dim)[:, :self.dim]).reshape(traj_dim[:-1] + (num2, num_probe, -1, 4, 4))
        costs = 0

        for cost in self.cost_list:
            costs += cost.eval(points, x_points=x_points, **observation)

        return costs.mean(-1)  # mean the probe point dim


class CostField(Cost):

    def __init__(
        self,
        dim: int,
        traj_range: Tuple[int, int],
        field: Callable = None,
        weight: float = 1.,
        tensor_args=None,
    ):
        super().__init__(dim, tensor_args=tensor_args)
        self.traj_range = traj_range
        self.field = field
        self.weight = weight

    def cost(self, points: torch.Tensor, x_points: torch.Tensor = None, **observation) -> torch.Tensor:
        traj_dim = observation.get('traj_dim', None)
        if self.field is None:
            return 0
        trajs = points.view(traj_dim)  # [..., traj_len, dim]
        states = trajs[..., :self.dim] if x_points is None else x_points
        field_cost = self.field(states, **observation) * self.weight
        return field_cost.view(traj_dim[:-1]).mean(-1)  # mean the traj_len

    def eval(self, points: torch.Tensor, x_points: torch.Tensor = None, **observation) -> torch.Tensor:
        traj_dim = observation.get('traj_dim', None)
        if self.field is None:
            return 0
        optim_dim = points.shape
        cost_dim = traj_dim[:-1] + optim_dim[1:3]  # [..., traj_len] + [nb2, num_probe]
        costs = torch.zeros(cost_dim, **self.tensor_args)
        points = points.view(cost_dim + traj_dim[-1:])
        states = points[..., self.traj_range[0]:self.traj_range[1], :, :, :self.dim] if x_points is None else x_points[..., self.traj_range[0]:self.traj_range[1], :, :, :, :, :]
        field_cost = self.field.compute_cost(states, **observation).view(cost_dim[:-3] + (self.traj_range[1] - self.traj_range[0], ) + cost_dim[-2:]) * self.weight
        costs[..., self.traj_range[0]:self.traj_range[1], :, :] = field_cost

        return costs.view((-1,) + optim_dim[1:3])


class CostGPHolonomic(Cost):

    def __init__(
        self,
        dim: int,
        traj_len: int,
        dt: float,
        sigma: float,
        probe_range: Tuple[int, int],
        weight: float = 1.,
        Q_c_inv: torch.Tensor = None,
        tensor_args=None,
    ):
        super().__init__(dim, tensor_args=tensor_args)
        self.dt = dt
        self.traj_len = traj_len
        self.phi = self.calc_phi()
        self.phi_T = self.phi.T
        if Q_c_inv is None:
            Q_c_inv = torch.eye(self.dim, **self.tensor_args) / sigma**2
        self.Q_c_inv = torch.zeros(self.traj_len - 1, self.dim, self.dim, **self.tensor_args) + Q_c_inv
        self.Q_inv = self.calc_Q_inv()
        self.single_Q_inv = self.Q_inv[[0]]
        self.probe_range = probe_range
        self.weight = weight

    def calc_phi(self) -> torch.Tensor:
        I = torch.eye(self.dim, **self.tensor_args)
        Z = torch.zeros(self.dim, self.dim, **self.tensor_args)
        phi_u = torch.cat((I, self.dt * I), dim=1)
        phi_l = torch.cat((Z, I), dim=1)
        phi = torch.cat((phi_u, phi_l), dim=0)
        return phi  # [dim, dim]

    def calc_Q_inv(self) -> torch.Tensor:
        m1 = 12. * (self.dt ** -3.) * self.Q_c_inv
        m2 = -6. * (self.dt ** -2.) * self.Q_c_inv
        m3 = 4. * (self.dt ** -1.) * self.Q_c_inv

        Q_inv_u = torch.cat((m1, m2), dim=-1)
        Q_inv_l = torch.cat((m2, m3), dim=-1)
        Q_inv  = torch.cat((Q_inv_u, Q_inv_l), dim=-2)
        return Q_inv

    def cost(self, points: torch.Tensor, x_points: torch.Tensor = None, **observation) -> torch.Tensor:
        traj_dim = observation.get('traj_dim', None)
        trajs = points.view(traj_dim)  # [..., traj_len, dim]
        errors = (trajs[..., 1:, :] - trajs[..., :-1, :] @ self.phi_T) * self.weight  # [..., traj_len-1, dim * 2]
        costs = torch.einsum('...ij,...ijk,...ik->...i', errors, self.single_Q_inv, errors)  # [..., traj_len-1]
        return costs.mean(dim=-1)  # mean the traj_len

    def eval(self, points: torch.Tensor, x_points: torch.Tensor = None, **observation) -> torch.Tensor:
        traj_dim = observation.get('traj_dim', None)
        current_trajs = observation.get('current_trajs', None)
        current_trajs = current_trajs.view(traj_dim)  # [..., traj_len, dim]
        current_trajs = current_trajs.unsqueeze(-2).unsqueeze(-2)  # [..., traj_len, 1, 1, dim]

        cost_dim = traj_dim[:-1] + points.shape[1:3]  # [..., traj_len] + [nb2, num_probe]
        costs = torch.zeros(cost_dim, **self.tensor_args)
        states = points
        probe_points = states[..., self.probe_range[0]:self.probe_range[1], :]  # [..., nb2, num_eval, dim * 2]
        len_probe = probe_points.shape[-2]
        probe_points = probe_points.view(traj_dim[:-1] + (points.shape[1], len_probe, self.state_dim,))  # [..., traj_len] + [nb2, num_eval, dim * 2]
        right_errors = probe_points[..., 1:self.traj_len, :, :, :] - current_trajs[..., 0:self.traj_len-1, :, :, :] @ self.phi_T # [..., traj_len-1, nb2, num_eval, dim * 2]
        left_errors = current_trajs[..., 1:self.traj_len, :, :, :] - probe_points[..., 0:self.traj_len-1, :, :, :] @ self.phi_T # [..., traj_len-1, nb2, num_eval, dim * 2]
        # mahalanobis distance
        left_cost_dist = torch.einsum('...ij,...ijk,...ik->...i', left_errors, self.single_Q_inv, left_errors) * (self.weight / 2)  # [..., traj_len-1, nb2, num_eval]
        right_cost_dist = torch.einsum('...ij,...ijk,...ik->...i', right_errors, self.single_Q_inv, right_errors) * (self.weight / 2)  # [..., traj_len-1, nb2, num_eval]
        # cost_dist = scale_cost_matrix(cost_dist)
        # cost_dist = torch.sqrt(cost_dist)
        costs[..., 0:self.traj_len-1, :, self.probe_range[0]:self.probe_range[1]] += left_cost_dist
        costs[..., 1:self.traj_len, :, self.probe_range[0]:self.probe_range[1]] += right_cost_dist

        return costs.view((-1,) + points.shape[1:3])
