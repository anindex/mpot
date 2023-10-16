from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple, List, Callable
import torch
import einops

from mpot.gp.field_factor import FieldFactor


class Cost(ABC):
    def __init__(self, robot, n_support_points, tensor_args=None, **kwargs):
        self.robot = robot
        self.n_dof = robot.q_dim
        self.dim = 2 * self.n_dof  # position + velocity
        self.n_support_points = n_support_points

        self.tensor_args = tensor_args

    def set_cost_factors(self):
        pass

    def __call__(self, trajs, **kwargs):
        return self.eval(trajs, **kwargs)

    @abstractmethod
    def eval(self, trajs, **kwargs):
        pass

    def get_q_pos_vel_and_fk_map(self, trajs, **kwargs):
        assert trajs.ndim == 3 or trajs.ndim == 4
        N = 1
        if trajs.ndim == 4:
            N, B, H, D = trajs.shape  # n_goals (or steps), batch of trajectories, length, dim
            trajs = einops.rearrange(trajs, 'N B H D -> (N B) H D')
        else:
            B, H, D = trajs.shape

        q_pos = self.robot.get_position(trajs)
        q_vel = self.robot.get_velocity(trajs)
        H_positions = self.robot.fk_map_collision(q_pos)  # I, taskspaces, x_dim+1, x_dim+1 (homogeneous transformation matrices)
        return trajs, q_pos, q_vel, H_positions


class CostField(Cost):

    def __init__(
        self,
        robot,
        n_support_points: int,
        field: Callable = None,
        sigma: float = 1.0,
        **kwargs
    ):
        super().__init__(robot, n_support_points, **kwargs)
        self.field = field
        self.sigma = sigma

        self.set_cost_factors()
    
    def set_cost_factors(self):
        # ========= Cost factors ===============
        self.obst_factor = FieldFactor(
            self.n_dof,
            self.sigma,
            [0, None]
        )

    def cost(self, trajs: torch.Tensor, **observation) -> torch.Tensor:
        traj_dim = observation.get('traj_dim', None)
        if self.field is None:
            return 0
        trajs = trajs.view(traj_dim)  # [..., traj_len, dim]
        states = trajs[..., :self.n_dof]
        field_cost = self.field.compute_cost(states, **observation)
        return field_cost.view(traj_dim[:-1]).mean(-1)  # mean the traj_len

    def eval(self, trajs: torch.Tensor, q_pos=None, q_vel=None, H_positions=None, **observation):
        optim_dim = observation.get('optim_dim')
        costs = 0
        if self.field is not None:
            # H_pos = link_pos_from_link_tensor(H)  # get translation part from transformation matrices
            H_pos = H_positions
            err_obst = self.obst_factor.get_error(
                trajs,
                self.field,
                q_pos=q_pos,
                q_vel=q_vel,
                H_pos=H_pos,
                calc_jacobian=False
            )
            w_mat = self.obst_factor.K
            obst_costs = w_mat * err_obst.mean(-1)
            costs = obst_costs.reshape(optim_dim[:2])

        return costs


class CostGPHolonomic(Cost):

    def __init__(
        self,
        robot,
        n_support_points: int,
        dt: float,
        sigma: float,
        probe_range: Tuple[int, int],
        Q_c_inv: torch.Tensor = None,
        **kwargs
    ):
        super().__init__(robot, n_support_points, **kwargs)
        self.dt = dt
        self.phi = self.calc_phi()
        self.phi_T = self.phi.T
        if Q_c_inv is None:
            Q_c_inv = torch.eye(self.n_dof, **self.tensor_args) / sigma**2
        self.Q_c_inv = torch.zeros(self.n_support_points - 1, self.n_dof, self.n_dof, **self.tensor_args) + Q_c_inv
        self.Q_inv = self.calc_Q_inv()
        self.single_Q_inv = self.Q_inv[[0]]
        self.probe_range = probe_range

    def calc_phi(self) -> torch.Tensor:
        I = torch.eye(self.n_dof, **self.tensor_args)
        Z = torch.zeros(self.n_dof, self.n_dof, **self.tensor_args)
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

    def cost(self, trajs: torch.Tensor, **observation) -> torch.Tensor:
        traj_dim = observation.get('traj_dim', None)
        trajs = trajs.view(traj_dim)  # [..., n_support_points, dim]
        errors = (trajs[..., 1:, :] - trajs[..., :-1, :] @ self.phi_T)  # [..., n_support_points-1, dim * 2]
        costs = torch.einsum('...ij,...ijk,...ik->...i', errors, self.single_Q_inv, errors)  # [..., n_support_points-1]
        return costs.mean(dim=-1)  # mean the n_support_points

    def eval(self, trajs: torch.Tensor, **observation) -> torch.Tensor:
        traj_dim = observation.get('traj_dim')
        optim_dim = observation.get('optim_dim')

        current_trajs = observation.get('current_trajs')
        current_trajs = current_trajs.view(traj_dim)  # [..., n_support_points, dim]
        current_trajs = current_trajs.unsqueeze(-2).unsqueeze(-2)  # [..., n_support_points, 1, 1, dim]

        cost_dim = traj_dim[:-1] + optim_dim[1:3]  # [..., n_support_points] + [nb2, num_probe]
        costs = torch.zeros(cost_dim, **self.tensor_args)
        states = trajs

        probe_points = states[..., self.probe_range[0]:self.probe_range[1], :]  # [..., nb2, num_eval, dim * 2]
        len_probe = probe_points.shape[-2]
        probe_points = probe_points.view(traj_dim[:-1] + (optim_dim[1], len_probe, self.dim,))  # [..., n_support_points] + [nb2, num_eval, dim * 2]
        right_errors = probe_points[..., 1:self.n_support_points, :, :, :] - current_trajs[..., 0:self.n_support_points-1, :, :, :] @ self.phi_T # [..., n_support_points-1, nb2, num_eval, dim * 2]
        left_errors = current_trajs[..., 1:self.n_support_points, :, :, :] - probe_points[..., 0:self.n_support_points-1, :, :, :] @ self.phi_T # [..., n_support_points-1, nb2, num_eval, dim * 2]
        # mahalanobis distance
        left_cost_dist = torch.einsum('...ij,...ijk,...ik->...i', left_errors, self.single_Q_inv, left_errors)  # [..., n_support_points-1, nb2, num_eval]
        right_cost_dist = torch.einsum('...ij,...ijk,...ik->...i', right_errors, self.single_Q_inv, right_errors)  # [..., n_support_points-1, nb2, num_eval]

        costs[..., 0:self.n_support_points-1, :, self.probe_range[0]:self.probe_range[1]] += left_cost_dist
        costs[..., 1:self.n_support_points, :, self.probe_range[0]:self.probe_range[1]] += right_cost_dist
        costs = costs.view(optim_dim).mean(dim=-1)  # mean the probe

        return costs


class CostComposite(Cost):

    def __init__(
        self,
        robot,
        n_support_points,
        cost_list,
        weights_cost_l=None,
        **kwargs
    ):
        super().__init__(robot, n_support_points, **kwargs)
        self.cost_l = cost_list
        self.weight_cost_l = weights_cost_l if weights_cost_l is not None else [1.0] * len(cost_list)

    def eval(self, trajs, trajs_interpolated=None, return_invidual_costs_and_weights=False, **kwargs):
        trajs, q_pos, q_vel, H_positions = self.get_q_pos_vel_and_fk_map(trajs)

        if not return_invidual_costs_and_weights:
            cost_total = 0
            for cost, weight_cost in zip(self.cost_l, self.weight_cost_l):
                if trajs_interpolated is not None:
                    trajs_tmp = trajs_interpolated
                else:
                    trajs_tmp = trajs
                cost_tmp = weight_cost * cost(trajs_tmp, q_pos=q_pos, q_vel=q_vel, H_positions=H_positions, **kwargs)
                cost_total += cost_tmp
            return cost_total
        else:
            cost_l = []
            for cost in self.cost_l:
                if trajs_interpolated is not None:
                    # Compute only collision costs with interpolated trajectories.
                    # Other costs are computed with non-interpolated trajectories, e.g. smoothness
                    trajs_tmp = trajs_interpolated
                else:
                    trajs_tmp = trajs

                cost_tmp = cost(trajs_tmp, q_pos=q_pos, q_vel=q_vel, H_positions=H_positions, **kwargs)
                cost_l.append(cost_tmp)

            if return_invidual_costs_and_weights:
                return cost_l, self.weight_cost_l
