from typing import Any, Optional, Tuple, List, Callable
import torch
import numpy as np
from mpot.ot.optimizer import optimize
from mpot.utils.polytopes import POLYTOPE_MAP
from mpot.utils.misc import MinMaxCenterScaler
from mpot.gp.mp_priors_multi import BatchGPPrior
from mpot.gp.gp_factor import GPFactor
from mpot.gp.unary_factor import UnaryFactor


class MPOT(object):

    def __init__(
            self,
            traj_len: int,
            num_particles_per_goal: int,
            method_params: dict,
            dt: float = 0.02,
            step_radius: float = 0.1,
            probe_radius: float = 0.2,
            num_bpoint: int =100,
            start_state: torch.Tensor = None,
            multi_goal_states: torch.Tensor = None,
            initial_particle_means: torch.Tensor = None,
            pos_limits=[-10, 10],
            vel_limits=[-10, 10],
            random_init: bool = True,
            polytope: str = 'orthoplex',
            random_step: bool = False,
            annealing: bool =False,
            eps_annealing: float =0.05,
            cost: Callable = None,
            sigma_start_init: float = None,
            sigma_goal_init: float = None,
            sigma_gp_init: float = None,
            seed: int = 0,
            tensor_args=None,
            **kwargs
    ):
        self.traj_len = traj_len
        self.dt = dt
        self.seed = seed
        if tensor_args is None:
            tensor_args = {'device': torch.device('cpu'), 'dtype': torch.float32}
        self.tensor_args = tensor_args
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.num_bpoint = num_bpoint
        self.step_radius = step_radius
        self.probe_radius = probe_radius
        self.method_params = method_params
        self.start_state = start_state
        self.multi_goal_states = multi_goal_states
        if multi_goal_states is None:  # NOTE: if there is no goal, we assume here is at least one goal
            self.num_goals = 1
        else:
            assert multi_goal_states.ndim == 2
            self.num_goals = multi_goal_states.shape[0]
        self.num_particles_per_goal = num_particles_per_goal
        self.num_particles = num_particles_per_goal * self.num_goals
        self.random_init = random_init
        self.polytope = polytope
        self.random_step = random_step
        self.annealing = annealing
        self.eps_annealing = eps_annealing
        self.cost = cost
        self.sigma_start_init = sigma_start_init
        self.sigma_goal_init = sigma_goal_init
        self.sigma_gp_init = sigma_gp_init
        self._traj_dist = None

        self.reset(initial_particle_means=initial_particle_means)
        # scaling operations
        if isinstance(pos_limits, torch.Tensor):
            self.pos_limits = pos_limits.clone().to(**self.tensor_args)
        else:
            self.pos_limits = torch.tensor(pos_limits, **self.tensor_args)
        if self.pos_limits.ndim == 1:
            self.pos_limits = self.pos_limits.unsqueeze(0).repeat(self.dim, 1)
        self.pos_scaler = MinMaxCenterScaler(dim_range=[0, self.dim], min=self.pos_limits[:, 0], max=self.pos_limits[:, 1])
        if isinstance(vel_limits, torch.Tensor):
            self.vel_limits = vel_limits.clone().to(**self.tensor_args)
        else:
            self.vel_limits = torch.tensor(vel_limits, **self.tensor_args)
        if self.vel_limits.ndim == 1:
            self.vel_limits = self.vel_limits.unsqueeze(0).repeat(self.dim, 1)
        self.vel_scaler = MinMaxCenterScaler(dim_range=[self.dim, self.state_dim], min=self.vel_limits[:, 0], max=self.vel_limits[:, 1])

    def reset(
            self,
            start_state: torch.Tensor = None,
            multi_goal_states: torch.Tensor = None,
            initial_particle_means: torch.Tensor = None,
    ):
        if start_state is not None:
            self.start_state = start_state.detach().clone()

        if multi_goal_states is not None:
            self.multi_goal_states = multi_goal_states.detach().clone()

        self.dim = self.start_state.shape[-1]
        self.state_dim = self.dim * 2
        self.get_prior_dists(initial_particle_means=initial_particle_means)

    def get_prior_dists(self, initial_particle_means: torch.Tensor = None):
        origin = torch.zeros(self.state_dim, **self.tensor_args)
        if self.random_step:
            self.step_dist = None
            self.step_weights = None
        else:
            self.step_dist = POLYTOPE_MAP[self.polytope](origin, self.step_radius, num_points=self.num_bpoint)
            self.step_weights = torch.ones(self.step_dist.shape[0], **self.tensor_args) / self.step_dist.shape[0]

        if initial_particle_means is None:
            if self.random_init:
                self.init_trajs = self.get_random_trajs()
            else:
                start, end = self.start_state.unsqueeze(0), self.multi_goal_states
                self.init_trajs = self.const_vel_trajectories(start, end)
                # copy traj for each particle
                self.init_trajs = self.init_trajs.unsqueeze(1).repeat(1, self.num_particles_per_goal, 1, 1)
                self.traj_dim = self.init_trajs.shape
                self.init_trajs = self.init_trajs.flatten(0, 1)
        else:
            self.init_trajs = initial_particle_means.clone()
        self.flatten_trajs = self.init_trajs.flatten(0, 1)

    def const_vel_trajectories(
        self,
        start_state: torch.Tensor,
        multi_goal_states: torch.Tensor,
    ) -> torch.Tensor:
        traj_dim = (multi_goal_states.shape[0], self.traj_len, self.state_dim)
        state_traj = torch.zeros(traj_dim, **self.tensor_args)
        mean_vel = (multi_goal_states[:, :self.dim] - start_state[:, :self.dim]) / (self.traj_len * self.dt)
        for i in range(self.traj_len):
            state_traj[:, i, :self.dim] = start_state[:, :self.dim] * (self.traj_len - i - 1) / (self.traj_len - 1) \
                                  + multi_goal_states[:, :self.dim] * i / (self.traj_len - 1)
        state_traj[:, :, self.dim:] = mean_vel.unsqueeze(1).repeat(1, self.traj_len, 1)
        return state_traj

    def get_GP_prior(
            self,
            start_K: torch.Tensor,
            gp_K: torch.Tensor,
            goal_K: torch.Tensor,
            state_init: torch.Tensor,
            particle_means: torch.Tensor = None,
            goal_states: torch.Tensor = None,
            tensor_args=None,
    ) -> BatchGPPrior:
        if tensor_args is None:
            tensor_args = self.tensor_args
        return BatchGPPrior(
            self.traj_len - 1,
            self.dt,
            self.dim,
            start_K,
            gp_K,
            state_init,
            means=particle_means,
            K_g_inv=goal_K,
            goal_states=goal_states,
            tensor_args=tensor_args,
        )

    def get_random_trajs(self) -> torch.Tensor:
        # force torch.float64
        tensor_args = dict(dtype=torch.float64, device=self.tensor_args['device'])
        # set zero velocity for GP prior
        start_state = torch.cat((self.start_state, torch.zeros_like(self.start_state)), dim=-1).to(**tensor_args)
        if self.multi_goal_states is not None:
            multi_goal_states = torch.cat((self.multi_goal_states, torch.zeros_like(self.multi_goal_states)), dim=-1).to(**tensor_args)
        else:
            multi_goal_states = None
        #========= Initialization factors ===============
        self.start_prior_init = UnaryFactor(
            self.dim * 2,
            self.sigma_start_init,
            start_state,
            tensor_args=tensor_args,
        )

        self.gp_prior_init = GPFactor(
            self.dim,
            self.sigma_gp_init,
            self.dt,
            self.traj_len - 1,
            tensor_args=tensor_args,
        )

        self.multi_goal_prior_init = []
        if multi_goal_states is not None:
            for i in range(self.num_goals):
                self.multi_goal_prior_init.append(
                    UnaryFactor(
                        self.dim * 2,
                        self.sigma_goal_init,
                        multi_goal_states[i],
                        tensor_args=tensor_args,
                    )
                )
        self._traj_dist = self.get_GP_prior(
                self.start_prior_init.K,
                self.gp_prior_init.Q_inv[0],
                self.multi_goal_prior_init[0].K if multi_goal_states is not None else None,
                start_state,
                goal_states=multi_goal_states,
                tensor_args=tensor_args,
            )
        particles = self._traj_dist.sample(self.num_particles_per_goal).to(**tensor_args)
        self.traj_dim = particles.shape
        del self._traj_dist  # free memory
        return particles.flatten(0, 1).to(**self.tensor_args)

    def optimize(self, **kwargs) -> Tuple[torch.Tensor, dict]:
        trajs, log_dict = optimize(self.step_dist, self.step_weights, self.flatten_trajs, self.cost, self.step_radius, self.probe_radius,
                                   polytope=self.polytope, num_sphere_point=self.num_bpoint, annealing=self.annealing, eps_annealing=self.eps_annealing,
                                   traj_dim=self.traj_dim, pos_scaler=self.pos_scaler, vel_scaler=self.vel_scaler,
                                   **self.method_params, **kwargs)
        trajs = trajs.view(self.traj_dim)
        return trajs, log_dict
