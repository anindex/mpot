from typing import Any, Optional, Tuple, List, Callable
import torch
import numpy as np
from mpot.ot.sinkhorn_step import SinkhornStep, SinkhornStepState
from mpot.ot.sinkhorn import Sinkhorn
from mpot.utils.misc import MinMaxCenterScaler
from mpot.gp.mp_priors_multi import BatchGPPrior
from mpot.gp.gp_factor import GPFactor
from mpot.gp.unary_factor import UnaryFactor


class MPOT(object):

    def __init__(
        self,
        objective_fn: Any,
        linear_ot_solver: Sinkhorn,
        ss_params: dict,
        traj_len: int = 64,
        num_particles_per_goal: int = 16,
        dt: float = 0.02,
        start_state: torch.Tensor = None,
        multi_goal_states: torch.Tensor = None,
        initial_particle_means: torch.Tensor = None,
        pos_limits=[-10, 10],
        vel_limits=[-10, 10],
        polytope: str = 'orthoplex',
        sigma_start_init: float = 0.001,
        sigma_goal_init: float = 1.,
        sigma_gp_init: float = 0.001,
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

        self.start_state = start_state
        self.multi_goal_states = multi_goal_states
        if multi_goal_states is None:  # NOTE: if there is no goal, we assume here is at least one goal
            self.num_goals = 1
        else:
            assert multi_goal_states.ndim == 2
            self.num_goals = multi_goal_states.shape[0]
        self.num_particles_per_goal = num_particles_per_goal
        self.num_particles = num_particles_per_goal * self.num_goals
        self.polytope = polytope
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

        # init solver
        self.ss_params = ss_params
        self.sinkhorn_step = SinkhornStep(
            self.state_dim,
            objective_fn=objective_fn,
            linear_ot_solver=linear_ot_solver,
            state_scalers=[self.pos_scaler, self.vel_scaler],
            **self.ss_params,
        )

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
        if initial_particle_means is None:
            self.init_trajs = self.get_random_trajs()
        else:
            self.init_trajs = initial_particle_means.clone()
        self.flatten_trajs = self.init_trajs.flatten(0, 1)

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

    def optimize(self) -> Tuple[torch.Tensor, SinkhornStepState, int]:
        state = self.sinkhorn_step.init_state(self.flatten_trajs)
        iteration = 0
        while self.sinkhorn_step._continue(state, iteration):
            state = self.sinkhorn_step.step(state, iteration, traj_dim=self.traj_dim)
            iteration += 1

        trajs = state.X.view(self.traj_dim)
        return trajs, state, iteration
