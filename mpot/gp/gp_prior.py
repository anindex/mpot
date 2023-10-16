import torch
import torch.distributions as dist


class BatchGPPrior:

    def __init__(
            self,
            traj_len: int,
            dt: float,
            dim: int,
            K_s_inv: torch.Tensor,
            K_gp_inv: torch.Tensor,
            start_state: torch.Tensor,
            means: torch.Tensor = None,
            K_g_inv: torch.Tensor = None,
            goal_states: torch.Tensor = None,
            tensor_args=None,
    ):
        """
        Motion-Planning prior.

        reference: "Continuous-time Gaussian process motion planning via
        probabilistic inference", Mukadam et al. (IJRR 2018)

        Parameters
        ----------
        traj_len : int
            Planning horizon length (not including start state).
        dt :  float
            Time-step size.
        state_dim : int
            State state_dimension.
        K_s_inv : Tensor
            Start-state inverse covariance. Shape: [state_dim, state_dim]
        K_gp_inv :  Tensor
            Gaussian-process single-step inverse covariance i.e. 'Q_inv'.
            Assumed constant, meaning homoscedastic noise with constant step-size.
            Shape: [2 * state_dim, 2 * state_dim]
        start_state : Tensor
            Shape: [state_dim]
        (Optional) K_g_inv : Tensor
            Goal-state inverse covariance. Shape: [state_dim, state_dim]
        (Optional) goal_state :  Tensor
            Shape: [state_dim]
        (Optional) dim : int
            Degrees of freedom.
        """
        self.dim = dim
        self.state_dim = dim * 2
        self.traj_len = traj_len
        self.M = self.state_dim * (traj_len + 1)
        self.tensor_args = tensor_args

        self.goal_directed = (goal_states is not None)

        if means is None:
            self.num_modes = goal_states.shape[0] if self.goal_directed else 1
            means = self.get_const_vel_mean(
                            start_state,
                            goal_states,
                            dt,
                            traj_len,
                            dim)
        else:
            self.num_modes = means.shape[0]

        # Flatten mean trajectories
        self.means = means.reshape(self.num_modes, -1)

        # TODO: Add different goal Covariances
        # Assume same goal Cov. for now
        Sigma_inv = self.get_const_vel_covariance(
            dt,
            K_s_inv,
            K_gp_inv,
            K_g_inv,
        )

        # self.Sigma_inv = Sigma_inv
        self.Sigma_inv = Sigma_inv # + torch.eye(Sigma_inv.shape[0], **tensor_args) * 1.e-3
        self.Sigma_invs = self.Sigma_inv.repeat(self.num_modes, 1, 1)
        self.update_dist(self.means, self.Sigma_invs)

    def update_dist(
            self,
            means: torch.Tensor,
            Sigma_invs: torch.Tensor,
    ) -> torch.Tensor:
        # Create Multi-variate Normal Distribution
        self.dist = dist.MultivariateNormal(
            means,
            precision_matrix=Sigma_invs,
        )

    def get_mean(self, reshape: bool = True) -> torch.Tensor:
        if reshape:
            return self.means.clone().detach().reshape(
                self.num_modes, self.traj_len + 1, self.state_dim,
            )
        else:
            self.means.clone().detach()

    def set_mean(self, means: torch.Tensor) -> torch.Tensor:
        assert means.shape == self.means.shape
        self.means = means.clone().detach()
        self.update_dist(self.means, self.Sigma_invs)

    def set_Sigma_invs(self, Sigma_invs: torch.Tensor) -> torch.Tensor:
        assert Sigma_invs.shape == self.Sigma_invs.shape
        self.Sigma_invs = Sigma_invs.clone().detach()
        self.update_dist(self.means, self.Sigma_invs)

    def const_vel_trajectory(
        self,
        start_state: torch.Tensor,
        goal_state: torch.Tensor,
        dt: float,
        traj_len: int,
        dim: int,
    ) -> torch.Tensor:
        state_traj = torch.zeros(traj_len + 1, 2 * dim, **self.tensor_args)
        mean_vel = (goal_state[:dim] - start_state[:dim]) / (traj_len * dt)
        for i in range(traj_len + 1):
            state_traj[i, :dim] = start_state[:dim] * (traj_len - i) * 1. / traj_len \
                                  + goal_state[:dim] * i * 1./traj_len
        state_traj[:, dim:] = mean_vel.unsqueeze(0)
        return state_traj

    def get_const_vel_mean(
        self,
        start_state: torch.Tensor,
        goal_states: torch.Tensor,
        dt: float,
        traj_len: int,
        dim: int,
    ) -> torch.Tensor:

        # Make mean goal-directed if goal_state is provided.
        if self.goal_directed:
            means = []
            for i in range(self.num_modes):
                means.append(self.const_vel_trajectory(
                    start_state,
                    goal_states[i],
                    dt,
                    traj_len,
                    dim,
                ))
            return torch.stack(means, dim=0)
        else:
            return start_state.repeat(traj_len + 1, 1)

    def get_const_vel_covariance(
        self,
        dt: float,
        K_s_inv: torch.Tensor,
        K_gp_inv: torch.Tensor,
        K_g_inv: torch.Tensor,
        precision_matrix: bool = True,
    ) -> torch.Tensor:
        # Transition matrix
        Phi = torch.eye(self.state_dim, **self.tensor_args)
        Phi[:self.dim, self.dim:] = torch.eye(self.dim, **self.tensor_args) * dt
        diag_Phis = Phi
        for _ in range(self.traj_len - 1):
            diag_Phis = torch.block_diag(diag_Phis, Phi)

        A = torch.eye(self.M, **self.tensor_args)
        A[self.state_dim:, :-self.state_dim] += -1. * diag_Phis
        if self.goal_directed:
            b = torch.zeros(self.state_dim, self.M,  **self.tensor_args)
            b[:, -self.state_dim:] = torch.eye(self.state_dim,  **self.tensor_args)
            A = torch.cat((A, b))

        Q_inv = K_s_inv
        for _ in range(self.traj_len):
            Q_inv = torch.block_diag(Q_inv, K_gp_inv).to(**self.tensor_args)
        if self.goal_directed:
            Q_inv = torch.block_diag(Q_inv, K_g_inv).to(**self.tensor_args)

        K_inv = A.t() @ Q_inv @ A
        if precision_matrix:
            return K_inv
        else:
            return torch.inverse(K_inv)

    def sample(self, num_samples: int) -> torch.Tensor:
        return self.dist.sample((num_samples,)).view(
            num_samples, self.num_modes, self.traj_len + 1, self.state_dim,
        ).transpose(1, 0)

    def log_prob(self, X: torch.Tensor) -> torch.Tensor:
        return self.dist.log_prob(X)