import torch
import torch.distributions as dist


class BatchGPPrior:
    """Gaussian Process prior for motion planning trajectories.

    Reference: "Continuous-time Gaussian process motion planning via
    probabilistic inference", Mukadam et al. (IJRR 2018)
    """

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
        Parameters
        ----------
        traj_len : int
            Planning horizon length (not including start state).
        dt : float
            Time-step size.
        dim : int
            Degrees of freedom (position only).
        K_s_inv : Tensor
            Start-state inverse covariance. Shape: [state_dim, state_dim]
        K_gp_inv : Tensor
            GP single-step inverse covariance (Q_inv).
            Shape: [2 * dim, 2 * dim]
        start_state : Tensor
            Shape: [dim]
        K_g_inv : Tensor, optional
            Goal-state inverse covariance. Shape: [state_dim, state_dim]
        goal_states : Tensor, optional
            Shape: [num_goals, dim]
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
                start_state, goal_states, dt, traj_len, dim)
        else:
            self.num_modes = means.shape[0]

        # Flatten mean trajectories
        self.means = means.reshape(self.num_modes, -1)

        if self.goal_directed and K_g_inv is None:
            raise ValueError("K_g_inv must be provided when goal_states is given")

        Sigma_inv = self.get_const_vel_covariance(
            dt, K_s_inv, K_gp_inv, K_g_inv)

        self.Sigma_inv = Sigma_inv
        self.Sigma_invs = self.Sigma_inv.repeat(self.num_modes, 1, 1)
        self.update_dist(self.means, self.Sigma_invs)

    def update_dist(self, means: torch.Tensor, Sigma_invs: torch.Tensor) -> None:
        self.dist = dist.MultivariateNormal(
            means, precision_matrix=Sigma_invs)

    def get_mean(self, reshape: bool = True) -> torch.Tensor:
        if reshape:
            return self.means.clone().detach().reshape(
                self.num_modes, self.traj_len + 1, self.state_dim)
        else:
            return self.means.clone().detach()

    def set_mean(self, means: torch.Tensor) -> None:
        assert means.shape == self.means.shape
        self.means = means.clone().detach()
        self.update_dist(self.means, self.Sigma_invs)

    def set_Sigma_invs(self, Sigma_invs: torch.Tensor) -> None:
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
        # Vectorized position interpolation
        t = torch.linspace(0, 1, traj_len + 1, **self.tensor_args)
        state_traj[:, :dim] = (1 - t).unsqueeze(-1) * start_state[:dim] + t.unsqueeze(-1) * goal_state[:dim]
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
        if self.goal_directed:
            means = []
            for i in range(self.num_modes):
                means.append(self.const_vel_trajectory(
                    start_state, goal_states[i], dt, traj_len, dim))
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

        # Build block-diagonal Phi matrix efficiently
        total_size = self.state_dim * self.traj_len
        diag_Phis = torch.zeros(total_size, total_size, **self.tensor_args)
        for i in range(self.traj_len):
            s = i * self.state_dim
            e = s + self.state_dim
            diag_Phis[s:e, s:e] = Phi

        A_rows = self.M
        if self.goal_directed:
            A_rows += self.state_dim
        A = torch.eye(self.M, **self.tensor_args)
        A[self.state_dim:self.M, :self.M - self.state_dim] -= diag_Phis[:self.M - self.state_dim, :self.M - self.state_dim]
        if self.goal_directed:
            b = torch.zeros(self.state_dim, self.M, **self.tensor_args)
            b[:, -self.state_dim:] = torch.eye(self.state_dim, **self.tensor_args)
            A = torch.cat((A, b))

        # Build block-diagonal Q_inv efficiently
        q_blocks = [K_s_inv]
        for _ in range(self.traj_len):
            q_blocks.append(K_gp_inv)
        if self.goal_directed:
            q_blocks.append(K_g_inv)
        Q_inv = torch.block_diag(*q_blocks).to(**self.tensor_args)

        K_inv = A.t() @ Q_inv @ A
        if precision_matrix:
            return K_inv
        else:
            return torch.linalg.inv(K_inv)

    def sample(self, num_samples: int) -> torch.Tensor:
        return self.dist.sample((num_samples,)).view(
            num_samples, self.num_modes, self.traj_len + 1, self.state_dim,
        ).transpose(1, 0)

    def log_prob(self, X: torch.Tensor) -> torch.Tensor:
        return self.dist.log_prob(X)