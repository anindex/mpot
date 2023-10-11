from typing import Union
import torch


class Epsilon:
    """Epsilon scheduler for Sinkhorn and Sinkhorn Step."""

    def __init__(
        self,
        target: float = 0.1,
        scale_epsilon: float = 1.0,
        init: float = 1.0,
        decay: float = 1.0
    ):
        self._target_init = target
        self._scale_epsilon = scale_epsilon
        self._init = init
        self._decay = decay

    @property
    def target(self) -> float:
        """Return the final regularizer value of scheduler."""
        target = 5e-2 if self._target_init is None else self._target_init
        scale = 1.0 if self._scale_epsilon is None else self._scale_epsilon
        return scale * target

    def at(self, iteration: int = 1) -> float:
        """Return (intermediate) regularizer value at a given iteration."""
        if iteration is None:
            return self.target
        # check the decay is smaller than 1.0.
        decay = min(self._decay, 1.0)
        # the multiple is either 1.0 or a larger init value that is decayed.
        multiple = max(self._init * (decay ** iteration), 1.0)
        return multiple * self.target

    def done(self, eps: float) -> bool:
        """Return whether the scheduler is done at a given value."""
        return eps == self.target

    def done_at(self, iteration: int) -> bool:
        """Return whether the scheduler is done at a given iteration."""
        return self.done(self.at(iteration))
    

class LinearEpsilon(Epsilon):

    def __init__(self, target: float = 0.1, 
                 scale_epsilon: float = 1, 
                 init: float = 1, 
                 decay: float = 1):
        super().__init__(target, scale_epsilon, init, decay)
    
    def at(self, iteration: int = 1) -> float:
        if iteration is None:
            return self.target
        
        eps = max(self._init - self._decay * iteration, self.target)
        return eps * self._scale_epsilon


class LinearProblem():

    def __init__(
        self,
        C: torch.Tensor,
        epsilon: Union[Epsilon, float] = 0.01,
        a: torch.Tensor = None,
        b: torch.Tensor = None,
        tau_a: float = 1.0,
        tau_b: float = 1.0,
        scaling_cost: bool = True,
    ) -> None:
        if scaling_cost:
            C = scale_cost_matrix(C)
        self.C = C
        self.epsilon = epsilon
        self.a = a if a is not None else (torch.ones(C.shape[0]).type_as(C) / C.shape[0])
        self.b = b if b is not None else (torch.ones(C.shape[1]).type_as(C) / C.shape[1])
        self.tau_a = tau_a
        self.tau_b = tau_b

    def potential_from_scaling(self, scaling: torch.Tensor) -> torch.Tensor:
        """Compute dual potential vector from scaling vector.

        Args:
        scaling: vector.

        Returns:
        a vector of the same size.
        """
        eps = self.epsilon.target if isinstance(self.epsilon, Epsilon) else self.epsilon
        return eps * torch.log(scaling)

    def marginal_from_potentials(
        self, f: torch.Tensor, g: torch.Tensor, dim: int
    ) -> torch.Tensor:
        eps = self.epsilon.target if isinstance(self.epsilon, Epsilon) else self.epsilon
        h = (f if dim == 1 else g)
        z = self.apply_lse_kernel(f, g, eps, dim=dim)
        return torch.exp((z + h) / eps)
    
    def update_potential(
        self, f: torch.Tensor, g: torch.Tensor, log_marginal: torch.Tensor,
        iteration: int = None, dim: int = 0,
    ) -> torch.Tensor:
        eps = self.epsilon.at(iteration) if isinstance(self.epsilon, Epsilon) else self.epsilon
        app_lse = self.apply_lse_kernel(f, g, eps, dim=dim)
        return eps * log_marginal - torch.where(torch.isfinite(app_lse), app_lse, 0)

    def transport_from_potentials(
        self, f: torch.Tensor, g: torch.Tensor
    ) -> torch.Tensor:
        """Output transport matrix from potentials."""
        eps = self.epsilon.target if isinstance(self.epsilon, Epsilon) else self.epsilon
        return torch.exp(self._center(f, g) / eps)

    def apply_lse_kernel(
        self, f: torch.Tensor, g: torch.Tensor, eps: float, dim: int
    ) -> torch.Tensor:
        w_res = self._softmax(f, g, eps, dim=dim)
        remove = f if dim == 1 else g
        return w_res - torch.where(torch.isfinite(remove), remove, 0)

    def _center(self, f: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        return f.unsqueeze(1) + g.unsqueeze(0) - self.C

    def _softmax(
        self, f: torch.Tensor, g: torch.Tensor, eps: float, dim: int
    ) -> torch.Tensor:
        """Apply softmax row or column wise"""

        lse_output = torch.logsumexp(
            self._center(f, g) / eps, dim=dim
        )
        return eps * lse_output


class SinkhornState():
    """Holds the state variables used to solve OT with Sinkhorn."""

    def __init__(
        self,
        errors: torch.Tensor = None,
        fu: torch.Tensor = None,
        gv: torch.Tensor = None,
    ):
        self.errors = errors
        self.fu = fu
        self.gv = gv
        self.converged = False

    def solution_error(
        self,
        ot_prob: LinearProblem,
        parallel_dual_updates: bool,
    ) -> torch.Tensor:
        """State dependent function to return error."""
        fu, gv = self.fu, self.gv

        return solution_error(
            fu,
            gv,
            ot_prob,
            parallel_dual_updates=parallel_dual_updates
        )

    def ent_reg_cost(
        self, ot_prob: LinearProblem,
    ) -> float:
        return ent_reg_cost(self.fu, self.gv, ot_prob)
    

class SinkhornStepState():
    """Holds the state of the barycenter solver.

    Args:
        costs: Holds the sequence of regularized GW costs seen through the outer
        loop of the solver.
        linear_convergence: Holds the sequence of bool convergence flags of the
        inner Sinkhorn iterations.
        X: optimizing points.
        a: weights of the barycenter. (not using)
    """

    def __init__(self,
                 X_init: torch.Tensor,
                 costs: torch.Tensor = None,
                 linear_convergence: torch.Tensor = None,
                 objective_vals: torch.Tensor = None,
                 X_history: torch.Tensor = None,
                 displacement_sqnorms: torch.Tensor = None,
                 a: torch.Tensor = None) -> None:
        self.X = X_init
        self.costs = costs
        self.linear_convergence = linear_convergence
        self.objective_vals = objective_vals
        self.X_history = X_history
        self.displacement_sqnorms = displacement_sqnorms
        self.a = a


def scale_cost_matrix(M: torch.Tensor) -> torch.Tensor:
    min_M = M.min()
    if min_M < 0:
        M -= min_M
    max_M = M.max()
    if max_M > 1.:
        M /= max_M   # for stability
    return M


def phi_star(h: torch.Tensor, rho: float) -> torch.Tensor:
  """Legendre transform of KL, :cite:`sejourne:19`, p. 9."""
  return rho * (torch.exp(h / rho) - 1)


def rho(epsilon: float, tau: float) -> float:
  return (epsilon * tau) / (1. - tau)


def derivative_phi_star(f: torch.Tensor, rho: float) -> torch.Tensor:
  return torch.exp(f / rho)


def grad_of_marginal_fit(
    c: torch.Tensor, h: torch.Tensor, tau: float, epsilon: float
) -> torch.Tensor:
  if tau == 1.0:
    return c
  r = rho(epsilon, tau)
  return torch.where(c > 0, c * derivative_phi_star(-h, r), 0.0)


def solution_error(
    f_u: torch.Tensor,
    g_v: torch.Tensor,
    ot_prob: LinearProblem,
    parallel_dual_updates: bool,
) -> torch.Tensor:
    """Compute error between Sinkhorn solution and target marginals."""
    if not parallel_dual_updates:
        return marginal_error(
            f_u, g_v, ot_prob.b, ot_prob, dim=0
        )

    grad_a = grad_of_marginal_fit(
        ot_prob.a, f_u, ot_prob.tau_a, ot_prob.epsilon
    )
    grad_b = grad_of_marginal_fit(
        ot_prob.b, g_v, ot_prob.tau_b, ot_prob.epsilon
    )

    err = marginal_error(f_u, g_v, grad_a, ot_prob, dim=1)
    err += marginal_error(f_u, g_v, grad_b, ot_prob, dim=0)
    return err


def marginal_error(
    f_u: torch.Tensor,
    g_v: torch.Tensor,
    target: torch.Tensor,
    ot_prob: LinearProblem,
    dim: int = 0
) -> torch.Tensor:
    """Output how far Sinkhorn solution is w.r.t target.

    Args:
        f_u: a vector of potentials or scalings for the first marginal.
        g_v: a vector of potentials or scalings for the second marginal.
        target: target marginal.
        dim: dim (0 or 1) along which to compute marginal.

    Returns:
        Array of floats, quantifying difference between target / marginal.
    """
    marginal = ot_prob.marginal_from_potentials(f_u, g_v, dim=dim)
    # L1 distance between target and marginal
    return torch.sum(
        torch.abs(marginal - target)
    )


def ent_reg_cost(
    f: torch.Tensor, g: torch.Tensor, ot_prob: LinearProblem,
) -> float:

    supp_a = ot_prob.a > 0
    supp_b = ot_prob.b > 0
    fa = ot_prob.potential_from_scaling(ot_prob.a)
    if ot_prob.tau_a == 1.0:
        div_a = torch.sum(torch.where(supp_a, ot_prob.a * (f - fa), 0.0))
    else:
        rho_a = rho(ot_prob.epsilon, ot_prob.tau_a)
        div_a = -torch.sum(
            torch.where(supp_a, ot_prob.a * phi_star(-(f - fa), rho_a), 0.0)
        )

    gb = ot_prob.potential_from_scaling(ot_prob.b)
    if ot_prob.tau_b == 1.0:
        div_b = torch.sum(torch.where(supp_b, ot_prob.b * (g - gb), 0.0))
    else:
        rho_b = rho(ot_prob.epsilon, ot_prob.tau_b)
        div_b = -torch.sum(
            torch.where(supp_b, ot_prob.b * phi_star(-(g - gb), rho_b), 0.0)
        )

    # Using https://arxiv.org/pdf/1910.12958.pdf (24)
    total_sum = torch.sum(ot_prob.marginal_from_potentials(f, g))
    return div_a + div_b + ot_prob.epsilon * (
        torch.sum(ot_prob.a) * torch.sum(ot_prob.b) - total_sum
    )