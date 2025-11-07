from typing import List
import torch

# =============== Scriptable helpers (no LinearProblem types) ===============

@torch.jit.script
def scale_cost_matrix(M: torch.Tensor) -> torch.Tensor:
    mn = torch.min(M)
    if bool(mn < 0):
        M = M - mn
    mx = torch.max(M)
    if bool(mx > 1.0):
        M = M / mx
    return M

@torch.jit.script
def rho(epsilon: float, tau: float) -> float:
    return (epsilon * tau) / (1.0 - tau)

@torch.jit.script
def phi_star(h: torch.Tensor, rho_val: float) -> torch.Tensor:
    # Legendre transform of KL
    return torch.tensor(rho_val, dtype=h.dtype, device=h.device) * (torch.exp(h / rho_val) - 1.0)

@torch.jit.script
def derivative_phi_star(f: torch.Tensor, rho_val: float) -> torch.Tensor:
    return torch.exp(f / rho_val)

@torch.jit.script
def grad_of_marginal_fit(c: torch.Tensor, h: torch.Tensor, tau: float, epsilon: float) -> torch.Tensor:
    if tau == 1.0:
        return c
    r = rho(epsilon, tau)
    return torch.where(c > 0, c * derivative_phi_star(-h, r), torch.zeros_like(c))


# =========================== Scripted classes ==============================

@torch.jit.script
class EpsilonScheduler:
    # mode 0: multiplicative; mode 1: linear
    target_init: float
    scale_epsilon: float
    init: float
    decay: float
    mode: int

    def __init__(self, target: float = 0.1, scale_epsilon: float = 1.0,
                 init: float = 1.0, decay: float = 1.0, mode: int = 0):
        self.target_init = float(target)
        self.scale_epsilon = float(scale_epsilon)
        self.init = float(init)
        self.decay = float(decay)
        self.mode = int(mode)

    def target(self) -> float:
        return self.scale_epsilon * self.target_init

    def at(self, iteration: int) -> float:
        if iteration < 0:
            return self.target()
        if self.mode == 0:
            d = self.decay if self.decay <= 1.0 else 1.0
            mult = self.init * (d ** float(iteration))
            if mult < 1.0:
                mult = 1.0
            return mult * self.target()
        else:
            eps = self.init - self.decay * float(iteration)
            t = self.target_init
            if eps < t:
                eps = t
            return eps * self.scale_epsilon

    def done(self, eps_val: float) -> bool:
        return float(eps_val) == self.target()

    def done_at(self, iteration: int) -> bool:
        return self.done(self.at(iteration))


@torch.jit.script
class LinearProblem:
    C: torch.Tensor
    epsilon: EpsilonScheduler
    a: torch.Tensor
    b: torch.Tensor
    tau_a: float
    tau_b: float

    def __init__(self, C: torch.Tensor, epsilon: EpsilonScheduler,
                 a: torch.Tensor, b: torch.Tensor,
                 tau_a: float = 1.0, tau_b: float = 1.0,
                 scaling_cost: bool = True):
        if scaling_cost:
            C = scale_cost_matrix(C)
        self.C = C
        self.epsilon = epsilon

        self.a = a
        self.b = b

        self.tau_a = float(tau_a)
        self.tau_b = float(tau_b)

    # ---- kernels / utilities ----

    def potential_from_scaling(self, scaling: torch.Tensor) -> torch.Tensor:
        eps = self.epsilon.target()
        return eps * torch.log(scaling)

    def _center(self, f: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        return f.unsqueeze(1) + g.unsqueeze(0) - self.C

    def _softmax(self, f: torch.Tensor, g: torch.Tensor, eps: float, dim: int) -> torch.Tensor:
        lse = torch.logsumexp(self._center(f, g) / eps, dim=dim)
        return eps * lse

    def apply_lse_kernel(self, f: torch.Tensor, g: torch.Tensor, eps: float, dim: int) -> torch.Tensor:
        w_res = self._softmax(f, g, eps, dim)
        remove = f if dim == 1 else g
        return w_res - torch.where(torch.isfinite(remove), remove, torch.zeros_like(remove))

    def marginal_from_potentials(self, f: torch.Tensor, g: torch.Tensor, dim: int) -> torch.Tensor:
        eps = self.epsilon.target()
        h = f if dim == 1 else g
        z = self.apply_lse_kernel(f, g, eps, dim)
        return torch.exp((z + h) / eps)

    def update_potential(self, f: torch.Tensor, g: torch.Tensor, log_marginal: torch.Tensor,
                         iteration: int, dim: int) -> torch.Tensor:
        eps = self.epsilon.at(iteration)
        app_lse = self.apply_lse_kernel(f, g, eps, dim)
        return eps * log_marginal - torch.where(torch.isfinite(app_lse), app_lse, torch.zeros_like(app_lse))

    def transport_from_potentials(self, f: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        eps = self.epsilon.target()
        return torch.exp(self._center(f, g) / eps)

    # ---- moved former free functions to methods (no forward refs) ----

    def marginal_error(self, f_u: torch.Tensor, g_v: torch.Tensor,
                       target: torch.Tensor, dim: int) -> torch.Tensor:
        marginal = self.marginal_from_potentials(f_u, g_v, dim)
        return torch.sum(torch.abs(marginal - target))

    def solution_error(self, f_u: torch.Tensor, g_v: torch.Tensor,
                       parallel_dual_updates: bool) -> torch.Tensor:
        if not parallel_dual_updates:
            return self.marginal_error(f_u, g_v, self.b, 0)

        eps_val = self.epsilon.target()
        grad_a = grad_of_marginal_fit(self.a, f_u, self.tau_a, eps_val)
        grad_b = grad_of_marginal_fit(self.b, g_v, self.tau_b, eps_val)
        err = self.marginal_error(f_u, g_v, grad_a, 1)
        err = err + self.marginal_error(f_u, g_v, grad_b, 0)
        return err

    def ent_reg_cost(self, f: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        supp_a = self.a > 0
        supp_b = self.b > 0

        fa = self.potential_from_scaling(self.a)
        gb = self.potential_from_scaling(self.b)

        zeros_a = torch.zeros_like(self.a)
        zeros_b = torch.zeros_like(self.b)

        if self.tau_a == 1.0:
            div_a = torch.sum(torch.where(supp_a, self.a * (f - fa), zeros_a))
        else:
            rho_a = rho(self.epsilon.target(), self.tau_a)
            div_a = -torch.sum(torch.where(supp_a, self.a * phi_star(-(f - fa), rho_a), zeros_a))

        if self.tau_b == 1.0:
            div_b = torch.sum(torch.where(supp_b, self.b * (g - gb), zeros_b))
        else:
            rho_b = rho(self.epsilon.target(), self.tau_b)
            div_b = -torch.sum(torch.where(supp_b, self.b * phi_star(-(g - gb), rho_b), zeros_b))

        total_sum = torch.sum(self.marginal_from_potentials(f, g, 0))
        eps_t = self.epsilon.target()
        return div_a + div_b + eps_t * (torch.sum(self.a) * torch.sum(self.b) - total_sum)


@torch.jit.script
class SinkhornState:
    errors: torch.Tensor
    fu: torch.Tensor
    gv: torch.Tensor
    converged_at: int

    def __init__(self, errors: torch.Tensor, fu: torch.Tensor, gv: torch.Tensor):
        self.errors = errors
        self.fu = fu
        self.gv = gv
        self.converged_at = -1

    def solution_error(self, ot_prob: LinearProblem, parallel_dual_updates: bool) -> torch.Tensor:
        return ot_prob.solution_error(self.fu, self.gv, parallel_dual_updates)

    def ent_reg_cost(self, ot_prob: LinearProblem) -> torch.Tensor:
        return ot_prob.ent_reg_cost(self.fu, self.gv)


@torch.jit.script
class SinkhornStepState:
    X: torch.Tensor
    costs: torch.Tensor
    linear_convergence: torch.Tensor
    objective_vals: torch.Tensor
    X_history: torch.Tensor
    displacement_sqnorms: torch.Tensor
    a: torch.Tensor

    def __init__(self, X_init: torch.Tensor, costs: torch.Tensor,
                 linear_convergence: torch.Tensor, # objective_vals: torch.Tensor,
                 X_history: torch.Tensor, displacement_sqnorms: torch.Tensor,
                 a: torch.Tensor):
        self.X = X_init
        self.costs = costs
        self.linear_convergence = linear_convergence
        # self.objective_vals = objective_vals
        self.X_history = X_history
        self.displacement_sqnorms = displacement_sqnorms
        self.a = a
