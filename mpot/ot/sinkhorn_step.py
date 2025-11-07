from typing import List, Tuple
import torch
from mpot.ot.problem import LinearProblem, EpsilonScheduler, SinkhornStepState
from mpot.ot.sinkhorn import Sinkhorn
from mpot.utils.polytopes import get_sampled_polytope_vertices, POLYTOPE_MAP
from mpot.utils.misc import MinMaxCenterScaler


# -----------------------------
# TorchScript-friendly helpers
# -----------------------------

@torch.jit.script
def _scheduled_radii(step_radius0: float, probe_radius0: float, eps: float) -> Tuple[float, float]:
    factor = 1.0 - eps
    return step_radius0 * factor, probe_radius0 * factor


# -----------------------------
# Fully scripted core
# -----------------------------
@torch.jit.script
class SinkhornStepCore:
    dim: int
    step_radius0: float
    probe_radius0: float
    num_probe: int
    min_iterations: int
    max_iterations: int
    threshold: float
    store_outer_evals: bool
    store_history: bool
    scale_cost: float

    epsilon: EpsilonScheduler
    ent_epsilon: EpsilonScheduler
    linear_ot_solver: Sinkhorn
    polytope_vertices: torch.Tensor
    state_scalers: List[MinMaxCenterScaler]

    def __init__(
        self,
        dim: int,
        linear_ot_solver: Sinkhorn,
        epsilon: EpsilonScheduler,
        ent_epsilon: EpsilonScheduler,
        polytope_vertices: torch.Tensor,
        state_scalers: List[MinMaxCenterScaler],
        scale_cost: float = 1.0,
        step_radius: float = 1.0,
        probe_radius: float = 2.0,
        num_probe: int = 5,
        min_iterations: int = 5,
        max_iterations: int = 50,
        threshold: float = 1e-3,
        store_outer_evals: bool = False,
        store_history: bool = False,
    ):
        self.dim = int(dim)
        self.linear_ot_solver = linear_ot_solver
        self.epsilon = epsilon
        self.ent_epsilon = ent_epsilon
        self.polytope_vertices = polytope_vertices
        self.state_scalers = state_scalers
        self.scale_cost = float(scale_cost)

        self.step_radius0 = float(step_radius)
        self.probe_radius0 = float(probe_radius)
        self.num_probe = int(num_probe)
        self.min_iterations = int(min_iterations)
        self.max_iterations = int(max_iterations)
        self.threshold = float(threshold)
        self.store_outer_evals = bool(store_outer_evals)
        self.store_history = bool(store_history)

    def init_state(self, X_init: torch.Tensor) -> SinkhornStepState:
        n = int(X_init.shape[0])
        T = self.max_iterations

        # Use zero-size tensors instead of None for JIT
        if self.store_history:
            X_history = torch.zeros((T, n, self.dim), dtype=X_init.dtype, device=X_init.device)
        else:
            X_history = torch.zeros((0, 1, 1), dtype=X_init.dtype, device=X_init.device)

        if self.store_outer_evals:
            costs = -torch.ones((T,), dtype=X_init.dtype, device=X_init.device)
        else:
            costs = torch.zeros((0,), dtype=X_init.dtype, device=X_init.device)

        displacement_sqnorms = -torch.ones((T,), dtype=X_init.dtype, device=X_init.device)
        linear_convergence = -torch.ones((T,), dtype=X_init.dtype, device=X_init.device)

        a = torch.ones((n,), dtype=X_init.dtype, device=X_init.device) / float(n)

        return SinkhornStepState(
            X_init=X_init,
            costs=costs,
            linear_convergence=linear_convergence,
            X_history=X_history,
            displacement_sqnorms=displacement_sqnorms,
            a=a,
        )

    def _converged(self, state: SinkhornStepState, iteration: int) -> bool:
        if iteration < 2:
            return False
        dq = state.displacement_sqnorms
        return bool(torch.isclose(dq[iteration - 2], dq[iteration - 1], rtol=self.threshold))

    def _continue(self, state: SinkhornStepState, iteration: int) -> bool:
        return (iteration <= self.min_iterations) or ((not self._converged(state, iteration)) and (iteration < self.max_iterations))

    def step_core(
        self,
        state: SinkhornStepState,
        iteration: int,
        # Precomputed inputs (script-friendly):
        C: torch.Tensor,                 # (n, m) cost matrix
        X_vertices: torch.Tensor         # (n, m, dim) sampled vertices matching C
    ) -> SinkhornStepState:
        # Dual-Sinkhorn (uniform a,b defaulted inside LinearProblem)
        ot_prob = LinearProblem(
            C, self.ent_epsilon,
            torch.ones((C.shape[0],), dtype=C.dtype, device=C.device) / float(C.shape[0]),
            torch.ones((C.shape[1],), dtype=C.dtype, device=C.device) / float(C.shape[1]),
            scaling_cost=True
        )

        fu0 = torch.zeros((C.shape[0],), dtype=C.dtype, device=C.device)
        gv0 = torch.zeros((C.shape[1],), dtype=C.dtype, device=C.device)
        W, res = self.linear_ot_solver.run(ot_prob, fu0, gv0, True)

        # Barycentric projection
        denom = state.a.unsqueeze(-1)          # (n,1)
        X_new = torch.einsum('nmd,nm->nd', X_vertices, W / denom)

        if int(state.X_history.numel()) > 0:
            state.X_history[iteration] = X_new

        state.linear_convergence[iteration] = float(res.converged_at)
        diff = X_new - state.X
        state.displacement_sqnorms[iteration] = (diff * diff).sum()
        state.X = X_new
        return state


# -----------------------------------------------------------
# Python wrapper (eager): computes C via objective_fn, then
# delegates to scripted core.step_core(...)
# -----------------------------------------------------------
class SinkhornStep:
    """Eager wrapper with the same ergonomics; JIT should use SinkhornStepCore."""

    def __init__(
        self,
        dim: int,
        objective_fn,                     # Python callable: __call__(X_probe, current_trajs, optim_dim, **kwargs), .cost(X, **kwargs)
        linear_ot_solver: Sinkhorn,
        epsilon: EpsilonScheduler,
        ent_epsilon: EpsilonScheduler,
        polytope_type: str = 'orthoplex',
        state_scalers: List[MinMaxCenterScaler] = (),
        scale_cost: float = 1.0,
        step_radius: float = 1.0,
        probe_radius: float = 2.0,
        num_probe: int = 5,
        min_iterations: int = 5,
        max_iterations: int = 50,
        threshold: float = 1e-3,
        store_outer_evals: bool = False,
        store_history: bool = False,
        tensor_args: dict = None,
    ):
        if tensor_args is None:
            tensor_args = {'device': 'cpu', 'dtype': torch.float32}

        self.objective_fn = objective_fn
        polytope_vertices = POLYTOPE_MAP[polytope_type](torch.zeros((dim,), **tensor_args))

        self.core = SinkhornStepCore(
            dim=dim,
            linear_ot_solver=linear_ot_solver,
            epsilon=epsilon,
            ent_epsilon=ent_epsilon,
            polytope_vertices=polytope_vertices,
            state_scalers=list(state_scalers),
            scale_cost=scale_cost,
            step_radius=step_radius,
            probe_radius=probe_radius,
            num_probe=num_probe,
            min_iterations=min_iterations,
            max_iterations=max_iterations,
            threshold=threshold,
            store_outer_evals=store_outer_evals,
            store_history=store_history,
        )

    def init_state(self, X_init: torch.Tensor) -> SinkhornStepState:
        return self.core.init_state(X_init)

    def step(self, state: SinkhornStepState, iteration: int, **kwargs) -> SinkhornStepState:
        # 1) schedule radii (don’t mutate inside core)
        eps = self.core.epsilon.at(iteration)
        sr, pr = _scheduled_radii(self.core.step_radius0, self.core.probe_radius0, eps)

        # 2) scale state (for consistent sampling)
        X = state.X.clone()
        for sc in self.core.state_scalers:
            sc(X)

        # 3) sample polytope around X (wrapper computes probes so we can build C)
        X_vertices, X_probe, _ = get_sampled_polytope_vertices(
            X, polytope_vertices=self.core.polytope_vertices,
            step_radius=sr, probe_radius=pr, num_probe=self.core.num_probe
        )

        # 4) inverse-scale for the objective
        for sc in self.core.state_scalers:
            sc.inverse(X_vertices)
            sc.inverse(X_probe)

        # 5) cost matrix from objective
        optim_dim = X_probe.shape[:-1]
        C = self.objective_fn(X_probe, current_trajs=state.X, optim_dim=optim_dim, **kwargs)

        # (optional) store outer eval
        if int(state.costs.numel()) > 0:
            state.costs[iteration] = self.objective_fn.cost(
                torch.einsum('nmd,nm->nd', X_vertices, torch.ones_like(C[..., 0]) / state.a.unsqueeze(-1)),
                **kwargs
            ).mean()

        # 6) run scripted core
        return self.core.step_core(state, iteration, C, X_vertices)
    
    def _converged(self, state, iteration: int) -> bool:
        return self.core._converged(state, iteration)

    def _continue(self, state, iteration: int) -> bool:
        return self.core._continue(state, iteration)
