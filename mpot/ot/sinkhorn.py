import torch
from typing import Tuple

from mpot.ot.problem import LinearProblem, SinkhornState

# -------------------------
# TorchScript-friendly utils
# -------------------------

@torch.jit.script
def _outer_iterations(max_iterations: int, inner_iterations: int) -> int:
    # ceil(max_it / inner_it) without numpy
    return (max_iterations + inner_iterations - 1) // inner_iterations

@torch.jit.script
def _coerce_dual(v: torch.Tensor, target_len: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    v = v.reshape(-1)
    if int(v.numel()) != int(target_len):
        return torch.zeros((target_len,), dtype=dtype, device=device)
    return v

@torch.jit.script
def _finite_or_zero(x: torch.Tensor) -> torch.Tensor:
    return torch.where(torch.isfinite(x), x, torch.zeros_like(x))


# -------------------------
# Momentum (scripted)
# -------------------------

@torch.jit.script
class Momentum:
    start: int
    error_threshold: float
    value: float
    inner_iterations: int

    def __init__(self, start: int = 0, error_threshold: float = 1.0e38,
                 value: float = 1.0, inner_iterations: int = 1):
        self.start = int(start)
        self.error_threshold = float(error_threshold)
        self.value = float(value)
        self.inner_iterations = int(inner_iterations)

    def weight(self, state: SinkhornState, iteration: int) -> float:
        if self.start == 0:
            return self.value
        idx = self.start // self.inner_iterations
        # require at least one completed error slot
        if iteration >= self.start and int(state.errors.numel()) >= idx:
            prev_err = state.errors[idx - 1]
            if prev_err < self.error_threshold:
                return float(self.lehmann(state))
        return self.value

    def lehmann(self, state: SinkhornState) -> float:
        # See: Lehmann et al. (2021), eq. (5)
        idx = self.start // self.inner_iterations
        # need two past errors: idx-1 and idx-2
        if idx < 2:
            return self.value
        e1 = state.errors[idx - 1]
        e2 = state.errors[idx - 2]
        ratio = e1 / e2
        cap = torch.tensor(0.99, dtype=state.errors.dtype, device=state.errors.device)
        ratio = torch.minimum(ratio, cap)
        power = 1.0 / float(self.inner_iterations)
        one = torch.tensor(1.0, dtype=state.errors.dtype, device=state.errors.device)
        two = torch.tensor(2.0, dtype=state.errors.dtype, device=state.errors.device)
        w = two / (one + torch.sqrt(one - torch.pow(ratio, power)))
        return float(w)

    def apply(self, weight: float, value: torch.Tensor, new_value: torch.Tensor) -> torch.Tensor:
        v = _finite_or_zero(value)
        w = float(weight)
        return (1.0 - w) * v + w * new_value


# -------------------------
# Sinkhorn (scripted)
# -------------------------

@torch.jit.script
class Sinkhorn:
    threshold: float
    inner_iterations: int
    min_iterations: int
    max_iterations: int
    parallel_dual_updates: bool
    init_type: int
    momentum: Momentum

    def __init__(self,
                 threshold: float = 1e-3,
                 inner_iterations: int = 1,
                 min_iterations: int = 1,
                 max_iterations: int = 100,
                 parallel_dual_updates: bool = False,
                 init_type: int = 0   # 0: default zeros, 1: random small noise
                 ):
        self.threshold = float(threshold)
        self.inner_iterations = int(inner_iterations)
        self.min_iterations = int(min_iterations)
        self.max_iterations = int(max_iterations)
        self.parallel_dual_updates = bool(parallel_dual_updates)
        self.init_type = int(init_type)
        self.momentum = Momentum(0, 1.0e38, 1.0, self.inner_iterations)

    # ---- initializer (scripted, no external deps) ----
    def _init_duals(self, ot_prob: LinearProblem,
                    fu0: torch.Tensor, gv0: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        n = int(ot_prob.a.shape[0])
        m = int(ot_prob.b.shape[0])
        dtype = ot_prob.C.dtype
        device = ot_prob.C.device

        fu = _coerce_dual(fu0, n, dtype, device)
        gv = _coerce_dual(gv0, m, dtype, device)

        if self.init_type == 1:
            # random small init around zero (log-space compatible)
            fu = fu + 1e-3 * torch.randn_like(fu)
            gv = gv + 1e-3 * torch.randn_like(gv)

        return fu, gv

    # ---- single LSE step ----
    def lse_step(self, ot_prob: LinearProblem, state: SinkhornState, iteration: int) -> SinkhornState:
        w = self.momentum.weight(state, iteration)
        tau_a = ot_prob.tau_a
        tau_b = ot_prob.tau_b

        old_fu = state.fu
        old_gv = state.gv

        # update g (dim=0)
        new_gv = tau_b * ot_prob.update_potential(old_fu, old_gv, torch.log(ot_prob.b), iteration, 0)
        gv = self.momentum.apply(w, old_gv, new_gv)

        # if not parallel, f update uses the freshly updated g
        g_for_f = gv if (not self.parallel_dual_updates) else old_gv

        # update f (dim=1)
        new_fu = tau_a * ot_prob.update_potential(old_fu, g_for_f, torch.log(ot_prob.a), iteration, 1)
        fu = self.momentum.apply(w, old_fu, new_fu)

        state.fu = fu
        state.gv = gv
        return state

    # ---- one outer iteration (may contain several inner steps in your design) ----
    def one_iteration(self, ot_prob: LinearProblem, state: SinkhornState,
                      iteration: int, compute_error: bool) -> SinkhornState:

        state = self.lse_step(ot_prob, state, iteration)

        # re-compute error if requested
        if compute_error:
            err = state.solution_error(ot_prob, self.parallel_dual_updates)
        else:
            err = torch.tensor(-1.0, dtype=state.errors.dtype, device=state.errors.device)

        idx = iteration // self.inner_iterations
        state.errors[idx] = err
        return state

    # ---- stopping logic ----
    def _converged(self, state: SinkhornState, iteration: int) -> bool:
        if iteration <= self.min_iterations:
            return False
        idx = (iteration // self.inner_iterations) - 1
        if idx < 0:
            return False
        return bool(state.errors[idx] < self.threshold)

    def _diverged(self, state: SinkhornState, iteration: int) -> bool:
        idx = (iteration // self.inner_iterations) - 1
        if idx < 0:
            return False
        return not bool(torch.isfinite(state.errors[idx]))

    def _continue(self, state: SinkhornState, iteration: int) -> bool:
        oit = _outer_iterations(self.max_iterations, self.inner_iterations)
        return (iteration < oit) and (not self._converged(state, iteration)) and (not self._diverged(state, iteration))

    # ---- state init / output ----
    def init_state(self, ot_prob: LinearProblem, fu: torch.Tensor, gv: torch.Tensor) -> SinkhornState:
        oit = _outer_iterations(self.max_iterations, self.inner_iterations)
        errors = -torch.ones((oit,), dtype=ot_prob.C.dtype, device=ot_prob.C.device)
        return SinkhornState(errors=errors, fu=fu, gv=gv)

    def output_from_state(self, ot_prob: LinearProblem, state: SinkhornState) -> torch.Tensor:
        return ot_prob.transport_from_potentials(state.fu, state.gv)

    # ---- main loop ----
    def iterations(self, ot_prob: LinearProblem, fu0: torch.Tensor, gv0: torch.Tensor,
                   compute_error: bool) -> SinkhornState:
        fu, gv = self._init_duals(ot_prob, fu0, gv0)
        state = self.init_state(ot_prob, fu, gv)

        iteration = 0
        while self._continue(state, iteration):
            state = self.one_iteration(ot_prob, state, iteration, compute_error)
            iteration += self.inner_iterations

        if self._converged(state, iteration):
            state.converged_at = iteration

        return state

    # ---- callable interface (avoids Optional/Union defaults) ----
    def run(self, ot_prob: LinearProblem, fu0: torch.Tensor, gv0: torch.Tensor,
            compute_error: bool = True) -> Tuple[torch.Tensor, SinkhornState]:

        final_state = self.iterations(ot_prob, fu0, gv0, compute_error)
        return self.output_from_state(ot_prob, final_state), final_state
