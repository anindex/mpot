from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Literal,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
)
import numpy as np
import torch
from mpot.ot.problem import LinearProblem, SinkhornState
from mpot.ot.initializer import DefaultInitializer, RandomInitializer, SinkhornInitializer


class Momentum:
    """Momentum for Sinkhorn updates.
    """

    def __init__(
        self,
        start: int = 0,
        error_threshold: float = torch.inf,
        value: float = 1.0,
        inner_iterations: int = 1,
    ) -> None:
        self.start = start
        self.error_threshold = error_threshold
        self.value = value
        self.inner_iterations = inner_iterations

    def weight(self, state: SinkhornState, iteration: int) -> float:
        if self.start == 0:
            return self.value
        idx = self.start // self.inner_iterations

        return self.lehmann(state) if iteration >= self.start and state.errors[idx - 1, -1] < self.error_threshold \
            else self.value

    def lehmann(self, state: SinkhornState) -> float:
        """See Lehmann, T., Von Renesse, M.-K., Sambale, A., and
            Uschmajew, A. (2021). A note on overrelaxation in the
            sinkhorn algorithm. Optimization Letters, pages 1–12. eq. 5."""
        idx = self.start // self.inner_iterations
        error_ratio = torch.minimum(
            state.errors[idx - 1, -1] / state.errors[idx - 2, -1], 0.99
        )
        power = 1.0 / self.inner_iterations
        return 2.0 / (1.0 + torch.sqrt(1.0 - error_ratio ** power))

    def __call__(  
        self,
        weight: float,
        value: torch.Tensor,
        new_value: torch.Tensor
    ) -> torch.Tensor:
        value = torch.where(torch.isfinite(value), value, 0.0)
        return (1.0 - weight) * value + weight * new_value


class Sinkhorn:

    def __init__(
        self,
        threshold: float = 1e-3,
        inner_iterations: int = 1,
        min_iterations: int = 10,
        max_iterations: int = 100,
        parallel_dual_updates: bool = False,
        initializer: Literal["default", "random"] = "default",
        kwargs_init: Optional[Mapping[str, Any]] = None,
    ):
        self.threshold = threshold
        self.inner_iterations = inner_iterations
        self.min_iterations = min_iterations
        self.max_iterations = max_iterations
        self.momentum = Momentum(inner_iterations=inner_iterations)

        self.parallel_dual_updates = parallel_dual_updates
        self.initializer = initializer
        self.kwargs_init = {} if kwargs_init is None else kwargs_init

    def __call__(
        self,
        ot_prob: LinearProblem,
        init: Tuple[Optional[torch.Tensor], Optional[torch.Tensor]] = (None, None),
        compute_error: bool = False,
    ) -> torch.Tensor:

        initializer = self.create_initializer()
        init_dual_a, init_dual_b = initializer(
            ot_prob, *init
        )
        final_state = self.iterations(ot_prob, (init_dual_a, init_dual_b), compute_error=compute_error)
        return self.output_from_state(ot_prob, final_state)

    def create_initializer(self) -> SinkhornInitializer:  
        if isinstance(self.initializer, SinkhornInitializer):
            return self.initializer
        if self.initializer == "default":
            return DefaultInitializer()
        if self.initializer == "random":
            return RandomInitializer()
        raise NotImplementedError(
            f"Initializer `{self.initializer}` is not yet implemented."
        )

    def lse_step(
        self, ot_prob: LinearProblem, state: SinkhornState,
        iteration: int
    ) -> SinkhornState:
        """Sinkhorn LSE update."""

        w = self.momentum.weight(state, iteration)
        tau_a, tau_b = ot_prob.tau_a, ot_prob.tau_b
        old_fu, old_gv = state.fu, state.gv

        # update g potential
        new_gv = tau_b * ot_prob.update_potential(
            old_fu, old_gv, torch.log(ot_prob.b), iteration, dim=0
        )
        gv = self.momentum(w, old_gv, new_gv)

        if not self.parallel_dual_updates:
            old_gv = gv

        # update f potential
        new_fu = tau_a * ot_prob.update_potential(
            old_fu, old_gv, torch.log(ot_prob.a), iteration, dim=1
        )
        fu = self.momentum(w, old_fu, new_fu)

        state.fu = fu
        state.gv = gv
        return state

    def one_iteration(
        self, ot_prob: LinearProblem, state: SinkhornState,
        iteration: int, compute_error: bool = False
    ) -> SinkhornState:

        state = self.lse_step(ot_prob, state, iteration)

        # re-computes error if compute_error is True, else set it to -1.
        if compute_error:
            err = state.solution_error(
                ot_prob,
                parallel_dual_updates=self.parallel_dual_updates,
            )
        else:
            err = -1
        state.errors[iteration // self.inner_iterations] = err
        return state

    def _converged(self, state: SinkhornState, iteration: int) -> bool:
        err = state.errors[iteration // self.inner_iterations - 1]
        return iteration > 0 and err < self.threshold

    def _diverged(self, state: SinkhornState, iteration: int) -> bool:
        err = state.errors[iteration // self.inner_iterations - 1]
        return not torch.isfinite(err)

    def _continue(self, state: SinkhornState, iteration: int) -> bool:
        """Continue while not(converged) and not(diverged)."""
        return not self._converged(state, iteration) and not self._diverged(state, iteration)

    @property
    def outer_iterations(self) -> int:
        """Upper bound on number of times inner_iterations are carried out.
        """
        return np.ceil(self.max_iterations / self.inner_iterations).astype(int)

    def init_state(
        self, init: Tuple[torch.Tensor, torch.Tensor]
    ) -> SinkhornState:
        """Return the initial state of the loop."""
        fu, gv = init
        errors = -torch.ones(self.outer_iterations).type_as(fu)
        state = SinkhornState(errors=errors, fu=fu, gv=gv)
        return state

    def output_from_state(
        self, ot_prob: LinearProblem, state: SinkhornState
    ) -> torch.Tensor:
        """Return the output of the Sinkhorn loop."""
        return ot_prob.transport_from_potentials(state.fu, state.gv)

    def iterations(
        self, ot_prob: LinearProblem, init: Tuple[torch.Tensor, torch.Tensor], compute_error: bool = False
    ) -> SinkhornState:
        state = self.init_state(init)
        iteration = 0
        while self._continue(state, iteration):
            state = self.one_iteration(ot_prob, state, iteration, compute_error=compute_error)
            iteration += self.inner_iterations
        return state


if __name__ == "__main__":
    from mpot.ot.problem import Epsilon
    from torch_robotics.torch_utils.torch_timer import TimerCUDA
    epsilon = Epsilon(target=0.1, init=1., decay=0.8)
    ot_prob = LinearProblem(
        torch.rand((1000, 1000)), epsilon
    )
    sinkhorn = Sinkhorn()
    with TimerCUDA() as t:
        W = sinkhorn(ot_prob)
    print(t.elapsed)
    print(W.shape)
