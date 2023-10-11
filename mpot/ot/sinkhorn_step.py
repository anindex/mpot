from typing import (
    TYPE_CHECKING,
    Any,
    List,
    Callable,
    Literal,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import torch
from mpot.ot.problem import LinearProblem, Epsilon, SinkhornStepState
from mpot.ot.sinkhorn import Sinkhorn
from mpot.utils.polytopes import POLYTOPE_MAP, get_sampled_polytope_vertices
from mpot.utils.misc import MinMaxCenterScaler


class SinkhornStep():
    """Sinkhorn Step solver."""

    def __init__(
        self,
        dim: int,
        objective_fn: Any,
        linear_ot_solver: Sinkhorn,
        epsilon: Union[Epsilon, float] ,
        ent_epsilon: Union[Epsilon, float] = 0.01,
        state_scalers: Optional[List[MinMaxCenterScaler]] = None,
        polytope_type: str = 'orthoplex',
        scale_cost: float = 1.0,
        step_radius: float = 1.,
        probe_radius: float = 2.,
        num_probe: int = 5,
        min_iterations: int = 5,
        max_iterations: int = 50,
        threshold: float = 1e-3,
        store_inner_errors: bool = False,
        store_outer_evals: bool = False,
        store_history: bool = False,
        tensor_args: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        if tensor_args is None:
            tensor_args = {'device': 'cpu', 'dtype': torch.float32}
        self.tensor_args = tensor_args
        self.dim = dim
        self.objective_fn = objective_fn

        # Sinkhorn Step params
        self.linear_ot_solver = linear_ot_solver
        self.polytope_vertices = POLYTOPE_MAP[polytope_type](torch.zeros((self.dim,), **tensor_args))
        self.epsilon = epsilon
        self.ent_epsilon = ent_epsilon
        self.state_scalers = state_scalers
        self.step_radius = step_radius
        self.probe_radius = probe_radius
        self.scale_cost = scale_cost
        self.num_probe = num_probe
        self.min_iterations = min_iterations
        self.max_iterations = max_iterations
        self.threshold = threshold
        self.store_inner_errors = store_inner_errors
        self.store_outer_evals = store_outer_evals
        self.store_history = store_history

        # TODO: support non-uniform weights for conditional sinkhorn step

    def init_state(
        self,
        X_init: torch.Tensor,
    ) -> SinkhornStepState:
        num_points, dim = X_init.shape
        num_iters = self.max_iterations
        if self.store_history:
            X_history = torch.zeros((num_iters, num_points, self.dim)).type_as(X_init)
        else:
            X_history = None
        
        if self.store_outer_evals:
            costs = -torch.ones(num_iters).type_as(X_init)
        else:
            costs = None
    
        self.displacement_sqnorms = -torch.ones(num_iters).type_as(X_init)
        self.linear_convergence = -torch.ones(num_iters).type_as(X_init)
        a = torch.ones((num_points,)).type_as(X_init) / num_points  # always uniform weights for now

        return SinkhornStepState(
            X_init=X_init,
            costs=costs,
            linear_convergence=self.linear_convergence,
            X_history=X_history,
            displacement_sqnorms=self.displacement_sqnorms,
            a=a,
        )

    def step(self, state: SinkhornStepState, iteration: int, **kwargs) -> SinkhornStepState:
        """Run Sinkhorn Step."""
        X = state.X.clone()

        # scale state features into same range
        if self.state_scalers is not None:
            for scaler in self.state_scalers:
                scaler(X)

        eps = self.epsilon.at(iteration) if isinstance(self.epsilon, Epsilon) else self.epsilon
        step_radius = self.step_radius * eps
        probe_radius = self.probe_radius * eps

        # compute sampled polytope vertices
        X_vertices, X_probe, vertices = get_sampled_polytope_vertices(X,
                                                                      polytope_vertices=self.polytope_vertices,
                                                                      step_radius=step_radius,
                                                                      probe_radius=probe_radius,
                                                                      num_probe=self.num_probe)
        
        # unscale for cost evaluation
        if self.state_scalers is not None:
            for scaler in self.state_scalers:
                scaler.inverse(X_vertices)
                scaler.inverse(X_probe)

        # solve Sinkhorn
        C = self.objective_fn(X_probe, current_trajs=state.X, **kwargs) 
        ot_prob = LinearProblem(C, epsilon=self.ent_epsilon, a=state.a, scaling_cost=False)
        W, res = self.linear_ot_solver(ot_prob)

        # barycentric projection
        X_new = torch.einsum('bik,bi->bk', X_vertices, W / state.a.unsqueeze(-1))

        if self.store_outer_evals:
            state.costs[iteration] = self.objective_fn.cost(X_new, **kwargs).mean()

        if self.store_history:
            state.X_history[iteration] = X_new

        state.linear_convergence[iteration] = res.converged
        state.displacement_sqnorms[iteration] = torch.square(X_new - state.X).sum()
        state.X = X_new
        return state

    def _converged(self, state: SinkhornStepState, iteration: int) -> bool:
        dqsnorm, i, tol = state.displacement_sqnorms, iteration, self.threshold
        return torch.isclose(dqsnorm[i - 2], dqsnorm[i - 1], rtol=tol)

    def _continue(self, state: SinkhornStepState, iteration: int) -> bool:
        """Continue while not(converged)"""
        return iteration <= self.min_iterations or (not self._converged(state, iteration) and iteration < self.max_iterations)
