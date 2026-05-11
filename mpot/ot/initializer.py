"""Sinkhorn dual potential initializers.

These initializers are provided for advanced use cases where custom
initialization strategies are needed. The default Sinkhorn solver
handles initialization internally via init_type parameter.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch

from mpot.ot.problem import LinearProblem


class SinkhornInitializer(ABC):
    """Base class for Sinkhorn initializers."""

    @abstractmethod
    def init_dual_a(self, ot_prob: LinearProblem) -> torch.Tensor:
        """Initialize Sinkhorn potential f_u.

        Returns:
            potential of size (n,).
        """

    @abstractmethod
    def init_dual_b(self, ot_prob: LinearProblem) -> torch.Tensor:
        """Initialize Sinkhorn potential g_v.

        Returns:
            potential of size (m,).
        """

    def __call__(
        self,
        ot_prob: LinearProblem,
        a: Optional[torch.Tensor] = None,
        b: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        n, m = ot_prob.C.shape
        if a is None:
            a = self.init_dual_a(ot_prob)
        if b is None:
            b = self.init_dual_b(ot_prob)

        assert a.shape == (n,), f"Expected f_u shape ({n},), got {a.shape}"
        assert b.shape == (m,), f"Expected g_v shape ({m},), got {b.shape}"

        # Cancel dual variables for zero weights
        a = torch.where(ot_prob.a > 0., a, torch.tensor(-torch.inf, dtype=a.dtype, device=a.device))
        b = torch.where(ot_prob.b > 0., b, torch.tensor(-torch.inf, dtype=b.dtype, device=b.device))

        return a, b


class DefaultInitializer(SinkhornInitializer):
    """Default zero initialization of Sinkhorn dual potentials."""

    def init_dual_a(self, ot_prob: LinearProblem) -> torch.Tensor:
        return torch.zeros_like(ot_prob.a)

    def init_dual_b(self, ot_prob: LinearProblem) -> torch.Tensor:
        return torch.zeros_like(ot_prob.b)


class RandomInitializer(SinkhornInitializer):
    """Random initialization of Sinkhorn dual potentials."""

    def __init__(self, seed: Optional[int] = None):
        self.seed = seed

    def init_dual_a(self, ot_prob: LinearProblem) -> torch.Tensor:
        return torch.randn_like(ot_prob.a)

    def init_dual_b(self, ot_prob: LinearProblem) -> torch.Tensor:
        return torch.randn_like(ot_prob.b)
