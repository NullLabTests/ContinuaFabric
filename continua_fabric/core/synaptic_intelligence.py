import jax
import jax.numpy as jnp
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from fabricpc.core.types import GraphParams


@dataclass
class SIBuffer:
    """Synaptic Intelligence buffer for a single task.

    Tracks per-parameter importance Omega computed from the path integral
    of (gradient x parameter update) along the learning trajectory.

    Reference:
        Zenke, Poole, Ganguli (2017) "Continual Learning Through Synaptic
        Intelligence" — https://arxiv.org/abs/1703.04200
    """

    omega: Optional[GraphParams] = None
    optimal_params: Optional[GraphParams] = None
    epsilon: float = 1e-8

    @staticmethod
    def init_tracker(params: GraphParams) -> GraphParams:
        """Initialise the running accumulators for a new task."""
        return jax.tree_util.tree_map(jnp.zeros_like, params)

    @staticmethod
    def update_tracker(
        tracker: GraphParams,
        grads: GraphParams,
        prev_params: GraphParams,
        current_params: GraphParams,
    ) -> GraphParams:
        """Accumulate the path integral omega during a training step.

        omega_i += -grad_i * (theta_i(t+1) - theta_i(t))
        """
        diff = jax.tree_util.tree_map(
            lambda c, p: c - p, current_params, prev_params
        )
        contribution = jax.tree_util.tree_map(
            lambda g, d: -g * d, grads, diff
        )
        return jax.tree_util.tree_map(
            lambda t, c: t + c, tracker, contribution
        )

    @staticmethod
    def compute_omega(
        tracker: GraphParams,
        prev_params: GraphParams,
        current_params: GraphParams,
        epsilon: float = 1e-8,
    ) -> GraphParams:
        """Convert accumulated tracker into per-parameter importance Omega.

        Omega_i = omega_i / (Delta_i^2 + epsilon)
        where Delta_i is the total parameter change over the task.
        """
        delta = jax.tree_util.tree_map(
            lambda c, p: c - p, current_params, prev_params
        )
        return jax.tree_util.tree_map(
            lambda t, d: t / (d ** 2 + epsilon), tracker, delta
        )

    def capture_params(self, params: GraphParams) -> None:
        self.optimal_params = jax.tree_util.tree_map(
            lambda x: jnp.array(x), params
        )


def compute_si_penalty(
    current_params: GraphParams,
    prev_params: GraphParams,
    si_buffers: List[SIBuffer],
    si_lambda: float = 1.0,
) -> GraphParams:
    """Compute the SI gradient penalty across all stored tasks.

    For each parameter theta_i:
      penalty = sum_k Omega_k[i] * (theta_i - theta_k*[i])

    Returns a tree of gradients to add to the current task gradients.
    """
    penalty = None

    for buf in si_buffers:
        if buf.omega is None or buf.optimal_params is None:
            continue

        diff = jax.tree_util.tree_map(
            lambda c, o: c - o, current_params, buf.optimal_params
        )
        weighted = jax.tree_util.tree_map(
            lambda w, d: w * d, buf.omega, diff
        )

        if penalty is None:
            penalty = weighted
        else:
            penalty = jax.tree_util.tree_map(
                lambda a, b: a + b, penalty, weighted
            )

    if penalty is None:
        penalty = jax.tree_util.tree_map(jnp.zeros_like, current_params)

    return jax.tree_util.tree_map(lambda p: si_lambda * p, penalty)
