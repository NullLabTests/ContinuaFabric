import jax
import jax.numpy as jnp
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from fabricpc.core.types import GraphParams
from fabricpc.core.inference import run_inference
from fabricpc.graph_initialization.state_initializer import (
    initialize_graph_state,
)


@dataclass
class EWCBuffer:
    """Stores Fisher information and optimal parameters for a learned task.

    The Fisher diagonal approximates how important each parameter is to the
    energy landscape of that task.  When learning a new task, the EWC penalty
    inflates the energy for changes to parameters that the old tasks depend on.
    """

    fisher: Optional[GraphParams] = None
    optimal_params: Optional[GraphParams] = None

    def capture_fisher(
        self,
        params: GraphParams,
        structure: Any,
        loader: Any,
        rng_key: jax.Array,
        n_samples: int = 200,
    ) -> None:
        """Approximate diagonal Fisher via Monte Carlo on the PC energy."""
        from fabricpc.core.learning import compute_local_weight_gradients

        fisher_dict = {}

        key, subkey = jax.random.split(rng_key)
        sample_keys = jax.random.split(subkey, n_samples)

        grad_sums = None
        count = 0

        for batch_idx, batch_data in enumerate(loader):
            if count >= n_samples:
                break

            if isinstance(batch_data, (list, tuple)):
                batch = {"x": jnp.array(batch_data[0]),
                         "y": jnp.array(batch_data[1])}
            else:
                batch = {k: jnp.array(v) for k, v in batch_data.items()}

            batch_size = next(iter(batch.values())).shape[0]
            clamps = {}
            for tname, tval in batch.items():
                if tname in structure.task_map:
                    clamps[structure.task_map[tname]] = tval

            init_state = initialize_graph_state(
                structure, batch_size, sample_keys[count],
                clamps=clamps, params=params,
            )
            final_state = run_inference(params, init_state, clamps, structure)
            grads = compute_local_weight_gradients(params, final_state, structure)

            sq_grads = jax.tree_util.tree_map(lambda g: g ** 2, grads)

            if grad_sums is None:
                grad_sums = sq_grads
            else:
                grad_sums = jax.tree_util.tree_map(
                    lambda a, b: a + b, grad_sums, sq_grads
                )

            count += 1

        if grad_sums is not None:
            self.fisher = jax.tree_util.tree_map(
                lambda s: s / max(count, 1), grad_sums
            )

    def capture_params(self, params: GraphParams) -> None:
        self.optimal_params = jax.tree_util.tree_map(
            lambda x: jnp.array(x), params
        )


def energy_importance(
    params: GraphParams,
    structure: Any,
    batch: Any,
    rng_key: jax.Array,
) -> GraphParams:
    """Compute per-parameter energy gradients as an importance measure.

    Parameters whose gradients have high magnitude are more important for the
    current task; this can be used as an alternative to the Fisher diagonal.
    """
    from fabricpc.core.learning import compute_local_weight_gradients
    from fabricpc.core.inference import run_inference
    from fabricpc.graph_initialization.state_initializer import (
        initialize_graph_state,
    )

    clamps = {}
    for tname, tval in batch.items():
        if tname in structure.task_map:
            clamps[structure.task_map[tname]] = tval

    batch_size = next(iter(batch.values())).shape[0]
    init_state = initialize_graph_state(
        structure, batch_size, rng_key,
        clamps=clamps, params=params,
    )
    final_state = run_inference(params, init_state, clamps, structure)
    grads = compute_local_weight_gradients(params, final_state, structure)
    return jax.tree_util.tree_map(lambda g: g ** 2, grads)


def compute_ewc_penalty(
    current_params: GraphParams,
    prev_params: GraphParams,
    ewc_buffers: List[EWCBuffer],
) -> GraphParams:
    """Compute the EWC gradient penalty across all stored tasks.

    For each parameter θᵢ:
      penalty = ∑_k F_k[i] · (θᵢ − θ_k*[i])

    where F_k is the Fisher diagonal for task k and θ_k* are the optimal
    parameters after learning task k.
    """
    penalty = None

    for buf in ewc_buffers:
        if buf.fisher is None or buf.optimal_params is None:
            continue

        diff = jax.tree_util.tree_map(
            lambda c, o: c - o, current_params, buf.optimal_params
        )

        weighted = jax.tree_util.tree_map(
            lambda f, d: f * d, buf.fisher, diff
        )

        if penalty is None:
            penalty = weighted
        else:
            penalty = jax.tree_util.tree_map(
                lambda a, b: a + b, penalty, weighted
            )

    if penalty is None:
        penalty = jax.tree_util.tree_map(
            jnp.zeros_like, current_params
        )

    return penalty
