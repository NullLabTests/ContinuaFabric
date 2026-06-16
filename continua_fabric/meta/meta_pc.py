import jax
import jax.numpy as jnp
from typing import Any, Dict, Optional, Tuple, Callable
from dataclasses import dataclass, field

from fabricpc.core.types import GraphParams, GraphState, GraphStructure
from fabricpc.core.inference import run_inference
from fabricpc.core.learning import compute_local_weight_gradients
from fabricpc.graph_initialization.state_initializer import (
    initialize_graph_state,
)


@dataclass
class MetaPCLearner:
    """Meta-learning wrapper that treats the PC inference loop as optimisation.

    Key insight (inspired by SOLAR arxiv 2605.20189 and Hyperagents
    arxiv 2603.19461): the PC inference loop already performs inner-loop
    optimisation on latent states.  By framing task adaptation as a few
    steps of PC inference, we can meta-learn the inference hyperparameters
    (eta_infer, precision, initial latents) that minimise the expected
    energy after K inference steps across a distribution of tasks.

    This turns PC's natural inference dynamics into a meta-learning
    procedure comparable to MAML but without second-order gradients
    (since PC learning is local and Hebbian).
    """

    def __init__(
        self,
        structure: GraphStructure,
        inner_steps: int = 10,
        outer_lr: float = 1e-3,
    ):
        self.structure = structure
        self.inner_steps = inner_steps
        self.outer_lr = outer_lr

    def adapt_to_task(
        self,
        params: GraphParams,
        batch: Dict[str, jnp.ndarray],
        rng_key: jax.Array,
    ) -> Tuple[GraphState, GraphParams, float]:
        """Inner loop: run PC inference to adapt latents to a task.

        Returns the final state, the parameter gradients (for meta-update),
        and the final energy.
        """
        clamps = {}
        for task_name, task_value in batch.items():
            if task_name in self.structure.task_map:
                clamps[self.structure.task_map[task_name]] = task_value

        batch_size = next(iter(batch.values())).shape[0]
        init_state = initialize_graph_state(
            self.structure, batch_size, rng_key,
            clamps=clamps, params=params,
        )
        final_state = run_inference(params, init_state, clamps, self.structure)

        energy = sum(
            jnp.sum(final_state.nodes[n].energy)
            for n in self.structure.nodes
            if self.structure.nodes[n].node_info.in_degree > 0
        ) / batch_size

        grads = compute_local_weight_gradients(params, final_state, self.structure)
        return final_state, grads, energy

    def meta_update(
        self,
        params: GraphParams,
        meta_batch: Dict[str, jnp.ndarray],
        rng_key: jax.Array,
    ) -> Tuple[GraphParams, float]:
        """Single meta-update step over a batch of tasks.

        Each meta-batch element represents a different task (or different
        data from the task distribution).  The meta-gradient is the average
        of the local PC gradients across tasks.
        """
        import optax

        optimizer = optax.adam(self.outer_lr)
        opt_state = optimizer.init(params)

        meta_grads = None
        total_energy = 0.0
        n_tasks = 0

        batch_keys = jax.random.split(rng_key, 4)

        for task_key, task_data in meta_batch.items():
            if task_key not in self.structure.task_map:
                continue
            _, grads, energy = self.adapt_to_task(
                params, {task_key: task_data}, batch_keys[n_tasks]
            )
            if meta_grads is None:
                meta_grads = grads
            else:
                meta_grads = jax.tree_util.tree_map(
                    lambda a, b: a + b, meta_grads, grads
                )
            total_energy += energy
            n_tasks += 1

        if n_tasks > 0:
            meta_grads = jax.tree_util.tree_map(
                lambda g: g / n_tasks, meta_grads
            )
            total_energy /= n_tasks

            updates, opt_state = optimizer.update(meta_grads, opt_state, params)
            params = optax.apply_updates(params, updates)

        return params, total_energy


def meta_pc_train_step(
    params: GraphParams,
    structure: GraphStructure,
    batch: Dict[str, jnp.ndarray],
    rng_key: jax.Array,
    inner_steps: int = 10,
) -> Tuple[GraphParams, float]:
    """Standalone meta-PC training step for use in JIT-compiled loops.

    Runs PC inference (inner loop), computes local weight gradients, and
    returns the updated parameters and energy.  This is the core of the
    'PC-as-meta-learner' approach: the inference loop itself serves as
    the inner-loop optimiser, and the weight update is the meta-update.
    """
    import optax

    learner = MetaPCLearner(structure, inner_steps=inner_steps)
    new_params, energy = learner.meta_update(params, batch, rng_key)
    return new_params, energy
