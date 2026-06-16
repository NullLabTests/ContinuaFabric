import jax
import jax.numpy as jnp
import optax
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Callable
from tqdm.auto import tqdm

from continua_fabric.core.elastic_weight import EWCBuffer, compute_ewc_penalty
from continua_fabric.core.replay import GenerativeReplayBuffer


@dataclass
class TaskID:
    identifier: str
    index: int


@dataclass
class TaskSchedule:
    tasks: List[TaskID] = field(default_factory=list)

    def add_task(self, identifier: str) -> TaskID:
        tid = TaskID(identifier=identifier, index=len(self.tasks))
        self.tasks.append(tid)
        return tid


@dataclass
class ContinualPCConfig:
    infer_steps: int = 20
    eta_infer: float = 0.05
    learning_rate: float = 3e-4
    ewc_lambda: float = 100.0
    replay_batch_ratio: float = 0.5
    replay_buffer_size: int = 2000
    use_ewc: bool = True
    use_replay: bool = True


class ContinualPCEngine:
    """Wraps a FabricPC graph with continual learning capabilities.

    Maintains task-specific state, EWC buffers, and a generative replay
    buffer so the network can learn tasks sequentially without forgetting.
    """

    def __init__(
        self,
        structure: Any,
        params: Any,
        config: Optional[ContinualPCConfig] = None,
        optimizer: Optional[Any] = None,
    ):
        self.structure = structure
        self.params = params
        self.config = config or ContinualPCConfig()
        self.optimizer = optimizer or optax.adam(self.config.learning_rate)
        self.opt_state = self.optimizer.init(params)

        self.task_schedule = TaskSchedule()
        self.ewc_buffers: Dict[str, EWCBuffer] = {}
        self.current_task: Optional[TaskID] = None
        self.replay_buffer = GenerativeReplayBuffer(
            max_size=self.config.replay_buffer_size
        )

    def learn_task(
        self,
        train_loader: Any,
        task_identifier: str,
        num_epochs: int = 10,
        rng_key: jax.Array = None,
    ) -> Dict[str, List[float]]:
        if rng_key is None:
            rng_key = jax.random.PRNGKey(0)

        task = self.task_schedule.add_task(task_identifier)
        self.current_task = task
        prev_params = self.params

        from fabricpc.core.types import GraphParams
        from fabricpc.core.inference import run_inference
        from fabricpc.graph_initialization.state_initializer import (
            initialize_graph_state,
        )
        from fabricpc.core.learning import compute_local_weight_gradients

        def train_step(p, opt_state, batch, rng_key):
            clamps = {}
            for task_name, task_value in batch.items():
                if task_name in self.structure.task_map:
                    node_name = self.structure.task_map[task_name]
                    clamps[node_name] = task_value

            batch_size = next(iter(batch.values())).shape[0]
            init_state = initialize_graph_state(
                self.structure, batch_size, rng_key,
                clamps=clamps, params=p,
            )
            final_state = run_inference(p, init_state, clamps, self.structure)

            energy = sum(
                jnp.sum(final_state.nodes[n].energy)
                for n in self.structure.nodes
                if self.structure.nodes[n].node_info.in_degree > 0
            ) / batch_size

            grads = compute_local_weight_gradients(p, final_state, self.structure)

            if self.config.use_ewc and len(self.ewc_buffers) > 0:
                ewc_penalty = compute_ewc_penalty(
                    p, prev_params, list(self.ewc_buffers.values())
                )
                grads = jax.tree_util.tree_map(
                    lambda g, e: g + self.config.ewc_lambda * e,
                    grads, ewc_penalty,
                )

            updates, opt_state = self.optimizer.update(grads, opt_state, p)
            p = optax.apply_updates(p, updates)
            return p, opt_state, energy, final_state

        jit_step = jax.jit(
            lambda p, o, b, k: train_step(p, o, b, k)
        )

        epoch_losses = []
        for epoch_idx in range(num_epochs):
            epoch_key, rng_key = jax.random.split(rng_key)
            batch_keys = jax.random.split(epoch_key, len(train_loader))

            batch_losses = []
            pbar = tqdm(enumerate(train_loader),
                        total=len(train_loader),
                        desc=f"Task {task_identifier} Epoch {epoch_idx + 1}",
                        leave=False)
            for batch_idx, batch_data in pbar:
                if isinstance(batch_data, (list, tuple)):
                    batch = {
                        "x": jnp.array(batch_data[0]),
                        "y": jnp.array(batch_data[1]),
                    }
                else:
                    batch = {k: jnp.array(v) for k, v in batch_data.items()}

                if self.config.use_replay and len(self.replay_buffer) > 0:
                    replay_x, replay_y = self.replay_buffer.sample(
                        int(self.config.replay_batch_ratio * batch["x"].shape[0]),
                        rng_key,
                    )
                    if replay_x is not None:
                        batch["x"] = jnp.concatenate([batch["x"], replay_x])
                        batch["y"] = jnp.concatenate([batch["y"], replay_y])

                self.params, self.opt_state, loss_val, final_state = jit_step(
                    self.params, self.opt_state, batch, batch_keys[batch_idx]
                )
                batch_losses.append(float(loss_val))
                pbar.set_postfix(loss=f"{loss_val:.4f}")

            avg_loss = sum(batch_losses) / len(batch_losses)
            epoch_losses.append(avg_loss)

            tqdm.write(
                f"  Task {task_identifier}, Epoch {epoch_idx + 1}: "
                f"energy = {avg_loss:.4f}"
            )

        ewc_buffer = EWCBuffer()
        ewc_buffer.capture_fisher(self.params, self.structure, train_loader, rng_key)
        ewc_buffer.capture_params(self.params)
        self.ewc_buffers[task_identifier] = ewc_buffer

        self.replay_buffer.update_from_model(
            self.params, self.structure, task_identifier, rng_key
        )

        return {"energy": epoch_losses}
