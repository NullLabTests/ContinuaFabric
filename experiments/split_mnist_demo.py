#!/usr/bin/env python3
"""Continual learning demo: Split MNIST with ContinuaFabric.

Trains a predictive-coding network on 5 binary classification tasks
(Split MNIST) sequentially, demonstrating EWC-based forgetting
protection and generative replay.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax

from jax_setup import set_jax_flags_before_importing_jax
set_jax_flags_before_importing_jax()

from fabricpc.nodes import Linear
from fabricpc.core.topology import Edge
from fabricpc.graph_assembly import TaskMap, graph
from fabricpc.graph_initialization import initialize_params
from fabricpc.core.inference import InferenceSGD
from fabricpc.training import train_pcn, evaluate_pcn

from continua_fabric.core import ContinualPCEngine, ContinualPCConfig
from continua_fabric.benchmarks import SplitMNISTBenchmark


def build_pc_mlp() -> tuple:
    """Build a 3-layer PC network for MNIST."""
    input_node = Linear(shape=(784,), name="input")
    hidden = Linear(shape=(256,), name="hidden", activation=jax.nn.relu)
    output = Linear(shape=(2,), name="output")

    structure = graph(
        nodes=[input_node, hidden, output],
        edges=[
            Edge(source=input_node, target=hidden.slot("in")),
            Edge(source=hidden, target=output.slot("in")),
        ],
        task_map=TaskMap(x=input_node, y=output),
        inference=InferenceSGD(eta_infer=0.05, infer_steps=20),
    )

    rng_key = jax.random.PRNGKey(0)
    params_key, train_key = jax.random.split(rng_key)
    params = initialize_params(structure, params_key)

    return structure, params, train_key


def evaluate_fn(params, structure, x_batch):
    """Run inference and return output latents."""
    from fabricpc.graph_initialization.state_initializer import (
        initialize_graph_state,
    )
    from fabricpc.core.inference import run_inference

    batch_size = x_batch.shape[0]
    x_node = structure.task_map["x"]
    clamps = {x_node: jnp.array(x_batch)}

    state = initialize_graph_state(
        structure, batch_size, jax.random.PRNGKey(0),
        clamps=clamps, params=params,
    )
    final_state = run_inference(params, state, clamps, structure)

    y_node = structure.task_map["y"]
    return np.array(final_state.nodes[y_node].z_mu)


def main():
    print("=" * 60)
    print("ContinuaFabric — Split MNIST Continual Learning Demo")
    print("=" * 60)

    benchmark = SplitMNISTBenchmark(batch_size=128)
    structure, params, rng_key = build_pc_mlp()

    config = ContinualPCConfig(
        infer_steps=20,
        eta_infer=0.05,
        learning_rate=3e-4,
        ewc_lambda=100.0,
        replay_batch_ratio=0.5,
        replay_buffer_size=2000,
        use_ewc=True,
        use_replay=True,
    )

    engine = ContinualPCEngine(
        structure=structure,
        params=params,
        config=config,
        optimizer=optax.adam(config.learning_rate),
    )

    all_accuracies = []
    task_names = ["0v1", "2v3", "4v5", "6v7", "8v9"]

    for task_id in range(5):
        task_name = task_names[task_id]
        print(f"\n--- Learning Task {task_id + 1}: {task_name} ---")

        train_loader = list(benchmark.get_task_loader(task_id, "train"))

        key, rng_key = jax.random.split(rng_key)
        engine.learn_task(
            train_loader=train_loader,
            task_identifier=task_name,
            num_epochs=5,
            rng_key=key,
        )

        def model_fn(x_batch):
            return evaluate_fn(engine.params, engine.structure, x_batch)

        accs = benchmark.evaluate_on_all_tasks(model_fn)
        all_accuracies.append(accs)

        avg = np.mean(accs)
        print(f"  Accuracies after task {task_id + 1}: "
              f"{[f'{a:.3f}' for a in accs]}")
        print(f"  Average: {avg:.3f}")

    final_accs = all_accuracies[-1]
    avg_final = np.mean(final_accs)
    forgetting = []
    for t in range(5):
        if len(all_accuracies) > t:
            best = max(accs[t] for accs in all_accuracies if len(accs) > t)
            final = final_accs[t]
            forgetting.append(best - final)
        else:
            forgetting.append(0.0)

    avg_forgetting = np.mean(forgetting)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Final per-task accuracies: {[f'{a:.3f}' for a in final_accs]}")
    print(f"Average final accuracy:    {avg_final:.3f}")
    print(f"Per-task forgetting:       {[f'{f:.3f}' for f in forgetting]}")
    print(f"Average forgetting:        {avg_forgetting:.3f}")
    print(f"EWC + Replay enabled:      Yes")
    print("=" * 60)


if __name__ == "__main__":
    main()
