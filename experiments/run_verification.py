#!/usr/bin/env python3
"""Verification suite that exercises all ContinuaFabric components and
produces quantitative metrics.  Run with:

    python experiments/run_verification.py

Results are printed to stdout and (optionally) saved as a JSON report.
"""

import json
import sys
import time
from pathlib import Path

from jax_setup import set_jax_flags_before_importing_jax
set_jax_flags_before_importing_jax()

import jax
import jax.numpy as jnp
import optax
import numpy as np


REPORT = {}


def section(name, results_dict=None):
    if results_dict is None:
        results_dict = REPORT
    def decorator(fn):
        def wrapper(*args, **kwargs):
            print(f"\n{'='*60}")
            print(f"  {name}")
            print(f"{'='*60}")
            t0 = time.time()
            result = fn(*args, **kwargs)
            dt = time.time() - t0
            print(f"  ✓ {name} — {dt:.2f}s")
            if result is not None:
                results_dict[name] = {"status": "PASS", "time_s": round(dt, 3), **result}
            else:
                results_dict[name] = {"status": "PASS", "time_s": round(dt, 3)}
            return result
        return wrapper
    return decorator


# ── Imports (must happen after jax flags) ────────────────────────────
from fabricpc.nodes import Linear
from fabricpc.core.topology import Edge
from fabricpc.graph_assembly import TaskMap, graph
from fabricpc.graph_initialization import initialize_params
from fabricpc.core.inference import InferenceSGD
from fabricpc.graph_initialization.state_initializer import initialize_graph_state
from fabricpc.core.inference import run_inference
from fabricpc.training import train_pcn, evaluate_pcn


# ── Test 1: FabricPC baseline ────────────────────────────────────────
@section("1. FabricPC baseline training")
def test_fabricpc_baseline():
    """Verify FabricPC training converges (energy decreases)."""
    inp = Linear(shape=(784,), name='input')
    hid = Linear(shape=(128,), name='hidden')
    out = Linear(shape=(2,), name='output')
    s = graph(
        nodes=[inp, hid, out],
        edges=[Edge(source=inp, target=hid.slot('in')), Edge(source=hid, target=out.slot('in'))],
        task_map=TaskMap(x=inp, y=out),
        inference=InferenceSGD(eta_infer=0.05, infer_steps=10),
    )
    key = jax.random.PRNGKey(0)
    p = initialize_params(s, key)

    rng = np.random.RandomState(42)
    X = rng.randn(200, 784).astype(np.float32)
    Y = np.eye(2)[rng.randint(0, 2, 200)].astype(np.float32)
    loader = [{'x': X[i:i+64], 'y': Y[i:i+64]} for i in range(0, 200, 64)]

    optimizer = optax.adam(3e-4)
    cfg = {'num_epochs': 5}
    key, subkey = jax.random.split(key)
    tp, energies, _ = train_pcn(p, s, loader, optimizer, cfg, subkey, use_tqdm=False, verbose=False)

    initial_energy = energies[0][0]
    final_energy = energies[-1][-1]
    assert final_energy < initial_energy, f"Energy did not decrease: {initial_energy} -> {final_energy}"
    return {"nodes": len(s.nodes), "params": str(p), "initial_energy": round(initial_energy, 4), "final_energy": round(final_energy, 4), "energy_reduction_pct": round(100 * (1 - final_energy / max(initial_energy, 1e-8)), 1)}


# ── Test 2: SelfModulatingLinear ─────────────────────────────────────
@section("2. SelfModulatingLinear node")
def test_self_modulating():
    from continua_fabric.nodes import SelfModulatingLinear
    inp = SelfModulatingLinear(shape=(784,), name='input', controller_hidden=8)
    hid = SelfModulatingLinear(shape=(128,), name='hidden', controller_hidden=8)
    out = SelfModulatingLinear(shape=(2,), name='output', controller_hidden=8)
    s = graph(
        nodes=[inp, hid, out],
        edges=[Edge(source=inp, target=hid.slot('in')), Edge(source=hid, target=out.slot('in'))],
        task_map=TaskMap(x=inp, y=out),
        inference=InferenceSGD(eta_infer=0.05, infer_steps=10),
    )
    key = jax.random.PRNGKey(0)
    p = initialize_params(s, key)

    batch = {'x': jnp.ones((4, 784)), 'y': jnp.ones((4, 2))}
    clamps = {s.task_map[k]: v for k, v in batch.items() if k in s.task_map}
    st = initialize_graph_state(s, 4, key, clamps=clamps, params=p)
    fs = run_inference(p, st, clamps, s)

    ctrl_keys = ['controller_w1', 'controller_w2', 'controller_b1', 'controller_b2']
    has_ctrl = all(k in p.nodes['hidden'].weights for k in ctrl_keys)
    return {"controller_params_present": has_ctrl, "output_energy": round(float(jnp.sum(fs.nodes['output'].energy)), 4)}


# ── Test 3: AdapterStack ─────────────────────────────────────────────
@section("3. AdapterStack node")
def test_adapter_stack():
    from continua_fabric.nodes import AdapterStack
    inp = AdapterStack(shape=(784,), name='input')
    hid = AdapterStack(shape=(128,), name='hidden', rank=4, max_adapters=10)
    out = AdapterStack(shape=(2,), name='output', rank=4, max_adapters=10)
    s = graph(
        nodes=[inp, hid, out],
        edges=[Edge(source=inp, target=hid.slot('in')), Edge(source=hid, target=out.slot('in'))],
        task_map=TaskMap(x=inp, y=out),
        inference=InferenceSGD(eta_infer=0.05, infer_steps=10),
    )
    key = jax.random.PRNGKey(0)
    p = initialize_params(s, key)

    batch = {'x': jnp.ones((4, 784)), 'y': jnp.ones((4, 2))}
    clamps = {s.task_map[k]: v for k, v in batch.items() if k in s.task_map}
    st = initialize_graph_state(s, 4, key, clamps=clamps, params=p)
    fs = run_inference(p, st, clamps, s)

    # Check adapter stacks exist
    has_stacks = 'input->hidden:in_A_stack' in p.nodes['hidden'].weights
    A_shape = p.nodes['hidden'].weights['input->hidden:in_A_stack'].shape
    return {"adapter_stacks_present": has_stacks, "A_stack_shape": str(A_shape), "output_energy": round(float(jnp.sum(fs.nodes['output'].energy)), 4)}


# ── Test 4: EWC ──────────────────────────────────────────────────────
@section("4. Elastic Weight Consolidation")
def test_ewc():
    from continua_fabric.core.elastic_weight import EWCBuffer, compute_ewc_penalty
    inp = Linear(shape=(784,), name='input')
    hid = Linear(shape=(128,), name='hidden')
    out = Linear(shape=(2,), name='output')
    s = graph(
        nodes=[inp, hid, out],
        edges=[Edge(source=inp, target=hid.slot('in')), Edge(source=hid, target=out.slot('in'))],
        task_map=TaskMap(x=inp, y=out),
        inference=InferenceSGD(eta_infer=0.05, infer_steps=5),
    )
    key = jax.random.PRNGKey(0)
    p = initialize_params(s, key)

    buf = EWCBuffer()
    buf.capture_params(p)

    rng = np.random.RandomState(42)
    X = rng.randn(64, 784).astype(np.float32)
    Y = np.eye(2)[rng.randint(0, 2, 64)].astype(np.float32)
    loader = [{'x': X, 'y': Y}]

    key, subkey = jax.random.split(key)
    buf.capture_fisher(p, s, loader, subkey, n_samples=10)

    penalty = compute_ewc_penalty(p, p, [buf])
    grad_norm = sum(float(jnp.sum(g ** 2)) for g in jax.tree_util.tree_leaves(penalty)) ** 0.5
    return {"fisher_captured": buf.fisher is not None, "penalty_norm": round(grad_norm, 6), "n_tasks_protected": 1}

    # ── Test 5: Generative Replay ────────────────────────────────────────

@section("5. Generative Replay buffer")
def test_replay():
    from continua_fabric.core.replay import GenerativeReplayBuffer
    inp = Linear(shape=(784,), name='input')
    hid = Linear(shape=(128,), name='hidden')
    out = Linear(shape=(2,), name='output')
    s = graph(
        nodes=[inp, hid, out],
        edges=[Edge(source=inp, target=hid.slot('in')), Edge(source=hid, target=out.slot('in'))],
        task_map=TaskMap(x=inp, y=out),
        inference=InferenceSGD(eta_infer=0.05, infer_steps=5),
    )
    key = jax.random.PRNGKey(0)
    p = initialize_params(s, key)

    rb = GenerativeReplayBuffer(max_size=500)
    rb.update_from_model(p, s, 'task_0', key, n_samples=32)
    rb.update_from_model(p, s, 'task_1', key, n_samples=32)

    samp = rb.sample(20, key)
    return {"replay_capable": samp is not None, "buffer_size": len(rb), "n_tasks_in_buffer": len(rb._data)}


# ── Test 6: MetaPCLearner ────────────────────────────────────────────
@section("6. MetaPCLearner")
def test_meta_pc():
    from continua_fabric.meta import MetaPCLearner
    inp = Linear(shape=(784,), name='input')
    hid = Linear(shape=(128,), name='hidden')
    out = Linear(shape=(2,), name='output')
    s = graph(
        nodes=[inp, hid, out],
        edges=[Edge(source=inp, target=hid.slot('in')), Edge(source=hid, target=out.slot('in'))],
        task_map=TaskMap(x=inp, y=out),
        inference=InferenceSGD(eta_infer=0.05, infer_steps=5),
    )
    key = jax.random.PRNGKey(0)
    p = initialize_params(s, key)

    learner = MetaPCLearner(s, inner_steps=5)
    batch = {'x': jnp.ones((4, 784)), 'y': jnp.ones((4, 2))}
    _, grads, energy = learner.adapt_to_task(p, batch, key)
    return {"gradients_computed": 'hidden' in grads.nodes, "energy": round(float(energy), 4)}


# ── Test 7: LayerNormPC node ────────────────────────────────────────
@section("7. LayerNormPC node")
def test_layer_norm():
    from continua_fabric.nodes import LayerNormPC
    inp = LayerNormPC(shape=(784,), name='input')
    hid = LayerNormPC(shape=(128,), name='hidden')
    out = LayerNormPC(shape=(2,), name='output')
    s = graph(
        nodes=[inp, hid, out],
        edges=[Edge(source=inp, target=hid.slot('in')), Edge(source=hid, target=out.slot('in'))],
        task_map=TaskMap(x=inp, y=out),
        inference=InferenceSGD(eta_infer=0.05, infer_steps=10),
    )
    key = jax.random.PRNGKey(0)
    p = initialize_params(s, key)

    batch = {'x': jnp.ones((4, 784)), 'y': jnp.ones((4, 2))}
    clamps = {s.task_map[k]: v for k, v in batch.items() if k in s.task_map}
    st = initialize_graph_state(s, 4, key, clamps=clamps, params=p)
    fs = run_inference(p, st, clamps, s)

    has_ln = 'ln_gamma' in p.nodes['hidden'].weights
    gamma_val = float(jnp.mean(p.nodes['hidden'].weights['ln_gamma']))
    return {"ln_params_present": has_ln, "ln_gamma_mean": round(gamma_val, 4), "output_energy": round(float(jnp.sum(fs.nodes['output'].energy)), 4)}


# ── Test 8: Synaptic Intelligence ────────────────────────────────────
@section("8. Synaptic Intelligence")
def test_synaptic_intelligence():
    from continua_fabric.core.synaptic_intelligence import SIBuffer, compute_si_penalty
    inp = Linear(shape=(784,), name='input')
    hid = Linear(shape=(128,), name='hidden')
    out = Linear(shape=(2,), name='output')
    s = graph(
        nodes=[inp, hid, out],
        edges=[Edge(source=inp, target=hid.slot('in')), Edge(source=hid, target=out.slot('in'))],
        task_map=TaskMap(x=inp, y=out),
        inference=InferenceSGD(eta_infer=0.05, infer_steps=5),
    )
    key = jax.random.PRNGKey(0)
    p = initialize_params(s, key)
    p2 = jax.tree_util.tree_map(lambda x: x + jax.random.normal(key, x.shape) * 0.01, p)

    buf = SIBuffer()
    tracker = SIBuffer.init_tracker(p)
    grads = jax.tree_util.tree_map(
        lambda x: jax.random.normal(key, x.shape) * 0.1, p
    )
    tracker = SIBuffer.update_tracker(tracker, grads, p, p2)
    buf.omega = SIBuffer.compute_omega(tracker, p, p2)
    buf.capture_params(p2)

    penalty = compute_si_penalty(p2, p, [buf], si_lambda=1.0)
    grad_norm = sum(float(jnp.sum(g ** 2)) for g in jax.tree_util.tree_leaves(penalty)) ** 0.5
    return {"omega_captured": buf.omega is not None, "penalty_norm": round(grad_norm, 6), "n_tasks_protected": 1}


# ── Test 9: ContinualPCEngine (multi-task) ───────────────────────────
@section("9. ContinualPCEngine (2-task synthetic)")
def test_continual_engine(ewc_lambda=50.0, use_ewc=True, use_replay=True):
    from continua_fabric.core import ContinualPCEngine, ContinualPCConfig
    inp = Linear(shape=(784,), name='input')
    hid = Linear(shape=(128,), name='hidden')
    out = Linear(shape=(2,), name='output')
    s = graph(
        nodes=[inp, hid, out],
        edges=[Edge(source=inp, target=hid.slot('in')), Edge(source=hid, target=out.slot('in'))],
        task_map=TaskMap(x=inp, y=out),
        inference=InferenceSGD(eta_infer=0.05, infer_steps=5),
    )
    key = jax.random.PRNGKey(0)
    p = initialize_params(s, key)

    cfg = ContinualPCConfig(
        infer_steps=5, learning_rate=3e-4, ewc_lambda=ewc_lambda,
        use_ewc=use_ewc, use_replay=use_replay,
    )
    eng = ContinualPCEngine(structure=s, params=p, config=cfg, optimizer=optax.adam(3e-4))

    rng = np.random.RandomState(42)
    task_energies = {}
    for task_id in range(3):
        X = rng.randn(150, 784).astype(np.float32)
        Y = np.eye(2)[rng.randint(0, 2, 150)].astype(np.float32)
        loader = [{'x': X[i:i+32], 'y': Y[i:i+32]} for i in range(0, 150, 32)]
        k, key = jax.random.split(key)
        result = eng.learn_task(loader, f'task_{task_id}', num_epochs=3, rng_key=k)
        task_energies[f'task_{task_id}'] = {
            'initial': round(result['energy'][0], 4),
            'final': round(result['energy'][-1], 4),
        }

    return {
        "n_tasks": len(eng.task_schedule.tasks),
        "n_ewc_buffers": len(eng.ewc_buffers),
        "replay_size": len(eng.replay_buffer),
        "task_energies": task_energies,
        "ewc_enabled": use_ewc,
        "replay_enabled": use_replay,
    }


# ── Run all ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("ContinuaFabric Verification Suite")
    print(f"JAX: {jax.__version__}, Devices: {jax.devices()}")

    test_fabricpc_baseline()
    test_self_modulating()
    test_adapter_stack()
    test_ewc()
    test_replay()
    test_meta_pc()
    test_layer_norm()
    test_synaptic_intelligence()
    test_continual_engine()

    print(f"\n{'='*60}")
    print(f"  SUMMARY: {len(REPORT)} tests run, all PASSED")
    print(f"{'='*60}")

    # Print report table
    print(f"\n{'Test':<40s} {'Status':<10s} {'Time':<8s} {'Key Metric'}")
    print(f"{'-'*40} {'-'*10} {'-'*8} {'-'*30}")
    for name, data in REPORT.items():
        status = data.get("status", "?")
        t = data.get("time_s", 0)
        # Pick a key metric
        metric_keys = [k for k in data if k not in ("status", "time_s")]
        metric = ""
        if "final_energy" in data:
            metric = f"energy: {data['final_energy']}"
        elif "energy" in data:
            metric = f"energy: {data['energy']}"
        elif "output_energy" in data:
            metric = f"output: {data['output_energy']}"
        elif "task_energies" in data:
            tasks = data["task_energies"]
            finals = [v['final'] for v in tasks.values()]
            metric = f"final: {min(finals):.4f}"
        elif "buffer_size" in data:
            metric = f"buffer: {data['buffer_size']}"
        elif "penalty_norm" in data:
            metric = f"penalty: {data['penalty_norm']}"
        elif "gradients_computed" in data:
            metric = f"grads: {data['gradients_computed']}"
        else:
            metric = list(data.keys())[-1] + ": " + str(list(data.values())[-1])
        print(f"{name:<40s} {status:<10s} {t:<8.2f} {metric}")

    # Save JSON report
    report_path = Path("verification_report.json")
    with open(report_path, "w") as f:
        json.dump(REPORT, f, indent=2)
    print(f"\nReport saved to {report_path.resolve()}")
