# ContinuaFabric

**Continuous Learning Predictive Coding Networks — JAX**

ContinuaFabric extends the predictive coding (PC) framework with explicit continual / lifelong learning capabilities. It builds on [FabricPC](https://github.com/trueagi-io/FabricPC)'s JAX-native PC engine and fuses ideas from the latest research on self-modulating architectures, composable expert stacks, meta-learning, and open-ended self-improvement.

## Why ContinuaFabric?

Predictive coding networks have a structural advantage for continual learning that has gone largely unexploited:

- **Local Hebbian learning** — weight updates are node-local, so learning a new task does not globally interfere with previous weights.
- **Separation of inference and learning** — latent states adapt rapidly to new inputs while weights change slowly, naturally supporting fast adaptation.
- **Energy-based prediction errors** — network uncertainty is expressed in per-node energies, which can drive dynamic precision scheduling and importance-weighted consolidation.

ContinuaFabric formalises these properties into a practical continual-learning framework.

## Key Ideas

| Layer | Inspiration | What It Does |
|-------|-------------|--------------|
| **Dual-Memory PC Core** | FabricPC + SOLAR + TFGN | Task-incremental PC with elastic weight consolidation, generative replay, and online streaming modes |
| **Self-Modulating Nodes** | Ouroboros + FlyPrompt | Micro-controller hypernetworks that modulate per-node inference dynamics (precision, learning rate, update order) based on the current energy landscape |
| **Composable Adapter Stacks** | Brainstacks + Share | Frozen base PC graph with lightweight learnable adapter subgraphs per task; adapters share an evolving low-rank subspace |
| **PC-as-Meta-Learner** | SOLAR + Hyperagents | The PC inference loop itself becomes a meta-optimizer — the network learns to adapt its own latents faster when entering a new task distribution |
| **Open-Ended Growth** | Darwin Godel Machine + Hyperagents | When new tasks exceed the current graph's capacity, the architecture autonomously grows new nodes and edges, validated by energy-based fitness |

## Quick Start

```bash
pip install -e ".[dev]"
python experiments/split_mnist_demo.py
```

## Project Structure

```
continua_fabric/
  core/           — continual learning engine, EWC, replay
  nodes/          — self-modulating nodes, adapter stacks
  meta/           — meta-PC, architecture search
  benchmarks/     — continual learning benchmark harnesses
experiments/      — runnable demos
```

## How It Differs

| Approach | ContinuaFabric |
|----------|----------------|
| **Replay-based CL** (GEM, A-GEM, ER) | Requires memory buffers; PC generative replay needs no stored data |
| **Regularisation-based CL** (EWC, SI) | Works on any network; ContinuaFabric uses the energy itself as the importance measure |
| **Architectural CL** (ProgNN, PackNet) | Grows networks heuristically; ContinuaFabric uses energy-based architecture search |
| **LoRA-based CL** (Share, Brainstacks) | Parameter-efficient but still backprop-based; ContinuaFabric brings the same idea to local PC learning |
| **Self-modifying systems** (DGM, Hyperagents) | Focus on code-gen agents; ContinuaFabric applies self-modulation to PC graph dynamics |

## License

MIT
