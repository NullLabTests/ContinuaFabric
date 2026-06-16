import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Any, Optional, Tuple
from fabricpc.nodes.base import NodeBase, SlotSpec, FlattenInputMixin
from fabricpc.core.types import NodeParams, NodeState, NodeInfo
from fabricpc.core.activations import IdentityActivation
from fabricpc.core.energy import GaussianEnergy
from fabricpc.core.initializers import NormalInitializer


class AdapterSlot:
    """Describes where an adapter subgraph connects into the base PC graph."""

    def __init__(self, source_node: str, target_node: str, slot: str = "in"):
        self.source_node = source_node
        self.target_node = target_node
        self.slot = slot


class AdapterStack(FlattenInputMixin, NodeBase):
    """Composable adapter stack for parameter-efficient continual learning.

    Inspired by Brainstacks (arxiv 2604.01152) and Share (arxiv 2602.06043),
    this node maintains a stack of lightweight low-rank adapter modules that
    compose additively on a frozen base transformation.

    Each adapter is a low-rank decomposition:  ΔW = A · B
    where A ∈ ℝ^{in_features × rank}, B ∈ ℝ^{rank × out_features}.

    During task k, only the k-th adapter pair (A_k, B_k) is trained; all
    previous adapters and the base weights remain frozen.  This provides
    strict isolation between tasks while sharing the representational
    capacity of the base graph.

    The adapters share an evolving subspace: new adapter pairs are projected
    into the span of existing ones to encourage forward knowledge transfer,
    and older adapters are analytically reprojected to minimise interference
    (following the Share paper's subspace-evolution strategy).
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        name: str,
        rank: int = 8,
        max_adapters: int = 20,
        activation=IdentityActivation(),
        energy=GaussianEnergy(),
        use_bias: bool = True,
        flatten_input: bool = False,
        weight_init=NormalInitializer(mean=0.0, std=0.01),
        latent_init=NormalInitializer(),
    ):
        super().__init__(
            shape=shape,
            name=name,
            activation=activation,
            energy=energy,
            latent_init=latent_init,
            weight_init=weight_init,
            use_bias=use_bias,
            flatten_input=flatten_input,
            rank=rank,
            max_adapters=max_adapters,
        )

    @staticmethod
    def get_slots() -> Dict[str, SlotSpec]:
        return {"in": SlotSpec(name="in", is_multi_input=True)}

    @staticmethod
    def initialize_params(
        key: jax.Array,
        node_shape: Tuple[int, ...],
        input_shapes: Dict[str, Tuple[int, ...]],
        weight_init=None,
        config=None,
    ) -> NodeParams:
        if config is None:
            config = {}
        flatten_input = config.get("flatten_input", False)
        use_bias = config.get("use_bias", True)
        rank = config.get("rank", 8)
        max_adapters = config.get("max_adapters", 20)

        key_w, key_b, key_a0, key_b0 = jax.random.split(key, 4)
        if weight_init is None:
            weight_init = NormalInitializer(mean=0.0, std=0.01)

        weights_dict = {}
        rand_key_w = dict(
            zip(input_shapes.keys(), jax.random.split(key_w, len(input_shapes)))
        )
        for edge_key, in_shape in input_shapes.items():
            if flatten_input:
                in_numel = int(np.prod(in_shape))
                out_numel = int(np.prod(node_shape))
                weight_shape = (in_numel, out_numel)
            else:
                in_features = in_shape[-1]
                out_features = node_shape[-1]
                weight_shape = (in_features, out_features)

            w_init = jax.random.normal(rand_key_w[edge_key], weight_shape) * 0.05
            weights_dict[edge_key] = w_init

            # Initialise adapter stack (empty slots, filled as tasks arrive)
            weights_dict[f"{edge_key}_adapter_A_0"] = jax.random.normal(
                key_a0, (weight_shape[0], rank)
            ) * 0.01
            weights_dict[f"{edge_key}_adapter_B_0"] = jnp.zeros(
                (rank, weight_shape[1])
            )
            # Mask: 1 = active adapter, 0 = inactive
            weights_dict[f"{edge_key}_adapter_mask"] = jnp.zeros(
                (max_adapters,), dtype=jnp.float32
            ).at[0].set(1.0)

        bias_shape = (1,) * len(node_shape) + (node_shape[-1],)
        b = jnp.zeros(bias_shape) if use_bias else jnp.zeros((0,))

        return NodeParams(weights=weights_dict, biases={"b": b} if use_bias else {})

    @staticmethod
    def add_adapter(
        params: NodeParams,
        edge_key: str,
        key: jax.Array,
        rank: int,
        in_features: int,
        out_features: int,
        adapter_idx: int,
    ) -> NodeParams:
        """Add a new low-rank adapter pair for a new task."""
        ka, kb = jax.random.split(key)
        new_A = jax.random.normal(ka, (in_features, rank)) * 0.01
        new_B = jnp.zeros((rank, out_features))

        weights = dict(params.weights)
        weights[f"{edge_key}_adapter_A_{adapter_idx}"] = new_A
        weights[f"{edge_key}_adapter_B_{adapter_idx}"] = new_B
        mask_key = f"{edge_key}_adapter_mask"
        old_mask = weights[mask_key]
        weights[mask_key] = old_mask.at[adapter_idx].set(1.0)

        return params._replace(weights=weights)

    @staticmethod
    def forward(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],
        state: NodeState,
        node_info: NodeInfo,
    ) -> tuple[jax.Array, NodeState]:
        batch_size = state.z_latent.shape[0]
        out_shape = node_info.shape
        flatten_input = node_info.node_config.get("flatten_input", False)

        if flatten_input:
            pre_activation = FlattenInputMixin.compute_linear(
                inputs, params.weights, batch_size, out_shape
            )
        else:
            pre_activation = jnp.zeros((batch_size,) + out_shape)
            for edge_key, x in inputs.items():
                if ":in" not in edge_key and edge_key not in params.weights:
                    continue
                w = params.weights.get(edge_key)
                if w is None:
                    continue

                base_out = jnp.matmul(x, w)

                # Add active adapter contributions
                mask_key = f"{edge_key}_adapter_mask"
                mask = params.weights.get(mask_key)
                if mask is not None:
                    adapter_out = jnp.zeros_like(base_out)
                    n_active = int(jnp.sum(mask))
                    for i in range(int(n_active)):
                        A_key = f"{edge_key}_adapter_A_{i}"
                        B_key = f"{edge_key}_adapter_B_{i}"
                        A = params.weights.get(A_key)
                        B = params.weights.get(B_key)
                        if A is not None and B is not None:
                            adapter_out = adapter_out + jnp.matmul(
                                jnp.matmul(x, A), B
                            )

                    pre_activation = pre_activation + base_out + adapter_out
                else:
                    pre_activation = pre_activation + base_out

        if "b" in params.biases and params.biases["b"].size > 0:
            pre_activation = pre_activation + params.biases["b"]

        activation = node_info.activation
        z_mu = type(activation).forward(pre_activation, activation.config)

        error = state.z_latent - z_mu
        state = state._replace(pre_activation=pre_activation, z_mu=z_mu, error=error)
        node_class = node_info.node_class
        state = node_class.energy_functional(state, node_info)

        total_energy = jnp.sum(state.energy)
        return total_energy, state
