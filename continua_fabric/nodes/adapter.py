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

    Each adapter is a low-rank decomposition: ΔW = A · B
    where A ∈ ℝ^{in_features × rank}, B ∈ ℝ^{rank × out_features}.

    Adapters are stored as stacked tensors so the forward pass is a single
    vectorised JAX operation (compatible with JIT and fori_loop).
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

        key_w, key_b, key_stacks = jax.random.split(key, 3)
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

            key_stacks, ka, kb = jax.random.split(key_stacks, 3)
            # Stacked adapters: (max_adapters, in_features, rank) and (max_adapters, rank, out_features)
            A_stack = jnp.zeros((max_adapters, weight_shape[0], rank), dtype=jnp.float32)
            B_stack = jnp.zeros((max_adapters, rank, weight_shape[1]), dtype=jnp.float32)
            # Initialise first adapter slot
            A_stack = A_stack.at[0].set(
                jax.random.normal(ka, (weight_shape[0], rank)) * 0.01
            )
            B_stack = B_stack.at[0].set(
                jnp.zeros((rank, weight_shape[1]))
            )
            weights_dict[f"{edge_key}_A_stack"] = A_stack
            weights_dict[f"{edge_key}_B_stack"] = B_stack
            # Mask: 1 = active adapter, 0 = inactive, shape (max_adapters,)
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
        adapter_idx: int,
        key: jax.Array,
        in_features: int,
        rank: int,
        out_features: int,
    ) -> NodeParams:
        """Activate a new adapter slot with random initialisation."""
        ka, kb = jax.random.split(key)
        weights = dict(params.weights)

        A_stack_key = f"{edge_key}_A_stack"
        B_stack_key = f"{edge_key}_B_stack"

        A_stack = weights[A_stack_key]
        B_stack = weights[B_stack_key]

        A_stack = A_stack.at[adapter_idx].set(
            jax.random.normal(ka, (in_features, rank)) * 0.01
        )

        weights[A_stack_key] = A_stack
        weights[B_stack_key] = B_stack

        mask_key = f"{edge_key}_adapter_mask"
        weights[mask_key] = weights[mask_key].at[adapter_idx].set(1.0)

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

                A_stack = params.weights.get(f"{edge_key}_A_stack")
                B_stack = params.weights.get(f"{edge_key}_B_stack")
                mask = params.weights.get(f"{edge_key}_adapter_mask")

                if A_stack is not None and B_stack is not None and mask is not None:
                    # Vectorised adapter computation
                    # x: (batch, in_features)
                    # A_stack: (max_adapters, in_features, rank)
                    # B_stack: (max_adapters, rank, out_features)
                    # mask: (max_adapters,)
                    # Result: (batch, out_features)
                    xA = jnp.einsum('bi,air->bar', x, A_stack)  # (batch, max_adapters, rank)
                    xAB = jnp.einsum('bar,aro->bao', xA, B_stack)  # (batch, max_adapters, out_features)
                    masked = xAB * mask[None, :, None]  # (batch, max_adapters, out_features)
                    adapter_out = jnp.sum(masked, axis=1)  # (batch, out_features)
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
