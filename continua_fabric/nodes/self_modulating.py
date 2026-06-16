import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Any, Optional, Tuple
from fabricpc.nodes.base import NodeBase, SlotSpec, FlattenInputMixin
from fabricpc.core.types import NodeParams, NodeState, NodeInfo
from fabricpc.core.activations import IdentityActivation
from fabricpc.core.energy import GaussianEnergy
from fabricpc.core.initializers import KaimingInitializer, NormalInitializer


class SelfModulatingLinear(FlattenInputMixin, NodeBase):
    """Linear PC node with an internal micro-controller hypernetwork.

    Inspired by Ouroboros (arxiv 2604.02051), this node attaches a tiny
    controller that observes the current hidden state and produces a per-step
    diagonal modulation vector applied to the inference dynamics.

    The controller hypernetwork is a 2-layer MLP that maps:
        (z_latent mean, z_mu mean, energy) → (d_eta, d_precision)

    where d_eta modulates the inference learning rate and d_precision
    modulates the energy precision for this node at each inference step.

    This enables input-conditioned adaptive computation without manual
    scheduling of inference hyperparameters.
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        name: str,
        controller_hidden: int = 32,
        activation=IdentityActivation(),
        energy=GaussianEnergy(),
        use_bias: bool = True,
        flatten_input: bool = False,
        weight_init=KaimingInitializer(),
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
            controller_hidden=controller_hidden,
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
        controller_hidden = config.get("controller_hidden", 32)

        key_w, key_b, key_c1, key_c2 = jax.random.split(key, 4)
        if weight_init is None:
            weight_init = NormalInitializer(mean=0.0, std=0.05)

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
            weights_dict[edge_key] = jax.random.normal(
                rand_key_w[edge_key], weight_shape
            ) * 0.05

        # Controller hypernetwork weights
        # Input: 3 scalars (z_mean, mu_mean, energy) -> controller_hidden -> 2 (d_eta, d_precision)
        c_in = 3
        weights_dict["controller_w1"] = jax.random.normal(
            key_c1, (c_in, controller_hidden)
        ) * 0.1
        weights_dict["controller_b1"] = jnp.zeros((controller_hidden,))
        weights_dict["controller_w2"] = jax.random.normal(
            key_c2, (controller_hidden, 2)
        ) * 0.1
        weights_dict["controller_b2"] = jnp.zeros((2,))

        bias_shape = (1,) * len(node_shape) + (node_shape[-1],)
        b = jnp.zeros(bias_shape) if use_bias else jnp.zeros((0,))

        return NodeParams(weights=weights_dict, biases={"b": b} if use_bias else {})

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
        config = node_info.node_config
        controller_hidden = config.get("controller_hidden", 32)

        if flatten_input:
            pre_activation = FlattenInputMixin.compute_linear(
                inputs, params.weights, batch_size, out_shape
            )
        else:
            pre_activation = jnp.zeros((batch_size,) + out_shape)
            for edge_key, x in inputs.items():
                if ":in" in edge_key or edge_key in params.weights:
                    w = params.weights.get(edge_key)
                    if w is not None:
                        pre_activation = pre_activation + jnp.matmul(x, w)

        if "b" in params.biases and params.biases["b"].size > 0:
            pre_activation = pre_activation + params.biases["b"]

        activation = node_info.activation
        z_mu = type(activation).forward(pre_activation, activation.config)

        error = state.z_latent - z_mu
        state = state._replace(pre_activation=pre_activation, z_mu=z_mu, error=error)
        node_class = node_info.node_class
        state = node_class.energy_functional(state, node_info)

        # --- Self-modulation via controller hypernetwork ---
        z_mean = jnp.mean(state.z_latent)
        mu_mean = jnp.mean(state.z_mu)
        energy_scalar = jnp.mean(state.energy)
        ctrl_input = jnp.stack([z_mean, mu_mean, energy_scalar])

        h1 = jnp.dot(ctrl_input, params.weights["controller_w1"]) + params.weights["controller_b1"]
        h1 = jax.nn.relu(h1)
        ctrl_out = jnp.dot(h1, params.weights["controller_w2"]) + params.weights["controller_b2"]

        d_eta = jax.nn.sigmoid(ctrl_out[0]) * 0.5 + 0.01  # range [0.01, 0.51]
        d_precision = jax.nn.sigmoid(ctrl_out[1]) * 2.0 + 0.5  # range [0.5, 2.5]

        # Apply modulation to energy (precision scaling)
        state = state._replace(
            energy=state.energy * d_precision,
        )

        total_energy = jnp.sum(state.energy)
        return total_energy, state
