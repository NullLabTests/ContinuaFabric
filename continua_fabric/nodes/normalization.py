import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Any, Optional, Tuple
from fabricpc.nodes.base import NodeBase, SlotSpec, FlattenInputMixin
from fabricpc.core.types import NodeParams, NodeState, NodeInfo
from fabricpc.core.activations import IdentityActivation
from fabricpc.core.energy import GaussianEnergy
from fabricpc.core.initializers import KaimingInitializer, NormalInitializer


class LayerNormPC(FlattenInputMixin, NodeBase):
    """PC node with layer normalisation and learnable gain/bias.

    Applies a linear transformation followed by layer normalisation:

        pre_activation = W @ x + b
        z_mu = LayerNorm(pre_activation) = gamma * (pre_activation - mu) / sigma + beta

    where gamma (gain) and beta (bias) are learnable per-node vectors.
    The PC energy is then E = ||z_latent - z_mu||^2.

    Layer normalisation stabilises inference dynamics by ensuring the
    latent distribution has consistent mean/variance regardless of input
    statistics, which is beneficial for continual learning (prevents
    latent shift between tasks).
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        name: str,
        eps: float = 1e-5,
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
            eps=eps,
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

        key_w, key_g, key_b = jax.random.split(key, 3)
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

        # LayerNorm gain and bias
        out_features = node_shape[-1]
        weights_dict["ln_gamma"] = jnp.ones((out_features,), dtype=jnp.float32)
        weights_dict["ln_beta"] = jnp.zeros((out_features,), dtype=jnp.float32)

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
        eps = node_info.node_config.get("eps", 1e-5)

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

        # Layer normalisation
        gamma = params.weights.get("ln_gamma", jnp.ones((out_shape[-1],)))
        beta = params.weights.get("ln_beta", jnp.zeros((out_shape[-1],)))

        mean = jnp.mean(pre_activation, axis=-1, keepdims=True)
        var = jnp.var(pre_activation, axis=-1, keepdims=True)
        pre_activation = gamma * (pre_activation - mean) / jnp.sqrt(var + eps) + beta

        activation = node_info.activation
        z_mu = type(activation).forward(pre_activation, activation.config)

        error = state.z_latent - z_mu
        state = state._replace(pre_activation=pre_activation, z_mu=z_mu, error=error)
        node_class = node_info.node_class
        state = node_class.energy_functional(state, node_info)

        total_energy = jnp.sum(state.energy)
        return total_energy, state
