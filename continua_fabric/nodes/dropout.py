import jax
import jax.numpy as jnp
from typing import Dict, Any, Optional, Tuple
from fabricpc.nodes.base import NodeBase, SlotSpec
from fabricpc.core.types import NodeParams, NodeState, NodeInfo
from fabricpc.core.energy import GaussianEnergy
from fabricpc.core.initializers import NormalInitializer


class DropoutPC(NodeBase):
    """PC node with stochastic dropout on the latent state during inference.

    During inference (training mode), each element of z_latent is set to
    zero with probability `drop_rate`.  The remaining elements are scaled
    by 1/(1 - drop_rate) to preserve expected magnitude.

    Dropout in a predictive coding context acts as a regulariser on the
    inference dynamics — it prevents the latent code from co-adapting
    to specific error signals and encourages robustness, which is
    beneficial for continual learning (reduces task-specific overfitting).

    At test / evaluation time, dropout is not applied and the node acts
    as an identity passthrough.

    Note: this node does *not* have learnable weights or a linear
    transformation.  It simply gates the latent state.  In a PC graph,
    it should be placed between a linear transform node and the next
    layer's input slot.
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        name: str,
        drop_rate: float = 0.2,
        energy=GaussianEnergy(),
        latent_init=NormalInitializer(),
    ):
        super().__init__(
            shape=shape,
            name=name,
            activation=None,
            energy=energy,
            latent_init=latent_init,
            weight_init=None,
            use_bias=False,
            flatten_input=False,
            drop_rate=drop_rate,
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
        return NodeParams(weights={}, biases={})

    @staticmethod
    def forward(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],
        state: NodeState,
        node_info: NodeInfo,
    ) -> tuple[jax.Array, NodeState]:
        batch_size = state.z_latent.shape[0]
        out_shape = node_info.shape
        drop_rate = node_info.node_config.get("drop_rate", 0.0)

        # Sum all inputs as the pre-activation (passthrough)
        pre_activation = jnp.zeros((batch_size,) + out_shape)
        for x in inputs.values():
            pre_activation = pre_activation + x

        # In a PC node, z_mu is the expected value of z_latent.
        # During training, z_latent is dropped out; z_mu remains clean.
        # The error = z_latent - z_mu then drives the learning signal
        # through the dropped-out pathway.
        z_mu = pre_activation

        # Apply dropout to the latent state only (not z_mu)
        training = node_info.node_config.get("training", True)
        if training and drop_rate > 0.0:
            key = node_info.node_config.get("dropout_key", jax.random.PRNGKey(0))
            mask = jax.random.bernoulli(key, 1.0 - drop_rate, (batch_size,) + out_shape)
            scale = 1.0 / (1.0 - drop_rate + 1e-8)
            z_latent_dropped = state.z_latent * mask * scale
        else:
            z_latent_dropped = state.z_latent

        error = z_latent_dropped - z_mu
        state = state._replace(
            pre_activation=pre_activation,
            z_mu=z_mu,
            z_latent=z_latent_dropped,
            error=error,
        )
        node_class = node_info.node_class
        state = node_class.energy_functional(state, node_info)

        total_energy = jnp.sum(state.energy)
        return total_energy, state
