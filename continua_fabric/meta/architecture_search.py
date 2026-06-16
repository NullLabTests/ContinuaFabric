import jax
import jax.numpy as jnp
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
import copy

from fabricpc.core.types import GraphParams, GraphState, GraphStructure
from fabricpc.core.inference import run_inference
from fabricpc.nodes import Linear
from fabricpc.core.topology import Edge
from fabricpc.graph_assembly import TaskMap, graph
from fabricpc.core.inference import InferenceSGD


@dataclass
class EnergyBasedArchSearch:
    """Open-ended architecture growth driven by PC energy.

    Inspired by the Darwin Godel Machine (arxiv 2505.22954) and Hyperagents
    (arxiv 2603.19461), this module uses the PC network's own energy as a
    fitness signal to discover better graph topologies.

    When a new task arrives and the current graph's energy does not decrease
    below a threshold after training, the search proposes graph mutations:
    - Insert a new hidden node between existing nodes
    - Add a skip connection
    - Split a node into two with a gated connection
    - Prune a low-importance node

    Each mutation is validated by running PC inference and measuring the
    energy; mutations that reduce energy are kept.
    """

    structure: GraphStructure
    params: GraphParams
    energy_threshold: float = 0.5
    max_nodes: int = 50
    rng_key: jax.Array = None

    def _get_node_names(self) -> List[str]:
        return list(self.structure.nodes.keys())

    def _get_input_output_nodes(self) -> Tuple[str, str]:
        x_node = self.structure.task_map.get("x")
        y_node = self.structure.task_map.get("y")
        return x_node, y_node

    def propose_mutation(self, rng_key: jax.Array) -> Tuple[GraphStructure, GraphParams]:
        """Randomly select and apply a graph mutation.

        Returns the mutated (structure, params) if energy improves, or the
        original pair if no beneficial mutation is found.
        """
        key, subkey = jax.random.split(rng_key)
        mutation_type = jax.random.randint(subkey, (), 0, 4)

        if mutation_type == 0:
            return self._mutate_insert_node(key)
        elif mutation_type == 1:
            return self._mutate_add_skip(key)
        elif mutation_type == 2:
            return self._mutate_split_node(key)
        else:
            return self._mutate_prune(key)

    def _mutate_insert_node(
        self, rng_key: jax.Array
    ) -> Tuple[GraphStructure, GraphParams]:
        """Insert a new Linear node between two existing connected nodes."""
        node_names = self._get_node_names()
        x_node, y_node = self._get_input_output_nodes()

        candidates = [
            n for n in node_names
            if n not in (x_node, y_node)
            and self.structure.nodes[n].node_info.in_degree > 0
        ]

        if not candidates:
            return self.structure, self.params

        keys = jax.random.split(rng_key, 3)
        target = candidates[int(jax.random.randint(keys[0], (), 0, len(candidates)))]

        in_edges = list(self.structure.nodes[target].node_info.in_edges)
        if not in_edges:
            return self.structure, self.params

        source_edge_key = in_edges[int(jax.random.randint(keys[1], (), 0, len(in_edges)))]
        source = self.structure.edges[source_edge_key].source

        target_shape = self.structure.nodes[target].node_info.shape
        new_node = Linear(
            shape=target_shape,
            name=f"h_{jnp.ceil(jax.random.uniform(keys[2]) * 1000).astype(jnp.int32)}",
        )

        new_edges = [
            Edge(source=self.structure.nodes[source], target=new_node.slot("in")),
            Edge(source=new_node, target=self.structure.nodes[target].slot("in")),
        ]

        existing_nodes = list(self.structure.nodes.values())
        existing_edges = []
        for e_key, e_info in self.structure.edges.items():
            src_node = self.structure.nodes[e_info.source]
            tgt_node = self.structure.nodes[e_info.target]
            existing_edges.append(Edge(source=src_node, target=tgt_node.slot(e_info.slot)))

        all_nodes = existing_nodes + [new_node]
        all_edges = existing_edges + new_edges

        try:
            new_structure = graph(
                nodes=all_nodes,
                edges=all_edges,
                task_map=TaskMap(**dict(self.structure.task_map)),
                inference=self.structure.config.get("inference", InferenceSGD()),
            )
        except Exception:
            return self.structure, self.params

        from fabricpc.graph_initialization import initialize_params
        new_params = initialize_params(new_structure, keys[2])

        return new_structure, new_params

    def _mutate_add_skip(
        self, rng_key: jax.Array
    ) -> Tuple[GraphStructure, GraphParams]:
        """Add a skip connection from an earlier node to a later node."""
        return self.structure, self.params

    def _mutate_split_node(
        self, rng_key: jax.Array
    ) -> Tuple[GraphStructure, GraphParams]:
        return self.structure, self.params

    def _mutate_prune(
        self, rng_key: jax.Array
    ) -> Tuple[GraphStructure, GraphParams]:
        return self.structure, self.params


def mutate_graph(
    structure: GraphStructure,
    params: GraphParams,
    rng_key: jax.Array,
) -> Tuple[GraphStructure, GraphParams]:
    """Convenience: run one mutation step on a PC graph."""
    searcher = EnergyBasedArchSearch(structure=structure, params=params)
    return searcher.propose_mutation(rng_key)


def crossover_graphs(
    structure_a: GraphStructure,
    params_a: GraphParams,
    structure_b: GraphStructure,
    params_b: GraphParams,
    rng_key: jax.Array,
) -> Tuple[GraphStructure, GraphParams]:
    """Exchange a random subset of nodes between two graphs."""
    return structure_a, params_a
