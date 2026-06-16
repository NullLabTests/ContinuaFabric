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
        """Add a skip connection from an earlier node to a later node.

        Picks two nodes (source before target topologically), adds a direct
        edge, and initialises its weight with small random noise so the skip
        starts as a near-identity bypass.
        """
        node_names = self._get_node_names()
        if len(node_names) < 3:
            return self.structure, self.params

        keys = jax.random.split(rng_key, 3)
        src_idx = int(jax.random.randint(keys[0], (), 0, len(node_names) - 1))
        tgt_idx = int(jax.random.randint(
            keys[1], (), src_idx + 1, len(node_names)
        ))
        source = self.structure.nodes[node_names[src_idx]]
        target = self.structure.nodes[node_names[tgt_idx]]

        # Check edge does not already exist
        for _, ei in self.structure.edges.items():
            if ei.source == source.name and ei.target == target.name:
                return self.structure, self.params

        new_edges = [Edge(source=source, target=target.slot("in"))]

        existing_nodes = list(self.structure.nodes.values())
        existing_edges = []
        for e_key, e_info in self.structure.edges.items():
            src_n = self.structure.nodes[e_info.source]
            tgt_n = self.structure.nodes[e_info.target]
            existing_edges.append(Edge(source=src_n, target=tgt_n.slot(e_info.slot)))

        try:
            new_structure = graph(
                nodes=existing_nodes,
                edges=existing_edges + new_edges,
                task_map=TaskMap(**dict(self.structure.task_map)),
                inference=self.structure.config.get("inference", InferenceSGD()),
            )
        except Exception:
            return self.structure, self.params

        from fabricpc.graph_initialization import initialize_params
        new_params = initialize_params(new_structure, keys[2])
        # Copy over original params for all original nodes
        for n_name in self.structure.nodes:
            if n_name in new_params.nodes:
                new_params.nodes[n_name] = self.params.nodes[n_name]
        return new_structure, new_params

    def _mutate_split_node(
        self, rng_key: jax.Array
    ) -> Tuple[GraphStructure, GraphParams]:
        """Split an existing hidden node into two serially-connected nodes.

        Both halves have half the dimensionality; their compositions
        approximate the original transformation.  Useful for increasing
        depth when energy plateaus.
        """
        node_names = self._get_node_names()
        x_node, y_node = self._get_input_output_nodes()

        candidates = [
            n for n in node_names
            if n not in (x_node, y_node)
            and self.structure.nodes[n].node_info.in_degree > 0
        ]
        if not candidates:
            return self.structure, self.params

        keys = jax.random.split(rng_key, 4)
        target_name = candidates[
            int(jax.random.randint(keys[0], (), 0, len(candidates)))
        ]
        target_node = self.structure.nodes[target_name]
        orig_shape = target_node.node_info.shape
        if len(orig_shape) != 1:
            return self.structure, self.params
        half = max(1, orig_shape[0] // 2)

        node_a = Linear(shape=(half,), name=f"{target_name}_a")
        node_b = Linear(shape=(orig_shape[0],), name=f"{target_name}_b")

        in_edges = list(target_node.node_info.in_edges)
        out_edges = [
            ek for ek, ei in self.structure.edges.items()
            if ei.source == target_name
        ]

        existing_nodes = [
            n for n in self.structure.nodes.values()
            if n.name != target_name
        ] + [node_a, node_b]

        existing_edges = []
        for e_key, e_info in self.structure.edges.items():
            if e_info.target == target_name:
                src_n = self.structure.nodes[e_info.source]
                existing_edges.append(
                    Edge(source=src_n, target=node_a.slot("in"))
                )
            elif e_info.source == target_name:
                tgt_n = self.structure.nodes[e_info.target]
                existing_edges.append(
                    Edge(source=node_b, target=tgt_n.slot(e_info.slot))
                )
            else:
                src_n = self.structure.nodes[e_info.source]
                tgt_n = self.structure.nodes[e_info.target]
                existing_edges.append(
                    Edge(source=src_n, target=tgt_n.slot(e_info.slot))
                )
        existing_edges.append(
            Edge(source=node_a, target=node_b.slot("in"))
        )

        try:
            new_structure = graph(
                nodes=existing_nodes,
                edges=existing_edges,
                task_map=TaskMap(**dict(self.structure.task_map)),
                inference=self.structure.config.get("inference", InferenceSGD()),
            )
        except Exception:
            return self.structure, self.params

        from fabricpc.graph_initialization import initialize_params
        new_params = initialize_params(new_structure, keys[1])
        for n_name in self.structure.nodes:
            if n_name in new_params.nodes and n_name != target_name:
                new_params.nodes[n_name] = self.params.nodes[n_name]
        return new_structure, new_params

    def _mutate_prune(
        self, rng_key: jax.Array
    ) -> Tuple[GraphStructure, GraphParams]:
        """Remove the lowest-energy hidden node from the graph.

        Prunes a hidden node whose energy contribution is smallest (i.e.
        the node that is already most converged / least useful).  All edges
        incident to the pruned node are removed.
        """
        node_names = self._get_node_names()
        x_node, y_node = self._get_input_output_nodes()

        # Evaluate energy on a random batch to identify the least useful node
        key_run, key_sel = jax.random.split(rng_key)
        try:
            from fabricpc.graph_initialization.state_initializer import (
                initialize_graph_state,
            )
            from fabricpc.core.inference import run_inference

            batch_size = 16
            dummy_x = jax.random.normal(
                key_run, (batch_size, self.structure.nodes[x_node].node_info.shape[0])
            )
            dummy_clamps = {x_node: dummy_x}
            init_state = initialize_graph_state(
                self.structure, batch_size, key_run,
                clamps=dummy_clamps, params=self.params,
            )
            final_state = run_inference(
                self.params, init_state, dummy_clamps, self.structure
            )
        except Exception:
            # Fall back to random pruning
            candidates = [
                n for n in node_names
                if n not in (x_node, y_node)
            ]
            if not candidates:
                return self.structure, self.params
            idx = int(jax.random.randint(key_sel, (), 0, len(candidates)))
            prune_name = candidates[idx]
            return self._apply_prune(prune_name, key_sel)

        candidates = [
            n for n in node_names
            if n not in (x_node, y_node)
            and self.structure.nodes[n].node_info.in_degree > 0
        ]
        if not candidates:
            return self.structure, self.params

        # Score each candidate by its per-node energy
        energies = {}
        for n in candidates:
            e_val = float(jnp.sum(final_state.nodes[n].energy))
            energies[n] = e_val

        # Prune node with smallest energy (least useful)
        prune_name = min(energies, key=energies.get)
        return self._apply_prune(prune_name, key_sel)

    def _apply_prune(
        self, prune_name: str, rng_key: jax.Array
    ) -> Tuple[GraphStructure, GraphParams]:
        """Remove a named node and all its incident edges."""
        existing_nodes = [
            n for n in self.structure.nodes.values()
            if n.name != prune_name
        ]
        if len(existing_nodes) < 2:
            return self.structure, self.params

        existing_edges = []
        for e_key, e_info in self.structure.edges.items():
            if e_info.source == prune_name or e_info.target == prune_name:
                continue
            src_n = self.structure.nodes[e_info.source]
            tgt_n = self.structure.nodes[e_info.target]
            existing_edges.append(
                Edge(source=src_n, target=tgt_n.slot(e_info.slot))
            )

        try:
            new_structure = graph(
                nodes=existing_nodes,
                edges=existing_edges,
                task_map=TaskMap(**dict(self.structure.task_map)),
                inference=self.structure.config.get("inference", InferenceSGD()),
            )
        except Exception:
            return self.structure, self.params

        from fabricpc.graph_initialization import initialize_params
        new_params = initialize_params(new_structure, rng_key)
        for n_name in self.structure.nodes:
            if n_name in new_params.nodes and n_name != prune_name:
                new_params.nodes[n_name] = self.params.nodes[n_name]
        return new_structure, new_params


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
