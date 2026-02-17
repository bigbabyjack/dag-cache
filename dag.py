"""DAG execution framework for orchestrating computation workflows."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import wraps
from typing import Any

import networkx as nx

from cache import DataRegistry


type NodeRegistry = dict[str, Node]


@dataclass(frozen=True)
class Context:
    """Execution context for DAG-based computations."""

    config: dict[str, Any] = field(default_factory=dict)

    dag: nx.DiGraph = field(default_factory=nx.DiGraph)
    data_registry: DataRegistry = field(default_factory=DataRegistry)
    node_registry: NodeRegistry = field(default_factory=dict)

    dry_run_enabled: bool = False

    @contextmanager
    def dry_run(self):
        """Enable dry run mode for dependency analysis without execution."""
        original_dry_run_enabled = self.dry_run_enabled
        object.__setattr__(self, "dry_run_enabled", True)
        try:
            yield
        finally:
            object.__setattr__(self, "dry_run_enabled", original_dry_run_enabled)

    @property
    def ordered_nodes(self) -> list[str]:
        """Get nodes in topological execution order."""
        return list(nx.topological_sort(self.dag))


def register_dependencies(func: Callable) -> Callable:
    """Decorator to register node dependencies during dry run."""

    @wraps(func)
    def wrapper(self: Node, context: Context) -> list[str] | None:
        if context.dry_run_enabled:
            print(f"Registering dependencies for node {self.id}")
            return self.data_dependencies

        return func(self, context)

    return wrapper


@dataclass(frozen=True)
class Node:
    """A computation node in the DAG."""

    id: str
    depends_on: list[str]
    data_dependencies: list[str] = field(default_factory=list)

    @register_dependencies
    def run(self, context: Context) -> list[str] | None:
        """Execute this node's computation."""
        print(f"Running node {self.id}")


def add_nodes_to_graph(g: nx.DiGraph, nodes: list[Node]) -> None:
    """Add nodes and their dependencies to a graph."""
    for node in nodes:
        g.add_node(node.id)
        for dependency in node.depends_on:
            g.add_edge(dependency, node.id)


def get_node(node: str, node_registry: NodeRegistry) -> Node | None:
    """Retrieve a node from the registry."""
    return node_registry[node]


def order_graph(g: nx.DiGraph) -> list[str]:
    """Determine execution order for a DAG."""
    # TODO: run in parallel using generations
    return list(nx.topological_sort(g))


def run(context: Context) -> None:
    """Execute all nodes in the DAG in topological order."""
    ordered_nodes = order_graph(context.dag)
    for node_id in ordered_nodes:
        node = get_node(node_id, context.node_registry)
        assert node is not None

        node.run(context)
