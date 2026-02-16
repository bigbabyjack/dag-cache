from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import polars as pl


def generate_dataset(
    name: str, n_rows: int, n_cols: int, path: Path = Path("./data/")
) -> DatasetSpec:
    df = pl.DataFrame({f"col_{i}": np.arange(0, n_rows) for i in range(n_cols)})
    dataset_spec = DatasetSpec(name=name, path=path / f"{name}.parquet")
    df.write_parquet(dataset_spec.path, mkdir=True)
    return dataset_spec


def make_mock_data() -> list[DatasetSpec]:
    if not Path("./data/").exists():
        Path("./data/").mkdir()
    return [
        generate_dataset(name="a", n_rows=100_000, n_cols=50),
        generate_dataset(name="b", n_rows=200_000, n_cols=100),
        generate_dataset(name="c", n_rows=50_000, n_cols=30),
    ]


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    loader: Callable = lambda p: pl.read_parquet(p)
    path: Path = Path("./data/name.parquet")


def estimate_memory_usage(dataset_spec: DatasetSpec) -> int:
    estimate = dataset_spec.path.stat().st_size
    return estimate


def plan_memory_management(
    context: Context, max_rss_mb: int = 512
) -> list[tuple[str, int]]:
    order = context.ordered_nodes
    memory_plan = []
    current_rss = 0
    for node_id in order:
        node = get_node(node_id, context.node_registry)
        assert node is not None

        for dataset_name in node.data_dependencies:
            dataset_spec = context.data_registry.get_spec(dataset_name)
            dataset_size = estimate_memory_usage(dataset_spec)
            current_rss += dataset_size

        memory_plan.append((node_id, current_rss))

        if current_rss > max_rss_mb * 1024 * 1024:
            print(
                f"Warning: Memory usage for node {node_id} exceeds {max_rss_mb} MB (estimated {current_rss / (1024 * 1024):.2f} MB)"
            )
    return memory_plan


@dataclass
class Cache:
    data: dict[str, pl.DataFrame] = field(default_factory=dict)

    _hits: dict[str, int] = field(default_factory=dict)
    _misses: dict[str, int] = field(default_factory=dict)

    def get(self, name: str) -> pl.DataFrame | None:
        if name not in self.data:
            self._misses[name] = self._misses.get(name, 0) + 1
            return None

        self._hits[name] = self._hits.get(name, 0) + 1
        return self.data[name]

    def set(self, name: str, df: pl.DataFrame) -> None:
        self.data[name] = df


def cached_method(
    func: Callable[[DataRegistry, str], pl.DataFrame],
) -> Callable[[DataRegistry, str], pl.DataFrame]:
    def wrapper(self: DataRegistry, name: str) -> pl.DataFrame:
        cached = self._cache.get(name)
        if cached is not None:
            return cached

        result = func(self, name)
        self._cache.set(name, result)
        return result

    return wrapper


@dataclass
class DataRegistry:
    datasets: dict[str, DatasetSpec] = field(default_factory=dict)
    _cache: Cache = field(default_factory=Cache)

    def get_spec(self, name: str) -> DatasetSpec:
        return self.datasets[name]

    @cached_method
    def get_data(self, name: str) -> pl.DataFrame:
        spec = self.get_spec(name)
        return spec.loader(spec.path)


@dataclass(frozen=True)
class Context:
    config: dict[str, Any] = field(default_factory=dict)

    dag: nx.DiGraph = field(default_factory=nx.DiGraph)
    data_registry: DataRegistry = field(default_factory=DataRegistry)
    node_registry: NodeRegistry = field(default_factory=dict)

    dry_run_enabled: bool = False

    @contextmanager
    def dry_run(self):
        original_dry_run_enabled = self.dry_run_enabled
        object.__setattr__(self, "dry_run_enabled", True)
        try:
            yield
        finally:
            object.__setattr__(self, "dry_run_enabled", original_dry_run_enabled)

    @property
    def ordered_nodes(self) -> list[str]:
        return list(nx.topological_sort(self.dag))


def register_dependencies(func):
    def wrapper(self: Node, context: Context) -> list[str] | None:
        if context.dry_run_enabled:
            print(f"Registering dependencies for node {self.id}")
            return self.data_dependencies

        return func(self, context)

    return wrapper


@dataclass(frozen=True)
class Node:
    id: str
    depends_on: list[str]
    data_dependencies: list[str] = field(default_factory=list)

    @register_dependencies
    def run(self, context: Context) -> list[str] | None:
        print(f"Running node {self.id}")


def add_nodes_to_graph(g: nx.DiGraph, nodes: list[Node]):
    for node in nodes:
        g.add_node(node.id)
        for dependency in node.depends_on:
            g.add_edge(dependency, node.id)


type NodeRegistry = dict[str, Node]


def get_node(
    node: str,
    node_registry: NodeRegistry,
) -> Node | None:
    return node_registry[node]


def order_graph(g: nx.DiGraph) -> list[str]:
    # TODO: run in parallel using generations
    return list(nx.topological_sort(g))


def run(context: Context):
    ordered_nodes = order_graph(context.dag)
    for node_id in ordered_nodes:
        node = get_node(node_id, context.node_registry)
        assert node is not None

        node.run(context)


def make_context() -> Context:
    data_registry = DataRegistry()
    for spec in make_mock_data():
        data_registry.datasets[spec.name] = spec

    node_registry: NodeRegistry = {
        "1": Node(id="1", depends_on=[]),
        "2": Node(id="2", depends_on=["1"]),
        "3": Node(id="3", depends_on=["1", "2"]),
        "4": Node(id="4", depends_on=["1", "3"]),
    }

    g = nx.DiGraph()
    add_nodes_to_graph(g, list(node_registry.values()))

    context = Context(data_registry=data_registry, node_registry=node_registry, dag=g)
    return context


def main():
    context = make_context()
    with context.dry_run():
        run(context)

    memory_plan = plan_memory_management(context, max_rss_mb=512)
    print("Memory plan:")
    for node_id, estimated_rss in memory_plan:
        print(f"Node {node_id}: estimated RSS {estimated_rss / (1024 * 1024):.2f} MB")

    run(context)


if __name__ == "__main__":
    main()
