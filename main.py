from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import Any, Literal

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
class DatasetQuery:
    dataset: str
    filters: pl.Expr | None = None
    columns: list[str] | None = None


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    loader: Callable = field(default=lambda path: pl.read_parquet(path))
    path: Path = Path("./data/name.parquet")


@dataclass(frozen=True)
class Dataset:
    spec: DatasetSpec
    data: pl.DataFrame

    query: DatasetQuery | None = None


def memory_usage(dataset: Dataset) -> int:
    return int(dataset.data.estimated_size())


def estimate_memory_usage(dataset_spec: DatasetSpec) -> int:
    estimate = dataset_spec.path.stat().st_size
    return estimate


@dataclass(frozen=True)
class MemoryEstimate:
    step: int
    node_id: str
    dataset: str
    estimated_mb: int


def estimate_memory_requirements(context: Context) -> list[MemoryEstimate]:
    assert context.dry_run_enabled, (
        "Memory estimation should only be run in dry run mode"
    )
    order = context.ordered_nodes
    print(f"Estimating memory requirements for execution order: {order}")
    memory_requirements = []
    for step, node_id in enumerate(order):
        print(f"Estimating memory for node {node_id} at step {step}")
        node = get_node(node_id, context.node_registry)
        assert node is not None

        for dataset_name in node.data_dependencies:
            print(
                f"Estimating memory for dataset {dataset_name} required by node {node_id}"
            )
            dataset_spec = context.data_registry.get_spec(dataset_name)
            dataset_size = estimate_memory_usage(dataset_spec)
            print(dataset_size)

            memory_requirements.append(
                MemoryEstimate(
                    step=step,
                    node_id=node_id,
                    dataset=dataset_name,
                    estimated_mb=int(dataset_size / (1024 * 1024)),
                )
            )

    return memory_requirements


@dataclass(frozen=True)
class CacheCommand:
    action: Literal["load", "evict"]
    dataset: str


type CachePlan = list[CacheCommand]


def plan_cache(estimates: list[MemoryEstimate], max_mb: int) -> CachePlan:
    plan = []
    loaded: dict[str, int] = {}
    current_rss = 0
    for estimate in estimates:
        if current_rss + estimate.estimated_mb < max_mb:
            plan.append(CacheCommand(action="load", dataset=estimate.dataset))
            loaded[estimate.dataset] = estimate.estimated_mb
            current_rss += estimate.estimated_mb

        if current_rss + estimate.estimated_mb >= max_mb:
            for dataset, size in loaded.items():
                plan.append(CacheCommand(action="evict", dataset=dataset))
                current_rss -= size
            loaded = {}
            plan.append(CacheCommand(action="load", dataset=estimate.dataset))
            loaded[estimate.dataset] = estimate.estimated_mb
            current_rss += estimate.estimated_mb


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
    @wraps(func)
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
    @wraps(func)
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
        "1": Node(id="1", depends_on=[], data_dependencies=["a"]),
        "2": Node(id="2", depends_on=["1"], data_dependencies=["b"]),
        "3": Node(id="3", depends_on=["1", "2"], data_dependencies=["a", "c"]),
        "4": Node(id="4", depends_on=["1", "3"], data_dependencies=["b", "c"]),
    }

    g = nx.DiGraph()
    add_nodes_to_graph(g, list(node_registry.values()))

    context = Context(data_registry=data_registry, node_registry=node_registry, dag=g)
    return context


def main():
    context = make_context()
    with context.dry_run():
        run(context)


if __name__ == "__main__":
    main()
