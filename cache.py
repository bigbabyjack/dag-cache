"""Cache management and planning strategies for dataset loading."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import wraps
from typing import TYPE_CHECKING, Callable, Literal

import polars as pl

from data import DatasetQuery, DatasetSpec, estimate_memory_usage

if TYPE_CHECKING:
    from dag import Context


@dataclass(frozen=True)
class MemoryEstimate:
    """Memory requirement estimate for a single execution step."""

    step: int
    node_id: str
    dataset: str
    estimated_mb: int


@dataclass(frozen=True)
class CacheCommand:
    """A command to load or evict a dataset from cache."""

    action: Literal["load", "evict"]
    dataset: str


type CachePlan = list[CacheCommand]


@dataclass
class Cache:
    """In-memory cache for datasets with hit/miss tracking."""

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
    """Decorator to cache dataset loading operations."""

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
    """Registry for dataset specifications with caching support."""

    datasets: dict[str, DatasetSpec] = field(default_factory=dict)
    _cache: Cache = field(default_factory=Cache)

    def get_spec(self, name: str) -> DatasetSpec:
        return self.datasets[name]

    @cached_method
    def get_data(self, name: str) -> pl.DataFrame:
        spec = self.get_spec(name)
        return spec.loader(spec.path)


def estimate_memory_requirements(context: Context) -> list[MemoryEstimate]:
    """Estimate memory requirements for all nodes in execution order."""
    from dag import get_node

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


def calculate_query_union(queries: list[DatasetQuery]) -> DatasetQuery:
    """Compute the minimal query that satisfies all given queries."""
    if not queries:
        return DatasetQuery(dataset="", filters=None, columns=None)

    # Union of columns: if any query needs None (all columns), union is None
    all_columns = [q.columns for q in queries if q.columns is not None]
    if len(all_columns) < len(queries):
        # At least one query needs all columns
        union_columns = None
    else:
        # Merge all column sets
        union_columns = sorted(set(col for cols in all_columns for col in cols))

    # Union of filters: if any query has no filter, union has no filter
    # Otherwise, we'd need to OR the filters (complex for now, simplify)
    all_filters = [q.filters for q in queries if q.filters is not None]
    if len(all_filters) < len(queries):
        union_filters = None
    else:
        # For now, if filters differ, load everything
        # In practice, you'd use pl.Expr.or_ to combine filters
        union_filters = None if len(set(map(str, all_filters))) > 1 else all_filters[0]

    return DatasetQuery(
        dataset=queries[0].dataset, filters=union_filters, columns=union_columns
    )


def estimate_query_memory(query: DatasetQuery, base_mb: int) -> int:
    """Estimate memory needed for a specific query."""
    if query.filters is None and query.columns is None:
        return base_mb

    # Simple heuristic: columns reduce proportionally
    column_factor = 1.0
    if query.columns is not None:
        # In practice, scan parquet schema to get actual column count
        # For now, assume ~50 columns total (from mock data)
        estimated_total_cols = 50
        column_factor = len(query.columns) / estimated_total_cols

    # Filter selectivity is harder - would need statistics or sampling
    # For now, assume filters reduce to 50% if present
    filter_factor = 0.5 if query.filters is not None else 1.0

    return int(base_mb * column_factor * filter_factor)


def look_ahead_dataset_usage(
    current_step: int,
    dataset: str,
    estimates: list[MemoryEstimate],
    queries: dict[str, DatasetQuery],
    window: int = 5,
) -> list[tuple[int, DatasetQuery]]:
    """
    Look ahead to find future steps that need this dataset.

    Returns: List of (step, query) tuples within the look-ahead window
    """
    future_usage = []
    for estimate in estimates[current_step + 1 : current_step + window + 1]:
        if estimate.dataset == dataset:
            query = queries.get(estimate.node_id)
            if query and query.dataset == dataset:
                future_usage.append((estimate.step, query))

    return future_usage


def should_expand_materialization(
    current_query: DatasetQuery,
    future_queries: list[DatasetQuery],
    base_mb: int,
    available_memory_mb: int,
) -> tuple[bool, DatasetQuery, int]:
    """
    Decide if we should load more data than minimally needed.

    Returns: (should_expand, expanded_query, io_savings)
    """
    if not future_queries:
        return False, current_query, 0

    # Calculate union query that satisfies current + future needs
    all_queries = [current_query] + future_queries
    union_query = calculate_query_union(all_queries)

    # Estimate costs
    minimal_memory = estimate_query_memory(current_query, base_mb)
    expanded_memory = estimate_query_memory(union_query, base_mb)
    expansion_cost = expanded_memory - minimal_memory

    # Estimate savings: each future query we satisfy avoids one I/O
    # I/O cost is roughly the data size we'd reload
    io_savings = sum(estimate_query_memory(q, base_mb) for q in future_queries)

    # Heuristic: expand if we have memory AND savings > expansion cost
    # This means: "pay memory now if it saves more I/O later"
    should_expand = (
        expansion_cost <= available_memory_mb
        and io_savings > expansion_cost * 1.5  # 1.5x multiplier: I/O is expensive!
    )

    return should_expand, union_query, io_savings


def plan_cache(estimates: list[MemoryEstimate], max_mb: int) -> CachePlan:
    """Simple greedy cache planner - kept for compatibility."""
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

    return plan


def plan_cache_with_lookahead(
    estimates: list[MemoryEstimate],
    queries: dict[str, DatasetQuery],
    max_mb: int,
    lookahead_window: int = 5,
) -> CachePlan:
    """
    Dynamic cache planning with look-ahead to optimize I/O vs memory.

    At each step:
    1. Look ahead to see what future steps need
    2. Decide whether to expand materialization beyond current needs
    3. Keep expanded data in cache if it's worth the memory cost
    """
    plan = []

    # Track what's currently loaded: dataset -> (query, memory_mb)
    loaded: dict[str, tuple[DatasetQuery, int]] = {}
    current_memory_mb = 0

    for i, estimate in enumerate(estimates):
        dataset = estimate.dataset
        node_query = queries.get(estimate.node_id, DatasetQuery(dataset=dataset))

        # Check if we already have this data cached
        if dataset in loaded:
            cached_query, cached_mb = loaded[dataset]

            # Check if cached data satisfies current query (query subsumption)
            # For simplicity: if cached has no filters/columns, it satisfies everything
            if cached_query.filters is None and cached_query.columns is None:
                print(f"Step {i}: Cache hit for {dataset} (already loaded)")
                continue

            # Otherwise, need to evict and reload (simplified)
            # In practice, check if cached columns ⊇ needed columns, etc.
            plan.append(CacheCommand(action="evict", dataset=dataset))
            current_memory_mb -= cached_mb
            del loaded[dataset]

        # Look ahead to see future usage of this dataset
        future_usage = look_ahead_dataset_usage(
            current_step=i,
            dataset=dataset,
            estimates=estimates,
            queries=queries,
            window=lookahead_window,
        )

        future_queries = [query for _, query in future_usage]
        available_memory = max_mb - current_memory_mb

        # Decide whether to expand materialization
        should_expand, materialized_query, io_savings = should_expand_materialization(
            current_query=node_query,
            future_queries=future_queries,
            base_mb=estimate.estimated_mb,
            available_memory_mb=available_memory,
        )

        # Calculate actual memory needed
        query_to_load = materialized_query if should_expand else node_query
        memory_needed = estimate_query_memory(query_to_load, estimate.estimated_mb)

        # Evict if needed to make space
        if current_memory_mb + memory_needed > max_mb:
            # Simple eviction: evict all (LRU would be smarter)
            for ds in list(loaded.keys()):
                plan.append(CacheCommand(action="evict", dataset=ds))
                current_memory_mb -= loaded[ds][1]
            loaded.clear()

        # Load the data
        action_type = "load (expanded)" if should_expand else "load"
        print(
            f"Step {i}: {action_type} {dataset} "
            f"({memory_needed}MB, saves {io_savings}MB I/O)"
        )

        plan.append(CacheCommand(action="load", dataset=dataset))
        loaded[dataset] = (query_to_load, memory_needed)
        current_memory_mb += memory_needed

    return plan


def fingerprint_query(query: DatasetQuery) -> str:
    """Create a hashable fingerprint for a query."""
    return f"{query.dataset}:{query.filters}:{tuple(query.columns) if query.columns else 'ALL'}"


def query_satisfies(loaded_query: DatasetQuery, needed_query: DatasetQuery) -> bool:
    """Check if a loaded query can satisfy a needed query (subsumption)."""
    # Same dataset check
    if loaded_query.dataset != needed_query.dataset:
        return False

    # Column check: loaded must include all needed columns
    if needed_query.columns is not None:
        if loaded_query.columns is None:
            return True  # Loaded has all columns
        if not set(needed_query.columns).issubset(set(loaded_query.columns)):
            return False

    # Filter check: simplified - in practice would check filter subsumption
    # For now: if loaded has no filter, it satisfies any filter
    if needed_query.filters is not None and loaded_query.filters is not None:
        if str(loaded_query.filters) != str(needed_query.filters):
            return False

    return True
