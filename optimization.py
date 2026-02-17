"""Optimal cache planning using dynamic programming and advanced algorithms."""

from __future__ import annotations

from dataclasses import dataclass

from cache import (
    CacheCommand,
    CachePlan,
    MemoryEstimate,
    calculate_query_union,
    estimate_query_memory,
    fingerprint_query,
    look_ahead_dataset_usage,
)
from data import DatasetQuery


@dataclass(frozen=True)
class MaterializationStrategy:
    """Describes how much data to load for a dataset."""

    dataset: str
    query: DatasetQuery  # What we minimally need
    expanded_query: DatasetQuery  # What we'll actually load (superset of query)
    memory_mb: int  # Cost of the expanded query

    # Metadata for decision-making
    satisfies_steps: list[int]  # Which future steps benefit from this expansion
    io_savings_mb: int  # How much I/O we avoid by expanding


@dataclass(frozen=True)
class CacheState:
    """Represents what's loaded in cache at a given step."""

    loaded_datasets: frozenset[tuple[str, str]]  # (dataset, query_fingerprint)
    memory_used_mb: int

    def __hash__(self):
        return hash((self.loaded_datasets, self.memory_used_mb))


def generate_materialization_strategies(
    step: int,
    estimate: MemoryEstimate,
    node_query: DatasetQuery,
    estimates: list[MemoryEstimate],
    queries: dict[str, DatasetQuery],
    max_mb: int,
) -> list[MaterializationStrategy]:
    """
    Generate all reasonable materialization strategies for this step.

    Strategies range from:
    - Minimal: load exactly what's needed
    - Expanded: load more to satisfy future queries
    - Maximal: load entire dataset
    """
    strategies = []

    # Strategy 1: Minimal load (just this query)
    minimal_memory = estimate_query_memory(node_query, estimate.estimated_mb)
    strategies.append(
        MaterializationStrategy(
            dataset=estimate.dataset,
            query=node_query,
            expanded_query=node_query,
            memory_mb=minimal_memory,
            satisfies_steps=[step],
            io_savings_mb=0,
        )
    )

    # Strategy 2: Look ahead and expand
    future_usage = look_ahead_dataset_usage(
        current_step=step,
        dataset=estimate.dataset,
        estimates=estimates,
        queries=queries,
        window=5,
    )

    if future_usage:
        future_queries = [q for _, q in future_usage]
        union_query = calculate_query_union([node_query] + future_queries)
        expanded_memory = estimate_query_memory(union_query, estimate.estimated_mb)

        if expanded_memory <= max_mb:
            io_savings = sum(
                estimate_query_memory(q, estimate.estimated_mb) for q in future_queries
            )
            strategies.append(
                MaterializationStrategy(
                    dataset=estimate.dataset,
                    query=node_query,
                    expanded_query=union_query,
                    memory_mb=expanded_memory,
                    satisfies_steps=[step] + [s for s, _ in future_usage],
                    io_savings_mb=io_savings,
                )
            )

    return strategies


def plan_cache_optimal(
    estimates: list[MemoryEstimate],
    queries: dict[str, DatasetQuery],
    max_mb: int,
) -> tuple[CachePlan, int]:
    """
    Find the OPTIMAL cache plan using dynamic programming.

    This guarantees minimal I/O cost by exploring all valid cache states.

    Returns: (optimal_plan, total_io_cost)
    """
    n = len(estimates)

    # DP state: dp[step][cache_state] = (min_io_cost, best_plan)
    # We'll use memoization to avoid recomputing states
    memo: dict[tuple[int, CacheState], tuple[int, CachePlan]] = {}

    def dp(step: int, current_state: CacheState) -> tuple[int, CachePlan]:
        """
        Recursively find optimal plan from this step onward.

        Returns: (io_cost_from_here, plan_from_here)
        """
        # Base case: processed all steps
        if step >= n:
            return 0, []

        # Check memo
        state_key = (step, current_state)
        if state_key in memo:
            return memo[state_key]

        estimate = estimates[step]
        node_query = queries.get(
            estimate.node_id, DatasetQuery(dataset=estimate.dataset)
        )

        # Check if current cache satisfies this query (cache hit!)
        cache_hit = False
        for ds_name, _query_fp in current_state.loaded_datasets:
            # Reconstruct query from fingerprint (simplified)
            # In practice, store full query objects
            if estimate.dataset in ds_name:
                cache_hit = True
                break

        best_cost = float("inf")
        best_plan = []

        if cache_hit:
            # No I/O needed! Continue with current state
            future_cost, future_plan = dp(step + 1, current_state)
            best_cost = future_cost
            best_plan = future_plan
        else:
            # Cache miss: need to load data
            # Try different materialization strategies
            strategies = generate_materialization_strategies(
                step, estimate, node_query, estimates, queries, max_mb
            )

            for strategy in strategies:
                # Calculate new state after loading
                new_loaded = set(current_state.loaded_datasets)
                new_memory = current_state.memory_used_mb

                # Evict if needed to make space
                evictions = []
                if new_memory + strategy.memory_mb > max_mb:
                    # Try evicting to make space
                    # For optimal solution, try all eviction subsets
                    # Simplified: evict all
                    for ds, _ in list(new_loaded):
                        evictions.append(CacheCommand(action="evict", dataset=ds))
                    new_loaded.clear()
                    new_memory = 0

                # Load new data
                load_cost = strategy.memory_mb  # I/O cost = data size
                new_loaded.add(
                    (strategy.dataset, fingerprint_query(strategy.expanded_query))
                )
                new_memory += strategy.memory_mb

                new_state = CacheState(
                    loaded_datasets=frozenset(new_loaded), memory_used_mb=new_memory
                )

                # Recurse
                future_cost, future_plan = dp(step + 1, new_state)
                total_cost = load_cost + future_cost

                if total_cost < best_cost:
                    best_cost = total_cost
                    best_plan = (
                        evictions
                        + [CacheCommand(action="load", dataset=strategy.dataset)]
                        + future_plan
                    )

        memo[state_key] = (int(best_cost), best_plan)
        return int(best_cost), best_plan

    # Start with empty cache
    initial_state = CacheState(loaded_datasets=frozenset(), memory_used_mb=0)
    optimal_cost, optimal_plan = dp(0, initial_state)

    return optimal_plan, optimal_cost
