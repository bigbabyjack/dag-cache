"""Demo and comparison scripts for DAG cache planning."""

from __future__ import annotations

import networkx as nx

from cache import (
    DataRegistry,
    estimate_memory_requirements,
    plan_cache_with_lookahead,
)
from dag import Context, Node, NodeRegistry, add_nodes_to_graph
from data import DatasetQuery, make_mock_data
from optimization import plan_cache_optimal


def make_context() -> Context:
    """Create a sample execution context for demos."""
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


def demo_lookahead_planning():
    """Demonstrate look-ahead cache planning with sample queries."""
    context = make_context()

    # Define queries for each node - showing different access patterns
    queries = {
        "1": DatasetQuery(
            dataset="a", columns=["col_0", "col_1", "col_2"], filters=None
        ),
        "2": DatasetQuery(dataset="b", columns=["col_0", "col_1"], filters=None),
        # Node 3 needs dataset "a" again - overlaps with node 1!
        "3": DatasetQuery(dataset="a", columns=["col_0", "col_1"], filters=None),
        "4": DatasetQuery(dataset="b", columns=None, filters=None),  # Needs all of "b"
    }

    print("=" * 60)
    print("Generating memory estimates...")
    print("=" * 60)

    with context.dry_run():
        estimates = estimate_memory_requirements(context)

    print("\n" + "=" * 60)
    print("HEURISTIC: Planning cache with look-ahead (budget: 50MB)...")
    print("=" * 60 + "\n")

    heuristic_plan = plan_cache_with_lookahead(
        estimates=estimates,
        queries=queries,
        max_mb=50,
        lookahead_window=3,
    )

    print("\n" + "=" * 60)
    print("Heuristic Plan:")
    print("=" * 60)
    for i, cmd in enumerate(heuristic_plan):
        print(f"{i}: {cmd.action:6s} {cmd.dataset}")

    print("\n" + "=" * 60)
    print("OPTIMAL: Planning cache with DP (budget: 50MB)...")
    print("=" * 60 + "\n")

    optimal_plan, optimal_cost = plan_cache_optimal(
        estimates=estimates,
        queries=queries,
        max_mb=50,
    )

    print("\n" + "=" * 60)
    print(f"Optimal Plan (Total I/O cost: {optimal_cost}MB):")
    print("=" * 60)
    for i, cmd in enumerate(optimal_plan):
        print(f"{i}: {cmd.action:6s} {cmd.dataset}")


def demo_optimization_comparison():
    """
    Compare different optimization approaches for cache planning.

    Demonstrates the trade-offs between:
    1. Greedy heuristic (fast, approximate)
    2. Look-ahead heuristic (medium speed, better)
    3. DP optimal (slower, provably best)
    4. ILP optimal (slowest, most flexible)
    """
    print(
        """
╔════════════════════════════════════════════════════════════════╗
║           CACHE PLANNING OPTIMIZATION COMPARISON               ║
╠════════════════════════════════════════════════════════════════╣
║                                                                ║
║  Approach         | Optimality | Time      | Use When         ║
║  ────────────────────────────────────────────────────────────  ║
║  Greedy           | ❌ No      | O(n)      | Quick prototypes ║
║  Look-ahead       | ❌ No      | O(n×w)    | Production       ║
║  DP               | ✅ Yes     | O(n×2^k)  | Known queries    ║
║  ILP              | ✅ Yes     | Exp       | Complex rules    ║
║                                                                ║
╠════════════════════════════════════════════════════════════════╣
║  Key Insights:                                                 ║
║                                                                ║
║  • DP is optimal when:                                         ║
║    - All queries known in advance (offline)                    ║
║    - State space is tractable (k datasets)                     ║
║    - Optimal substructure exists (DAGs ✓)                      ║
║                                                                ║
║  • ILP is optimal when:                                        ║
║    - Complex constraints (quotas, SLAs, costs)                 ║
║    - Willing to wait for solver (seconds to minutes)           ║
║    - Need certifiable optimality for compliance                ║
║                                                                ║
║  • Heuristics are best when:                                   ║
║    - Need sub-second planning                                  ║
║    - "Good enough" beats "perfect but slow"                    ║
║    - Online setting (queries arrive dynamically)               ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
    """
    )


if __name__ == "__main__":
    demo_lookahead_planning()
