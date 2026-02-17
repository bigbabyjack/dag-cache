# CLAUDE.md

## Project Overview

dag-cache is a Python library for dynamic cache optimization using dry-run DAG (Directed Acyclic Graph) execution. It models computation as a graph of nodes with data dependencies, performs dry-run execution to estimate memory requirements, and plans cache load/evict strategies to stay within memory budgets.

## Tech Stack

- **Language**: Python 3.13 (pinned in `.python-version`)
- **Package manager**: [uv](https://docs.astral.sh/uv/)
- **Core dependencies**: polars (DataFrames), networkx (DAG operations), matplotlib (visualization)
- **Dev tools**: ruff (linting + formatting), ty (type checking)

## Project Structure

This is a single-module project. All code lives in `main.py`.

```
main.py          # All source code (DAG, cache, data registry, execution engine)
pyproject.toml   # Project config, dependencies, tool settings
uv.lock          # Locked dependencies
data/            # Generated parquet files (gitignored)
```

## Commands

All commands use `uv run` to execute within the project's virtual environment.

```bash
# Run the application
uv run python main.py

# Lint
uv run ruff check .

# Format (check only)
uv run ruff format --check .

# Format (apply)
uv run ruff format .

# Type check
uv run ty check main.py
```

There are no tests in this project currently.

## Architecture

### Key Data Types

- **`DatasetSpec`** — Metadata for a dataset: name, loader function, file path.
- **`Dataset`** — A loaded dataset: spec + polars DataFrame.
- **`DatasetQuery`** — A query against a dataset with optional filters/column selection.
- **`Node`** — A DAG node with an ID, execution dependencies (`depends_on`), and data dependencies (`data_dependencies`).
- **`Context`** — Central execution context holding the DAG, data registry, node registry, and config. Supports a `dry_run()` context manager.
- **`Cache`** — In-memory DataFrame cache with hit/miss tracking.
- **`DataRegistry`** — Registry of dataset specs with cached data loading via `@cached_method`.

### Execution Flow

1. `make_context()` — Creates mock datasets (parquet files) and builds a DAG of nodes with dependencies.
2. `run(context)` — Topologically sorts the DAG and executes nodes in order.
3. **Dry-run mode** — `context.dry_run()` enables dry-run where `Node.run()` returns data dependency names instead of executing. Used by `estimate_memory_requirements()`.
4. **Cache planning** — `plan_cache()` takes memory estimates and a budget, producing a sequence of load/evict commands.

### Key Patterns

- **Frozen dataclasses** — Most data types use `@dataclass(frozen=True)` for immutability.
- **Decorator-based behavior** — `@register_dependencies` intercepts `Node.run()` during dry-run; `@cached_method` adds caching to `DataRegistry.get_data()`.
- **Type aliases** — `NodeRegistry = dict[str, Node]`, `CachePlan = list[CacheCommand]`.

## Code Style & Conventions

- **Formatter**: ruff format (88 char line length, double quotes, space indentation)
- **Linting rules**: pycodestyle, pyflakes, isort, pep8-naming, pyupgrade, flake8-bugbear, flake8-comprehensions, flake8-simplify, ruff-specific rules (see `pyproject.toml [tool.ruff.lint]`)
- **Type checking**: ty with `all = "warn"`, `possibly-unresolved-reference = "error"`, `unresolved-import = "error"`
- **Imports**: Use `from __future__ import annotations`. Sort with isort (handled by ruff).
- **Type hints**: Use modern Python 3.13 syntax (`list[str]`, `dict[str, int]`, `X | None`, `type` aliases).

## Known Issues

- `plan_cache()` is missing a `return plan` statement (ty reports `invalid-return-type`).
- `order_graph()` has a TODO comment about parallel execution using generations.
