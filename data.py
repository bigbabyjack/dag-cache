"""Data models and dataset specifications for the DAG cache system."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl


@dataclass(frozen=True)
class DatasetQuery:
    """Specification for querying a subset of a dataset."""

    dataset: str
    filters: pl.Expr | None = None
    columns: list[str] | None = None


@dataclass(frozen=True)
class DatasetSpec:
    """Specification for loading a dataset from storage."""

    name: str
    loader: Callable = field(default=lambda path: pl.read_parquet(path))
    path: Path = Path("./data/name.parquet")


@dataclass(frozen=True)
class Dataset:
    """A loaded dataset with its specification and data."""

    spec: DatasetSpec
    data: pl.DataFrame
    query: DatasetQuery | None = None


def memory_usage(dataset: Dataset) -> int:
    """Calculate actual memory usage of a loaded dataset."""
    return int(dataset.data.estimated_size())


def estimate_memory_usage(dataset_spec: DatasetSpec) -> int:
    """Estimate memory usage based on file size."""
    estimate = dataset_spec.path.stat().st_size
    return estimate


def generate_dataset(
    name: str, n_rows: int, n_cols: int, path: Path = Path("./data/")
) -> DatasetSpec:
    """Generate a mock dataset and save it to disk."""
    df = pl.DataFrame({f"col_{i}": np.arange(0, n_rows) for i in range(n_cols)})
    dataset_spec = DatasetSpec(name=name, path=path / f"{name}.parquet")
    df.write_parquet(dataset_spec.path, mkdir=True)
    return dataset_spec


def make_mock_data() -> list[DatasetSpec]:
    """Create mock datasets for testing and demos."""
    if not Path("./data/").exists():
        Path("./data/").mkdir()
    return [
        generate_dataset(name="a", n_rows=100_000, n_cols=50),
        generate_dataset(name="b", n_rows=200_000, n_cols=100),
        generate_dataset(name="c", n_rows=50_000, n_cols=30),
    ]
