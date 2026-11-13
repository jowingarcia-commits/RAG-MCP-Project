"""MCP tools for tabular data. Run alone with: python pandas_mcp/pandas_server.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastmcp import FastMCP

from config.settings import setup_logging
from pandas_mcp import analytics

setup_logging()
mcp = FastMCP("pandas-analytics")


def list_datasets() -> list[dict]:
    """List the CSV and Excel files available for analysis."""
    return analytics.list_datasets()


def dataset_summary(dataset: str | None = None) -> dict:
    """Row count, columns, dtypes, missing values and first rows. Omit dataset for the default."""
    return analytics.dataset_summary(dataset)


def describe_columns(dataset: str | None = None, columns: list[str] | None = None) -> dict:
    """Descriptive statistics for all columns, or only the named ones."""
    return analytics.describe_columns(dataset, columns)


def value_counts(column: str, dataset: str | None = None, top: int = 10) -> list[dict]:
    """Most frequent values in a column."""
    return analytics.value_counts(dataset, column, top)


def group_stats(group_by: str, metric: str, agg: str = "mean", dataset: str | None = None) -> list[dict]:
    """Aggregate a metric per group. group_by may list several columns separated by commas.
    agg is one of mean, sum, min, max, median, count."""
    return analytics.group_stats(dataset, group_by, metric, agg)


def correlations(dataset: str | None = None) -> dict:
    """Correlation matrix of the numeric columns."""
    return analytics.correlations(dataset)


def filter_rows(column: str, op: str, value: str, dataset: str | None = None, limit: int = 20) -> dict:
    """Return rows where column <op> value. op is ==, !=, >, >=, <, <= or contains."""
    return analytics.filter_rows(dataset, column, op, value, limit)


TOOLS = [list_datasets, dataset_summary, describe_columns, value_counts, group_stats, correlations, filter_rows]
for _tool in TOOLS:
    mcp.tool(_tool)


if __name__ == "__main__":
    mcp.run()
