"""Pure pandas helpers. No MCP code here, so they are easy to test."""
import json
from pathlib import Path

import pandas as pd

from config.settings import DATA_DIR, DATASET_EXTENSIONS, DEFAULT_DATASET, UPLOAD_DIR

_AGGS = {"mean", "sum", "min", "max", "median", "count"}
_OPS = {"==", "!=", ">", ">=", "<", "<=", "contains"}


def _records(df: pd.DataFrame) -> list[dict]:
    return json.loads(df.to_json(orient="records"))


def list_datasets() -> list[dict]:
    found = []
    for folder in (DATA_DIR, UPLOAD_DIR):
        for p in sorted(folder.iterdir()):
            if p.suffix.lower() in DATASET_EXTENSIONS:
                found.append({"name": p.name, "location": folder.name,
                              "size_kb": round(p.stat().st_size / 1024, 1)})
    return found


def _read(path: Path) -> pd.DataFrame:
    return pd.read_excel(path) if path.suffix.lower() == ".xlsx" else pd.read_csv(path)


def load_df(name: str | None = None) -> pd.DataFrame:
    safe = Path(name or DEFAULT_DATASET).name  # blocks path traversal
    for folder in (UPLOAD_DIR, DATA_DIR):
        path = folder / safe
        if path.is_file() and path.suffix.lower() in DATASET_EXTENSIONS:
            return _read(path)
    available = list_datasets()
    names = [d["name"] for d in available]
    if not name and available:  # default dataset is missing: fall back to the first one
        first = available[0]
        return _read((DATA_DIR if first["location"] == DATA_DIR.name else UPLOAD_DIR) / first["name"])
    raise FileNotFoundError(f"Dataset '{safe}' not found. Available: {names}")


def _need(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Unknown column(s) {missing}. Columns are: {list(df.columns)}")


def dataset_summary(name: str | None = None) -> dict:
    df = load_df(name)
    return {"rows": len(df), "columns": list(df.columns),
            "dtypes": {c: str(t) for c, t in df.dtypes.items()},
            "missing": {c: int(n) for c, n in df.isna().sum().items() if n},
            "head": _records(df.head(5))}


def describe_columns(name: str | None = None, columns: list[str] | None = None) -> dict:
    df = load_df(name)
    if columns:
        _need(df, columns)
        df = df[columns]
    return json.loads(df.describe(include="all").round(3).to_json(orient="index"))


def value_counts(name: str | None, column: str, top: int = 10) -> list[dict]:
    df = load_df(name)
    _need(df, [column])
    counts = df[column].value_counts(dropna=False).head(max(1, int(top)))
    return [{"value": str(k), "count": int(v)} for k, v in counts.items()]


def group_stats(name: str | None, group_by: str, metric: str, agg: str = "mean") -> list[dict]:
    df = load_df(name)
    cols = [c.strip() for c in group_by.split(",") if c.strip()]
    _need(df, cols + [metric])
    if agg not in _AGGS:
        raise ValueError(f"agg must be one of {sorted(_AGGS)}")
    out = df.groupby(cols)[metric].agg(agg).reset_index().round(3)
    return _records(out)


def correlations(name: str | None = None) -> dict:
    df = load_df(name).select_dtypes("number")
    return json.loads(df.corr().round(3).to_json())


def filter_rows(name: str | None, column: str, op: str, value: str, limit: int = 20) -> dict:
    df = load_df(name)
    _need(df, [column])
    if op not in _OPS:
        raise ValueError(f"op must be one of {sorted(_OPS)}")
    series = df[column]
    if op == "contains":
        mask = series.astype(str).str.contains(value, case=False, na=False)
    else:
        v = value
        if pd.api.types.is_numeric_dtype(series):
            try:
                v = float(value)
            except ValueError:
                raise ValueError(f"Column '{column}' is numeric; '{value}' is not a number.") from None
        mask = {"==": series == v, "!=": series != v, ">": series > v,
                ">=": series >= v, "<": series < v, "<=": series <= v}[op]
    hits = df[mask]
    return {"matches": len(hits), "rows": _records(hits.head(max(1, min(int(limit), 100))))}
