"""Preprocessing, fold-safe transforms, and PyTorch datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold
from torch.utils.data import Dataset

from .config import CONFIG, DLConfig

MASK_COLUMNS = ("row_exists_mask", "meter_missing_mask", "label_mask")

def _required_columns(is_train: bool, config: DLConfig) -> list[str]:
    d = config.data
    cols = [d.group_column, d.time_column, *d.raw_numeric_features, *d.categorical_features]
    cols += [d.target_column] if is_train else [d.test_id_column]
    return list(dict.fromkeys(cols))

def validate_raw_schema(path: Path, is_train: bool, config: DLConfig = CONFIG) -> None:
    header = pd.read_csv(path, nrows=0).columns
    missing = sorted(set(_required_columns(is_train, config)) - set(header))
    if missing:
        raise ValueError(f"{path.name} is missing columns: {missing}")

def _run_length(values: pd.Series, missing: pd.Series, tolerance: float = 1e-6) -> np.ndarray:
    """Causal length of a near-constant run; missing values break runs."""
    x, m = values.to_numpy(np.float64), missing.to_numpy(bool)
    out = np.ones(len(x), dtype=np.float32)
    for i in range(1, len(x)):
        if not m[i] and not m[i - 1] and np.isfinite(x[i]) and np.isfinite(x[i - 1]):
            if abs(x[i] - x[i - 1]) <= tolerance * max(1.0, abs(x[i - 1])):
                out[i] = out[i - 1] + 1.0
    out[m] = 0.0
    return out

def _reindex_building(group: pd.DataFrame, is_train: bool, config: DLConfig) -> pd.DataFrame:
    d = config.data
    group = group.sort_values(d.time_column).drop_duplicates(d.time_column, keep="first")
    timeline = pd.date_range(group[d.time_column].iloc[0], group[d.time_column].iloc[-1], freq="h")
    group = group.set_index(d.time_column).reindex(timeline).rename_axis(d.time_column).reset_index()
    group[d.group_column] = group[d.group_column].ffill().bfill()
    group["row_exists_mask"] = group[d.source_row_column].notna().astype("uint8")
    group["meter_missing_mask"] = group["meter_reading"].isna().astype("uint8")
    if is_train:
        group["label_mask"] = (group["row_exists_mask"].eq(1) & group[d.target_column].notna()).astype("uint8")
        group[d.target_column] = group[d.target_column].fillna(0).astype("uint8")
    else:
        group["label_mask"] = np.uint8(0)

    static = ["site_id", "primary_use", "square_feet", "year_built", "floor_count"]
    weather = [c for c in d.raw_numeric_features if c not in {"meter_reading", *static}]
    group[static] = group[static].ffill().bfill()
    group[weather] = group[weather].ffill()  # forward-only: causal
    meter = group["meter_reading"].astype("float32")
    safe = meter.clip(lower=0)
    group["meter_log1p"] = np.log1p(safe)
    group["meter_delta_1"] = safe.diff(1)
    group["meter_delta_24"] = safe.diff(24)
    group["meter_run_length"] = _run_length(meter, group["meter_missing_mask"])
    ts = group[d.time_column]
    group["hour_sin"] = np.sin(2 * np.pi * ts.dt.hour / 24).astype("float32")
    group["hour_cos"] = np.cos(2 * np.pi * ts.dt.hour / 24).astype("float32")
    group["weekday_sin"] = np.sin(2 * np.pi * ts.dt.weekday / 7).astype("float32")
    group["weekday_cos"] = np.cos(2 * np.pi * ts.dt.weekday / 7).astype("float32")
    group["month_sin"] = np.sin(2 * np.pi * (ts.dt.month - 1) / 12).astype("float32")
    group["month_cos"] = np.cos(2 * np.pi * (ts.dt.month - 1) / 12).astype("float32")
    return group

def preprocess_one(path: Path, output: Path, is_train: bool, config: DLConfig = CONFIG) -> dict:
    validate_raw_schema(path, is_train, config)
    d = config.data
    frame = pd.read_csv(path, usecols=_required_columns(is_train, config), low_memory=False)
    frame[d.time_column] = pd.to_datetime(frame[d.time_column], errors="raise")
    frame[d.source_row_column] = np.arange(len(frame), dtype=np.int64)
    if not is_train:
        frame[d.test_id_column] = frame[d.test_id_column].astype("Int64")
    parts = [_reindex_building(g, is_train, config) for _, g in frame.groupby(d.group_column, sort=True)]
    processed = pd.concat(parts, ignore_index=True)
    ordered = [d.group_column, d.time_column, d.source_row_column]
    ordered += [d.test_id_column] if not is_train else [d.target_column]
    ordered += [*MASK_COLUMNS, *d.model_features]
    processed = processed[list(dict.fromkeys(ordered))]
    for col in d.numeric_features:
        processed[col] = pd.to_numeric(processed[col], errors="coerce").astype("float32")
    processed[d.group_column] = processed[d.group_column].astype("int32")
    processed[d.source_row_column] = processed[d.source_row_column].astype("Int64")
    output.parent.mkdir(parents=True, exist_ok=True)
    processed.to_parquet(output, index=False, compression="zstd")
    return {
        "raw_rows": int(len(frame)), "processed_rows": int(len(processed)),
        "inserted_rows": int((processed["row_exists_mask"] == 0).sum()),
        "buildings": int(processed[d.group_column].nunique()),
        "start": str(processed[d.time_column].min()), "end": str(processed[d.time_column].max()),
    }

def preprocess_all(config: DLConfig = CONFIG, force: bool = False) -> dict:
    p = config.paths
    p.processed_dir.mkdir(parents=True, exist_ok=True)
    train_out, test_out = p.processed_dir / p.processed_train, p.processed_dir / p.processed_test
    if not force and train_out.exists() and test_out.exists():
        raise FileExistsError("Processed DL files exist. Pass --force to replace them.")
    train_audit = preprocess_one(p.raw_dir / p.raw_train_features, train_out, True, config)
    test_audit = preprocess_one(p.raw_dir / p.raw_test_features, test_out, False, config)
    metadata = {
        "version": 1, "train": train_audit, "test": test_audit,
        "model_features": list(config.data.model_features),
        "categorical_features": list(config.data.categorical_features),
        "causal_feature_engineering": True,
        "notes": [
            "gte_* and raw string interactions are excluded.",
            "Inserted rows never enter metrics or submission.",
            "Transforms are fitted independently inside each fold.",
        ],
    }
    (p.processed_dir / p.processed_metadata).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata

def load_processed(config: DLConfig = CONFIG) -> tuple[pd.DataFrame, pd.DataFrame]:
    p = config.paths
    train_path, test_path = p.processed_dir / p.processed_train, p.processed_dir / p.processed_test
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("Run `python -m model_DL.pipeline preprocess` first.")
    return pd.read_parquet(train_path), pd.read_parquet(test_path)

class FoldTransformer:
    """Fold-fitted median/scaler and unknown-safe ordinal categoricals."""
    def __init__(self, numeric: Sequence[str], categorical: Sequence[str]):
        self.numeric, self.categorical = list(numeric), list(categorical)
        self.medians: dict[str, float] = {}
        self.means: dict[str, float] = {}
        self.stds: dict[str, float] = {}
        self.categories: dict[str, list[str]] = {}

    def fit(self, frame: pd.DataFrame) -> "FoldTransformer":
        for col in self.numeric:
            x = pd.to_numeric(frame[col], errors="coerce").to_numpy(np.float64)
            finite = x[np.isfinite(x)]
            median = float(np.median(finite)) if len(finite) else 0.0
            x = np.nan_to_num(x, nan=median, posinf=median, neginf=median)
            self.medians[col] = median
            self.means[col] = float(x.mean())
            self.stds[col] = max(float(x.std()), 1e-6)
        for col in self.categorical:
            self.categories[col] = sorted(frame[col].fillna("__MISSING__").astype(str).unique().tolist())
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        columns: list[np.ndarray] = []
        for col in self.numeric:
            x = pd.to_numeric(frame[col], errors="coerce").to_numpy(np.float32)
            x = np.nan_to_num(x, nan=self.medians[col], posinf=self.medians[col], neginf=self.medians[col])
            columns.append(((x - self.means[col]) / self.stds[col]).astype(np.float32))
        for col in self.categorical:
            mapping = {v: i + 1 for i, v in enumerate(self.categories[col])}
            x = frame[col].fillna("__MISSING__").astype(str).map(mapping).fillna(0).to_numpy(np.float32)
            columns.append(x / max(1, len(mapping)))
        return np.column_stack(columns).astype(np.float32, copy=False)

    def state_dict(self) -> dict:
        return {k: getattr(self, k) for k in ("numeric", "categorical", "medians", "means", "stds", "categories")}

    @classmethod
    def from_state_dict(cls, state: dict) -> "FoldTransformer":
        obj = cls(state["numeric"], state["categorical"])
        for name in ("medians", "means", "stds", "categories"):
            setattr(obj, name, state[name])
        return obj

def group_folds(frame: pd.DataFrame, n_splits: int, config: DLConfig = CONFIG) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    groups = frame[config.data.group_column].to_numpy()
    yield from GroupKFold(n_splits=n_splits).split(frame, groups=groups)

class PointDataset(Dataset):
    def __init__(self, features: np.ndarray, positions: np.ndarray, labels: np.ndarray | None = None):
        self.features, self.positions, self.labels = features, positions.astype(np.int64), labels
    def __len__(self) -> int:
        return len(self.positions)
    def __getitem__(self, index: int):
        pos = self.positions[index]
        y = -1.0 if self.labels is None else float(self.labels[pos])
        return torch.from_numpy(self.features[pos]), torch.tensor(y), torch.tensor(pos)

class SequenceDataset(Dataset):
    """Left-padded context+target windows; every target row occurs once."""
    def __init__(self, features: np.ndarray, frame: pd.DataFrame, allowed_positions: np.ndarray,
                 labels: np.ndarray | None = None, config: DLConfig = CONFIG):
        self.features, self.frame, self.labels, self.config = features, frame, labels, config
        self.label_mask = frame["label_mask"].to_numpy(np.float32)
        self.row_exists_mask = frame["row_exists_mask"].to_numpy(np.float32)
        d = config.data
        allowed = np.zeros(len(frame), dtype=bool)
        allowed[allowed_positions] = True
        self.windows: list[tuple[np.ndarray, np.ndarray]] = []
        for _, group in frame.groupby(d.group_column, sort=False):
            pos = group.index.to_numpy(np.int64)
            valid = pos[allowed[pos]]
            if not len(valid):
                continue
            if len(valid) != len(pos):
                raise ValueError("Sequence split cuts through a building; use GroupKFold.")
            for start in range(0, len(pos), d.chunk_stride):
                target = pos[start:start + d.target_length]
                context = pos[max(0, start - d.context_length):start]
                self.windows.append((context, target))
    def __len__(self) -> int:
        return len(self.windows)
    def __getitem__(self, index: int):
        d = self.config.data
        context, target = self.windows[index]
        positions = np.concatenate([context, target])
        x = np.zeros((d.sequence_length, self.features.shape[1]), np.float32)
        seq_mask = np.zeros(d.sequence_length, np.float32)
        x[-len(positions):], seq_mask[-len(positions):] = self.features[positions], 1.0
        y, loss_mask = np.zeros(d.target_length, np.float32), np.zeros(d.target_length, np.float32)
        out_pos = np.full(d.target_length, -1, np.int64)
        if self.labels is not None:
            y[:len(target)] = self.labels[target]
            loss_mask[:len(target)] = self.label_mask[target]
        else:
            loss_mask[:len(target)] = self.row_exists_mask[target]
        out_pos[:len(target)] = target
        return tuple(torch.from_numpy(v) for v in (x, y, loss_mask, out_pos, seq_mask))

def prediction_rows(frame: pd.DataFrame) -> np.ndarray:
    return np.flatnonzero(frame["row_exists_mask"].to_numpy() == 1)
