"""Read the timestamped chunked datasets and build gap-safe DL windows."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import PROJECT_ROOT, DLConfig
from .prepare_chunked_data import sha256

DATA_DIR = PROJECT_ROOT / "data" / "processed_DL"
MANIFEST = DATA_DIR / "chunked_timestamp_metadata.json"
VARIANTS = ("baseline", "zone", "context")
ONE_HOUR_NS = pd.Timedelta(hours=1).value


def load_variant(variant: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}; expected one of {VARIANTS}")
    if not MANIFEST.is_file():
        raise FileNotFoundError(f"Missing {MANIFEST}; run python -m model_DL.prepare_chunked_data")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    records = {(item["split"], item["variant"]): item for item in manifest["datasets"]}
    frames = {}
    for split in ("train", "test"):
        record = records[(split, variant)]
        # The manifest may have been created on Windows and trained on Linux.
        path = PROJECT_ROOT / Path(record["output"].replace("\\", "/"))
        if sha256(path) != record["output_sha256"]:
            raise ValueError(f"Dataset hash differs from manifest: {path}")
        frame = pd.read_parquet(path)
        validate_frame(frame, split, variant, record)
        frames[split] = frame
    return frames["train"], frames["test"], manifest


def validate_frame(frame: pd.DataFrame, split: str, variant: str, record: dict) -> None:
    required = {"building_id", "timestamp", "anomaly", "site_id"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{split}/{variant}: missing {sorted(missing)}")
    if len(frame) != record["rows"] or len(frame.columns) != record["columns"]:
        raise ValueError(f"{split}/{variant}: shape differs from manifest")
    if frame[["building_id", "timestamp", "anomaly"]].isna().any().any():
        raise ValueError(f"{split}/{variant}: null building, timestamp, or label")
    if not pd.api.types.is_datetime64_any_dtype(frame["timestamp"]):
        raise TypeError(f"{split}/{variant}: timestamp is not datetime64")
    if frame.duplicated(["building_id", "timestamp"]).any():
        raise ValueError(f"{split}/{variant}: duplicate building/timestamp")
    days = frame["timestamp"].dt.day
    valid_days = days.le(20) if split == "train" else days.ge(25)
    if not valid_days.all():
        raise ValueError(f"{split}/{variant}: row falls outside its time split")
    if not frame["anomaly"].isin([0, 1]).all():
        raise ValueError(f"{split}/{variant}: anomaly must be binary")
    for name in frame.select_dtypes(include=[np.number]).columns:
        if not np.isfinite(frame[name].to_numpy(dtype=np.float64, copy=False)).all():
            raise ValueError(f"{split}/{variant}: nonfinite numeric value in {name}")


def feature_columns(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Preserve the original DL policy of excluding uncertain gte_* features."""
    excluded = {"building_id", "timestamp", "anomaly", "site_id"}
    categorical = ["site_id"]
    numeric = [name for name in frame.columns
               if name not in excluded and not name.startswith("gte_")]
    if not numeric or any(not pd.api.types.is_numeric_dtype(frame[name]) for name in numeric):
        raise ValueError("The chunked model feature set must be nonempty and numeric")
    return numeric, categorical


class ChunkedSequenceDataset(Dataset):
    """Every selected row is a target exactly once; context stays within hourly runs."""

    def __init__(self, features: np.ndarray, frame: pd.DataFrame, positions: np.ndarray,
                 labels: np.ndarray | None, config: DLConfig):
        self.features, self.labels, self.config = features, labels, config
        if config.data.chunk_stride != config.data.target_length:
            raise ValueError("chunk_stride must equal target_length for one prediction per row")
        if config.data.context_length + config.data.target_length != config.data.sequence_length:
            raise ValueError("sequence_length must equal context_length + target_length")
        selected = frame.iloc[np.asarray(positions, dtype=np.int64)][["building_id", "timestamp"]].copy()
        selected["_position"] = np.asarray(positions, dtype=np.int64)
        sizes = frame.groupby("building_id", sort=False).size()
        self.windows: list[tuple[np.ndarray, np.ndarray]] = []
        for building, group in selected.groupby("building_id", sort=False):
            if len(group) != sizes.loc[building]:
                raise ValueError(f"Selection cuts through building {building}")
            group = group.sort_values("timestamp")
            ordered_pos = group["_position"].to_numpy(np.int64)
            times = group["timestamp"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
            starts = np.r_[0, np.flatnonzero(np.diff(times) != ONE_HOUR_NS) + 1]
            ends = np.r_[starts[1:], len(ordered_pos)]
            for lo, hi in zip(starts, ends):
                segment = ordered_pos[lo:hi]
                for start in range(0, len(segment), config.data.target_length):
                    target = segment[start:start + config.data.target_length]
                    context = segment[max(0, start - config.data.context_length):start]
                    self.windows.append((context, target))

        targets = np.concatenate([target for _, target in self.windows]) if self.windows else np.array([], dtype=np.int64)
        expected = np.asarray(positions, dtype=np.int64)
        if len(targets) != len(expected) or not np.array_equal(np.sort(targets), np.sort(expected)):
            raise ValueError("Sequence windows do not cover every selected row exactly once")

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int):
        context, target = self.windows[index]
        target_length, sequence_length = self.config.data.target_length, self.config.data.sequence_length
        positions = np.concatenate([context, target])
        x = np.zeros((sequence_length, self.features.shape[1]), np.float32)
        seq_mask = np.zeros(sequence_length, np.float32)
        # Keep the last observed target at the end of the sequence. Short
        # targets use the final slots of the model's target_length output.
        x[-len(positions):] = self.features[positions]
        seq_mask[-len(positions):] = 1
        y = np.zeros(target_length, np.float32)
        mask = np.zeros(target_length, np.float32)
        out_pos = np.full(target_length, -1, np.int64)
        if self.labels is not None:
            y[-len(target):] = self.labels[target]
        mask[-len(target):] = 1
        out_pos[-len(target):] = target
        return tuple(torch.from_numpy(array) for array in (x, y, mask, out_pos, seq_mask))
