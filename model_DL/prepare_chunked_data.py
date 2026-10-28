"""Add the original timestamps to the immutable chunked processed datasets.

Run from the repository root with ``python -m model_DL.prepare_chunked_data``.
The raw CSV is only read to recover row-aligned timestamps. All feature values
come from data/processed, which is never written by this module.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .config import PROJECT_ROOT

RAW_PATH = PROJECT_ROOT / "energy-anomaly-detection" / "train_features.csv"
INPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed_DL"
VARIANTS = ("baseline", "zone", "context")
SPLITS = ("train", "test")
MANIFEST = OUTPUT_DIR / "chunked_timestamp_metadata.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_source_rows() -> tuple[dict[str, pd.DataFrame], dict]:
    split_meta = json.loads((INPUT_DIR / "split_metadata.json").read_text(encoding="utf-8"))
    raw = pd.read_csv(
        RAW_PATH,
        usecols=["building_id", "timestamp", "anomaly"],
        dtype={"building_id": "int32", "anomaly": "uint8"},
        low_memory=False,
    )
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], errors="raise")
    if raw["timestamp"].isna().any():
        raise ValueError("Raw CSV has null timestamps")

    day = raw["timestamp"].dt.day
    masks = {"train": day.le(20), "test": day.ge(25)}
    counts = {
        "total_raw_rows": len(raw),
        "n_train": int(masks["train"].sum()),
        "n_gap_purged": int(day.between(21, 24).sum()),
        "n_test": int(masks["test"].sum()),
    }
    for name, actual in counts.items():
        if actual != split_meta[name]:
            raise ValueError(f"Split mismatch for {name}: {actual} != {split_meta[name]}")

    # preprocessingforALL.py selects these masks in original CSV row order.
    rows = {split: raw.loc[mask].reset_index(drop=True) for split, mask in masks.items()}
    return rows, counts


def verify_alignment(table: pa.Table, rows: pd.DataFrame, split: str, variant: str) -> None:
    if table.num_rows != len(rows):
        raise ValueError(f"{split}/{variant}: row count {table.num_rows} != {len(rows)}")
    if "timestamp" in table.column_names:
        raise ValueError(f"{split}/{variant}: source already has timestamp")

    for column in ("building_id", "anomaly"):
        actual = table[column].to_numpy(zero_copy_only=False)
        expected = rows[column].to_numpy()
        if not np.array_equal(actual, expected):
            first = int(np.flatnonzero(actual != expected)[0])
            raise ValueError(f"{split}/{variant}: {column} differs from raw row {first}")

    # A second, independent alignment check catches a wrong row order even if
    # adjacent rows happen to share building_id and anomaly.
    timestamps = rows["timestamp"]
    for unit, values, period, offset in (
        ("hour", timestamps.dt.hour.to_numpy(), 24, 0),
        ("weekday", timestamps.dt.weekday.to_numpy(), 7, 0),
        ("month", timestamps.dt.month.to_numpy(), 12, 1),
    ):
        angle = 2 * np.pi * ((values - offset) % period) / period
        for suffix, expected in (("sin", np.sin(angle)), ("cos", np.cos(angle))):
            column = f"{unit}_{suffix}"
            actual = table[column].to_numpy(zero_copy_only=False)
            if not np.allclose(actual, expected, atol=1e-5, rtol=0, equal_nan=False):
                raise ValueError(f"{split}/{variant}: {column} does not match raw timestamps")


def build() -> None:
    inputs = {
        (split, variant): INPUT_DIR / f"{split}_chunked_{variant}.parquet"
        for split in SPLITS for variant in VARIANTS
    }
    outputs = {
        key: OUTPUT_DIR / f"{key[0]}_chunked_{key[1]}_timestamp.parquet"
        for key in inputs
    }
    missing = [str(path) for path in (RAW_PATH, INPUT_DIR / "split_metadata.json", *inputs.values())
               if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing input files: {missing}")
    existing = [str(path) for path in (*outputs.values(), MANIFEST) if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite existing outputs: {existing}")

    rows, counts = load_source_rows()
    split_meta = json.loads((INPUT_DIR / "split_metadata.json").read_text(encoding="utf-8"))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    for key, source in inputs.items():
        split, variant = key
        table = pq.read_table(source)
        expected_columns = split_meta[f"columns_{variant}"]
        if table.column_names != expected_columns:
            raise ValueError(f"{source.name}: schema/order differs from split_metadata.json")
        verify_alignment(table, rows[split], split, variant)

        timestamps = pa.array(rows[split]["timestamp"].to_numpy(dtype="datetime64[ns]"))
        enriched = table.add_column(1, pa.field("timestamp", pa.timestamp("ns")), timestamps)
        destination = outputs[key]
        temporary = destination.with_name(destination.name + ".tmp")
        try:
            pq.write_table(enriched, temporary, compression="snappy")
            saved = pq.read_table(temporary)
            if not saved.drop_columns(["timestamp"]).equals(table, check_metadata=True):
                raise ValueError(f"{destination.name}: features changed during serialization")
            if not saved["timestamp"].equals(enriched["timestamp"]):
                raise ValueError(f"{destination.name}: timestamp changed during serialization")
            temporary.rename(destination)
        finally:
            temporary.unlink(missing_ok=True)

        record = {
            "split": split,
            "variant": variant,
            "rows": table.num_rows,
            "columns": enriched.num_columns,
            "timestamp_min": str(rows[split]["timestamp"].min()),
            "timestamp_max": str(rows[split]["timestamp"].max()),
            "input": str(source.relative_to(PROJECT_ROOT)),
            "input_sha256": sha256(source),
            "output": str(destination.relative_to(PROJECT_ROOT)),
            "output_sha256": sha256(destination),
        }
        records.append(record)
        print(f"[OK] {destination.name}: {record['rows']:,} rows, {record['columns']} columns", flush=True)

    metadata = {
        "purpose": "Row-aligned timestamps for chunked DL training; no feature values modified",
        "timestamp_source": str(RAW_PATH.relative_to(PROJECT_ROOT)),
        "timestamp_source_sha256": sha256(RAW_PATH),
        "split_rule": {"train_days": "1-20", "purged_days": "21-24", "test_days": "25-end"},
        "counts": counts,
        "datasets": records,
    }
    MANIFEST.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"[OK] {MANIFEST.relative_to(PROJECT_ROOT)}", flush=True)


if __name__ == "__main__":
    build()
