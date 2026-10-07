"""Configuration and stable contracts for the deep-learning pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

@dataclass(frozen=True)
class PathConfig:
    raw_dir: Path = PROJECT_ROOT / "energy-anomaly-detection"
    processed_dir: Path = PROJECT_ROOT / "data" / "processed_DL"
    artifact_dir: Path = PROJECT_ROOT / "model_DL" / "artifacts"
    raw_train_features: str = "train_features.csv"
    raw_test_features: str = "test_features.csv"
    raw_sample_submission: str = "sample_submission.csv"
    processed_train: str = "train_processed_dl.parquet"
    processed_test: str = "test_processed_dl.parquet"
    processed_metadata: str = "processed_dl_metadata.json"

@dataclass(frozen=True)
class DataConfig:
    target_column: str = "anomaly"
    group_column: str = "building_id"
    time_column: str = "timestamp"
    test_id_column: str = "row_id"
    source_row_column: str = "source_row"
    target_length: int = 168
    context_length: int = 252
    sequence_length: int = 420
    chunk_stride: int = 168
    categorical_features: tuple[str, ...] = ("primary_use", "site_id")
    raw_numeric_features: tuple[str, ...] = (
        "meter_reading", "square_feet", "year_built", "floor_count",
        "air_temperature", "cloud_coverage", "dew_temperature",
        "precip_depth_1_hr", "sea_level_pressure", "wind_direction",
        "wind_speed", "is_holiday",
    )
    generated_numeric_features: tuple[str, ...] = (
        "meter_log1p", "meter_delta_1", "meter_delta_24", "meter_run_length",
        "meter_missing_mask", "row_exists_mask", "hour_sin", "hour_cos",
        "weekday_sin", "weekday_cos", "month_sin", "month_cos",
    )

    @property
    def numeric_features(self) -> tuple[str, ...]:
        return self.raw_numeric_features + self.generated_numeric_features

    @property
    def model_features(self) -> tuple[str, ...]:
        return self.numeric_features + self.categorical_features

@dataclass(frozen=True)
class TrainConfig:
    seed: int = 42
    n_group_folds: int = 5
    batch_size: int = 64
    max_epochs: int = 30
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    early_stopping_patience: int = 5
    # Windows spawn would pickle multi-million-row arrays per worker.
    num_workers: int = 0
    positive_class_weight: float = 20.0
    threshold_grid_size: int = 200

@dataclass(frozen=True)
class ModelConfig:
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    tcn_channels: int = 64
    tcn_kernel_size: int = 3
    tcn_dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32)

@dataclass(frozen=True)
class DLConfig:
    paths: PathConfig = field(default_factory=PathConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["paths"] = {k: str(v) for k, v in result["paths"].items()}
        return result

MODEL_NAMES = ("mlp", "tcn", "bigru", "bilstm", "lstm_ae")
CONFIG = DLConfig()
