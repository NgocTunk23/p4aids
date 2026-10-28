"""Train the existing five DL models on labeled chunked temporal splits.

The legacy Kaggle pipeline is left intact. This command uses only the
timestamped copies in data/processed_DL and writes isolated run artifacts.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from torch import nn
from torch.utils.data import DataLoader

from .chunked_data import ChunkedSequenceDataset, VARIANTS, feature_columns, load_variant
from .config import CONFIG, MODEL_NAMES, PROJECT_ROOT, DLConfig
from .data_pipeline import FoldTransformer, PointDataset
from .models import build_model

RUN_ROOT = PROJECT_ROOT / "model_DL" / "artifacts" / "chunked"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ranking_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    if len(labels) != len(scores) or not np.isfinite(scores).all():
        raise ValueError("Missing or nonfinite prediction scores")
    if len(np.unique(labels)) != 2:
        raise ValueError("Both classes are required to calculate AP and ROC-AUC")
    return {
        "average_precision": float(average_precision_score(labels, scores)),
        "roc_auc": float(roc_auc_score(labels, scores)),
    }


def oof_threshold(labels: np.ndarray, scores: np.ndarray, grid_size: int) -> tuple[float, float]:
    """Choose a diagnostic F1 threshold on T_in OOF, never on T_out."""
    thresholds = np.unique(np.quantile(scores, np.linspace(0, 1, grid_size)))
    order = np.argsort(scores)
    sorted_scores, sorted_labels = scores[order], labels[order].astype(np.int64)
    cumulative = np.r_[0, np.cumsum(sorted_labels)]
    first = np.searchsorted(sorted_scores, thresholds, side="left")
    predicted = len(scores) - first
    positives = int(cumulative[-1])
    true_positive = positives - cumulative[first]
    f1 = np.divide(2 * true_positive, predicted + positives,
                   out=np.zeros(len(thresholds), dtype=np.float64), where=(predicted + positives) > 0)
    best = int(np.argmax(f1))
    return float(thresholds[best]), float(f1[best])


def summarize_predictions(frame: pd.DataFrame, threshold: float) -> dict[str, float | int]:
    labels = frame["anomaly_true"].to_numpy(np.int64)
    scores = frame["score"].to_numpy(np.float64)
    result: dict[str, float | int] = ranking_metrics(labels, scores)
    result["f1_at_oof_threshold"] = float(f1_score(labels, scores >= threshold, zero_division=0))
    building_ap, building_roc = [], []
    for _, part in frame.groupby("building_id", sort=False):
        y = part["anomaly_true"].to_numpy(np.int64)
        if len(np.unique(y)) != 2:
            continue
        s = part["score"].to_numpy(np.float64)
        building_ap.append(average_precision_score(y, s))
        building_roc.append(roc_auc_score(y, s))
    result["building_average_precision_macro"] = float(np.mean(building_ap)) if building_ap else float("nan")
    result["building_roc_auc_macro"] = float(np.mean(building_roc)) if building_roc else float("nan")
    result["evaluable_buildings"] = len(building_ap)
    return result


def make_dataset(model_name: str, features: np.ndarray, frame: pd.DataFrame,
                 positions: np.ndarray, labels: np.ndarray | None, config: DLConfig):
    if model_name == "mlp":
        return PointDataset(features, positions, labels)
    return ChunkedSequenceDataset(features, frame, positions, labels, config)


def make_loader(dataset, batch_size: int, workers: int, shuffle: bool, device: torch.device) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      pin_memory=device.type == "cuda", persistent_workers=workers > 0)


def unpack_batch(batch, model_name: str, device: torch.device):
    if model_name == "mlp":
        x, y, pos = (value.to(device, non_blocking=True) for value in batch)
        mask = torch.ones_like(y)
    else:
        x, y, mask, pos, _ = (value.to(device, non_blocking=True) for value in batch)
    return x, y, mask, pos


def loss_and_scores(model: nn.Module, model_name: str, x: torch.Tensor, y: torch.Tensor,
                    mask: torch.Tensor, criterion: nn.Module, config: DLConfig):
    raw = model(x)
    if model_name == "lstm_ae":
        point_loss = ((raw - x[:, -config.data.target_length:]) ** 2).mean(dim=-1)
        effective = mask * (y == 0).float()  # Only normal rows train the AE.
        scores = point_loss
    else:
        point_loss = criterion(raw, y)
        effective = mask
        scores = torch.sigmoid(raw)
    numerator = (point_loss * effective).sum()
    denominator = effective.sum()
    return numerator / denominator.clamp_min(1), numerator, denominator, scores


def train_epoch(model: nn.Module, loader: DataLoader, model_name: str, optimizer,
                scaler, device: torch.device, config: DLConfig) -> float:
    model.train()
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(config.train.positive_class_weight, device=device), reduction="none"
    )
    numerator_total = denominator_total = 0.0
    for batch in loader:
        x, y, mask, _ = unpack_batch(batch, model_name, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            loss, numerator, denominator, _ = loss_and_scores(model, model_name, x, y, mask, criterion, config)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), config.train.gradient_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        numerator_total += float(numerator.detach())
        denominator_total += float(denominator.detach())
    if denominator_total == 0:
        raise ValueError(f"No usable training rows for {model_name}")
    return numerator_total / denominator_total


@torch.inference_mode()
def score_loader(model: nn.Module, loader: DataLoader, model_name: str, device: torch.device,
                 config: DLConfig, compute_loss: bool) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(config.train.positive_class_weight, device=device), reduction="none"
    )
    positions, scores = [], []
    numerator_total = denominator_total = 0.0
    for batch in loader:
        x, y, mask, pos = unpack_batch(batch, model_name, device)
        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            if compute_loss:
                _, numerator, denominator, score = loss_and_scores(model, model_name, x, y, mask, criterion, config)
            else:
                raw = model(x)
                score = (((raw - x[:, -config.data.target_length:]) ** 2).mean(dim=-1)
                         if model_name == "lstm_ae" else torch.sigmoid(raw))
        valid = (pos >= 0) & (mask > 0)
        positions.append(pos[valid].cpu().numpy())
        scores.append(score[valid].float().cpu().numpy())
        if compute_loss:
            numerator_total += float(numerator)
            denominator_total += float(denominator)
    if not positions:
        raise ValueError("Empty prediction loader")
    loss = numerator_total / denominator_total if compute_loss and denominator_total else float("nan")
    return np.concatenate(positions), np.concatenate(scores), loss


def score_frame(frame: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({
        "processed_position": np.arange(len(frame), dtype=np.int64),
        "building_id": frame["building_id"].to_numpy(),
        "timestamp": frame["timestamp"].to_numpy(),
        "anomaly_true": frame["anomaly"].to_numpy(np.uint8),
        "score": scores.astype(np.float32),
    })


def train_fold(model_name: str, variant: str, fold: int, train: pd.DataFrame, test: pd.DataFrame,
               train_idx: np.ndarray, valid_idx: np.ndarray, numeric: list[str], categorical: list[str],
               output_dir: Path, device: torch.device, config: DLConfig, workers: int,
               batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    seed_everything(config.train.seed + fold)
    transformer = FoldTransformer(numeric, categorical).fit(train.iloc[train_idx])
    train_x, test_x = transformer.transform(train), transformer.transform(test)
    if not np.isfinite(train_x).all() or not np.isfinite(test_x).all():
        raise ValueError(f"{variant}/{model_name}/fold_{fold}: nonfinite transformed features")
    labels = train["anomaly"].to_numpy(np.float32)
    train_ds = make_dataset(model_name, train_x, train, train_idx, labels, config)
    valid_ds = make_dataset(model_name, train_x, train, valid_idx, labels, config)
    test_ds = make_dataset(model_name, test_x, test, np.arange(len(test)), None, config)
    train_loader = make_loader(train_ds, batch_size, workers, True, device)
    valid_loader = make_loader(valid_ds, batch_size, workers, False, device)
    test_loader = make_loader(test_ds, batch_size, workers, False, device)
    model = build_model(model_name, train_x.shape[1], config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.train.learning_rate,
                                  weight_decay=config.train.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    checkpoint_dir = output_dir / "checkpoints" / model_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_dir / f"fold_{fold}.pt"
    history_path = output_dir / "logs" / f"history_{model_name}_fold_{fold}.csv"
    best_ap, best_epoch, stale, history = -np.inf, 0, 0, []
    for epoch in range(1, config.train.max_epochs + 1):
        train_loss = train_epoch(model, train_loader, model_name, optimizer, scaler, device, config)
        pos, score, valid_loss = score_loader(model, valid_loader, model_name, device, config, True)
        metrics = ranking_metrics(labels[pos], score)
        history.append({"epoch": epoch, "train_loss": train_loss, "valid_loss": valid_loss, **metrics})
        pd.DataFrame(history).to_csv(history_path, index=False)
        print(f"[{variant}/{model_name} fold={fold} epoch={epoch}] "
              f"loss={train_loss:.6f} val_loss={valid_loss:.6f} "
              f"AP={metrics['average_precision']:.6f} ROC={metrics['roc_auc']:.6f}", flush=True)
        if metrics["average_precision"] > best_ap:
            best_ap, best_epoch, stale = metrics["average_precision"], epoch, 0
            torch.save({
                "data_source": "chunked", "variant": variant, "model_name": model_name,
                "fold": fold, "best_epoch": epoch, "best_average_precision": best_ap,
                "input_size": train_x.shape[1], "numeric_features": numeric,
                "categorical_features": categorical, "model_state": model.state_dict(),
                "transformer": transformer.state_dict(), "config": config.to_dict(),
                "valid_buildings": sorted(train.iloc[valid_idx]["building_id"].unique().astype(int).tolist()),
            }, checkpoint)
        else:
            stale += 1
            if stale >= config.train.early_stopping_patience:
                break
    if best_epoch == 0:
        raise ValueError(f"No usable AP checkpoint for {variant}/{model_name}/fold_{fold}")
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(saved["model_state"])
    valid_pos, valid_score, _ = score_loader(model, valid_loader, model_name, device, config, False)
    test_pos, test_score, _ = score_loader(model, test_loader, model_name, device, config, False)
    if not np.array_equal(np.sort(valid_pos), np.sort(valid_idx)):
        raise ValueError(f"{variant}/{model_name}/fold_{fold}: validation coverage mismatch")
    if not np.array_equal(np.sort(test_pos), np.arange(len(test))):
        raise ValueError(f"{variant}/{model_name}/fold_{fold}: holdout coverage mismatch")
    report = {"fold": fold, "best_epoch": best_epoch, "best_average_precision": best_ap,
              "validation": ranking_metrics(labels[valid_pos], valid_score),
              "checkpoint": str(checkpoint.relative_to(PROJECT_ROOT)), "batch_size": batch_size}
    del train_x, test_x, train_ds, valid_ds, test_ds, train_loader, valid_loader, test_loader, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return valid_pos, valid_score, test_pos, test_score, report


def pick_smoke_buildings(train: pd.DataFrame, count: int = 8) -> list[int]:
    positives = train.groupby("building_id")["anomaly"].sum()
    buildings = sorted(int(building) for building in positives.index[positives > 0])[:count]
    if len(buildings) < count:
        raise ValueError(f"Need {count} positive-bearing buildings for smoke test")
    return buildings


def train_run(args: argparse.Namespace) -> Path:
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    folds = 2 if args.smoke else args.folds
    epochs = 1 if args.smoke else args.epochs
    if folds < 2 or epochs < 1 or args.workers < 0 or (args.batch_size is not None and args.batch_size < 1):
        raise ValueError("folds>=2, epochs>=1, workers>=0 and batch-size>=1 are required")
    run_id = args.run_id or (("smoke_" if args.smoke else "full_") +
                             datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_id):
        raise ValueError("run-id may only contain letters, digits, dots, underscores and hyphens")
    root = RUN_ROOT / run_id
    root.mkdir(parents=True, exist_ok=False)
    config = replace(CONFIG, train=replace(CONFIG.train, n_group_folds=folds, max_epochs=epochs,
                                            early_stopping_patience=1 if args.smoke else args.patience))
    run_info = {"status": "running", "run_id": run_id, "smoke": bool(args.smoke),
                "device": str(device), "models": args.models, "variants": args.variants,
                "folds": folds, "max_epochs": epochs, "seed": config.train.seed,
                "feature_policy": "all numeric except IDs, labels, timestamps and gte_*; site_id categorical",
                "validation": "GroupKFold by building_id within T_in",
                "holdout": "T_out; no fitting, early stopping or threshold selection on T_out",
                "metric_note": "average_precision is sklearn AP, not trapezoidal PR area",
                "data_note": "chunked features were precomputed on full T_in before GroupKFold"}
    run_config = root / "run_config.json"
    run_config.write_text(json.dumps(run_info, indent=2) + "\n", encoding="utf-8")
    summaries: list[dict] = []
    smoke_buildings: list[int] | None = None
    try:
        for variant in args.variants:
            train, test, source_manifest = load_variant(variant)
            if args.smoke:
                if smoke_buildings is None:
                    smoke_buildings = pick_smoke_buildings(train)
                train = train.loc[train["building_id"].isin(smoke_buildings)].reset_index(drop=True)
                test = test.loc[test["building_id"].isin(smoke_buildings)].reset_index(drop=True)
            numeric, categorical = feature_columns(train)
            test_numeric, test_categorical = feature_columns(test)
            if (numeric, categorical) != (test_numeric, test_categorical):
                raise ValueError(f"{variant}: train/test feature schemas differ")
            variant_dir = root / variant
            (variant_dir / "logs").mkdir(parents=True)
            (variant_dir / "predictions").mkdir()
            (variant_dir / "checkpoints").mkdir()
            records = [item for item in source_manifest["datasets"] if item["variant"] == variant]
            variant_info = {"variant": variant, "train_rows": len(train), "holdout_rows": len(test),
                            "numeric_features": numeric, "categorical_features": categorical,
                            "source_datasets": records, "smoke_buildings": smoke_buildings}
            (variant_dir / "variant_config.json").write_text(json.dumps(variant_info, indent=2) + "\n",
                                                             encoding="utf-8")
            labels = train["anomaly"].to_numpy(np.uint8)
            groups = train["building_id"].to_numpy()
            cv = list(GroupKFold(n_splits=folds).split(train, groups=groups))
            for fold, (train_idx, valid_idx) in enumerate(cv):
                if len(np.unique(labels[train_idx])) != 2 or len(np.unique(labels[valid_idx])) != 2:
                    raise ValueError(f"{variant}/fold_{fold}: both classes required in train and validation")
            for model_name in args.models:
                batch_size = args.batch_size or (1024 if model_name == "mlp" else 16)
                model_config = replace(config, train=replace(config.train, batch_size=batch_size))
                oof = np.full(len(train), np.nan, np.float32)
                holdout_sum = np.zeros(len(test), np.float64)
                holdout_count = np.zeros(len(test), np.int16)
                fold_reports = []
                for fold, (train_idx, valid_idx) in enumerate(cv):
                    vp, vs, tp, ts, report = train_fold(
                        model_name, variant, fold, train, test, train_idx, valid_idx,
                        numeric, categorical, variant_dir, device, model_config, args.workers, batch_size
                    )
                    if not np.isnan(oof[vp]).all():
                        raise ValueError(f"{variant}/{model_name}: duplicate OOF predictions")
                    oof[vp] = vs
                    holdout_sum[tp] += ts
                    holdout_count[tp] += 1
                    fold_reports.append(report)
                if not np.isfinite(oof).all() or not np.all(holdout_count == folds):
                    raise ValueError(f"{variant}/{model_name}: incomplete prediction coverage")
                holdout_scores = (holdout_sum / holdout_count).astype(np.float32)
                oof_frame = score_frame(train, oof)
                holdout_frame = score_frame(test, holdout_scores)
                threshold, oof_f1 = oof_threshold(labels, oof, config.train.threshold_grid_size)
                oof_metrics = summarize_predictions(oof_frame, threshold)
                holdout_metrics = summarize_predictions(holdout_frame, threshold)
                oof_frame.to_parquet(variant_dir / "predictions" / f"oof_{model_name}.parquet", index=False)
                holdout_frame.to_parquet(variant_dir / "predictions" / f"holdout_{model_name}.parquet", index=False)
                report = {"variant": variant, "model": model_name, "oof_threshold": threshold,
                          "oof_f1_at_selected_threshold": oof_f1, "oof": oof_metrics,
                          "holdout": holdout_metrics, "folds": fold_reports}
                (variant_dir / "logs" / f"metrics_{model_name}.json").write_text(
                    json.dumps(report, indent=2) + "\n", encoding="utf-8"
                )
                summaries.append({"variant": variant, "model": model_name,
                                  "oof_ap": oof_metrics["average_precision"],
                                  "oof_roc_auc": oof_metrics["roc_auc"],
                                  "holdout_ap": holdout_metrics["average_precision"],
                                  "holdout_roc_auc": holdout_metrics["roc_auc"],
                                  "holdout_f1_at_oof_threshold": holdout_metrics["f1_at_oof_threshold"],
                                  "oof_threshold": threshold})
                pd.DataFrame(summaries).to_csv(root / "summary.csv", index=False)
                print(f"[DONE] {variant}/{model_name}: OOF AP={oof_metrics['average_precision']:.6f}, "
                      f"T_out AP={holdout_metrics['average_precision']:.6f}", flush=True)
        run_info["status"] = "complete"
    except Exception as error:
        run_info["status"] = "failed"
        run_info["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        run_config.write_text(json.dumps(run_info, indent=2) + "\n", encoding="utf-8")
    print(f"[COMPLETE] {root}", flush=True)
    return root


def evaluate_run(run_id: str) -> pd.DataFrame:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_id):
        raise ValueError("Invalid run-id")
    root = RUN_ROOT / run_id
    info = json.loads((root / "run_config.json").read_text(encoding="utf-8"))
    if info["status"] != "complete":
        raise ValueError(f"Run {run_id} is not complete: {info['status']}")
    rows = []
    for variant in info["variants"]:
        for model_name in info["models"]:
            pred_dir = root / variant / "predictions"
            oof = pd.read_parquet(pred_dir / f"oof_{model_name}.parquet")
            holdout = pd.read_parquet(pred_dir / f"holdout_{model_name}.parquet")
            threshold, _ = oof_threshold(oof["anomaly_true"].to_numpy(),
                                         oof["score"].to_numpy(), CONFIG.train.threshold_grid_size)
            a, b = summarize_predictions(oof, threshold), summarize_predictions(holdout, threshold)
            rows.append({"variant": variant, "model": model_name, "oof_ap": a["average_precision"],
                         "oof_roc_auc": a["roc_auc"], "holdout_ap": b["average_precision"],
                         "holdout_roc_auc": b["roc_auc"],
                         "holdout_f1_at_oof_threshold": b["f1_at_oof_threshold"]})
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P4AIDS DL on timestamped chunked datasets")
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train", help="Train and evaluate on T_in/T_out")
    train.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    train.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    train.add_argument("--folds", type=int, default=CONFIG.train.n_group_folds)
    train.add_argument("--epochs", type=int, default=CONFIG.train.max_epochs)
    train.add_argument("--patience", type=int, default=CONFIG.train.early_stopping_patience)
    train.add_argument("--batch-size", type=int, default=None,
                       help="Default: 1024 for MLP, 16 for sequence models")
    train.add_argument("--workers", type=int, default=0)
    train.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    train.add_argument("--run-id", default=None)
    train.add_argument("--smoke", action="store_true", help="8 buildings, 2 folds, 1 epoch")
    evaluate = sub.add_parser("evaluate", help="Recalculate metrics from saved predictions")
    evaluate.add_argument("--run-id", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "train":
        train_run(args)
    else:
        print(evaluate_run(args.run_id).to_string(index=False))


if __name__ == "__main__":
    main()
