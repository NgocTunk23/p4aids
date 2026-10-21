"""Train, evaluate, predict, and export Kaggle submissions.

Examples (run from repository root)::

    python -m model_DL.pipeline preprocess
    python -m model_DL.pipeline benchmark --models mlp tcn bigru
    python -m model_DL.pipeline predict-test --models mlp tcn bigru
    python -m model_DL.pipeline ensemble --models mlp tcn bigru
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from dataclasses import replace
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from .config import CONFIG, MODEL_NAMES, DLConfig
from .data_pipeline import (
    FoldTransformer, PointDataset, SequenceDataset, group_folds,
    load_processed, prediction_rows, preprocess_all,
)
from .models import build_model

def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def artifact_dirs(config: DLConfig) -> dict[str, Path]:
    result = {name: config.paths.artifact_dir / name for name in ("checkpoints", "logs", "predictions")}
    for path in result.values(): path.mkdir(parents=True, exist_ok=True)
    return result

def make_loader(dataset, config: DLConfig, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=config.train.batch_size, shuffle=shuffle,
                      num_workers=config.train.num_workers, pin_memory=torch.cuda.is_available(),
                      persistent_workers=config.train.num_workers > 0)

def safe_metrics(y_true: np.ndarray, score: np.ndarray, grid_size: int = 200) -> dict[str, float]:
    finite = np.isfinite(score)
    y, s = y_true[finite].astype(int), score[finite]
    if len(y) == 0 or len(np.unique(y)) < 2:
        return {"roc_auc": float("nan"), "pr_auc": float("nan"), "f1": float("nan"), "threshold": float("nan")}
    roc, pr = roc_auc_score(y, s), average_precision_score(y, s)
    quantiles = np.unique(np.quantile(s, np.linspace(0, 1, grid_size)))
    f1s = np.asarray([f1_score(y, s >= t, zero_division=0) for t in quantiles])
    best = int(f1s.argmax())
    return {"roc_auc": float(roc), "pr_auc": float(pr), "f1": float(f1s[best]), "threshold": float(quantiles[best])}

def metrics_with_buildings(frame: pd.DataFrame, positions: np.ndarray, labels: np.ndarray,
                           score: np.ndarray, config: DLConfig) -> dict[str, float]:
    """Point-wise metrics plus macro averages over evaluable buildings."""
    result = safe_metrics(labels[positions], score, config.train.threshold_grid_size)
    groups = frame.iloc[positions][config.data.group_column].to_numpy()
    building_auc, building_pr = [], []
    for building in np.unique(groups):
        take = groups == building
        y, s = labels[positions[take]].astype(int), score[take]
        if len(np.unique(y)) < 2:
            continue
        building_auc.append(roc_auc_score(y, s))
        building_pr.append(average_precision_score(y, s))
    result["building_roc_auc_macro"] = float(np.mean(building_auc)) if building_auc else float("nan")
    result["building_pr_auc_macro"] = float(np.mean(building_pr)) if building_pr else float("nan")
    result["evaluable_buildings"] = int(len(building_auc))
    return result

def _forward_scores(model: nn.Module, model_name: str, x: torch.Tensor, target_length: int):
    output = model(x)
    if model_name == "lstm_ae":
        return ((output - x[:, -target_length:]) ** 2).mean(dim=-1)
    return output

def _epoch(model: nn.Module, loader: DataLoader, model_name: str, device: torch.device,
           config: DLConfig, optimizer=None, scaler=None) -> tuple[np.ndarray, np.ndarray]:
    training = optimizer is not None
    model.train(training)
    all_pos, all_score = [], []
    supervised_loss = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(config.train.positive_class_weight, device=device), reduction="none"
    )
    for batch in loader:
        if model_name == "mlp":
            x, y, pos = (v.to(device, non_blocking=True) for v in batch)
            mask = torch.ones_like(y)
        else:
            x, y, mask, pos, _ = (v.to(device, non_blocking=True) for v in batch)
        if training: optimizer.zero_grad(set_to_none=True)
        amp_enabled = device.type == "cuda"
        with torch.set_grad_enabled(training), torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            raw = model(x)
            if model_name == "lstm_ae":
                point_loss = ((raw - x[:, -config.data.target_length:]) ** 2).mean(dim=-1)
                # AE learns normal patterns only.
                effective = mask * (y == 0).float()
                loss = (point_loss * effective).sum() / effective.sum().clamp_min(1)
                scores = point_loss
            else:
                point_loss = supervised_loss(raw, y)
                loss = (point_loss * mask).sum() / mask.sum().clamp_min(1)
                scores = torch.sigmoid(raw)
        if training:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), config.train.gradient_clip_norm)
            scaler.step(optimizer); scaler.update()
        valid = (pos >= 0) & (mask > 0)
        all_pos.append(pos[valid].detach().cpu().numpy())
        all_score.append(scores[valid].detach().float().cpu().numpy())
    return np.concatenate(all_pos), np.concatenate(all_score)

@torch.inference_mode()
def predict_loader(model: nn.Module, loader: DataLoader, model_name: str, device: torch.device,
                   config: DLConfig) -> tuple[np.ndarray, np.ndarray]:
    model.eval(); all_pos, all_score = [], []
    for batch in loader:
        if model_name == "mlp":
            x, _, pos = (v.to(device, non_blocking=True) for v in batch)
            score = torch.sigmoid(model(x)); valid = pos >= 0
        else:
            x, _, mask, pos, _ = (v.to(device, non_blocking=True) for v in batch)
            raw = model(x)
            score = ((raw - x[:, -config.data.target_length:]) ** 2).mean(-1) if model_name == "lstm_ae" else torch.sigmoid(raw)
            valid = (pos >= 0) & (mask > 0)
        all_pos.append(pos[valid].cpu().numpy()); all_score.append(score[valid].float().cpu().numpy())
    return np.concatenate(all_pos), np.concatenate(all_score)

def make_dataset(model_name: str, features: np.ndarray, frame: pd.DataFrame, positions: np.ndarray,
                 labels: np.ndarray | None, config: DLConfig):
    if model_name == "mlp":
        if labels is not None:
            positions = positions[frame["label_mask"].to_numpy()[positions] == 1]
        else:
            positions = positions[frame["row_exists_mask"].to_numpy()[positions] == 1]
        return PointDataset(features, positions, labels)
    return SequenceDataset(features, frame, positions, labels, config)

def _save_submission(test: pd.DataFrame, processed_scores: np.ndarray, name: str, config: DLConfig) -> Path:
    d, p = config.data, config.paths
    existing = test["row_exists_mask"].to_numpy() == 1
    pred = pd.DataFrame({d.test_id_column: test.loc[existing, d.test_id_column].astype("int64"),
                         d.target_column: processed_scores[existing]})
    sample = pd.read_csv(p.raw_dir / p.raw_sample_submission)
    submission = sample[[d.test_id_column]].merge(pred, on=d.test_id_column, how="left", validate="one_to_one")
    if submission[d.target_column].isna().any():
        raise ValueError(f"Missing predictions for {submission[d.target_column].isna().sum()} test rows")
    output = artifact_dirs(config)["predictions"] / f"submission_{name}.csv"
    submission.to_csv(output, index=False)
    return output

def train_fold(model_name: str, fold: int, train: pd.DataFrame, test: pd.DataFrame,
               train_idx: np.ndarray, valid_idx: np.ndarray, device: torch.device,
               config: DLConfig) -> tuple[np.ndarray, np.ndarray, dict]:
    d, dirs = config.data, artifact_dirs(config)
    transformer = FoldTransformer(d.numeric_features, d.categorical_features).fit(train.iloc[train_idx])
    train_x, test_x = transformer.transform(train), transformer.transform(test)
    labels = train[d.target_column].to_numpy(np.float32)
    train_ds = make_dataset(model_name, train_x, train, train_idx, labels, config)
    valid_ds = make_dataset(model_name, train_x, train, valid_idx, labels, config)
    test_ds = make_dataset(model_name, test_x, test, np.arange(len(test)), None, config)
    train_loader, valid_loader = make_loader(train_ds, config, True), make_loader(valid_ds, config, False)
    model = build_model(model_name, train_x.shape[1], config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.train.learning_rate, weight_decay=config.train.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    checkpoint_dir = dirs["checkpoints"] / model_name; checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_dir / f"fold_{fold}.pt"
    best_auc, stale, history = -np.inf, 0, []
    for epoch in range(1, config.train.max_epochs + 1):
        _epoch(model, train_loader, model_name, device, config, optimizer, scaler)
        pos, score = predict_loader(model, valid_loader, model_name, device, config)
        metrics = metrics_with_buildings(train, pos, labels, score, config)
        history.append({"epoch": epoch, **metrics})
        current = metrics["roc_auc"] if np.isfinite(metrics["roc_auc"]) else -np.inf
        print(f"[{model_name} fold={fold} epoch={epoch}] AUC={metrics['roc_auc']:.6f} PR={metrics['pr_auc']:.6f}")
        if current > best_auc:
            best_auc, stale = current, 0
            torch.save({"model_name": model_name, "fold": fold, "input_size": train_x.shape[1],
                        "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        "transformer": transformer.state_dict(), "config": config.to_dict(),
                        "valid_buildings": sorted(train.iloc[valid_idx][d.group_column].unique().astype(int).tolist()),
                        "history": history}, checkpoint)
        else:
            stale += 1
            if stale >= config.train.early_stopping_patience: break
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(saved["model_state"])
    valid_pos, valid_score = predict_loader(model, valid_loader, model_name, device, config)
    test_pos, test_score = predict_loader(model, make_loader(test_ds, config, False), model_name, device, config)
    del train_x, test_x, model
    if device.type == "cuda": torch.cuda.empty_cache()
    return (np.column_stack([valid_pos, valid_score]), np.column_stack([test_pos, test_score]),
            metrics_with_buildings(train, valid_pos, labels, valid_score, config))

def benchmark(models: Iterable[str], device: torch.device, config: DLConfig) -> pd.DataFrame:
    seed_everything(config.train.seed)
    train, test = load_processed(config)
    labels = train[config.data.target_column].to_numpy(np.float32)
    folds = list(group_folds(train, config.train.n_group_folds, config))
    summaries = []
    for model_name in models:
        oof = np.full(len(train), np.nan, np.float32)
        test_sum, test_count = np.zeros(len(test), np.float64), np.zeros(len(test), np.int16)
        fold_metrics = []
        for fold, (train_idx, valid_idx) in enumerate(folds):
            val_result, test_result, metrics = train_fold(
                model_name, fold, train, test, train_idx, valid_idx, device, config
            )
            vp, vs = val_result[:, 0].astype(np.int64), val_result[:, 1]
            tp, ts = test_result[:, 0].astype(np.int64), test_result[:, 1]
            oof[vp] = vs; test_sum[tp] += ts; test_count[tp] += 1
            fold_metrics.append({"fold": fold, **metrics})
        test_pred = np.divide(test_sum, test_count, out=np.full(len(test), np.nan), where=test_count > 0)
        valid = np.isfinite(oof) & (train["label_mask"].to_numpy() == 1)
        valid_pos = np.flatnonzero(valid)
        overall = metrics_with_buildings(train, valid_pos, labels, oof[valid_pos], config)
        pred_dir = artifact_dirs(config)["predictions"]
        pd.DataFrame({"processed_position": np.flatnonzero(valid), "anomaly": oof[valid]}).to_parquet(
            pred_dir / f"oof_{model_name}.parquet", index=False
        )
        output = _save_submission(test, test_pred, model_name, config)
        report = {"model": model_name, "overall": overall, "folds": fold_metrics, "submission": str(output)}
        (artifact_dirs(config)["logs"] / f"metrics_{model_name}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        summaries.append({"model": model_name, **overall,
                          "fold_auc_std": float(np.nanstd([m["roc_auc"] for m in fold_metrics]))})
    result = pd.DataFrame(summaries).sort_values("roc_auc", ascending=False)
    result.to_csv(artifact_dirs(config)["logs"] / "benchmark_summary.csv", index=False)
    print(result.to_string(index=False))
    return result

def predict_test(models: Iterable[str], device: torch.device, config: DLConfig) -> list[Path]:
    _, test = load_processed(config)
    outputs = []
    for model_name in models:
        checkpoint_dir = artifact_dirs(config)["checkpoints"] / model_name
        checkpoints = sorted(checkpoint_dir.glob("fold_*.pt"))
        if not checkpoints: raise FileNotFoundError(f"No checkpoints for {model_name} in {checkpoint_dir}")
        total, count = np.zeros(len(test), np.float64), np.zeros(len(test), np.int16)
        for path in checkpoints:
            saved = torch.load(path, map_location=device, weights_only=False)
            transformer = FoldTransformer.from_state_dict(saved["transformer"])
            features = transformer.transform(test)
            ds = make_dataset(model_name, features, test, np.arange(len(test)), None, config)
            model = build_model(model_name, saved["input_size"], config).to(device)
            model.load_state_dict(saved["model_state"])
            pos, score = predict_loader(model, make_loader(ds, config, False), model_name, device, config)
            total[pos] += score; count[pos] += 1
            del features, model
            if device.type == "cuda": torch.cuda.empty_cache()
        scores = np.divide(total, count, out=np.full(len(test), np.nan), where=count > 0)
        outputs.append(_save_submission(test, scores, model_name, config))
    return outputs

def ensemble(models: Iterable[str], config: DLConfig) -> Path:
    """Equal-weight average of percentile ranks, robust to AE score scale."""
    pred_dir = artifact_dirs(config)["predictions"]
    frames = []
    for model in models:
        frame = pd.read_csv(pred_dir / f"submission_{model}.csv")
        frame = frame.sort_values(config.data.test_id_column)
        frames.append(frame[config.data.target_column].rank(pct=True).to_numpy())
    sample = pd.read_csv(config.paths.raw_dir / config.paths.raw_sample_submission)
    order = np.argsort(sample[config.data.test_id_column].to_numpy())
    inverse = np.empty_like(order); inverse[order] = np.arange(len(order))
    averaged_sorted = np.mean(frames, axis=0)
    sample[config.data.target_column] = averaged_sorted[inverse]
    output = pred_dir / "submission_ensemble.csv"; sample.to_csv(output, index=False)
    return output

def evaluate(models: Iterable[str], config: DLConfig) -> pd.DataFrame:
    train, _ = load_processed(config); labels = train[config.data.target_column].to_numpy(np.float32)
    rows = []
    for model in models:
        oof = pd.read_parquet(artifact_dirs(config)["predictions"] / f"oof_{model}.parquet")
        pos, score = oof["processed_position"].to_numpy(int), oof["anomaly"].to_numpy(float)
        rows.append({"model": model, **metrics_with_buildings(train, pos, labels, score, config)})
    return pd.DataFrame(rows).sort_values("roc_auc", ascending=False)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P4AIDS deep-learning pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("preprocess"); prep.add_argument("--force", action="store_true")
    for command in ("benchmark", "predict-test", "evaluate", "ensemble"):
        p = sub.add_parser(command)
        p.add_argument("--models", nargs="+", choices=MODEL_NAMES,
                       default=list(MODEL_NAMES if command == "benchmark" else MODEL_NAMES[:-1]))
        if command in {"benchmark", "predict-test"}:
            p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
            p.add_argument("--workers", type=int, default=CONFIG.train.num_workers)
        if command == "benchmark":
            p.add_argument("--folds", type=int, default=CONFIG.train.n_group_folds)
            p.add_argument("--epochs", type=int, default=CONFIG.train.max_epochs)
            p.add_argument("--batch-size", type=int, default=CONFIG.train.batch_size)
    return parser.parse_args()

def main() -> None:
    args = parse_args(); config = CONFIG
    if args.command in {"benchmark", "predict-test"}:
        train_cfg = replace(config.train, num_workers=args.workers)
        if args.command == "benchmark":
            train_cfg = replace(train_cfg, n_group_folds=args.folds, max_epochs=args.epochs,
                                batch_size=args.batch_size)
        config = replace(config, train=train_cfg)
    if args.command == "preprocess": print(json.dumps(preprocess_all(config, args.force), indent=2))
    elif args.command == "benchmark": benchmark(args.models, torch.device(args.device), config)
    elif args.command == "predict-test":
        for path in predict_test(args.models, torch.device(args.device), config): print(path)
    elif args.command == "evaluate": print(evaluate(args.models, config).to_string(index=False))
    elif args.command == "ensemble": print(ensemble(args.models, config))

if __name__ == "__main__":
    main()
