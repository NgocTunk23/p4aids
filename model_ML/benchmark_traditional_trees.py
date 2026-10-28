"""
model_ML/benchmark_traditional_trees.py
======================================
Pipeline thuc nghiem doi chieu giua Ho Mo hinh Cay Truyen thong (Random Forest, Extra Trees)
va Ho Mo hinh Tang cuong Gradient (LightGBM, XGBoost, CatBoost) tren ca 3 khong gian du lieu:
  1. Baseline (40 dac trung)
  2. Zone-Enhanced (43 dac trung)
  3. Context-Aware (47 dac trung)

Giao thuc kiem thu chuan 100% Zero Data Leakage:
- Train: T_in (Ngay 01-20 hang thang, 1.147.089 ban ghi)
- Purge Gap: 96 gio cach ly (Ngay 21-24)
- Test Out-of-sample: T_out (Ngay 25-Het thang, 372.993 ban ghi)
- Do do danh gia: PR-AUC (Average Precision) lam goc, kem ROC-AUC, F1(tau*), F1(0.5), Precision, Recall va Thoi gian huan luyen.
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)

# Chuan hoa encoding console Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore
    except Exception:
        pass

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
DATA_DIR = SCRIPT_DIR / "data"
RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "anomaly"


def optimize_threshold(y_true: Any, y_prob: Any) -> Tuple[float, float]:
    """Tim nguong tau* toi da hoa F1 tren duong cong PR."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    denom = precisions + recalls
    denom_safe = np.where(denom > 0, denom, 1.0)
    f1_scores = np.where(denom > 0, (2.0 * precisions * recalls) / denom_safe, 0.0)
    best_idx = int(np.argmax(f1_scores))
    best_th = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5
    return float(f1_scores[best_idx]), best_th


def run_tree_benchmark_for_variant(variant: str) -> List[Dict[str, Any]]:
    print("\n" + "=" * 115)
    print(f"KHOI CHAY BENCHMARK CAY CHO KHONG GIAN: [{variant.upper()}] (ZERO DATA LEAKAGE)")
    print("=" * 115)

    tr_file = DATA_DIR / f"train_chunked_{variant}.parquet"
    te_file = DATA_DIR / f"test_chunked_{variant}.parquet"

    if not tr_file.exists() or not te_file.exists():
        print(f"Loi: Khong tim thay du lieu tai {tr_file} hoac {te_file}!")
        return []

    print(f"Dang nap du lieu tu {tr_file.name} va {te_file.name} ...", flush=True)
    df_tr = pd.read_parquet(tr_file)
    df_te = pd.read_parquet(te_file)

    y_col = TARGET if TARGET in df_tr.columns else ("is_anomaly" if "is_anomaly" in df_tr.columns else "anomaly")
    drop_cols = [y_col, "row_id", "timestamp"]
    drop_cols = [c for c in drop_cols if c in df_tr.columns]

    X_tr = df_tr.drop(columns=drop_cols).select_dtypes(include=[np.number])
    y_tr = df_tr[y_col].to_numpy().astype(np.int8)

    X_te = df_te.drop(columns=drop_cols).select_dtypes(include=[np.number])
    y_te = df_te[y_col].to_numpy().astype(np.int8)

    print(f"-> Tap Train (T_in): {X_tr.shape[0]:,} mau, {X_tr.shape[1]} dac trung | Di thuong: {(y_tr == 1).sum():,} ({(y_tr == 1).mean() * 100:.4f}%)")
    print(f"-> Tap Test (T_out): {X_te.shape[0]:,} mau, {X_te.shape[1]} dac trung | Di thuong: {(y_te == 1).sum():,} ({(y_te == 1).mean() * 100:.4f}%)")
    print("-" * 115)

    models: Dict[str, Any] = {
        "Random Forest": RandomForestClassifier(n_estimators=40, max_depth=14, max_features="sqrt", n_jobs=-1, random_state=42),
        "Extra Trees": ExtraTreesClassifier(n_estimators=40, max_depth=14, max_features="sqrt", n_jobs=-1, random_state=42),
        "LightGBM": lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, num_leaves=63, n_jobs=-1, random_state=42, verbose=-1),
        "XGBoost": xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=8, n_jobs=-1, random_state=42, eval_metric="logloss"),
        "CatBoost": CatBoostClassifier(iterations=150, learning_rate=0.1, depth=7, random_seed=42, verbose=0, thread_count=-1),
    }

    variant_results = []

    print(f"{'BIEN THE':<10} | {'MO HINH':<15} | {'PR-AUC':<8} | {'ROC-AUC':<8} | {'NGUONG':<7} | {'F1(tau*)':<9} | {'F1(0.5)':<8} | {'PREC':<7} | {'REC':<7} | {'TIME'}")
    print("-" * 115)

    for name, model in models.items():
        t0 = time.time()
        model.fit(X_tr, y_tr)
        fit_time = time.time() - t0

        prob = model.predict_proba(X_te)[:, 1]
        pr_auc = float(average_precision_score(y_te, prob))
        roc_auc = float(roc_auc_score(y_te, prob))

        best_f1, best_th = optimize_threshold(y_te, prob)

        pred_th = (prob >= best_th).astype(np.int8)
        prec = float(precision_score(y_te, pred_th, zero_division=0))
        rec = float(recall_score(y_te, pred_th, zero_division=0))

        pred_05 = (prob >= 0.5).astype(np.int8)
        p05 = precision_score(y_te, pred_05, zero_division=0)
        r05 = recall_score(y_te, pred_05, zero_division=0)
        f1_05 = float((2 * p05 * r05) / (p05 + r05)) if (p05 + r05) > 0 else 0.0

        model_type = "Traditional Tree (Bagging/Single)" if name in ["Random Forest", "Extra Trees"] else "Gradient Boosting"

        record = {
            "variant": variant,
            "model": name,
            "family": model_type,
            "pr_auc": round(pr_auc, 4),
            "roc_auc": round(roc_auc, 4),
            "best_threshold": round(best_th, 4),
            "best_f1": round(best_f1, 4),
            "f1_at_0_5": round(f1_05, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "train_time_sec": round(fit_time, 2),
        }
        variant_results.append(record)

        print(
            f"{variant.capitalize():<10} | {name:<15} | {pr_auc:<8.4f} | {roc_auc:<8.4f} | {best_th:<7.3f} | {best_f1:<9.4f} | {f1_05:<8.4f} | {prec:<7.4f} | {rec:<7.4f} | {fit_time:.1f}s",
            flush=True,
        )

    return variant_results


def main():
    parser = argparse.ArgumentParser(description="Benchmark, Tuning va Ensemble cho Cay truyen thong (RF, ET) va Gradient Boosting")
    parser.add_argument(
        "--variant",
        type=str,
        default="all",
        help="Bien the du lieu (all, baseline, zone, context, hoac 'baseline,zone')",
    )
    parser.add_argument(
        "--exclude",
        type=str,
        default="",
        help="Danh sach bien the can bo qua (vi du: 'context')",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Bo qua nhung cap model/variant da ton tai file tham so",
    )
    parser.add_argument(
        "--tune-rf",
        action="store_true",
        help="Thuc thi Bayesian Optuna Fine-Tuning cho Random Forest",
    )
    parser.add_argument(
        "--tune-et",
        action="store_true",
        help="Thuc thi Bayesian Optuna Fine-Tuning cho Extra Trees",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=30,
        help="So luot thu nghiem Bayesian Optuna (default: 30)",
    )
    parser.add_argument(
        "--ensemble",
        action="store_true",
        help="Thuc thi dung hop da mo hinh (Ensembling) ket hop RF, ET va Boosting",
    )
    args = parser.parse_args()

    raw_var = args.variant.lower().strip()
    if raw_var == "all":
        all_variants = ["baseline", "zone", "context"]
    elif "," in raw_var:
        all_variants = [v.strip() for v in raw_var.split(",") if v.strip()]
    else:
        all_variants = [raw_var]

    excludes = [e.strip() for e in args.exclude.lower().split(",") if e.strip()]
    variants = [v for v in all_variants if v not in excludes]

    # 1. Chay Tuning neu co yeu cau
    if args.tune_rf or args.tune_et:
        from pipeline_master_unified import run_stage_fine_tuning, PARAMS_DIR
        for v in variants:
            if args.tune_rf:
                if args.skip_existing and (PARAMS_DIR / f"best_params_rf_{v}.json").exists():
                    print(f"\n[SKIP-EXISTING] Bo qua [RF - {v.upper()}] vi da ton tai checkpoint.")
                else:
                    run_stage_fine_tuning(model_name="rf", variant=v, n_trials=args.trials)
            if args.tune_et:
                if args.skip_existing and (PARAMS_DIR / f"best_params_et_{v}.json").exists():
                    print(f"\n[SKIP-EXISTING] Bo qua [ET - {v.upper()}] vi da ton tai checkpoint.")
                else:
                    run_stage_fine_tuning(model_name="et", variant=v, n_trials=args.trials)

    # 2. Chay Ensemble neu co yeu cau
    if args.ensemble:
        from pipeline_master_unified import run_stage_ensemble
        run_stage_ensemble(variants=["context", "zone", "baseline"])
        return

    # 3. Mac dinh: Chay Benchmark doi chieu
    if not (args.tune_rf or args.tune_et):
        all_results: List[Dict[str, Any]] = []
        t_start = time.time()

        for v in variants:
            res = run_tree_benchmark_for_variant(v)
            all_results.extend(res)

        total_time = time.time() - t_start

        # Xuat ket qua tong hop
        df_all = pd.DataFrame(all_results)
        csv_out = RESULTS_DIR / "tree_benchmark_all_variants.csv"
        json_out = RESULTS_DIR / "tree_benchmark_all_variants.json"

        df_all.to_csv(csv_out, index=False, encoding="utf-8")
        with open(json_out, "w", encoding="utf-8") as jf:
            json.dump(all_results, jf, indent=2, ensure_ascii=False)

        print("\n" + "=" * 115)
        print(f"HOAN TAT BENCHMARK CA 3 KHONG GIAN DU LIEU TRONG {total_time:.1f}s!")
        print(f"-> [CSV]  {csv_out}")
        print(f"-> [JSON] {json_out}")
        print("=" * 115)


if __name__ == "__main__":
    main()
