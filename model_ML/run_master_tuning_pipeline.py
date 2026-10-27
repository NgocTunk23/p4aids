"""
model_ML/run_master_tuning_pipeline.py
======================================
Master Tuning Pipeline cho 3 Mo hinh Machine Learning (XGBoost, LightGBM, CatBoost)
tren 3 Bien the Du lieu (Baseline 40 cot, Zone 43 cot, Context 47 cot - Champion).

Toan bo quy trinh tuan thu 100% Zero Data Leakage voi chu ky Seasonal Chunking:
  - T_in (Ngay 01-20): Train
  - GAP 96h (Ngay 21-24): Huy bo cach ly
  - T_out (Ngay 25-Het): Test doc lap
"""

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

# UTF-8 cho Windows console
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            getattr(sys.stdout, "reconfigure")(encoding="utf-8")
        if hasattr(sys.stderr, "reconfigure"):
            getattr(sys.stderr, "reconfigure")(encoding="utf-8")
    except Exception:
        pass

import joblib
import numpy as np
import optuna
import pandas as pd
import scipy.stats as stats  # type: ignore
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Thiet lap thu muc chuan hoa
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
DATA_DIR = SCRIPT_DIR / "data"
PARAMS_DIR = SCRIPT_DIR / "hyperparameters"
MODELS_DIR = SCRIPT_DIR / "saved_models"
RESULTS_DIR = SCRIPT_DIR / "results"

for d in [DATA_DIR, PARAMS_DIR, MODELS_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

TARGET = "anomaly"

# Import truc tiep tu pipeline_master_unified de dong bo tuyet doi
try:
    from pipeline_master_unified import (
        run_stage_fine_tuning,
        run_stage_ensemble,
        run_stage_benchmark,
        optimize_threshold as base_optimize_threshold,
        compute_metrics as base_compute_metrics,
    )
except ImportError:
    # Neu chay doc lap tu thu muc khac
    sys.path.insert(0, str(SCRIPT_DIR))
    from pipeline_master_unified import (
        run_stage_fine_tuning,
        run_stage_ensemble,
        run_stage_benchmark,
        optimize_threshold as base_optimize_threshold,
        compute_metrics as base_compute_metrics,
    )


# ---------------------------------------------------------------------------------
# 1. HAM TAI VA CHUAN BI DU LIEU CHUNKER (ZERO DATA LEAKAGE)
# ---------------------------------------------------------------------------------
def load_chunked_dataset(
    variant: str = "context",
) -> Tuple[pd.DataFrame, Any, pd.DataFrame, Any, Any]:
    """
    Tai bo du lieu Parquet tuong ung voi 100% Zero Data Leakage:
      - 'baseline': 40 dac trung co ban (du lieu sach + sin/cos chu ky thoi gian)
      - 'zone': 43 dac trung (baseline + site_sqft_mean + use_sqft_mean + zone_interaction)
      - 'context': 47 dac trung Champion (zone + 4 dac trung ngu canh cuc bo z-score)
    """
    var = variant.lower().strip()
    tr_path = DATA_DIR / f"train_chunked_{var}.parquet"
    te_path = DATA_DIR / f"test_chunked_{var}.parquet"

    # Du phong tim trong data/processed neu can
    if not tr_path.exists():
        alt_tr = ROOT_DIR / "data" / "processed" / f"train_chunked_{var}.parquet"
        if alt_tr.exists():
            tr_path = alt_tr
    if not te_path.exists():
        alt_te = ROOT_DIR / "data" / "processed" / f"test_chunked_{var}.parquet"
        if alt_te.exists():
            te_path = alt_te

    if not tr_path.exists() or not te_path.exists():
        raise FileNotFoundError(
            f"Khong tim thay du lieu cho bien the '{var}'. "
            f"Vui long chay 'python preprocessingforALL.py' hoac pipeline Stage 1 truoc!"
        )

    print(f"-> Dang nap du lieu [{var.upper()}]: {tr_path.name} & {te_path.name}...", end=" ", flush=True)
    t0 = time.time()
    df_tr = pd.read_parquet(tr_path)
    df_te = pd.read_parquet(te_path)

    y_train = df_tr[TARGET].astype("uint8")
    y_test = df_te[TARGET].astype("uint8")

    X_train = df_tr.drop(columns=[TARGET, "row_id"], errors="ignore")
    X_test = df_te.drop(columns=[TARGET, "row_id"], errors="ignore")

    b_test: Any = df_te["building_id"].to_numpy() if "building_id" in df_te.columns else None

    print(f"Xong trong {time.time() - t0:.1f}s. Kich thuoc Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, y_train, X_test, y_test, b_test


# ---------------------------------------------------------------------------------
# 2. OPTIMIZE THRESHOLD & EVALUATION
# ---------------------------------------------------------------------------------
def optimize_threshold(y_true: Any, y_prob: Any) -> Tuple[float, float]:
    """Quet duong cong Precision-Recall tim nguong tau* toi da hoa F1-Score."""
    return base_optimize_threshold(y_true, y_prob)


def evaluate_predictions(y_true: Any, y_prob: Any, buildings: Any = None) -> Dict[str, float]:
    """Tinh toan he thong chi so toan dien: PR-AUC, ROC-AUC, F1, Precision, Recall."""
    return base_compute_metrics(y_true, y_prob, buildings)


# ---------------------------------------------------------------------------------
# 3. CLI RUNNER
# ---------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Master Tuning Pipeline cho 5 Mo hinh (XGB, LGBM, CAT, RF, ET) va 3 Bien the (Baseline, Zone, Context)."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["lgbm", "xgb", "cat", "rf", "et", "all", "all5", "boosting", "traditional"],
        help="Mo hinh can tuning (lgbm, xgb, cat, rf, et, all (4 core: RF+Boosting), all5, boosting, hoac traditional)",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="context",
        help="Bien the du lieu (baseline, zone, context, all, hoac ket hop phan tach bang dau phay vi du 'baseline,zone')",
    )
    parser.add_argument(
        "--exclude",
        type=str,
        default="",
        help="Danh sach bien the can bo qua (vi du: 'context' hoac 'context,zone')",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Bo qua cac cap model/variant da co file best_params_{m}_{v}.json",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=50,
        help="So luot thu nghiem Bayesian Optuna (default: 50)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=180,
        help="Gioi han thoi gian tuning moi mo hinh theo phut (default: 180)",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Chay danh gia Benchmark doi chieu truoc khi tuning",
    )
    parser.add_argument(
        "--ensemble",
        action="store_true",
        help="Chay dung hop Ensemble toan dien sau khi tuning",
    )
    parser.add_argument(
        "--ensemble-models",
        type=str,
        default="all",
        help="Danh sach cac mo hinh tham gia ensemble (vi du: 'all', 'lgbm,xgb,cat,rf', 'cat,xgb,rf')",
    )

    args = parser.parse_args()

    if args.model == "all":
        models_to_run = ["lgbm", "xgb", "cat", "rf"]
    elif args.model == "all5":
        models_to_run = ["lgbm", "xgb", "cat", "rf", "et"]
    elif args.model == "boosting":
        models_to_run = ["lgbm", "xgb", "cat"]
    elif args.model == "traditional":
        models_to_run = ["rf"]
    else:
        models_to_run = [args.model.lower().strip()]

    raw_var = args.variant.lower().strip()
    if raw_var == "all":
        all_variants = ["baseline", "zone", "context"]
    elif "," in raw_var:
        all_variants = [v.strip() for v in raw_var.split(",") if v.strip()]
    else:
        all_variants = [raw_var]

    excludes = [e.strip() for e in args.exclude.lower().split(",") if e.strip()]
    variants_to_run = [v for v in all_variants if v not in excludes]

    print("\n" + "=" * 95)
    print("KHOI CHAY MASTER TUNING PIPELINE (ZERO DATA LEAKAGE)")
    print(f"-> Mo hinh : {models_to_run}")
    print(f"-> Bien the: {variants_to_run} (Loai tru: {excludes if excludes else 'Khong'})")
    print(f"-> So trials: {args.trials} | Timeout: {args.timeout} phut | Skip existing: {args.skip_existing}")
    print("=" * 95)

    if args.benchmark:
        run_stage_benchmark(models=models_to_run, variants=variants_to_run)

    for m in models_to_run:
        for v in variants_to_run:
            if args.skip_existing:
                param_f = PARAMS_DIR / f"best_params_{m}_{v}.json"
                if param_f.exists():
                    print(f"\n[SKIP-EXISTING] Bo qua [{m.upper()} - {v.upper()}] vi da ton tai: {param_f.name}", flush=True)
                    continue

            run_stage_fine_tuning(
                model_name=m,
                variant=v,
                n_trials=args.trials,
                timeout_min=args.timeout,
            )

    if args.ensemble:
        # Khi ensemble, mac dinh dung hop tat ca cac tap du lieu da co model san sang
        ens_variants = ["context", "zone", "baseline"]
        raw_ens_m = args.ensemble_models.lower().strip()
        ens_models = (
            [m.strip() for m in raw_ens_m.split(",") if m.strip()]
            if raw_ens_m != "all"
            else ["lgbm", "xgb", "cat", "rf"]
        )
        run_stage_ensemble(variants=ens_variants, models=ens_models)

    print("\n[HOAN TAT] Master Tuning Pipeline da thuc thi thanh cong!")


if __name__ == "__main__":
    main()
