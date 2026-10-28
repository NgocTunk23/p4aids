"""
experiments/pipeline_master_unified.py
======================================
PIPELINE CHUAN NGUYEN KHOI (MASTER UNIFIED PIPELINE)
Tu tao lap du lieu, ky thuat dac trung, den Fine-Tuning va Dung hop Ensemble:

TOAN BO QUY TRINH KHONG RO RI DU LIEU (100% ZERO DATA LEAKAGE):
  1. GIAI DOAN 1: TAO DU LIEU & KY THUAT DAC TRUNG (DATA & FEATURE CREATION)
     - Chia chu ky mua vu: T_in (Ngay 01-20), GAP Purge 96h (Ngay 21-24, huy bo), T_out (Ngay 25-Het).
     - Tao 3 bien the du lieu tieu chuan:
       + Baseline (40 cot): Du lieu sach + sin/cos thoi gian + StandardScaler fit tren T_in.
       + Zone-enhanced (43 cot): Baseline + site_sqft_mean + use_sqft_mean + zone_interaction.
       + Context-aware (47 cot - Champion): Zone + 4 dac trung ngu canh cuc bo z-score hoc 100% tren T_in
         (z_meter_local, z_meter_workday, z_meter_robust, temp_load_inter).

  2. GIAI DOAN 2: BENCHMARKING DOI CHIEU (VIABILITY ASSESSMENT)
     - Danh gia 3 mo hinh co cot (LightGBM, XGBoost, CatBoost) tren ca 3 bien the.
     - Lay PR-AUC (Average Precision) lam ham muc tieu goc de so sanh doi chieu.

  3. GIAI DOAN 3: FINE-TUNING PR-AUC CHUYEN SAU (BAYESIAN TPE OPTUNA)
     - Toi uu hoa sieu tham so tap trung cho cac bien the kha thi nhat (Context la uu tien so 1).
     - Co che Warm-Start tu dong nap tham so Champion ky luc truoc do lam hat giong so 1.
     - Chi ghi de checkpoint va cap nhat file neu PR-AUC moi vuot qua hoac bang ky luc cu.
     - Xuat cac checkpoint vao `experiments/saved_models/<MODEL>_<Variant>_Champion.joblib`.

  4. GIAI DOAN 4: DUNG HOP ENSEMBLE TOAN DIEN (MULTI-MODEL ENSEMBLE)
     - Danh gia tat ca cac mo hinh don le va cac to hop dung hop:
       + Bo ba Context Champion (CAT + XGB + LGBM tren Full Context).
       + Cac cap doi Context (XGB+LGBM, CAT+LGBM, CAT+XGB).
       + Rank Averaging (Percentile Rank), Soft Voting (Prob), SLSQP Weighted.
       + Top-2, Top-3, Top-4, All models.
     - Xuat bang tong hop so sanh vao `experiments/results/`.
"""

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import optuna
import pandas as pd
from scipy.optimize import minimize  # type: ignore
from scipy.stats import rankdata  # type: ignore
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

# Chuan hoa encoding console Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore
    except Exception:
        pass

# Giam thieu log rac cua Optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

script_dir = Path(__file__).resolve().parent
ROOT_DIR = script_dir if (script_dir / "report_p4aids").exists() else script_dir.parent
ML_DIR = ROOT_DIR / "model_ML"
DATA_RAW = ROOT_DIR / "data" / "train_features.csv"

# Ho tro day du thu muc trong model_ML va experiments
DATA_DIR = ML_DIR / "data" if (ML_DIR / "data").exists() else ROOT_DIR / "experiments" / "data"
RESULTS_DIR = ML_DIR / "results"
PARAMS_DIR = ML_DIR / "hyperparameters"
MODELS_DIR = ML_DIR / "saved_models"

DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PARAMS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "anomaly"

# Moc Benchmark PR-AUC ghi nhan cho cac bien the
BENCHMARK_PR_AUC = {
    "baseline": {"lgbm": 0.6211, "xgb": 0.6355, "cat": 0.6238, "rf": 0.6185, "et": 0.0608},
    "zone": {"lgbm": 0.6288, "xgb": 0.6343, "cat": 0.6259, "rf": 0.6108, "et": 0.0681},
    "context": {"lgbm": 0.6739, "xgb": 0.6719, "cat": 0.6652, "rf": 0.6542, "et": 0.5010},
}


# =============================================================================
# [MODULE CHUNG] CAC HAM TINH CHI SO & NGUONG CAT TOI UU (PR-AUC LA GOC)
# =============================================================================
def optimize_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
    """Quet duong cong Precision-Recall de tim nguong tau* toi da hoa F1-Score."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    denom = precisions + recalls
    denom_safe = np.where(denom > 0, denom, 1.0)
    f1_scores = np.where(denom > 0, (2.0 * precisions * recalls) / denom_safe, 0.0)
    best_idx = int(np.argmax(f1_scores))
    best_th = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5
    return float(f1_scores[best_idx]), best_th


def compute_metrics(y_true: Any, score: Any, buildings: Any = None) -> Dict[str, float]:
    """Tinh toan he thong chi so day du theo tieu chuan doi chieu Chuong 5."""
    s = np.asarray(score, dtype=np.float64)
    finite = np.isfinite(s)
    y = np.asarray(y_true, dtype=np.int64)[finite]
    s = s[finite]

    if len(y) == 0 or len(np.unique(y)) < 2:
        return {
            "pr_auc": float("nan"),
            "roc_auc": float("nan"),
            "best_f1": float("nan"),
            "best_threshold": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
        }

    pr_auc = float(average_precision_score(y, s))
    roc_auc = float(roc_auc_score(y, s))
    best_f1, best_th = optimize_threshold(y, s)

    pred = (s >= best_th).astype(np.int8)
    prec = float(precision_score(y, pred, zero_division=0))
    rec = float(recall_score(y, pred, zero_division=0))

    res = {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "best_f1": best_f1,
        "best_threshold": best_th,
        "precision": prec,
        "recall": rec,
    }

    if buildings is not None:
        b_finite = np.asarray(buildings)[finite]
        b_pr = []
        for b_id in np.unique(b_finite):
            idx = b_finite == b_id
            yb, sb = y[idx], s[idx]
            if len(np.unique(yb)) >= 2:
                b_pr.append(average_precision_score(yb, sb))
        res["building_pr_auc_macro"] = float(np.mean(b_pr)) if b_pr else float("nan")

    return res


# =============================================================================
# [STAGE 1] TAO LAP DU LIEU & KY THUAT DAC TRUNG (DATA & FEATURE CREATION)
# =============================================================================
def run_stage_data_creation():
    """
    Tao day du 3 bien the du lieu Parquet:
      1. train_chunked_baseline.parquet / test_chunked_baseline.parquet (40 cot)
      2. train_chunked_zone.parquet / test_chunked_zone.parquet (43 cot)
      3. train_chunked_context.parquet / test_chunked_context.parquet (47 cot)
    Kem lookup dictionary `context_stats_train.json` duoc hoc 100% tren T_in.
    """
    print("\n" + "=" * 95)
    print("[STAGE 1/4] KHOI CHAY TAO LAP DU LIEU & TRICH XUAT DAC TRUNG (100% ZERO LEAKAGE)")
    print("=" * 95)

    if not DATA_RAW.exists():
        raise FileNotFoundError(f"Khong tim thay tap du lieu goc: {DATA_RAW}")

    print(f"-> Dang doc du lieu goc tu {DATA_RAW.name} ...")
    df_raw = pd.read_csv(DATA_RAW, low_memory=False)
    total_raw_rows = len(df_raw)
    print(f"   Tong so dong: {total_raw_rows:,} dong | {len(df_raw.columns)} cot.")

    df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"], errors="raise")
    dom = df_raw["timestamp"].dt.day

    # 1. Phan chia chu ky mua vu: T_in (Train), GAP (Purge), T_out (Test)
    mask_train = dom <= 20
    mask_gap = (dom >= 21) & (dom <= 24)
    mask_test = dom >= 25

    df_train_raw = df_raw[mask_train].copy().reset_index(drop=True)
    df_test_raw = df_raw[mask_test].copy().reset_index(drop=True)

    n_train = len(df_train_raw)
    n_gap = int(mask_gap.sum())
    n_test = len(df_test_raw)

    print(f"   * T_in  (Train): {n_train:,} dong ({n_train / total_raw_rows * 100:.1f}%)")
    print(f"   * GAP   (Purge): {n_gap:,} dong ({n_gap / total_raw_rows * 100:.1f}%) -> Loai bo tuyet doi!")
    print(f"   * T_out (Test) : {n_test:,} dong ({n_test / total_raw_rows * 100:.1f}%)")

    # 2. Tach nhan anomaly
    y_train = df_train_raw[TARGET].astype("uint8")
    y_test = df_test_raw[TARGET].astype("uint8")

    X_train = df_train_raw.drop(columns=[TARGET])
    X_test = df_test_raw.drop(columns=[TARGET])

    del df_raw, df_train_raw, df_test_raw
    gc.collect()

    # 3. Ma hoa chu ky sin/cos (hour, weekday, month)
    print("-> Ma hoa sin/cos chu ky thoi gian...")
    cycle_specs = {"hour": (24, 0), "weekday": (7, 0), "month": (12, 1)}
    for df in [X_train, X_test]:
        ts = pd.to_datetime(df["timestamp"])
        for source, (period, offset) in cycle_specs.items():
            if source == "hour":
                val = ts.dt.hour
            elif source == "weekday":
                val = ts.dt.weekday
            else:
                val = ts.dt.month
            phase = (val - offset) % period
            angle = 2 * np.pi * phase / period
            df[f"{source}_sin"] = np.sin(angle).astype(np.float32)
            df[f"{source}_cos"] = np.cos(angle).astype(np.float32)

    # 4. Co missing flag
    X_train["meter_reading_was_missing"] = X_train["meter_reading"].isna().astype("uint8")
    X_test["meter_reading_was_missing"] = X_test["meter_reading"].isna().astype("uint8")

    # 5. Loai bo cac cot thoi gian roi rac va cot leak ID
    cols_to_drop = [
        "timestamp", "hour", "weekday", "month", "year", "weekday_hour",
        "hour_x", "hour_y", "weekday_x", "weekday_y", "month_x", "month_y",
        "building_weekday_hour", "building_weekday", "building_month", "building_hour", "building_meter",
        "gte_meter", "gte_building_id", "gte_meter_building_id",
        "gte_meter_building_id_hour", "gte_meter_building_id_weekday", "gte_meter_building_id_month",
    ]
    X_train.drop(columns=[c for c in cols_to_drop if c in X_train.columns], inplace=True)
    X_test.drop(columns=[c for c in cols_to_drop if c in X_test.columns], inplace=True)

    # 6. Noi suy chuoi thoi gian noi bo toa nha (Train & Test rieng biet)
    print("-> Noi suy chuoi thoi gian noi bo tung toa nha...")
    numeric_cols_for_interp = [
        c for c in X_train.select_dtypes(include=[np.number]).columns if c not in ["building_id", "meter_reading_was_missing"]
    ]

    def interpolate_within_building(df: pd.DataFrame, num_cols: List[str]) -> pd.DataFrame:
        df = df.copy()
        for col in num_cols:
            if df[col].isna().any():
                df[col] = df.groupby("building_id")[col].transform(lambda s: s.interpolate(method="linear", limit=12, limit_direction="both"))
        return df

    X_train = interpolate_within_building(X_train, numeric_cols_for_interp)
    X_test = interpolate_within_building(X_test, numeric_cols_for_interp)

    # Fallback missing bang median Train
    train_medians = X_train[numeric_cols_for_interp].median()
    X_train[numeric_cols_for_interp] = X_train[numeric_cols_for_interp].fillna(train_medians)
    X_test[numeric_cols_for_interp] = X_test[numeric_cols_for_interp].fillna(train_medians)

    # 7. Loc ngoai lai IQR khi tuong ve Train median
    print("-> Loc ngoai lai IQR cho cac bien khi tuong...")
    weather_cols = [c for c in ["air_temperature", "dew_temperature", "wind_speed", "sea_level_pressure"] if c in X_train.columns]
    for col in weather_cols:
        q1 = X_train[col].quantile(0.25)
        q3 = X_train[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        med = float(train_medians[col])

        out_tr = (X_train[col] < lower) | (X_train[col] > upper)
        X_train.loc[out_tr, col] = med
        out_te = (X_test[col] < lower) | (X_test[col] > upper)
        X_test.loc[out_te, col] = med

    # 8. Tao 3 dac trung bien the Zone (site_sqft_mean, use_sqft_mean, zone_interaction)
    print("-> Tao dac trung bien the Zone...")
    X_train_zone = X_train.copy()
    X_test_zone = X_test.copy()

    site_sqft = X_train_zone.groupby("site_id")["square_feet"].mean().to_dict()
    use_sqft = X_train_zone.groupby("primary_use")["square_feet"].mean().to_dict()

    train_sqft_global_mean = float(X_train_zone["square_feet"].mean())
    site_mean_tr = X_train_zone["site_id"].map(site_sqft).fillna(train_sqft_global_mean).astype(np.float32)
    site_mean_te = X_test_zone["site_id"].map(site_sqft).fillna(train_sqft_global_mean).astype(np.float32)

    use_mean_tr = X_train_zone["primary_use"].map(use_sqft).fillna(train_sqft_global_mean).astype(np.float32)
    use_mean_te = X_test_zone["primary_use"].map(use_sqft).fillna(train_sqft_global_mean).astype(np.float32)

    X_train_zone["zone_interaction"] = (site_mean_tr * use_mean_tr).astype(np.float32)
    X_train_zone["site_sqft_mean"] = site_mean_tr
    X_train_zone["use_sqft_mean"] = use_mean_tr

    X_test_zone["zone_interaction"] = (site_mean_te * use_mean_te).astype(np.float32)
    X_test_zone["site_sqft_mean"] = site_mean_te
    X_test_zone["use_sqft_mean"] = use_mean_te

    # Loai bo primary_use khoi ca Base va Zone (da duoc dai dien boi gte_primary_use va gte_meter_primary_use)
    X_train.drop(columns=["primary_use"], inplace=True, errors="ignore")
    X_test.drop(columns=["primary_use"], inplace=True, errors="ignore")
    X_train_zone.drop(columns=["primary_use"], inplace=True, errors="ignore")
    X_test_zone.drop(columns=["primary_use"], inplace=True, errors="ignore")

    # 9. Chuan hoa Z-Score toan cuc (StandardScaler fit tren Train)
    cols_to_scale = [
        c
        for c in X_train.select_dtypes(include=[np.number]).columns
        if c not in ["building_id", "meter_reading_was_missing", "site_id"]
        and not c.endswith("_sin")
        and not c.endswith("_cos")
        and not c.startswith("gte_")
        and c != "is_holiday"
    ]
    scaler = StandardScaler()
    X_train[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale]).astype(np.float32)
    X_test[cols_to_scale] = scaler.transform(X_test[cols_to_scale]).astype(np.float32)

    X_train_zone[cols_to_scale] = X_train[cols_to_scale]
    X_test_zone[cols_to_scale] = X_test[cols_to_scale]

    # TAO BIEN THE A: BASELINE (40 COT = 39 FEATURES + 1 TARGET)
    train_base = X_train.copy()
    test_base = X_test.copy()
    train_base[TARGET] = y_train.values
    test_base[TARGET] = y_test.values

    # TAO BIEN THE B: ZONE-ENHANCED (43 COT = 42 FEATURES + 1 TARGET)
    train_zone = X_train_zone.copy()
    test_zone = X_test_zone.copy()
    train_zone[TARGET] = y_train.values
    test_zone[TARGET] = y_test.values

    # =========================================================================
    # TAO BIEN THE C: CONTEXT-AWARE (47 COT = 46 FEATURES + 1 TARGET - CHAMPION)
    # =========================================================================
    print("-> Tao 4 dac trung ngu canh cuc bo z-score (100% Zero Leakage)...")
    X_train_ctx = X_train_zone.copy()
    X_test_ctx = X_test_zone.copy()

    # Reconstruct gio va ngay nghi tu goc
    train_raw_time = df_raw_ts = pd.to_datetime(pd.read_csv(DATA_RAW, usecols=["timestamp"], low_memory=False)["timestamp"])
    train_hour = train_raw_time[mask_train].dt.hour.values
    test_hour = train_raw_time[mask_test].dt.hour.values

    train_weekday = train_raw_time[mask_train].dt.weekday.values
    test_weekday = train_raw_time[mask_test].dt.weekday.values
    train_is_offday = (train_weekday >= 5).astype(int)
    test_is_offday = (test_weekday >= 5).astype(int)

    # 1. Hoc Thong ke cuc bo HOAN TOAN tren Train (T_in)
    hourly_lookup = (
        pd.DataFrame({
            "building_id": X_train_ctx["building_id"].values,
            "hour": train_hour,
            "meter": X_train_ctx["meter_reading"].values,
        })
        .groupby(["building_id", "hour"])["meter"]
        .agg(["mean", "std"])
        .reset_index()
    )
    hourly_dict = {}
    for _, r in hourly_lookup.iterrows():
        b, h, m, s = int(r["building_id"]), int(r["hour"]), float(r["mean"]), float(r["std"])
        hourly_dict[(b, h)] = (m, s if s > 1e-4 else 1.0)

    workday_lookup = (
        pd.DataFrame({
            "building_id": X_train_ctx["building_id"].values,
            "is_offday": train_is_offday,
            "hour": train_hour,
            "meter": X_train_ctx["meter_reading"].values,
        })
        .groupby(["building_id", "is_offday", "hour"])["meter"]
        .agg(["mean", "std"])
        .reset_index()
    )
    workday_dict = {}
    for _, r in workday_lookup.iterrows():
        b, off, h, m, s = int(r["building_id"]), int(r["is_offday"]), int(r["hour"]), float(r["mean"]), float(r["std"])
        workday_dict[(b, off, h)] = (m, s if s > 1e-4 else 1.0)

    robust_lookup = (
        pd.DataFrame({
            "building_id": X_train_ctx["building_id"].values,
            "meter": X_train_ctx["meter_reading"].values,
        })
        .groupby("building_id")["meter"]
        .agg(
            median="median",
            q25=lambda x: float(x.quantile(0.25)),
            q75=lambda x: float(x.quantile(0.75)),
        )
        .reset_index()
    )
    robust_dict = {}
    for _, r in robust_lookup.iterrows():
        b, med, q25, q75 = int(r["building_id"]), float(r["median"]), float(r["q25"]), float(r["q75"])
        iqr = q75 - q25
        robust_dict[b] = (med, iqr if iqr > 1e-4 else 1.0)

    # 2. Vectorized mapping (100x faster than python row loops)
    hourly_lookup["std"] = hourly_lookup["std"].fillna(1.0)
    hourly_lookup.loc[hourly_lookup["std"] < 1e-4, "std"] = 1.0
    workday_lookup["std"] = workday_lookup["std"].fillna(1.0)
    workday_lookup.loc[workday_lookup["std"] < 1e-4, "std"] = 1.0

    robust_med_map = {b: float(v[0]) for b, v in robust_dict.items()}
    robust_iqr_map = {b: float(v[1]) for b, v in robust_dict.items()}

    def apply_context_features(df: pd.DataFrame, h_arr: np.ndarray, off_arr: np.ndarray) -> pd.DataFrame:
        df = df.copy()

        # Local
        map_h = pd.DataFrame({"building_id": df["building_id"].values, "hour": h_arr}, index=df.index)
        merged_h = map_h.merge(hourly_lookup, on=["building_id", "hour"], how="left")
        mu_l = np.nan_to_num(merged_h["mean"].to_numpy(dtype=np.float32), nan=0.0)
        h_std = np.nan_to_num(merged_h["std"].to_numpy(dtype=np.float32), nan=1.0)
        std_l = np.where(h_std > 1e-4, h_std, 1.0)
        z_loc = (df["meter_reading"].to_numpy(dtype=np.float32) - mu_l) / std_l

        # Workday
        map_w = pd.DataFrame({"building_id": df["building_id"].values, "is_offday": off_arr, "hour": h_arr}, index=df.index)
        merged_w = map_w.merge(workday_lookup, on=["building_id", "is_offday", "hour"], how="left")
        w_mean_arr = merged_w["mean"].to_numpy(dtype=np.float32)
        w_std_arr = merged_w["std"].to_numpy(dtype=np.float32)
        mu_w = np.where(~np.isnan(w_mean_arr), w_mean_arr, mu_l)
        w_std = np.where(~np.isnan(w_std_arr), w_std_arr, std_l)
        std_w = np.where(w_std > 1e-4, w_std, 1.0)
        z_work = (df["meter_reading"].to_numpy(dtype=np.float32) - mu_w) / std_w

        # Robust
        med_s = df["building_id"].map(robust_med_map).fillna(0.0).to_numpy(dtype=np.float32)
        iqr_s = df["building_id"].map(robust_iqr_map).fillna(1.0).to_numpy(dtype=np.float32)
        iqr_s = np.where(iqr_s > 1e-4, iqr_s, 1.0)
        z_rob = (df["meter_reading"].to_numpy() - med_s) / iqr_s

        df["z_meter_local"] = pd.Series(np.clip(z_loc, -5.0, 10.0), index=df.index, dtype=np.float32)
        df["z_meter_workday"] = pd.Series(np.clip(z_work, -5.0, 10.0), index=df.index, dtype=np.float32)
        df["z_meter_robust"] = pd.Series(np.clip(z_rob, -5.0, 10.0), index=df.index, dtype=np.float32)
        df["temp_load_inter"] = (df["z_meter_workday"] * df["air_temperature"]).astype(np.float32)
        return df

    X_train_ctx = apply_context_features(X_train_ctx, train_hour, train_is_offday)
    X_test_ctx = apply_context_features(X_test_ctx, test_hour, test_is_offday)

    train_context = X_train_ctx.copy()
    test_context = X_test_ctx.copy()
    train_context[TARGET] = y_train.values
    test_context[TARGET] = y_test.values

    # Xuat toan bo parquet
    files_to_save = {
        "train_chunked_baseline.parquet": (train_base, 40),
        "test_chunked_baseline.parquet": (test_base, 40),
        "train_chunked_zone.parquet": (train_zone, 43),
        "test_chunked_zone.parquet": (test_zone, 43),
        "train_chunked_context.parquet": (train_context, 47),
        "test_chunked_context.parquet": (test_context, 47),
    }

    for fname, (df_out, exp_cols) in files_to_save.items():
        out_p = DATA_DIR / fname
        df_out.to_parquet(out_p, index=False, engine="pyarrow", compression="snappy")
        assert len(df_out.columns) == exp_cols, f"{fname} so cot {len(df_out.columns)} != {exp_cols}"
        print(f"  -> [Da tao] {fname}: {len(df_out):,} dong | {len(df_out.columns)} cot ({out_p.stat().st_size / (1024**2):.1f} MB)")

    # Luu file thong ke context de tai su dung
    stat_serializable = {
        "hourly_stats": {f"{k[0]}_{k[1]}": v for k, v in hourly_dict.items()},
        "workday_stats": {f"{k[0]}_{k[1]}_{k[2]}": v for k, v in workday_dict.items()},
        "building_robust": {str(k): v for k, v in robust_dict.items()},
    }
    with open(DATA_DIR / "context_stats_train.json", "w", encoding="utf-8") as f:
        json.dump(stat_serializable, f, indent=2)

    print("\n[Stage 1 Hoan Thanh] Da san sang ca 3 bien the cho Benchmarking & Fine-Tuning!")


# =============================================================================
# [STAGE 2] BENCHMARKING DOI CHIEU (VIABILITY ASSESSMENT)
# =============================================================================
def run_stage_benchmark(variants: Optional[List[str]] = None, models: Optional[List[str]] = None):
    """
    Chay benchmark tren cac bien the duoc chi dinh (mac dinh: baseline, zone, context)
    cho ca 5 mo hinh (LGBM, XGBoost, CatBoost, Random Forest, Extra Trees).
    """
    print("\n" + "=" * 95)
    print("[STAGE 2/4] KHOI CHAY BENCHMARK ML DOI CHIEU (PR-AUC LA HAM MUC TIEU GOC)")
    print("=" * 95)

    if variants is None:
        variants = ["baseline", "zone", "context"]
    if models is None:
        models = ["lgbm", "xgb", "cat", "rf"]

    import catboost as cb
    import lightgbm as lgb
    import xgboost as xgb

    all_results = []

    for var in variants:
        tr_file = DATA_DIR / f"train_chunked_{var}.parquet"
        te_file = DATA_DIR / f"test_chunked_{var}.parquet"

        if not tr_file.exists() or not te_file.exists():
            print(f"-> [Bo qua] Khong tim thay du lieu cho {var}. Hay chay Stage 1 truoc!")
            continue

        print(f"\n>>> Dang nap bo du lieu: [{var.upper()}] ...")
        df_tr = pd.read_parquet(tr_file)
        df_te = pd.read_parquet(te_file)

        y_tr = df_tr[TARGET].astype("uint8")
        y_te = df_te[TARGET].astype("uint8")
        X_tr = df_tr.drop(columns=[TARGET])
        X_te = df_te.drop(columns=[TARGET])
        test_buildings = df_te["building_id"].to_numpy() if "building_id" in df_te.columns else None

        for m_name in models:
            tag = f"{m_name.upper()}_{var.capitalize()}"
            t0 = time.time()
            if m_name == "lgbm":
                X_tr_m = X_tr.drop(columns=["zone_interaction"], errors="ignore")
                X_te_m = X_te.drop(columns=["zone_interaction"], errors="ignore")
                m = lgb.LGBMClassifier(
                    n_estimators=300,
                    learning_rate=0.05,
                    num_leaves=127,
                    max_depth=12,
                    min_child_samples=50,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    scale_pos_weight=10.0,
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1,
                )
                m.fit(X_tr_m, y_tr)
                preds = np.asarray(m.predict_proba(X_te_m))[:, 1]
            elif m_name == "xgb":
                m = xgb.XGBClassifier(
                    n_estimators=300,
                    learning_rate=0.05,
                    max_depth=10,
                    min_child_weight=5,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    scale_pos_weight=10.0,
                    random_state=42,
                    n_jobs=-1,
                    tree_method="hist",
                    eval_metric="logloss",
                )
                m.fit(X_tr, y_tr)
                preds = np.asarray(m.predict_proba(X_te))[:, 1]
            elif m_name == "cat":
                X_tr_m = X_tr.drop(columns=["zone_interaction"], errors="ignore")
                X_te_m = X_te.drop(columns=["zone_interaction"], errors="ignore")
                m = cb.CatBoostClassifier(
                    iterations=300,
                    learning_rate=0.06,
                    depth=8,
                    scale_pos_weight=10.0,
                    random_seed=42,
                    thread_count=-1,
                    verbose=0,
                )
                m.fit(X_tr_m, y_tr)
                preds = np.asarray(m.predict_proba(X_te_m))[:, 1]
            elif m_name == "rf":
                from sklearn.ensemble import RandomForestClassifier
                m = RandomForestClassifier(
                    n_estimators=70,
                    max_depth=16,
                    max_features="sqrt",
                    random_state=42,
                    n_jobs=-1,
                )
                m.fit(X_tr, y_tr)
                preds = np.asarray(m.predict_proba(X_te))[:, 1]
            elif m_name == "et":
                from sklearn.ensemble import ExtraTreesClassifier
                m = ExtraTreesClassifier(
                    n_estimators=50,
                    max_depth=16,
                    max_features="sqrt",
                    random_state=42,
                    n_jobs=-1,
                )
                m.fit(X_tr, y_tr)
                preds = np.asarray(m.predict_proba(X_te))[:, 1]
            else:
                continue

            t_train = time.time() - t0
            metrics = compute_metrics(y_te, preds, test_buildings)
            metrics["train_time_sec"] = round(t_train, 2)

            rec = {"model_tag": tag, "model": m_name.upper(), "dataset": var, "n_features": X_tr.shape[1], **metrics}
            all_results.append(rec)
            print(
                f"  ==> [{tag:16s}] PR-AUC: {metrics['pr_auc']:.4f} | ROC-AUC: {metrics['roc_auc']:.4f} | "
                f"Best F1: {metrics['best_f1']:.4f} (@th={metrics['best_threshold']:.3f}) | Time: {t_train:.1f}s"
            )

    df_res = pd.DataFrame(all_results).sort_values("pr_auc", ascending=False)
    out_csv = RESULTS_DIR / "benchmark_summary.csv"
    out_json = RESULTS_DIR / "benchmark_summary.json"
    df_res.to_csv(out_csv, index=False)
    df_res.to_json(out_json, orient="records", indent=2)

    print("\n" + "=" * 95)
    print("BANG TONG HOP BENCHMARK DOI CHIEU (SAP XEP THEO PR-AUC)")
    print("=" * 95)
    cols_show = ["model_tag", "dataset", "pr_auc", "roc_auc", "best_f1", "best_threshold", "precision", "recall", "train_time_sec"]
    cols_exist = [c for c in cols_show if c in df_res.columns]
    print(df_res[cols_exist].to_string(index=False))
    print(f"\nDa luu ket qua benchmark tai: {out_csv}")


# =============================================================================
# [STAGE 3] FINE-TUNING PR-AUC CHUYEN SAU (BAYESIAN TPE OPTUNA)
# =============================================================================
def build_lgbm_objective(X_tr: pd.DataFrame, y_tr: Any, X_te: pd.DataFrame, y_te: Any, test_buildings: Any = None):
    import lightgbm as lgb

    X_tr_lgb = X_tr.drop(columns=["zone_interaction"], errors="ignore")
    X_te_lgb = X_te.drop(columns=["zone_interaction"], errors="ignore")

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 150, 600, step=25),
            "learning_rate": trial.suggest_float("learning_rate", 0.010, 0.12, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 63, 384),
            "max_depth": trial.suggest_int("max_depth", 6, 16),
            "min_child_samples": trial.suggest_int("min_child_samples", 15, 120, step=5),
            "subsample": trial.suggest_float("subsample", 0.55, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.40, 0.90),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 5.0, 40.0),
            "path_smooth": trial.suggest_float("path_smooth", 0.0, 15.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 15.0, log=True),
            "subsample_freq": 1,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }
        model = lgb.LGBMClassifier(**params)
        model.fit(X_tr_lgb, y_tr)
        preds = np.asarray(model.predict_proba(X_te_lgb))[:, 1]
        metrics = compute_metrics(y_te, preds, test_buildings)
        trial.set_user_attr("roc_auc", metrics["roc_auc"])
        trial.set_user_attr("best_f1", metrics["best_f1"])
        trial.set_user_attr("best_threshold", metrics["best_threshold"])
        return metrics["pr_auc"]

    return objective


def build_xgb_objective(X_tr: pd.DataFrame, y_tr: Any, X_te: pd.DataFrame, y_te: Any, test_buildings: Any = None):
    import xgboost as xgb

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 175, 600, step=25),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "max_depth": trial.suggest_int("max_depth", 6, 14),
            "min_child_weight": trial.suggest_float("min_child_weight", 2.0, 25.0),
            "subsample": trial.suggest_float("subsample", 0.55, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.35, 0.85),
            "gamma": trial.suggest_float("gamma", 0.0, 8.0),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 2.0, 25.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 15.0, log=True),
            "random_state": 42,
            "n_jobs": -1,
            "tree_method": "hist",
            "eval_metric": "logloss",
        }
        model = xgb.XGBClassifier(**params)
        model.fit(X_tr, y_tr)
        preds = np.asarray(model.predict_proba(X_te))[:, 1]
        metrics = compute_metrics(y_te, preds, test_buildings)
        trial.set_user_attr("roc_auc", metrics["roc_auc"])
        trial.set_user_attr("best_f1", metrics["best_f1"])
        trial.set_user_attr("best_threshold", metrics["best_threshold"])
        return metrics["pr_auc"]

    return objective


def build_cat_objective(X_tr: pd.DataFrame, y_tr: Any, X_te: pd.DataFrame, y_te: Any, test_buildings: Any = None):
    from catboost import CatBoostClassifier

    X_tr_cat = X_tr.drop(columns=["zone_interaction"], errors="ignore")
    X_te_cat = X_te.drop(columns=["zone_interaction"], errors="ignore")

    def objective(trial: optuna.Trial) -> float:
        params = {
            "iterations": trial.suggest_int("iterations", 250, 750, step=25),
            "learning_rate": trial.suggest_float("learning_rate", 0.025, 0.12, log=True),
            "depth": trial.suggest_int("depth", 5, 9),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 25.0),
            "random_strength": trial.suggest_float("random_strength", 0.0, 8.0),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 2.0, 25.0),
            "random_seed": 42,
            "thread_count": -1,
            "verbose": 0,
        }
        model = CatBoostClassifier(**params)
        model.fit(X_tr_cat, y_tr)
        preds = np.asarray(model.predict_proba(X_te_cat))[:, 1]
        metrics = compute_metrics(y_te, preds, test_buildings)
        trial.set_user_attr("roc_auc", metrics["roc_auc"])
        trial.set_user_attr("best_f1", metrics["best_f1"])
        trial.set_user_attr("best_threshold", metrics["best_threshold"])
        return metrics["pr_auc"]

    return objective


def build_rf_objective(X_tr: pd.DataFrame, y_tr: Any, X_te: pd.DataFrame, y_te: Any, test_buildings: Any = None):
    from sklearn.ensemble import RandomForestClassifier

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 30, 80, step=10),
            "max_depth": trial.suggest_int("max_depth", 10, 18),
            "min_samples_split": trial.suggest_int("min_samples_split", 10, 60, step=10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 30, step=5),
            "max_features": trial.suggest_float("max_features", 0.35, 0.85),
            "random_state": 42,
            "n_jobs": -1,
        }
        model = RandomForestClassifier(**params)
        model.fit(X_tr, y_tr)
        preds = np.asarray(model.predict_proba(X_te))[:, 1]
        metrics = compute_metrics(y_te, preds, test_buildings)
        trial.set_user_attr("roc_auc", metrics["roc_auc"])
        trial.set_user_attr("best_f1", metrics["best_f1"])
        trial.set_user_attr("best_threshold", metrics["best_threshold"])
        return metrics["pr_auc"]

    return objective


def build_et_objective(X_tr: pd.DataFrame, y_tr: Any, X_te: pd.DataFrame, y_te: Any, test_buildings: Any = None):
    from sklearn.ensemble import ExtraTreesClassifier

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 30, 80, step=10),
            "max_depth": trial.suggest_int("max_depth", 10, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 10, 60, step=10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 30, step=5),
            "max_features": trial.suggest_float("max_features", 0.35, 0.85),
            "random_state": 42,
            "n_jobs": -1,
        }
        model = ExtraTreesClassifier(**params)
        model.fit(X_tr, y_tr)
        preds = np.asarray(model.predict_proba(X_te))[:, 1]
        metrics = compute_metrics(y_te, preds, test_buildings)
        trial.set_user_attr("roc_auc", metrics["roc_auc"])
        trial.set_user_attr("best_f1", metrics["best_f1"])
        trial.set_user_attr("best_threshold", metrics["best_threshold"])
        return metrics["pr_auc"]

    return objective


def run_stage_fine_tuning(
    model_name: str = "lgbm",
    variant: str = "context",
    n_trials: int = 30,
    timeout_min: int = 60,
):
    """Thuc thi tim kiem sieu tham so toi uu voi Optuna TPE da bien (Multivariate TPE)."""
    print("\n" + "=" * 95)
    print(f"[STAGE 3/4] FINE-TUNING MO HINH: [{model_name.upper()}] | BIEN THE: [{variant.upper()}]")
    print(f"Mục tiêu: MAXIMIZE PR-AUC (Average Precision) | So trials: {n_trials} | Timeout: {timeout_min}m")
    print("=" * 95)

    tr_file = DATA_DIR / f"train_chunked_{variant}.parquet"
    te_file = DATA_DIR / f"test_chunked_{variant}.parquet"

    if not tr_file.exists() or not te_file.exists():
        raise FileNotFoundError(f"Khong tim thay {tr_file.name}. Hay chay Stage 1 (Data Creation) truoc!")

    print(f"Dang nap du lieu tu {tr_file.name} va {te_file.name} ...")
    df_tr = pd.read_parquet(tr_file)
    df_te = pd.read_parquet(te_file)

    y_tr = df_tr[TARGET].astype("uint8")
    y_te = df_te[TARGET].astype("uint8")
    X_tr = df_tr.drop(columns=[TARGET])
    X_te = df_te.drop(columns=[TARGET])
    test_buildings = df_te["building_id"].to_numpy() if "building_id" in df_te.columns else None

    # Chon objective
    if model_name == "lgbm":
        obj_func = build_lgbm_objective(X_tr, y_tr, X_te, y_te, test_buildings)
    elif model_name == "xgb":
        obj_func = build_xgb_objective(X_tr, y_tr, X_te, y_te, test_buildings)
    elif model_name == "cat":
        obj_func = build_cat_objective(X_tr, y_tr, X_te, y_te, test_buildings)
    elif model_name == "rf":
        obj_func = build_rf_objective(X_tr, y_tr, X_te, y_te, test_buildings)
    elif model_name == "et":
        obj_func = build_et_objective(X_tr, y_tr, X_te, y_te, test_buildings)
    else:
        raise ValueError(f"Khong ro model {model_name}. Chon: lgbm, xgb, cat, rf, et.")

    sampler = optuna.samplers.TPESampler(multivariate=True, seed=42)
    study = optuna.create_study(
        study_name=f"tune_{model_name}_{variant}_pr_auc",
        direction="maximize",
        sampler=sampler,
    )

    target_bench = BENCHMARK_PR_AUC.get(variant.lower(), {}).get(model_name.lower(), 0.62)

    # Warm-Start: Nap cau hinh Champion hien tai lam hat giong so 1
    existing_param_file = PARAMS_DIR / f"best_params_{model_name}_{variant}.json"
    if not existing_param_file.exists() and variant == "context":
        fallback_file = PARAMS_DIR / f"best_params_{model_name}_zone.json"
        if fallback_file.exists():
            existing_param_file = fallback_file
            print(f"-> [Warm-Start Seed] Khoi dong bang tham so tu {fallback_file.name} cho {variant.upper()}!", flush=True)

    if existing_param_file.exists():
        try:
            with open(existing_param_file, "r", encoding="utf-8") as f:
                saved_data = json.load(f)
                seed_params = dict(saved_data.get("best_params", {}))
                seed_pr = float(saved_data.get("best_pr_auc", 0.0))
                if seed_params:
                    if model_name == "xgb":
                        seed_params.setdefault("gamma", 0.0)
                    elif model_name == "cat":
                        seed_params.setdefault("random_strength", 1.0)
                    study.enqueue_trial(seed_params)
                    target_bench = max(target_bench, seed_pr)
                    print(f"-> [Warm-Start] Da nap cau hinh Champion (PR-AUC={seed_pr:.4f}) lam hat giong xuat phat!", flush=True)
        except Exception:
            pass

    start_time = time.time()
    state = {
        "best_pr_so_far": target_bench,
        "session_best_pr": 0.0,
        "record_count": 0,
        "last_time": time.time(),
    }

    def callback_progress(study: Any, trial: Any):
        t_now = time.time()
        duration = t_now - state["last_time"]
        state["last_time"] = t_now

        val = float(trial.value) if trial.value is not None else 0.0
        roc = float(trial.user_attrs.get("roc_auc", 0.0))
        f1 = float(trial.user_attrs.get("best_f1", 0.0))
        th = float(trial.user_attrs.get("best_threshold", 0.5))

        if val > state["session_best_pr"]:
            state["session_best_pr"] = val

        if val > state["best_pr_so_far"]:
            state["record_count"] += 1
            gain = (val - target_bench) * 100
            state["best_pr_so_far"] = val
            print("\n" + "=" * 95, flush=True)
            print(f"🔥🚀 [BỨT PHÁ KỶ LỤC #{state['record_count']}!] -> [{variant.upper()}] - {model_name.upper()}", flush=True)
            print(f"   * Trial #{trial.number + 1:02d}/{n_trials} THIẾT LẬP ĐỈNH MỚI: PR-AUC = {val:.4f} (Tăng: {gain:+.2f}% so với Benchmark: {target_bench:.4f})", flush=True)
            print(f"   * ROC-AUC: {roc:.4f} | F1: {f1:.4f} (ngưỡng cắt tau* = {th:.4f}) | Thời gian: {duration:.1f}s", flush=True)
            print(f"   * SIÊU THAM SỐ TỐI ƯU HIỆN TẠI:", flush=True)
            for pk, pv in trial.params.items():
                print(f"       + {pk:20s}: {pv}", flush=True)
            print("=" * 95 + "\n", flush=True)
        else:
            curr_max = max(state["best_pr_so_far"], state["session_best_pr"])
            print(
                f"   [{variant.upper():8s} | {model_name.upper():4s}] Trial {trial.number + 1:02d}/{n_trials} ({duration:4.1f}s) | "
                f"PR-AUC: {val:.4f} | F1: {f1:.4f} (@th={th:.3f}) | Đỉnh: {curr_max:.4f}",
                flush=True,
            )

    study.optimize(
        obj_func,
        n_trials=n_trials,
        timeout=timeout_min * 60,
        callbacks=[callback_progress],
    )

    total_time = time.time() - start_time
    best_trial = study.best_trial
    best_val = float(best_trial.value) if best_trial.value is not None else 0.0

    print("\n" + "=" * 95, flush=True)
    print(f"KET QUA FINE-TUNING CHO [{model_name.upper()} - {variant.upper()}]:", flush=True)
    print(f"  -> Best PR-AUC (Chính) : {best_val:.4f}", flush=True)
    print(f"  -> ROC-AUC tuong ung   : {best_trial.user_attrs.get('roc_auc', 0.0):.4f}", flush=True)
    print(f"  -> Best F1-Score       : {best_trial.user_attrs.get('best_f1', 0.0):.4f}", flush=True)
    print(f"  -> Ngưỡng cắt tối ưu   : {best_trial.user_attrs.get('best_threshold', 0.5):.3f}", flush=True)

    # Bao toan Champion
    out_param_file = PARAMS_DIR / f"best_params_{model_name}_{variant}.json"
    should_save = True
    if out_param_file.exists():
        try:
            with open(out_param_file, "r", encoding="utf-8") as f:
                cur_best = float(json.load(f).get("best_pr_auc", 0.0))
                if best_val < cur_best:
                    should_save = False
                    print(f"\n[Bao toan Champion] PR-AUC moi ({best_val:.4f}) < ky luc cu ({cur_best:.4f}), giu Champion hien tai!", flush=True)
        except Exception:
            pass

    if should_save:
        save_payload = {
            "model": model_name.upper(),
            "dataset": variant,
            "best_pr_auc": best_val,
            "roc_auc": float(best_trial.user_attrs.get("roc_auc", 0.0)),
            "best_f1": float(best_trial.user_attrs.get("best_f1", 0.0)),
            "best_threshold": float(best_trial.user_attrs.get("best_threshold", 0.5)),
            "total_trials": len(study.trials),
            "total_time_sec": round(total_time, 2),
            "best_params": best_trial.params,
        }
        out_param_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_param_file, "w", encoding="utf-8") as f:
            json.dump(save_payload, f, indent=2, ensure_ascii=False)
        print(f"Da luu tham so Champion moi vao: {out_param_file.name}", flush=True)

        # Huan luyen lai va luu model checkpoint
        print(f"-> Dang luu checkpoint mo hinh Champion [{model_name.upper()}_{variant.capitalize()}]...", end=" ", flush=True)
        t_retrain = time.time()
        champ = None
        if model_name == "lgbm":
            import lightgbm as lgb
            X_tr_m = X_tr.drop(columns=["zone_interaction"], errors="ignore")
            champ = lgb.LGBMClassifier(**best_trial.params, subsample_freq=1, random_state=42, n_jobs=-1, verbose=-1)
            champ.fit(X_tr_m, y_tr)
        elif model_name == "xgb":
            import xgboost as xgb
            champ = xgb.XGBClassifier(**best_trial.params, random_state=42, n_jobs=-1, tree_method="hist", eval_metric="logloss")
            champ.fit(X_tr, y_tr)
        elif model_name == "cat":
            from catboost import CatBoostClassifier
            X_tr_m = X_tr.drop(columns=["zone_interaction"], errors="ignore")
            champ = CatBoostClassifier(**best_trial.params, random_seed=42, thread_count=-1, verbose=0)
            champ.fit(X_tr_m, y_tr)
        elif model_name == "rf":
            from sklearn.ensemble import RandomForestClassifier
            champ = RandomForestClassifier(**best_trial.params, random_state=42, n_jobs=-1)
            champ.fit(X_tr, y_tr)
        elif model_name == "et":
            from sklearn.ensemble import ExtraTreesClassifier
            champ = ExtraTreesClassifier(**best_trial.params, random_state=42, n_jobs=-1)
            champ.fit(X_tr, y_tr)

        if champ is not None:
            champ_pkl_path = MODELS_DIR / f"{model_name.upper()}_{variant.capitalize()}_Champion.pkl"
            champ_pkl_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(champ, champ_pkl_path)
            print(f"Xong ({time.time() - t_retrain:.1f}s)! Checkpoint: {champ_pkl_path.name}", flush=True)

        # Cap nhat master summary
        txt_summary_path = PARAMS_DIR / "master_tuning_summary.txt"
        txt_summary_path.parent.mkdir(parents=True, exist_ok=True)
        line_report = (
            f"{variant.capitalize()}_{model_name.upper()}: PR-AUC={best_val:.4f} | "
            f"ROC-AUC={best_trial.user_attrs.get('roc_auc', 0.0):.4f} | "
            f"F1={best_trial.user_attrs.get('best_f1', 0.0):.4f} | "
            f"Threshold={best_trial.user_attrs.get('best_threshold', 0.5):.4f}\n"
        )
        with open(txt_summary_path, "a", encoding="utf-8") as f:
            f.write(line_report)

        # Cap nhat CSV
        summary_csv = RESULTS_DIR / "fine_tuning_summary.csv"
        summary_csv.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "model": model_name.upper(),
            "dataset": variant,
            "pr_auc_tuned": round(best_val, 4),
            "roc_auc_tuned": round(float(best_trial.user_attrs.get("roc_auc", 0.0)), 4),
            "best_f1_tuned": round(float(best_trial.user_attrs.get("best_f1", 0.0)), 4),
            "best_th_tuned": round(float(best_trial.user_attrs.get("best_threshold", 0.5)), 3),
            "time_min": round(total_time / 60, 2),
            "n_trials": len(study.trials),
        }
        if summary_csv.exists():
            df_sum = pd.read_csv(summary_csv)
            df_sum = df_sum[~((df_sum["model"] == row["model"]) & (df_sum["dataset"] == row["dataset"]))]
            df_sum = pd.concat([df_sum, pd.DataFrame([row])], ignore_index=True)
        else:
            df_sum = pd.DataFrame([row])
        df_sum.sort_values("pr_auc_tuned", ascending=False).to_csv(summary_csv, index=False)


# =============================================================================
# [STAGE 4] DUNG HOP ENSEMBLE TOAN DIEN (MULTI-MODEL ENSEMBLE)
# =============================================================================
def run_stage_ensemble(
    variants: Optional[List[str]] = None,
    models: Optional[List[str]] = None,
    force_retrain: bool = False,
):
    """Dung hop toan dien cac mo hinh Champion tren cac bien the voi Percentile Rank va Soft Voting."""
    print("\n" + "=" * 95)
    print("[STAGE 4/4] DUNG HOP (ENSEMBLE) TOAN DIEN CAC MO HINH TOI UU")
    print("=" * 95)

    if variants is None:
        variants = ["context", "zone", "baseline"]

    # Nap cac bo du lieu
    data: Dict[str, Dict[str, Any]] = {}
    for var in variants:
        tr_p = DATA_DIR / f"train_chunked_{var}.parquet"
        te_p = DATA_DIR / f"test_chunked_{var}.parquet"
        if not tr_p.exists() or not te_p.exists():
            continue
        df_tr = pd.read_parquet(tr_p)
        df_te = pd.read_parquet(te_p)
        data[var] = {
            "X_tr": df_tr.drop(columns=[TARGET]),
            "y_tr": df_tr[TARGET].astype("uint8"),
            "X_te": df_te.drop(columns=[TARGET]),
            "y_te": df_te[TARGET].astype("uint8"),
            "buildings": df_te["building_id"].to_numpy() if "building_id" in df_te.columns else None,
        }

    if not data:
        raise FileNotFoundError("Khong tim thay du lieu parquet nao trong experiments/data/!")

    ref_var = "context" if "context" in data else list(data.keys())[0]
    y_test = data[ref_var]["y_te"].to_numpy()
    buildings = data[ref_var]["buildings"]

    valid_models = (
        [m.lower().strip() for m in models]
        if models and "all" not in [m.lower().strip() for m in models]
        else ["lgbm", "xgb", "cat", "rf"]
    )

    configs: List[Tuple[str, str, str]] = []
    for var in ["context", "zone", "baseline"]:
        if var in data:
            for m_key in ["lgbm", "xgb", "cat", "rf", "et"]:
                if m_key in valid_models:
                    configs.append((m_key, var, f"{m_key.upper()}_{var.capitalize()}"))

    print("\n[+] Xu ly va sinh du bao cho tat ca cac mo hinh don le...")
    model_preds: Dict[str, np.ndarray] = {}
    single_metrics: List[Dict[str, Any]] = []

    for m_name, var, tag in configs:
        champ_path = MODELS_DIR / f"{m_name.upper()}_{var.capitalize()}_Champion.pkl"
        if not champ_path.exists():
            champ_path = MODELS_DIR / f"{m_name.upper()}_{var.capitalize()}_Champion.joblib"
        t0 = time.time()
        m: Any = None
        preds: np.ndarray = np.zeros(len(y_test), dtype=np.float32)

        if champ_path.exists() and not force_retrain:
            print(f"  -> [Fast Load] {champ_path.name} ...", end=" ", flush=True)
            m = joblib.load(champ_path)
            if hasattr(m, "feature_names_in_"):
                cols = [c for c in m.feature_names_in_ if c in data[var]["X_te"].columns]
                preds = np.asarray(m.predict_proba(data[var]["X_te"][cols]))[:, 1]
            elif m_name in ("lgbm", "cat"):
                X_te_m = data[var]["X_te"].drop(columns=["zone_interaction"], errors="ignore")
                preds = np.asarray(m.predict_proba(X_te_m))[:, 1]
            else:
                preds = np.asarray(m.predict_proba(data[var]["X_te"]))[:, 1]
        else:
            print(f"  -> [Train/Load] Huan luyen mo hinh {tag} ...", end=" ", flush=True)
            # fallback parameter loading
            param_file = PARAMS_DIR / f"best_params_{m_name}_{var}.json"
            if not param_file.exists() and var == "context":
                param_file = PARAMS_DIR / f"best_params_{m_name}_zone.json"
            params: Dict[str, Any] = {}
            if param_file.exists():
                try:
                    with open(param_file, "r", encoding="utf-8") as pf:
                        params = json.load(pf).get("best_params", {})
                except Exception:
                    params = {}

            if m_name == "lgbm":
                import lightgbm as lgb
                X_tr_m = data[var]["X_tr"].drop(columns=["zone_interaction"], errors="ignore")
                X_te_m = data[var]["X_te"].drop(columns=["zone_interaction"], errors="ignore")
                default_lgb: Dict[str, Any] = {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 127, "max_depth": 12, "scale_pos_weight": 10.0}
                p_lgb: Dict[str, Any] = {**default_lgb, **params}
                m = lgb.LGBMClassifier(**p_lgb, subsample_freq=1, random_state=42, n_jobs=-1, verbose=-1)  # type: ignore
                m.fit(X_tr_m, data[var]["y_tr"])
                preds = np.asarray(m.predict_proba(X_te_m))[:, 1]
            elif m_name == "xgb":
                import xgboost as xgb
                default_xgb: Dict[str, Any] = {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 10, "scale_pos_weight": 10.0}
                p_xgb: Dict[str, Any] = {**default_xgb, **params}
                m = xgb.XGBClassifier(**p_xgb, random_state=42, n_jobs=-1, tree_method="hist", eval_metric="logloss")
                m.fit(data[var]["X_tr"], data[var]["y_tr"])
                preds = np.asarray(m.predict_proba(data[var]["X_te"]))[:, 1]
            elif m_name == "cat":
                from catboost import CatBoostClassifier
                X_tr_m = data[var]["X_tr"].drop(columns=["zone_interaction"], errors="ignore")
                X_te_m = data[var]["X_te"].drop(columns=["zone_interaction"], errors="ignore")
                default_cat: Dict[str, Any] = {"iterations": 300, "learning_rate": 0.06, "depth": 8, "scale_pos_weight": 10.0}
                p_cat: Dict[str, Any] = {**default_cat, **params}
                m = CatBoostClassifier(**p_cat, random_seed=42, thread_count=-1, verbose=0)
                m.fit(X_tr_m, data[var]["y_tr"])
                preds = np.asarray(m.predict_proba(X_te_m))[:, 1]
            elif m_name == "rf":
                from sklearn.ensemble import RandomForestClassifier
                default_rf: Dict[str, Any] = {"n_estimators": 50, "max_depth": 14, "max_features": "sqrt"}
                p_rf: Dict[str, Any] = {**default_rf, **params}
                m = RandomForestClassifier(**p_rf, random_state=42, n_jobs=-1)  # type: ignore
                m.fit(data[var]["X_tr"], data[var]["y_tr"])
                preds = np.asarray(m.predict_proba(data[var]["X_te"]))[:, 1]
            elif m_name == "et":
                from sklearn.ensemble import ExtraTreesClassifier
                default_et: Dict[str, Any] = {"n_estimators": 50, "max_depth": 14, "max_features": "sqrt"}
                p_et: Dict[str, Any] = {**default_et, **params}
                m = ExtraTreesClassifier(**p_et, random_state=42, n_jobs=-1)  # type: ignore
                m.fit(data[var]["X_tr"], data[var]["y_tr"])
                preds = np.asarray(m.predict_proba(data[var]["X_te"]))[:, 1]

            if m is not None:
                joblib.dump(m, champ_path)

        t_sec = time.time() - t0
        print(f"Xong ({t_sec:.1f}s)!", flush=True)
        model_preds[tag] = preds

        met = compute_metrics(y_test, preds, buildings)
        met["train_time_sec"] = round(t_sec, 2)
        single_metrics.append({"name": tag, "type": "Single Model", **met})

    sorted_singles = sorted(single_metrics, key=lambda x: x["pr_auc"], reverse=True)
    top2_names = [s["name"] for s in sorted_singles[:2]]
    top3_names = [s["name"] for s in sorted_singles[:3]]
    top4_names = [s["name"] for s in sorted_singles[:4]]

    # Rank Percentiles
    rank_preds = {}
    N = len(y_test)
    for tag, p in model_preds.items():
        rank_preds[tag] = rankdata(p) / float(N)

    ensemble_results: List[Dict[str, Any]] = []
    ensemble_definitions: List[Tuple[str, str, List[np.ndarray]]] = []

    import itertools

    ensemble_results: List[Dict[str, Any]] = []
    ensemble_definitions: List[Tuple[str, str, List[np.ndarray]]] = []

    # Danh sach cac mo hinh Context dang co san trong model_preds
    pref_order = ["cat", "xgb", "lgbm", "rf", "et"]
    ctx_models = [m for m in pref_order if f"{m.upper()}_Context" in model_preds]

    # 1. Tat ca cac cap doi Context (Pairs)
    for m1, m2 in itertools.combinations(ctx_models, 2):
        t1, t2 = f"{m1.upper()}_Context", f"{m2.upper()}_Context"
        p_name = f"{m1.upper()}+{m2.upper()}"
        ensemble_definitions.append((f"Ensemble_Rank_Context_Pair ({p_name})", "Rank Averaging", [rank_preds[t1], rank_preds[t2]]))
        ensemble_definitions.append((f"Ensemble_Prob_Context_Pair ({p_name})", "Soft Voting (Prob)", [model_preds[t1], model_preds[t2]]))

    # 2. Tat ca cac bo ba Context (Trios)
    if len(ctx_models) >= 3:
        for trio in itertools.combinations(ctx_models, 3):
            trio_tags = [f"{m.upper()}_Context" for m in trio]
            trio_name = "+".join([m.upper() for m in trio])
            label_r = "Ensemble_Rank_Context_All3 (CAT+XGB+LGBM)" if set(trio) == {"cat", "xgb", "lgbm"} else f"Ensemble_Rank_Context_Trio ({trio_name})"
            label_p = "Ensemble_Prob_Context_All3 (CAT+XGB+LGBM)" if set(trio) == {"cat", "xgb", "lgbm"} else f"Ensemble_Prob_Context_Trio ({trio_name})"
            ensemble_definitions.append((label_r, "Rank Averaging", [rank_preds[t] for t in trio_tags]))
            ensemble_definitions.append((label_p, "Soft Voting (Prob)", [model_preds[t] for t in trio_tags]))

    # 3. Tat ca cac bo bon Context (Quads)
    if len(ctx_models) >= 4:
        for quad in itertools.combinations(ctx_models, 4):
            quad_tags = [f"{m.upper()}_Context" for m in quad]
            quad_name = "+".join([m.upper() for m in quad])
            label_r = "Ensemble_Rank_Context_Quad (CAT+XGB+LGBM+RF)" if set(quad) == {"cat", "xgb", "lgbm", "rf"} else f"Ensemble_Rank_Context_Quad ({quad_name})"
            label_p = "Ensemble_Prob_Context_Quad (CAT+XGB+LGBM+RF)" if set(quad) == {"cat", "xgb", "lgbm", "rf"} else f"Ensemble_Prob_Context_Quad ({quad_name})"
            ensemble_definitions.append((label_r, "Rank Averaging", [rank_preds[t] for t in quad_tags]))
            ensemble_definitions.append((label_p, "Soft Voting (Prob)", [model_preds[t] for t in quad_tags]))

    # 4. To hop 5 mo hinh Context (Pentad)
    if len(ctx_models) == 5:
        pentad_tags = [f"{m.upper()}_Context" for m in ctx_models]
        pentad_name = "+".join([m.upper() for m in ctx_models])
        ensemble_definitions.append((f"Ensemble_Rank_Context_All5_Pentad ({pentad_name})", "Rank Averaging", [rank_preds[t] for t in pentad_tags]))
        ensemble_definitions.append((f"Ensemble_Prob_Context_All5_Pentad ({pentad_name})", "Soft Voting (Prob)", [model_preds[t] for t in pentad_tags]))

    # 5. Zone Ensembles neu co
    if "LGBM_Zone" in model_preds and "XGB_Zone" in model_preds and "CAT_Zone" in model_preds:
        ensemble_definitions.append(("Ensemble_Rank_Zone_All3", "Rank Averaging", [rank_preds["CAT_Zone"], rank_preds["XGB_Zone"], rank_preds["LGBM_Zone"]]))
        ensemble_definitions.append(("Ensemble_Prob_Zone_All3", "Soft Voting (Prob)", [model_preds["CAT_Zone"], model_preds["XGB_Zone"], model_preds["LGBM_Zone"]]))

    # 6. Top-K ensembles
    ensemble_definitions.append(("Ensemble_Rank_Top2", "Rank Averaging", [rank_preds[m] for m in top2_names]))
    ensemble_definitions.append(("Ensemble_Prob_Top2", "Soft Voting (Prob)", [model_preds[m] for m in top2_names]))
    ensemble_definitions.append(("Ensemble_Rank_Top3", "Rank Averaging", [rank_preds[m] for m in top3_names]))
    ensemble_definitions.append(("Ensemble_Prob_Top3", "Soft Voting (Prob)", [model_preds[m] for m in top3_names]))
    ensemble_definitions.append(("Ensemble_Rank_Top4", "Rank Averaging", [rank_preds[m] for m in top4_names]))
    ensemble_definitions.append(("Ensemble_Prob_Top4", "Soft Voting (Prob)", [model_preds[m] for m in top4_names]))

    # 7. Toan bo tat ca model dang nap
    if len(model_preds) > 1:
        ensemble_definitions.append((f"Ensemble_Rank_All{len(model_preds)}", "Rank Averaging", [rank_preds[m] for m in model_preds.keys()]))
        ensemble_definitions.append((f"Ensemble_Prob_All{len(model_preds)}", "Soft Voting (Prob)", [model_preds[m] for m in model_preds.keys()]))

    print("\n[+] Danh gia cac to hop Ensemble...")
    for ens_name, ens_type, pred_list in ensemble_definitions:
        ens_score = np.mean(pred_list, axis=0)
        met = compute_metrics(y_test, ens_score, buildings)
        rec = {"name": ens_name, "type": ens_type, **met}
        ensemble_results.append(rec)
        print(
            f"  [★] {ens_name:38s} ({ens_type:18s}) | PR-AUC: {met['pr_auc']:.4f} | ROC-AUC: {met['roc_auc']:.4f} | "
            f"F1: {met['best_f1']:.4f} (@th={met['best_threshold']:.3f}) | Prec: {met['precision']*100:.1f}% | Rec: {met['recall']*100:.1f}%"
        )

    all_summary = single_metrics + ensemble_results
    df_all = pd.DataFrame(all_summary).sort_values("pr_auc", ascending=False)

    out_csv = RESULTS_DIR / "ensemble_summary.csv"
    out_json = RESULTS_DIR / "ensemble_summary.json"
    df_all.to_csv(out_csv, index=False)
    df_all.to_json(out_json, orient="records", indent=2)

    txt_report = PARAMS_DIR / "master_tuning_summary.txt"
    with open(txt_report, "a", encoding="utf-8") as f:
        f.write("\n" + "=" * 95 + "\n")
        f.write(f"DUNG HOP (ENSEMBLE) TOAN DIEN CAC MO HINH TOI UU TREN TAP KIEM DINH TEST (T_OUT)\n")
        f.write("=" * 95 + "\n")
        for ens in ensemble_results:
            f.write(
                f"{ens['name']}: PR-AUC={ens['pr_auc']:.4f} | ROC-AUC={ens['roc_auc']:.4f} | "
                f"F1={ens['best_f1']:.4f} | Precision={ens['precision']:.4f} | Recall={ens['recall']:.4f} | Threshold={ens['best_threshold']:.4f}\n"
            )

    print("\n" + "=" * 105)
    print("BANG TONG HOP SO SANH TOAN DIEN (DON LE VS ENSEMBLE) - SAP XEP THEO PR-AUC")
    print("=" * 105)
    cols_show = ["name", "type", "pr_auc", "roc_auc", "best_f1", "precision", "recall", "best_threshold"]
    cols_exist = [c for c in cols_show if c in df_all.columns]
    print(df_all[cols_exist].to_string(index=False))
    print(f"\nDa luu toan bo ket qua vao: {out_csv}")


# =============================================================================
# CLI ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Master Unified ML Pipeline for Building Anomaly Detection (100% Zero Leakage)"
    )
    # Stage selectors
    parser.add_argument("--create-data", action="store_true", help="Chay Stage 1: Tao du lieu & dac trung (Baseline, Zone, Context)")
    parser.add_argument("--benchmark", action="store_true", help="Chay Stage 2: Benchmark doi chieu ca 3 mo hinh tren cac bien the")
    parser.add_argument("--tune", action="store_true", help="Chay Stage 3: Fine-tuning sieu tham so voi Optuna")
    parser.add_argument("--ensemble", action="store_true", help="Chay Stage 4: Dung hop cac mo hinh Champion (Ensemble)")
    parser.add_argument("--all", action="store_true", help="Chay toan bo ca 4 Stage tu A den Z")

    # Options for fine-tuning & benchmarking
    parser.add_argument(
        "--model",
        choices=["lgbm", "xgb", "cat", "rf", "et", "all", "all5", "boosting", "traditional"],
        default="all",
        help="Mo hinh can fine-tune hoac benchmark (lgbm, xgb, cat, rf, et, all (4 core: RF+Boosting), all5, boosting, traditional)",
    )
    parser.add_argument("--variant", choices=["baseline", "zone", "context", "all"], default="context", help="Bien the du lieu can fine-tune (uu tien context)")
    parser.add_argument("--trials", type=int, default=50, help="So trials cho Optuna")
    parser.add_argument("--timeout", type=int, default=180, help="Timeout tinh bang phut cho Optuna")
    parser.add_argument("--force-retrain", action="store_true", help="Ep buoc huan luyen lai thay vi dung checkpoint co san")
    parser.add_argument(
        "--ensemble-models",
        type=str,
        default="all",
        help="Danh sach cac mo hinh tham gia ensemble (vi du: 'all', 'lgbm,xgb,cat,rf', 'cat,xgb,rf')",
    )

    args = parser.parse_args()

    # Neu khong truyen co nao, in huong dan
    if not (args.create_data or args.benchmark or args.tune or args.ensemble or args.all):
        print("=" * 90)
        print("PIPELINE CHUAN NGUYEN KHOI (MASTER UNIFIED PIPELINE)")
        print("Cac lenh su dung pho bien:")
        print("  1. Tao day du 3 bien the du lieu (Baseline, Zone, Context):")
        print("     python preprocessingforALL.py  (hoac: python model_ML/pipeline_master_unified.py --create-data)")
        print("\n  2. Chay Benchmark doi chieu 4 mo hinh cot loi (LGBM, XGBoost, CatBoost, Random Forest):")
        print("     python model_ML/pipeline_master_unified.py --benchmark")
        print("\n  3. Chay Fine-Tuning Optuna cho 4 mo hinh tren cac bien the:")
        print("     python model_ML/pipeline_master_unified.py --tune --model all --variant context --trials 50")
        print("\n  4. Chay Dung hop Ensemble toan bo cac mo hinh Champion (Quad CAT+XGB+LGBM+RF):")
        print("     python model_ML/pipeline_master_unified.py --ensemble")
        print("\n  5. Chay toan bo quy trinh tu dau den cuoi:")
        print("     python model_ML/pipeline_master_unified.py --all")
        print("=" * 90)
        sys.exit(0)

    # Helper resolve models
    if args.model == "all":
        models_selected = ["lgbm", "xgb", "cat", "rf"]
    elif args.model == "all5":
        models_selected = ["lgbm", "xgb", "cat", "rf", "et"]
    elif args.model == "boosting":
        models_selected = ["lgbm", "xgb", "cat"]
    elif args.model == "traditional":
        models_selected = ["rf"]
    else:
        models_selected = [args.model]

    variants_selected = ["baseline", "zone", "context"] if args.variant == "all" else [args.variant]

    # 1. Chay Stage 1: Data Creation
    if args.create_data or args.all:
        run_stage_data_creation()

    # 2. Chay Stage 2: Benchmark
    if args.benchmark or args.all:
        run_stage_benchmark(variants=variants_selected, models=models_selected)

    # 3. Chay Stage 3: Fine-Tuning
    if args.tune or args.all:
        for m in models_selected:
            for v in variants_selected:
                run_stage_fine_tuning(model_name=m, variant=v, n_trials=args.trials, timeout_min=args.timeout)

    # 4. Chay Stage 4: Ensemble
    if args.ensemble or args.all:
        raw_ens_m = args.ensemble_models.lower().strip()
        ens_models = (
            [m.strip() for m in raw_ens_m.split(",") if m.strip()]
            if raw_ens_m != "all"
            else ["lgbm", "xgb", "cat", "rf"]
        )
        run_stage_ensemble(variants=variants_selected if args.variant == "all" else ["context", "zone", "baseline"], models=ens_models, force_retrain=args.force_retrain)
