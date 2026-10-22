import os
import sys
import time
import json
import warnings
from pathlib import Path

# Đảm bảo UTF-8 cho Windows Console
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8')
        if hasattr(sys.stderr, 'reconfigure'):
            sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, f1_score, precision_score, recall_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

TARGET = "anomaly"
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_BASELINE = BASE_DIR / "data" / "processed" / "train_preprocessed.parquet"
DATA_ZONE = BASE_DIR / "data" / "processed" / "train_preprocessedzone.parquet"
DATA_RAW = BASE_DIR / "data" / "train_features.csv"
OUT_JSON = BASE_DIR / "model_ML" / "hyperparameters" / "lag_vs_zone_test_results.json"

def find_optimal_threshold(y_true, y_prob):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_th = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5
    best_f1 = float(f1_scores[best_idx])
    
    y_pred_opt = (y_prob >= best_th).astype(int)
    prec_opt = float(precision_score(y_true, y_pred_opt, zero_division=0))
    rec_opt = float(recall_score(y_true, y_pred_opt, zero_division=0))
    
    return {
        'optimal_threshold': round(best_th, 4),
        'precision': round(prec_opt, 4),
        'recall': round(rec_opt, 4),
        'f1_score': round(best_f1, 4)
    }

def get_tuned_baseline_models():
    """Bộ siêu tham số đã tuning từ Optuna cho Baseline / Lag"""
    return {
        "XGBoost": XGBClassifier(
            learning_rate=0.2375786861052034,
            max_depth=15,
            subsample=0.9599947358815135,
            colsample_bytree=0.6210976217070321,
            n_estimators=164,
            tree_method='hist',
            n_jobs=-1,
            random_state=42,
            eval_metric='logloss'
        ),
        "LightGBM": LGBMClassifier(
            learning_rate=0.15367758401415987,
            num_leaves=233,
            max_depth=17,
            feature_fraction=0.8458886576427025,
            min_child_samples=62,
            n_estimators=200,
            n_jobs=-1,
            random_state=42,
            verbose=-1
        ),
        "CatBoost": CatBoostClassifier(
            learning_rate=0.262571031377097,
            depth=14,
            l2_leaf_reg=1.3969479625934749,
            iterations=178,
            task_type='GPU',
            random_seed=42,
            verbose=0
        )
    }

def get_tuned_zone_models():
    """Bộ siêu tham số đã tuning từ Optuna cho Zone"""
    return {
        "XGBoost": XGBClassifier(
            learning_rate=0.2382238758466043,
            max_depth=16,
            subsample=0.9755387922832883,
            colsample_bytree=0.6088134291940173,
            n_estimators=152,
            tree_method='hist',
            n_jobs=-1,
            random_state=42,
            eval_metric='logloss'
        ),
        "LightGBM": LGBMClassifier(
            learning_rate=0.1576597043447496,
            num_leaves=252,
            max_depth=16,
            feature_fraction=0.5720622388332075,
            min_child_samples=73,
            n_estimators=215,
            n_jobs=-1,
            random_state=42,
            verbose=-1
        ),
        "CatBoost": CatBoostClassifier(
            learning_rate=0.346734305948443,
            depth=14,
            l2_leaf_reg=2.257276285054781,
            iterations=176,
            task_type='GPU',
            random_seed=42,
            verbose=0
        )
    }

def main():
    print("=" * 85)
    print("   THỰC NGHIỆM ĐỐI CHỨNG: LAG FEATURES vs ZONE FEATURES (VỚI MÔ HÌNH ĐÃ TUNING)")
    print("=" * 85)
    print("-> Sử dụng siêu tham số ĐÃ TUNING TỪ OPTUNA (Best Params)")
    print("-> Quét ngưỡng tối ưu tau* để tối đa hóa F1-Score")
    print("-" * 85, flush=True)

    # 1. TẠO DỮ LIỆU NHÁNH 1: LAG FEATURES
    print("-> Đang nạp và trích xuất Đặc trưng Trễ (Lag Features)...", end=" ", flush=True)
    t0 = time.time()
    df_base = pd.read_parquet(DATA_BASELINE)
    df_raw_ts = pd.read_csv(DATA_RAW, usecols=['timestamp'])
    df_base['timestamp'] = pd.to_datetime(df_raw_ts['timestamp'])

    # Sắp xếp theo building_id và timestamp để tạo biến trễ
    df_base = df_base.sort_values(['building_id', 'timestamp'])
    df_base['meter_reading_lag1'] = df_base.groupby('building_id')['meter_reading'].shift(1).fillna(-999)
    df_base['meter_reading_lag2'] = df_base.groupby('building_id')['meter_reading'].shift(2).fillna(-999)
    
    # Trả về thứ tự ban đầu
    df_base = df_base.sort_index()
    X_lag = df_base.drop(columns=[TARGET, 'timestamp', 'row_id'], errors='ignore')
    y_lag = df_base[TARGET]
    print(f"Xong trong {time.time() - t0:.1f}s. Kích thước X_lag: {X_lag.shape}")

    X_train_lag, X_val_lag, y_train_lag, y_val_lag = train_test_split(
        X_lag, y_lag, test_size=0.2, random_state=42, stratify=y_lag
    )

    # 2. TẠO DỮ LIỆU NHÁNH 2: ZONE FEATURES
    print("-> Đang nạp Đặc trưng Khu vực (Zone Features)...", end=" ", flush=True)
    t0 = time.time()
    df_zone = pd.read_parquet(DATA_ZONE)
    if 'timestamp' in df_zone.columns:
        df_zone = df_zone.drop(columns=['timestamp'])
    X_zone = df_zone.drop(columns=[TARGET, 'row_id'], errors='ignore')
    y_zone = df_zone[TARGET]
    print(f"Xong trong {time.time() - t0:.1f}s. Kích thước X_zone: {X_zone.shape}")

    X_train_zone, X_val_zone, y_train_zone, y_val_zone = train_test_split(
        X_zone, y_zone, test_size=0.2, random_state=42, stratify=y_zone
    )
    print("-" * 85)

    results = []

    # 3. HUẤN LUYỆN VÀ ĐÁNH GIÁ NHÁNH 1 (LAG FEATURES - TUNED)
    print("--- [BẮT ĐẦU ĐÁNH GIÁ NHÁNH 1: LAG FEATURES (TUNED)] ---")
    models_lag = get_tuned_baseline_models()
    for name in ["XGBoost", "LightGBM", "CatBoost"]:
        clf = models_lag[name]
        print(f"-> Huấn luyện [{name}] trên Lag Features...", end=" ", flush=True)
        t_start = time.time()
        clf.fit(X_train_lag, y_train_lag)
        y_prob = clf.predict_proba(X_val_lag)[:, 1]
        m = find_optimal_threshold(y_val_lag, y_prob)
        m['branch'] = 'Nhánh 1: Lag Features'
        m['model'] = name
        results.append(m)
        print(f"Xong {time.time() - t_start:.1f}s | tau*: {m['optimal_threshold']:.4f} | Prec: {m['precision']:.4f} | Rec: {m['recall']:.4f} | F1: {m['f1_score']:.4f}")

    print("\n--- [BẮT ĐẦU ĐÁNH GIÁ NHÁNH 2: ZONE FEATURES (TUNED)] ---")
    # 4. HUẤN LUYỆN VÀ ĐÁNH GIÁ NHÁNH 2 (ZONE FEATURES - TUNED)
    models_zone = get_tuned_zone_models()
    for name in ["XGBoost", "LightGBM", "CatBoost"]:
        clf = models_zone[name]
        print(f"-> Huấn luyện [{name}] trên Zone Features...", end=" ", flush=True)
        t_start = time.time()
        clf.fit(X_train_zone, y_train_zone)
        y_prob = clf.predict_proba(X_val_zone)[:, 1]
        m = find_optimal_threshold(y_val_zone, y_prob)
        m['branch'] = 'Nhánh 2: Zone Features'
        m['model'] = name
        results.append(m)
        print(f"Xong {time.time() - t_start:.1f}s | tau*: {m['optimal_threshold']:.4f} | Prec: {m['precision']:.4f} | Rec: {m['recall']:.4f} | F1: {m['f1_score']:.4f}")

    # 5. Lưu kết quả ra JSON
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    # 6. Hiển thị bảng tổng hợp (không có cột thời gian theo yêu cầu)
    df_res = pd.DataFrame(results)[['branch', 'model', 'optimal_threshold', 'precision', 'recall', 'f1_score']]
    df_res.columns = ['Chiến thuật Đặc trưng', 'Mô hình', 'Ngưỡng tau*', 'Precision', 'Recall', 'F1-Score']
    print("\n" + "=" * 85)
    print("      BẢNG SO SÁNH THỰC NGHIỆM: LAG FEATURES vs ZONE FEATURES (TUNED)")
    print("=" * 85)
    print(df_res.to_string(index=False))
    print("=" * 85)

if __name__ == "__main__":
    main()
