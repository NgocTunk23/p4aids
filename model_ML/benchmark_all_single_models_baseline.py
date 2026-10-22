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

# Kích hoạt tăng tốc phần cứng Intel oneAPI / AVX-512 cho Scikit-learn
try:
    from sklearnex import patch_sklearn
    patch_sklearn()
    sklearnex_active = True
except ImportError:
    sklearnex_active = False

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import precision_recall_curve, f1_score, precision_score, recall_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

N_CORES = os.cpu_count() or 4
TARGET = "anomaly"

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_BASELINE = BASE_DIR / "data" / "processed" / "train_preprocessed.parquet"
OUT_JSON = BASE_DIR / "model_ML" / "hyperparameters" / "single_models_baseline_benchmark.json"

def print_banner():
    print("=" * 90)
    print("      HUẤN LUYỆN ĐỐI CHỨNG TẤT CẢ 5 MÔ HÌNH ĐƠN (TUNED) TRÊN BASELINE ĐA LÕI")
    print("=" * 90)
    print(f"-> Hệ thống: CPU {N_CORES} Luồng | Intel oneAPI: {'Kích hoạt' if sklearnex_active else 'Không có'}")
    print(f"-> Tập dữ liệu: Baseline (train_preprocessed.parquet) | Phân chia: 80% Train / 20% Val (Stratified)")
    print(f"-> Mục tiêu: Đo đạc chính xác thời gian fit và F1 (ngưỡng 0.5 vs ngưỡng tối ưu tau*) trên cùng môi trường")
    print("=" * 90, flush=True)

def find_optimal_threshold(y_true, y_prob):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_th = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5
    best_f1 = float(f1_scores[best_idx])
    
    y_pred_opt = (y_prob >= best_th).astype(int)
    prec_opt = float(precision_score(y_true, y_pred_opt, zero_division=0))
    rec_opt = float(recall_score(y_true, y_pred_opt, zero_division=0))
    
    y_pred_05 = (y_prob >= 0.5).astype(int)
    f1_05 = float(f1_score(y_true, y_pred_05, zero_division=0))
    prec_05 = float(precision_score(y_true, y_pred_05, zero_division=0))
    rec_05 = float(recall_score(y_true, y_pred_05, zero_division=0))
    
    return {
        'optimal_threshold': round(best_th, 4),
        'f1_optimal': round(best_f1, 4),
        'precision_optimal': round(prec_opt, 4),
        'recall_optimal': round(rec_opt, 4),
        'f1_05': round(f1_05, 4),
        'precision_05': round(prec_05, 4),
        'recall_05': round(rec_05, 4)
    }

def main():
    print_banner()
    
    # 1. Nạp dữ liệu
    t0 = time.time()
    print("-> Đang nạp dữ liệu Baseline...", end=" ", flush=True)
    df = pd.read_parquet(DATA_BASELINE)
    if 'timestamp' in df.columns:
        df = df.drop(columns=['timestamp'])
    X = df.drop(columns=[TARGET, "row_id"], errors="ignore")
    y = df[TARGET]
    print(f"Xong trong {time.time() - t0:.1f}s. Kích thước: {X.shape}")
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Train: {X_train.shape[0]:,} dòng | Val: {X_val.shape[0]:,} dòng")
    print("-" * 90)
    
    # 2. Định nghĩa danh sách mô hình kèm Siêu tham số đã tối ưu bởi Optuna
    models_dict = {
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
        "CatBoost": CatBoostClassifier(
            learning_rate=0.262571031377097,
            depth=14,
            l2_leaf_reg=1.3969479625934749,
            iterations=178,
            thread_count=-1,
            random_seed=42,
            verbose=0
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=90,
            max_depth=22,
            min_samples_split=13,
            min_samples_leaf=9,
            max_features=0.41809091556421524,
            max_samples=0.6301321323053057,
            bootstrap=True,
            n_jobs=-1,
            random_state=42
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=90,
            max_depth=22,
            min_samples_split=13,
            min_samples_leaf=9,
            max_features=0.41809091556421524,
            max_samples=0.6301321323053057,
            bootstrap=True,
            n_jobs=-1,
            random_state=42
        )
    }
    
    results = []
    
    for name, model in models_dict.items():
        print(f"-> Đang huấn luyện [{name}]...", end=" ", flush=True)
        t_start = time.time()
        model.fit(X_train, y_train)
        fit_time = time.time() - t_start
        print(f"Xong trong {fit_time:5.2f}s.", flush=True)
        
        # Dự đoán
        y_prob = model.predict_proba(X_val)[:, 1]
        metrics = find_optimal_threshold(y_val, y_prob)
        metrics['model'] = name
        metrics['fit_time_sec'] = round(fit_time, 2)
        results.append(metrics)
        
        print(f"   Ngưỡng tau*: {metrics['optimal_threshold']:.4f} | "
              f"Precision: {metrics['precision_optimal']:.4f} | "
              f"Recall: {metrics['recall_optimal']:.4f} | "
              f"F1(0.5): {metrics['f1_05']:.4f} | "
              f"F1(tau*): {metrics['f1_optimal']:.4f}")
        print("-" * 90)

    # Lưu kết quả
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
        
    df_res = pd.DataFrame(results)[[
        'model', 'optimal_threshold', 'precision_optimal', 'recall_optimal', 
        'f1_05', 'f1_optimal', 'fit_time_sec'
    ]]
    df_res.columns = ['Mô hình', 'Ngưỡng tau*', 'Precision', 'Recall', 'F1 (0.5)', 'F1 (tau*)', 'Thời gian fit (s)']
    print("\n" + "=" * 90)
    print("           BẢNG TỔNG HỢP HIỆU NĂNG VÀ THỜI GIAN THỰC TẾ (BASELINE)")
    print("=" * 90)
    print(df_res.to_string(index=False))
    print("=" * 90)

if __name__ == "__main__":
    main()
