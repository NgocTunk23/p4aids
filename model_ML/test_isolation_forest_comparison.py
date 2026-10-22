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

# Kích hoạt tăng tốc phần cứng Intel oneAPI / AVX cho Scikit-learn
try:
    from sklearnex import patch_sklearn
    patch_sklearn()
    sklearnex_active = True
except ImportError:
    sklearnex_active = False

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_curve, f1_score, precision_score, recall_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

TARGET = "anomaly"
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_BASELINE = BASE_DIR / "data" / "processed" / "train_preprocessed.parquet"
OUT_JSON = BASE_DIR / "model_ML" / "hyperparameters" / "isolation_forest_test_results.json"

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

def get_tuned_models():
    """Bộ siêu tham số ĐÃ TUNING từ Optuna cho Baseline"""
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
            thread_count=-1,
            random_seed=42,
            verbose=0
        )
    }

def main():
    print("=" * 85)
    print("   THỰC NGHIỆM ĐÁNH GIÁ ĐẶC TRƯNG ISOLATION FOREST (IF) VỚI MÔ HÌNH ĐÃ TUNING")
    print("=" * 85)
    print("-> Sử dụng siêu tham số ĐÃ TUNING TỪ OPTUNA (Baseline Tuned)")
    print("-> Quét ngưỡng tối ưu tau* để tối đa hóa F1-Score")
    print("-" * 85, flush=True)

    # 1. Nạp dữ liệu
    df = pd.read_parquet(DATA_BASELINE)
    if 'timestamp' in df.columns:
        df = df.drop(columns=['timestamp'])
    X = df.drop(columns=[TARGET, "row_id"], errors="ignore")
    y = df[TARGET]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"-> Train: {X_train.shape[0]:,} dòng | Val: {X_val.shape[0]:,} dòng")

    # 2. Huấn luyện Isolation Forest và tạo meta-feature
    print("-> Đang trích xuất meta-feature từ Isolation Forest...", end=" ", flush=True)
    t0 = time.time()
    iso = IsolationForest(n_estimators=100, max_samples=512, contamination=0.0213, random_state=42, n_jobs=-1)
    iso.fit(X_train)
    
    if_score_train = -iso.score_samples(X_train)
    if_score_val = -iso.score_samples(X_val)
    print(f"Xong trong {time.time() - t0:.1f}s.")

    X_train_if = X_train.copy()
    X_train_if['if_score'] = if_score_train

    X_val_if = X_val.copy()
    X_val_if['if_score'] = if_score_val

    results = []

    # 3. Huấn luyện và đánh giá đối chứng
    for name in ["XGBoost", "LightGBM", "CatBoost"]:
        # A. Preprocessed (Không có IF)
        models_base = get_tuned_models()
        clf_base = models_base[name]
        print(f"-> Huấn luyện [{name}] trên Preprocessed (Tuned)...", end=" ", flush=True)
        t_start = time.time()
        clf_base.fit(X_train, y_train)
        y_prob_base = clf_base.predict_proba(X_val)[:, 1]
        m_base = find_optimal_threshold(y_val, y_prob_base)
        m_base['model'] = name
        m_base['dataset'] = 'Preprocessed'
        results.append(m_base)
        print(f"Xong {time.time() - t_start:.1f}s | tau*: {m_base['optimal_threshold']:.4f} | Prec: {m_base['precision']:.4f} | Rec: {m_base['recall']:.4f} | F1: {m_base['f1_score']:.4f}")

        # B. Tích hợp IF
        models_if = get_tuned_models()
        clf_if = models_if[name]
        print(f"-> Huấn luyện [{name}] trên Tích hợp IF (Tuned)...", end=" ", flush=True)
        t_start = time.time()
        clf_if.fit(X_train_if, y_train)
        y_prob_if = clf_if.predict_proba(X_val_if)[:, 1]
        m_if = find_optimal_threshold(y_val, y_prob_if)
        m_if['model'] = name
        m_if['dataset'] = 'Tích hợp IF'
        results.append(m_if)
        print(f"Xong {time.time() - t_start:.1f}s | tau*: {m_if['optimal_threshold']:.4f} | Prec: {m_if['precision']:.4f} | Rec: {m_if['recall']:.4f} | F1: {m_if['f1_score']:.4f}")
        print("-" * 85)

    # 4. Lưu ra file JSON
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    # 5. Hiển thị bảng kết quả (không có cột thời gian theo yêu cầu)
    df_res = pd.DataFrame(results)[['model', 'dataset', 'optimal_threshold', 'precision', 'recall', 'f1_score']]
    df_res.columns = ['Mô hình', 'Dữ liệu', 'Ngưỡng tau*', 'Precision', 'Recall', 'F1-Score']
    print("\n" + "=" * 85)
    print("      BẢNG SO SÁNH HIỆU NĂNG TRƯỚC VÀ SAU KHI TÍCH HỢP ISOLATION FOREST (IF)")
    print("=" * 85)
    print(df_res.to_string(index=False))
    print("=" * 85)

if __name__ == "__main__":
    main()
