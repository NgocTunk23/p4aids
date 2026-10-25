"""
Thực nghiệm Đối chứng Độc lập:
So sánh Hiệu năng giữa Tập Baseline Cũ (One-Hot 13 cột primary_use)
và Tập Baseline Mới (1 cột Số nguyên primary_use_code) trên cùng mô hình đã Tuning từ Optuna.
"""

import os
import sys
import time
import json
import warnings
from pathlib import Path
from typing import Any, Dict

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
DATA_BASE_OLD = BASE_DIR / "data" / "processed" / "train_preprocessed.parquet"
DATA_BASE_NEW = BASE_DIR / "data" / "processed" / "train_preprocessed_single_cat.parquet"
OUT_JSON = BASE_DIR / "model_ML" / "hyperparameters" / "baseline_single_cat_vs_onehot_comparison.json"


def create_baseline_single_cat_dataset():
    """Tạo và lưu trữ tập dữ liệu Baseline mới với 1 cột primary_use_code thay vì 13 cột One-Hot."""
    print("=" * 95)
    print("BƯỚC 1: KHỞI TẠO TẬP DỮ LIỆU BASELINE MỚI (CHUYỂN ONE-HOT THÀNH 1 CỘT SỐ NGUYÊN)")
    print("=" * 95)
    
    t0 = time.time()
    print("-> Đang đọc dữ liệu train_preprocessed.parquet...", end=" ", flush=True)
    df_train = pd.read_parquet(DATA_BASE_OLD)
    print(f"Xong ({len(df_train):,} dòng, {len(df_train.columns)} cột).")
    
    pu_cols = [c for c in df_train.columns if c.startswith('primary_use_')]
    category_names = [c.replace('primary_use_', '') for c in pu_cols]
    cat_map = {c: i for i, c in enumerate(category_names)}
    
    # Nghịch đảo One-Hot về 1 cột số nguyên
    primary_use_series = df_train[pu_cols].idxmax(axis=1).map(lambda x: cat_map[x.replace('primary_use_', '')]).astype('int8')
    
    df_train_new = df_train.drop(columns=pu_cols).copy()
    df_train_new['primary_use_code'] = primary_use_series
    
    DATA_BASE_NEW.parent.mkdir(parents=True, exist_ok=True)
    df_train_new.to_parquet(DATA_BASE_NEW, index=False)
    print(f"-> Đã xuất thành công: {DATA_BASE_NEW.name}")
    print(f"   Kích thước mới: {df_train_new.shape} (giảm từ 53 cột xuống còn {len(df_train_new.columns)} cột, gồm 40 đặc trưng + 1 nhãn)")
    
    # Xử lý tương tự cho tập test nếu tồn tại
    test_old = BASE_DIR / "data" / "processed" / "test_preprocessed.parquet"
    test_new = BASE_DIR / "data" / "processed" / "test_preprocessed_single_cat.parquet"
    if test_old.exists():
        df_test = pd.read_parquet(test_old)
        pu_test = [c for c in df_test.columns if c.startswith('primary_use_')]
        test_pu_series = df_test[pu_test].idxmax(axis=1).map(lambda x: cat_map[x.replace('primary_use_', '')]).astype('int8')
        df_test_new = df_test.drop(columns=pu_test).copy()
        df_test_new['primary_use_code'] = test_pu_series
        df_test_new.to_parquet(test_new, index=False)
        print(f"-> Đã đồng bộ xuất tập test: {test_new.name} ({df_test_new.shape})")
    
    print(f"Thời gian tạo dữ liệu: {time.time() - t0:.2f}s.\n", flush=True)
    return cat_map


def find_optimal_threshold(y_true, y_prob) -> Dict[str, Any]:
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


def get_models():
    """Tải siêu tham số tối ưu từ Optuna trên tập Baseline."""
    return {
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
        )
    }


def train_and_eval_dataset(dataset_path: Path, dataset_label: str):
    print("=" * 95)
    print(f"BƯỚC ĐÁNH GIÁ: {dataset_label}")
    print(f"File: {dataset_path.name}")
    print("=" * 95)
    
    df = pd.read_parquet(dataset_path)
    if 'timestamp' in df.columns:
        df = df.drop(columns=['timestamp'])
    X = df.drop(columns=[TARGET, "row_id"], errors="ignore")
    y = df[TARGET]
    
    # Cần format lại tên cột cho LightGBM nếu có khoảng trắng
    X_lgb = X.rename(columns=lambda col: str(col).replace(' ', '_'))
    
    print(f"-> Số chiều đặc trưng đầu vào: {X.shape[1]} đặc trưng | Tổng mẫu: {X.shape[0]:,}")
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train_lgb, X_val_lgb, _, _ = train_test_split(
        X_lgb, y, test_size=0.2, random_state=42, stratify=y
    )
    
    results = {}
    prob_dict = {}
    
    models = get_models()
    for name, model in models.items():
        print(f"-> Đang huấn luyện [{name}]...", end=" ", flush=True)
        t_start = time.time()
        
        if name == "LightGBM":
            model.fit(X_train_lgb, y_train)
            raw_prob = model.predict_proba(X_val_lgb)
        else:
            model.fit(X_train, y_train)
            raw_prob = model.predict_proba(X_val)
            
        fit_time = time.time() - t_start
        y_prob = np.asarray(raw_prob)[:, 1]
        prob_dict[name] = y_prob
        
        metrics = find_optimal_threshold(y_val, y_prob)
        metrics['fit_time_sec'] = round(fit_time, 2)
        results[name] = metrics
        
        print(f"Xong trong {fit_time:5.2f}s | tau*: {metrics['optimal_threshold']:.4f} | "
              f"Prec: {metrics['precision_optimal']:.4f} | Rec: {metrics['recall_optimal']:.4f} | "
              f"F1(0.5): {metrics['f1_05']:.4f} | F1(tau*): {metrics['f1_optimal']:.4f}")

    # Đánh giá thêm Weighted Blending Baseline (0.68 XGB + 0.26 LGB + 0.06 Cat)
    print("-> Đánh giá mô hình Dung hợp Weighted Blending [0.68 XGB + 0.26 LGB + 0.06 Cat]...", end=" ", flush=True)
    blend_prob = 0.68 * prob_dict["XGBoost"] + 0.26 * prob_dict["LightGBM"] + 0.06 * prob_dict["CatBoost"]
    blend_metrics = find_optimal_threshold(y_val, blend_prob)
    blend_metrics['fit_time_sec'] = round(sum(results[m]['fit_time_sec'] for m in results), 2)
    results["Weighted Blending"] = blend_metrics
    print(f"Xong | tau*: {blend_metrics['optimal_threshold']:.4f} | "
          f"Prec: {blend_metrics['precision_optimal']:.4f} | Rec: {blend_metrics['recall_optimal']:.4f} | "
          f"F1(0.5): {blend_metrics['f1_05']:.4f} | F1(tau*): {blend_metrics['f1_optimal']:.4f}\n")
    
    return results


def main():
    cat_map = create_baseline_single_cat_dataset()
    
    # 1. Chạy đánh giá trên tập Baseline Cũ (One-Hot 13 cột)
    res_onehot = train_and_eval_dataset(
        DATA_BASE_OLD, 
        "TẬP BASELINE CŨ: MÃ HÓA ONE-HOT (52 ĐẶC TRƯNG, 13 CỘT primary_use_)"
    )
    
    # 2. Chạy đánh giá trên tập Baseline Mới (1 cột Số nguyên primary_use_code)
    res_single = train_and_eval_dataset(
        DATA_BASE_NEW, 
        "TẬP BASELINE MỚI: MÃ HÓA 1 CỘT SỐ NGUYÊN (40 ĐẶC TRƯNG, primary_use_code)"
    )
    
    # 3. Tạo bảng tổng hợp so sánh trực tiếp
    print("=" * 115)
    print("                       BẢNG ĐỐI CHỨNG TOÀN DIỆN HIỆU NĂNG VÀ THỜI GIAN FIT")
    print("               BASELINE CŨ (ONE-HOT 13 CỘT)  vs  BASELINE MỚI (1 CỘT SỐ NGUYÊN primary_use_code)")
    print("=" * 115)
    
    print(f"{'Mô hình':<20} | {'Dạng mã hóa':<15} | {'Số cột':<7} | {'Fit time':<9} | {'tau*':<7} | {'Prec':<7} | {'Rec':<7} | {'F1(0.5)':<8} | {'F1(tau*)':<8} | {'Delta F1':<10}")
    print("-" * 115)
    
    comparison_data = []
    
    for m in ["XGBoost", "LightGBM", "CatBoost", "Weighted Blending"]:
        oh = res_onehot[m]
        sc = res_single[m]
        delta_f1 = sc['f1_optimal'] - oh['f1_optimal']
        delta_pct = (delta_f1 / oh['f1_optimal']) * 100
        sign = "+" if delta_f1 >= 0 else ""
        delta_str = f"{sign}{delta_f1:.4f} ({sign}{delta_pct:.2f}%)"
        
        # Dòng One-hot
        print(f"{m:<20} | {'One-Hot (Cũ)':<15} | {'52':<7} | {oh['fit_time_sec']:<7.1f}s | {oh['optimal_threshold']:<7.4f} | {oh['precision_optimal']:<7.4f} | {oh['recall_optimal']:<7.4f} | {oh['f1_05']:<8.4f} | {oh['f1_optimal']:<8.4f} | {'Mốc chuẩn':<10}")
        # Dòng Single Cat
        print(f"{'':<20} | {'1 Cột Code':<15} | {'40':<7} | {sc['fit_time_sec']:<7.1f}s | {sc['optimal_threshold']:<7.4f} | {sc['precision_optimal']:<7.4f} | {sc['recall_optimal']:<7.4f} | {sc['f1_05']:<8.4f} | {sc['f1_optimal']:<8.4f} | {delta_str:<10}")
        print("-" * 115)
        
        comparison_data.append({
            "model": m,
            "one_hot_52": oh,
            "single_cat_40": sc,
            "delta_f1": round(delta_f1, 4),
            "delta_f1_pct": round(delta_pct, 2)
        })
        
    # Ghi ra JSON
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "category_mapping": cat_map,
            "comparison": comparison_data
        }, f, indent=4, ensure_ascii=False)
    print(f"\n-> Đã lưu toàn bộ kết quả so sánh tại: {OUT_JSON}")


if __name__ == "__main__":
    main()
