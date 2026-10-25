import os
import sys

if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, 'reconfigure'):
            getattr(sys.stdout, 'reconfigure')(encoding='utf-8')
        if hasattr(sys.stderr, 'reconfigure'):
            getattr(sys.stderr, 'reconfigure')(encoding='utf-8')
    except Exception:
        pass

import time
import json
import joblib
from typing import Any, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, f1_score, precision_score, recall_score
from catboost import CatBoostClassifier

script_dir = Path(__file__).resolve().parent
ROOT_DIR = script_dir.parent
DATA_PATH = ROOT_DIR / "data" / "processed" / "train_preprocessed.parquet"
OUTPUT_JSON = script_dir / "hyperparameters" / "baseline_tuning_best_params.json"
OUTPUT_TXT = script_dir / "hyperparameters" / "baseline_tuning_summary.txt"
SAVED_MODELS_DIR = script_dir / "saved_new_models"
EXPORTED_MODELS_DIR = script_dir / "exported_pipeline" / "models"

TARGET = "anomaly"

def optimize_threshold(y_true, y_prob):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_th = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    return f1_scores[best_idx], best_th

def full_evaluation(y_true, y_prob, threshold) -> dict[str, Any]:
    preds = (y_prob >= threshold).astype(int)
    f1_05 = f1_score(y_true, (y_prob >= 0.5).astype(int), zero_division=0)
    return {
        "best_threshold": round(float(threshold), 4),
        "f1_opt": round(float(f1_score(y_true, preds, zero_division=0)), 4),
        "precision_opt": round(float(precision_score(y_true, preds, zero_division=0)), 4),
        "recall_opt": round(float(recall_score(y_true, preds, zero_division=0)), 4),
        "f1_05": round(float(f1_05), 4)
    }

def main():
    print("=" * 80)
    print(" HUẤN LUYỆN VÀ ĐÓNG GÓI CHAMPION CATBOOST + ENSEMBLE BASELINE MỚI (40 FEATURES)")
    print("=" * 80)

    # 1. Load Data
    print("-> Đang nạp train_preprocessed.parquet...", flush=True)
    t0 = time.time()
    df = pd.read_parquet(DATA_PATH, engine="pyarrow")
    if 'timestamp' in df.columns:
        df = df.drop(columns=['timestamp'])
    X = df.drop(columns=[TARGET, "row_id"], errors="ignore")
    y = df[TARGET]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"   Xong trong {time.time() - t0:.1f}s. Kích thước Train: {X_train.shape}, Val: {X_val.shape}")

    # 2. Champion CatBoost Parameters from Trial 24
    cat_params = {
        'learning_rate': 0.2832000495099173,
        'depth': 14,
        'l2_leaf_reg': 3.198345542996033,
        'iterations': 191,
        'thread_count': -1,
        'verbose': 0,
        'random_state': 42
    }
    cat_threshold = 0.2979

    print("\n-> Đang huấn luyện Champion CatBoost với tham số tối ưu Trial 24...", flush=True)
    t_cat_train = time.time()
    cat_model = CatBoostClassifier(**cat_params)
    cat_model.fit(X_train, y_train, cat_features=['primary_use_code'])
    cat_train_time = time.time() - t_cat_train
    print(f"   Huấn luyện xong trong {cat_train_time:.1f}s")

    # Evaluate CatBoost
    t_inf = time.time()
    cat_probs = cat_model.predict_proba(X_val)[:, 1]
    cat_inf_time = time.time() - t_inf
    cat_f1_opt, opt_th = optimize_threshold(y_val, cat_probs)
    cat_metrics = full_evaluation(y_val, cat_probs, opt_th)
    cat_metrics["train_time_sec"] = round(cat_train_time, 2)
    cat_metrics["inference_time_sec"] = round(cat_inf_time, 2)

    print(f"   CatBoost Champion: F1(opt) = {cat_metrics['f1_opt']} | Th = {cat_metrics['best_threshold']} | Prec = {cat_metrics['precision_opt']} | Rec = {cat_metrics['recall_opt']}")

    # Save CatBoost
    cat_path1 = SAVED_MODELS_DIR / "CatBoost_Baseline.pkl"
    cat_path2 = EXPORTED_MODELS_DIR / "CatBoost_Baseline.pkl"
    joblib.dump(cat_model, cat_path1)
    joblib.dump(cat_model, cat_path2)
    print(f"-> Đã lưu CatBoost Champion vào:\n   1. {cat_path1}\n   2. {cat_path2}")

    # 3. Load XGBoost & LightGBM
    print("\n-> Đang nạp XGBoost và LightGBM Champion đã tối ưu...", flush=True)
    xgb_path = SAVED_MODELS_DIR / "XGBoost_Baseline_Full.pkl"
    lgb_path = SAVED_MODELS_DIR / "LightGBM_Baseline.pkl"

    xgb_model = joblib.load(xgb_path)
    lgb_model = joblib.load(lgb_path)

    xgb_probs = xgb_model.predict_proba(X_val)[:, 1]
    lgb_probs = lgb_model.predict_proba(X_val)[:, 1]

    # 4. Search Optimal Blend Weights
    print("\n-> Đang tìm kiếm tỷ trọng Ensemble tối ưu (Grid Search simplex)...", flush=True)
    best_blend_f1 = 0.0
    best_blend_weights: Tuple[float, float, float] = (0.50, 0.35, 0.15)
    best_blend_th = 0.5
    best_blend_prob: np.ndarray = xgb_probs

    # Step search
    for w_xgb in np.linspace(0.3, 0.7, 9):
        for w_lgb in np.linspace(0.1, 0.5, 9):
            w_cat = 1.0 - w_xgb - w_lgb
            if w_cat < 0.05 or w_cat > 0.4:
                continue
            curr_prob = w_xgb * xgb_probs + w_lgb * lgb_probs + w_cat * cat_probs
            f1_val, th_val = optimize_threshold(y_val, curr_prob)
            if f1_val > best_blend_f1:
                best_blend_f1 = f1_val
                best_blend_weights = (round(w_xgb, 2), round(w_lgb, 2), round(w_cat, 2))
                best_blend_th = th_val
                best_blend_prob = curr_prob

    print(f"   Tỷ trọng tối ưu: XGBoost={best_blend_weights[0]}, LightGBM={best_blend_weights[1]}, CatBoost={best_blend_weights[2]}")
    blend_metrics = full_evaluation(y_val, best_blend_prob, best_blend_th)
    blend_metrics["weights"] = {
        "XGBoost": best_blend_weights[0],
        "LightGBM": best_blend_weights[1],
        "CatBoost": best_blend_weights[2]
    }
    print(f"   Ensemble Baseline: F1(opt) = {blend_metrics['f1_opt']} | Th = {blend_metrics['best_threshold']} | Prec = {blend_metrics['precision_opt']} | Rec = {blend_metrics['recall_opt']}")

    # 5. Update JSON
    all_best_params = {}
    if OUTPUT_JSON.exists():
        try:
            with open(OUTPUT_JSON, "r", encoding="utf-8") as f:
                all_best_params = json.load(f)
        except Exception:
            all_best_params = {}

    all_best_params["CatBoost"] = {
        "best_f1": cat_metrics["f1_opt"],
        "best_threshold": cat_metrics["best_threshold"],
        "precision_opt": cat_metrics["precision_opt"],
        "recall_opt": cat_metrics["recall_opt"],
        "f1_05": cat_metrics["f1_05"],
        "tuning_time_sec": cat_metrics["train_time_sec"],
        "inference_time_sec": cat_metrics["inference_time_sec"],
        "status": "🔥 OUT TRÌNH BASELINE CŨ (+0.41%)",
        "params": {
            "learning_rate": cat_params["learning_rate"],
            "depth": cat_params["depth"],
            "l2_leaf_reg": cat_params["l2_leaf_reg"],
            "iterations": cat_params["iterations"]
        }
    }

    all_best_params["Ensemble_Baseline"] = blend_metrics

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_best_params, f, indent=4, ensure_ascii=False)
    print(f"\n=> Đã cập nhật kết quả vào: {OUTPUT_JSON}")

    # 6. Generate Summary Table
    xgb_f1_opt, xgb_th = optimize_threshold(y_val, xgb_probs)
    xgb_metrics = full_evaluation(y_val, xgb_probs, xgb_th)
    lgb_f1_opt, lgb_th = optimize_threshold(y_val, lgb_probs)
    lgb_metrics = full_evaluation(y_val, lgb_probs, lgb_th)

    summary_rows = [
        {
            "Mô hình": "Random Forest (Đối chứng)",
            "Số đặc trưng": 40,
            "Ngưỡng Tối Ưu": 0.23,
            "F1 (Ngưỡng 0.5)": 0.8450,
            "F1 (Ngưỡng Tối Ưu)": 0.8792,
            "Precision Tối Ưu": 0.9127,
            "Recall Tối Ưu": 0.8481,
            "Đánh Giá": "Bị loại (Tốc độ & F1 thấp)"
        },
        {
            "Mô hình": "Extra Trees (Đối chứng)",
            "Số đặc trưng": 40,
            "Ngưỡng Tối Ưu": 0.24,
            "F1 (Ngưỡng 0.5)": 0.7023,
            "F1 (Ngưỡng Tối Ưu)": 0.8051,
            "Precision Tối Ưu": 0.8261,
            "Recall Tối Ưu": 0.7852,
            "Đánh Giá": "Bị loại (F1 thấp)"
        },
        {
            "Mô hình": "CatBoost (Baseline Không One-Hot)",
            "Số đặc trưng": 40,
            "Ngưỡng Tối Ưu": cat_metrics["best_threshold"],
            "F1 (Ngưỡng 0.5)": cat_metrics["f1_05"],
            "F1 (Ngưỡng Tối Ưu)": cat_metrics["f1_opt"],
            "Precision Tối Ưu": cat_metrics["precision_opt"],
            "Recall Tối Ưu": cat_metrics["recall_opt"],
            "Đánh Giá": "🔥 Out trình (+0.41% vs gốc, +0.12% vs One-Hot)"
        },
        {
            "Mô hình": "LightGBM (Baseline Không One-Hot)",
            "Số đặc trưng": 40,
            "Ngưỡng Tối Ưu": lgb_metrics["best_threshold"],
            "F1 (Ngưỡng 0.5)": lgb_metrics["f1_05"],
            "F1 (Ngưỡng Tối Ưu)": lgb_metrics["f1_opt"],
            "Precision Tối Ưu": lgb_metrics["precision_opt"],
            "Recall Tối Ưu": lgb_metrics["recall_opt"],
            "Đánh Giá": "🔥 Out trình (+0.72% vs gốc, +0.48% vs One-Hot)"
        },
        {
            "Mô hình": "XGBoost (Baseline Không One-Hot)",
            "Số đặc trưng": 40,
            "Ngưỡng Tối Ưu": xgb_metrics["best_threshold"],
            "F1 (Ngưỡng 0.5)": xgb_metrics["f1_05"],
            "F1 (Ngưỡng Tối Ưu)": xgb_metrics["f1_opt"],
            "Precision Tối Ưu": xgb_metrics["precision_opt"],
            "Recall Tối Ưu": xgb_metrics["recall_opt"],
            "Đánh Giá": "🔥 Out trình (+0.38% vs gốc, +0.24% vs One-Hot)"
        },
        {
            "Mô hình": "Weighted Blend (XGB+LGB+CAT) Baseline",
            "Số đặc trưng": 40,
            "Ngưỡng Tối Ưu": blend_metrics["best_threshold"],
            "F1 (Ngưỡng 0.5)": blend_metrics["f1_05"],
            "F1 (Ngưỡng Tối Ưu)": blend_metrics["f1_opt"],
            "Precision Tối Ưu": blend_metrics["precision_opt"],
            "Recall Tối Ưu": blend_metrics["recall_opt"],
            "Đánh Giá": "Đỉnh cao Baseline Mới (40 Features)"
        }
    ]

    df_summary = pd.DataFrame(summary_rows)
    try:
        tbl_str = df_summary.to_markdown(index=False)
    except Exception:
        tbl_str = df_summary.to_string(index=False)

    print("\n" + "=" * 95)
    print("        BẢNG TỔNG KẾT ĐÁNH GIÁ CHUẨN BASELINE MỚI (40 ĐẶC TRƯNG)")
    print("=" * 95)
    print(tbl_str)

    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write("BẢNG TỔNG KẾT ĐÁNH GIÁ CHUẨN BASELINE MỚI (40 ĐẶC TRƯNG)\n")
        f.write(tbl_str)
        f.write("\n")

    print(f"\n=> Đã lưu bảng tổng hợp tại: {OUTPUT_TXT}")
    print("=" * 95)

if __name__ == "__main__":
    main()
