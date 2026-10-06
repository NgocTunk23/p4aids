import pandas as pd
import numpy as np
import time
import os
import json
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, log_loss, recall_score, precision_score
from sklearn.linear_model import LogisticRegression

# Import class mô hình
from model_classes import XGBoostModel, LightGBMModel, CatBoostModel, RandomForestModel, ExtraTreesModel

import warnings
warnings.filterwarnings('ignore')

# ----------------- CONFIGURATION -----------------
script_dir = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = Path(script_dir).parent / "data" / "processed"

DATASET_PREPROCESSED = PROCESSED_DIR / "train_preprocessed.parquet"
DATASET_IF = PROCESSED_DIR / "train_IF.parquet"
TARGET = "anomaly"

# Thư mục lưu mô hình
SAVED_MODELS_DIR = os.path.join(script_dir, "saved_models")
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

def prepare_data(dataset_path):
    print(f"Đang nạp dữ liệu từ: {dataset_path.name}...")
    df = pd.read_parquet(dataset_path, engine="fastparquet")
    X = df.drop(columns=[TARGET, "row_id", "timestamp"], errors="ignore").fillna(0)
    y = df[TARGET]
    
    y = df[TARGET]
    # Chia ngẫu nhiên để đảm bảo tập Train bao phủ đủ 4 mùa trong 1 năm duy nhất của dataset
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("="*80)
print("BƯỚC 1: CHUẨN BỊ DỮ LIỆU ĐỒNG BỘ")
print("="*80)
X_train_pre, X_val_pre, y_train, y_val = prepare_data(DATASET_PREPROCESSED)
X_train_fe, X_val_fe, _, _ = prepare_data(DATASET_IF)

def find_best_threshold(y_true, y_prob):
    """Quét các ngưỡng từ 0.05 đến 0.95 để tìm ngưỡng tối ưu hóa F1 Score"""
    thresholds = np.arange(0.01, 0.99, 0.02)
    best_thresh = 0.5
    best_score = -999
    
    base_log_loss = log_loss(y_true, y_prob)
    
    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_score:
            best_score = f1
            best_thresh = thresh
            
    return best_thresh, best_score, base_log_loss

# =================================================================================
# ĐỊNH NGHĨA CÁC MÔ HÌNH VỚI SIÊU THAM SỐ ĐÃ TÌM ĐƯỢC
# =================================================================================
xgb_params = {'learning_rate': 0.23187303509421126, 'max_depth': 15, 'subsample': 0.9524355862143259, 'n_estimators': 141}
lgb_params = {'learning_rate': 0.0859101054837918, 'num_leaves': 147, 'max_depth': 15, 'feature_fraction': 0.6267447348759597}
cat_params = {'learning_rate': 0.3282464476846472, 'depth': 14, 'iterations': 159, 'l2_leaf_reg': 1.483228351433837}    
rf_params = {'n_estimators': 142, 'random_state': 42, 'n_jobs': -1}
et_params = {'n_estimators': 142, 'random_state': 42, 'n_jobs': -1}

runs = [
    ("XGBoost", XGBoostModel(**xgb_params), "Preprocessed", X_train_pre, X_val_pre),
    ("XGBoost", XGBoostModel(**xgb_params), "IF", X_train_fe, X_val_fe),
    ("LightGBM", LightGBMModel(**lgb_params), "Preprocessed", X_train_pre, X_val_pre),
    ("LightGBM", LightGBMModel(**lgb_params), "IF", X_train_fe, X_val_fe),
    ("CatBoost", CatBoostModel(**cat_params), "Preprocessed", X_train_pre, X_val_pre),
    ("CatBoost", CatBoostModel(**cat_params), "IF", X_train_fe, X_val_fe),
    ("Random Forest", RandomForestModel(**rf_params), "Preprocessed", X_train_pre, X_val_pre),
    ("Random Forest", RandomForestModel(**rf_params), "IF", X_train_fe, X_val_fe),
    ("Extra Trees", ExtraTreesModel(**et_params), "Preprocessed", X_train_pre, X_val_pre),
    ("Extra Trees", ExtraTreesModel(**et_params), "IF", X_train_fe, X_val_fe)
]

print("\n" + "="*80)
print("BƯỚC 2: HUẤN LUYỆN VÀ ĐÁNH GIÁ (CÓ NGƯỠNG VÀ KHÔNG NGƯỠNG)")
print("="*80)

final_results = []
val_probas = {} # Lưu xác suất để dùng cho Giai đoạn 4 Ensemble
optimal_thresholds = {} # Lưu lại ngưỡng tối ưu

for model_name, model_obj, ds_name, X_t, X_v in runs:
    print(f"Đang huấn luyện {model_name} trên {ds_name}...")
    start_time = time.time()
    model_obj.model.fit(X_t, y_train)
    
    proba = model_obj.model.predict_proba(X_v)
    if proba.ndim > 1:
        proba = proba[:, 1]
        
    train_time = time.time() - start_time
    
    # Đánh giá ngưỡng mặc định (0.5)
    pred_default = (proba >= 0.5).astype(int)
    score_default = f1_score(y_val, pred_default, zero_division=0)
    recall_default = recall_score(y_val, pred_default)
    precision_default = precision_score(y_val, pred_default, zero_division=0)
    
    # Tìm ngưỡng tối ưu
    best_th, best_score, _ = find_best_threshold(y_val, proba)
    pred_opt = (proba >= best_th).astype(int)
    recall_opt = recall_score(y_val, pred_opt)
    precision_opt = precision_score(y_val, pred_opt, zero_division=0)
    
    final_results.append({
        "Giai đoạn": "GĐ 2-3: Mô hình Đơn",
        "Mô hình": model_name,
        "Dữ liệu": ds_name,
        "Ngưỡng Tối Ưu": round(best_th, 2),
        "F1 Score (Ngưỡng 0.5)": round(score_default, 4),
        "F1 Score (Ngưỡng Tối Ưu)": round(best_score, 4),
        "Recall (Ngưỡng 0.5)": round(recall_default, 4),
        "Recall (Ngưỡng Tối Ưu)": round(recall_opt, 4),
        "Precision (Ngưỡng 0.5)": round(precision_default, 4),
        "Precision (Ngưỡng Tối Ưu)": round(precision_opt, 4),
        "Tốc độ (s)": round(train_time, 1)
    })
    
    # Lưu lại xác suất cho Giai đoạn 4
    key = f"{model_name}_{ds_name}"
    val_probas[key] = proba
    
    # --- LƯU MÔ HÌNH VÀ NGƯỠNG ---
    model_path = os.path.join(SAVED_MODELS_DIR, f"{key}.pkl")
    joblib.dump(model_obj.model, model_path)
    optimal_thresholds[key] = round(best_th, 4)
    print(f"   -> Đã lưu mô hình tại: {model_path}")


# Đã chuyển Giai đoạn 4 sang các script chuyên biệt (run_advanced_ensemble.py, run_oof_blending.py, run_moe_ensemble.py)

print("\n" + "="*90)
print("BẢNG TỔNG HỢP TOÀN DIỆN MỌI GIAI ĐOẠN (Từ Pre, FE đến Threshold & Ensemble)")
print("="*90)

results_df = pd.DataFrame(final_results)
results_df = results_df.sort_values(by=["F1 Score (Ngưỡng Tối Ưu)"], ascending=False)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

markdown_table = results_df.to_markdown(index=False)
print(markdown_table)

# --- LƯU TOÀN BỘ NGƯỠNG VÀO JSON ---
thresh_path = os.path.join(SAVED_MODELS_DIR, "optimal_thresholds.json")
with open(thresh_path, "w", encoding="utf-8") as f:
    json.dump(optimal_thresholds, f, indent=4)

out_md = os.path.join(script_dir, "final_comprehensive_report.md")

with open(out_md, "w", encoding="utf-8") as f:
    f.write("# BẢNG TỔNG HỢP TOÀN DIỆN MỌI GIAI ĐOẠN (Từ Pre, FE đến Threshold & Ensemble)\n\n")
    f.write(markdown_table)
    f.write("\n")

print(f"Đã lưu báo cáo Markdown vào: {out_md}")
    