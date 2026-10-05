import pandas as pd
import numpy as np
import time
import os
import json
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score, log_loss
from sklearn.linear_model import LogisticRegression

# Import class mô hình
from model_classes import XGBoostModel, LightGBMModel, CatBoostModel

import warnings
warnings.filterwarnings('ignore')

# ----------------- CONFIGURATION -----------------
script_dir = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = Path(script_dir).parent / "data" / "processed"

DATASET_PREPROCESSED = PROCESSED_DIR / "train_preprocessed.parquet"
DATASET_ENGINEERED = PROCESSED_DIR / "train_engineered.parquet"
TARGET = "anomaly"

# Thư mục lưu mô hình
SAVED_MODELS_DIR = os.path.join(script_dir, "saved_models")
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

def prepare_data(dataset_path):
    print(f"Đang nạp dữ liệu từ: {dataset_path.name}...")
    df = pd.read_parquet(dataset_path, engine="fastparquet")
    X = df.drop(columns=[TARGET, "row_id", "timestamp"], errors="ignore").fillna(0)
    y = df[TARGET]
    # Cố định random_state=42 để 2 bộ dữ liệu khớp nhau từng dòng
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("="*80)
print("BƯỚC 1: CHUẨN BỊ DỮ LIỆU ĐỒNG BỘ")
print("="*80)
X_train_pre, X_val_pre, y_train, y_val = prepare_data(DATASET_PREPROCESSED)
X_train_fe, X_val_fe, _, _ = prepare_data(DATASET_ENGINEERED)

def find_best_threshold(y_true, y_prob):
    """Quét các ngưỡng từ 0.05 đến 0.95 để tìm ngưỡng tối ưu hóa Custom Score"""
    thresholds = np.arange(0.01, 0.99, 0.02)
    best_thresh = 0.5
    best_score = -999
    
    base_log_loss = log_loss(y_true, y_prob)
    
    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        f2 = fbeta_score(y_true, y_pred, beta=2.0)
        custom = f2 - (0.1 * base_log_loss)
        if custom > best_score:
            best_score = custom
            best_thresh = thresh
            
    return best_thresh, best_score, base_log_loss

# =================================================================================
# ĐỊNH NGHĨA CÁC MÔ HÌNH VỚI SIÊU THAM SỐ ĐÃ TÌM ĐƯỢC
# =================================================================================
xgb_params = {'learning_rate': 0.2588966101472009, 'max_depth': 12, 'subsample': 0.9469427996754303, 'n_estimators': 148}
lgb_params = {'learning_rate': 0.12327192527974386, 'max_depth': 13, 'num_leaves': 130, 'feature_fraction': 0.7025615258894086}
cat_params = {'learning_rate': 0.2859403421268123, 'depth': 12, 'iterations': 145, 'l2_leaf_reg': 2.0604935809714204}

runs = [
    ("XGBoost", XGBoostModel(**xgb_params), "Preprocessed", X_train_pre, X_val_pre),
    ("XGBoost", XGBoostModel(**xgb_params), "Engineered", X_train_fe, X_val_fe),
    ("LightGBM", LightGBMModel(**lgb_params), "Preprocessed", X_train_pre, X_val_pre),
    ("LightGBM", LightGBMModel(**lgb_params), "Engineered", X_train_fe, X_val_fe),
    ("CatBoost", CatBoostModel(**cat_params), "Preprocessed", X_train_pre, X_val_pre),
    ("CatBoost", CatBoostModel(**cat_params), "Engineered", X_train_fe, X_val_fe)
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
    f2_default = fbeta_score(y_val, pred_default, beta=2.0)
    loss = log_loss(y_val, proba)
    score_default = f2_default - (0.1 * loss)
    
    # Tìm ngưỡng tối ưu
    best_th, best_score, _ = find_best_threshold(y_val, proba)
    
    final_results.append({
        "Giai đoạn": "GĐ 2-3: Mô hình Đơn",
        "Mô hình": model_name,
        "Dữ liệu": ds_name,
        "Ngưỡng Tối Ưu": round(best_th, 2),
        "Custom Score (Ngưỡng 0.5)": round(score_default, 4),
        "Custom Score (Ngưỡng Tối Ưu)": round(best_score, 4),
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


print("\n" + "="*80)
print("BƯỚC 3: GIAI ĐOẠN 4 - MODEL ENSEMBLE")
print("="*80)

# Để Ensemble mạnh nhất, ta kết hợp bản xuất sắc nhất của từng mô hình:
# 1. XGBoost (Pre)
# 2. LightGBM (FE)
# 3. CatBoost (FE)
p_xgb = val_probas["XGBoost_Preprocessed"]
p_lgb = val_probas["LightGBM_Engineered"]
p_cat = val_probas["CatBoost_Engineered"]

def evaluate_ensemble(name, proba_matrix, y_true):
    # Blending (Trung bình)
    blend_proba = proba_matrix.mean(axis=1)
    
    score_default = fbeta_score(y_true, (blend_proba >= 0.5).astype(int), beta=2.0) - (0.1 * log_loss(y_true, blend_proba))
    best_th, best_score, _ = find_best_threshold(y_true, blend_proba)
    
    final_results.append({
        "Giai đoạn": "GĐ 4: Ensemble",
        "Mô hình": f"{name} (Blending)",
        "Dữ liệu": "Hỗn hợp",
        "Ngưỡng Tối Ưu": round(best_th, 2),
        "Custom Score (Ngưỡng 0.5)": round(score_default, 4),
        "Custom Score (Ngưỡng Tối Ưu)": round(best_score, 4),
        "Tốc độ (s)": 0.0
    })
    
    # Stacking (Logistic Regression)
    meta = LogisticRegression()
    meta.fit(proba_matrix, y_true)
    stack_proba = meta.predict_proba(proba_matrix)[:, 1]
    
    score_default_s = fbeta_score(y_true, (stack_proba >= 0.5).astype(int), beta=2.0) - (0.1 * log_loss(y_true, stack_proba))
    best_th_s, best_score_s, _ = find_best_threshold(y_true, stack_proba)
    
    final_results.append({
        "Giai đoạn": "GĐ 4: Ensemble",
        "Mô hình": f"{name} (Stacking)",
        "Dữ liệu": "Hỗn hợp",
        "Ngưỡng Tối Ưu": round(best_th_s, 2),
        "Custom Score (Ngưỡng 0.5)": round(score_default_s, 4),
        "Custom Score (Ngưỡng Tối Ưu)": round(best_score_s, 4),
        "Tốc độ (s)": 0.0
    })
    
    # --- LƯU META-LEARNER VÀ NGƯỠNG CHO ENSEMBLE ---
    meta_path = os.path.join(SAVED_MODELS_DIR, f"MetaLearner_{name.replace(' + ', '_')}.pkl")
    joblib.dump(meta, meta_path)
    optimal_thresholds[f"Ensemble_Blending_{name}"] = round(best_th, 4)
    optimal_thresholds[f"Ensemble_Stacking_{name}"] = round(best_th_s, 4)
    print(f"Hoàn tất cấu hình: {name} (Đã lưu Meta-Learner)")

# Kịch bản 1: Cả 3 mô hình
evaluate_ensemble("XGB + LightGBM + CatBoost", np.column_stack([p_xgb, p_lgb, p_cat]), y_val)

# Kịch bản 2: Chỉ LightGBM + CatBoost (FE)
evaluate_ensemble("LightGBM + CatBoost", np.column_stack([p_lgb, p_cat]), y_val)

print("\n" + "="*90)
print("BẢNG TỔNG HỢP TOÀN DIỆN MỌI GIAI ĐOẠN (Từ Pre, FE đến Threshold & Ensemble)")
print("="*90)

results_df = pd.DataFrame(final_results)
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
    