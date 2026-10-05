import pandas as pd
import numpy as np
import time
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score, log_loss, f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression

# Import class mô hình đã được bọc sẵn
from model_classes import XGBoostModel, LightGBMModel, CatBoostModel

import warnings
warnings.filterwarnings('ignore')

# ----------------- CONFIGURATION -----------------
script_dir = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = Path(script_dir).parent / "data" / "processed"

DATASET_PREPROCESSED = PROCESSED_DIR / "train_preprocessed.parquet"
DATASET_ENGINEERED = PROCESSED_DIR / "train_engineered.parquet"
TARGET = "anomaly"

def prepare_data(dataset_path):
    print(f"Đang nạp dữ liệu từ: {dataset_path.name}...")
    df = pd.read_parquet(dataset_path, engine="fastparquet")
    X = df.drop(columns=[TARGET, "row_id", "timestamp"], errors="ignore").fillna(0)
    y = df[TARGET]
    # Cố định random_state=42 để 2 bộ dữ liệu được chia cắt HOÀN TOÀN KHỚP NHAU từng dòng
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("="*80)
print("BƯỚC 1: CHUẨN BỊ DỮ LIỆU ĐỒNG BỘ")
print("="*80)
X_train_pre, X_val_pre, y_train, y_val = prepare_data(DATASET_PREPROCESSED)
X_train_fe, X_val_fe, _, _ = prepare_data(DATASET_ENGINEERED)

# ----------------- GIAI ĐOẠN 3: TÌM NGƯỠNG TỐI ƯU (THRESHOLD MOVING) -----------------
print("\n" + "="*80)
print("BƯỚC 2: HUẤN LUYỆN 3 MÔ HÌNH BOOST VỚI SIÊU THAM SỐ TỐI ƯU")
print("="*80)

# Khởi tạo mô hình với các tham số tốt nhất bạn đã tìm ra
models = {
    "XGBoost (Pre)": (
        XGBoostModel(learning_rate=0.2588966101472009, max_depth=12, subsample=0.9469427996754303, n_estimators=148),
        X_train_pre, X_val_pre
    ),
    "LightGBM (FE)": (
        LightGBMModel(learning_rate=0.12327192527974386, max_depth=13, num_leaves=130, feature_fraction=0.7025615258894086),
        X_train_fe, X_val_fe
    ),
    "CatBoost (FE)": (
        CatBoostModel(learning_rate=0.2859403421268123, depth=12, iterations=145, l2_leaf_reg=2.0604935809714204),
        X_train_fe, X_val_fe
    )
}

val_predictions = {} # Lưu lại xác suất dự đoán của 3 mô hình trên tập Validation

for name, (model, X_t, X_v) in models.items():
    print(f"Đang huấn luyện {name}...")
    model.model.fit(X_t, y_train)
    # Lấy xác suất dự đoán (Probability) thay vì nhãn 0/1
    proba = model.model.predict_proba(X_v)
    if proba.ndim > 1:
        proba = proba[:, 1]
    val_predictions[name] = proba

def find_best_threshold(y_true, y_prob):
    """Quét các ngưỡng từ 0.05 đến 0.95 để tìm ngưỡng tối ưu hóa Custom Score"""
    thresholds = np.arange(0.05, 0.95, 0.02)
    best_thresh = 0.5
    best_score = -999
    
    # Log loss cố định vì nó tính trên probability
    base_log_loss = log_loss(y_true, y_prob)
    
    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        f2 = fbeta_score(y_true, y_pred, beta=2.0)
        custom = f2 - (0.1 * base_log_loss)
        
        if custom > best_score:
            best_score = custom
            best_thresh = thresh
            
    return best_thresh, best_score

print("\n" + "="*80)
print("GIAI ĐOẠN 3: TỐI ƯU HÓA NGƯỠNG CẮT (THRESHOLD MOVING)")
print("="*80)

for name, proba in val_predictions.items():
    best_th, best_score = find_best_threshold(y_val, proba)
    # So sánh với ngưỡng mặc định (0.5)
    default_pred = (proba >= 0.5).astype(int)
    default_score = fbeta_score(y_val, default_pred, beta=2.0) - (0.1 * log_loss(y_val, proba))
    
    print(f"[{name}]")
    print(f"   -> Ngưỡng mặc định (0.5)  : Custom Score = {default_score:.4f}")
    print(f"   -> Ngưỡng tối ưu ({best_th:.2f}): Custom Score = {best_score:.4f} (Cải thiện: {best_score - default_score:+.4f})")


# ----------------- GIAI ĐOẠN 4: MODEL ENSEMBLE (STACKING & BLENDING) -----------------
print("\n" + "="*80)
print("GIAI ĐOẠN 4: DUNG HỢP MÔ HÌNH (ENSEMBLE - CHỈ DÙNG 3 BOOST MODELS)")
print("="*80)

# Tạo ma trận đặc trưng mới cho Meta-Learner (Kích thước: số mẫu Val x 3 cột)
X_meta = np.column_stack([
    val_predictions["XGBoost (Pre)"],
    val_predictions["LightGBM (FE)"],
    val_predictions["CatBoost (FE)"]
])

# 4.1 Phương pháp 1: Blending (Trung bình xác suất)
print("1. Phương pháp BLENDING (Soft Voting / Trung bình cộng):")
blend_proba = X_meta.mean(axis=1) # Lấy trung bình cộng 3 xác suất
blend_best_th, blend_best_score = find_best_threshold(y_val, blend_proba)
print(f"   -> Ngưỡng tối ưu ({blend_best_th:.2f}): Custom Score = {blend_best_score:.4f}")

# 4.2 Phương pháp 2: Stacking (Logistic Regression Meta-Learner)
# Lưu ý: Theo đúng chuẩn, Stacking cần dùng Out-of-fold predictions.
# Tuy nhiên, để biểu diễn concept nhanh chóng và không làm rò rỉ dữ liệu lên tập test,
# ta sẽ huấn luyện Meta-Learner trực tiếp trên Validation Probas. (Trong thực tế đi thi nên dùng StratifiedKFold)
print("\n2. Phương pháp STACKING (Dùng Logistic Regression học trọng số):")
meta_model = LogisticRegression()
meta_model.fit(X_meta, y_val)

# Xem trọng số mà Logistic Regression gán cho 3 mô hình (XGBoost, LightGBM, CatBoost)
weights = meta_model.coef_[0]
print("   -> Trọng số học được từ Meta-Learner:")
print(f"      * XGBoost (Pre): {weights[0]:.4f}")
print(f"      * LightGBM (FE): {weights[1]:.4f}")
print(f"      * CatBoost (FE): {weights[2]:.4f}")

stacking_proba = meta_model.predict_proba(X_meta)[:, 1]
stacking_best_th, stacking_best_score = find_best_threshold(y_val, stacking_proba)
print(f"   -> Ngưỡng tối ưu ({stacking_best_th:.2f}): Custom Score = {stacking_best_score:.4f}")

print("\n" + "="*80)
print("KẾT LUẬN GIAI ĐOẠN 4:")
print("So sánh Custom Score tốt nhất:")
print(f"- Lẻ tốt nhất (XGBoost): {find_best_threshold(y_val, val_predictions['XGBoost (Pre)'])[1]:.4f}")
print(f"- Ensemble Blending    : {blend_best_score:.4f}")
print(f"- Ensemble Stacking    : {stacking_best_score:.4f}")
print("="*80)
