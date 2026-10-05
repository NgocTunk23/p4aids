import pandas as pd
import numpy as np
import time
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score, log_loss
from imblearn.over_sampling import SMOTE
from model_classes import XGBoostModel
import warnings
warnings.filterwarnings('ignore')

# ----------------- CONFIGURATION -----------------
script_dir = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = Path(script_dir).parent / "data" / "processed"
DATASET_PREPROCESSED = PROCESSED_DIR / "train_preprocessed.parquet"
TARGET = "anomaly"

def find_best_threshold(y_true, y_prob):
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

print("="*80)
print("BƯỚC 1: NẠP DỮ LIỆU")
print("="*80)
print("Đang nạp dữ liệu Preprocessed (Vì XGBoost hoạt động tốt nhất trên tập này)...")
df = pd.read_parquet(DATASET_PREPROCESSED, engine="fastparquet")
X = df.drop(columns=[TARGET, "row_id", "timestamp"], errors="ignore").fillna(0)
y = df[TARGET]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("\n" + "="*80)
print("BƯỚC 2: ÁP DỤNG SMOTE ĐỂ CÂN BẰNG DỮ LIỆU")
print("="*80)
print("Đang chạy thuật toán SMOTE để sinh dữ liệu nhân tạo cho nhóm thiểu số...")
print("Cảnh báo: Quá trình này có thể tốn khá nhiều RAM và CPU!")
start_smote = time.time()
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print(f"-> Hoàn tất SMOTE trong {time.time() - start_smote:.1f}s")
print(f"Kích thước tập Train GỐC : {X_train.shape[0]:,} dòng (Nhãn 1: {sum(y_train==1):,} dòng)")
print(f"Kích thước tập Train SMOTE: {X_train_smote.shape[0]:,} dòng (Nhãn 1: {sum(y_train_smote==1):,} dòng)")


print("\n" + "="*80)
print("BƯỚC 3: HUẤN LUYỆN VÀ SO SÁNH XGBOOST")
print("="*80)
xgb_params = {'learning_rate': 0.2588966101472009, 'max_depth': 12, 'subsample': 0.9469427996754303, 'n_estimators': 148}

results = []

for name, X_t, y_t in [("Mặc định (Không SMOTE)", X_train, y_train), ("Có áp dụng SMOTE", X_train_smote, y_train_smote)]:
    print(f"Đang huấn luyện XGBoost - {name}...")
    start_time = time.time()
    
    # Khởi tạo mô hình
    model = XGBoostModel(**xgb_params)
    model.model.fit(X_t, y_t)
    train_time = time.time() - start_time
    
    # Dự đoán trên TẬP VALIDATION (Tập Val hoàn toàn không bị dính dáng gì tới SMOTE)
    proba = model.model.predict_proba(X_val)
    if proba.ndim > 1: 
        proba = proba[:, 1]
    
    # Đánh giá ở ngưỡng mặc định 0.5
    pred_default = (proba >= 0.5).astype(int)
    score_default = fbeta_score(y_val, pred_default, beta=2.0) - (0.1 * log_loss(y_val, proba))
    
    # Tìm ngưỡng tối ưu
    best_th, best_score, _ = find_best_threshold(y_val, proba)
    
    results.append({
        "Phiên bản XGBoost": name,
        "Ngưỡng Tối Ưu": round(best_th, 2),
        "Custom Score (Ngưỡng 0.5)": round(score_default, 4),
        "Custom Score (Ngưỡng Tối Ưu)": round(best_score, 4),
        "Thời gian Train (s)": round(train_time, 1)
    })

results_df = pd.DataFrame(results)

print("\n" + "="*85)
print("BẢNG KẾT QUẢ: SO SÁNH HIỆU QUẢ CỦA SMOTE LÊN MÔ HÌNH XGBOOST")
print("="*85)
print(results_df.to_markdown(index=False))

out_md = os.path.join(script_dir, "smote_comparison_report.md")
with open(out_md, "w", encoding="utf-8") as f:
    f.write("# BẢNG KẾT QUẢ: SO SÁNH HIỆU QUẢ CỦA SMOTE LÊN MÔ HÌNH XGBOOST\n\n")
    f.write(results_df.to_markdown(index=False))
    f.write("\n")

print(f"Đã lưu báo cáo Markdown vào: {out_md}")
