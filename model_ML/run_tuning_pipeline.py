import pandas as pd
import numpy as np
import time
import os
import optuna
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score, log_loss
from model_classes import XGBoostModel, LightGBMModel, CatBoostModel, RandomForestModel, ExtraTreesModel
import warnings
warnings.filterwarnings('ignore')

# ----------------- CONFIGURATION -----------------
script_dir = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = Path(script_dir).parent / "data" / "processed"

# Nạp cả 2 bộ dữ liệu để phục vụ chiến thuật ghép cặp
DATASET_PREPROCESSED = PROCESSED_DIR / "train_preprocessed.parquet"
DATASET_ENGINEERED = PROCESSED_DIR / "train_engineered.parquet"
TARGET = "anomaly"

# Số lượng mẫu dùng để Tuning (Nên dùng số nhỏ để chạy nhanh, ví dụ 200k - 500k dòng)
TUNING_SAMPLE_SIZE = None
N_TRIALS = 30 # Số lần thử nghiệm cấu hình cho mỗi mô hình

def objective(trial, model_name, X_train, y_train, X_val, y_val):
    """
    Định nghĩa không gian tìm kiếm (Search Space) siêu tham số cho từng loại mô hình
    """
    # if model_name == "XGBoost":
    #     params = {
    #         'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
    #         'max_depth': trial.suggest_int('max_depth', 3, 12),
    #         'subsample': trial.suggest_float('subsample', 0.5, 1.0),
    #         'n_estimators': trial.suggest_int('n_estimators', 50, 150)
    #     }
    #     model = XGBoostModel(**params)
        
    # if model_name == "LightGBM":
    #     params = {
    #         'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
    #         'num_leaves': trial.suggest_int('num_leaves', 20, 150),
    #         'max_depth': trial.suggest_int('max_depth', 3, 15),
    #         'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0)
    #     }
    #     model = LightGBMModel(**params)
        
    if model_name == "CatBoost":
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'depth': trial.suggest_int('depth', 4, 12),
            'iterations': trial.suggest_int('iterations', 50, 150),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 20.0, log=True)
        }
        model = CatBoostModel(**params)
        
    elif model_name == "Random Forest":
        params = {
            'max_depth': trial.suggest_int('max_depth', 5, 20),
            'n_estimators': trial.suggest_int('n_estimators', 30, 100),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'class_weight': 'balanced' # Rất quan trọng để trị mất cân bằng
        }
        model = RandomForestModel(**params)
        
    elif model_name == "Extra Trees":
        params = {
            'max_depth': trial.suggest_int('max_depth', 5, 20),
            'n_estimators': trial.suggest_int('n_estimators', 30, 100),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'class_weight': 'balanced' # Bắt buộc phải có với Extra Trees
        }
        model = ExtraTreesModel(**params)

    # Huấn luyện và dự đoán
    # Để Optuna chạy im lặng, ta tắt print trong lúc fit
    model.model.fit(X_train, y_train) 
    y_pred = model.model.predict(X_val)
    y_prob = model.model.predict_proba(X_val)
    if y_prob.ndim > 1: # Xử lý trường hợp predict_proba trả về mảng 2D (như RF)
        y_prob = y_prob[:, 1]
    
    # Tính toán Custom Score (Nặng Recall, Phạt Log Loss)
    f2 = fbeta_score(y_val, y_pred, beta=2.0)
    loss = log_loss(y_val, y_prob)
    
    custom_score = f2 - (0.1 * loss)
    return custom_score

# =================================================================================
# MAIN PIPELINE
# =================================================================================

print("BẮT ĐẦU QUÁ TRÌNH TÌM KIẾM SIÊU THAM SỐ TỐI ƯU (HYPERPARAMETER TUNING)")
print(f"Số vòng lặp (Trials) cho mỗi mô hình: {N_TRIALS}")
print(f"Số lượng dữ liệu dùng để Tuning: {TUNING_SAMPLE_SIZE if TUNING_SAMPLE_SIZE else 'Toàn bộ'} dòng")
print("="*80)

def prepare_data(dataset_path):
    print(f"\nĐang chuẩn bị dữ liệu từ: {dataset_path.name}...")
    df = pd.read_parquet(dataset_path, engine="fastparquet")
    if TUNING_SAMPLE_SIZE is not None and len(df) > TUNING_SAMPLE_SIZE:
        df = df.sample(n=TUNING_SAMPLE_SIZE, random_state=42)
    X = df.drop(columns=[TARGET, "row_id", "timestamp"], errors="ignore").fillna(0)
    y = df[TARGET]
    # Cố định tập Train/Val cho mọi Trials để công bằng
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Tải trước 2 bộ dữ liệu
print("\n[1] Chuẩn bị bộ dữ liệu PREPROCESSED (Dành riêng cho XGBoost)")
X_train_pre, X_val_pre, y_train_pre, y_val_pre = prepare_data(DATASET_PREPROCESSED)

print("\n[2] Chuẩn bị bộ dữ liệu ENGINEERED (FE) (Dành cho các mô hình còn lại)")
X_train_fe, X_val_fe, y_train_fe, y_val_fe = prepare_data(DATASET_ENGINEERED)

# Cấu hình chiến thuật (Mô hình -> Bộ dữ liệu)
tuning_strategy = {
    "XGBoost": (X_train_pre, X_val_pre, y_train_pre, y_val_pre, "Preprocessed"),
    "LightGBM": (X_train_fe, X_val_fe, y_train_fe, y_val_fe, "Engineered"),
    "CatBoost": (X_train_fe, X_val_fe, y_train_fe, y_val_fe, "Engineered"),
    "Random Forest": (X_train_fe, X_val_fe, y_train_fe, y_val_fe, "Engineered"),
    "Extra Trees": (X_train_fe, X_val_fe, y_train_fe, y_val_fe, "Engineered")
}

best_results_log = []

# Tắt log hệ thống của Optuna cho đỡ rác màn hình
optuna.logging.set_verbosity(optuna.logging.WARNING)

for m_name, (X_t, X_v, y_t, y_v, ds_name) in tuning_strategy.items():
    print(f"\n  -> Đang tìm tham số tối ưu cho {m_name} trên tập {ds_name}...")
    start_time = time.time()
    
    # Khởi tạo Optuna Study
    study = optuna.create_study(direction="maximize") # Tối đa hóa Custom Score
    
    # Chạy tối ưu hóa
    try:
        study.optimize(lambda trial: objective(trial, m_name, X_t, y_t, X_v, y_v), n_trials=N_TRIALS)
        
        best_score = study.best_value
        best_params = study.best_params
        
        best_results_log.append({
            "Model": m_name,
            "Dataset Used": ds_name,
            "Best Custom Score": round(best_score, 4),
            "Best Hyperparameters": str(best_params)
        })
        
        print(f"     [XONG] Best Score: {best_score:.4f} | Time: {time.time() - start_time:.1f}s")
        print(f"     [PARAMS] {best_params}")
        
    except Exception as e:
        print(f"     [LỖI] {e}")

# In Bảng tổng kết
print("\n" + "="*90)
print("BẢNG TỔNG KẾT SIÊU THAM SỐ TỐI ƯU THEO CHIẾN THUẬT LỰA CHỌN")
print("="*90)

results_df = pd.DataFrame(best_results_log)
if not results_df.empty:
    results_df = results_df.sort_values(by=["Best Custom Score"], ascending=False)
    
    # Hiển thị đẹp hơn
    pd.set_option('display.max_colwidth', None) 
    print(results_df.to_markdown(index=False))
    
    output_csv = os.path.join(script_dir, "tuning_best_params.csv")
    results_df.to_csv(output_csv, index=False)
    print(f"\nĐã lưu chi tiết cấu hình tối ưu tại: {output_csv}")
