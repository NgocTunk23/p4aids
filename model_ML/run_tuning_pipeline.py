import os
import sys

# Đảm bảo UTF-8 cho Windows Console để in tiếng Việt có dấu không bị lỗi UnicodeEncodeError
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, 'reconfigure'):
            getattr(sys.stdout, 'reconfigure')(encoding='utf-8')
        if hasattr(sys.stderr, 'reconfigure'):
            getattr(sys.stderr, 'reconfigure')(encoding='utf-8')
    except Exception:
        pass

# In thông báo ngay lập tức để người dùng biết script đã bắt đầu chạy
print("=" * 85, flush=True)
print("     TỐI ƯU SIÊU THAM SỐ ĐA LÕI TRÊN DỮ LIỆU BASELINE (OPTUNA 50 VÒNG)", flush=True)
print("=" * 85, flush=True)
print("-> Đang khởi động và nạp các thư viện Machine Learning (XGBoost, LightGBM, CatBoost)...", flush=True)

import time
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
import optuna

# Thêm đường dẫn gốc để import model_classes
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model_ML.model_classes import XGBoostModel, LightGBMModel, CatBoostModel

import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------------
# CẤU HÌNH ĐA LÕI VÀ TUNING
# ---------------------------------------------------------------------------------
N_CORES = os.cpu_count() or 4
N_TRIALS = 50  # Mặc định 50 vòng cho mỗi mô hình
TARGET = "anomaly"

script_dir = Path(__file__).resolve().parent
DATA_PATH = script_dir.parent / "data" / "processed" / "train_preprocessed.parquet"
OUTPUT_JSON = script_dir / "baseline_tuning_best_params.json"
OUTPUT_TXT = script_dir / "baseline_tuning_summary.txt"

print("=" * 85)
print("     TỐI ƯU SIÊU THAM SỐ ĐA LÕI TRÊN DỮ LIỆU BASELINE (THƯỜNG) - OPTUNA 50 VÒNG")
print("=" * 85)
print(f"-> Hệ thống phát hiện: {N_CORES} CPU Cores. Kích hoạt toàn bộ lõi cho huấn luyện.")
print(f"-> Số vòng lặp (Trials) cho mỗi mô hình: {N_TRIALS}")
print(f"-> Nguồn dữ liệu: {DATA_PATH.name}")
print("-" * 85)

# ---------------------------------------------------------------------------------
# 1. TẢI VÀ CHUẨN BỊ DỮ LIỆU BASELINE
# ---------------------------------------------------------------------------------
def load_baseline_data():
    t0 = time.time()
    print("Đang nạp dữ liệu Baseline bằng PyArrow đa luồng...", end=" ", flush=True)
    df = pd.read_parquet(DATA_PATH, engine="pyarrow")
    if 'timestamp' in df.columns:
        df = df.drop(columns=['timestamp'])
    X = df.drop(columns=[TARGET, "row_id"], errors="ignore")
    y = df[TARGET]
    print(f"Xong trong {time.time() - t0:.1f}s. Kích thước: {X.shape}")
    
    # Chia Train/Val cố định 80/20 với stratify=y để đảm bảo phân phối nhãn dị thường
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

X_train, X_val, y_train, y_val = load_baseline_data()
print(f"-> Tập Train: {X_train.shape[0]:,} dòng | Tập Val: {X_val.shape[0]:,} dòng")
print("-" * 85)

# ---------------------------------------------------------------------------------
# 2. HÀM TỐI ƯU NGƯỠNG TĨNH DỰA TRÊN PRECISION-RECALL CURVE (VECTOR HÓA SIÊU TỐC)
# ---------------------------------------------------------------------------------
def optimize_threshold(y_true, y_prob):
    """Tính toán F1 Score tối ưu dọc theo toàn bộ các ngưỡng xác suất bằng C++"""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_th = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    return float(f1_scores[best_idx]), float(best_th)

# ---------------------------------------------------------------------------------
# 3. ĐỊNH NGHĨA OBJECTIVE CHO TỪNG MÔ HÌNH (SỬ DỤNG FULL CORES)
# ---------------------------------------------------------------------------------

def objective_lgb(trial):
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.03, 0.25, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 63, 255),
        'max_depth': trial.suggest_int('max_depth', 10, 18),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 0.9),
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
        'n_estimators': trial.suggest_int('n_estimators', 120, 220),
        'n_jobs': -1,          # Tận dụng 100% CPU cores
        'random_state': 42,
        'verbose': -1
    }
    model = LightGBMModel(**params)
    model.train(X_train, y_train)
    preds = model.predict_proba(X_val)
    if len(preds.shape) > 1 and preds.shape[1] == 2:
        preds = preds[:, 1]
    best_f1, best_th = optimize_threshold(y_val, preds)
    trial.set_user_attr('best_threshold', best_th)
    return best_f1

def objective_xgb(trial):
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 10, 16),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'n_estimators': trial.suggest_int('n_estimators', 120, 180),
        'tree_method': 'hist', # Tối ưu hóa histogram C++ đa luồng cực nhanh
        'n_jobs': -1,          # Tận dụng 100% CPU cores
        'eval_metric': 'logloss',
        'random_state': 42
    }
    model = XGBoostModel(**params)
    model.train(X_train, y_train)
    preds = model.predict_proba(X_val)
    if len(preds.shape) > 1 and preds.shape[1] == 2:
        preds = preds[:, 1]
    best_f1, best_th = optimize_threshold(y_val, preds)
    trial.set_user_attr('best_threshold', best_th)
    return best_f1

def objective_cat(trial):
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.08, 0.35, log=True),
        'depth': trial.suggest_int('depth', 8, 14),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 5.0, log=True),
        'iterations': trial.suggest_int('iterations', 120, 180),
        'thread_count': -1,    # Tận dụng 100% CPU cores
        'verbose': 0,
        'random_state': 42
    }
    model = CatBoostModel(**params)
    model.train(X_train, y_train)
    preds = model.predict_proba(X_val)
    if len(preds.shape) > 1 and preds.shape[1] == 2:
        preds = preds[:, 1]
    best_f1, best_th = optimize_threshold(y_val, preds)
    trial.set_user_attr('best_threshold', best_th)
    return best_f1

# ---------------------------------------------------------------------------------
# 4. CALLBACK THEO DÕI TIẾN ĐỘ THỜI GIAN THỰC
# ---------------------------------------------------------------------------------
def create_callback(model_name):
    def optuna_callback(study, trial):
        duration = (trial.datetime_complete - trial.datetime_start).total_seconds()
        val = trial.value
        best_val = study.best_value
        is_record = (trial.number == study.best_trial.number)
        
        status_tag = "[KỶ LỤC MỚI ⭐]" if is_record else "             "
        th_val = trial.user_attrs.get('best_threshold', 0.5)
        print(f"  [{model_name}] Trial {trial.number + 1:02d}/{N_TRIALS} | "
              f"F1: {val:.4f} (Th: {th_val:.3f}) | {duration:5.1f}s | {status_tag} Best F1: {best_val:.4f}")
    return optuna_callback

# ---------------------------------------------------------------------------------
# 5. ĐIỀU PHỐI VÀ LƯU KẾT QUẢ
# ---------------------------------------------------------------------------------
def run_tuning(models_to_run):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    # Đọc kết quả cũ nếu có để cập nhật
    all_best_params = {}
    if OUTPUT_JSON.exists():
        try:
            with open(OUTPUT_JSON, "r", encoding="utf-8") as f:
                all_best_params = json.load(f)
        except Exception:
            all_best_params = {}
            
    summary_results = []
    
    mapping = {
        "LightGBM": objective_lgb,
        "XGBoost": objective_xgb,
        "CatBoost": objective_cat
    }
    
    for name in models_to_run:
        print(f"\n>>> BẮT ĐẦU TỐI ƯU SIÊU THAM SỐ CHO: {name.upper()} ({N_TRIALS} VÒNG) <<<")
        t_model_start = time.time()
        
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
        callback = create_callback(name)
        
        study.optimize(mapping[name], n_trials=N_TRIALS, callbacks=[callback])
        
        elapsed = time.time() - t_model_start
        best_trial = study.best_trial
        best_score = best_trial.value
        best_th = best_trial.user_attrs.get('best_threshold', 0.5)
        best_params = best_trial.params
        
        # Cập nhật kết quả
        all_best_params[name] = {
            "best_f1": round(best_score, 4),
            "best_threshold": round(best_th, 4),
            "tuning_time_sec": round(elapsed, 1),
            "params": best_params
        }
        
        summary_results.append({
            "Mô hình": name,
            "Best F1 Score": round(best_score, 4),
            "Ngưỡng Tối Ưu": round(best_th, 4),
            "Thời Gian (s)": round(elapsed, 1),
            "Siêu tham số tốt nhất": str(best_params)
        })
        
        # Lưu JSON liên tục sau mỗi model hoàn tất
        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(all_best_params, f, indent=4, ensure_ascii=False)
            
        print(f"--- ĐÃ HOÀN TẤT {name} ---")
        print(f"  -> Best F1: {best_score:.4f} (Ngưỡng {best_th:.3f}) | Tổng thời gian: {elapsed:.1f}s")
        print(f"  -> Params: {best_params}")

    # In Bảng Tổng Kết
    print("\n" + "=" * 90)
    print("                 BẢNG TỔNG KẾT TUNING BASELINE (OPTUNA 50 VÒNG)")
    print("=" * 90)
    summary_df = pd.DataFrame(summary_results)
    if not summary_df.empty:
        try:
            tbl_str = summary_df.to_markdown(index=False)
        except Exception:
            tbl_str = summary_df.to_string(index=False)
        print(tbl_str)
        with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
            f.write("BẢNG TỔNG KẾT TUNING BASELINE OPTUNA\n")
            f.write(tbl_str)
            f.write("\n")
            
    print(f"\n=> Đã lưu cấu hình siêu tham số chuẩn dạng JSON tại: {OUTPUT_JSON}")
    print(f"=> Bạn có thể chạy script đánh giá toàn diện để đối chiếu trực tiếp Baseline vs Zone!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna Multi-core Tuning Pipeline cho Baseline")
    parser.add_argument("--model", type=str, default="all", choices=["all", "lgb", "xgb", "cat"],
                        help="Chọn model cần tune: 'all' (cả 3), 'lgb', 'xgb', hoặc 'cat'")
    parser.add_argument("--trials", type=int, default=N_TRIALS, help="Số vòng lặp Optuna (mặc định 50)")
    args = parser.parse_args()
    
    N_TRIALS = args.trials
    
    if args.model == "lgb":
        models_to_run = ["LightGBM"]
    elif args.model == "xgb":
        models_to_run = ["XGBoost"]
    elif args.model == "cat":
        models_to_run = ["CatBoost"]
    else:
        models_to_run = ["LightGBM", "XGBoost", "CatBoost"]
        
    run_tuning(models_to_run)
