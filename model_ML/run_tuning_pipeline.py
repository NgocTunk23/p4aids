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

import time
import json
import joblib
import argparse
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, f1_score, precision_score, recall_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import optuna

# ---------------------------------------------------------------------------------
# MỐC CHUẨN CỦA BASELINE MỚI (KHÔNG DÙNG MÃ HÓA ONE-HOT) ĐỂ TÌM "OUT TRÌNH"
# ---------------------------------------------------------------------------------
N_CORES = os.cpu_count() or 4
N_TRIALS = 50
TARGET = "anomaly"

BASE_NO_ONEHOT_RECORDS = {
    "LightGBM": {
        "best_f1": 0.9212,
        "best_threshold": 0.3475,
        "precision": 0.9642,
        "recall": 0.8819,
        "desc": "Baseline Không One-Hot (Trước Tuning)"
    },
    "XGBoost": {
        "best_f1": 0.9294,
        "best_threshold": 0.3241,
        "precision": 0.9609,
        "recall": 0.8999,
        "desc": "Baseline Không One-Hot (Trước Tuning)"
    },
    "CatBoost": {
        "best_f1": 0.9159,
        "best_threshold": 0.2774,
        "precision": 0.9473,
        "recall": 0.8864,
        "desc": "Baseline Không One-Hot (Trước Tuning)"
    }
}

BASE_ONEHOT_RECORDS = {
    "LightGBM": 0.9236,
    "XGBoost": 0.9308,
    "CatBoost": 0.9188
}

script_dir = Path(__file__).resolve().parent
ROOT_DIR = script_dir.parent
DATA_PATH = ROOT_DIR / "data" / "processed" / "train_preprocessed.parquet"
OUTPUT_JSON = script_dir / "hyperparameters" / "baseline_tuning_best_params.json"
OUTPUT_TXT = script_dir / "hyperparameters" / "baseline_tuning_summary.txt"
SAVED_MODELS_DIR = script_dir / "saved_new_models"
EXPORTED_MODELS_DIR = script_dir / "exported_pipeline" / "models"

SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
EXPORTED_MODELS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 95, flush=True)
print("     TỐI ƯU SIÊU THAM SỐ BASELINE MỚI (CHUYÊN SÂU LIGHTGBM & CATBOOST)", flush=True)
print("     MỤC TIÊU: 50 TRIALS OPTUNA - BỨT PHÁ VÙNG TIỀM NĂNG CAO", flush=True)
print("=" * 95, flush=True)
print(f"-> Hệ thống phát hiện: {N_CORES} CPU Cores. Sử dụng đa luồng tối đa cho quá trình huấn luyện.")
print(f"-> Dữ liệu đầu vào: {DATA_PATH} (Đã loại bỏ 13 One-Hot, dùng primary_use_code dạng số)")
print("-> MỐC CŨ CỦA BASELINE KHÔNG ONE-HOT CẦN VƯỢT QUA (OUT TRÌNH):")
for m, rec in BASE_NO_ONEHOT_RECORDS.items():
    print(f"   * {m:10s}: F1 = {rec['best_f1']:.4f} (Ngưỡng: {rec['best_threshold']:.4f}) [Mốc One-Hot cũ: {BASE_ONEHOT_RECORDS[m]:.4f}]")
print("-" * 95, flush=True)


# ---------------------------------------------------------------------------------
# 1. TẢI VÀ CHUẨN BỊ DỮ LIỆU BASELINE MỚI
# ---------------------------------------------------------------------------------
def load_baseline_data():
    t0 = time.time()
    print("-> Đang nạp dữ liệu train_preprocessed.parquet...", end=" ", flush=True)
    df = pd.read_parquet(DATA_PATH, engine="pyarrow")
    if 'timestamp' in df.columns:
        df = df.drop(columns=['timestamp'])
    X = df.drop(columns=[TARGET, "row_id"], errors="ignore")
    y = df[TARGET]
    print(f"Xong trong {time.time() - t0:.1f}s. Kích thước: {X.shape}")
    
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


X_train, X_val, y_train, y_val = load_baseline_data()
print(f"-> Tập Train: {X_train.shape[0]:,} dòng | Tập Val: {X_val.shape[0]:,} dòng ({X_train.shape[1]} đặc trưng)")
print("-" * 95, flush=True)


# ---------------------------------------------------------------------------------
# 2. HÀM TỐI ƯU NGƯỠNG DỰA TRÊN PRECISION-RECALL CURVE
# ---------------------------------------------------------------------------------
def optimize_threshold(y_true, y_prob):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_th = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    return float(f1_scores[best_idx]), float(best_th)


def full_evaluation(y_true, y_prob, best_th):
    y_pred_opt = (y_prob >= best_th).astype(int)
    prec_opt = float(precision_score(y_true, y_pred_opt, zero_division=0))
    rec_opt = float(recall_score(y_true, y_pred_opt, zero_division=0))
    f1_opt = float(f1_score(y_true, y_pred_opt, zero_division=0))

    y_pred_05 = (y_prob >= 0.5).astype(int)
    prec_05 = float(precision_score(y_true, y_pred_05, zero_division=0))
    rec_05 = float(recall_score(y_true, y_pred_05, zero_division=0))
    f1_05 = float(f1_score(y_true, y_pred_05, zero_division=0))

    return {
        "best_threshold": round(best_th, 4),
        "f1_opt": round(f1_opt, 4),
        "precision_opt": round(prec_opt, 4),
        "recall_opt": round(rec_opt, 4),
        "f1_05": round(f1_05, 4),
        "precision_05": round(prec_05, 4),
        "recall_05": round(rec_05, 4)
    }


# ---------------------------------------------------------------------------------
# 3. ĐỊNH NGHĨA OBJECTIVE NÂNG CẤP VÙNG TIỀM NĂNG CAO
# ---------------------------------------------------------------------------------
def objective_lgb(trial):
    # Tập trung vào vùng lá lớn (190-300) và độ sâu sâu (15-20) đã chứng minh hiệu quả
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.11, 0.22),
        'num_leaves': trial.suggest_int('num_leaves', 190, 300),
        'max_depth': trial.suggest_int('max_depth', 15, 20),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.78, 0.95),
        'min_child_samples': trial.suggest_int('min_child_samples', 35, 75),
        'n_estimators': trial.suggest_int('n_estimators', 180, 240),
        'n_jobs': -1,
        'random_state': 42,
        'verbose': -1
    }
    model = LGBMClassifier(**params)
    # Khai báo categorical_feature để LightGBM áp dụng thuật toán gom cụm tối ưu
    model.fit(X_train, y_train, categorical_feature=['primary_use_code'])
    preds = np.asarray(model.predict_proba(X_val))[:, 1]
    best_f1, best_th = optimize_threshold(y_val, preds)
    trial.set_user_attr('best_threshold', best_th)
    return best_f1


def objective_xgb(trial):
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.15, 0.25),
        'max_depth': trial.suggest_int('max_depth', 14, 17),
        'subsample': trial.suggest_float('subsample', 0.85, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.50, 0.75),
        'n_estimators': trial.suggest_int('n_estimators', 140, 190),
        'tree_method': 'hist',
        'n_jobs': -1,
        'eval_metric': 'logloss',
        'random_state': 42
    }
    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    preds = model.predict_proba(X_val)[:, 1]
    best_f1, best_th = optimize_threshold(y_val, preds)
    trial.set_user_attr('best_threshold', best_th)
    return best_f1


def objective_cat(trial):
    # Cấu hình CatBoost với độ sâu lớn (11-14) và Target Statistics trên primary_use_code
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.18, 0.32),
        'depth': trial.suggest_int('depth', 11, 14),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 3.5),
        'iterations': trial.suggest_int('iterations', 140, 200),
        'thread_count': -1,
        'verbose': 0,
        'random_state': 42
    }
    model = CatBoostClassifier(**params)
    model.fit(X_train, y_train, cat_features=['primary_use_code'])
    preds = model.predict_proba(X_val)[:, 1]
    best_f1, best_th = optimize_threshold(y_val, preds)
    trial.set_user_attr('best_threshold', best_th)
    return best_f1


# ---------------------------------------------------------------------------------
# 4. CALLBACK XUẤT RA MÀN HÌNH KHI OUT TRÌNH
# ---------------------------------------------------------------------------------
def create_out_trinh_callback(model_name, total_trials):
    prev_no_onehot_f1 = BASE_NO_ONEHOT_RECORDS[model_name]["best_f1"]
    prev_onehot_f1 = BASE_ONEHOT_RECORDS[model_name]
    
    state = {
        "best_f1_so_far": prev_no_onehot_f1,
        "out_trinh_count": 0
    }

    def callback(study, trial):
        duration = (trial.datetime_complete - trial.datetime_start).total_seconds()
        val = trial.value
        th_val = trial.user_attrs.get('best_threshold', 0.5)

        if val > state["best_f1_so_far"]:
            state["out_trinh_count"] += 1
            gain_vs_old_base = (val - prev_no_onehot_f1) * 100
            gain_vs_onehot = (val - prev_onehot_f1) * 100
            state["best_f1_so_far"] = val
            
            beat_onehot_str = f"🚀 VƯỢT CẢ ONE-HOT CŨ ({gain_vs_onehot:+.2f}%)" if val > prev_onehot_f1 else f"Tiệm cận One-Hot cũ ({gain_vs_onehot:+.2f}%)"
            
            print("\n" + "=" * 95, flush=True)
            print(f"🔥🚀 [OUT TRÌNH KỶ LỤC CŨ #{state['out_trinh_count']}!] -> MÔ HÌNH: {model_name.upper()} (BASELINE KHÔNG ONE-HOT)", flush=True)
            print(f"   * Trial #{trial.number + 1:02d}/{total_trials} THIẾT LẬP MỐC CHUẨN TỐI ƯU MỚI!", flush=True)
            print(f"   * F1 Mới: {val:.4f} > Mốc cũ Baseline không One-hot: {prev_no_onehot_f1:.4f} (Vượt trội: {gain_vs_old_base:+.2f}%)", flush=True)
            print(f"   * So sánh đối chứng: {beat_onehot_str} [Mốc One-Hot cũ: {prev_onehot_f1:.4f}]", flush=True)
            print(f"   * Ngưỡng phân loại tối ưu: {th_val:.4f} | Thời gian tính: {duration:.1f}s", flush=True)
            print(f"   * Cấu hình siêu tham số: {trial.params}", flush=True)
            print("=" * 95 + "\n", flush=True)
        else:
            print(f"   [{model_name}] Trial {trial.number + 1:02d}/{total_trials} xong ({duration:4.1f}s) | F1: {val:.4f} (Ngưỡng {th_val:.3f}) | Mốc cao nhất: {state['best_f1_so_far']:.4f}", flush=True)

    return callback


# ---------------------------------------------------------------------------------
# 5. HUẤN LUYỆN LẠI VÀ LƯU CHECKPOINT MÔ HÌNH TỐI ƯU
# ---------------------------------------------------------------------------------
def refit_and_save_champion(model_name, best_params, best_threshold):
    print(f"\n-> Đang huấn luyện lại mô hình Champion {model_name} với bộ siêu tham số tốt nhất...", flush=True)
    t_train_start = time.time()
    
    if model_name == "LightGBM":
        p = best_params.copy()
        p['n_jobs'] = -1
        p['random_state'] = 42
        p['verbose'] = -1
        model = LGBMClassifier(**p)
        model.fit(X_train, y_train, categorical_feature=['primary_use_code'])
        filename = "LightGBM_Baseline.pkl"
    elif model_name == "XGBoost":
        p = best_params.copy()
        p['tree_method'] = 'hist'
        p['n_jobs'] = -1
        p['eval_metric'] = 'logloss'
        p['random_state'] = 42
        model = XGBClassifier(**p)
        model.fit(X_train, y_train)
        filename = "XGBoost_Baseline_Full.pkl"
    else:
        p = best_params.copy()
        p['thread_count'] = -1
        p['verbose'] = 0
        p['random_state'] = 42
        model = CatBoostClassifier(**p)
        model.fit(X_train, y_train, cat_features=['primary_use_code'])
        filename = "CatBoost_Baseline.pkl"
        
    train_time = time.time() - t_train_start
    
    t_inf_start = time.time()
    preds = np.asarray(model.predict_proba(X_val))[:, 1]
    inf_time = time.time() - t_inf_start
    
    metrics = full_evaluation(y_val, preds, best_threshold)
    metrics["train_time_sec"] = round(train_time, 2)
    metrics["inference_time_sec"] = round(inf_time, 2)
    
    save_path1 = SAVED_MODELS_DIR / filename
    save_path2 = EXPORTED_MODELS_DIR / filename
    
    joblib.dump(model, save_path1)
    joblib.dump(model, save_path2)
    print(f"-> Đã lưu mô hình Champion chuẩn tại:\n   1. {save_path1}\n   2. {save_path2}")
    
    return metrics, preds


# ---------------------------------------------------------------------------------
# 6. ĐIỀU PHỐI TIẾN TRÌNH TUNING
# ---------------------------------------------------------------------------------
def run_tuning_all(models_to_run, n_trials):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    mapping = {
        "LightGBM": objective_lgb,
        "XGBoost": objective_xgb,
        "CatBoost": objective_cat
    }
    
    all_best_params = {}
    if OUTPUT_JSON.exists():
        try:
            with open(OUTPUT_JSON, "r", encoding="utf-8") as f:
                all_best_params = json.load(f)
        except Exception:
            all_best_params = {}
            
    comprehensive_records = {}
    validation_probs = {}
    
    # Nạp lại xác suất của các mô hình đã có sẵn nếu không cần chạy lại
    for existing_name in ["LightGBM", "XGBoost", "CatBoost"]:
        if existing_name not in models_to_run:
            fname = f"{existing_name}_Baseline.pkl" if existing_name != "XGBoost" else "XGBoost_Baseline_Full.pkl"
            fpath = SAVED_MODELS_DIR / fname
            if fpath.exists() and existing_name in all_best_params:
                try:
                    m = joblib.load(fpath)
                    validation_probs[existing_name] = m.predict_proba(X_val)[:, 1]
                    info = all_best_params[existing_name]
                    comprehensive_records[existing_name] = {
                        "Mô hình": f"{existing_name} (Đã có sẵn)",
                        "Ngưỡng Tối Ưu": info.get("best_threshold", 0.5),
                        "F1 (Ngưỡng 0.5)": info.get("f1_05", 0.0),
                        "F1 (Ngưỡng Tối Ưu)": info.get("best_f1", 0.0),
                        "Precision Tối Ưu": info.get("precision_opt", 0.0),
                        "Recall Tối Ưu": info.get("recall_opt", 0.0),
                        "Thời Gian Inference": f"{info.get('inference_time_sec', 0.0)}s",
                        "Đánh Giá": info.get("status", "Hoàn tất")
                    }
                except Exception:
                    pass
    
    for name in models_to_run:
        prev_no_oh = BASE_NO_ONEHOT_RECORDS[name]["best_f1"]
        print(f"\n{'#' * 95}", flush=True)
        print(f"### BẮT ĐẦU 50 TRIALS TỐI ƯU CHO: {name.upper()} TRÊN BASELINE KHÔNG ONE-HOT (VÙNG TIỀM NĂNG CAO)", flush=True)
        print(f"### MỐC CŨ: F1 = {prev_no_oh:.4f} | Ngưỡng cũ = {BASE_NO_ONEHOT_RECORDS[name]['best_threshold']:.4f}", flush=True)
        print(f"{'#' * 95}\n", flush=True)
        
        t_start = time.time()
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
        callback = create_out_trinh_callback(name, n_trials)
        
        study.optimize(mapping[name], n_trials=n_trials, callbacks=[callback])
        
        total_tune_time = time.time() - t_start
        best_trial = study.best_trial
        best_f1 = float(best_trial.value) if best_trial.value is not None else 0.0
        best_th = float(best_trial.user_attrs.get('best_threshold', 0.5))
        best_p = best_trial.params
        
        metrics, prob_val = refit_and_save_champion(name, best_p, best_th)
        validation_probs[name] = prob_val
        
        gain = (best_f1 - prev_no_oh) * 100
        out_msg = f"🔥 OUT TRÌNH BASELINE CŨ ({gain:+.2f}%)" if best_f1 > prev_no_oh else "Ngang bằng kỷ lục cũ"
        
        all_best_params[name] = {
            "best_f1": round(best_f1, 4),
            "best_threshold": round(best_th, 4),
            "precision_opt": metrics["precision_opt"],
            "recall_opt": metrics["recall_opt"],
            "f1_05": metrics["f1_05"],
            "tuning_time_sec": round(total_tune_time, 1),
            "inference_time_sec": metrics["inference_time_sec"],
            "status": out_msg,
            "params": best_p
        }
        
        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(all_best_params, f, indent=4, ensure_ascii=False)
            
        comprehensive_records[name] = {
            "Mô hình": f"{name} (Baseline Không One-Hot)",
            "Ngưỡng Tối Ưu": metrics["best_threshold"],
            "F1 (Ngưỡng 0.5)": metrics["f1_05"],
            "F1 (Ngưỡng Tối Ưu)": metrics["f1_opt"],
            "Precision Tối Ưu": metrics["precision_opt"],
            "Recall Tối Ưu": metrics["recall_opt"],
            "Thời Gian Inference": f"{metrics['inference_time_sec']}s",
            "Đánh Giá": out_msg
        }

    # Tính toán thêm Weighted Ensemble Baseline nếu có đủ 3 model
    p_lgb = validation_probs.get("LightGBM")
    p_xgb = validation_probs.get("XGBoost")
    p_cat = validation_probs.get("CatBoost")
    
    if p_lgb is not None and p_xgb is not None and p_cat is not None:
        print("\n-> Đang tính toán hiệu năng Ensemble kết hợp trên tập Baseline mới...", flush=True)
        # Tỷ trọng tối ưu: XGBoost 0.55 + LightGBM 0.30 + CatBoost 0.15
        blend_prob = 0.55 * p_xgb + 0.30 * p_lgb + 0.15 * p_cat
        b_f1, b_th = optimize_threshold(y_val, blend_prob)
        b_metrics = full_evaluation(y_val, blend_prob, b_th)
        comprehensive_records["Ensemble"] = {
            "Mô hình": "Weighted Blend (XGB+LGB+CAT) Baseline",
            "Ngưỡng Tối Ưu": b_metrics["best_threshold"],
            "F1 (Ngưỡng 0.5)": b_metrics["f1_05"],
            "F1 (Ngưỡng Tối Ưu)": b_metrics["f1_opt"],
            "Precision Tối Ưu": b_metrics["precision_opt"],
            "Recall Tối Ưu": b_metrics["recall_opt"],
            "Thời Gian Inference": "<0.1s",
            "Đánh Giá": "Đỉnh cao Baseline Mới"
        }
        all_best_params["Ensemble_Baseline"] = b_metrics
        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(all_best_params, f, indent=4, ensure_ascii=False)

    print("\n" + "=" * 105, flush=True)
    print("        BẢNG TỔNG KẾT ĐÁNH GIÁ CHUẨN BASELINE MỚI (40 ĐẶC TRƯNG - OPTUNA 50 VÒNG)", flush=True)
    print("=" * 105, flush=True)
    df_summary = pd.DataFrame(list(comprehensive_records.values()))
    try:
        tbl_str = df_summary.to_markdown(index=False)
    except Exception:
        tbl_str = df_summary.to_string(index=False)
    print(tbl_str, flush=True)
    
    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write("BẢNG TỔNG KẾT ĐÁNH GIÁ CHUẨN BASELINE MỚI (40 ĐẶC TRƯNG - OPTUNA 50 VÒNG)\n")
        f.write(tbl_str)
        f.write("\n")
        
    print(f"\n=> Đã cập nhật kết quả và tham số vào: {OUTPUT_JSON}")
    print(f"=> Đã lưu bảng tổng hợp tại: {OUTPUT_TXT}")
    print("=" * 105, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tối ưu hóa siêu tham số Baseline mới 40 đặc trưng")
    parser.add_argument("--model", type=str, default="lgb_cat", 
                        choices=["all", "lgb", "xgb", "cat", "lgb_cat"],
                        help="Chọn model cần tune: 'lgb_cat' (chỉ LGB và CAT), 'all', 'lgb', 'xgb', hoặc 'cat'")
    parser.add_argument("--trials", type=int, default=N_TRIALS, help="Số vòng lặp Optuna (mặc định 50)")
    args = parser.parse_args()
    
    if args.model == "lgb":
        targets = ["LightGBM"]
    elif args.model == "xgb":
        targets = ["XGBoost"]
    elif args.model == "cat":
        targets = ["CatBoost"]
    elif args.model == "lgb_cat":
        targets = ["LightGBM", "CatBoost"]
    else:
        targets = ["LightGBM", "XGBoost", "CatBoost"]
        
    run_tuning_all(targets, args.trials)
