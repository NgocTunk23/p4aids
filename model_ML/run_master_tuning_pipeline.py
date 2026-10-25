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
from typing import Any, Dict, Tuple, List

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, f1_score, precision_score, recall_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import optuna

# ---------------------------------------------------------------------------------
# CẤU HÌNH HỆ THỐNG VÀ ĐƯỜNG DẪN DỮ LIỆU
# ---------------------------------------------------------------------------------
N_CORES = os.cpu_count() or 4
TARGET = "anomaly"

script_dir = Path(__file__).resolve().parent
ROOT_DIR = script_dir.parent

DATA_BASE_PATH = ROOT_DIR / "data" / "processed" / "train_preprocessed.parquet"
DATA_ZONE_PATH = ROOT_DIR / "data" / "processed" / "train_preprocessedzone.parquet"

SAVED_MODELS_DIR = script_dir / "saved_new_models"
EXPORTED_MODELS_DIR = script_dir / "exported_pipeline" / "models"
HYPERPARAMS_DIR = script_dir / "hyperparameters"

SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
EXPORTED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
HYPERPARAMS_DIR.mkdir(parents=True, exist_ok=True)

# Mốc kỷ lục trước đây để theo dõi tiến trình
BENCHMARK_RECORDS = {
    "Baseline": {
        "XGBoost": {"f1": 0.9332, "th": 0.2405, "desc": "Baseline XGBoost cũ"},
        "LightGBM": {"f1": 0.9284, "th": 0.2784, "desc": "Baseline LightGBM cũ"}
    },
    "Zone": {
        "XGBoost": {"f1": 0.9317, "th": 0.2631, "desc": "Zone XGBoost 55 cột cũ"},
        "LightGBM": {"f1": 0.9240, "th": 0.3174, "desc": "Zone LightGBM 55 cột cũ"}
    }
}


# ---------------------------------------------------------------------------------
# 1. HÀM TẢI VÀ CHUẨN BỊ 2 BỘ DỮ LIỆU (KHÔNG DÙNG ONE-HOT)
# ---------------------------------------------------------------------------------
def get_clean_datasets(dataset_name: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Tải bộ dữ liệu tương ứng:
    - Baseline: 40 đặc trưng (đã loại bỏ 13 One-Hot, dùng primary_use_code dạng số)
    - Zone: 43 đặc trưng (40 đặc trưng Baseline + zone_interaction, site_sqft_mean, use_sqft_mean)
    """
    t0 = time.time()
    if dataset_name.lower() == "baseline":
        print(f"-> Đang nạp bộ dữ liệu BASELINE (40 đặc trưng)...", end=" ", flush=True)
        df = pd.read_parquet(DATA_BASE_PATH, engine="pyarrow")
        if 'timestamp' in df.columns:
            df = df.drop(columns=['timestamp'])
        df['primary_use_code'] = df['primary_use_code'].astype('category')
        X = df.drop(columns=[TARGET, 'row_id'], errors='ignore')
        y = df[TARGET].astype(int)
        print(f"Xong trong {time.time() - t0:.1f}s. Kích thước: {X.shape}")
        
    elif dataset_name.lower() == "zone":
        print(f"-> Đang nạp bộ dữ liệu ZONE (43 đặc trưng, Không One-Hot)...", end=" ", flush=True)
        load_path = DATA_ZONE_PATH if DATA_ZONE_PATH.exists() else ROOT_DIR / "data" / "processed" / "train_preprocessedzone_43.parquet"
        df = pd.read_parquet(load_path, engine="pyarrow")
        if 'timestamp' in df.columns:
            df = df.drop(columns=['timestamp'])
        df['primary_use_code'] = df['primary_use_code'].astype('category')
        df['zone_interaction'] = df['zone_interaction'].astype('category')
        X = df.drop(columns=[TARGET, 'row_id'], errors='ignore')
        y = df[TARGET].astype(int)
        print(f"Xong trong {time.time() - t0:.1f}s. Kích thước: {X.shape}")
    else:
        raise ValueError(f"Tên dataset không hợp lệ: {dataset_name}")
        
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return X_tr, X_va, y_tr, y_va


# ---------------------------------------------------------------------------------
# 2. HÀM TỐI ƯU HÓA NGƯỠNG PHÂN LOẠI TAU*
# ---------------------------------------------------------------------------------
def optimize_threshold(y_true: pd.Series, y_prob: np.ndarray) -> Tuple[float, float]:
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = int(np.argmax(f1_scores))
    best_th = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5
    return float(f1_scores[best_idx]), best_th


def evaluate_predictions(y_true: pd.Series, y_prob: np.ndarray, optimal_th: float) -> Dict[str, Any]:
    pred_opt = (y_prob >= optimal_th).astype(int)
    pred_05 = (y_prob >= 0.5).astype(int)
    
    return {
        "best_threshold": round(float(optimal_th), 4),
        "best_f1": round(float(f1_score(y_true, pred_opt, zero_division=0)), 4),
        "precision_opt": round(float(precision_score(y_true, pred_opt, zero_division=0)), 4),
        "recall_opt": round(float(recall_score(y_true, pred_opt, zero_division=0)), 4),
        "f1_05": round(float(f1_score(y_true, pred_05, zero_division=0)), 4),
        "precision_05": round(float(precision_score(y_true, pred_05, zero_division=0)), 4),
        "recall_05": round(float(recall_score(y_true, pred_05, zero_division=0)), 4)
    }


# ---------------------------------------------------------------------------------
# 3. ĐỊNH NGHĨA SEARCH SPACE TỐI ƯU CHO XGBOOST & LIGHTGBM
# ---------------------------------------------------------------------------------
def build_objective_xgb(X_tr, y_tr, X_va, y_va):
    def objective_xgb(trial: optuna.Trial) -> float:
        # Tối ưu chuyên sâu cho dữ liệu mất cân bằng:
        # min_child_weight được quét trong dải vàng [0.3 - 1.2]
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 0.15, 0.25),
            'max_depth': trial.suggest_int('max_depth', 14, 17),
            'subsample': trial.suggest_float('subsample', 0.85, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.52, 0.72),
            'n_estimators': trial.suggest_int('n_estimators', 160, 220),
            'min_child_weight': trial.suggest_float('min_child_weight', 0.3, 1.2),
            'gamma': 0.0,
            'tree_method': 'hist',
            'enable_categorical': True,
            'n_jobs': -1,
            'eval_metric': 'logloss',
            'random_state': 42
        }
        model = XGBClassifier(**params)
        model.fit(X_tr, y_tr)
        preds = model.predict_proba(X_va)[:, 1]
        best_f1, best_th = optimize_threshold(y_va, preds)
        trial.set_user_attr('best_threshold', best_th)
        return best_f1
    return objective_xgb


def build_objective_lgb(X_tr, y_tr, X_va, y_va, is_zone: bool = False):
    # Nếu là Zone, loại bỏ zone_interaction đối với LightGBM để tránh overfit phân cụm
    if is_zone:
        X_tr_lgb = X_tr.drop(columns=['zone_interaction'], errors='ignore')
        X_va_lgb = X_va.drop(columns=['zone_interaction'], errors='ignore')
    else:
        X_tr_lgb = X_tr
        X_va_lgb = X_va

    def objective_lgb(trial: optuna.Trial) -> float:
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 0.12, 0.24),
            'num_leaves': trial.suggest_int('num_leaves', 220, 310),
            'max_depth': trial.suggest_int('max_depth', 16, 22),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.80, 0.96),
            'min_child_samples': trial.suggest_int('min_child_samples', 45, 85),
            'n_estimators': trial.suggest_int('n_estimators', 190, 260),
            'n_jobs': -1,
            'random_state': 42,
            'verbose': -1
        }
        model = LGBMClassifier(**params)
        model.fit(X_tr_lgb, y_tr, categorical_feature=['primary_use_code'])
        preds = np.asarray(model.predict_proba(X_va_lgb))[:, 1]
        best_f1, best_th = optimize_threshold(y_va, preds)
        trial.set_user_attr('best_threshold', best_th)
        return best_f1
    return objective_lgb


# ---------------------------------------------------------------------------------
# 4. CALLBACK THEO DÕI TIẾN TRÌNH THỜI GIAN THỰC (PROGRESS MONITOR)
# ---------------------------------------------------------------------------------
def create_monitor_callback(dataset_name: str, model_name: str, total_trials: int):
    target_bench = BENCHMARK_RECORDS[dataset_name][model_name]["f1"]
    
    state = {
        "best_f1_so_far": target_bench,
        "out_trinh_count": 0
    }

    def callback(study: optuna.Study, trial: optuna.trial.FrozenTrial):
        if trial.datetime_complete and trial.datetime_start:
            duration = (trial.datetime_complete - trial.datetime_start).total_seconds()
        else:
            duration = 0.0
        val = float(trial.value) if trial.value is not None else 0.0
        th_val = trial.user_attrs.get('best_threshold', 0.5)

        if val > state["best_f1_so_far"]:
            state["out_trinh_count"] += 1
            gain = (val - state["best_f1_so_far"]) * 100
            gain_target = (val - target_bench) * 100
            state["best_f1_so_far"] = val
            
            print("\n" + "=" * 95, flush=True)
            print(f"🔥🚀 [BỨT PHÁ KỶ LỤC #{state['out_trinh_count']}!] -> [{dataset_name.upper()}] - {model_name.upper()}", flush=True)
            print(f"   * Trial #{trial.number + 1:02d}/{total_trials} THIẾT LẬP ĐỈNH MỚI: F1 = {val:.4f} (Tăng: {gain_target:+.2f}%)", flush=True)
            print(f"   * Ngưỡng tối ưu: {th_val:.4f} | Thời gian: {duration:.1f}s", flush=True)
            print(f"   * Tham số: {trial.params}", flush=True)
            print("=" * 95 + "\n", flush=True)
        else:
            print(f"   [{dataset_name} | {model_name}] Trial {trial.number + 1:02d}/{total_trials} xong ({duration:4.1f}s) | F1: {val:.4f} (tau* {th_val:.3f}) | Đỉnh: {state['best_f1_so_far']:.4f}", flush=True)

    return callback


# ---------------------------------------------------------------------------------
# 5. HUẤN LUYỆN LẠI VÀ XUẤT MÔ HÌNH CHAMPION
# ---------------------------------------------------------------------------------
def retrain_and_export(dataset_name: str, model_name: str, best_params: dict, X_tr, y_tr, X_va, y_va):
    print(f"-> Đang huấn luyện lại mô hình Champion [{dataset_name} - {model_name}]...", end=" ", flush=True)
    t0 = time.time()
    
    if model_name == "XGBoost":
        p = best_params.copy()
        p.update({
            'tree_method': 'hist',
            'enable_categorical': True,
            'gamma': 0.0,
            'n_jobs': -1,
            'eval_metric': 'logloss',
            'random_state': 42
        })
        model = XGBClassifier(**p)
        model.fit(X_tr, y_tr)
        preds = model.predict_proba(X_va)[:, 1]
        fname = f"XGBoost_{dataset_name}_Champion.pkl"
        
    elif model_name == "LightGBM":
        p = best_params.copy()
        p.update({
            'n_jobs': -1,
            'random_state': 42,
            'verbose': -1
        })
        model = LGBMClassifier(**p)
        if dataset_name.lower() == "zone":
            X_tr_lgb = X_tr.drop(columns=['zone_interaction'], errors='ignore')
            X_va_lgb = X_va.drop(columns=['zone_interaction'], errors='ignore')
        else:
            X_tr_lgb = X_tr
            X_va_lgb = X_va
            
        model.fit(X_tr_lgb, y_tr, categorical_feature=['primary_use_code'])
        preds = np.asarray(model.predict_proba(X_va_lgb))[:, 1]
        fname = f"LightGBM_{dataset_name}_Champion.pkl"
    else:
        raise ValueError(f"Mô hình không được hỗ trợ: {model_name}")

    train_time = time.time() - t0
    _, best_th = optimize_threshold(y_va, preds)
    metrics = evaluate_predictions(y_va, preds, best_th)
    metrics["train_time_sec"] = round(train_time, 2)
    metrics["params"] = best_params
    
    joblib.dump(model, SAVED_MODELS_DIR / fname)
    print(f"Xong ({train_time:.1f}s)! Lưu tại {fname}")
    return metrics, preds


# ---------------------------------------------------------------------------------
# 6. ĐIỀU PHỐI VÀ CHẠY TOÀN BỘ TIẾN TRÌNH
# ---------------------------------------------------------------------------------
def run_master_pipeline(dataset_mode: str, model_mode: str, n_trials: int):
    print("=" * 95)
    print("      MASTER TUNING PIPELINE: TỐI ƯU SIÊU THAM SỐ TOÀN DIỆN CHO BASELINE VÀ ZONE")
    print("      (KHÔNG SỬ DỤNG ONE-HOT - TẬP TRUNG: XGBOOST & LIGHTGBM VỚI DẢI MIN TỐI ƯU)")
    print("=" * 95)
    print(f"-> Hệ thống: {N_CORES} CPU Cores kích hoạt tối đa.")
    print(f"-> Chế độ dữ liệu: {dataset_mode.upper()} | Chế độ mô hình: {model_mode.upper()} | Số Trials: {n_trials}")
    print("=" * 95 + "\n")

    datasets_to_run = []
    if dataset_mode.lower() in ["both", "all"]:
        datasets_to_run = ["Baseline", "Zone"]
    elif dataset_mode.lower() == "baseline":
        datasets_to_run = ["Baseline"]
    elif dataset_mode.lower() == "zone":
        datasets_to_run = ["Zone"]
    else:
        raise ValueError(f"dataset_mode không hợp lệ: {dataset_mode}")

    models_to_run = []
    if model_mode.lower() in ["both", "all"]:
        models_to_run = ["XGBoost", "LightGBM"]
    elif model_mode.lower() in ["xgb", "xgboost"]:
        models_to_run = ["XGBoost"]
    elif model_mode.lower() in ["lgb", "lightgbm"]:
        models_to_run = ["LightGBM"]
    else:
        raise ValueError(f"model_mode không hợp lệ: {model_mode}")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    all_summary_results = {}
    
    json_path = HYPERPARAMS_DIR / "master_tuning_results.json"
    existing_records = {}
    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                existing_records = json.load(f)
        except Exception:
            existing_records = {}

    for dname in datasets_to_run:
        print(f"\n{'#'*35} KHỞI CHẠY BỘ DỮ LIỆU: {dname.upper()} {'#'*35}")
        X_tr, X_va, y_tr, y_va = get_clean_datasets(dname)
        
        predictions_map = {}
        dataset_records = {}

        for mname in models_to_run:
            print(f"\n{'-'*30} BẮT ĐẦU TUNING: [{dname}] {mname.upper()} ({n_trials} TRIALS) {'-'*30}")
            
            if mname == "XGBoost":
                obj_fn = build_objective_xgb(X_tr, y_tr, X_va, y_va)
            else:
                obj_fn = build_objective_lgb(X_tr, y_tr, X_va, y_va, is_zone=(dname == "Zone"))

            monitor_cb = create_monitor_callback(dname, mname, n_trials)
            
            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=42)
            )
            
            # Khởi tạo Warm-start bằng hạt giống Champion đã kiểm chứng
            if mname == "XGBoost":
                seed_mcw = 0.5 if dname == "Zone" else 0.5
                study.enqueue_trial({
                    'learning_rate': 0.196,
                    'max_depth': 16,
                    'subsample': 0.927,
                    'colsample_bytree': 0.629,
                    'n_estimators': 170,
                    'min_child_weight': seed_mcw
                })
            elif mname == "LightGBM":
                study.enqueue_trial({
                    'learning_rate': 0.195,
                    'num_leaves': 279,
                    'max_depth': 20,
                    'feature_fraction': 0.926,
                    'min_child_samples': 73,
                    'n_estimators': 229
                })

            t_tune_start = time.time()
            study.optimize(obj_fn, n_trials=n_trials, callbacks=[monitor_cb], n_jobs=1)
            tune_duration = time.time() - t_tune_start
            
            best_trial = study.best_trial
            print(f"-> Hoàn tất Tuning [{dname} - {mname}] ({tune_duration/60:.2f} phút)! F1: {best_trial.value:.4f}")
            
            # Xuất checkpoint mô hình
            metrics, preds = retrain_and_export(dname, mname, best_trial.params, X_tr, y_tr, X_va, y_va)
            metrics["tuning_time_sec"] = round(tune_duration, 2)
            
            dataset_records[mname] = metrics
            predictions_map[mname] = preds
            all_summary_results[f"{dname}_{mname}"] = metrics

        # Tính toán Blend nếu chạy cả XGBoost và LightGBM trên tập dữ liệu này
        if "XGBoost" in predictions_map and "LightGBM" in predictions_map:
            p_xgb = predictions_map["XGBoost"]
            p_lgb = predictions_map["LightGBM"]
            best_b_f1, best_b_th, best_w = 0.0, 0.5, 0.5
            
            for w in np.linspace(0.35, 0.75, 9):
                p_blend = w * p_xgb + (1 - w) * p_lgb
                f1_b, th_b = optimize_threshold(y_va, p_blend)
                if f1_b > best_b_f1:
                    best_b_f1, best_b_th, best_w = f1_b, th_b, w
                    
            pred_b_final = (best_w * p_xgb + (1 - best_w) * p_lgb >= best_b_th).astype(int)
            blend_metrics = {
                "weights": {"XGBoost": round(float(best_w), 2), "LightGBM": round(float(1 - best_w), 2)},
                "best_f1": round(float(best_b_f1), 4),
                "best_threshold": round(float(best_b_th), 4),
                "precision_opt": round(float(precision_score(y_va, pred_b_final, zero_division=0)), 4),
                "recall_opt": round(float(recall_score(y_va, pred_b_final, zero_division=0)), 4)
            }
            dataset_records["Ensemble_Blend"] = blend_metrics
            all_summary_results[f"{dname}_Blend"] = blend_metrics
            
            print(f"\n-> [TỔ HỢP BLEND {dname.upper()}]: {best_w*100:.0f}% XGB + {(1-best_w)*100:.0f}% LGB -> F1 = {best_b_f1:.4f} (Ngưỡng {best_b_th:.4f})")

        existing_records[dname] = dataset_records
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(existing_records, f, indent=4, ensure_ascii=False)
        print(f"-> [LƯU KẾT QUẢ] Đã cập nhật file: {json_path.name}")

    # Bảng tổng kết đối chứng cuối cùng
    print("\n" + "=" * 105)
    print("                     BẢNG TỔNG KẾT SO SÁNH ĐỐI CHỨNG CUỐI CÙNG (MASTER BENCHMARK)")
    print("=" * 105)
    print(f"{'Bộ Dữ Liệu':<12} | {'Mô Hình / Phương Pháp':<24} | {'Ngưỡng tau*':<11} | {'F1-Score':<10} | {'Precision':<10} | {'Recall':<10}")
    print("-" * 105)
    for k, v in all_summary_results.items():
        parts = k.split("_")
        d_label = parts[0]
        m_label = parts[1]
        th_str = f"{v.get('best_threshold', 0.5):.4f}"
        f1_str = f"{v.get('best_f1', 0.0):.4f}"
        pr_str = f"{v.get('precision_opt', 0.0):.4f}"
        rc_str = f"{v.get('recall_opt', 0.0):.4f}"
        if m_label == "Blend":
            m_label = f"Blend ({v['weights']['XGBoost']*100:.0f}%XGB + {v['weights']['LightGBM']*100:.0f}%LGB)"
        print(f"{d_label:<12} | {m_label:<24} | {th_str:<11} | {f1_str:<10} | {pr_str:<10} | {rc_str:<10}")
    print("=" * 105)
    
    summary_txt = HYPERPARAMS_DIR / "master_tuning_summary.txt"
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write("MASTER TUNING BENCHMARK REPORT (BASELINE VS ZONE - NO ONE-HOT)\n")
        f.write("=" * 80 + "\n")
        for k, v in all_summary_results.items():
            f.write(f"{k}: F1={v.get('best_f1')} | Precision={v.get('precision_opt')} | Recall={v.get('recall_opt')} | Threshold={v.get('best_threshold')}\n")
    print(f"\n-> Đã lưu toàn bộ báo cáo tổng kết tại: {summary_txt}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Master Tuning Pipeline cho Baseline & Zone (XGBoost + LightGBM)")
    parser.add_argument("--dataset", type=str, default="both", choices=["both", "zone", "baseline"],
                        help="Chọn bộ dữ liệu: 'both' (cả Baseline & Zone), 'zone', hoặc 'baseline'")
    parser.add_argument("--model", type=str, default="both", choices=["both", "xgb", "lgb"],
                        help="Chọn mô hình: 'both' (cả XGBoost & LightGBM), 'xgb', hoặc 'lgb'")
    parser.add_argument("--trials", type=int, default=50,
                        help="Số lượng trials Optuna cho mỗi mô hình (mặc định: 50)")
    args = parser.parse_args()
    
    run_master_pipeline(args.dataset, args.model, args.trials)
