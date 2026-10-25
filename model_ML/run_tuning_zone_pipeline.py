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
from typing import Any, Dict, Tuple

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
# CẤU HÌNH PHẦN CỨNG VÀ MỐC KỶ LỤC CẦN PHÁ ĐỔI (TARGET BENCHMARKS)
# ---------------------------------------------------------------------------------
N_CORES = os.cpu_count() or 4
TARGET = "anomaly"

# Mốc kỷ lục trước đây để theo dõi sự bứt phá (Out Trình)
HISTORICAL_BENCHMARKS = {
    "XGBoost": {
        "zone_old_f1": 0.9317,
        "baseline_record_f1": 0.9332,
        "desc": "Baseline Không One-Hot (XGBoost)"
    },
    "LightGBM": {
        "zone_old_f1": 0.9240,
        "baseline_record_f1": 0.9284,
        "desc": "Baseline Không One-Hot (LightGBM)"
    },
    "CatBoost": {
        "zone_old_f1": 0.9202,
        "baseline_record_f1": 0.9200,
        "desc": "Baseline Không One-Hot (CatBoost)"
    }
}

script_dir = Path(__file__).resolve().parent
ROOT_DIR = script_dir.parent
DATA_CLEAN_PATH = ROOT_DIR / "data" / "processed" / "train_preprocessedzone.parquet"
DATA_BASE_PATH = ROOT_DIR / "data" / "processed" / "train_preprocessed.parquet"

OUTPUT_JSON_HYPER = script_dir / "hyperparameters" / "zone_tuning_best_params.json"
OUTPUT_TXT = script_dir / "hyperparameters" / "zone_tuning_summary.txt"
SAVED_MODELS_DIR = script_dir / "saved_new_models"
EXPORTED_MODELS_DIR = script_dir / "exported_pipeline" / "models"

SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
(script_dir / "hyperparameters").mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------------
# 1. TẢI VÀ CHUẨN BỊ DỮ LIỆU ZONE TINH GỌN (43 ĐẶC TRƯNG, KHÔNG ONE-HOT)
# ---------------------------------------------------------------------------------
def load_zone_43_dataset() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Tải bộ dữ liệu Zone tinh gọn 43 đặc trưng:
    - Loại bỏ hoàn toàn 13 cột One-Hot thưa (tránh pha loãng subsampling colsample_bytree).
    - Duy trì primary_use_code và 3 đặc trưng Zone: zone_interaction, site_sqft_mean, use_sqft_mean.
    - Ép kiểu category cho primary_use_code và zone_interaction để kích hoạt phân vùng Fisher tối ưu.
    """
    t0 = time.time()
    load_path = DATA_CLEAN_PATH if DATA_CLEAN_PATH.exists() else ROOT_DIR / "data" / "processed" / "train_preprocessedzone_43.parquet"
    print(f"-> Đang nạp nhanh {load_path.name} qua PyArrow đa luồng...", end=" ", flush=True)

    df = pd.read_parquet(load_path, engine="pyarrow")
    if 'timestamp' in df.columns:
        df = df.drop(columns=['timestamp'])
    
    # Ép kiểu danh mục (category) chuẩn hóa
    df['primary_use_code'] = df['primary_use_code'].astype('category')
    df['zone_interaction'] = df['zone_interaction'].astype('category')
    
    X = df.drop(columns=[TARGET], errors="ignore")
    y = df[TARGET].astype(int)
    
    print(f"Xong trong {time.time() - t0:.1f}s. Kích thước dữ liệu: {X.shape}")
    
    # Phân chia Train/Val 80/20 với stratify=y chuẩn hóa toàn đề tài
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return X_tr, X_va, y_tr, y_va


# ---------------------------------------------------------------------------------
# 2. HÀM TỐI ƯU HÓA NGƯỠNG PHÂN LOẠI TAU*
# ---------------------------------------------------------------------------------
def optimize_threshold(y_true: pd.Series, y_prob: np.ndarray) -> Tuple[float, float]:
    """Xác định ngưỡng phân loại cắt tối ưu dọc theo đường cong Precision-Recall"""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = int(np.argmax(f1_scores))
    best_th = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5
    return float(f1_scores[best_idx]), best_th


def full_evaluation(y_true: pd.Series, y_prob: np.ndarray, optimal_th: float) -> Dict[str, Any]:
    """Tính toán chi tiết các chỉ số F1, Precision, Recall ở cả ngưỡng 0.5 và ngưỡng tối ưu"""
    pred_opt = (y_prob >= optimal_th).astype(int)
    pred_05 = (y_prob >= 0.5).astype(int)
    
    f1_opt = f1_score(y_true, pred_opt, zero_division=0)
    prec_opt = precision_score(y_true, pred_opt, zero_division=0)
    rec_opt = recall_score(y_true, pred_opt, zero_division=0)
    
    f1_05 = f1_score(y_true, pred_05, zero_division=0)
    prec_05 = precision_score(y_true, pred_05, zero_division=0)
    rec_05 = recall_score(y_true, pred_05, zero_division=0)
    
    return {
        "best_threshold": round(float(optimal_th), 4),
        "best_f1": round(float(f1_opt), 4),
        "precision_opt": round(float(prec_opt), 4),
        "recall_opt": round(float(rec_opt), 4),
        "f1_05": round(float(f1_05), 4),
        "precision_05": round(float(prec_05), 4),
        "recall_05": round(float(rec_05), 4)
    }


# ---------------------------------------------------------------------------------
# 3. ĐỊNH NGHĨA KHÔNG GIAN TÌM KIẾM CHO XGBOOST, LIGHTGBM, CATBOOST
# ---------------------------------------------------------------------------------
def get_objective_xgb(X_tr, y_tr, X_va, y_va):
    def objective_xgb(trial: optuna.Trial) -> float:
        # Cấu hình tối ưu chuẩn cho dữ liệu mất cân bằng (anomalies 1.8%):
        # min_child_weight = 1.0 (mặc định) là BẮT BUỘC để cây được phép phân nhánh cô lập các cụm dị thường nhỏ.
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


def get_objective_lgb(X_tr, y_tr, X_va, y_va):
    # Với LightGBM, loại bỏ zone_interaction (47 categories gây overfit bin) và giữ site_sqft_mean, use_sqft_mean
    # giúp LightGBM đạt F1 >= 0.9290
    X_tr_lgb = X_tr.drop(columns=['zone_interaction'], errors='ignore')
    X_va_lgb = X_va.drop(columns=['zone_interaction'], errors='ignore')
    
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


def get_objective_cat(X_tr, y_tr, X_va, y_va):
    # Chuẩn bị dữ liệu định danh riêng cho CatBoost (CatBoost hoạt động tốt nhất với integer/string cho cat_features)
    X_tr_cat = X_tr.copy()
    X_va_cat = X_va.copy()
    X_tr_cat['primary_use_code'] = X_tr_cat['primary_use_code'].astype(int)
    X_tr_cat['zone_interaction'] = X_tr_cat['zone_interaction'].astype(int)
    X_va_cat['primary_use_code'] = X_va_cat['primary_use_code'].astype(int)
    X_va_cat['zone_interaction'] = X_va_cat['zone_interaction'].astype(int)
    
    def objective_cat(trial: optuna.Trial) -> float:
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 0.18, 0.35),
            'depth': trial.suggest_int('depth', 11, 14),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 4.0),
            'iterations': trial.suggest_int('iterations', 140, 200),
            'thread_count': -1,
            'verbose': 0,
            'random_state': 42
        }
        model = CatBoostClassifier(**params)
        model.fit(X_tr_cat, y_tr, cat_features=['primary_use_code', 'zone_interaction'])
        preds = model.predict_proba(X_va_cat)[:, 1]
        best_f1, best_th = optimize_threshold(y_va, preds)
        trial.set_user_attr('best_threshold', best_th)
        return best_f1
    return objective_cat


# ---------------------------------------------------------------------------------
# 4. CALLBACK THEO DÕI TIẾN TRÌNH VÀ BÁO HIỆU "OUT TRÌNH" KỶ LỤC
# ---------------------------------------------------------------------------------
def create_progress_callback(model_name: str, total_trials: int):
    bench = HISTORICAL_BENCHMARKS[model_name]
    zone_old_f1 = bench["zone_old_f1"]
    base_rec_f1 = bench["baseline_record_f1"]
    
    state = {
        "best_f1_so_far": max(zone_old_f1, 0.9200),
        "breakthrough_count": 0
    }

    def callback(study: optuna.Study, trial: optuna.trial.FrozenTrial):
        if trial.datetime_complete and trial.datetime_start:
            duration = (trial.datetime_complete - trial.datetime_start).total_seconds()
        else:
            duration = 0.0
        val = float(trial.value) if trial.value is not None else 0.0
        th_val = trial.user_attrs.get('best_threshold', 0.5)

        if val > state["best_f1_so_far"]:
            state["breakthrough_count"] += 1
            state["best_f1_so_far"] = val
            gain_vs_old_zone = (val - zone_old_f1) * 100
            gain_vs_base = (val - base_rec_f1) * 100
            
            beat_baseline_str = f"🚀 [ĐÃ PHÁ KỶ LỤC BASELINE ({gain_vs_base:+.2f}%)]" if val > base_rec_f1 else f"Đang tiệm cận Baseline ({gain_vs_base:+.2f}%)"
            
            print("\n" + "=" * 95, flush=True)
            print(f"🔥🚀 [BỨT PHÁ KỶ LỤC MỚI #{state['breakthrough_count']}] -> MÔ HÌNH: {model_name.upper()} (ZONE FEATURES 43 CỘT)", flush=True)
            print(f"   * Trial #{trial.number + 1:02d}/{total_trials} THIẾT LẬP ĐỈNH CAO MỚI!", flush=True)
            print(f"   * F1 Mới: {val:.4f} > Mốc Zone cũ: {zone_old_f1:.4f} (Tăng: {gain_vs_old_zone:+.2f}%)", flush=True)
            print(f"   * Đối chứng Baseline: {beat_baseline_str} [Mốc Baseline: {base_rec_f1:.4f}]", flush=True)
            print(f"   * Ngưỡng tối ưu: {th_val:.4f} | Thời gian tính: {duration:.1f}s", flush=True)
            print(f"   * Siêu tham số: {trial.params}", flush=True)
            print("=" * 95 + "\n", flush=True)
        else:
            print(f"   [{model_name}] Trial {trial.number + 1:02d}/{total_trials} xong ({duration:4.1f}s) | F1: {val:.4f} (Ngưỡng {th_val:.3f}) | Đỉnh hiện tại: {state['best_f1_so_far']:.4f}", flush=True)

    return callback


# ---------------------------------------------------------------------------------
# 5. HUẤN LUYỆN LẠI VÀ LƯU CHECKPOINT MÔ HÌNH TỐI ƯU
# ---------------------------------------------------------------------------------
def train_and_export_champion(model_name: str, best_params: dict, X_tr, y_tr, X_va, y_va):
    print(f"\n-> Đang huấn luyện lại mô hình Champion [{model_name}] với bộ tham số tốt nhất...", flush=True)
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
        t_infer_0 = time.time()
        preds = model.predict_proba(X_va)[:, 1]
        inf_time = time.time() - t_infer_0
        fname = "XGBoost_Zone_43.pkl"
        
    elif model_name == "LightGBM":
        p = best_params.copy()
        p.update({
            'n_jobs': -1,
            'random_state': 42,
            'verbose': -1
        })
        model = LGBMClassifier(**p)
        X_tr_lgb = X_tr.drop(columns=['zone_interaction'], errors='ignore')
        X_va_lgb = X_va.drop(columns=['zone_interaction'], errors='ignore')
        model.fit(X_tr_lgb, y_tr, categorical_feature=['primary_use_code'])
        t_infer_0 = time.time()
        preds = np.asarray(model.predict_proba(X_va_lgb))[:, 1]
        inf_time = time.time() - t_infer_0
        fname = "LightGBM_Zone_43.pkl"
        
    elif model_name == "CatBoost":
        p = best_params.copy()
        p.update({
            'thread_count': -1,
            'verbose': 0,
            'random_state': 42
        })
        model = CatBoostClassifier(**p)
        X_tr_cat = X_tr.copy()
        X_va_cat = X_va.copy()
        X_tr_cat['primary_use_code'] = X_tr_cat['primary_use_code'].astype(int)
        X_tr_cat['zone_interaction'] = X_tr_cat['zone_interaction'].astype(int)
        X_va_cat['primary_use_code'] = X_va_cat['primary_use_code'].astype(int)
        X_va_cat['zone_interaction'] = X_va_cat['zone_interaction'].astype(int)
        model.fit(X_tr_cat, y_tr, cat_features=['primary_use_code', 'zone_interaction'])
        t_infer_0 = time.time()
        preds = model.predict_proba(X_va_cat)[:, 1]
        inf_time = time.time() - t_infer_0
        fname = "CatBoost_Zone_43.pkl"
    else:
        raise ValueError(f"Mô hình không hợp lệ: {model_name}")
        
    train_duration = time.time() - t0
    _, best_th = optimize_threshold(y_va, preds)
    metrics = full_evaluation(y_va, preds, best_th)
    metrics["train_time_sec"] = round(train_duration, 2)
    metrics["inference_time_sec"] = round(inf_time, 2)
    
    save_path1 = SAVED_MODELS_DIR / fname
    save_path2 = EXPORTED_MODELS_DIR / fname
    joblib.dump(model, save_path1)
    joblib.dump(model, save_path2)
    print(f"-> Đã lưu mô hình Champion chuẩn tại:\n   1. {save_path1}\n   2. {save_path2}")
    
    return metrics, preds


# ---------------------------------------------------------------------------------
# 6. ĐIỀU PHỐI QUÁ TRÌNH TUNING
# ---------------------------------------------------------------------------------
def run_tuning(model_choice: str, n_trials: int):
    print("=" * 95, flush=True)
    print("     CHƯƠNG TRÌNH TỐI ƯU SIÊU THAM SỐ ZONE FEATURES 43 ĐẶC TRƯNG", flush=True)
    print("     (KHÔNG SỬ DỤNG MÃ HÓA ONE-HOT - KÍCH HOẠT PHÂN HOẠCH CATEGORICAL TỐI ƯU)", flush=True)
    print("=" * 95, flush=True)
    print(f"-> Hệ thống phát hiện: {N_CORES} CPU Cores. Sử dụng đa luồng tối đa cho quá trình huấn luyện.")
    print(f"-> Lựa chọn mô hình: {model_choice.upper()} | Số trials mỗi mô hình: {n_trials}")
    print("-> MỐC SO SÁNH ĐỐI CHỨNG CẦN BỨT PHÁ (TARGETS):")
    for m, rec in HISTORICAL_BENCHMARKS.items():
        print(f"   * {m:10s}: Mốc Zone cũ = {rec['zone_old_f1']:.4f} | Mốc Baseline cần vượt = {rec['baseline_record_f1']:.4f}")
    print("-" * 95, flush=True)
    
    # 1. Nạp dữ liệu
    X_tr, X_va, y_tr, y_va = load_zone_43_dataset()
    print(f"-> Phân chia hoàn tất: Tập Huấn luyện: {X_tr.shape[0]:,} dòng | Tập Kiểm định: {X_va.shape[0]:,} dòng (43 đặc trưng)")
    print("-" * 95, flush=True)
    
    models_to_run = []
    if model_choice.lower() in ["both", "xgb_lgb"]:
        models_to_run = ["XGBoost", "LightGBM"]
    elif model_choice.lower() in ["xgb", "xgboost"]:
        models_to_run = ["XGBoost"]
    elif model_choice.lower() in ["lgb", "lightgbm"]:
        models_to_run = ["LightGBM"]
    elif model_choice.lower() in ["cat", "catboost"]:
        models_to_run = ["CatBoost"]
    elif model_choice.lower() == "all":
        models_to_run = ["XGBoost", "LightGBM", "CatBoost"]
    else:
        raise ValueError(f"Mô hình không hợp lệ: {model_choice}. Chọn trong: both, xgb, lgb, cat, all")
        
    objective_builders = {
        "XGBoost": lambda: get_objective_xgb(X_tr, y_tr, X_va, y_va),
        "LightGBM": lambda: get_objective_lgb(X_tr, y_tr, X_va, y_va),
        "CatBoost": lambda: get_objective_cat(X_tr, y_tr, X_va, y_va)
    }
    
    all_best_params = {}
    if OUTPUT_JSON_HYPER.exists():
        try:
            with open(OUTPUT_JSON_HYPER, "r", encoding="utf-8") as f:
                all_best_params = json.load(f)
        except Exception:
            all_best_params = {}

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    results_summary = {}
    val_predictions = {}

    for name in models_to_run:
        print(f"\n{'='*38} BẮT ĐẦU TUNING: {name.upper()} ({n_trials} TRIALS) {'='*38}")
        obj_fn = objective_builders[name]()
        cb = create_progress_callback(name, n_trials)
        
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        # Warm-start Optuna bằng bộ tham số Champion Baseline đã được chứng minh hiệu quả cao
        if name == "XGBoost":
            study.enqueue_trial({
                'learning_rate': 0.196,
                'max_depth': 16,
                'subsample': 0.927,
                'colsample_bytree': 0.629,
                'n_estimators': 170,
                'min_child_weight': 0.5
            })
        elif name == "LightGBM":
            study.enqueue_trial({
                'learning_rate': 0.195,
                'num_leaves': 279,
                'max_depth': 20,
                'feature_fraction': 0.926,
                'min_child_samples': 73,
                'n_estimators': 229
            })
        elif name == "CatBoost":
            study.enqueue_trial({
                'learning_rate': 0.283,
                'depth': 14,
                'l2_leaf_reg': 3.198,
                'iterations': 191
            })
        
        t_start = time.time()
        study.optimize(obj_fn, n_trials=n_trials, callbacks=[cb], n_jobs=1)
        tuning_time = time.time() - t_start
        
        best_trial = study.best_trial
        best_val = float(best_trial.value) if best_trial.value is not None else 0.0
        print(f"\n-> HOÀN THÀNH TUNING [{name}] trong {tuning_time/60:.2f} phút!")
        print(f"   * F1 Score tối ưu: {best_val:.4f}")
        print(f"   * Siêu tham số tối ưu: {best_trial.params}")
        
        # Huấn luyện mô hình tối ưu và xuất checkpoint
        eval_metrics, preds = train_and_export_champion(name, best_trial.params, X_tr, y_tr, X_va, y_va)
        eval_metrics["tuning_time_sec"] = round(tuning_time, 2)
        eval_metrics["params"] = best_trial.params
        val_predictions[name] = preds
        
        all_best_params[name] = eval_metrics
        results_summary[name] = eval_metrics
        
        # Lưu file JSON ngay lập tức sau mỗi mô hình (chống mất mát dữ liệu)
        with open(OUTPUT_JSON_HYPER, "w", encoding="utf-8") as f:
            json.dump(all_best_params, f, indent=4, ensure_ascii=False)
        print(f"-> [TỰ ĐỘNG LƯU] Đã cập nhật kết quả vào: {OUTPUT_JSON_HYPER.name}")

    # Ghi báo cáo tổng hợp mô hình đơn
    print("\n" + "=" * 95)
    print("                    BẢNG TỔNG KẾT KẾT QUẢ TUNING ZONE 43 CỘT")
    print("=" * 95)
    print(f"{'Mô hình':<12} | {'Ngưỡng tau*':<11} | {'F1 (tau*)':<10} | {'Precision':<10} | {'Recall':<10} | {'Thời gian':<10}")
    print("-" * 95)
    for name, m in results_summary.items():
        print(f"{name:<12} | {m['best_threshold']:<11.4f} | {m['best_f1']:<10.4f} | {m['precision_opt']:<10.4f} | {m['recall_opt']:<10.4f} | {m.get('train_time_sec', 0):<8.1f} s")
    print("=" * 95)
    
    # Đánh giá tổ hợp Blend nếu chạy cả XGBoost và LightGBM
    if "XGBoost" in val_predictions and "LightGBM" in val_predictions:
        print("\n" + "=" * 95)
        print("          ĐÁNH GIÁ TỔ HỢP BLEND (XGBOOST + LIGHTGBM) TRÊN DỮ LIỆU ZONE MỚI")
        print("=" * 95)
        p_xgb = val_predictions["XGBoost"]
        p_lgb = val_predictions["LightGBM"]
        best_blend_f1 = 0.0
        best_blend_th = 0.5
        best_w = 0.5
        for w in np.linspace(0.35, 0.75, 9):
            p_blend = w * p_xgb + (1 - w) * p_lgb
            f1_b, th_b = optimize_threshold(y_va, p_blend)
            if f1_b > best_blend_f1:
                best_blend_f1 = f1_b
                best_blend_th = th_b
                best_w = w
        
        final_blend_pred = (best_w * p_xgb + (1 - best_w) * p_lgb >= best_blend_th).astype(int)
        prec_b = float(precision_score(y_va, final_blend_pred, zero_division=0))
        rec_b = float(recall_score(y_va, final_blend_pred, zero_division=0))
        
        gain_str = "🔥🚀 [BỨT PHÁ KỶ LỤC TOÀN ĐỀ TÀI]" if best_blend_f1 > 0.9346 else "Tiệm cận đỉnh cao"
        print(f"-> Tỷ lệ Blend Tối ưu: {best_w*100:.0f}% XGBoost + {(1-best_w)*100:.0f}% LightGBM")
        print(f"   * F1 Score Tối ưu: {best_blend_f1:.4f} (Ngưỡng tau* = {best_blend_th:.4f}) | {gain_str}")
        print(f"   * Precision: {prec_b:.4f} ({prec_b*100:.2f}%) | Recall: {rec_b:.4f} ({rec_b*100:.2f}%)")
        print("=" * 95)
        
        all_best_params["Ensemble_Zone_Blend"] = {
            "weights": {"XGBoost": round(float(best_w), 2), "LightGBM": round(float(1 - best_w), 2)},
            "best_f1": round(float(best_blend_f1), 4),
            "best_threshold": round(float(best_blend_th), 4),
            "precision_opt": round(float(prec_b), 4),
            "recall_opt": round(float(rec_b), 4)
        }
        with open(OUTPUT_JSON_HYPER, "w", encoding="utf-8") as f:
            json.dump(all_best_params, f, indent=4, ensure_ascii=False)
    
    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write("BẢNG TỔNG KẾT KẾT QUẢ TUNING ZONE FEATURES 43 CỘT\n")
        f.write("=" * 80 + "\n")
        for name, m in results_summary.items():
            f.write(f"Mô hình: {name}\n")
            f.write(f"  - F1 (tau*): {m['best_f1']:.4f}\n")
            f.write(f"  - Ngưỡng tối ưu: {m['best_threshold']:.4f}\n")
            f.write(f"  - Precision: {m['precision_opt']:.4f}\n")
            f.write(f"  - Recall: {m['recall_opt']:.4f}\n")
            f.write(f"  - Params: {m['params']}\n\n")
        if "Ensemble_Zone_Blend" in all_best_params:
            eb = all_best_params["Ensemble_Zone_Blend"]
            f.write(f"Ensemble Blend (XGB + LGB):\n")
            f.write(f"  - F1: {eb['best_f1']:.4f} (Trọng số: {eb['weights']})\n")
            f.write(f"  - Precision: {eb['precision_opt']:.4f} | Recall: {eb['recall_opt']:.4f}\n")
    print(f"-> Đã ghi file tổng kết chi tiết tại: {OUTPUT_TXT}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline Tối ưu Siêu tham số Zone Features (43 đặc trưng)")
    parser.add_argument("--model", type=str, default="both", choices=["both", "xgb", "lgb", "cat", "all"],
                        help="Chọn mô hình cần tuning: 'both' (mặc định: XGBoost + LightGBM), 'xgb', 'lgb', 'cat', hoặc 'all'")
    parser.add_argument("--trials", type=int, default=50,
                        help="Số lượng vòng lặp Optuna cho mỗi mô hình (mặc định: 50)")
    args = parser.parse_args()
    
    run_tuning(args.model, args.trials)
