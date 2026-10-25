"""
PIPELINE TẠO FILE SUBMISSION CHUẨN XÁC CHO HỆ THỐNG LEAD
=============================================================================
Hỗ trợ tạo các file submission cho:
  1. [Mô hình hiệu năng cao nhất] Ensemble Zone Features (Weighted Blending: 0.65 XGB + 0.25 LGB + 0.10 CAT)
     - F1 đạt 0.9373 với 16 ngưỡng động theo từng Site ID (top 1 trong benchmark toàn diện).
  2. [XGBoost Zone] XGBoost Classifier với đặc trưng phân cụm và tương tác Zone
     - F1 đạt 0.9341 với ngưỡng động theo Site ID.
  3. [XGBoost Không Zone] XGBoost Classifier trên dữ liệu Baseline chuẩn hóa
     - F1 đạt 0.9349 với ngưỡng động theo Site ID.

Cách sử dụng:
  - Chạy cả 3 mô hình (mặc định):
      python model_ML/pipeline_full_train_and_submit.py
  - Chỉ chạy Model hiệu năng cao nhất:
      python model_ML/pipeline_full_train_and_submit.py --model best
  - Chỉ chạy XGBoost Zone:
      python model_ML/pipeline_full_train_and_submit.py --model xgb_zone
  - Chỉ chạy XGBoost Không Zone:
      python model_ML/pipeline_full_train_and_submit.py --model xgb_no_zone
  - Buộc huấn luyện lại từ đầu (không load model .pkl cũ):
      python model_ML/pipeline_full_train_and_submit.py --force-retrain
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import joblib
from typing import Any
from sklearn.metrics import f1_score

# Đảm bảo in ký tự tiếng Việt an toàn trên Windows Console / PowerShell
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

warnings.filterwarnings('ignore')

# Đảm bảo import được các module từ thư mục gốc
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from model_ML.model_classes import XGBoostModel, LightGBMModel, CatBoostModel

# ĐƯỜNG DẪN DỮ LIỆU VÀ THƯ MỤC
DATA_DIR = ROOT_DIR / "data" / "processed"
SUB_DIR = ROOT_DIR / "model_ML" / "submission"
SAVED_MODELS_DIR = ROOT_DIR / "model_ML" / "saved_new_models"
HYPERPARAMS_DIR = ROOT_DIR / "model_ML" / "hyperparameters"

SUB_DIR.mkdir(parents=True, exist_ok=True)
SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)

# SIÊU THAM SỐ TỐI ƯU TỪ OPTUNA BENCHMARK
BASELINE_PARAMS = {
    'XGBoost': {
        'learning_rate': 0.23187, 'max_depth': 15, 'subsample': 0.9524,
        'n_estimators': 140, 'n_jobs': -1, 'tree_method': 'hist',
        'eval_metric': 'logloss', 'random_state': 42
    },
    'LightGBM': {
        'learning_rate': 0.08591, 'num_leaves': 147, 'max_depth': 15,
        'feature_fraction': 0.6267, 'n_estimators': 150, 'n_jobs': -1,
        'random_state': 42, 'verbose': -1
    },
    'CatBoost': {
        'learning_rate': 0.32825, 'depth': 14, 'iterations': 150,
        'l2_leaf_reg': 1.4832, 'thread_count': -1, 'verbose': 0, 'random_state': 42
    }
}

ZONE_PARAMS = {
    'XGBoost': {
        'learning_rate': 0.22271, 'max_depth': 15, 'subsample': 0.9521,
        'n_estimators': 144, 'n_jobs': -1, 'tree_method': 'hist',
        'eval_metric': 'logloss', 'random_state': 42
    },
    'LightGBM': {
        'learning_rate': 0.08834, 'num_leaves': 168, 'max_depth': 16,
        'feature_fraction': 0.7605, 'n_estimators': 200, 'n_jobs': -1,
        'random_state': 42, 'verbose': -1
    },
    'CatBoost': {
        'learning_rate': 0.35090, 'depth': 14, 'iterations': 154,
        'l2_leaf_reg': 1.2227, 'thread_count': -1, 'verbose': 0, 'random_state': 42
    }
}

# TRỌNG SỐ BLENDING TỐI ƯU CỦA MÔ HÌNH HIỆU NĂNG CAO NHẤT (F1 = 0.9373)
BEST_ENSEMBLE_WEIGHTS = {
    'XGBoost': 0.65,
    'LightGBM': 0.25,
    'CatBoost': 0.10
}

# 16 NGƯỠNG ĐỘNG TỐI ƯU MẶC ĐỊNH CHO TỪNG SITE
DEFAULT_ZONE_DYNAMIC_THRESHOLDS = {
    'site_0': 0.1464, 'site_1': 0.2002, 'site_2': 0.2102, 'site_3': 0.3919,
    'site_4': 0.0841, 'site_5': 0.2236, 'site_7': 0.1518, 'site_8': 0.0620,
    'site_9': 0.3299, 'site_10': 0.2032, 'site_12': 0.9997, 'site_13': 0.2479,
    'site_14': 0.2933, 'site_15': 0.0756
}

DEFAULT_XGB_DYNAMIC_THRESHOLDS = {
    'site_0': 0.1019, 'site_1': 0.2699, 'site_2': 0.2659, 'site_3': 0.4548,
    'site_4': 0.3823, 'site_5': 0.1956, 'site_7': 0.1524, 'site_8': 0.0727,
    'site_9': 0.2874, 'site_10': 0.1262, 'site_12': 0.9996, 'site_13': 0.2872,
    'site_14': 0.2656, 'site_15': 0.0982
}


def load_dynamic_thresholds(method_name: str, default_dict: dict) -> dict:
    """Nạp ngưỡng động từ file json nếu có, ngược lại dùng dictionary mặc định."""
    json_path = HYPERPARAMS_DIR / "final_comprehensive_dynamic_thresholds.json"
    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if method_name in data:
                return data[method_name]
        except Exception as e:
            print(f"[CẢNH BÁO] Không đọc được file ngưỡng: {e}")
    return default_dict


def align_features(X: pd.DataFrame, model) -> pd.DataFrame:
    """Căn chỉnh đúng thứ tự và tên cột theo mô hình đã huấn luyện để tránh lỗi feature_names mismatch."""
    # Xử lý riêng cho LightGBM (vốn tự động thay dấu khoảng trắng và / thành dấu gạch dưới)
    if hasattr(model, 'booster_') and hasattr(model.booster_, 'feature_name'):
        lgb_cols = model.booster_.feature_name()
        name_map = {c.replace(' ', '_').replace('/', '_'): c for c in X.columns}
        orig_cols = [name_map.get(col, col) for col in lgb_cols]
        if all(c in X.columns for c in orig_cols):
            X_sub = X[orig_cols].copy()
            X_sub.columns = lgb_cols
            return X_sub

    expected = None
    if hasattr(model, 'feature_names_in_') and model.feature_names_in_ is not None:
        expected = list(model.feature_names_in_)
    elif hasattr(model, 'get_booster'):
        try:
            expected = model.get_booster().feature_names
        except Exception:
            pass
    elif hasattr(model, 'feature_names_'):
        expected = list(model.feature_names_)

    if expected is not None:
        valid_cols = [c for c in expected if c in X.columns]
        if len(valid_cols) == len(expected):
            return X[valid_cols]
    return X


def save_submission_csv(test_df: pd.DataFrame, probas: np.ndarray, out_file: Path, binary_pred: np.ndarray | None = None):
    """Lưu file submission chuẩn ROC-AUC (xác suất liên tục float [0.0 - 1.0]) và file nhị phân đối chứng kèm theo."""
    # Lấy row_id từ tập test_df hoặc từ sample_submission
    sample_path = ROOT_DIR / "data" / "sample_submission.csv"
    if sample_path.exists():
        sample_df = pd.read_csv(sample_path)
        row_ids = sample_df['row_id'].values
    elif 'row_id' in test_df.columns:
        row_ids = test_df['row_id'].values
    else:
        row_ids = np.arange(len(probas))

    # 1. FILE NỘP CHUẨN ROC-AUC: Cột anomaly là XÁC SUẤT LIÊN TỤC FLOAT [0.0 - 1.0]
    sub_roc_df = pd.DataFrame({
        'row_id': row_ids,
        'anomaly': np.clip(probas, 0.0, 1.0).round(6)
    })
    sub_roc_df.to_csv(out_file, index=False)

    total_rows = len(sub_roc_df)
    file_size_mb = out_file.stat().st_size / (1024 * 1024)
    mean_prob = float(sub_roc_df['anomaly'].mean())
    p95_prob = float(sub_roc_df['anomaly'].quantile(0.95))
    p99_prob = float(sub_roc_df['anomaly'].quantile(0.99))

    print(f" -> ĐÃ XUẤT SUBMISSION CHUẨN ROC-AUC: {out_file.name}")
    print(f"    Định dạng: Xác suất liên tục float [0.0 - 1.0] | Tổng dòng: {total_rows:,} | Dung lượng: {file_size_mb:.2f} MB")
    print(f"    Phân phối xác suất: Mean={mean_prob:.5f} | P95={p95_prob:.5f} | P99={p99_prob:.5f}")

    # 2. FILE NHỊ PHÂN KÈM THEO (_binary.csv): Phục vụ phân tích ngưỡng và F1-score offline
    if binary_pred is not None:
        bin_file = out_file.with_name(out_file.stem + "_binary.csv")
        sub_bin_df = pd.DataFrame({
            'row_id': row_ids,
            'anomaly': binary_pred.astype(int)
        })
        sub_bin_df.to_csv(bin_file, index=False)
        anom_count = int(sub_bin_df['anomaly'].sum())
        anom_pct = (anom_count / total_rows) * 100
        print(f"    [Kèm file nhị phân F1]: {bin_file.name} | Dị thường (1): {anom_count:,} ({anom_pct:.2f}%)")


def apply_dynamic_thresholds(probas: np.ndarray, site_ids: Any, site_thresholds: dict, global_th: float) -> np.ndarray:
    """Áp dụng ngưỡng động theo từng Site ID cho mảng xác suất."""
    y_pred = np.zeros_like(probas, dtype=int)
    unique_sites = np.unique(site_ids)

    for site in unique_sites:
        mask = (site_ids == site)
        site_key = f"site_{site}" if not str(site).startswith("site_") else str(site)
        th = site_thresholds.get(site_key, site_thresholds.get(str(site), global_th))
        y_pred[mask] = (probas[mask] >= th).astype(int)

    return y_pred


def get_trained_model(model_name: str, model_class, params: dict, X_train: pd.DataFrame, y_train: pd.Series, pkl_path: Path, force_retrain: bool = False):
    """Nạp mô hình đã lưu nếu có, hoặc huấn luyện mới nếu chưa có / force_retrain."""
    if pkl_path.exists() and not force_retrain:
        try:
            print(f" -> [NẠP SẴN] Đang tải mô hình {model_name} từ {pkl_path.name}...")
            model = joblib.load(pkl_path)
            return model
        except Exception as e:
            print(f" -> [CẢNH BÁO] Không thể tải {pkl_path.name}: {e}. Huấn luyện lại...")

    print(f" -> [HUẤN LUYỆN] Đang train FULL 100% {model_name}...")
    wrapper = model_class(**params)
    t0 = time.time()
    wrapper.train(X_train, y_train)
    elapsed = time.time() - t0
    print(f"    Hoàn tất huấn luyện {model_name} trong {elapsed:.1f}s.")
    joblib.dump(wrapper.model, pkl_path)
    print(f"    Đã lưu mô hình vào: {pkl_path.name}")
    return wrapper.model


def run_xgboost_no_zone(force_retrain: bool = False):
    """Pipeline XGBoost Không Zone (Baseline)."""
    print("\n" + "=" * 80)
    print("1. PIPELINE: XGBOOST KHÔNG ZONE (BASELINE)")
    print("=" * 80)

    train_path = DATA_DIR / "train_preprocessed.parquet"
    test_path = DATA_DIR / "test_preprocessed.parquet"
    out_csv = SUB_DIR / "submission_xgboost_no_zone.csv"
    pkl_path = SAVED_MODELS_DIR / "XGBoost_Baseline_Full.pkl"

    print(f"Đang nạp dữ liệu: {train_path.name}...")
    df_train = pd.read_parquet(train_path)
    X_train = df_train.drop(columns=['anomaly', 'timestamp'], errors='ignore')
    y_train = df_train['anomaly']

    model = get_trained_model(
        model_name="XGBoost Baseline",
        model_class=XGBoostModel,
        params=BASELINE_PARAMS['XGBoost'],
        X_train=X_train,
        y_train=y_train,
        pkl_path=pkl_path,
        force_retrain=force_retrain
    )

    print(f"Đang nạp tập test: {test_path.name} và dự đoán xác suất...")
    df_test = pd.read_parquet(test_path)
    X_test = df_test.drop(columns=['anomaly', 'timestamp', 'row_id'], errors='ignore')
    X_test_aligned = align_features(X_test, model)
    probas = model.predict_proba(X_test_aligned)
    if len(probas.shape) > 1:
        probas = probas[:, 1]

    # Áp dụng ngưỡng động theo site
    site_thresholds = load_dynamic_thresholds("XGBoost", DEFAULT_XGB_DYNAMIC_THRESHOLDS)
    y_pred = apply_dynamic_thresholds(probas, X_test['site_id'].values, site_thresholds, global_th=0.303)

    save_submission_csv(df_test, probas=probas, out_file=out_csv, binary_pred=y_pred)
    return out_csv


def run_xgboost_zone(force_retrain: bool = False):
    """Pipeline XGBoost Có Zone."""
    print("\n" + "=" * 80)
    print("2. PIPELINE: XGBOOST CÓ ZONE (ZONE FEATURES)")
    print("=" * 80)

    train_path = DATA_DIR / "train_preprocessedzone.parquet"
    test_path = DATA_DIR / "test_preprocessedzone.parquet"
    out_csv = SUB_DIR / "submission_xgboost_zone.csv"
    pkl_path = SAVED_MODELS_DIR / "XGBoost_Zone.pkl"

    print(f"Đang nạp dữ liệu: {train_path.name}...")
    df_train = pd.read_parquet(train_path)
    X_train = df_train.drop(columns=['anomaly', 'timestamp'], errors='ignore')
    y_train = df_train['anomaly']

    model = get_trained_model(
        model_name="XGBoost Zone",
        model_class=XGBoostModel,
        params=ZONE_PARAMS['XGBoost'],
        X_train=X_train,
        y_train=y_train,
        pkl_path=pkl_path,
        force_retrain=force_retrain
    )

    print(f"Đang nạp tập test: {test_path.name} và dự đoán xác suất...")
    df_test = pd.read_parquet(test_path)
    X_test = df_test.drop(columns=['anomaly', 'timestamp', 'row_id'], errors='ignore')
    X_test_aligned = align_features(X_test, model)
    probas = model.predict_proba(X_test_aligned)
    if len(probas.shape) > 1:
        probas = probas[:, 1]

    # Áp dụng ngưỡng động theo site
    site_thresholds = load_dynamic_thresholds("XGBoost", DEFAULT_XGB_DYNAMIC_THRESHOLDS)
    y_pred = apply_dynamic_thresholds(probas, X_test['site_id'].values, site_thresholds, global_th=0.263)

    save_submission_csv(df_test, probas=probas, out_file=out_csv, binary_pred=y_pred)
    return out_csv


def run_best_model_ensemble_zone(force_retrain: bool = False):
    """Pipeline Model Hiệu Năng Cao Nhất: Ensemble Zone Features (Weighted Blending: 0.65 XGB + 0.25 LGB + 0.10 CAT)."""
    print("\n" + "=" * 80)
    print("3. PIPELINE: MODEL HIỆU NĂNG CAO NHẤT (ENSEMBLE ZONE - F1 = 0.9373)")
    print("=" * 80)

    train_path = DATA_DIR / "train_preprocessedzone.parquet"
    test_path = DATA_DIR / "test_preprocessedzone.parquet"
    out_csv = SUB_DIR / "submission_best_model.csv"

    print(f"Đang nạp dữ liệu: {train_path.name}...")
    df_train = pd.read_parquet(train_path)
    X_train = df_train.drop(columns=['anomaly', 'timestamp'], errors='ignore')
    y_train = df_train['anomaly']

    # 1. Nạp / Huấn luyện bộ ba mô hình Zone
    models = {}
    model_configs = [
        ("XGBoost", XGBoostModel, ZONE_PARAMS['XGBoost'], SAVED_MODELS_DIR / "XGBoost_Zone.pkl"),
        ("LightGBM", LightGBMModel, ZONE_PARAMS['LightGBM'], SAVED_MODELS_DIR / "LightGBM_Zone.pkl"),
        ("CatBoost", CatBoostModel, ZONE_PARAMS['CatBoost'], SAVED_MODELS_DIR / "CatBoost_Zone.pkl"),
    ]

    for m_name, cls, prms, pkl in model_configs:
        models[m_name] = get_trained_model(
            model_name=f"{m_name} (Zone)",
            model_class=cls,
            params=prms,
            X_train=X_train,
            y_train=y_train,
            pkl_path=pkl,
            force_retrain=force_retrain
        )

    # 2. Dự đoán trên tập Test
    print(f"\nĐang nạp tập test: {test_path.name}...")
    df_test = pd.read_parquet(test_path)
    X_test = df_test.drop(columns=['anomaly', 'timestamp', 'row_id'], errors='ignore')

    test_probas = {}
    for m_name, m in models.items():
        print(f" -> Dự đoán xác suất với {m_name}...")
        X_test_aligned = align_features(X_test, m)
        p = m.predict_proba(X_test_aligned)
        if len(p.shape) > 1:
            p = p[:, 1]
        test_probas[m_name] = p

    # 3. Kết hợp trọng số tối ưu: 0.65 XGBoost + 0.25 LightGBM + 0.10 CatBoost
    w_xgb = BEST_ENSEMBLE_WEIGHTS['XGBoost']
    w_lgb = BEST_ENSEMBLE_WEIGHTS['LightGBM']
    w_cat = BEST_ENSEMBLE_WEIGHTS['CatBoost']
    print(f"\n-> Áp dụng Trọng số Blending tối ưu: XGB={w_xgb:.2f}, LGB={w_lgb:.2f}, CAT={w_cat:.2f}")

    ensemble_proba = (
        w_xgb * test_probas['XGBoost'] +
        w_lgb * test_probas['LightGBM'] +
        w_cat * test_probas['CatBoost']
    )

    # 4. Áp dụng 16 ngưỡng động tối ưu theo từng Site ID
    ensemble_key = "Weighted Blending LogLoss [0.65 0.25 0.1 ]"
    site_thresholds = load_dynamic_thresholds(ensemble_key, DEFAULT_ZONE_DYNAMIC_THRESHOLDS)
    print("-> Áp dụng 16 ngưỡng động theo từng Site ID để tối đa hóa F1...")
    y_pred = apply_dynamic_thresholds(ensemble_proba, X_test['site_id'].values, site_thresholds, global_th=0.232)

    save_submission_csv(df_test, probas=ensemble_proba, out_file=out_csv, binary_pred=y_pred)
    return out_csv


def main():
    parser = argparse.ArgumentParser(description="Pipeline tạo file submission LEAD Anomaly Detection")
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["all", "best", "xgb_zone", "xgb_no_zone"],
        help="Lựa chọn mô hình để chạy: 'all' (cả 3), 'best' (mô hình cao nhất), 'xgb_zone', 'xgb_no_zone'"
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Buộc huấn luyện lại từ đầu thay vì nạp file model .pkl đã lưu sẵn"
    )
    args = parser.parse_args()

    t_start = time.time()
    print("=" * 80)
    print("     KHỞI CHẠY PIPELINE TẠO FILE SUBMISSION CHO BÀI TOÁN LEAD")
    print("=" * 80)
    print(f"Chế độ chạy: --model {args.model}")
    print(f"Force retrain: {args.force_retrain}")

    generated_files = []

    if args.model in ["all", "xgb_no_zone"]:
        f_no_zone = run_xgboost_no_zone(force_retrain=args.force_retrain)
        generated_files.append(("XGBoost Không Zone (Baseline)", f_no_zone))

    if args.model in ["all", "xgb_zone"]:
        f_zone = run_xgboost_zone(force_retrain=args.force_retrain)
        generated_files.append(("XGBoost Có Zone (Zone Features)", f_zone))

    if args.model in ["all", "best"]:
        f_best = run_best_model_ensemble_zone(force_retrain=args.force_retrain)
        generated_files.append(("Model Hiệu Năng Cao Nhất (Ensemble Zone)", f_best))

    total_time = time.time() - t_start

    print("\n" + "=" * 80)
    print("                    TỔNG KẾT KẾT QUẢ TẠO SUBMISSION")
    print("=" * 80)
    print(f"Thời gian hoàn tất: {total_time:.2f} giây")
    print("\nDanh sách các file submission đã sẵn sàng nộp hệ thống:")
    for label, path in generated_files:
        print(f"  + [{label}]:\n    --> {path}")
    print("\nThư mục chứa các file: " + str(SUB_DIR))
    print("=" * 80)


if __name__ == '__main__':
    main()
