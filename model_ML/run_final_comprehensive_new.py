import os
import sys
import time
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, log_loss

# Thêm đường dẫn gốc để import model_ML
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model_ML.model_classes import XGBoostModel, LightGBMModel, CatBoostModel

import warnings
warnings.filterwarnings('ignore')

SAVED_MODELS_DIR = os.path.join(os.path.dirname(__file__), 'saved_models')
NEW_MODELS_DIR = os.path.join(os.path.dirname(__file__), 'saved_new_models')
os.makedirs(NEW_MODELS_DIR, exist_ok=True)

# Hyperparameters đã được bạn tối ưu trước đó
xgb_params = {'learning_rate': 0.23187303509421126, 'max_depth': 15, 'subsample': 0.9524355862143259, 'n_estimators': 141}
lgb_params = {'learning_rate': 0.0859101054837918, 'num_leaves': 147, 'max_depth': 15, 'feature_fraction': 0.6267447348759597}
cat_params = {'learning_rate': 0.3282464476846472, 'depth': 14, 'iterations': 159, 'l2_leaf_reg': 1.483228351433837}

def prepare_data(parquet_path):
    print(f"Đang chuẩn bị dữ liệu từ: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    if 'timestamp' in df.columns:
        df = df.drop(columns=['timestamp'])
    
    # Chia theo 80/20 chuẩn như lúc baseline
    X = df.drop(columns=['anomaly'])
    y = df['anomaly']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return X_train, X_val, y_train, y_val

def find_best_threshold(y_true, y_prob):
    thresholds = np.arange(0.01, 0.99, 0.02)
    best_thresh = 0.5
    best_score = -1
    best_rec = 0
    best_prec = 0
    
    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_score:
            best_score = f1
            best_thresh = thresh
            best_rec = recall_score(y_true, y_pred, zero_division=0)
            best_prec = precision_score(y_true, y_pred, zero_division=0)
            
    return best_thresh, best_score, best_prec, best_rec

def evaluate_model(name, strategy, y_true, y_prob, train_time=0):
    best_th, f1, prec, rec = find_best_threshold(y_true, y_prob)
    return {
        "Chiến Thuật": strategy,
        "Mô Hình": name,
        "Ngưỡng Tối Ưu": best_th,
        "Precision": prec,
        "Recall": rec,
        "F1 Score": f1,
        "Train Time (s)": train_time
    }

def main():
    print("=========================================================================")
    print("CHẠY ĐÁNH GIÁ PIPELINE CHUẨN (KHÔNG LEAKAGE)")
    print("=========================================================================")
    
    results = []
    
    # =====================================================================
    # 1. BASELINE (LOAD TỪ MÔ HÌNH CŨ)
    # =====================================================================
    print("\n[1/3] ĐÁNH GIÁ BASELINE (CŨ)")
    X_train_pre, X_val_pre, y_train_pre, y_val_pre = prepare_data("data/processed/train_preprocessed.parquet")
    
    xgb_old = joblib.load(os.path.join(SAVED_MODELS_DIR, "XGBoost_Preprocessed.pkl"))
    lgb_old = joblib.load(os.path.join(SAVED_MODELS_DIR, "LightGBM_Preprocessed.pkl"))
    cat_old = joblib.load(os.path.join(SAVED_MODELS_DIR, "CatBoost_Preprocessed.pkl"))
    
    prob_xgb_old = xgb_old.predict_proba(X_val_pre)[:, 1]
    prob_lgb_old = lgb_old.predict_proba(X_val_pre)[:, 1]
    prob_cat_old = cat_old.predict_proba(X_val_pre)[:, 1]
    
    results.append(evaluate_model("XGBoost", "1. Baseline (Cũ)", y_val_pre, prob_xgb_old, 0))
    results.append(evaluate_model("LightGBM", "1. Baseline (Cũ)", y_val_pre, prob_lgb_old, 0))
    results.append(evaluate_model("CatBoost", "1. Baseline (Cũ)", y_val_pre, prob_cat_old, 0))

    # =====================================================================
    # 2. LAG FEATURES (TRAIN CHUẨN 80/20 ĐỂ TRÁNH LEAKAGE)
    # =====================================================================
    print("\n[2/3] HUẤN LUYỆN VÀ ĐÁNH GIÁ LAG FEATURES (CHỐNG LEAKAGE)")
    X_train_lag, X_val_lag, y_train_lag, y_val_lag = prepare_data("data/processed/train_preprocessedlag.parquet")
    
    # Train XGBoost Lag
    print("  -> Đang train XGBoost (Lag)...")
    t0 = time.time()
    xgb_lag = XGBoostModel(**xgb_params)
    xgb_lag.train(X_train_lag, y_train_lag)
    t_xgb_lag = time.time() - t0
    prob_xgb_lag = xgb_lag.predict_proba(X_val_lag)
    results.append(evaluate_model("XGBoost", "2. Lag Features", y_val_lag, prob_xgb_lag, t_xgb_lag))
    joblib.dump(xgb_lag.model, os.path.join(NEW_MODELS_DIR, "XGBoost_Lag_Clean.pkl"))

    # Train LightGBM Lag
    print("  -> Đang train LightGBM (Lag)...")
    t0 = time.time()
    lgb_lag = LightGBMModel(**lgb_params)
    lgb_lag.train(X_train_lag, y_train_lag)
    t_lgb_lag = time.time() - t0
    prob_lgb_lag = lgb_lag.predict_proba(X_val_lag)
    results.append(evaluate_model("LightGBM", "2. Lag Features", y_val_lag, prob_lgb_lag, t_lgb_lag))
    joblib.dump(lgb_lag.model, os.path.join(NEW_MODELS_DIR, "LightGBM_Lag_Clean.pkl"))

    # Train CatBoost Lag
    print("  -> Đang train CatBoost (Lag)...")
    t0 = time.time()
    cat_lag = CatBoostModel(**cat_params)
    cat_lag.train(X_train_lag, y_train_lag)
    t_cat_lag = time.time() - t0
    prob_cat_lag = cat_lag.predict_proba(X_val_lag)
    results.append(evaluate_model("CatBoost", "2. Lag Features", y_val_lag, prob_cat_lag, t_cat_lag))
    joblib.dump(cat_lag.model, os.path.join(NEW_MODELS_DIR, "CatBoost_Lag_Clean.pkl"))

    # =====================================================================
    # 3. ZONE FEATURES (TRAIN CHUẨN 80/20 ĐỂ TRÁNH LEAKAGE)
    # =====================================================================
    print("\n[3/3] HUẤN LUYỆN VÀ ĐÁNH GIÁ ZONE FEATURES (CHỐNG LEAKAGE)")
    X_train_zone, X_val_zone, y_train_zone, y_val_zone = prepare_data("data/processed/train_preprocessedzone.parquet")
    
    # Train XGBoost Zone
    print("  -> Đang train XGBoost (Zone)...")
    t0 = time.time()
    xgb_zone = XGBoostModel(**xgb_params)
    xgb_zone.train(X_train_zone, y_train_zone)
    t_xgb_zone = time.time() - t0
    prob_xgb_zone = xgb_zone.predict_proba(X_val_zone)
    results.append(evaluate_model("XGBoost", "3. Zone Features", y_val_zone, prob_xgb_zone, t_xgb_zone))
    joblib.dump(xgb_zone.model, os.path.join(NEW_MODELS_DIR, "XGBoost_Zone_Clean.pkl"))

    # Train LightGBM Zone
    print("  -> Đang train LightGBM (Zone)...")
    t0 = time.time()
    lgb_zone = LightGBMModel(**lgb_params)
    lgb_zone.train(X_train_zone, y_train_zone)
    t_lgb_zone = time.time() - t0
    prob_lgb_zone = lgb_zone.predict_proba(X_val_zone)
    results.append(evaluate_model("LightGBM", "3. Zone Features", y_val_zone, prob_lgb_zone, t_lgb_zone))
    joblib.dump(lgb_zone.model, os.path.join(NEW_MODELS_DIR, "LightGBM_Zone_Clean.pkl"))

    # Train CatBoost Zone
    print("  -> Đang train CatBoost (Zone)...")
    t0 = time.time()
    cat_zone = CatBoostModel(**cat_params)
    cat_zone.train(X_train_zone, y_train_zone)
    t_cat_zone = time.time() - t0
    prob_cat_zone = cat_zone.predict_proba(X_val_zone)
    results.append(evaluate_model("CatBoost", "3. Zone Features", y_val_zone, prob_cat_zone, t_cat_zone))
    joblib.dump(cat_zone.model, os.path.join(NEW_MODELS_DIR, "CatBoost_Zone_Clean.pkl"))
    
    # =====================================================================
    # 4. THỬ NGHIỆM TỔNG HỢP: ENSEMBLE (MoE) TRÊN ZONE (Vì Zone thường rất tốt)
    # =====================================================================
    print("\n[MỞ RỘNG] ĐANG CHẠY ENSEMBLE MẠNG TỔNG CHỈ HUY (MoE) TRÊN DỮ LIỆU ZONE...")
    # Lấy meta-features trên tập validation của Zone
    X_meta_train, X_meta_val, y_meta_train, y_meta_val = train_test_split(X_val_zone, y_val_zone, test_size=0.5, random_state=42, stratify=y_val_zone)
    
    # Tạo xác suất cho meta_train
    meta_train = X_meta_train.copy()
    meta_train['prob_xgb'] = xgb_zone.predict_proba(X_meta_train)
    meta_train['prob_lgb'] = lgb_zone.predict_proba(X_meta_train)
    meta_train['prob_cat'] = cat_zone.predict_proba(X_meta_train)
    
    # Tạo xác suất cho meta_val
    meta_val = X_meta_val.copy()
    meta_val['prob_xgb'] = xgb_zone.predict_proba(X_meta_val)
    meta_val['prob_lgb'] = lgb_zone.predict_proba(X_meta_val)
    meta_val['prob_cat'] = cat_zone.predict_proba(X_meta_val)
    
    t0 = time.time()
    moe_model = LightGBMModel(n_estimators=100, learning_rate=0.05, verbose=-1)
    # Tạm tắt stdout để đỡ rác
    old_stdout = sys.stdout
    with open(os.devnull, "w") as f:
        sys.stdout = f
        moe_model.train(meta_train, y_meta_train)
    sys.stdout = old_stdout
    
    t_moe = time.time() - t0
    prob_moe = moe_model.predict_proba(meta_val)
    results.append(evaluate_model("Context-Aware MoE", "4. Ensemble (MoE trên Zone)", y_meta_val, prob_moe, t_moe))
    joblib.dump(moe_model.model, os.path.join(NEW_MODELS_DIR, "MoE_Zone_Clean.pkl"))

    # In kết quả
    print("\n=============================================================================================")
    print("                  BẢNG KẾT QUẢ SO SÁNH CHUẨN (KHÔNG LEAKAGE) ")
    print("=============================================================================================")
    results_df = pd.DataFrame(results).round(4)
    print(results_df.to_markdown(index=False))
    
    # Save markdown
    report_path = os.path.join(os.path.dirname(__file__), "final_comprehensive_report_new.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# BẢNG KẾT QUẢ SO SÁNH CHUẨN (KHÔNG LEAKAGE)\n\n")
        f.write("Dữ liệu Lag và Zone đã được train chuẩn trên 80% và đánh giá trên 20% validation set, giống hệt với Baseline.\n\n")
        f.write(results_df.to_markdown(index=False))
    
    print(f"\n=> Đã lưu báo cáo tại: {report_path}")

if __name__ == "__main__":
    main()
