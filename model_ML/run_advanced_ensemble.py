import pandas as pd
import numpy as np
import os
import joblib
import gc
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, log_loss, recall_score, precision_score
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize
import warnings
import re

warnings.filterwarnings('ignore')

# Thư mục lưu mô hình Ensemble
ENSEMBLE_DIR = Path("model_ML/saved_ensembles")
ENSEMBLE_DIR.mkdir(parents=True, exist_ok=True)

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model_ML.model_classes import XGBoostModel, LightGBMModel, CatBoostModel

def find_best_threshold(y_true, proba):
    thresholds = np.arange(0.01, 0.99, 0.01)
    best_score = -1
    best_th = 0.5
    for th in thresholds:
        pred = (proba >= th).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_score:
            best_score = f1
            best_th = th
    return best_th, best_score

def get_dynamic_f1(y_true, proba, site_ids):
    val_df = pd.DataFrame({'site_id': site_ids, 'y_true': y_true, 'y_prob': proba})
    y_pred_dynamic = np.zeros_like(proba)
    for site in val_df['site_id'].unique():
        mask = val_df['site_id'] == site
        site_data = val_df[mask]
        if site_data['y_true'].sum() == 0:
            best_th = 1.0
        else:
            best_th, _ = find_best_threshold(site_data['y_true'], site_data['y_prob'])
        y_pred_dynamic[mask] = (site_data['y_prob'] >= best_th).astype(int)
    return f1_score(y_true, y_pred_dynamic, zero_division=0)

def record_result(results_list, name, method_name, proba, y_true, site_ids):
    pred_default = (proba >= 0.5).astype(int)
    score_default = f1_score(y_true, pred_default, zero_division=0)
    
    best_th, best_score = find_best_threshold(y_true, proba)
    
    dynamic_score = get_dynamic_f1(y_true, proba, site_ids)
    
    results_list.append({
        "Tổ Hợp": name,
        "Phương pháp": method_name,
        "Ngưỡng Chung": round(best_th, 4),
        "F1 (Ngưỡng 0.5)": round(score_default, 4),
        "F1 (Ngưỡng Chung)": round(best_score, 4),
        "F1 (Ngưỡng Động 16 Khu)": round(dynamic_score, 4)
    })

def main():
    print("="*80)
    print("BƯỚC 1: CHUẨN BỊ DỮ LIỆU BASELINE & ZONE")
    print("="*80)
    
    data_path = Path("data/processed")
    
    # 1. BASELINE DATA
    print("Đang nạp dữ liệu Baseline...")
    df_pre = pd.read_parquet(data_path / "train_preprocessed.parquet")
    if 'timestamp' in df_pre.columns: df_pre.drop(columns=['timestamp'], inplace=True)
    X_pre = df_pre.drop(columns=['anomaly'])
    y_pre = df_pre['anomaly']
    _, X_val, _, y_val = train_test_split(X_pre, y_pre, test_size=0.2, random_state=42, stratify=y_pre)
    
    # 2. ZONE DATA
    print("Đang nạp dữ liệu Zone...")
    df_zone = pd.read_parquet(data_path / "train_preprocessedzone.parquet")
    if 'timestamp' in df_zone.columns: df_zone.drop(columns=['timestamp'], inplace=True)
    X_z = df_zone.drop(columns=['anomaly'])
    y_z = df_zone['anomaly']
    X_train_z, X_val_z, y_train_z, y_val_z = train_test_split(X_z, y_z, test_size=0.2, random_state=42, stratify=y_z)
    
    print("\n" + "="*80)
    print("BƯỚC 2: TẢI & HUẤN LUYỆN CÁC MÔ HÌNH")
    print("="*80)
    
    models_dir = Path("model_ML/saved_models")
    
    print("1. Đang tải các mô hình Baseline...")
    xgb_model = joblib.load(models_dir / "XGBoost_Preprocessed.pkl")
    lgb_model = joblib.load(models_dir / "LightGBM_Preprocessed.pkl")
    cat_model = joblib.load(models_dir / "CatBoost_Preprocessed.pkl")
    
    xgb_wrapper = XGBoostModel(); xgb_wrapper.model = xgb_model
    lgb_wrapper = LightGBMModel(); lgb_wrapper.model = lgb_model
    cat_wrapper = CatBoostModel(); cat_wrapper.model = cat_model
    
    p_xgb = xgb_wrapper.predict_proba(X_val)
    p_lgb = lgb_wrapper.predict_proba(X_val)
    p_cat = cat_wrapper.predict_proba(X_val)
    
    if len(p_xgb.shape) > 1: p_xgb = p_xgb[:, 1]
    if len(p_lgb.shape) > 1: p_lgb = p_lgb[:, 1]
    if len(p_cat.shape) > 1: p_cat = p_cat[:, 1]
    
    print("2. Đang huấn luyện các mô hình Zone với tham số Tối ưu nhất...")
    xgb_zone_params = {'learning_rate': 0.2227117196758846, 'max_depth': 15, 'subsample': 0.9521445174611336, 'n_estimators': 144, 'random_state': 42}
    lgb_zone_params = {'learning_rate': 0.08833924691306173, 'num_leaves': 168, 'max_depth': 16, 'feature_fraction': 0.7604568174122006, 'n_estimators': 200, 'random_state': 42, 'verbose': -1}
    cat_zone_params = {'learning_rate': 0.35090221036945446, 'depth': 14, 'l2_leaf_reg': 1.2226533778340363, 'iterations': 154, 'random_state': 42, 'verbose': 0}
    
    xgb_zone = XGBoostModel(**xgb_zone_params)
    xgb_zone.train(X_train_z, y_train_z)
    joblib.dump(xgb_zone.model, models_dir / "XGBoost_Zone.pkl")
    print(f"  -> Đã lưu XGBoost Zone tại: {models_dir / 'XGBoost_Zone.pkl'}")
    p_xgb_z = xgb_zone.predict_proba(X_val_z)
    if len(p_xgb_z.shape) > 1: p_xgb_z = p_xgb_z[:, 1]
    
    lgb_zone = LightGBMModel(**lgb_zone_params)
    lgb_zone.train(X_train_z, y_train_z)
    joblib.dump(lgb_zone.model, models_dir / "LightGBM_Zone.pkl")
    print(f"  -> Đã lưu LightGBM Zone tại: {models_dir / 'LightGBM_Zone.pkl'}")
    p_lgb_z = lgb_zone.predict_proba(X_val_z)
    if len(p_lgb_z.shape) > 1: p_lgb_z = p_lgb_z[:, 1]
    
    print("  -> Đang huấn luyện CatBoost Zone (Depth 14, sẽ hơi lâu chút)...")
    cat_zone = CatBoostModel(**cat_zone_params)
    cat_zone.train(X_train_z, y_train_z)
    joblib.dump(cat_zone.model, models_dir / "CatBoost_Zone.pkl")
    print(f"  -> Đã lưu CatBoost Zone tại: {models_dir / 'CatBoost_Zone.pkl'}")
    p_cat_z = cat_zone.predict_proba(X_val_z)
    if len(p_cat_z.shape) > 1: p_cat_z = p_cat_z[:, 1]
    
    print("\n" + "="*80)
    print("BƯỚC 3: ĐÁNH GIÁ (BASELINE vs ZONE) VỚI NGƯỠNG TĨNH VÀ NGƯỠNG ĐỘNG")
    print("="*80)
    
    results = []
    y_true = y_val.values
    site_ids = X_val['site_id'].values
    
    print("Đang đánh giá các mô hình đơn lẻ...")
    record_result(results, "Baseline Đơn lẻ", "XGBoost", p_xgb, y_true, site_ids)
    record_result(results, "Baseline Đơn lẻ", "LightGBM", p_lgb, y_true, site_ids)
    record_result(results, "Baseline Đơn lẻ", "CatBoost", p_cat, y_true, site_ids)
    
    record_result(results, "Zone Đơn lẻ", "XGBoost", p_xgb_z, y_true, site_ids)
    record_result(results, "Zone Đơn lẻ", "LightGBM", p_lgb_z, y_true, site_ids)
    record_result(results, "Zone Đơn lẻ", "CatBoost", p_cat_z, y_true, site_ids)
    
    # Chuẩn bị cho Ensemble
    def do_weighted_blending(proba_matrix, name):
        def loss_blend(w):
            w = w / w.sum()
            p = np.dot(proba_matrix, w)
            return log_loss(y_true, p)
        init_w = np.ones(proba_matrix.shape[1]) / proba_matrix.shape[1]
        bounds = [(0, 1)] * proba_matrix.shape[1]
        res_b = minimize(loss_blend, init_w, bounds=bounds, method='SLSQP')
        best_wb = res_b.x / res_b.x.sum()
        blend_proba = np.dot(proba_matrix, best_wb)
        record_result(results, name, f"Weighted Blending {np.round(best_wb, 2)}", blend_proba, y_true, site_ids)

    print("Đang đánh giá Ensemble (Weighted Blending)...")
    do_weighted_blending(np.column_stack([p_xgb, p_lgb, p_cat]), "Baseline Ensemble (Cả 3)")
    do_weighted_blending(np.column_stack([p_xgb, p_cat]), "Baseline Ensemble (XGB+Cat)")
    do_weighted_blending(np.column_stack([p_xgb_z, p_lgb_z, p_cat_z]), "Zone Ensemble (Cả 3)")
    do_weighted_blending(np.column_stack([p_xgb_z, p_cat_z]), "Zone Ensemble (XGB+Cat)")
    
    print("\n" + "="*80)
    print("BƯỚC 4: XUẤT BÁO CÁO")
    print("="*80)
    
    import tabulate
    df_res = pd.DataFrame(results)
    df_res = df_res.sort_values(by=["F1 (Ngưỡng Động 16 Khu)"], ascending=False)
    md_table = tabulate.tabulate(df_res, headers='keys', tablefmt='pipe', showindex=False)
    
    report_path = Path("model_ML/advanced_ensemble_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# BÁO CÁO TỔNG HỢP CUỐI CÙNG: MÔ HÌNH vs DỮ LIỆU vs NGƯỠNG ĐỘNG\n\n")
        f.write(md_table)
        f.write("\n")
        
    print(md_table)
    print(f"\n=> Đã lưu báo cáo siêu tổng hợp vào: {report_path}")

if __name__ == "__main__":
    main()
