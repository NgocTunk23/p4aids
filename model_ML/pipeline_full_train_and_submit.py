import pandas as pd
import numpy as np
import os
import joblib
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from pathlib import Path
import sys

sys.path.append(os.path.abspath('.'))
from model_ML.model_classes import XGBoostModel, LightGBMModel, CatBoostModel
import warnings
warnings.filterwarnings('ignore')

# CHỈ ĐỊNH THAM SỐ CỤ THỂ CHO CẢ BASELINE VÀ ZONE
baseline_params = {
    'XGBoost': {'learning_rate': 0.23187303509421126, 'max_depth': 15, 'subsample': 0.9524355862143259, 'n_estimators': 141, 'random_state': 42},
    'LightGBM': {'learning_rate': 0.0859101054837918, 'num_leaves': 147, 'max_depth': 15, 'feature_fraction': 0.6267447348759597, 'n_estimators': 100, 'random_state': 42, 'verbose': -1},
    'CatBoost': {'learning_rate': 0.3282464476846472, 'depth': 14, 'iterations': 159, 'l2_leaf_reg': 1.483228351433837, 'random_state': 42, 'verbose': 0}
}

# Tham số Zone lấy đúng từ model_ML/run_advanced_ensemble.py (dòng 113-115)
zone_params = {
    'XGBoost': {'learning_rate': 0.2227117196758846, 'max_depth': 15, 'subsample': 0.9521445174611336, 'n_estimators': 144, 'random_state': 42},
    'LightGBM': {'learning_rate': 0.08833924691306173, 'num_leaves': 168, 'max_depth': 16, 'feature_fraction': 0.7604568174122006, 'n_estimators': 200, 'random_state': 42, 'verbose': -1},
    'CatBoost': {'learning_rate': 0.35090221036945446, 'depth': 14, 'iterations': 154, 'l2_leaf_reg': 1.2226533778340363, 'random_state': 42, 'verbose': 0}
}

# Trọng số Ensemble [XGB, LGB, CAT] lấy từ advanced_ensemble_report_new.md
# (Không tối ưu lại trên tập train vì XGB overfit -> luôn ra 100% XGB)
ENSEMBLE_WEIGHTS = {
    'Baseline': np.array([0.83, 0.00, 0.17]),
    'Zone': np.array([0.81, 0.04, 0.15]),
}

# Số fold OOF dùng để tính NGƯỠNG (model nộp bài vẫn train 100% dữ liệu)
N_FOLDS_OOF = 5

MODEL_CLASSES = {'XGBoost': XGBoostModel, 'LightGBM': LightGBMModel, 'CatBoost': CatBoostModel}

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
    return best_th

def get_dynamic_thresholds(y_true, proba, site_ids):
    val_df = pd.DataFrame({'site_id': site_ids, 'y_true': y_true, 'y_prob': proba})
    zone_thresholds = {}
    for site in val_df['site_id'].unique():
        site_data = val_df[val_df['site_id'] == site]
        if site_data['y_true'].sum() == 0:
            zone_thresholds[site] = 1.0 
        else:
            zone_thresholds[site] = find_best_threshold(site_data['y_true'], site_data['y_prob'])
    global_th = find_best_threshold(y_true, proba)
    return zone_thresholds, global_th

def get_oof_probas(X, y, model_params_dict, n_folds=N_FOLDS_OOF):
    """Out-of-fold: mỗi dòng được dự đoán bởi model KHÔNG thấy dòng đó khi train.
    Chỉ dùng để tìm ngưỡng, không dùng làm model nộp bài."""
    oof = {m: np.zeros(len(X)) for m in MODEL_CLASSES}
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
        print(f"    [OOF] Fold {fold}/{n_folds}...")
        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        X_va = X.iloc[va_idx]
        for m_name, cls in MODEL_CLASSES.items():
            m = cls(**model_params_dict.get(m_name, {}))
            m.train(X_tr, y_tr)
            p = m.predict_proba(X_va)
            if len(p.shape) > 1: p = p[:, 1]
            oof[m_name][va_idx] = p
    return oof

def create_submission(test_df, proba, test_sites, zone_th, global_th, out_path):
    y_pred = np.zeros_like(proba)
    for site, th in zone_th.items():
        mask = (test_sites == site)
        y_pred[mask] = (proba[mask] >= th).astype(int)
    missing_mask = ~np.isin(test_sites, list(zone_th.keys()))
    if missing_mask.sum() > 0:
        y_pred[missing_mask] = (proba[missing_mask] >= global_th).astype(int)
        
    sample_df = pd.read_csv("data/sample_submission.csv")
    sub_df = pd.DataFrame({'row_id': sample_df['row_id'], 'anomaly': y_pred.astype(int)})
    sub_df.to_csv(out_path, index=False)
    print(f" -> LƯU KẾT QUẢ TẠI: {out_path} | Số dị thường (1): {sub_df['anomaly'].sum()}")

def process_pipeline(name_prefix, train_parquet, test_parquet, model_params_dict):
    print(f"\n{'='*80}\n[PIPELINE] BẮT ĐẦU XỬ LÝ CỤM: {name_prefix}\n{'='*80}")
    
    data_path = Path("data/processed")
    sub_dir = Path("submission")
    model_dir = Path("model_submission")
    
    sub_dir.mkdir(exist_ok=True)
    model_dir.mkdir(exist_ok=True)
    
    print(f"1. Nạp dữ liệu {name_prefix} (100% DATA)...")
    df_train = pd.read_parquet(data_path / train_parquet)
    if 'timestamp' in df_train.columns: df_train.drop(columns=['timestamp'], inplace=True)
    X = df_train.drop(columns=['anomaly'])
    y = df_train['anomaly']
    
    models = {m_name: cls(**model_params_dict.get(m_name, {})) for m_name, cls in MODEL_CLASSES.items()}
    
    print("2. HUẤN LUYỆN TRỰC TIẾP TRÊN 100% DỮ LIỆU (model dùng để nộp)...")
    for m_name, model in models.items():
        print(f" -> Đang train FULL {m_name}...")
        model.train(X, y)
        
        # Save model ngay sau khi train full
        m_path = model_dir / f"{name_prefix}_{m_name}_Full.pkl"
        joblib.dump(model.model, m_path)
        print(f"    Đã lưu mô hình vào {m_path}")
        
    print(f"3. Tính NGƯỠNG động bằng {N_FOLDS_OOF}-fold OOF trên 100% dữ liệu...")
    oof_probas = get_oof_probas(X, y, model_params_dict)
    
    opt_weights = ENSEMBLE_WEIGHTS[name_prefix]
    print(f"    Trọng số Ensemble {name_prefix} (cố định từ report): XGB={opt_weights[0]:.2f}, LGB={opt_weights[1]:.2f}, CAT={opt_weights[2]:.2f}")
    oof_matrix = np.column_stack([oof_probas['XGBoost'], oof_probas['LightGBM'], oof_probas['CatBoost']])
    oof_probas['Ensemble'] = np.dot(oof_matrix, opt_weights)
    
    thresholds_info = {}
    for m_name, p in oof_probas.items():
        z_th, g_th = get_dynamic_thresholds(y.values, p, X['site_id'].values)
        thresholds_info[m_name] = (z_th, g_th)
        f1_g = f1_score(y.values, (p >= g_th).astype(int), zero_division=0)
        print(f"    {m_name}: Ngưỡng chung={g_th:.2f} | OOF F1 (global)={f1_g:.4f}")

    print("\n4. Nạp dữ liệu TEST và Dự đoán...")
    test_df = pd.read_parquet(data_path / test_parquet)
    cols_to_drop = ['anomaly', 'timestamp', 'row_id']
    X_test = test_df.drop(columns=[c for c in cols_to_drop if c in test_df.columns])
    test_sites = X_test['site_id'].values
    
    test_probas = {}
    for m_name, model in models.items():
        print(f" -> Dự đoán tập TEST với {m_name}...")
        p = model.predict_proba(X_test)
        if len(p.shape) > 1: p = p[:, 1]
        test_probas[m_name] = p
        
        z_th, g_th = thresholds_info[m_name]
        out_path = sub_dir / f"submission_{name_prefix.lower()}_{m_name.lower()}.csv"
        create_submission(test_df, p, test_sites, z_th, g_th, out_path)
        
    print(f" -> Dự đoán tập TEST với Ensemble...")
    test_probas['Ensemble'] = (test_probas['XGBoost']*opt_weights[0] + 
                               test_probas['LightGBM']*opt_weights[1] + 
                               test_probas['CatBoost']*opt_weights[2])
    
    z_th, g_th = thresholds_info['Ensemble']
    out_path = sub_dir / f"submission_{name_prefix.lower()}_ensemble.csv"
    create_submission(test_df, test_probas['Ensemble'], test_sites, z_th, g_th, out_path)

def main():
    # 1. Chạy Baseline Pipeline
    process_pipeline(
        name_prefix="Baseline", 
        train_parquet="train_preprocessed.parquet", 
        test_parquet="test_preprocessed.parquet", 
        model_params_dict=baseline_params
    )
    
    # 2. Chạy Zone Pipeline
    process_pipeline(
        name_prefix="Zone", 
        train_parquet="train_preprocessedzone.parquet", 
        test_parquet="test_preprocessedzone.parquet", 
        model_params_dict=zone_params
    )
    
    print("\n[HOÀN TẤT TẤT CẢ PIPELINE]")
    print("Mời bạn kiểm tra thư mục 'submission' để lấy các file csv nộp hệ thống")
    print("Các mô hình train 100% pkl được lưu trong 'model_submission'")

if __name__ == '__main__':
    main()
