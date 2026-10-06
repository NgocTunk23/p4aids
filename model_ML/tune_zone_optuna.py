import os
import sys
import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_recall_curve

# Import model_classes
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model_ML.model_classes import XGBoostModel, LightGBMModel, CatBoostModel

import warnings
warnings.filterwarnings('ignore')

def optimize_threshold(y_true, y_prob):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)
    return f1_scores[best_idx]

def load_zone_data():
    df = pd.read_parquet("data/processed/train_preprocessedzone.parquet")
    if 'timestamp' in df.columns:
        df = df.drop(columns=['timestamp'])
    X = df.drop(columns=['anomaly'])
    y = df['anomaly']
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

X_train, X_val, y_train, y_val = load_zone_data()

def objective_lgb(trial):
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.2, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 130, 170),
        'max_depth': trial.suggest_int('max_depth', 12, 17),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 0.8),
        'n_estimators': 200, # Tăng nhẹ estimators
        'n_jobs': -1,
        'random_state': 42,
        'verbose': -1
    }
    model = LightGBMModel(**params)
    model.train(X_train, y_train)
    preds = model.predict_proba(X_val)
    if len(preds.shape) > 1 and preds.shape[1] == 2:
        preds = preds[:, 1]
    return optimize_threshold(y_val, preds)

def objective_xgb(trial):
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.1, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 13, 16),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'n_estimators': trial.suggest_int('n_estimators', 140, 160),
        'n_jobs': -1,
        'eval_metric': 'logloss',
        'random_state': 42
    }
    model = XGBoostModel(**params)
    model.train(X_train, y_train)
    preds = model.predict_proba(X_val)
    if len(preds.shape) > 1 and preds.shape[1] == 2:
        preds = preds[:, 1]
    return optimize_threshold(y_val, preds)

def objective_cat(trial):
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.1, 0.4, log=True),
        'depth': trial.suggest_int('depth', 10, 14),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 3.0),
        'iterations': trial.suggest_int('iterations', 140, 170),
        'thread_count': -1,
        'verbose': 0,
        'random_state': 42
    }
    model = CatBoostModel(**params)
    model.train(X_train, y_train)
    preds = model.predict_proba(X_val)
    if len(preds.shape) > 1 and preds.shape[1] == 2:
        preds = preds[:, 1]
    return optimize_threshold(y_val, preds)

if __name__ == "__main__":
    # print("=== TUNE LIGHTGBM (Mục tiêu đánh bại 0.8836) ===")
    # study_lgb = optuna.create_study(direction="maximize")
    # study_lgb.optimize(objective_lgb, n_trials=10)
    # print("LightGBM Best Params:", study_lgb.best_params)
    # print("LightGBM Best F1:", study_lgb.best_value)

    # print("\n=== TUNE XGBOOST (Mục tiêu đánh bại 0.9291) ===")
    # study_xgb = optuna.create_study(direction="maximize")
    # study_xgb.optimize(objective_xgb, n_trials=10)
    # print("XGBoost Best Params:", study_xgb.best_params)
    # print("XGBoost Best F1:", study_xgb.best_value)
    
    print("\n=== TUNE CATBOOST (Mục tiêu đánh bại 0.9162) ===")
    print("(CatBoost chạy khá lâu nên chỉ tune 5 vòng)")
    study_cat = optuna.create_study(direction="maximize")
    study_cat.optimize(objective_cat, n_trials=10)
    print("CatBoost Best Params:", study_cat.best_params)
    print("CatBoost Best F1:", study_cat.best_value)
    
    # Ghi nhận kết quả
    with open("model_ML/zone_tuning_results_cat.txt", "w") as f:
        f.write(f"CatBoost Best F1: {study_cat.best_value}\n")
        f.write(f"CatBoost Best Params: {study_cat.best_params}\n")
