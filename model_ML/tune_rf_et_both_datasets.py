import os
import sys
import time
import json
import warnings
from pathlib import Path

# Đảm bảo UTF-8 cho Windows Console
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8')
        if hasattr(sys.stderr, 'reconfigure'):
            sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

warnings.filterwarnings('ignore')

# Kích hoạt tăng tốc phần cứng Intel oneAPI / AVX vectorization cho Scikit-learn
try:
    from sklearnex import patch_sklearn
    patch_sklearn()
    sklearnex_active = True
except ImportError:
    sklearnex_active = False

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import precision_recall_curve, f1_score, precision_score, recall_score
import optuna

# Tắt log rác của optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

N_CORES = os.cpu_count() or 4
N_TRIALS = 10
TARGET = "anomaly"

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_BASELINE = BASE_DIR / "data" / "processed" / "train_preprocessed.parquet"
DATA_ZONE = BASE_DIR / "data" / "processed" / "train_preprocessedzone.parquet"

OUT_JSON = BASE_DIR / "model_ML" / "hyperparameters" / "rf_et_tuning_results.json"
OUT_MD = BASE_DIR / "model_ML" / "rf_et_benchmark_summary.md"

def print_banner():
    print("=" * 90)
    print("      HUẤN LUYỆN & TUNING RANDOM FOREST VÀ EXTRA TREES (BỘ DỮ LIỆU 1 & 2)")
    print("=" * 90)
    print(f"-> Phần cứng: CPU {N_CORES} Cores (n_jobs=-1)")
    print(f"-> Gia tốc oneAPI/AVX (sklearnex): {'Đã kích hoạt (Tối ưu hóa đa nhân)' if sklearnex_active else 'Không có'}")
    print(f"-> Số vòng lặp Tuning (Optuna Trials): {N_TRIALS} vòng/mô hình")
    print(f"-> Mục tiêu: Tối ưu F1-Score kết hợp dò tìm ngưỡng tối ưu (tau*) và so sánh với ngưỡng mặc định (0.5)")
    print(f"-> Nơi lưu kết quả: {OUT_JSON.name} & {OUT_MD.name}")
    print("=" * 90, flush=True)

def load_data(filepath, name):
    t0 = time.time()
    print(f"-> Đang nạp {name} ({filepath.name})...", end=" ", flush=True)
    df = pd.read_parquet(filepath)
    if 'timestamp' in df.columns:
        df = df.drop(columns=['timestamp'])
    X = df.drop(columns=[TARGET, "row_id"], errors="ignore")
    y = df[TARGET]
    print(f"Xong trong {time.time() - t0:.1f}s. Kích thước: {X.shape}")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Train: {X_train.shape[0]:,} dòng | Val: {X_val.shape[0]:,} dòng | Tỷ lệ dị thường: {y.mean()*100:.2f}%")
    return X_train, X_val, y_train, y_val

def find_optimal_threshold(y_true, y_prob):
    """Tính ngưỡng tối ưu cực đại hóa F1-Score qua precision_recall_curve"""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_th = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5
    best_f1 = float(f1_scores[best_idx])
    
    # Đo đạc chi tiết tại ngưỡng tối ưu
    y_pred_opt = (y_prob >= best_th).astype(int)
    prec_opt = float(precision_score(y_true, y_pred_opt, zero_division=0))
    rec_opt = float(recall_score(y_true, y_pred_opt, zero_division=0))
    
    # Đo đạc tại ngưỡng mặc định 0.5
    y_pred_05 = (y_prob >= 0.5).astype(int)
    f1_05 = float(f1_score(y_true, y_pred_05, zero_division=0))
    prec_05 = float(precision_score(y_true, y_pred_05, zero_division=0))
    rec_05 = float(recall_score(y_true, y_pred_05, zero_division=0))
    
    return {
        'optimal_threshold': round(best_th, 4),
        'f1_optimal': round(best_f1, 4),
        'precision_optimal': round(prec_opt, 4),
        'recall_optimal': round(rec_opt, 4),
        'f1_05': round(f1_05, 4),
        'precision_05': round(prec_05, 4),
        'recall_05': round(rec_05, 4)
    }

def run_experiment(model_type, dataset_name, X_train, X_val, y_train, y_val):
    print("\n" + "-" * 90)
    print(f"BẮT ĐẦU: {model_type} trên {dataset_name} ({N_TRIALS} Trials)")
    print("-" * 90, flush=True)
    
    trials_history = []
    
    def objective(trial):
        n_estimators = trial.suggest_int('n_estimators', 50, 120, step=10)
        max_depth = trial.suggest_int('max_depth', 12, 22)
        min_samples_split = trial.suggest_int('min_samples_split', 10, 50)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 5, 25)
        max_features = trial.suggest_float('max_features', 0.4, 0.8)
        max_samples = trial.suggest_float('max_samples', 0.5, 0.9)
        
        params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'max_features': max_features,
            'bootstrap': True,
            'max_samples': max_samples,
            'n_jobs': -1,
            'random_state': 42
        }
        
        t_start = time.time()
        if model_type == "Random Forest":
            clf = RandomForestClassifier(**params)
        else:
            clf = ExtraTreesClassifier(**params)
            
        clf.fit(X_train, y_train)
        fit_time = time.time() - t_start
        
        y_prob = clf.predict_proba(X_val)[:, 1]
        metrics = find_optimal_threshold(y_val, y_prob)
        metrics['fit_time_sec'] = round(fit_time, 2)
        metrics['trial_number'] = trial.number
        metrics['params'] = params
        
        trials_history.append(metrics)
        
        print(f"  [Trial {trial.number+1:02d}/{N_TRIALS:02d}] "
              f"Fit: {fit_time:5.1f}s | "
              f"F1(0.5): {metrics['f1_05']:.4f} | "
              f"tau*: {metrics['optimal_threshold']:.3f} -> "
              f"F1(tau*): {metrics['f1_optimal']:.4f} "
              f"(Prec: {metrics['precision_optimal']:.4f}, Rec: {metrics['recall_optimal']:.4f})",
              flush=True)
              
        trial.set_user_attr('metrics', metrics)
        return metrics['f1_optimal']

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    t_opt_start = time.time()
    study.optimize(objective, n_trials=N_TRIALS)
    total_tuning_time = time.time() - t_opt_start
    
    best_trial = study.best_trial
    best_metrics = best_trial.user_attrs['metrics']
    best_params = best_trial.params
    
    print(f"\n=> HOÀN TẤT: {model_type} ({dataset_name}) | Tổng thời gian tuning: {total_tuning_time:.1f}s")
    print(f"   Best Trial: #{best_trial.number+1} | F1(tau*): {best_metrics['f1_optimal']:.4f} | tau*: {best_metrics['optimal_threshold']:.4f}")
    print(f"   Precision: {best_metrics['precision_optimal']:.4f} | Recall: {best_metrics['recall_optimal']:.4f} | F1(0.5): {best_metrics['f1_05']:.4f}")
    print(f"   Best Params: {best_params}")
    
    return {
        'model': model_type,
        'dataset': dataset_name,
        'total_tuning_time_sec': round(total_tuning_time, 2),
        'best_trial_number': best_trial.number + 1,
        'best_fit_time_sec': best_metrics['fit_time_sec'],
        'optimal_threshold': best_metrics['optimal_threshold'],
        'precision_optimal': best_metrics['precision_optimal'],
        'recall_optimal': best_metrics['recall_optimal'],
        'f1_optimal': best_metrics['f1_optimal'],
        'precision_05': best_metrics['precision_05'],
        'recall_05': best_metrics['recall_05'],
        'f1_05': best_metrics['f1_05'],
        'best_params': best_params,
        'all_trials': trials_history
    }

def main():
    print_banner()
    
    # 1. Nạp Bộ dữ liệu 1 (Baseline)
    X_tr_base, X_va_base, y_tr_base, y_va_base = load_data(DATA_BASELINE, "Bộ dữ liệu 1 (Baseline)")
    
    # 2. Chạy Random Forest & Extra Trees trên Baseline
    res_rf_base = run_experiment("Random Forest", "Baseline", X_tr_base, X_va_base, y_tr_base, y_va_base)
    res_et_base = run_experiment("Extra Trees", "Baseline", X_tr_base, X_va_base, y_tr_base, y_va_base)
    
    # Dọn dẹp RAM trước khi nạp Bộ dữ liệu 2
    del X_tr_base, X_va_base, y_tr_base, y_va_base
    import gc
    gc.collect()
    
    # 3. Nạp Bộ dữ liệu 2 (Zone)
    X_tr_zone, X_va_zone, y_tr_zone, y_va_zone = load_data(DATA_ZONE, "Bộ dữ liệu 2 (Zone)")
    
    # 4. Chạy Random Forest & Extra Trees trên Zone
    res_rf_zone = run_experiment("Random Forest", "Zone", X_tr_zone, X_va_zone, y_tr_zone, y_va_zone)
    res_et_zone = run_experiment("Extra Trees", "Zone", X_tr_zone, X_va_zone, y_tr_zone, y_va_zone)
    
    del X_tr_zone, X_va_zone, y_tr_zone, y_va_zone
    gc.collect()
    
    all_results = [res_rf_base, res_et_base, res_rf_zone, res_et_zone]
    
    # 5. Lưu ra JSON
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)
    print(f"\n-> Đã lưu toàn bộ kết quả chi tiết vào: {OUT_JSON}")
    
    # 6. Tạo bảng so sánh Markdown & LaTeX
    summary_rows = []
    for r in all_results:
        summary_rows.append({
            "Mô hình": r['model'],
            "Bộ dữ liệu": r['dataset'],
            "Ngưỡng tau*": r['optimal_threshold'],
            "Precision (tau*)": r['precision_optimal'],
            "Recall (tau*)": r['recall_optimal'],
            "F1 (0.5)": r['f1_05'],
            "F1 (tau*)": r['f1_optimal'],
            "Thời gian fit (s)": r['best_fit_time_sec'],
            "Tổng thời gian tuning (s)": r['total_tuning_time_sec']
        })
    df_summary = pd.DataFrame(summary_rows)
    
    print("\n" + "=" * 90)
    print("                      BẢNG TỔNG HỢP KẾT QUẢ THỰC NGHIỆM")
    print("=" * 90)
    print(df_summary.to_string(index=False))
    print("=" * 90)
    
    # Ghi ra file Markdown
    md_content = f"""# Báo cáo Thực nghiệm Random Forest & Extra Trees sau Tuning (10 Trials)
- **Ngày thực hiện:** {time.strftime('%Y-%m-%d %H:%M:%S')}
- **Phần cứng:** CPU {N_CORES} Cores đa luồng, hỗ trợ Intel oneAPI (scikit-learn-intelex)
- **Quy trình:** 10 vòng lặp Optuna Bayesian TPE trên tập 80% Train / 20% Stratified Validation

| Mô hình | Bộ dữ liệu | Ngưỡng $\\tau^*$ | Precision | Recall | F1 (0.5) | F1 ($\\tau^*$) | Fit Time (s) | Tổng Tuning (s) |
|---|---|---|---|---|---|---|---|---|
"""
    for r in all_results:
        md_content += f"| **{r['model']}** | {r['dataset']} | {r['optimal_threshold']:.4f} | {r['precision_optimal']:.4f} | {r['recall_optimal']:.4f} | {r['f1_05']:.4f} | **{r['f1_optimal']:.4f}** | {r['best_fit_time_sec']:.1f}s | {r['total_tuning_time_sec']:.1f}s |\n"
        
    md_content += "\n## Chi tiết Siêu tham số Tối ưu nhất:\n"
    for r in all_results:
        md_content += f"### {r['model']} - {r['dataset']}:\n```json\n{json.dumps(r['best_params'], indent=2)}\n```\n"

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"-> Đã ghi báo cáo tóm tắt vào: {OUT_MD}")

if __name__ == "__main__":
    main()
