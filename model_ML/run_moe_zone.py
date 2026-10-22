import os
import sys

# Đảm bảo UTF-8 cho Windows Console để in tiếng Việt có dấu
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, 'reconfigure'):
            getattr(sys.stdout, 'reconfigure')(encoding='utf-8')
        if hasattr(sys.stderr, 'reconfigure'):
            getattr(sys.stderr, 'reconfigure')(encoding='utf-8')
    except Exception:
        pass

print("=" * 95, flush=True)
print("     CHUYÊN BIỆT: CONTEXT-AWARE MoE (META-LEARNER) VỚI THAM SỐ LIGHTGBM TỐI ƯU", flush=True)
print("=" * 95, flush=True)
print("-> Đang khởi động và nạp các mô hình Zone đã huấn luyện...", flush=True)

import time
import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, log_loss

# Thêm đường dẫn gốc để import model_classes
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model_ML.model_classes import LightGBMModel

import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------------
# CẤU HÌNH ĐƯỜNG DẪN
# ---------------------------------------------------------------------------------
script_dir = Path(__file__).resolve().parent
DATA_ZONE = script_dir.parent / "data" / "processed" / "train_preprocessedzone.parquet"
NEW_MODELS_DIR = script_dir / "saved_new_models"
TUNING_FILE = script_dir / "zone_tuning_best_params.json"
THRESHOLDS_OUTPUT = script_dir / "moe_dynamic_site_thresholds.json"
REPORT_OUTPUT = script_dir / "moe_zone_report.md"

TARGET = "anomaly"

# ---------------------------------------------------------------------------------
# 1. ĐỌC SIÊU THAM SỐ LIGHTGBM TỐI ƯU TỪ OPTUNA
# ---------------------------------------------------------------------------------
default_lgb_params = {
    'learning_rate': 0.157659,
    'num_leaves': 252,
    'max_depth': 16,
    'feature_fraction': 0.572062,
    'min_child_samples': 73,
    'n_estimators': 215,
    'n_jobs': -1,
    'random_state': 42,
    'verbose': -1
}

if TUNING_FILE.exists():
    try:
        with open(TUNING_FILE, "r", encoding="utf-8") as f:
            tuned_data = json.load(f)
        if "LightGBM" in tuned_data and "params" in tuned_data["LightGBM"]:
            default_lgb_params.update(tuned_data["LightGBM"]["params"])
            print(f"-> [THÀNH CÔNG] Đã nạp siêu tham số LightGBM tối ưu từ {TUNING_FILE.name}:")
            for k, v in tuned_data["LightGBM"]["params"].items():
                print(f"   + {k}: {v}")
    except Exception as e:
        print(f"-> [CẢNH BÁO] Không thể đọc {TUNING_FILE.name}: {e}. Dùng thông số tối ưu mặc định.")
else:
    print(f"-> [LƯU Ý] Chưa tìm thấy {TUNING_FILE.name}. Dùng thông số tối ưu mặc định.")

default_lgb_params['random_state'] = 42
default_lgb_params['n_jobs'] = -1
default_lgb_params['verbose'] = -1

# ---------------------------------------------------------------------------------
# 2. HÀM TÌM NGƯỠNG TỐI ƯU VÀ NGƯỠNG ĐỘNG THEO KHU VỰC
# ---------------------------------------------------------------------------------
def find_best_threshold(y_true, y_prob):
    """Quét các ngưỡng xác suất từ 0.01 đến 0.99 để tìm ngưỡng tối đa hóa F1-score"""
    thresholds = np.arange(0.01, 0.99, 0.01)
    best_th = 0.5
    best_f1 = -1.0
    best_prec = 0.0
    best_rec = 0.0
    
    for th in thresholds:
        pred = (y_prob >= th).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_th = th
            best_prec = precision_score(y_true, pred, zero_division=0)
            best_rec = recall_score(y_true, pred, zero_division=0)
            
    return float(best_th), float(best_f1), float(best_prec), float(best_rec)

# ---------------------------------------------------------------------------------
# 3. CHƯƠNG TRÌNH CHÍNH
# ---------------------------------------------------------------------------------
def main():
    t_start = time.time()
    
    # BƯỚC 1: Nạp các mô hình Base Zone đã lưu sẵn
    print("\n" + "=" * 90)
    print("BƯỚC 1: TẢI BỘ 3 MÔ HÌNH ZONE ĐÃ HUẤN LUYỆN (XGBoost, LightGBM, CatBoost)")
    print("=" * 90)
    xgb_path = NEW_MODELS_DIR / "XGBoost_Zone.pkl"
    lgb_path = NEW_MODELS_DIR / "LightGBM_Zone.pkl"
    cat_path = NEW_MODELS_DIR / "CatBoost_Zone.pkl"
    
    if not (xgb_path.exists() and lgb_path.exists() and cat_path.exists()):
        raise FileNotFoundError(
            f"Thiếu file model trong {NEW_MODELS_DIR}! Cần có: "
            f"XGBoost_Zone.pkl, LightGBM_Zone.pkl, CatBoost_Zone.pkl"
        )
        
    print(f"-> Đang nạp XGBoost Zone từ {xgb_path.name}...", end=" ", flush=True)
    xgb_zone = joblib.load(xgb_path)
    print("OK")
    
    print(f"-> Đang nạp LightGBM Zone từ {lgb_path.name}...", end=" ", flush=True)
    lgb_zone = joblib.load(lgb_path)
    print("OK")
    
    print(f"-> Đang nạp CatBoost Zone từ {cat_path.name}...", end=" ", flush=True)
    cat_zone = joblib.load(cat_path)
    print("OK")
    
    # BƯỚC 2: Nạp dữ liệu Zone và phân chia tập Meta-Train / Meta-Val
    print("\n" + "=" * 90)
    print("BƯỚC 2: NẠP DỮ LIỆU ZONE FEATURES (PyArrow đa luồng)")
    print("=" * 90)
    t0 = time.time()
    df = pd.read_parquet(DATA_ZONE, engine="pyarrow")
    if 'timestamp' in df.columns:
        df = df.drop(columns=['timestamp'])
    X = df.drop(columns=[TARGET, "row_id"], errors="ignore")
    y = df[TARGET]
    print(f"-> Dữ liệu tải xong trong {time.time() - t0:.1f}s. Kích thước: {X.shape}")
    
    # Chia Train (80%) và Validation (20%) cố định giống quy trình chuẩn
    _, X_val, _, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Chia tập Validation thành Meta-Train (50%) và Meta-Val (50%)
    X_meta_tr, X_meta_va, y_meta_tr, y_meta_va = train_test_split(
        X_val, y_val, test_size=0.5, random_state=42, stratify=y_val
    )
    site_ids_meta_val = X_meta_va['site_id'].values if 'site_id' in X_meta_va else None
    
    print(f"-> Tập Meta-Train: {X_meta_tr.shape[0]:,} dòng")
    print(f"-> Tập Meta-Val:   {X_meta_va.shape[0]:,} dòng")
    
    # BƯỚC 3: Sinh xác suất dự đoán từ 3 mô hình Base
    print("\n" + "=" * 90)
    print("BƯỚC 3: TRÍCH XUẤT XÁC SUẤT DỰ ĐOÁN (PROBABILITIES) LÀM ĐẶC TRƯNG META")
    print("=" * 90)
    print("-> Đang sinh xác suất trên tập Meta-Train...", end=" ", flush=True)
    t0 = time.time()
    meta_tr = X_meta_tr.copy()
    meta_tr['prob_xgb'] = xgb_zone.predict_proba(X_meta_tr)[:, 1]
    meta_tr['prob_lgb'] = lgb_zone.predict_proba(X_meta_tr)[:, 1]
    meta_tr['prob_cat'] = cat_zone.predict_proba(X_meta_tr)[:, 1]
    print(f"Xong ({time.time() - t0:.1f}s)")
    
    print("-> Đang sinh xác suất trên tập Meta-Val...", end=" ", flush=True)
    t0 = time.time()
    meta_va = X_meta_va.copy()
    meta_va['prob_xgb'] = xgb_zone.predict_proba(X_meta_va)[:, 1]
    meta_va['prob_lgb'] = lgb_zone.predict_proba(X_meta_va)[:, 1]
    meta_va['prob_cat'] = cat_zone.predict_proba(X_meta_va)[:, 1]
    print(f"Xong ({time.time() - t0:.1f}s)")
    
    # BƯỚC 4: Huấn luyện Meta-Learner bằng tham số LightGBM tối ưu
    print("\n" + "=" * 90)
    print("BƯỚC 4: HUẤN LUYỆN CONTEXT-AWARE MoE VỚI SIÊU THAM SỐ LIGHTGBM TỐI ƯU")
    print("=" * 90)
    t0 = time.time()
    moe_model = LightGBMModel(**default_lgb_params)
    moe_model.train(meta_tr, y_meta_tr)
    t_train = time.time() - t0
    print(f"-> Huấn luyện Meta-Learner hoàn tất trong {t_train:.1f}s!")
    
    # Lưu mô hình MoE tối ưu
    moe_save_path = NEW_MODELS_DIR / "MoE_Zone_Optimal.pkl"
    joblib.dump(moe_model.model, moe_save_path)
    print(f"-> Đã lưu mô hình MoE tối ưu tại: {moe_save_path}")
    
    # BƯỚC 5: Đánh giá và Tìm Bộ 16 Ngưỡng Động
    print("\n" + "=" * 90)
    print("BƯỚC 5: ĐÁNH GIÁ VÀ XÁC ĐỊNH BỘ 16 NGƯỠNG ĐỘNG CHI TIẾT TỪNG SITE/ZONE")
    print("=" * 90)
    p_moe = moe_model.predict_proba(meta_va)
    y_true_va = y_meta_va.values
    
    # 5.1 Đánh giá ngưỡng tĩnh 0.5
    pred_05 = (p_moe >= 0.5).astype(int)
    f1_05 = float(f1_score(y_true_va, pred_05, zero_division=0))
    prec_05 = float(precision_score(y_true_va, pred_05, zero_division=0))
    rec_05 = float(recall_score(y_true_va, pred_05, zero_division=0))
    
    # 5.2 Đánh giá ngưỡng tối ưu toàn cục
    best_global_th, best_global_f1, prec_global, rec_global = find_best_threshold(y_true_va, p_moe)
    
    # 5.3 Xác định chi tiết từng ngưỡng động cho 16 Site
    site_details = []
    site_thresholds_dict = {}
    val_df = pd.DataFrame({'site_id': site_ids_meta_val, 'y_true': y_true_va, 'y_prob': p_moe})
    y_pred_dynamic = np.zeros_like(p_moe)
    
    sorted_sites = sorted(val_df['site_id'].unique())
    for site in sorted_sites:
        mask = val_df['site_id'] == site
        s_data = val_df[mask]
        s_total = len(s_data)
        s_anom = int(s_data['y_true'].sum())
        s_rate = (s_anom / s_total * 100) if s_total > 0 else 0.0
        
        if s_anom == 0:
            s_th = 1.0
            s_f1 = 1.0
        else:
            s_th, s_f1, _, _ = find_best_threshold(s_data['y_true'].values, s_data['y_prob'].values)
            
        y_pred_dynamic[mask] = (s_data['y_prob'] >= s_th).astype(int)
        site_thresholds_dict[f"site_{site}"] = round(float(s_th), 4)
        
        site_details.append({
            "Site ID": int(site),
            "Số mẫu": f"{s_total:,}",
            "Số dị thường": f"{s_anom:,}",
            "Tỷ lệ (%)": f"{s_rate:.2f}%",
            "Ngưỡng Động": round(float(s_th), 4),
            "F1-Score": round(float(s_f1), 4)
        })
        
    dynamic_f1_overall = float(f1_score(y_true_va, y_pred_dynamic, zero_division=0))
    dynamic_prec_overall = float(precision_score(y_true_va, y_pred_dynamic, zero_division=0))
    dynamic_rec_overall = float(recall_score(y_true_va, y_pred_dynamic, zero_division=0))
    
    # In bảng tổng kết hiệu năng MoE
    print("\n" + "-" * 90)
    print("                    KẾT QUẢ ĐÁNH GIÁ HIỆU NĂNG CONTEXT-AWARE MoE")
    print("-" * 90)
    overview_df = pd.DataFrame([
        {
            "Chiến lược Ngưỡng": "Mặc định (0.500)",
            "Ngưỡng áp dụng": "0.5000",
            "F1-Score": round(f1_05, 4),
            "Precision": round(prec_05, 4),
            "Recall": round(rec_05, 4)
        },
        {
            "Chiến lược Ngưỡng": "Ngưỡng Tối Ưu Toàn Cục",
            "Ngưỡng áp dụng": f"{best_global_th:.4f}",
            "F1-Score": round(best_global_f1, 4),
            "Precision": round(prec_global, 4),
            "Recall": round(rec_global, 4)
        },
        {
            "Chiến lược Ngưỡng": "⭐ Ngưỡng Động 16 Khu Vực",
            "Ngưỡng áp dụng": "Từng Site riêng biệt",
            "F1-Score": round(dynamic_f1_overall, 4),
            "Precision": round(dynamic_prec_overall, 4),
            "Recall": round(dynamic_rec_overall, 4)
        }
    ])
    try:
        print(overview_df.to_markdown(index=False))
    except Exception:
        print(overview_df.to_string(index=False))
        
    # In bảng chi tiết 16 Site
    print("\n" + "-" * 90)
    print("           BẢNG CHI TIẾT BỘ 16 NGƯỠNG ĐỘNG TỐI ƯU THEO TỪNG SITE/ZONE")
    print("-" * 90)
    sites_df = pd.DataFrame(site_details)
    try:
        print(sites_df.to_markdown(index=False))
    except Exception:
        print(sites_df.to_string(index=False))
        
    # BƯỚC 6: Xuất file JSON bộ 16 ngưỡng động và Báo cáo Markdown
    export_payload = {
        "model_name": "Context-Aware MoE (Stacking Meta-Learner)",
        "meta_learner_type": "LightGBM",
        "hyperparameters": default_lgb_params,
        "performance_summary": {
            "f1_at_05": round(f1_05, 4),
            "global_optimal_threshold": round(best_global_th, 4),
            "f1_at_global_optimal": round(best_global_f1, 4),
            "dynamic_f1_overall": round(dynamic_f1_overall, 4),
            "dynamic_precision_overall": round(dynamic_prec_overall, 4),
            "dynamic_recall_overall": round(dynamic_rec_overall, 4)
        },
        "dynamic_site_thresholds": site_thresholds_dict
    }
    
    with open(THRESHOLDS_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(export_payload, f, indent=4, ensure_ascii=False)
    print(f"\n=> ĐÃ XUẤT THÀNH CÔNG BỘ 16 NGƯỠNG ĐỘNG VÀO: {THRESHOLDS_OUTPUT}")
    
    # Xuất Markdown
    with open(REPORT_OUTPUT, "w", encoding="utf-8") as f:
        f.write("# BÁO CÁO HIỆU NĂNG CONTEXT-AWARE MoE & BỘ 16 NGƯỠNG ĐỘNG\n\n")
        f.write(f"- **Mô hình Meta-Learner:** LightGBM với siêu tham số tối ưu Optuna\n")
        f.write(f"- **Thời gian huấn luyện:** {t_train:.1f}s | **Tổng thời gian pipeline:** {time.time() - t_start:.1f}s\n\n")
        f.write("### 1. Bảng Tổng Hợp Hiệu Năng:\n\n")
        try:
            f.write(overview_df.to_markdown(index=False))
        except Exception:
            f.write(overview_df.to_string(index=False))
        f.write("\n\n### 2. Chi Tiết Bộ 16 Ngưỡng Động Theo Từng Site/Zone:\n\n")
        try:
            f.write(sites_df.to_markdown(index=False))
        except Exception:
            f.write(sites_df.to_string(index=False))
        f.write(f"\n\n- File JSON ngưỡng động cho inference: `{THRESHOLDS_OUTPUT.name}`\n")
        f.write(f"- File model Meta-Learner đã lưu: `{moe_save_path.name}`\n")
        
    print(f"=> ĐÃ XUẤT BÁO CÁO CHI TIẾT TẠI: {REPORT_OUTPUT}")
    print(f"=> Hoàn tất pipeline chỉ trong {time.time() - t_start:.1f}s!")

if __name__ == "__main__":
    main()
