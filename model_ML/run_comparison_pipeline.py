import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, f1_score, log_loss, brier_score_loss
from sklearn.preprocessing import OrdinalEncoder
from model_classes import XGBoostModel, LightGBMModel, CatBoostModel, RandomForestModel, ExtraTreesModel
import os

# Đường dẫn tuyệt đối dựa trên vị trí của file script này
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)

datasets = {
    "Preprocessed (Chưa qua FE)": os.path.join(project_dir, "data", "processed", "train_preprocessed.parquet"),
    "Engineered (Đã qua FE)": os.path.join(project_dir, "data", "processed", "train_engineered.parquet")
}

# HƯỚNG DẪN ĐIỀU CHỈNH SIÊU THAM SỐ (HYPERPARAMETER TUNING):
# Bạn chỉ cần truyền các tham số (arguments) trực tiếp vào trong ngoặc () của Class.
# Bất kỳ tham số nào bạn truyền vào đây sẽ ĐÈ LÊN tham số mặc định của hệ thống.
models = [
    # {'learning_rate': 0.2588966101472009, 'max_depth': 12, 'subsample': 0.9469427996754303, 'n_estimators': 148}
    # XGBoost 0.8882
    XGBoostModel(learning_rate=0.2588966101472009, max_depth=12, subsample= 0.9469427996754303, n_estimators=148),
    
    # {'learning_rate': 0.12327192527974386, 'num_leaves': 130, 'max_depth': 13, 'feature_fraction': 0.7025615258894086}
    # LightGBM 0.8204
    LightGBMModel(learning_rate=0.12327192527974386, max_depth=13, num_leaves=130, feature_fraction=0.7025615258894086),
    
    # {'learning_rate': 0.2859403421268123, 'depth': 12, 'iterations': 145, 'l2_leaf_reg': 2.0604935809714204}
    # CatBoost 0.8361
    CatBoostModel(learning_rate=0.2859403421268123, depth=12, iterations=145, l2_leaf_reg=2.0604935809714204),
    
    # Random Forest
    RandomForestModel(max_depth=10, n_estimators=100, min_samples_split=7),
    
    # Extra Trees
    ExtraTreesModel(max_depth=10, n_estimators=100, min_samples_split=7)
]

# (Mẹo nâng cao: Nếu bạn dùng Optuna / GridSearch, bạn có thể gọi model.update_params(learning_rate=0.01) giữa các vòng lặp)

results = []

for ds_name, ds_path in datasets.items():
    print(f"\n{'='*60}")
    print(f"ĐANG CHẠY TRÊN BỘ DỮ LIỆU: {ds_name}")
    print(f"{'='*60}")
    
    try:
        df = pd.read_parquet(ds_path, engine="fastparquet")
    except Exception as e:
        print(f"Lỗi nạp dữ liệu {ds_path}: {e}")
        continue
        
    TARGET = "anomaly"
    X = df.drop(columns=[TARGET, "row_id", "timestamp"], errors="ignore")
    y = df[TARGET]
    
    print("Điền khuyết các giá trị NaN còn sót lại bằng 0...")
    X = X.fillna(0)
    
    print("Chia tập Train/Validation (80/20) với stratify...")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    X_train_res, y_train_res = X_train, y_train
    
    for model in models:
        try:
            model.train(X_train_res, y_train_res)
            metrics = model.evaluate(X_val, y_val)
            
            # Thêm tên Dataset vào dict kết quả
            res_dict = {"Tập dữ liệu": ds_name.split(" ")[0]}
            res_dict.update({
                "Mô hình": metrics["Model"],
                "Custom Score": metrics["Custom Score"],
                "F2-Score (Nặng Recall)": metrics["F2-Score"],
                "Recall (Độ phủ)": metrics["Recall"],
                "Custom Score": metrics["Custom Score"],
                "Log Loss (Càng nhỏ càng tốt)": metrics["Log Loss"],
                "Brier Score": metrics["Brier Score"],
                "Tốc độ (Giây)": metrics["Time (s)"]
            })
            
            results.append(res_dict)
            print(f"  -> Xong! Custom Score: {metrics['Custom Score']} | Recall: {metrics['Recall']} | Tốc độ: {metrics['Time (s)']}s")
        except Exception as e:
            print(f"  -> [LỖI] Mô hình {model.name} thất bại: {e}")

# In bảng kết quả so sánh
print("\n" + "="*90)
print("BẢNG TỔNG HỢP KẾT QUẢ SO SÁNH GIỮA CÁC MÔ HÌNH VÀ BỘ DỮ LIỆU")
print("="*90)
results_df = pd.DataFrame(results)

if not results_df.empty:
    # Sắp xếp bảng theo F1-Score giảm dần để dễ tìm mô hình tốt nhất
    results_df = results_df.sort_values(by=["Tập dữ liệu", "Custom Score"], ascending=[True, False])
    print(results_df.to_markdown(index=False))

    # Lưu file kết quả
    output_csv = os.path.join(script_dir, "comparison_results.csv")
    results_df.to_csv(output_csv, index=False)
    print(f"\nĐã lưu kết quả chi tiết tại: {output_csv}")
else:
    print("Không có kết quả nào được tạo ra do lỗi nạp dữ liệu!")
