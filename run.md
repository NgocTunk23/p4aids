python model_ML/run_tuning_pipeline.py
python model_ML/run_comparison_pipeline.py
python model_ML/run_final_comprehensive.py


==========================================================================================
BẢNG TỔNG HỢP KẾT QUẢ SO SÁNH GIỮA CÁC MÔ HÌNH VÀ BỘ DỮ LIỆU LẦN 1
==========================================================================================
| Tập dữ liệu   | Mô hình       |   Custom Score |   F2-Score (Nặng Recall) |   Recall (Độ phủ) |   Log Loss (Càng nhỏ càng tốt) |   Brier Score |   Tốc độ (Giây) |
|:--------------|:--------------|---------------:|-------------------------:|------------------:|-------------------------------:|--------------:|----------------:|
| Engineered    | XGBoost       |         0.8405 |                   0.842  |            0.8141 |                         0.015  |        0.0035 |           13.1  |
| Engineered    | LightGBM      |         0.7619 |                   0.7672 |            0.7496 |                         0.0529 |        0.007  |            9.58 |
| Engineered    | CatBoost      |         0.7248 |                   0.7276 |            0.6833 |                         0.0275 |        0.0059 |           48.15 |
| Engineered    | Random Forest |         0.6894 |                   0.7177 |            0.796  |                         0.282  |        0.0674 |          134.27 |
| Engineered    | Extra Trees   |         0.3023 |                   0.353  |            0.7463 |                         0.5073 |        0.1627 |          100.16 |
| Preprocessed  | XGBoost       |         0.8457 |                   0.8471 |            0.8201 |                         0.0147 |        0.0034 |           17.33 |
| Preprocessed  | LightGBM      |         0.7499 |                   0.7565 |            0.771  |                         0.066  |        0.01   |           12.62 |
| Preprocessed  | CatBoost      |         0.7236 |                   0.7264 |            0.682  |                         0.0278 |        0.006  |           47.83 |
| Preprocessed  | Random Forest |         0.6839 |                   0.7124 |            0.7961 |                         0.2843 |        0.0683 |          135.7  |
| Preprocessed  | Extra Trees   |         0.2905 |                   0.3411 |            0.7568 |                         0.5063 |        0.1624 |           99.26 |

Trình bày nó khả thi cho các dạng Tree nhưng quá lâu và hiệu suất không cao như 2 thằng kia
Ngưỡng tối ưu áp dụng được
