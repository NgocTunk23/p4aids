# BẢNG TỔNG HỢP TOÀN DIỆN MỌI GIAI ĐOẠN (Từ Pre, FE đến Threshold & Ensemble)

| Giai đoạn           | Mô hình                              | Dữ liệu      |   Ngưỡng Tối Ưu |   Custom Score (Ngưỡng 0.5) |   Custom Score (Ngưỡng Tối Ưu) |   Tốc độ (s) |
|:--------------------|:-------------------------------------|:-------------|----------------:|----------------------------:|-------------------------------:|-------------:|
| GĐ 2-3: Mô hình Đơn | XGBoost                              | Preprocessed |            0.07 |                      0.8882 |                         0.9215 |         84   |
| GĐ 2-3: Mô hình Đơn | XGBoost                              | Engineered   |            0.07 |                      0.8824 |                         0.9216 |         92.1 |
| GĐ 2-3: Mô hình Đơn | LightGBM                             | Preprocessed |            0.11 |                      0.8271 |                         0.8762 |         93.3 |
| GĐ 2-3: Mô hình Đơn | LightGBM                             | Engineered   |            0.11 |                      0.8204 |                         0.8678 |         30.9 |
| GĐ 2-3: Mô hình Đơn | CatBoost                             | Preprocessed |            0.11 |                      0.8352 |                         0.8941 |        113.9 |
| GĐ 2-3: Mô hình Đơn | CatBoost                             | Engineered   |            0.13 |                      0.8361 |                         0.889  |        120   |
| GĐ 4: Ensemble      | XGB + LightGBM + CatBoost (Blending) | Hỗn hợp      |            0.11 |                      0.8593 |                         0.9099 |          0   |
| GĐ 4: Ensemble      | XGB + LightGBM + CatBoost (Stacking) | Hỗn hợp      |            0.01 |                      0.8916 |                         0.9141 |          0   |
| GĐ 4: Ensemble      | LightGBM + CatBoost (Blending)       | Hỗn hợp      |            0.15 |                      0.8328 |                         0.8912 |          0   |
| GĐ 4: Ensemble      | LightGBM + CatBoost (Stacking)       | Hỗn hợp      |            0.03 |                      0.8619 |                         0.8921 |          0   |
