# BẢNG KẾT QUẢ SO SÁNH CHUẨN (KHÔNG LEAKAGE)

Dữ liệu Lag và Zone đã được train chuẩn trên 80% và đánh giá trên 20% validation set, giống hệt với Baseline.

| Chiến Thuật                 | Mô Hình           |   Ngưỡng Tối Ưu |   Precision |   Recall |   F1 Score |   Train Time (s) |
|:----------------------------|:------------------|----------------:|------------:|---------:|-----------:|-----------------:|
| 1. Baseline (Cũ)            | XGBoost           |            0.27 |      0.9607 |   0.9016 |     0.9302 |           0      |
| 1. Baseline (Cũ)            | LightGBM          |            0.29 |      0.9298 |   0.8452 |     0.8855 |           0      |
| 1. Baseline (Cũ)            | CatBoost          |            0.29 |      0.9505 |   0.886  |     0.9172 |           0      |
| 2. Lag Features             | XGBoost           |            0.15 |      0.4997 |   0.3354 |     0.4014 |          53.3464 |
| 2. Lag Features             | LightGBM          |            0.13 |      0.4149 |   0.3199 |     0.3612 |          23.6305 |
| 2. Lag Features             | CatBoost          |            0.19 |      0.4686 |   0.308  |     0.3717 |         207.403  |
| 3. Zone Features            | XGBoost           |            0.25 |      0.9533 |   0.9062 |     0.9291 |          32.7089 |
| 3. Zone Features            | LightGBM          |            0.21 |      0.9047 |   0.8634 |     0.8836 |          17.8136 |
| 3. Zone Features            | CatBoost          |            0.33 |      0.9571 |   0.8787 |     0.9162 |         186.023  |
| 4. Ensemble (MoE trên Zone) | Context-Aware MoE |            0.41 |      0.9445 |   0.9174 |     0.9308 |           2.4514 |