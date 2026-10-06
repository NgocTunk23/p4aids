# BÁO CÁO TỔNG HỢP CUỐI CÙNG: MÔ HÌNH vs DỮ LIỆU vs NGƯỠNG ĐỘNG

| Tổ Hợp                      | Phương pháp                        |   Ngưỡng Chung |   F1 (Ngưỡng 0.5) |   F1 (Ngưỡng Chung) |   F1 (Ngưỡng Động 16 Khu) |
|:----------------------------|:-----------------------------------|---------------:|------------------:|--------------------:|--------------------------:|
| Zone Ensemble (Cả 3)        | Weighted Blending [0.81 0.04 0.15] |           0.26 |            0.9212 |              0.9294 |                    0.9346 |
| Zone Ensemble (XGB+Cat)     | Weighted Blending [0.82 0.18]      |           0.21 |            0.9213 |              0.9295 |                    0.9345 |
| Baseline Ensemble (Cả 3)    | Weighted Blending [0.83 0.   0.17] |           0.24 |            0.9202 |              0.9309 |                    0.9343 |
| Baseline Ensemble (XGB+Cat) | Weighted Blending [0.82 0.18]      |           0.24 |            0.92   |              0.9312 |                    0.9342 |
| Baseline Đơn lẻ             | XGBoost                            |           0.26 |            0.9209 |              0.9304 |                    0.9341 |
| Zone Đơn lẻ                 | XGBoost                            |           0.25 |            0.9218 |              0.9291 |                    0.9331 |
| Baseline Đơn lẻ             | CatBoost                           |           0.3  |            0.9057 |              0.9173 |                    0.9207 |
| Zone Đơn lẻ                 | CatBoost                           |           0.27 |            0.9077 |              0.917  |                    0.9192 |
| Zone Đơn lẻ                 | LightGBM                           |           0.32 |            0.8929 |              0.9054 |                    0.9103 |
| Baseline Đơn lẻ             | LightGBM                           |           0.28 |            0.8617 |              0.8855 |                    0.8867 |
