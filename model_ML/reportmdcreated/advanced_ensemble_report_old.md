# BÁO CÁO: CÁC KỸ THUẬT ENSEMBLE NÂNG CAO VỚI CÁC TỔ HỢP (Theo ASHRAE Top 1)

| Giai đoạn               | Tổ Hợp                      | Phương pháp       |   Ngưỡng Tối Ưu |   F1 Score (Ngưỡng 0.5) |   F1 Score (Ngưỡng Tối Ưu) |   Recall (Tối Ưu) |   Precision (Tối Ưu) |
|:------------------------|:----------------------------|:------------------|----------------:|------------------------:|---------------------------:|------------------:|---------------------:|
| GĐ 4: Ensemble Nâng Cao | XGB + Cat (1 cao + 1 thấp)  | Weighted Blending |            0.24 |                  0.92   |                     0.9312 |            0.9075 |               0.9561 |
| GĐ 4: Ensemble Nâng Cao | XGB + LGBM + Cat (Cả 3)     | Weighted Blending |            0.24 |                  0.9202 |                     0.9309 |            0.9072 |               0.9558 |
| GĐ 4: Ensemble Nâng Cao | Mô hình Đơn lẻ              | XGBoost           |            0.26 |                  0.9209 |                     0.9304 |            0.9029 |               0.9595 |
| GĐ 4: Ensemble Nâng Cao | XGB + LGBM (1 cao + 1 thấp) | Generalized Mean  |            0.26 |                  0.9209 |                     0.9304 |            0.9029 |               0.9595 |
| GĐ 4: Ensemble Nâng Cao | XGB + LGBM (1 cao + 1 thấp) | Weighted Blending |            0.27 |                  0.9207 |                     0.9304 |            0.9019 |               0.9609 |
| GĐ 4: Ensemble Nâng Cao | XGB + Cat (1 cao + 1 thấp)  | Generalized Mean  |            0.26 |                  0.9207 |                     0.9303 |            0.9025 |               0.9598 |
| GĐ 4: Ensemble Nâng Cao | XGB + LGBM + Cat (Cả 3)     | Generalized Mean  |            0.26 |                  0.9206 |                     0.9303 |            0.9025 |               0.9598 |
| GĐ 4: Ensemble Nâng Cao | XGB + Cat (1 cao + 1 thấp)  | Stacking LR       |            0.07 |                  0.9237 |                     0.9292 |            0.9087 |               0.9506 |
| GĐ 4: Ensemble Nâng Cao | XGB + Cat (1 cao + 1 thấp)  | Soft Voting       |            0.23 |                  0.9181 |                     0.9286 |            0.91   |               0.9479 |
| GĐ 4: Ensemble Nâng Cao | XGB + LGBM + Cat (Cả 3)     | Stacking LR       |            0.1  |                  0.9223 |                     0.9283 |            0.9058 |               0.952  |
| GĐ 4: Ensemble Nâng Cao | XGB + LGBM (1 cao + 1 thấp) | Stacking LR       |            0.09 |                  0.9212 |                     0.9277 |            0.9017 |               0.9553 |
| GĐ 4: Ensemble Nâng Cao | XGB + LGBM + Cat (Cả 3)     | Soft Voting       |            0.25 |                  0.9097 |                     0.9249 |            0.9012 |               0.9498 |
| GĐ 4: Ensemble Nâng Cao | XGB + LGBM (1 cao + 1 thấp) | Soft Voting       |            0.24 |                  0.9054 |                     0.9211 |            0.9007 |               0.9425 |
| GĐ 4: Ensemble Nâng Cao | LGBM + Cat (2 model thấp)   | Stacking LR       |            0.16 |                  0.9115 |                     0.9179 |            0.8874 |               0.9506 |
| GĐ 4: Ensemble Nâng Cao | LGBM + Cat (2 model thấp)   | Weighted Blending |            0.27 |                  0.9052 |                     0.9177 |            0.8901 |               0.9471 |
| GĐ 4: Ensemble Nâng Cao | Mô hình Đơn lẻ              | CatBoost          |            0.3  |                  0.9057 |                     0.9173 |            0.8843 |               0.9529 |
| GĐ 4: Ensemble Nâng Cao | LGBM + Cat (2 model thấp)   | Generalized Mean  |            0.3  |                  0.9057 |                     0.9172 |            0.8842 |               0.9529 |
| GĐ 4: Ensemble Nâng Cao | LGBM + Cat (2 model thấp)   | Soft Voting       |            0.27 |                  0.8937 |                     0.9142 |            0.8847 |               0.9458 |
| GĐ 4: Ensemble Nâng Cao | Mô hình Đơn lẻ              | LightGBM          |            0.28 |                  0.8617 |                     0.8855 |            0.8484 |               0.926  |
