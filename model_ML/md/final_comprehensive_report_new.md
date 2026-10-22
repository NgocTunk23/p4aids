# BÁO CÁO ĐÁNH GIÁ HIỆU NĂNG TOÀN DIỆN: BASELINE vs ZONE FEATURES vs ENSEMBLE

- **Ngày cập nhật:** 21/10/2026 23:47:39
- **Tập dữ liệu:** Dữ liệu chuẩn hóa Baseline (`train_preprocessed.parquet`) vs Dữ liệu Zone (`train_preprocessedzone.parquet`)
- **Tỷ lệ phân chia:** 80% Train / 20% Validation (Stratified sampling chống rò rỉ dữ liệu)

### Bảng Kết Quả Hiệu Năng Chi Tiết (Sắp xếp theo F1 Ngưỡng Tối Ưu):

| Nhóm Thực Nghiệm               | Mô Hình / Phương Pháp                      |   F1 (Ngưỡng 0.5) |   Ngưỡng Tối Ưu |   F1 (Ngưỡng Tối Ưu) |   F1 (Ngưỡng Động 16 Khu) |   Precision |   Recall |   Thời Gian (s) |
|:-------------------------------|:-------------------------------------------|------------------:|----------------:|---------------------:|--------------------------:|------------:|---------:|----------------:|
| 4. Ensemble (Zone Features)    | Weighted Blending LogLoss [0.65 0.25 0.1 ] |            0.9241 |           0.232 |               0.9336 |                    0.9373 |      0.9519 |   0.9159 |             0   |
| 4. Ensemble (Zone Features)    | Dynamic Thresh Blend [0.6  0.25 0.15]      |            0.9244 |           0.234 |               0.9335 |                    0.9372 |      0.9524 |   0.9153 |             0   |
| 4. Ensemble (Zone Features)    | Top-2 Blend (XGB+CAT) [0.8 0.2]            |            0.923  |           0.251 |               0.933  |                    0.9348 |      0.956  |   0.9111 |             0   |
| 3. Ensemble (Baseline)         | Weighted Blending LogLoss [0.68 0.26 0.06] |            0.9229 |           0.295 |               0.932  |                    0.9359 |      0.9639 |   0.9021 |             0   |
| 4. Ensemble (Zone Features)    | Context-Aware MoE (Meta-Learner)           |            0.9304 |           0.294 |               0.932  |                    0.9354 |      0.9532 |   0.9118 |             3.3 |
| 2. Zone Features (Đã phân cụm) | XGBoost                                    |            0.9238 |           0.263 |               0.9317 |                    0.9341 |      0.9568 |   0.9079 |            10.9 |
| 4. Ensemble (Zone Features)    | Soft Voting (XGB + LGB + CAT)              |            0.9219 |           0.234 |               0.9315 |                    0.9359 |      0.9492 |   0.9145 |             0   |
| 3. Ensemble (Baseline)         | Top-2 Blend (XGB+CAT) [0.84 0.16]          |            0.9228 |           0.334 |               0.9311 |                    0.9354 |      0.9688 |   0.8962 |             0   |
| 3. Ensemble (Baseline)         | Dynamic Thresh Blend [0.5  0.25 0.25]      |            0.9213 |           0.319 |               0.9311 |                    0.9363 |      0.9665 |   0.8981 |             0   |
| 1. Baseline (Dữ liệu thường)   | XGBoost                                    |            0.9234 |           0.303 |               0.9308 |                    0.9349 |      0.9637 |   0.9001 |            12.8 |
| 3. Ensemble (Baseline)         | Soft Voting (XGB + LGB + CAT)              |            0.9201 |           0.263 |               0.9301 |                    0.935  |      0.9539 |   0.9074 |             0   |
| 2. Zone Features (Đã phân cụm) | LightGBM                                   |            0.9134 |           0.317 |               0.924  |                    0.9287 |      0.9612 |   0.8895 |             9.2 |
| 1. Baseline (Dữ liệu thường)   | LightGBM                                   |            0.9122 |           0.285 |               0.9236 |                    0.926  |      0.9528 |   0.8961 |             8.1 |
| 2. Zone Features (Đã phân cụm) | CatBoost                                   |            0.9101 |           0.293 |               0.9202 |                    0.922  |      0.952  |   0.8905 |           111.5 |
| 1. Baseline (Dữ liệu thường)   | CatBoost                                   |            0.9056 |           0.255 |               0.9188 |                    0.9211 |      0.9435 |   0.8954 |           160.1 |

---
### Nhận xét & Kết luận chính:
1. **Hiệu quả của Zone Features:** Việc phân cụm và trích xuất đặc trưng tương tác theo từng Zone/Site giúp mô hình thích ứng tốt hơn với đặc thù vi khí hậu và hành vi tải điện cục bộ.
2. **Sức mạnh của Ngưỡng Tối Ưu (Optimal Threshold):** Do mất cân bằng nhãn nghiêm trọng (chỉ 1-5% dị thường), việc dùng ngưỡng cố định 0.5 bỏ sót nhiều điểm bất thường. Chuyển sang ngưỡng tối ưu giúp F1-score nhảy vọt.
3. **Tổ hợp Ensemble:** Kỹ thuật Weighted Blending và Context-Aware MoE khai thác tối đa điểm mạnh bù trừ của bộ 3 XGBoost, LightGBM và CatBoost, mang lại độ tin cậy và điểm F1 cao nhất.
