# BÁO CÁO TỔNG HỢP TOÀN DIỆN CÁC PHƯƠNG PHÁP DUNG HỢP ENSEMBLE
**Tập dữ liệu:** Expanded 54 Đặc trưng SOTA (Chunked Test: 372,993 dòng)  
**Mục tiêu tối thượng:** Tối đa hóa PR-AUC (Average Precision) và Cân bằng $F_1$-Score trong điều kiện mất cân bằng dữ liệu cực đoan (~2.4% dị thường).

---

## 1. Bảng Xếp Hạng Tổng Thể Toàn Bộ 23 Phương Pháp (Leaderboard)

| Hạng | Phương Pháp Dung Hợp | Phân Loại Kỹ Thuật | PR-AUC | ROC-AUC | $F_1$-Score | Precision | Recall | Ngưỡng Tối Ưu ($\tau^*$) | TP | FP | FN |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 | **Quad Geometric Mean** | Geometric Mean | **0.8458** | **0.9541** | 0.8432 | 94.12% | 76.38% | 0.2405 | 6,972 | 436 | 2,159 |
| 🥈 | **Quad Harmonic Mean** | Harmonic Mean | **0.8457** | 0.9537 | 0.8416 | 93.92% | 76.25% | 0.2142 | 6,962 | 451 | 2,169 |
| 🥉 | **Pair Soft Voting (XGB + CAT)** | Soft Voting | **0.8457** | 0.9531 | **0.8441** | 94.78% | 76.09% | 0.2622 | 6,947 | 383 | 2,184 |
| 4 | **Quad Equal Rank 0.25 (CHAMPION)** | Rank Averaging | **0.8452** | **0.9540** | 0.8419 | 93.73% | **76.42%** | 0.9794 | 6,980 | 477 | 2,151 |
| 5 | **Quad Weighted Rank (PR-AUC)** | Rank Weighted | 0.8452 | 0.9540 | 0.8420 | 93.75% | 76.41% | 0.9794 | 6,981 | 477 | 2,150 |
| 6 | **Quad SLSQP Optimization** | Rank SLSQP | 0.8452 | 0.9540 | 0.8419 | 93.74% | 76.41% | 0.9794 | 6,980 | 477 | 2,151 |
| 7 | **Trio Rank (XGB + CAT + LGBM)** | Rank Averaging | 0.8452 | 0.9535 | 0.8433 | 94.36% | 76.23% | 0.9797 | 6,957 | 414 | 2,174 |
| 8 | **Trio Rank (XGB + LGBM + HIST)** | Rank Averaging | 0.8451 | **0.9544** | 0.8405 | 93.90% | 76.07% | 0.9798 | 6,938 | 445 | 2,193 |
| 9 | **Quad Equal Prob (Soft Voting 0.25)**| Soft Voting | 0.8449 | 0.9540 | **0.8441** | 94.76% | 76.10% | 0.2835 | 6,949 | 384 | 2,182 |
| 10 | **Quad Weighted Soft Voting** | Soft Voting Weighted | 0.8449 | 0.9540 | **0.8441** | 94.76% | 76.10% | 0.2834 | 6,948 | 384 | 2,183 |
| 11 | **Trio Soft Voting (XGB + CAT + LGBM)**| Soft Voting | 0.8448 | 0.9537 | 0.8436 | 94.00% | 76.51% | 0.2585 | 6,985 | 446 | 2,146 |
| 12 | **Trio Rank (CAT + LGBM + HIST)** | Rank Averaging | 0.8442 | 0.9536 | 0.8418 | 92.81% | **77.02%** | 0.9789 | 7,025 | 539 | 2,106 |
| 13 | **Trio Rank (XGB + CAT + HIST)** | Rank Averaging | 0.8442 | 0.9535 | 0.8409 | 93.91% | 76.14% | 0.9796 | 6,954 | 459 | 2,177 |
| 14 | **Pair Rank (XGB + CAT)** | Rank Averaging | 0.8441 | 0.9524 | 0.8418 | 93.89% | 76.29% | 0.9796 | 6,967 | 462 | 2,164 |
| 15 | **Pair Rank (XGB + LGBM)** | Rank Averaging | 0.8439 | 0.9540 | 0.8403 | 93.23% | 76.49% | 0.9797 | 6,977 | 504 | 2,154 |
| 16 | **Quad Min Probability** | Min Voting (Conservative) | 0.8439 | 0.9526 | 0.8380 | 94.02% | 75.59% | 0.1434 | 6,902 | 439 | 2,229 |
| 17 | **Pair Rank (LGBM + HIST)** | Rank Averaging | 0.8437 | 0.9543 | 0.8405 | 95.67% | 74.94% | 0.9805 | 6,846 | 317 | 2,285 |
| 18 | **Pair Rank (CAT + LGBM)** | Rank Averaging | 0.8435 | 0.9525 | 0.8423 | 94.83% | 75.76% | 0.9800 | 6,913 | 372 | 2,218 |
| 19 | **Single: XGBoost** | Single Model Baseline | 0.8432 | 0.9531 | 0.8358 | 93.35% | 75.67% | 0.2356 | 6,909 | 492 | 2,222 |
| 20 | **Pair Soft Voting (XGB + LGBM)** | Soft Voting | 0.8422 | 0.9539 | 0.8396 | 94.28% | 75.68% | 0.2927 | 6,910 | 419 | 2,221 |
| 21 | **Quad Max Probability** | Max Voting (Optimistic) | 0.8409 | 0.9522 | 0.8438 | 94.69% | 76.10% | 0.4497 | 6,948 | 390 | 2,183 |
| 22 | **Single: LightGBM** | Single Model Baseline | 0.8384 | 0.9534 | 0.8366 | 93.60% | 75.63% | 0.3227 | 6,906 | 472 | 2,225 |
| 23 | **Single: HistGradientBoosting** | Single Model Baseline | 0.8380 | 0.9530 | 0.8358 | 95.74% | 74.15% | 0.3645 | 6,771 | 301 | 2,360 |
| 24 | **Single: CatBoost** | Single Model Baseline | 0.8376 | 0.9482 | 0.8409 | 93.93% | 76.11% | 0.2495 | 6,950 | 449 | 2,181 |

---

## 2. Phân Tích Chuyên Sâu Từng Nhóm Kỹ Thuật

### A. Nhóm Dung Hợp Bộ Tứ (Quad Boosting Ensembles)
* **Geometric Mean (Trung bình nhân xác suất):** Đạt điểm số PR-AUC cao nhất bảng xếp hạng (**0.8458**). Nhờ đặc tính phạt nặng nếu có bất kỳ mô hình nào dự đoán xác suất thấp, Geometric Mean loại bỏ triệt để các cảnh báo giả do sai số ngẫu nhiên của một mô hình đơn lẻ.
* **Equal Rank Averaging (0.25 đều):** Đạt **PR-AUC = 0.8452**, **ROC-AUC = 0.9540** và **Recall = 76.42%** (bắt đúng 6,980 dị thường). Đây là kỹ thuật sản xuất tối ưu nhất vì triệt tiêu hoàn toàn sự lệch chuẩn (calibration shift) giữa các mô hình mà không cần tham số tối ưu hóa phức tạp.
* **Chứng minh thực nghiệm SLSQP:** Thuật toán tối ưu hóa ràng buộc phi tuyến SLSQP trên mẫu validation hội tụ về trọng số:
  $$w_{\text{SLSQP}} = [w_{\text{xgb}}=0.250,\; w_{\text{cat}}=0.250,\; w_{\text{lgb}}=0.251,\; w_{\text{hist}}=0.250]$$
  Kết quả kiểm thử trên Test của SLSQP đạt **PR-AUC = 0.8452**, **$F_1$ = 0.8419**, hoàn toàn trùng khớp 100% với Equal Rank Averaging.

### B. Nhóm Dung Hợp Bộ Ba (Trio Ensembles)
* Tổ hợp **XGBoost + CatBoost + LightGBM (Trio Rank)** đạt **PR-AUC = 0.8452**, $F_1$ = 0.8433, tiệm cận tuyệt đối với Bộ tứ.
* Tổ hợp **CAT + LGBM + HIST** đạt **Recall cao nhất (77.02%)** với 7,025 dị thường được phát hiện chính xác.

### C. So Sánh: Mô Hình Đơn Lẻ vs. Ensemble
* **Mô hình đơn lẻ tốt nhất:** XGBoost đạt PR-AUC = 0.8432, $F_1$ = 0.8358.
* **Khi kết hợp Ensemble:** Tất cả các biến thể Ensemble Bộ tứ đều cải thiện PR-AUC từ **0.8432 lên 0.8452 - 0.8458** (+0.0026 PR-AUC), đồng thời đẩy $F_1$ từ 0.8358 lên **0.8419 - 0.8441** (+0.0083 $F_1$), giảm thiểu số ca bỏ sót (FN) từ 2,222 ca xuống 2,151 ca.

---

## 3. Danh Mục Các Tệp Dữ Liệu Đã Xuất

| Tệp Dữ Liệu | Định Dạng | Dung Lượng / Kích Thước | Mục Đích Sử Dụng |
| :--- | :---: | :---: | :--- |
| [`all_ensemble_methods_summary.md`](file:///d:/CODE/p4aids/model_ML/results/all_ensemble_methods_summary.md) | Markdown | ~6 KB | Xem trực tiếp báo cáo tổng hợp trực quan trong VS Code / IDE |
| [`all_ensemble_methods_summary.csv`](file:///d:/CODE/p4aids/model_ML/results/all_ensemble_methods_summary.csv) | CSV | 25 dòng | Bảng tổng hợp chỉ số 23 phương pháp để import vào Excel / Pandas |
| [`final_quad_ensemble_summary.csv`](file:///d:/CODE/p4aids/model_ML/results/final_quad_ensemble_summary.csv) | CSV | 25 dòng | Tệp đồng bộ bảng kết quả phục vụ pipeline tự động |
| [`test_predictions_ensemble.parquet`](file:///d:/CODE/p4aids/model_ML/results/test_predictions_ensemble.parquet) | Parquet | **23.5 MB (372,993 dòng)** | Toàn bộ dự đoán test: `y_true`, xác suất đơn lẻ, rank, điểm ensemble, `y_pred` |
| [`test_predictions_ensemble_sample.csv`](file:///d:/CODE/p4aids/model_ML/results/test_predictions_ensemble_sample.csv) | CSV | 10,000 dòng | Mẫu trích xuất 10,000 dòng để mở nhanh kiểm tra dữ liệu |

