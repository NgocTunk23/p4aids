# HƯỚNG DẪN VẬN HÀNH DỰ ÁN & BẢN ĐỒ KIẾN TRÚC HỆ THỐNG
**Đề tài:** Phát hiện Bất thường Phụ tải Điện quy mô lớn (*Large-scale Energy Anomaly Detection - LEAD*)  
**Tiêu chuẩn Thiết kế:** 100% Zero Data Leakage (Seasonal Chunking với 96h Purge Gap)  
**Môi trường khuyến nghị:** Windows 10/11 | Python 3.10+ (CUDA GPU NVIDIA hỗ trợ tăng tốc)

---

## 1. Cấu trúc Thư mục Toàn diện & Ý nghĩa Từng Tệp (Directory Map)

```text
p4aids/
├── data/                                   # Thư mục dữ liệu gốc và dữ liệu xử lý chung
│   ├── train_features.csv                  # Dữ liệu gốc huấn luyện (1,75 triệu dòng)
│   ├── test_features.csv                   # Dữ liệu gốc kiểm tra (1,84 triệu dòng)
│   ├── processed/                          # Dữ liệu Parquet Zero Data Leakage cho Machine Learning
│   │   ├── train_chunked_expanded.parquet  # [SOTA CHAMPION] Tập train mở rộng (1.147.089 dòng, 55 cột)
│   │   ├── test_chunked_expanded.parquet   # [SOTA CHAMPION] Tập test độc lập Out-of-Sample (372.993 dòng, 55 cột)
│   │   ├── train_chunked_context.parquet   # Tập train Ngữ cảnh vi mô (47 cột)
│   │   ├── test_chunked_context.parquet    # Tập test Ngữ cảnh vi mô (47 cột)
│   │   ├── train_chunked_zone.parquet      # Tập train Tương tác Zone (43 cột)
│   │   ├── test_chunked_zone.parquet       # Tập test Tương tác Zone (43 cột)
│   │   ├── train_chunked_baseline.parquet  # Tập train Baseline cơ sở (40 cột)
│   │   ├── test_chunked_baseline.parquet   # Tập test Baseline cơ sở (40 cột)
│   │   ├── split_metadata.json             # Metadata cấu trúc phân chia mùa vụ & kiểm toán tỷ lệ nhãn
│   │   └── preprocessing_step_log.csv      # Bảng kiểm toán (audit log) 8 bước tiền xử lý
│   └── processed_DL/                       # Dữ liệu định dạng chuỗi tuần tự cho Deep Learning
│       ├── train_processed_dl.parquet      # Tập train chuỗi tuần tự theo giờ (hourly grid)
│       ├── test_processed_dl.parquet       # Tập test chuỗi tuần tự theo giờ
│       └── processed_dl_metadata.json      # Metadata cấu trúc chuỗi và thống kê chuẩn hóa
│
├── preprocessingforALL.py                  # Module lõi tiền xử lý dữ liệu chung & trích xuất 8 đặc trưng Expanded (xuất đủ 8 bộ Parquet train/test)
├── preprocessingforALL.ipynb               # Notebook trực quan hóa EDA và kiểm toán phân phối dữ liệu
├── run_ML.md                               # [TÀI LIỆU NÀY] Hướng dẫn vận hành hệ thống chuẩn hóa
├── requirements.txt                        # Danh sách thư viện Python cần thiết
│
├── model_ML/                               # Phân hệ Học máy Tinh gọn (Streamlined Machine Learning)
│   ├── pipeline_ml.py                      # [PIPELINE DUY NHẤT] Hợp nhất Tuning, Ensemble 23 cách, Đóng gói .pkl & Inference
│   ├── run_tuning_expanded.py              # Wrapper tương thích (tự động chuyển tiếp vào pipeline_ml.py)
│   │
│   ├── saved_models/                       # Thư mục lưu trữ mô hình Champion định dạng .pkl
│   │   ├── XGB_Expanded_Champion.pkl       # Mô hình XGBoost Champion GPU (1.62 MB)
│   │   ├── CAT_Expanded_Champion.pkl       # Mô hình CatBoost Champion GPU (0.61 MB)
│   │   ├── LGBM_Expanded_Champion.pkl      # Mô hình LightGBM Champion CPU (1.19 MB)
│   │   ├── HIST_Expanded_Champion.pkl      # Mô hình HistGradientBoosting Champion CPU (0.58 MB)
│   │   ├── champion_models_metadata.json   # Metadata cấu hình 54 đặc trưng và ngưỡng tối ưu tau* = 0.9794
│   │   └── *_Context_Champion.pkl ...      # Checkpoints mô hình vòng trước để đối chứng
│   │
│   ├── hyperparameters/                    # Tham số tối ưu hóa Optuna (Best Hyperparameters)
│   │   ├── best_params_xgb_expanded.json   # Siêu tham số XGBoost Expanded (PR-AUC: 0.8432)
│   │   ├── best_params_cat_expanded.json   # Siêu tham số CatBoost Expanded (PR-AUC: 0.8376)
│   │   ├── best_params_lgbm_expanded.json  # Siêu tham số LightGBM Expanded (PR-AUC: 0.8384)
│   │   ├── best_params_hist_expanded.json  # Siêu tham số HistGB Expanded (PR-AUC: 0.8380)
│   │   └── best_params_*_context.json ...  # Tham số tối ưu của các bộ dữ liệu nền tảng
│   │
│   ├── results/                            # Kết quả đánh giá và dữ liệu dự đoán xuất ra
│   │   ├── all_ensemble_methods_summary.md # [BÁO CÁO TRỰC QUAN] Xếp hạng 23 phương pháp dung hợp, so sánh TP, FP, FN
│   │   ├── all_ensemble_methods_summary.csv# Bảng số liệu tổng hợp 23 phương pháp dung hợp
│   │   ├── final_quad_ensemble_summary.csv # File đồng bộ kết quả Ensemble SOTA
│   │   ├── test_predictions_ensemble.parquet # Dữ liệu dự đoán chi tiết 373k dòng test (y_true, xác suất, rank, y_pred)
│   │   ├── test_predictions_ensemble_sample.csv # Bản trích mẫu 10k dòng để mở nhanh bằng Excel
│   │   ├── tuning_expanded_summary.csv     # Bảng tổng hợp hiệu năng 4 mô hình đơn lẻ sau khi tuning
│   │   └── technique_leave_one_out_summary.csv # Kết quả thực nghiệm triệt tiêu Leave-One-Out từng kỹ thuật
│   │
│   └── data/                               # Dữ liệu parquet cục bộ đồng bộ cho ML (gồm 4 cặp train/test)
│
├── model_DL/                               # Phân hệ Học sâu (Deep Learning Pipeline)
│   ├── config.py                           # Cấu hình siêu tham số, thiết lập sliding window 252h
│   ├── data_pipeline.py                    # Nạp chuỗi tuần tự theo giờ, tạo mask khuyết thiếu
│   ├── models.py                           # Kiến trúc PyTorch (MLP, Causal TCN, BiGRU, BiLSTM, LSTM-AE)
│   └── pipeline.py                         # Điều phối huấn luyện GroupKFold theo building_id & Ensemble
│
└── report_p4aids/                          # Báo cáo khoa học đồ án (LaTeX)
    ├── main.tex                            # Tệp nguồn chính luận văn 52 trang
    ├── compile.bat                         # Kịch bản 1-click tự động biên dịch sang PDF bằng Tectonic
    ├── bin/tectonic.exe                    # Trình biên dịch LaTeX độc lập (không cần TeX Live)
    └── main.pdf                            # Báo cáo đồ án khoa học hoàn chỉnh chất lượng cao
```

---

## 2. Hướng dẫn Vận hành Chi tiết Từng Giai đoạn (Execution Guide)

Tất cả các lệnh dưới đây được thực hiện từ thư mục gốc `d:\CODE\p4aids` trong môi trường Windows PowerShell.

### Giai đoạn 1: Tiền Xử Lý Dữ Liệu Chung (Zero Data Leakage)
Tạo ra toàn bộ 8 tệp Parquet (Baseline, Zone, Context, Expanded) vào `data/processed/` và đồng bộ vào `model_ML/data/`:
```powershell
python preprocessingforALL.py
```
*Thời gian chạy:* ~45 giây. Đảm bảo triệt để Zero Data Leakage bằng Seasonal Chunking với 96h Purge Gap.

---

### Giai đoạn 2: Vận hành Pipeline Machine Learning Duy Nhất (`pipeline_ml.py`)

Tệp [`model_ML/pipeline_ml.py`](file:///d:/CODE/p4aids/model_ML/pipeline_ml.py) hợp nhất toàn bộ quy trình ML. Tùy theo nhu cầu, bạn chỉ cần chọn một trong các cách chạy sau:

#### Cách 2.1: Chạy Tinh Chỉnh Siêu Tham Số (Optuna Fine-Tuning)
Chạy tìm kiếm siêu tham số tối ưu với Optuna TPE, in tiến độ chi tiết từng trial ra terminal (`🔥 [NEW BEST!]`):

* **Tuning 3 mô hình còn lại (Bỏ qua XGBoost vì đã có sẵn file tối ưu):**
  ```powershell
  python model_ML/pipeline_ml.py --model cat,lgbm,hist --trials 30
  ```
* **Tuning lại toàn bộ cả 4 mô hình từ đầu:**
  ```powershell
  python model_ML/pipeline_ml.py --model all --trials 30
  ```
* **Tuning riêng một mô hình đơn lẻ (ví dụ CatBoost):**
  ```powershell
  python model_ML/pipeline_ml.py --model cat --trials 30
  ```
*(Sau khi tuning xong, script sẽ tự động chuyển sang bước Ensemble và đóng gói model).*

---

#### Cách 2.2: Chạy Riêng Ensemble Đánh Giá 23 Phương Pháp & Đóng Gói Model .pkl
Nếu đã có sẵn các file siêu tham số trong `model_ML/hyperparameters/`, chạy lệnh sau để:
1. Huấn luyện 4 mô hình Champion trên 1.15M dòng train.
2. Đánh giá và so sánh toàn bộ **23 phương pháp dung hợp** trên 373k dòng test.
3. Xuất báo cáo Markdown [`all_ensemble_methods_summary.md`](file:///d:/CODE/p4aids/model_ML/results/all_ensemble_methods_summary.md) và file CSV.
4. Xuất file dự đoán chi tiết [`test_predictions_ensemble.parquet`](file:///d:/CODE/p4aids/model_ML/results/test_predictions_ensemble.parquet).
5. **Tự động đóng gói 4 file model `.pkl`** vào [`model_ML/saved_models/`](file:///d:/CODE/p4aids/model_ML/saved_models/).

```powershell
python model_ML/pipeline_ml.py --only-ensemble
```
*Thời gian chạy:* Chỉ ~51 giây cho toàn bộ quy trình!

---

#### Cách 2.3: Chạy Suy Luận Sản Xuất Siêu Tốc Từ Model .pkl (Production Inference)
Nạp trực tiếp 4 file `.pkl` đã lưu trong `saved_models/` để ra quyết định nhị phân tại ngưỡng tối ưu $\tau^* = 0.9794$:
```powershell
python model_ML/pipeline_ml.py --inference
```
*Thời gian suy luận:* **< 2.3 giây** cho toàn bộ 372.993 dòng dữ liệu kiểm thử!

*(Ghi chú: Lệnh `python model_ML/run_tuning_expanded.py` cũng hỗ trợ 100% các tham số trên do đã được bọc tự động chuyển tiếp vào `pipeline_ml.py`).*

---

### Giai đoạn 3: Huấn luyện Phân Hệ Học Sâu (Deep Learning Pipeline)

```powershell
# 1. Tiền xử lý dữ liệu chuỗi tuần tự cho Deep Learning:
python -m model_DL.pipeline preprocess

# 2. Huấn luyện và đánh giá 5 kiến trúc Học sâu (MLP, Causal TCN, BiGRU, BiLSTM, LSTM-AE):
python -m model_DL.pipeline benchmark --models mlp tcn bigru bilstm lstm_ae

# 3. Chạy tổ hợp Percentile-Rank Ensemble cho Deep Learning:
python -m model_DL.pipeline ensemble --models mlp tcn bigru bilstm lstm_ae
```

---

### Giai đoạn 4: Biên Dịch Báo Cáo Khoa Học Sang PDF (LaTeX)

Dự án tích hợp sẵn trình biên dịch độc lập `bin/tectonic.exe`, không yêu cầu cài đặt TeX Live hay MiKTeX:

```powershell
# Cách 1: Chạy file batch tự động một chạm:
.\report_p4aids\compile.bat

# Cách 2: Gọi trực tiếp trình biên dịch Tectonic từ thư mục report:
cd report_p4aids
.\bin\tectonic.exe main.tex
cd ..
```
*File kết quả xuất ra ngay tại:* **`report_p4aids/main.pdf`** *(Báo cáo đồ án khoa học hoàn chỉnh 52 trang chất lượng cao)*.

---

## 3. Bảng Tổng Hợp Hiệu Năng Đỉnh Cao (Master SOTA Leaderboard)

Kết quả đo đạc khách quan trên **372.993 mẫu Out-of-Sample** tuân thủ 100% Zero Data Leakage trên bộ dữ liệu **Expanded 54 Đặc Trưng**:

| Hạng | Phương Pháp Dung Hợp | Phân Loại Kỹ Thuật | PR-AUC (Average Precision) | ROC-AUC | $F_1$-Score | Precision | Recall | Ngưỡng $\tau^*$ |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 | **Quad Geometric Mean** | Geometric Mean | **0.8458** | **0.9541** | 0.8432 | 94.12% | 76.38% | 0.2405 |
| 🥈 | **Quad Harmonic Mean** | Harmonic Mean | **0.8457** | 0.9537 | 0.8416 | 93.92% | 76.25% | 0.2142 |
| 🥉 | **Pair Soft Voting (XGB + CAT)** | Soft Voting | **0.8457** | 0.9531 | **0.8441** | 94.78% | 76.09% | 0.2622 |
| 4 | **Quad Equal Rank 0.25 (CHAMPION SOTA)** | Rank Averaging | **0.8452** | **0.9540** | 0.8419 | 93.73% | **76.42%** | 0.9794 |
| 5 | **Quad Weighted Rank (PR-AUC)** | Rank Weighted | 0.8452 | 0.9540 | 0.8420 | 93.75% | 76.41% | 0.9794 |
| 6 | **Quad SLSQP Optimization** | Rank SLSQP | 0.8452 | 0.9540 | 0.8419 | 93.74% | 76.41% | 0.9794 |
| 7 | **Trio Rank (XGB + CAT + LGBM)** | Rank Averaging | 0.8452 | 0.9535 | 0.8433 | 94.36% | 76.23% | 0.9797 |
| 8 | **Trio Rank (XGB + LGBM + HIST)** | Rank Averaging | 0.8451 | **0.9544** | 0.8405 | 93.90% | 76.07% | 0.9798 |
| 9 | **Quad Equal Prob (Soft Voting 0.25)**| Soft Voting | 0.8449 | 0.9540 | **0.8441** | 94.76% | 76.10% | 0.2835 |
| 10 | **Trio Rank (CAT + LGBM + HIST)** | Rank Averaging | 0.8442 | 0.9536 | 0.8418 | 92.81% | **77.02%** | 0.9789 |
| 11 | **Pair Rank (XGB + CAT)** | Rank Averaging | 0.8441 | 0.9524 | 0.8418 | 93.89% | 76.29% | 0.9796 |
| 12 | **Single: XGBoost** | Single Model Baseline | 0.8432 | 0.9531 | 0.8358 | 93.35% | 75.67% | 0.2356 |
| 13 | **Single: LightGBM** | Single Model Baseline | 0.8384 | 0.9534 | 0.8366 | 93.60% | 75.63% | 0.3227 |
| 14 | **Single: HistGradientBoosting** | Single Model Baseline | 0.8380 | 0.9530 | 0.8358 | 95.74% | 74.15% | 0.3645 |
| 15 | **Single: CatBoost** | Single Model Baseline | 0.8376 | 0.9482 | 0.8409 | 93.93% | 76.11% | 0.2495 |
| --- | *Nhóm mô hình trên tập Context (46 đặc trưng)* | --- | 0.686 - 0.696 | 0.892 - 0.898 | 0.708 - 0.727 | --- | --- | --- |
| --- | *Nhóm mô hình trên tập Zone Features (43 đặc trưng)* | --- | 0.634 - 0.646 | 0.847 - 0.862 | 0.706 - 0.709 | --- | --- | --- |
| --- | *Nhóm mô hình trên tập Baseline (39 đặc trưng)* | --- | 0.638 - 0.642 | 0.854 - 0.863 | 0.701 - 0.710 | --- | --- | --- |

---

## 4. Chi Tiết Pipeline Tiền Xử Lý Dữ Liệu & Điểm Nhấn Kỹ Thuật Đặc Trưng

Pipeline tiền xử lý dữ liệu của dự án được triển khai tập trung và thống nhất 100% trong [`preprocessingforALL.py`](file:///d:/CODE/p4aids/preprocessingforALL.py). Toàn bộ quy trình tuân thủ nghiêm ngặt chuẩn mực **Zero Data Leakage** tuyệt đối.

### 4.1. Nguyên Lý Vàng Zero Data Leakage & Giao Thức Seasonal Chunking (96h Purge Gap)

Trong dữ liệu chuỗi thời gian phụ tải điện, việc phân chia ngẫu nhiên (Random K-Fold) sẽ gây rò rỉ dữ liệu nghiêm trọng do tự tương quan thời gian. Dự án áp dụng giao thức phân chia chu kỳ mùa vụ độc quyền:

```text
       Ngày 01 -> Ngày 20                 Ngày 21 -> Ngày 24             Ngày 25 -> Cuối tháng
┌─────────────────────────────────┐   ┌─────────────────────────┐   ┌─────────────────────────────┐
│  T_in (Tập Train Nội bộ Mùa vụ) │   │ 96h PURGE GAP (CÁCH LY) │   │ T_out (Tập Test Ngoại suy)  │
│    1.147.089 dòng (~65.6%)      │   │  229.412 dòng (~13.1%)  │   │   372.993 dòng (~21.3%)     │
└─────────────────────────────────┘   └─────────────────────────┘   └─────────────────────────────┘
  * Chỉ Fit thống kê tại đây!          * Bỏ hẳn để cắt rò rỉ          * Out-of-Sample thuần túy!
```

* **Vùng Purge Gap 96 giờ (Ngày 21–24 hàng tháng):** Bị loại bỏ hoàn toàn khỏi cả quá trình huấn luyện và đánh giá. Vùng đệm 4 ngày này triệt tiêu hoàn toàn khả năng rò rỉ thông tin của các đặc trưng trễ dài hạn (lag 24h, 48h, 72h).
* **Quy tắc Fit-Transform 100% trên Train:** Tất cả các tham số thống kê (Mean, Std, Median, Quantile IQR, Lookup Tables, Target Encoding) đều **CHỈ ĐƯỢC FIT trên $T_{\text{in}}$**, sau đó áp dụng phép chiếu (Transform) sang $T_{\text{out}}$.

---

### 4.2. Quy Trình Tiền Xử Lý Chuẩn Hóa & Kỹ Thuật Trích Xuất Expanded Toàn Diện

Quy trình tiền xử lý trong [`preprocessingforALL.py`](file:///d:/CODE/p4aids/preprocessingforALL.py) được thực thi tuần tự và chia thành 3 giai đoạn rõ rệt:

#### Giai đoạn I: Tiền Xử Lý Cơ Sở & Phân Chia Chu Kỳ Mùa Vụ
* **Bước 1 – Phân chia chu kỳ mùa vụ & Cách ly Purge Gap 96h:**
  * Chia tách $T_{\text{in}}$ (Ngày 01–20: 1.147.089 dòng) và $T_{\text{out}}$ (Ngày 25–Cuối tháng: 372.993 dòng).
  * Loại bỏ hoàn toàn vùng Purge Gap (Ngày 21–24: 229.412 dòng) để cách ly 96 giờ chống rò rỉ tín hiệu trễ.
* **Bước 2 – Tách Vector Nhãn Dị thường:**
  * Tách nhãn `anomaly` thành vector mục tiêu riêng biệt, bảo toàn phân phối mất cân bằng tự nhiên (~2.4% dị thường).
* **Bước 3 – Mã hóa Chu kỳ Thời gian hình học ($\sin / \cos$):**
  * Biến đổi các đại lượng thời gian sang dạng sóng liên tục: giờ trong ngày ($T=24$), ngày trong tuần ($T=7$) và tháng trong năm ($T=12$).
* **Bước 4 – Gắn cờ Khuyết thiếu & Khử Rò rỉ ID:**
  * Tạo cờ nhị phân `meter_reading_was_missing`, đồng thời loại bỏ toàn bộ các biến định danh có nguy cơ rò rỉ (`building_id`, `site_id`, `timestamp` thô).
* **Bước 5 – Nội suy Chuỗi thời gian Cục bộ Nội bộ Tòa nhà:**
  * Nội suy tuyến tính cục bộ (`interpolate limit=12`) riêng biệt cho từng tòa nhà. Các giá trị khuyết còn lại được điền bằng Median **fit 100% từ tập Train ($T_{\text{in}}$)**.
* **Bước 6 – Lọc Ngoại lai Khí tượng bằng IQR:**
  * Xác định ngưỡng biên $[Q_1 - 1.5\text{IQR}, Q_3 + 1.5\text{IQR}]$ của các biến thời tiết chỉ dựa trên tập Train; các điểm ngoại lai trên cả Train và Test được thay thế bằng Median của Train.
* **Bước 7 – Mã hóa Biến Phân loại & Target Encoding:**
  * Xây dựng bảng tra cứu thống kê phụ tải theo loại hình công trình (`primary_use`) chỉ từ tập Train rồi ánh xạ sang Test.

---

#### Giai đoạn II: Quy Trình Trích Xuất 8 Đặc Trưng Động Học Chuỗi Thời Gian & STL (Tầng Mở Rộng - Expanded Layer)
Được thực hiện trong hàm [`extract_temporal_dynamics_and_stl()`](file:///d:/CODE/p4aids/preprocessingforALL.py#L57) trước khi phân tách dữ liệu, tuân thủ nguyên lý nhân quả (causality):

1. **Sắp xếp Nhân quả (Causal Sorting):**
   * Dữ liệu được sắp xếp theo từng tòa nhà (`building_id`) và thời gian (`timestamp`) tăng dần để đảm bảo mọi phép trượt chỉ lấy thông tin từ quá khứ, loại trừ hoàn toàn việc nhìn trước tương lai (*Look-ahead bias*).
2. **Trích xuất Vi mô Trễ (Micro-Lags):**
   * Tính toán công suất tiêu thụ thực tế 1 giờ trước (`meter_lag_1h` = $\text{shift}(1)$) và 2 giờ trước (`meter_lag_2h` = $\text{shift}(2)$) nội bộ từng tòa nhà.
3. **Tính toán Tốc độ & Gia tốc Biến thiên (Differential Dynamics):**
   * Vận tốc biến thiên phụ tải tức thời: $\Delta_1 = m_t - m_{t-1}$ (`meter_diff_1h`).
   * Gia tốc biến thiên phụ tải 2 giờ: $\Delta_2 = m_t - m_{t-2}$ (`meter_diff_2h`).
4. **Tính toán Tỉ số Động năng Phụ tải (Kinetic Ratios):**
   * Tỉ số biến thiên cấp 1: $R_1 = \frac{m_t + 1}{m_{t-1} + 1}$ (`meter_ratio_1h`).
   * Tỉ số biến thiên cấp 2: $R_2 = \frac{m_t + 1}{m_{t-2} + 1}$ (`meter_ratio_2h`).
   * Phép đo này giúp khuếch đại cực mạnh các xung gai dị thường (Spike anomalies) khi phụ tải tăng vọt đột ngột so với giờ trước.
5. **Bộ đếm Thời lượng Treo Cảm biến (Sensor Flatline Duration Tracking):**
   * Xác định trạng thái cảm biến bị đứng yên: $|\Delta_1| \le 10^{-4}$.
   * Sử dụng thuật toán tích lũy khối (`cumsum` trên đảo ngược boolean) để đếm chính xác số giờ liên tục mà cảm biến không thay đổi giá trị (`flatline_duration`), chuyên trị phát hiện dạng dị thường đóng băng dữ liệu (sensor freeze).
6. **Khử Chu kỳ Mùa vụ 24h & Tính STL Residual (Seasonal Decomposition):**
   * So sánh công suất hiện tại với Median của chính giờ đó trong chu kỳ 7 ngày trước đó (`timestamp - 7 ngày`).
   * Tính toán độ lệch phần dư: $\text{STL Residual} = m_t - \text{Median}_{\text{hour, 7d}}$, loại bỏ triệt để tính tuần hoàn ngày/đêm để làm nổi bật các biến động bất thường thực chất.
7. **Bảo toàn Thứ tự Dòng Gốc & Ánh Xạ Không Rò Rỉ (Zero Index Leakage):**
   * Ánh xạ toàn bộ 8 cột đặc trưng mới về đúng vị trí chỉ số gốc ban đầu (`_orig_idx`), đảm bảo 100% khớp nối dữ liệu.
   * Áp dụng `train_mask` và `test_mask` để đưa 8 đặc trưng vào $T_{\text{in}}$ và $T_{\text{out}}$, hoàn toàn không chạm qua vùng Purge Gap.

---

#### Giai đoạn III: Hợp Nhất Đa Tầng & Xuất Bản Dữ Liệu
* **Bước 8 – Hợp nhất Đặc trưng Đa tầng (Multi-Tier Feature Merging):**
  * Ghép nối các lớp đặc trưng tương ứng để hình thành 4 bộ dữ liệu: Baseline (40 cột) $\rightarrow$ Zone (43 cột) $\rightarrow$ Context (47 cột) $\rightarrow$ Expanded (55 cột).
* **Bước 9 – Xuất Tệp Parquet Chuẩn Hóa & Audit Log:**
  * Xuất đồng thời 8 tệp Parquet vào cả [`data/processed/`](file:///d:/CODE/p4aids/data/processed/) và [`model_ML/data/`](file:///d:/CODE/p4aids/model_ML/data/).
  * Xuất tệp nhật ký kiểm toán `preprocessing_step_log.csv` ghi nhận đầy đủ thống kê phân phối của từng bước.

---

### 4.3. Kiến Trúc 4 Cấp Độ Dữ Liệu Đa Tầng (Multi-Tier Datasets)

| Cấp Độ Bộ Dữ Liệu | Số Cột | Đặc Trưng Bổ Sung Nổi Bật | Mục Đích Thực Nghiệm |
| :--- | :---: | :--- | :--- |
| **1. Baseline** | 40 cột | Đặc trưng khí tượng, diện tích, thời gian cơ sở | Làm mốc đối chứng cơ bản (Benchmark gốc) |
| **2. Zone Features** | 43 cột | `zone_interaction`, `site_sqft_mean`, `use_sqft_mean` | Khảo sát tương tác vĩ mô không gian vùng |
| **3. Context-Aware** | 47 cột | Target encoding phụ tải, độ lệch chuẩn theo loại công trình | Khảo sát làm giàu ngữ cảnh vi mô công trình |
| **4. Expanded Champion** | **55 cột** | **8 đặc trưng vi mô trễ, động năng, flatline, STL residual** | **Bộ dữ liệu Champion SOTA đạt PR-AUC 0.8458** |

---

### 4.4. Đột Phá 8 Đặc Trưng Động Học Chuỗi Thời Gian & Phân Rã STL (`preprocessingforALL.py`)

Việc bổ sung 8 đặc trưng này đã tạo ra bước nhảy vọt lớn nhất toàn bộ nghiên cứu (**tăng vọt $+0.1492$ PR-AUC từ 0.6960 lên 0.8458**):

1. `meter_lag_1h`: Công suất tiêu thụ thực tế 1 giờ trước của cùng tòa nhà.
2. `meter_lag_2h`: Công suất tiêu thụ thực tế 2 giờ trước của cùng tòa nhà.
3. `meter_diff_1h` ($\Delta_1 = m_t - m_{t-1}$): Vận tốc biến thiên phụ tải tức thời.
4. `meter_diff_2h` ($\Delta_2 = m_t - m_{t-2}$): Gia tốc biến thiên phụ tải 2 giờ.
5. `meter_ratio_1h` ($\frac{m_t + 1}{m_{t-1} + 1}$): Tỉ số động năng phụ tải (bắt các đột biến xung nhọn tức thời).
6. `meter_ratio_2h` ($\frac{m_t + 1}{m_{t-2} + 1}$): Tỉ số động năng phụ tải chu kỳ 2 giờ.
7. `flatline_duration`: **Bộ đếm thời lượng treo cảm biến** (tính thời gian liên tục mà $|\Delta_1| \le 10^{-4}$), chuyên trị phát hiện dạng dị thường đóng băng dữ liệu đo đạc (sensor freeze).
8. `stl_residual`: **Phần dư sau phân rã mùa vụ**, tính bằng độ lệch giữa giá trị đo đạc và Median cùng giờ trong chu kỳ 7 ngày trước đó, loại bỏ triệt để tính tuần hoàn ngày/đêm.

---

### 4.5. Điểm Nhấn Thực Nghiệm & Tinh Gọn Pipeline Sản Xuất

1. **Chứng minh Toán học & Thực nghiệm Nghiệm SLSQP $\equiv$ Equal Weights 0.25:**
   * Tối ưu hóa ràng buộc phi tuyến SLSQP trên mẫu validation hội tụ về nghiệm:
     $$w_{\text{SLSQP}} = [w_{\text{xgb}}=0.250,\; w_{\text{cat}}=0.250,\; w_{\text{lgb}}=0.251,\; w_{\text{hist}}=0.250]$$
   * Thực nghiệm trên tập Test khẳng định **Equal Rank Averaging 0.25** đạt PR-AUC = **0.8452**, $F_1$ = **0.8419**, tương đương 100% với SLSQP nhưng loại bỏ hoàn toàn chi phí tính toán tối ưu phức tạp trong sản xuất.
2. **Loại bỏ Hậu Xử Lý Chuỗi Thời Gian (Post-Processing Elimination):**
   * Khảo sát Leave-One-Out chứng minh việc áp dụng hậu xử lý lọc trễ làm giảm $F_1$ từ 0.8441 xuống 0.8335 do gây độ trễ phát hiện tín hiệu. Việc loại bỏ hậu xử lý giúp hệ thống đạt đỉnh $F_1$ cao nhất toàn bài (**0.8441**) và tối ưu hóa tốc độ cảnh báo thời gian thực.
