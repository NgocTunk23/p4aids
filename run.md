# HƯỚNG DẪN VẬN HÀNH DỰ ÁN & BẢN ĐỒ KIẾN TRÚC HỆ THỐNG
**Đề tài:** Phát hiện Bất thường Phụ tải Điện quy mô lớn (*Large-scale Energy Anomaly Detection - LEAD*)  
**Tiêu chuẩn Thiết kế:** 100% Zero Data Leakage (Seasonal Chunking với 96h Purge Gap)  
**Môi trường khuyến nghị:** Windows 10/11 | Python 3.10+ (Đã kích hoạt môi trường ảo `.venv`)

---

## 1. Cấu trúc Thư mục Toàn diện & Ý nghĩa Từng File (Directory Map)

```text
p4aids/
├── data/                                   # Dữ liệu dự án
│   ├── train_features.csv                  # Dữ liệu gốc huấn luyện (1,75 triệu dòng)
│   ├── test_features.csv                   # Dữ liệu gốc kiểm tra (1,84 triệu dòng)
│   ├── processed/                          # Dữ liệu Parquet Zero Data Leakage cho Machine Learning
│   │   ├── train_chunked_baseline.parquet  # Tập train Baseline (40 cột = 39 đặc trưng + 1 nhãn anomaly)
│   │   ├── test_chunked_baseline.parquet   # Tập test Baseline (40 cột, 372.993 dòng độc lập Out-of-Sample)
│   │   ├── train_chunked_zone.parquet      # Tập train mở rộng tương tác Zone (43 cột)
│   │   ├── test_chunked_zone.parquet       # Tập test mở rộng tương tác Zone (43 cột)
│   │   ├── train_chunked_context.parquet   # Tập train làm giàu Ngữ cảnh vi mô (47 cột Champion)
│   │   ├── test_chunked_context.parquet    # Tập test làm giàu Ngữ cảnh vi mô (47 cột Champion)
│   │   ├── split_metadata.json             # Metadata cấu trúc phân chia mùa vụ & kiểm toán tỷ lệ nhãn
│   │   └── preprocessing_step_log.csv      # Bảng kiểm toán (audit log) 8 bước tiền xử lý
│   └── processed_DL/                       # Dữ liệu định dạng chuỗi tuần tự cho Deep Learning
│       ├── train_processed_dl.parquet      # Tập train chuỗi tuần tự theo giờ (hourly grid)
│       ├── test_processed_dl.parquet       # Tập test chuỗi tuần tự theo giờ
│       └── processed_dl_metadata.json      # Metadata cấu trúc chuỗi và thống kê chuẩn hóa
│
├── preprocessingforALL.py                  # Module lõi tiền xử lý chuẩn hóa Zero Data Leakage (xuất 6 bộ Parquet)
├── preprocessingforALL.ipynb               # Notebook Jupyter trực quan hóa kiểm toán dữ liệu và EDA
│
├── model_ML/                               # Phân hệ Học máy (Machine Learning Master Pipeline)
│   ├── pipeline_master_unified.py          # Pipeline chuẩn nguyên khối tích hợp 4 Stage (Data, Benchmark, Tune, Ensemble)
│   ├── run_master_tuning_pipeline.py       # Script chuyên sâu Optuna Bayesian Tuning cho 3 mô hình x 3 bộ dữ liệu
│   ├── benchmark_traditional_trees.py      # Pipeline đối chứng họ Cây truyền thống (RF, Extra Trees) vs Boosting trên cả 3 tập
│   ├── generate_distribution_diagrams.py   # Tự động xuất biểu đồ phân phối EDA (Histogram) & Train vs Test audit (300 DPI)
│   ├── data/                               # Thư mục dữ liệu cục bộ đồng bộ cho Machine Learning (6 Parquet)
│   ├── hyperparameters/                    # 9 tệp cấu hình tham số tối ưu và nhật ký tuning Optuna
│   │   ├── best_params_xgb_baseline.json   # XGBoost trên Baseline (PR-AUC: 0.6417)
│   │   ├── best_params_xgb_zone.json       # XGBoost trên Zone (PR-AUC: 0.6459)
│   │   ├── best_params_xgb_context.json    # XGBoost trên Context Champion (PR-AUC: 0.6915)
│   │   ├── best_params_lgbm_baseline.json  # LightGBM trên Baseline (PR-AUC: 0.6399)
│   │   ├── best_params_lgbm_zone.json      # LightGBM trên Zone (PR-AUC: 0.6443)
│   │   ├── best_params_lgbm_context.json   # LightGBM trên Context Champion (PR-AUC: 0.6863)
│   │   ├── best_params_cat_baseline.json   # CatBoost trên Baseline (PR-AUC: 0.6383)
│   │   ├── best_params_cat_zone.json       # CatBoost trên Zone (PR-AUC: 0.6379)
│   │   ├── best_params_cat_context.json    # CatBoost trên Context Champion (PR-AUC: 0.6828)
│   │   └── master_tuning_summary.txt       # Bản ghi vắn tắt tiến trình các lượt bứt phá kỷ lục
│   ├── saved_models/                       # 9 mô hình Champion chuẩn hóa định dạng .pkl
│   │   ├── XGB_Context_Champion.pkl        # XGBoost Context Champion
│   │   ├── LGBM_Context_Champion.pkl       # LightGBM Context Champion
│   │   ├── CAT_Context_Champion.pkl        # CatBoost Context Champion
│   │   ├── XGB_Zone_Champion.pkl           # XGBoost Zone Champion
│   │   ├── LGBM_Zone_Champion.pkl          # LightGBM Zone Champion
│   │   ├── CAT_Zone_Champion.pkl           # CatBoost Zone Champion
│   │   ├── XGB_Baseline_Champion.pkl       # XGBoost Baseline Champion
│   │   ├── LGBM_Baseline_Champion.pkl      # LightGBM Baseline Champion
│   │   └── CAT_Baseline_Champion.pkl       # CatBoost Baseline Champion
│   └── results/                            # Bảng tổng hợp đối chứng và hiệu năng xuất ra
│       ├── tree_benchmark_all_variants.csv # Thực nghiệm đối chứng Cây truyền thống vs Boosting trên cả 3 tập
│       ├── ensemble_summary.csv            # Tổng hợp đầy đủ tất cả mô hình đơn và các tổ hợp Ensemble
│       ├── ensemble_summary.json           # Dữ liệu JSON chi tiết của Stage 4 Ensemble
│       └── fine_tuning_summary.csv         # Tiến trình tối ưu qua các lượt Tuning
│
├── model_DL/                               # Phân hệ Học sâu (Deep Learning Pipeline)
│   ├── config.py                           # Cấu hình siêu tham số, thiết lập GPU/CPU, sliding window 252h
│   ├── data_pipeline.py                    # Pipeline nạp dữ liệu chuỗi tuần tự theo giờ, tạo mask khuyết thiếu
│   ├── models.py                           # Định nghĩa kiến trúc PyTorch (MLP, Causal TCN, BiGRU, BiLSTM, LSTM-AE)
│   ├── pipeline.py                         # Điều phối huấn luyện GroupKFold theo building_id, đánh giá và ensemble
│   └── README.md                           # Hướng dẫn chi tiết phân hệ Deep Learning
│
├── report_p4aids/                          # Báo cáo khoa học đồ án (LaTeX)
│   ├── main.tex                            # Tệp nguồn gốc tích hợp toàn bộ các chương
│   ├── compile.bat                         # Kịch bản 1-click tự động biên dịch báo cáo sang PDF
│   ├── bin/tectonic.exe                    # Trình biên dịch LaTeX độc lập (không cần cài TeX Live)
│   ├── main.pdf                            # Báo cáo đồ án hoàn chỉnh 52 trang chất lượng cao
│   ├── 0 Bảng phân công.tex                # Bảng phân công công việc các thành viên
│   ├── 1 Tổng quan về bài toán và dữ liệu.tex # Chương 1: Giới thiệu đề tài & Phân tích khám phá EDA
│   ├── 2 ML.tex & 2 DL.tex                 # Chương 2: Tổng quan tài liệu Machine Learning & Deep Learning
│   ├── 3 Tiền xử lý dữ liệu.tex            # Chương 3: 8 bước tiền xử lý chuẩn hóa & Kỹ thuật đặc trưng Zero Leakage
│   ├── 4 ML.tex                            # Chương 4: Triển khai ML, Tinh chỉnh Optuna & Đa mô hình Ensemble
│   ├── 5 DL.tex                            # Chương 5: Triển khai Học sâu tuần tự & LSTM-Autoencoder
│   ├── 6 Đánh giá, So sánh và Phân tích kết quả hiệu năng.tex # Chương 6: Đối chứng toàn diện & Trade-off
│   ├── 7 Kết luận và Hướng phát triển tương lai.tex # Chương 7: Kết luận, Đóng góp & Tài liệu tham khảo
│   └── IMAGE/                              # Thư mục hình ảnh, biểu đồ trực quan hóa nhúng trong báo cáo
│
├── requirements.txt                        # Danh sách thư viện Python cần thiết
└── run.md                                  # Hướng dẫn vận hành nhanh dự án (Tài liệu này)
```

---

## 2. Hướng dẫn Chạy Chi tiết Từng Giai đoạn (Step-by-Step Execution)

Tất cả các lệnh dưới đây được thực hiện từ thư mục gốc `d:\CODE\p4aids` trong môi trường Windows PowerShell.

### Giai đoạn 0: Khởi tạo Môi trường Ảo (Virtual Environment)
```powershell
# Tạo môi trường ảo nếu chưa có
python -m venv .venv

# Kích hoạt môi trường ảo
.\.venv\Scripts\Activate.ps1

# Cài đặt các thư viện cần thiết
pip install -r requirements.txt
```

---

### Giai đoạn 1: Tiền xử lý Dữ liệu Chuẩn hóa 100% Zero Data Leakage
Quy trình thực thi chia tách chuỗi thời gian mùa vụ kết hợp khoảng cách ly:
- **$T_{\text{in}}$ (Train):** Ngày 01 đến Ngày 20 hàng tháng (**1.147.089 dòng**). Tất cả các tham số thống kê ($\mu, \sigma$, Median, IQR, StandardScaler) được fit độc quyền trên tập này.
- **GAP (Purge):** Ngày 21 đến Ngày 24 hàng tháng (**229.412 dòng** - khoảng cách ly 96 giờ nhằm triệt tiêu hoàn toàn rò rỉ của đặc trưng trễ 73 ngày `lag73`).
- **$T_{\text{out}}$ (Test):** Ngày 25 đến Hết tháng (**372.993 dòng** - kiểm toán ngoài mẫu Out-of-Sample thực tế).

```powershell
# Chạy trực tiếp qua script Python (tạo đầy đủ 6 bộ Parquet):
python preprocessingforALL.py

# Hoặc mở Jupyter Notebook để xem trực quan hóa kiểm toán:
jupyter notebook preprocessingforALL.ipynb
```

---

### Giai đoạn 2: Xuất Biểu đồ Khám phá EDA và Kiểm toán Phân phối Dữ liệu (300 DPI)
```powershell
python model_ML/generate_distribution_diagrams.py
```
*Tự động trích xuất dữ liệu, kiểm toán thống kê và xuất 2 hình ảnh phân giải cao vào `report_p4aids/IMAGE/`:*
- `eda_histogram_distribution.png`: 6 lược đồ phân phối tần suất đặc trưng ban đầu (Skewness $+9{,}15$, biến đổi Log1p, mật độ lưỡng đỉnh nhiệt độ không khí).
- `ml_train_test_distribution.png`: Kiểm định Kolmogorov-Smirnov 2-mẫu chứng minh phân phối đồng nhất giữa $T_{\text{in}}$ và $T_{\text{out}}$, không có Covariate Shift hay rò rỉ dữ liệu.

---

### Giai đoạn 3: Khởi chạy Pipeline Học máy Toàn diện (Master Unified Pipeline)

File `model_ML/pipeline_master_unified.py` là trung tâm điều phối toàn bộ các giai đoạn Machine Learning.

#### 0. Thực nghiệm Đối chứng Cây Truyền thống vs Gradient Boosting (Cả 3 tập):
Đánh giá đối chứng Random Forest, Extra Trees so với LightGBM, XGBoost, CatBoost trên cả 3 không gian dữ liệu (Baseline, Zone, Context):
```powershell
# Chạy toàn bộ cả 3 tập dữ liệu:
python model_ML/benchmark_traditional_trees.py --variant all

# Hoặc chạy riêng từng tập:
python model_ML/benchmark_traditional_trees.py --variant baseline
python model_ML/benchmark_traditional_trees.py --variant zone
python model_ML/benchmark_traditional_trees.py --variant context
```
*Kết quả xuất ra tại `model_ML/results/tree_benchmark_all_variants.csv` và `.json`.*

#### 1. Đánh giá Mốc chuẩn Đối chứng (Viability Benchmark):
Đánh giá nhanh các mô hình trên cả 3 biến thể (Baseline, Zone, Context):
```powershell
python model_ML/pipeline_master_unified.py --benchmark
```

#### 2. Tinh chỉnh Siêu tham số Bayesian TPE với Optuna (Hỗ trợ Boosting, RF & ET):
Tối ưu hóa trực tiếp hàm mục tiêu **PR-AUC (Average Precision)** trên tập dữ liệu mất cân bằng nhãn:
```powershell
# Tinh chỉnh chuyên sâu biến thể Context Champion cho Gradient Boosting (50 trials):
python model_ML/run_master_tuning_pipeline.py --model boosting --variant context --trials 50

# Tinh chỉnh Random Forest hoặc Extra Trees (30 trials):
python model_ML/benchmark_traditional_trees.py --variant context --tune-rf --trials 30
python model_ML/benchmark_traditional_trees.py --variant context --tune-et --trials 30
# Hoặc:
python model_ML/run_master_tuning_pipeline.py --model traditional --variant context --trials 30

# Tinh chỉnh toàn bộ cả 5 mô hình (XGB, LGBM, CAT, RF, ET):
python model_ML/run_master_tuning_pipeline.py --model all --variant context --trials 30
```

#### 3. Dung hợp Đa mô hình Toàn diện (Stage 4 Multi-Model Ensemble):
Đánh giá toàn bộ các mô hình đơn lẻ và các tổ hợp dung hợp kết hợp cả Boosting (CAT, XGB, LGBM) và Cây tổ hợp (RF, ET):
- Pentad Ensemble All-5: `CAT + XGB + LGBM + RF + ET` (Rank Averaging & Soft Voting)
- Quad Ensemble: `CAT + XGB + LGBM + RF`
- Pair Ensembles: `RF + XGB`, `RF + CAT`, `CAT + XGB`, `XGB + LGBM`
```powershell
# Thực thi dung hợp toàn diện:
python model_ML/pipeline_master_unified.py --ensemble
# Hoặc:
python model_ML/benchmark_traditional_trees.py --variant context --ensemble
```
*Kết quả chi tiết được tự động xuất ra `model_ML/results/ensemble_summary.csv`.*

#### 4. Chạy toàn bộ chu trình Machine Learning từ A đến Z:
```powershell
python model_ML/pipeline_master_unified.py --all
```

---

### Giai đoạn 4: Huấn luyện Phân hệ Học sâu (Deep Learning Pipeline)

```powershell
# 1. Tiền xử lý dữ liệu chuỗi tuần tự cho Deep Learning:
python -m model_DL.pipeline preprocess

# 2. Huấn luyện và đánh giá 5 kiến trúc Học sâu (MLP, Causal TCN, BiGRU, BiLSTM, LSTM-AE):
python -m model_DL.pipeline benchmark --models mlp tcn bigru bilstm lstm_ae

# 3. Chạy tổ hợp Percentile-Rank Ensemble cho Deep Learning:
python -m model_DL.pipeline ensemble --models mlp tcn bigru bilstm lstm_ae
```

---

### Giai đoạn 5: Biên dịch Báo cáo Khoa học Luận văn sang PDF (LaTeX)

Dự án tích hợp sẵn trình biên dịch độc lập `bin/tectonic.exe`, không yêu cầu người dùng phải cài đặt TeX Live hay MiKTeX cồng kềnh:

```powershell
# Cách 1: Chạy file kịch bản tự động một chạm:
.\report_p4aids\compile.bat

# Cách 2: Gọi trực tiếp trình biên dịch Tectonic từ thư mục report_p4aids:
cd report_p4aids
.\bin\tectonic.exe main.tex
cd ..
```
*File kết quả xuất ra ngay tại:* **`report_p4aids/main.pdf`** *(Báo cáo đồ án khoa học hoàn chỉnh chất lượng cao 52 trang)*.

---

## 3. Bảng Tổng hợp Hiệu năng Đỉnh cao (Master Champion Leaderboard)

Kết quả đo đạc khách quan trên **$372.993$ mẫu Out-of-Sample ($T_{\text{out}}$)** tuân thủ 100% Zero Data Leakage:

| Xếp hạng | Mô hình / Kỹ thuật Dung hợp | PR-AUC (Gốc) | ROC-AUC | F1-Score | Precision | Recall | Ngưỡng $\tau^*$ |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 | **Ensemble_Rank_Context_All3 (CAT + XGB + LGBM)** | **0.6953** | **0.8972** | **0.7203** | **97.0%** | **57.3%** | 0.984 |
| 🥈 | **Ensemble_Rank_Context_Pair (XGB + LGBM)** | **0.6938** | **0.8976** | **0.7171** | **96.5%** | **57.1%** | 0.985 |
| 🥉 | **Ensemble_Prob_Top3 (Soft Voting Context Trio)** | **0.6931** | **0.8965** | **0.7215** | **96.1%** | **57.7%** | 0.695 |
| 4 | **Ensemble_Prob_Context_Pair (CAT + XGB)** | **0.6892** | 0.8934 | **0.7252** *(F1 Đỉnh)* | **95.0%** | **58.6%** | 0.635 |
| 5 | **XGB_Context (Single Champion Tuned)** | **0.6915** | 0.8969 | **0.7225** | **95.3%** | **58.1%** | 0.623 |
| 6 | **LGBM_Context (Single Champion Tuned)** | **0.6863** | 0.8988 | **0.7089** | **95.5%** | **56.8%** | 0.845 |
| 7 | **CAT_Context (Single Champion Tuned)** | **0.6828** | 0.8924 | **0.7113** | **87.4%** | **59.2%** | 0.755 |
| --- | *Nhóm mô hình trên tập Zone Features (43 cột)* | 0.634 - 0.646 | 0.847 - 0.862 | 0.706 - 0.709 | --- | --- | --- |
| --- | *Nhóm mô hình trên tập Baseline (40 cột)* | 0.638 - 0.642 | 0.854 - 0.863 | 0.701 - 0.710 | --- | --- | --- |
