# HƯỚNG DẪN VẬN HÀNH DỰ ÁN & BẢN ĐỒ KIẾN TRÚC HỆ THỐNG
**Đề tài:** Phát hiện Bất thường Phụ tải Điện quy mô lớn (*Large-scale Energy Anomaly Detection - LEAD*)  
**Môi trường khuyến nghị:** Windows 10/11 | Python 3.10+ (Đã kích hoạt môi trường ảo `.venv`)

---

## 1. Cấu trúc Thư mục Toàn diện & Ý nghĩa Từng File (Directory Map)

```text
p4aids/
├── data/                                   # Dữ liệu dự án
│   ├── train_features.csv                  # Dữ liệu gốc huấn luyện (1,75 triệu dòng)
│   ├── test_features.csv                   # Dữ liệu gốc kiểm tra (1,84 triệu dòng)
│   ├── processed/                          # Dữ liệu đã qua tiền xử lý cho Machine Learning
│   │   ├── train_preprocessed.parquet      # Tập train Baseline sau 17 bước làm sạch chuẩn hóa
│   │   ├── test_preprocessed.parquet       # Tập test Baseline
│   │   ├── train_preprocessedzone.parquet  # Tập train mở rộng tích hợp Đặc trưng Tương tác Zone
│   │   ├── test_preprocessedzone.parquet   # Tập test tích hợp Đặc trưng Tương tác Zone
│   │   └── preprocessing_step_log.csv      # Bảng ghi vết (audit log) 17 bước tiền xử lý
│   └── processed_DL/                       # Dữ liệu định dạng chuỗi tuần tự cho Deep Learning
│       ├── train_processed_dl.parquet      # Tập train chuỗi tuần tự theo giờ (hourly grid)
│       ├── test_processed_dl.parquet       # Tập test chuỗi tuần tự theo giờ
│       └── processed_dl_metadata.json      # Metadata cấu trúc chuỗi và thống kê chuẩn hóa
│
├── preprocessingforALL.ipynb               # Notebook Jupyter thực hiện tiền xử lý & trích xuất đặc trưng
│
├── model_ML/                               # Phân hệ Học máy (Machine Learning Pipeline)
│   ├── model_classes.py                    # Khung bao bọc mô hình chuẩn hóa (XGBoost, LightGBM, CatBoost, RF, ET)
│   ├── run_tuning_pipeline.py              # Script Optuna tuning 50 trials cho nhánh dữ liệu Baseline
│   ├── tune_zone_optuna.py                 # Script Optuna tuning 50 trials cho nhánh dữ liệu Zone Features
│   ├── test_isolation_forest_comparison.py # Kiểm định thực nghiệm đối chứng & bác bỏ Isolation Forest
│   ├── test_lag_vs_zone_comparison.py      # Kiểm định đối chứng 3 chiều: Baseline vs Lag Features vs Zone Features
│   ├── tune_rf_et_both_datasets.py         # Kiểm định thực nghiệm & tuning mô hình cây truyền thống (RF, ET)
│   ├── benchmark_all_single_models_baseline.py # Đo thời gian fit & hiệu năng 5 mô hình đơn đa lõi trên Baseline
│   ├── run_moe_zone.py                     # Huấn luyện mô hình Context-Aware MoE (LightGBM Meta-Learner)
│   ├── run_final_comprehensive_new.py      # Chạy đánh giá toàn diện tất cả mô hình đơn & 6 kỹ thuật Ensemble
│   ├── pipeline_full_train_and_submit.py   # Huấn luyện trên 100% dữ liệu và suy luận tạo file nộp bài
│   ├── hyperparameters/                    # Thư mục lưu siêu tham số tối ưu & kết quả thực nghiệm (.json)
│   │   ├── baseline_tuning_best_params.json         # Siêu tham số tối ưu Optuna cho Baseline
│   │   ├── zone_tuning_best_params.json             # Siêu tham số tối ưu Optuna cho Zone Features
│   │   ├── isolation_forest_test_results.json       # Kết quả đối chứng đặc trưng Isolation Forest
│   │   ├── lag_vs_zone_test_results.json            # Kết quả đối chứng Lag vs Zone vs Baseline
│   │   ├── rf_et_tuning_results.json                # Kết quả tuning Random Forest và Extra Trees
│   │   ├── single_models_baseline_benchmark.json    # Benchmark chi tiết 5 mô hình đơn trên Baseline
│   │   ├── moe_dynamic_site_thresholds.json         # Bộ 16 ngưỡng động tối ưu của Context-Aware MoE
│   │   └── final_comprehensive_dynamic_thresholds.json # Bộ 16 ngưỡng động cho tất cả các mô hình
│   ├── saved_new_models/                   # Lưu trữ checkpoint mô hình đã huấn luyện (.pkl)
│   │   ├── XGBoost_Baseline_Full.pkl       # Mô hình XGBoost Baseline
│   │   ├── XGBoost_Zone.pkl                # Mô hình XGBoost Zone
│   │   ├── LightGBM_Zone.pkl               # Mô hình LightGBM Zone
│   │   ├── CatBoost_Zone.pkl               # Mô hình CatBoost Zone
│   │   ├── MoE_Zone.pkl                    # Checkpoint Meta-Learner LightGBM MoE
│   │   └── MoE_Zone_Optimal.pkl            # Checkpoint MoE tối ưu toàn diện
│   ├── submission/                         # Thư mục lưu kết quả dự đoán trên tập test (.csv)
│   │   ├── submission_best_model.csv       # Kết quả mô hình tốt nhất (Zone Ensemble - F1 = 0.9373)
│   │   ├── submission_xgboost_zone.csv     # Kết quả XGBoost Zone (F1 = 0.9341)
│   │   └── submission_xgboost_no_zone.csv  # Kết quả XGBoost Baseline (F1 = 0.9349)
│   └── md/                                 # Báo cáo đánh giá tự động dạng Markdown
│       ├── final_comprehensive_report_new.md # Báo cáo hiệu năng toàn diện tất cả mô hình & ensemble
│       ├── moe_zone_report.md               # Báo cáo đánh giá chi tiết Context-Aware MoE
│       └── rf_et_benchmark_summary.md       # Báo cáo so sánh Random Forest và Extra Trees
│
├── model_DL/                               # Phân hệ Học sâu (Deep Learning Pipeline)
│   ├── config.py                           # Cấu hình siêu tham số, thiết lập GPU/CPU, sliding window
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
│   ├── 3 Tiền xử lý dữ liệu.tex            # Chương 3: 17 bước tiền xử lý chuẩn hóa & Kỹ thuật đặc trưng
│   ├── 4 ML.tex                            # Chương 4: Triển khai ML, Tinh chỉnh Optuna & 6 Kỹ thuật Ensemble
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

### Giai đoạn 1: Tiền xử lý Dữ liệu & Trích xuất Đặc trưng (Data Preprocessing)
* **Tiền xử lý cho Machine Learning (Baseline & Zone Features):**  
  Chạy toàn bộ các cells trong notebook:
  ```powershell
  # Khởi chạy Jupyter Lab / Notebook để chạy file
  jupyter notebook preprocessingforALL.ipynb
  ```
  *Quá trình này thực hiện 17 bước xử lý nghiêm ngặt (nội suy chuỗi thời gian nội bộ tòa nhà, mã hóa chu kỳ lượng giác $\sin/\cos$, xử lý dữ liệu ngoại lai, trích xuất tương tác `zone_interaction`, `site_sqft_mean`, `use_sqft_mean`) và xuất dữ liệu ra `data/processed/`.*

* **Tiền xử lý cho Deep Learning (Chuỗi tuần tự theo giờ):**
  ```powershell
  .\.venv\Scripts\python.exe -m model_DL.pipeline preprocess
  ```
  *Tạo lưới chuỗi thời gian đều đặn (hourly grid) và xuất dữ liệu ra `data/processed_DL/`.*

---

### Giai đoạn 2: Các Thực nghiệm Kiểm định Độc lập & Bác bỏ Kỹ thuật

1. **Kiểm định và bác bỏ đặc trưng Isolation Forest (Unsupervised Meta-Feature):**
   ```powershell
   .\.venv\Scripts\python.exe model_ML/test_isolation_forest_comparison.py
   ```
   *Chứng minh việc đưa điểm số dị biệt từ Isolation Forest vào làm suy giảm F1-Score và Recall của cả 3 mô hình Boosting.*

2. **Kiểm định so sánh Đối chứng: Baseline vs Lag Features vs Zone Features:**
   ```powershell
   .\.venv\Scripts\python.exe model_ML/test_lag_vs_zone_comparison.py
   ```
   *Chứng minh Đặc trưng Trễ (Lag) gây nhiễu và sụt giảm hiệu năng, trong khi Đặc trưng Phân vùng (Zone) nâng trần F1-Score lên mức cao nhất.*

3. **Kiểm định và Tinh chỉnh Họ mô hình cây truyền thống (Random Forest vs Extra Trees):**
   ```powershell
   .\.venv\Scripts\python.exe model_ML/tune_rf_et_both_datasets.py
   ```
   *Đo đạc hiệu năng của RF ($F_1 = 0{,}8559$) và ET ($F_1 = 0{,}8786$), làm cơ sở thực nghiệm loại bỏ Bagging truyền thống.*

4. **Benchmark Đối chứng Đa lõi 5 Mô hình Đơn trên Baseline (Đo thời gian fit & F1):**
   ```powershell
   .\.venv\Scripts\python.exe model_ML/benchmark_all_single_models_baseline.py
   ```
   *Đo đạc khách quan thời gian huấn luyện đa lõi và chỉ số F1 (ngưỡng 0.5 vs ngưỡng tối ưu $\tau^*$).*

---

### Giai đoạn 3: Tinh chỉnh Siêu tham số với Optuna (Hyperparameter Tuning)

1. **Tuning 50 trials cho 3 thuật toán Boosting trên tập Baseline:**
   ```powershell
   .\.venv\Scripts\python.exe model_ML/run_tuning_pipeline.py
   ```
   *Tối ưu hóa các siêu tham số cho XGBoost, LightGBM, CatBoost; kết quả tự động ghi vào `model_ML/hyperparameters/baseline_tuning_best_params.json`.*

2. **Tuning 50 trials cho 3 thuật toán Boosting trên tập Zone Features:**
   ```powershell
   .\.venv\Scripts\python.exe model_ML/tune_zone_optuna.py
   ```
   *Tối ưu hóa các siêu tham số cho XGBoost, LightGBM, CatBoost trên tập Zone; kết quả tự động ghi vào `model_ML/hyperparameters/zone_tuning_best_params.json`.*

---

### Giai đoạn 4: Đánh giá Toàn diện & Huấn luyện Dung hợp (Ensemble Pipeline)

1. **Chạy Đánh giá Toàn diện Tất cả Mô hình Đơn & 6 Phương pháp Dung hợp (Ensemble Benchmark):**
   ```powershell
   .\.venv\Scripts\python.exe model_ML/run_final_comprehensive_new.py
   ```
   *Huấn luyện và so sánh đồng thời:*
   - *3 Mô hình đơn Baseline (XGBoost, LightGBM, CatBoost)*
   - *3 Mô hình đơn Zone Features*
   - *6 Kỹ thuật Ensemble: Soft Voting, Top-2 Blend, Weighted Blending LogLoss, Dynamic Thresh Blend, Stacking LR, Context-Aware MoE.*
   - *Tự động xuất báo cáo toàn diện tại `model_ML/md/final_comprehensive_report_new.md`.*

2. **Chạy riêng Mô hình Context-Aware MoE (Meta-Learner LightGBM) và tìm 16 Ngưỡng Động:**
   ```powershell
   .\.venv\Scripts\python.exe model_ML/run_moe_zone.py
   ```
   *Nạp 3 model Zone đã huấn luyện, khớp Meta-Learner LightGBM và xuất chi tiết 16 ngưỡng động từng khu vực ra `model_ML/hyperparameters/moe_dynamic_site_thresholds.json`.*

3. **Chạy Pipeline Phân hệ Học sâu (Deep Learning):**
   ```powershell
   # 1. Chạy thử nghiệm nhanh (Smoke test)
   .\.venv\Scripts\python.exe -m model_DL.pipeline benchmark --models mlp --folds 2 --epochs 1 --workers 0

   # 2. Huấn luyện và đánh giá toàn diện 5 mô hình DL (MLP, Causal TCN, BiGRU, BiLSTM, LSTM-AE)
   .\.venv\Scripts\python.exe -m model_DL.pipeline benchmark --models mlp tcn bigru bilstm lstm_ae

   # 3. Chạy tổ hợp Percentile-Rank Ensemble cho Deep Learning
   .\.venv\Scripts\python.exe -m model_DL.pipeline ensemble --models mlp tcn bigru bilstm lstm_ae
   ```

---

### Giai đoạn 5: Huấn luyện Toàn diện & Suy luận Dự đoán (Inference & Submission)

Chạy script tạo các file dự đoán trên toàn bộ tập kiểm tra (`test_features.csv`):
```powershell
# Chạy tất cả các cấu hình
.\.venv\Scripts\python.exe model_ML/pipeline_full_train_and_submit.py
```

* **Chạy riêng lẻ từng mô hình cụ thể:**
  - **Mô hình Tốt nhất Đề tài (Zone Ensemble + 16 Ngưỡng Động - $F_1 = 0{,}9373$):**
    ```powershell
    .\.venv\Scripts\python.exe model_ML/pipeline_full_train_and_submit.py --model best
    ```
    *File xuất ra:* `model_ML/submission/submission_best_model.csv`
  - **Mô hình Đơn XGBoost Zone ($F_1 = 0{,}9341$):**
    ```powershell
    .\.venv\Scripts\python.exe model_ML/pipeline_full_train_and_submit.py --model xgb_zone
    ```
    *File xuất ra:* `model_ML/submission/submission_xgboost_zone.csv`
  - **Mô hình Đơn XGBoost Baseline ($F_1 = 0{,}9349$):**
    ```powershell
    .\.venv\Scripts\python.exe model_ML/pipeline_full_train_and_submit.py --model xgb_no_zone
    ```
    *File xuất ra:* `model_ML/submission/submission_xgboost_no_zone.csv`

---

### Giai đoạn 6: Biên dịch Báo cáo Khoa học LaTeX sang PDF

1. **Cách 1: Chạy file kịch bản tự động một chạm (Khuyến nghị):**
   ```powershell
   .\report_p4aids\compile.bat
   ```
   *(Hoặc nhấp đúp chuột trực tiếp vào file `compile.bat` trong thư mục `report_p4aids/`)*.

2. **Cách 2: Biên dịch trực tiếp bằng trình biên dịch Tectonic nhúng:**
   ```powershell
   cd report_p4aids
   .\bin\tectonic.exe main.tex
   cd ..
   ```
*File kết quả xuất ra ngay tại thư mục:* **`report_p4aids/main.pdf`** *(Báo cáo đồ án khoa học hoàn chỉnh 52 trang chất lượng cao)*.
