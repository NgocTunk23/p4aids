# Deep Learning pipeline

Pipeline đọc trực tiếp dữ liệu Kaggle trong `energy-anomaly-detection/`, tạo
dataset riêng tại `data/processed_DL/`, đánh giá mọi model bằng cùng GroupKFold
theo `building_id`, rồi sinh submission riêng cho từng model.

## Kiến trúc

```text
energy-anomaly-detection/                 # raw, git-ignored
  train_features.csv
  test_features.csv
  sample_submission.csv
              |
              v
model_DL/data_pipeline.py                 # schema, hourly grid, mask, feature
              |
              v
data/processed_DL/
  train_processed_dl.parquet
  test_processed_dl.parquet
  processed_dl_metadata.json
              |
              v
model_DL/pipeline.py
  GroupKFold(building_id) -> fold transform -> train/evaluate/predict test
              |
              +--> artifacts/checkpoints/<model>/fold_*.pt
              +--> artifacts/predictions/oof_<model>.parquet
              +--> artifacts/predictions/submission_<model>.csv
              +--> artifacts/logs/benchmark_summary.csv
```

`models.py` cung cấp MLP, causal TCN, BiGRU, BiLSTM và LSTM autoencoder. TCN
dùng 252 giờ causal context và dự đoán 168 giờ tiếp theo. Mỗi target row chỉ
được chấm một lần. LSTM-AE học reconstruction trên điểm normal và dùng
reconstruction error làm anomaly score.

## Chạy

Chạy từ thư mục gốc repo `p4aids`:

```powershell
python -m model_DL.pipeline preprocess

# Smoke test trước
python -m model_DL.pipeline benchmark --models mlp --folds 2 --epochs 1 --workers 0

# Benchmark đầy đủ; mỗi fold đồng thời predict test
python -m model_DL.pipeline benchmark --models mlp tcn bigru bilstm lstm_ae

# Predict lại test từ checkpoint mà không train lại
python -m model_DL.pipeline predict-test --models mlp tcn bigru bilstm lstm_ae

# Ensemble percentile-rank (phù hợp cả AE score và classifier probability)
python -m model_DL.pipeline ensemble --models mlp tcn bigru bilstm lstm_ae

# Tính lại metric từ OOF đã lưu
python -m model_DL.pipeline evaluate --models mlp tcn bigru bilstm lstm_ae
```

Với GPU 6 GB, dùng `--batch-size 16` cho TCN/RNN nếu batch 64 gây CUDA OOM.
Model và fold chạy tuần tự, AMP bật tự động trên CUDA, checkpoint tốt nhất được
chọn bằng validation ROC-AUC.

`workers=0` là mặc định chủ ý trên Windows để không pickle/copy toàn bộ mảng dữ
liệu cho subprocess. Có thể thử `--workers 1` sau nếu chuyển dataset sang
memory-map.

## Data contract và chống leakage

- `row_exists_mask`: 0 cho giờ reindex thêm; chỉ cung cấp context, không vào
  loss/metric/submission.
- `meter_missing_mask`: phân biệt meter thiếu với giá trị scaler điền median.
- `label_mask`: chỉ train/evaluate nhãn có thật.
- Feature thời gian, delta và run length đều causal; weather chỉ forward-fill.
- Median/scaler/category mapping chỉ fit trên building train của từng fold.
- `gte_*` và interaction dạng chuỗi bị loại vì provenance/mapping không an toàn.
- Submission giữ đúng `row_id` của `sample_submission.csv`, chứa continuous
  score chứ không threshold thành 0/1.

F1/threshold trong report chỉ là diagnostic chọn trên OOF. ROC-AUC là metric
chính để xếp hạng và chấm submission Kaggle. Report có cả point-wise metric và
macro ROC-AUC/PR-AUC theo building.
