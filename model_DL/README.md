# Deep Learning pipeline

## Chunked datasets with timestamps

The six immutable datasets in `data/processed/` do not contain `timestamp`.
To prepare separate timestamped copies for later DL training, run from the
repository root:

```powershell
python -m model_DL.prepare_chunked_data
```

This reads `energy-anomaly-detection/train_features.csv` for timestamps only,
checks row order against `building_id`, `anomaly`, and the six time features,
then writes these files to `data/processed_DL/`:

```text
train_chunked_baseline_timestamp.parquet
train_chunked_zone_timestamp.parquet
train_chunked_context_timestamp.parquet
test_chunked_baseline_timestamp.parquet
test_chunked_zone_timestamp.parquet
test_chunked_context_timestamp.parquet
chunked_timestamp_metadata.json
```

All original columns and values are preserved. The metadata records input and
output SHA-256 hashes and row counts. Existing outputs are never overwritten.
The legacy `model_DL.pipeline` still uses `train_processed_dl.parquet` and
`test_processed_dl.parquet`. The chunked training command below reads the new
timestamped files explicitly.

## Train on the timestamped chunked datasets

Use the separate entry point below. It keeps the original Kaggle pipeline and
its saved artifacts intact:

```powershell
# Short end-to-end check: 8 buildings, 2 folds, 1 epoch.
python -m model_DL.chunked_pipeline train --smoke

# Full run: three variants, five models, five GroupKFold splits.
python -m model_DL.chunked_pipeline train

# Recalculate the metrics from saved OOF and T_out prediction files.
python -m model_DL.chunked_pipeline evaluate --run-id full_YYYYMMDDTHHMMSSZ
```

`--variants` and `--models` can narrow a run. `--epochs`, `--folds`,
`--patience`, `--batch-size`, `--workers`, `--device`, and `--run-id` are also
available. Full runs default to 30 epochs at most, with early stopping after
five epochs without a better validation Average Precision (AP). The default
batch size is 1024 for MLP and 16 for each sequence model; reduce it if GPU
memory is insufficient. Every run uses a new directory under
`model_DL/artifacts/chunked/<run-id>/`. Generated runs stay on disk and are
ignored by Git until explicitly added. Each run writes:

```text
run_config.json
summary.csv
<variant>/variant_config.json
<variant>/checkpoints/<model>/fold_<n>.pt
<variant>/logs/history_<model>_fold_<n>.csv
<variant>/logs/metrics_<model>.json
<variant>/predictions/oof_<model>.parquet
<variant>/predictions/holdout_<model>.parquet
```

The per-row prediction files contain `processed_position`, `building_id`,
`timestamp`, `anomaly_true`, and `score`. GroupKFold uses `T_in` only. Best
checkpoints use validation AP, with ROC-AUC and loss logged each epoch. The
F1 threshold comes from `T_in` OOF scores and is applied unchanged to labeled
`T_out`; the holdout is never used for training or model selection. Sequence
windows reset at each non-hourly gap and cover every selected row once. The
training feature set excludes `gte_*`, as in the original DL pipeline, until
their provenance has been established. `variant_config.json` records the exact
feature list. Precomputed chunked features were fitted on full `T_in` before
the model's GroupKFold, so its CV results are exploratory for features that
depend on building-level statistics.

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
