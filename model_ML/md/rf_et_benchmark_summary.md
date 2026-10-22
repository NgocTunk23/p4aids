# Báo cáo Thực nghiệm Random Forest & Extra Trees sau Tuning (10 Trials)
- **Ngày thực hiện:** 2026-10-22 05:09:17
- **Phần cứng:** CPU 20 Cores đa luồng, hỗ trợ Intel oneAPI (scikit-learn-intelex)
- **Quy trình:** 10 vòng lặp Optuna Bayesian TPE trên tập 80% Train / 20% Stratified Validation

| Mô hình | Bộ dữ liệu | Ngưỡng $\tau^*$ | Precision | Recall | F1 (0.5) | F1 ($\tau^*$) | Fit Time (s) | Tổng Tuning (s) |
|---|---|---|---|---|---|---|---|---|
| **Random Forest** | Baseline | 0.4507 | 0.9336 | 0.7902 | 0.8501 | **0.8559** | 36.0s | 424.8s |
| **Extra Trees** | Baseline | 0.1985 | 0.9019 | 0.8564 | 0.8285 | **0.8786** | 30.6s | 528.1s |
| **Random Forest** | Zone | 0.4186 | 0.9239 | 0.7993 | 0.8479 | **0.8571** | 38.8s | 410.4s |
| **Extra Trees** | Zone | 0.2041 | 0.9053 | 0.8591 | 0.8355 | **0.8816** | 31.7s | 397.0s |

## Chi tiết Siêu tham số Tối ưu nhất:
### Random Forest - Baseline:
```json
{
  "n_estimators": 90,
  "max_depth": 22,
  "min_samples_split": 13,
  "min_samples_leaf": 9,
  "max_features": 0.41809091556421524,
  "max_samples": 0.6301321323053057
}
```
### Extra Trees - Baseline:
```json
{
  "n_estimators": 90,
  "max_depth": 22,
  "min_samples_split": 13,
  "min_samples_leaf": 9,
  "max_features": 0.41809091556421524,
  "max_samples": 0.6301321323053057
}
```
### Random Forest - Zone:
```json
{
  "n_estimators": 90,
  "max_depth": 22,
  "min_samples_split": 13,
  "min_samples_leaf": 9,
  "max_features": 0.41809091556421524,
  "max_samples": 0.6301321323053057
}
```
### Extra Trees - Zone:
```json
{
  "n_estimators": 90,
  "max_depth": 22,
  "min_samples_split": 13,
  "min_samples_leaf": 9,
  "max_features": 0.41809091556421524,
  "max_samples": 0.6301321323053057
}
```
