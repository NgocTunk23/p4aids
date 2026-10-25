# HƯỚNG DẪN TÍCH HỢP PIPELINE MÔ HÌNH HỌC MÁY CHO FRONT-END (FE)
**Dự án:** Hệ thống Phát hiện Bất thường Phụ tải Điện quy mô lớn (*Large-scale Energy Anomaly Detection - LEAD*)  
**Dành cho:** Bạn phụ trách phát triển Giao diện người dùng (Front-End / Web App / API Backend)

---

## 1. Cấu trúc Thư mục Đóng gói (`model_ML/exported_pipeline/`)

Toàn bộ mô hình đã được đóng gói độc lập, tự động xử lý tiền xử lý và tương thích mọi thư viện:

```text
model_ML/exported_pipeline/
├── models/                                 # Thư mục chứa trọng số mô hình đã huấn luyện (.pkl)
│   ├── XGBoost_Zone.pkl                    # Model XGBoost Zone tối ưu
│   ├── LightGBM_Zone.pkl                   # Model LightGBM Zone tối ưu
│   ├── CatBoost_Zone.pkl                   # Model CatBoost Zone tối ưu
│   ├── MoE_Zone.pkl                        # Meta-Learner LightGBM (Context-Aware MoE)
│   ├── XGBoost_Baseline.pkl                # Model XGBoost Baseline
│   ├── LightGBM_Baseline.pkl               # Model LightGBM Baseline
│   └── CatBoost_Baseline.pkl               # Model CatBoost Baseline
│
├── ensemble_metadata.json                  # Cấu hình Metadata: 16 ngưỡng động, trọng số, F1-scores
├── fe_inference_engine.py                  # Class suy luận thông minh LeadInferenceEngine
├── demo_fe_usage.py                        # Script mẫu chạy thử nghiệm mọi tính năng (Smoke Test)
└── README_FE.md                            # Tài liệu hướng dẫn này
```

---

## 2. Cách Sử dụng Siêu Nhanh (Chỉ với 3 dòng code!)

```python
from model_ML.exported_pipeline.fe_inference_engine import LeadInferenceEngine

# 1. Khởi tạo engine (Mặc định tự động chọn mô hình Champion F1 = 0.9373)
engine = LeadInferenceEngine()

# 2. Dự báo từ Form Web (1 bản ghi dạng dictionary)
result = engine.predict_single({
    "building_id": 439,
    "site_id": 4,
    "meter_reading": 125.0,
    "square_feet": 100000,
    "air_temperature": 28.5,
    "timestamp": "2026-10-25 14:00:00"
})

# 3. Kết quả trả về dạng JSON trực tiếp cho Frontend
print(result["is_anomaly"])          # True hoặc False
print(result["anomaly_probability"]) # Xác suất dị thường (0.00 đến 1.00)
print(result["risk_level"])          # "NORMAL", "MEDIUM", "HIGH", "CRITICAL"
print(result["risk_color"])          # "#2ECC71", "#F1C40F", "#E67E22", "#E74C3C"
print(result["explanation"])         # Chuỗi diễn giải cảnh báo tự nhiên
```

---

## 3. Bảng Danh mục Mô hình (Dùng để Render Dropdown Menu trên FE)

Bạn có thể gọi hàm `engine.get_available_models()` để lấy danh sách mô hình đưa vào thẻ `<select>` hoặc Dropdown:

| Model Key | Tên Hiển Thị (Display Name) | Phân Loại | F1 (Ngưỡng Động) | Khuyến Nghị Sử Dụng |
| :--- | :--- | :--- | :---: | :--- |
| **`zone_weighted_blend`** | **Weighted Blending LogLoss [0.65 XGB + 0.25 LGB + 0.10 Cat]** | **Ensemble** | **0.9373** | ★ **CHAMPION TỐT NHẤT TOÀN ĐỀ TÀI** |
| `zone_dynamic_thresh_blend` | Dynamic Threshold Blend [0.60 XGB + 0.25 LGB + 0.15 Cat] | Ensemble | 0.9372 | Phương án đối chứng ngưỡng động |
| `zone_soft_voting` | Soft Voting [XGB + LGB + Cat] (Trung bình cộng) | Ensemble | 0.9359 | Phương án voting cào bằng |
| `context_aware_moe` | Context-Aware Mixture of Experts (Meta-Learner LightGBM) | Ensemble | 0.9354 | Định tuyến thích nghi theo bối cảnh |
| `zone_top2_blend` | Top-2 Blend [0.80 XGB + 0.20 Cat] | Ensemble | 0.9348 | Gọn nhẹ, tiết kiệm tài nguyên |
| `xgboost_zone` | XGBoost Tuned (Zone Single Model) | Single Model | 0.9341 | Đơn mô hình nhanh nhất |
| `xgboost_baseline` | XGBoost Tuned (Baseline Single Model) | Single Model | 0.9349 | Dữ liệu cơ sở không cần Zone |

**Cách đổi mô hình khi người dùng chọn Dropdown:**
```python
engine.set_model("context_aware_moe")  # hoặc "zone_top2_blend", "xgboost_zone"...
```

---

## 4. Dự báo Hàng loạt (Batch Prediction - Khi Người dùng Tải file CSV)

```python
# Dự báo cho file CSV người dùng tải lên
result_df, summary = engine.predict_batch("user_uploaded_data.csv")

# 1. Thống kê tóm tắt để vẽ Thẻ KPI trên Dashboard:
print(summary["total_samples"])      # Tổng số bản ghi đo lường
print(summary["anomaly_count"])      # Số lượng điểm bất thường
print(summary["anomaly_percentage"]) # Tỷ lệ dị thường (%)
print(summary["critical_cases"])     # Số ca nguy cấp báo động đỏ

# 2. DataFrame kết quả đã được gán sẵn các cột:
# 'anomaly_prob', 'is_anomaly', 'applied_threshold', 'risk_level'
result_df.to_csv("analyzed_result.csv", index=False)
```

---

## 5. Mẫu Tích hợp Backend API (FastAPI / Streamlit)

### Mẫu 1: Xây dựng REST API bằng FastAPI (File `api.py`):
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from fe_inference_engine import LeadInferenceEngine

app = FastAPI(title="LEAD Anomaly Detection API")
engine = LeadInferenceEngine()

class EnergyData(BaseModel):
    building_id: int
    site_id: int
    meter_reading: float
    square_feet: Optional[float] = 100000.0
    air_temperature: Optional[float] = 20.0
    dew_temperature: Optional[float] = 10.0
    wind_speed: Optional[float] = 3.0
    timestamp: Optional[str] = None
    model_key: Optional[str] = "zone_weighted_blend"

@app.get("/models")
def get_models():
    """Endpoint cho Dropdown danh sách mô hình trên Frontend"""
    return engine.get_available_models()

@app.post("/predict")
def predict_anomaly(data: EnergyData):
    """Endpoint phân tích dị thường phụ tải điện"""
    if data.model_key != engine.model_key:
        engine.set_model(data.model_key)
    result = engine.predict_single(data.dict())
    return result
```

### Mẫu 2: Xây dựng Giao diện Web bằng Streamlit (File `app.py`):
```python
import streamlit as st
import pandas as pd
from fe_inference_engine import LeadInferenceEngine

st.set_page_config(page_title="LEAD Anomaly Detection", layout="wide")
st.title("⚡ Hệ thống Giám sát & Phát hiện Bất thường Phụ tải Điện")

@st.cache_resource
def get_engine():
    return LeadInferenceEngine()

engine = get_engine()

# 1. Sidebar chọn mô hình
models = engine.get_available_models()
model_opts = {m["display_name"]: m["model_key"] for m in models}
selected_disp = st.sidebar.selectbox("Lựa chọn Mô hình:", list(model_opts.keys()))
engine.set_model(model_opts[selected_disp])

# 2. Form nhập liệu
col1, col2, col3 = st.columns(3)
with col1:
    meter_val = st.number_input("Phụ tải điện đo được (kWh):", value=120.0)
    building_id = st.number_input("ID Tòa nhà:", value=439)
with col2:
    site_id = st.selectbox("Khu vực (Site ID):", list(range(16)), index=4)
    air_temp = st.number_input("Nhiệt độ môi trường (°C):", value=28.0)
with col3:
    sqft = st.number_input("Diện tích sàn (ft²):", value=100000)

if st.button("🔍 Phân tích Phụ tải Ngay", type="primary"):
    res = engine.predict_single({
        "meter_reading": meter_val,
        "building_id": building_id,
        "site_id": site_id,
        "air_temperature": air_temp,
        "square_feet": sqft
    })
    
    if res["is_anomaly"]:
        st.error(f"🚨 {res['explanation']}")
    else:
        st.success(f"✅ {res['explanation']}")
        
    st.metric("Xác suất Dị thường", f"{res['anomaly_probability']*100:.2f}%")
    st.metric("Ngưỡng áp dụng", f"{res['applied_threshold']} ({res['threshold_type']})")
```
