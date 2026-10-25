"""
DEMO HƯỚNG DẪN SỬ DỤNG PIPELINE MÔ HÌNH CHO FRONT-END DEVELOPER
Dự án: Large-scale Energy Anomaly Detection (LEAD)

Chạy file này bằng lệnh:
python model_ML/exported_pipeline/demo_fe_usage.py
"""

import sys
import json
from pathlib import Path

# Đảm bảo UTF-8 cho Windows Console
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, 'reconfigure'):
            getattr(sys.stdout, 'reconfigure')(encoding='utf-8')
        if hasattr(sys.stderr, 'reconfigure'):
            getattr(sys.stderr, 'reconfigure')(encoding='utf-8')
    except Exception:
        pass

import pandas as pd
from fe_inference_engine import LeadInferenceEngine

def main():
    print("=" * 95)
    print("   DEMO TÍCH HỢP PIPELINE MÔ HÌNH HỌC MÁY (MACHINE LEARNING) VÀO FRONTEND")
    print("=" * 95)
    
    # 1. Khởi tạo Engine (Mặc định nạp mô hình Champion: Zone Weighted Blending F1 = 0.9373)
    print("\n[Bước 1] Khởi tạo LeadInferenceEngine...")
    engine = LeadInferenceEngine(model_key="zone_weighted_blend")
    print(f"-> Đang sử dụng mô hình: {engine.model_info['display_name']}")
    
    # 2. Liệt kê danh sách các mô hình khả dụng để render Dropdown trên Web UI
    print("\n[Bước 2] Danh sách các mô hình sẵn có (Render Menu Dropdown trên FE):")
    models = engine.get_available_models()
    for i, m in enumerate(models, 1):
        star = "★ [KHUYÊN DÙNG]" if m["is_recommended"] else " "
        print(f"  {i}. {star:<15} {m['display_name']}")
        print(f"     -> Phân loại: {m['category']} | F1 (Ngưỡng Động): {m['f1_score_dynamic']:.4f} | Precision: {m['precision']:.4f} | Recall: {m['recall']:.4f}")
    
    # 3. Nạp mẫu thực tế từ tập dữ liệu để kiểm nghiệm
    df_sample = pd.read_parquet(Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "train_preprocessedzone.parquet")
    normal_row = df_sample[df_sample["anomaly"] == 0].iloc[100].to_dict()
    anomaly_row = df_sample[df_sample["anomaly"] == 1].iloc[50].to_dict()
    
    # Xóa cột nhãn để giả lập dữ liệu mới từ Frontend
    normal_row.pop("anomaly", None)
    normal_row.pop("timestamp", None)
    anomaly_row.pop("anomaly", None)
    anomaly_row.pop("timestamp", None)

    # 4. Test Case 1: Dự báo Đơn lẻ - Mẫu Phụ tải VẬN HÀNH BÌNH THƯỜNG (Normal)
    print("\n" + "-" * 95)
    print("[Bước 3] Test Case 1: Mô phỏng người dùng nhập Form phụ tải BÌNH THƯỜNG")
    res_normal = engine.predict_single(normal_row, use_dynamic_threshold=True)
    print("-> Kết quả trả về (JSON cho Frontend):")
    print(json.dumps(res_normal, indent=2, ensure_ascii=False))
    
    # 5. Test Case 2: Dự báo Đơn lẻ - Mẫu Phụ tải SỰ CỐ DỊ THƯỜNG (Anomaly)
    print("\n" + "-" * 95)
    print("[Bước 4] Test Case 2: Mô phỏng phát hiện SỰ CỐ DỊ THƯỜNG (Quá tải / Sụt áp điện)")
    res_anomaly = engine.predict_single(anomaly_row, use_dynamic_threshold=True)
    print("-> Kết quả trả về (JSON cho Frontend):")
    print(json.dumps(res_anomaly, indent=2, ensure_ascii=False))
    
    # 6. Test Case 3: Thử chuyển sang mô hình Context-Aware MoE (LightGBM Meta-Learner)
    print("\n" + "-" * 95)
    print("[Bước 5] Test Case 3: Chuyển đổi mô hình sang Context-Aware MoE (Meta-Learner):")
    engine.set_model("context_aware_moe")
    res_moe = engine.predict_single(anomaly_row, use_dynamic_threshold=True)
    print(f"-> Mô hình: {res_moe['model_used']}")
    print(f"   + Xác suất Dị thường: {res_moe['anomaly_probability']*100:.2f}%")
    print(f"   + Ngưỡng áp dụng: {res_moe['applied_threshold']} ({res_moe['threshold_type']})")
    print(f"   + Đánh giá rủi ro: {res_moe['risk_label']}")
    print(f"   + Diễn giải cảnh báo: {res_moe['explanation']}")
    
    # 7. Test Case 4: Thử nghiệm Dự báo Hàng loạt (Batch Prediction từ File CSV)
    print("\n" + "-" * 95)
    print("[Bước 6] Test Case 4: Mô phỏng người dùng Tải lên File CSV đo lường nhiều công trình:")
    sample_batch_df = df_sample.sample(10, random_state=42).drop(columns=["anomaly", "timestamp"], errors="ignore")
    
    engine.set_model("zone_weighted_blend") # Quay lại mô hình Champion
    batch_res_df, summary = engine.predict_batch(sample_batch_df, use_dynamic_threshold=True)
    
    print("\n-> Bảng Thống kê Tóm tắt Dashboard (Render Thẻ KPI trên UI):")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    
    print("\n-> Bảng Chi tiết Kết quả Dự báo từng dòng (Render Bảng Dữ liệu trên FE):")
    display_cols = ["building_id", "site_id", "meter_reading", "anomaly_prob", "applied_threshold", "is_anomaly", "risk_level"]
    print(batch_res_df[display_cols].to_string(index=False))
    
    print("\n" + "=" * 95)
    print("   DEMO HOÀN TẤT THÀNH CÔNG! ENGINE SẴN SÀNG CHO BẠN BẬC FRONT-END TÍCH HỢP.")
    print("=" * 95)

if __name__ == "__main__":
    main()
