"""
MODULE SUY LUẬN CHUYÊN DỤNG CHO FRONT-END (FE INFERENCE ENGINE)
Dự án: Phát hiện Dị thường Phụ tải Điện quy mô lớn (LEAD)

Cung cấp Class `LeadInferenceEngine` tiện dụng, an toàn, chống crash, 
hỗ trợ cả dự đoán đơn lẻ (từ Form web UI) và dự đoán hàng loạt (từ File CSV người dùng tải lên).
"""

import os
import sys
import json
import joblib
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

import numpy as np
import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
MODELS_DIR = CURRENT_DIR / "models"
METADATA_FILE = CURRENT_DIR / "ensemble_metadata.json"

class LeadInferenceEngine:
    """
    Công cụ suy luận phân loại dị thường phụ tải điện phục vụ Frontend.
    Hỗ trợ cả mô hình đơn (XGBoost, LightGBM, CatBoost) và 5 phương pháp Ensemble hàng đầu.
    """
    
    def __init__(self, model_key="zone_weighted_blend", models_dir=None):
        self.models_dir = Path(models_dir) if models_dir else MODELS_DIR
        self.metadata = self._load_metadata()
        self.loaded_models = {}
        self.model_key = None
        self.set_model(model_key)
        
    def _load_metadata(self):
        if not METADATA_FILE.exists():
            raise FileNotFoundError(f"Không tìm thấy tệp metadata tại: {METADATA_FILE}")
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
            
    def get_available_models(self):
        """Trả về danh sách các mô hình và ensemble sẵn sàng cho Dropdown FE."""
        models_info = []
        for key, info in self.metadata["available_models"].items():
            models_info.append({
                "model_key": key,
                "display_name": info["display_name"],
                "category": info["category"],
                "is_recommended": info.get("is_recommended", False),
                "f1_score_dynamic": info.get("f1_score_dynamic", 0.0),
                "f1_score_global": info.get("f1_score_global", 0.0),
                "precision": info.get("precision", 0.0),
                "recall": info.get("recall", 0.0),
                "description": info.get("description", "")
            })
        return sorted(models_info, key=lambda x: x["f1_score_dynamic"], reverse=True)

    def set_model(self, model_key):
        """Thay đổi mô hình đang sử dụng."""
        if model_key not in self.metadata["available_models"]:
            raise ValueError(f"Mô hình '{model_key}' không hợp lệ. Chọn một trong: {list(self.metadata['available_models'].keys())}")
        self.model_key = model_key
        self.model_info = self.metadata["available_models"][model_key]
        self._preload_required_models()
        return self.model_info["display_name"]
        
    def _get_model(self, model_filename):
        """Tải mô hình (sử dụng cache bộ nhớ để tăng tốc)."""
        if model_filename not in self.loaded_models:
            model_path = self.models_dir / model_filename
            if not model_path.exists():
                raise FileNotFoundError(f"Không tìm thấy tệp trọng số mô hình: {model_path}")
            self.loaded_models[model_filename] = joblib.load(model_path)
        return self.loaded_models[model_filename]

    def _preload_required_models(self):
        """Tải trước các tệp trọng số cần thiết cho mô hình hiện tại."""
        if "model_file" in self.model_info:
            self._get_model(self.model_info["model_file"])
        if "sub_models" in self.model_info:
            for sub_m in self.model_info["sub_models"]:
                self._get_model(sub_m)
        if "meta_model" in self.model_info:
            self._get_model(self.model_info["meta_model"])

    def _prepare_features(self, df_input, dataset_type="zone"):
        """
        Chuẩn hóa, sắp xếp đúng thứ tự cột và tự động bù đắp các đặc trưng thiếu 
        để đảm bảo không bao giờ bị crash khi người dùng nhập thiếu trường trên Web UI.
        """
        schema_info = self.metadata["feature_schemas"][dataset_type]
        expected_cols = schema_info["feature_names"]
        
        df = df_input.copy()
        
        # Nếu có timestamp, tự động tạo các biến chu kỳ sin/cos nếu người dùng chưa tính
        if "timestamp" in df.columns:
            ts = pd.to_datetime(df["timestamp"], errors="coerce")
            if "hour_sin" not in df.columns:
                df["hour_sin"] = np.sin(2 * np.pi * ts.dt.hour / 24.0)
                df["hour_cos"] = np.cos(2 * np.pi * ts.dt.hour / 24.0)
            if "weekday_sin" not in df.columns:
                df["weekday_sin"] = np.sin(2 * np.pi * ts.dt.weekday / 7.0)
                df["weekday_cos"] = np.cos(2 * np.pi * ts.dt.weekday / 7.0)
            if "month_sin" not in df.columns:
                df["month_sin"] = np.sin(2 * np.pi * ts.dt.month / 12.0)
                df["month_cos"] = np.cos(2 * np.pi * ts.dt.month / 12.0)
        
        # Nếu có hour/weekday/month số nguyên
        if "hour" in df.columns and "hour_sin" not in df.columns:
            df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
            df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
        if "weekday" in df.columns and "weekday_sin" not in df.columns:
            df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7.0)
            df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7.0)
        if "month" in df.columns and "month_sin" not in df.columns:
            df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
            df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)

        # Tự động ánh xạ primary_use sang cả dạng One-Hot (cho Zone) và dạng số nguyên primary_use_code (cho Baseline)
        cat_map = {
            'Education': 0, 'Office': 1, 'Parking': 2, 'Lodging/residential': 3,
            'Entertainment/public assembly': 4, 'Public services': 5,
            'Manufacturing/industrial': 6, 'Services': 7, 'Other': 8,
            'Healthcare': 9, 'Food sales and service': 10, 'Religious worship': 11,
            '__UNKNOWN__': 12
        }
        if "primary_use" in df.columns:
            if "primary_use_code" not in df.columns:
                df["primary_use_code"] = df["primary_use"].map(lambda x: cat_map.get(str(x), 0)).astype("int8")
            for cat_name in cat_map:
                col_name = f"primary_use_{cat_name}"
                if col_name not in df.columns:
                    df[col_name] = (df["primary_use"] == cat_name).astype(float)

        # Điền mặc định cho các cột thiếu
        for col in expected_cols:
            if col not in df.columns:
                if col.startswith("primary_use_") or col == "is_holiday" or col == "meter_reading_was_missing":
                    df[col] = 0.0
                elif col == "primary_use_code":
                    df[col] = 0
                elif col in ["site_sqft_mean", "use_sqft_mean", "zone_interaction"]:
                    df[col] = 0.0
                elif col.startswith("gte_"):
                    df[col] = 0.02 # Tỷ lệ dị thường trung bình nền
                elif col.endswith("_lag7") or col.endswith("_lag73"):
                    df[col] = df.get("air_temperature", 0.0)
                else:
                    df[col] = 0.0
                    
        return df

    def _predict_model_proba(self, model, X_prepared):
        """Dự báo xác suất tương thích hoàn hảo cho từng kiến trúc mô hình."""
        # 1. XGBoost
        if hasattr(model, "get_booster"):
            xgb_cols = model.get_booster().feature_names
            X_xgb = X_prepared[[c for c in xgb_cols if c in X_prepared.columns]].astype(float)
            return model.predict_proba(X_xgb)[:, 1]
            
        # 2. LightGBM
        if hasattr(model, "feature_name_"):
            lgb_cols = model.feature_name_
            X_renamed = X_prepared.rename(columns=lambda c: c.replace(' ', '_'))
            X_lgb = X_renamed[[c for c in lgb_cols if c in X_renamed.columns]].astype(float)
            return model.predict_proba(X_lgb)[:, 1]
            
        # 3. CatBoost
        if hasattr(model, "feature_names_"):
            cb_cols = model.feature_names_
            X_cb = X_prepared[[c for c in cb_cols if c in X_prepared.columns]].astype(float)
            return model.predict_proba(X_cb)[:, 1]
            
        return model.predict_proba(X_prepared)[:, 1]

    def predict_proba(self, data):
        """
        Tính xác suất dị thường liên tục p in [0, 1].
        Hỗ trợ đầu vào dạng: dict (1 bản ghi) hoặc DataFrame (nhiều bản ghi).
        """
        is_single = isinstance(data, dict)
        df_input = pd.DataFrame([data]) if is_single else pd.DataFrame(data)
        
        dataset_type = self.model_info.get("dataset_type", "zone")
        X = self._prepare_features(df_input, dataset_type)
        
        # 1. Trường hợp mô hình đơn
        if "model_file" in self.model_info:
            model = self._get_model(self.model_info["model_file"])
            probs = self._predict_model_proba(model, X)
            return float(probs[0]) if is_single else probs

        # 2. Trường hợp Context-Aware MoE
        if self.model_key == "context_aware_moe":
            xgb = self._get_model("XGBoost_Zone.pkl")
            lgb = self._get_model("LightGBM_Zone.pkl")
            cat = self._get_model("CatBoost_Zone.pkl")
            moe = self._get_model(self.model_info["meta_model"])
            
            p_xgb = self._predict_model_proba(xgb, X)
            p_lgb = self._predict_model_proba(lgb, X)
            p_cat = self._predict_model_proba(cat, X)
            
            # Chuẩn bị ma trận Meta-Features cho LightGBM Meta-Learner
            X_lgb = X.rename(columns=lambda c: c.replace(' ', '_'))[moe.feature_name_[:55]]
            meta_X = X_lgb.copy()
            meta_X["prob_xgb"] = p_xgb
            meta_X["prob_lgb"] = p_lgb
            meta_X["prob_cat"] = p_cat
            
            probs = moe.predict_proba(meta_X)[:, 1]
            return float(probs[0]) if is_single else probs

        # 3. Trường hợp Blending / Voting
        sub_probs = []
        model_names = []
        if "sub_models" in self.model_info:
            for sm_name in self.model_info["sub_models"]:
                sm = self._get_model(sm_name)
                sub_probs.append(self._predict_model_proba(sm, X))
                model_names.append(sm_name.split("_")[0].lower())
                
        weights = self.model_info.get("weights", {})
        if len(weights) > 0:
            final_prob = np.zeros(len(X))
            for m_name, prob_arr in zip(model_names, sub_probs):
                w = weights.get(m_name, 0.0)
                final_prob += w * prob_arr
            return float(final_prob[0]) if is_single else final_prob
        else:
            final_prob = np.mean(sub_probs, axis=0)
            return float(final_prob[0]) if is_single else final_prob

    def _determine_threshold(self, site_id, use_dynamic=True):
        """Xác định ngưỡng cắt (ngưỡng động 16 khu vực hoặc ngưỡng toàn cục)."""
        global_th = self.model_info.get("global_threshold", 0.3)
        if not use_dynamic or site_id is None:
            return global_th, "global"
            
        dyn_dict = self.model_info.get("dynamic_thresholds", {})
        site_key = f"site_{int(site_id)}"
        if site_key in dyn_dict:
            return float(dyn_dict[site_key]), f"dynamic ({site_key})"
        return global_th, "global (fallback)"

    def _determine_risk_level(self, prob):
        """Xác định cấp độ rủi ro (CRITICAL, HIGH, MEDIUM, NORMAL) cho giao diện UI."""
        levels = self.metadata["risk_levels"]
        if prob >= levels["CRITICAL"]["min_prob"]:
            return "CRITICAL", levels["CRITICAL"]
        elif prob >= levels["HIGH"]["min_prob"]:
            return "HIGH", levels["HIGH"]
        elif prob >= levels["MEDIUM"]["min_prob"]:
            return "MEDIUM", levels["MEDIUM"]
        else:
            return "NORMAL", levels["NORMAL"]

    def predict_single(self, input_dict, use_dynamic_threshold=True):
        """
        Dự báo cho 1 bản ghi duy nhất (dành cho Form nhập liệu trực tiếp trên Web UI).
        Trả về kết quả chi tiết chuẩn JSON sẵn sàng cho Frontend render.
        """
        prob = self.predict_proba(input_dict)
        site_id = input_dict.get("site_id", None)
        applied_th, th_type = self._determine_threshold(site_id, use_dynamic_threshold)
        
        is_anomaly = bool(prob >= applied_th)
        risk_code, risk_info = self._determine_risk_level(prob)
        
        # Diễn giải ngôn ngữ tự nhiên hỗ trợ người dùng cuối
        if is_anomaly:
            explanation = (f"CẢNH BÁO DỊ THƯỜNG: Xác suất bất thường đạt {prob*100:.2f}%, "
                           f"vượt ngưỡng kiểm soát {applied_th:.3f} [{th_type}]. "
                           f"Khuyến nghị kiểm tra ngay lập tức hệ thống đo lường tại Site {site_id}.")
        else:
            explanation = (f"VẬN HÀNH BÌNH THƯỜNG: Xác suất phụ tải bất thường chỉ {prob*100:.2f}%, "
                           f"nằm trong phạm vi an toàn dưới ngưỡng {applied_th:.3f} [{th_type}].")

        return {
            "is_anomaly": is_anomaly,
            "anomaly_label": 1 if is_anomaly else 0,
            "anomaly_probability": round(float(prob), 4),
            "applied_threshold": round(float(applied_th), 4),
            "threshold_type": th_type,
            "risk_level": risk_code,
            "risk_label": risk_info["label"],
            "risk_color": risk_info["color"],
            "model_used": self.model_info["display_name"],
            "model_f1_score": self.model_info.get("f1_score_dynamic", self.model_info.get("f1_score_global")),
            "explanation": explanation
        }

    def predict_batch(self, df_or_csv, use_dynamic_threshold=True):
        """
        Dự báo hàng loạt cho danh sách bản ghi hoặc tệp CSV người dùng tải lên.
        Trả về DataFrame kết quả kèm báo cáo tóm tắt để vẽ Dashboard.
        """
        if isinstance(df_or_csv, (str, Path)):
            df = pd.read_csv(df_or_csv)
        else:
            df = df_or_csv.copy()
            
        probs = self.predict_proba(df)
        df["anomaly_prob"] = np.round(probs, 4)
        
        site_ids = df["site_id"].values if "site_id" in df.columns else [None] * len(df)
        
        thresholds = []
        is_anomalies = []
        risk_levels = []
        
        for p, s in zip(probs, site_ids):
            th, _ = self._determine_threshold(s, use_dynamic_threshold)
            thresholds.append(round(float(th), 4))
            anom = int(p >= th)
            is_anomalies.append(anom)
            r_code, _ = self._determine_risk_level(p)
            risk_levels.append(r_code)
            
        df["applied_threshold"] = thresholds
        df["is_anomaly"] = is_anomalies
        df["risk_level"] = risk_levels
        
        total_samples = len(df)
        anomaly_count = int(sum(is_anomalies))
        normal_count = total_samples - anomaly_count
        anomaly_rate = round((anomaly_count / total_samples) * 100.0, 2) if total_samples > 0 else 0.0
        
        summary = {
            "total_samples": total_samples,
            "anomaly_count": anomaly_count,
            "normal_count": normal_count,
            "anomaly_percentage": anomaly_rate,
            "model_used": self.model_info["display_name"],
            "critical_cases": int((df["risk_level"] == "CRITICAL").sum()),
            "high_cases": int((df["risk_level"] == "HIGH").sum()),
            "medium_cases": int((df["risk_level"] == "MEDIUM").sum())
        }
        
        return df, summary
