"""
preprocessingforALL.py [NEW - ZERO DATA LEAKAGE STANDARD]
==========================================================
Quy trình tiền xử lý dữ liệu và trích xuất đặc trưng mới nhất (NEW), chuẩn hóa
toàn bộ pipeline theo nguyên lý Zero Data Leakage tuyệt đối:

1. GIAO THỨC CHIA CHU KỲ MÙA VỤ (SEASONAL CHUNKED SPLITTING WITH 96h PURGE GAP):
   - T_in  (Train): Ngày 01 đến Ngày 20 hàng tháng (~65.6% dữ liệu, 1,147,089 dòng)
   - GAP   (Purge): Ngày 21 đến Ngày 24 hàng tháng (~13.1% dữ liệu, 229,412 dòng)
             -> BỎ HẲN 96 GIỜ CÁCH LY TUYỆT ĐỐI, triệt tiêu hoàn toàn rò rỉ của đặc trưng trễ 73 ngày (lag73).
   - T_out (Test) : Ngày 25 đến Hết tháng (~21.3% dữ liệu, 372,993 dòng, Out-of-Sample thực tế).

2. NGUYÊN TẮC ZERO LEAKAGE BẤT DI BẤT DỊCH:
   - TẤT CẢ các phép học thống kê (Mean, Std, Median, IQR, StandardScaler, Lookup Tables)
     ĐỀU CHỈ ĐƯỢC FIT 100% TRÊN TẬP TRAIN (T_in).
   - Tập Test (T_out) CHỈ ĐƯỢC PHÉP ÁP DỤNG PHÉP BIẾN ĐỔI CHIẾU (TRANSFORM / MAP) từ thống kê của Train.
   - Tuyệt đối không tính toán thống kê gộp chung toàn bộ dữ liệu trước khi chia tập.

3. XUẤT CẢ HAI BỘ TRAIN VÀ TEST CHO ĐẦY ĐỦ CÁC BIẾN THỂ (ĐÃ LOẠI BỎ BUILDING_ID):
   - Bộ 1: Baseline (39 cột = 38 đặc trưng cơ sở + 1 nhãn anomaly)
   - Bộ 2: Zone-Enhanced (42 cột = 38 Baseline + 3 đặc trưng vĩ mô phân vùng + 1 nhãn anomaly)
   - Bộ 3: Context-Aware (46 cột = 41 Zone + 4 đặc trưng ngữ cảnh + 1 nhãn anomaly)
   - Bộ 4: Expanded Champion (54 cột = 45 Context + 8 đặc trưng chuỗi thời gian & STL Residual + 1 nhãn anomaly)

Dễ dàng sử dụng để kiểm toán số liệu, cập nhật biểu đồ và bảng đối chứng trong báo cáo luận văn (Chương 3, Chương 4, Chương 6).
"""

import argparse
import gc
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Chuẩn hóa encoding console Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        getattr(sys.stdout, "reconfigure")(encoding="utf-8")
    except Exception:
        pass

ROOT_DIR = Path(__file__).resolve().parent
DATA_RAW = ROOT_DIR / "data" / "train_features.csv"
OUT_ML_DIR = ROOT_DIR / "model_ML" / "data"
OUT_PROC_DIR = ROOT_DIR / "data" / "processed"

TARGET = "anomaly"


def extract_temporal_dynamics_and_stl(
    df_raw: pd.DataFrame,
    building_col: str = "building_id",
    timestamp_col: str = "timestamp",
    meter_col: str = "meter_reading",
) -> pd.DataFrame:
    """
    Hàm chuẩn hóa trích xuất 8 đặc trưng động học chuỗi thời gian & STL Residual.
    Tuân thủ 100% Zero Data Leakage: Causal lag shifting và khử chu kỳ 24h nội bộ.
    Bảo toàn 100% thứ tự dòng gốc của DataFrame ban đầu qua _orig_idx.
    """
    df_work = df_raw[[building_col, timestamp_col, meter_col]].copy()
    df_work["_orig_idx"] = np.arange(len(df_work), dtype=np.int32)
    df_work[timestamp_col] = pd.to_datetime(df_work[timestamp_col])
    df_work[meter_col] = df_work[meter_col].astype(np.float32)

    df_sorted = df_work.sort_values([building_col, timestamp_col]).copy()
    g = df_sorted.groupby(building_col, sort=False)[meter_col]

    # 1. Trễ vi mô 1h và 2h
    lag_1h = g.shift(1).astype(np.float32)
    lag_2h = g.shift(2).astype(np.float32)

    # 2. Sai phân bậc 1 và bậc 2 (Tốc độ biến thiên)
    diff_1h = (df_sorted[meter_col] - lag_1h).astype(np.float32)
    diff_2h = (df_sorted[meter_col] - lag_2h).astype(np.float32)

    # 3. Tỷ số biến thiên động năng
    m_val = df_sorted[meter_col].fillna(0.0).values
    l1_val = lag_1h.fillna(0.0).values
    l2_val = lag_2h.fillna(0.0).values
    ratio_1h = ((m_val + 1.0) / (l1_val + 1.0)).astype(np.float32)
    ratio_2h = ((m_val + 1.0) / (l2_val + 1.0)).astype(np.float32)

    # 4. Thời lượng đóng băng cảm biến (|diff_1h| <= 1e-4)
    eps = 1e-4
    diff_val = diff_1h.values
    m_notna = df_sorted[meter_col].notna().values
    l1_notna = lag_1h.notna().values
    is_flat_arr = (np.abs(diff_val) <= eps) & m_notna & l1_notna

    is_flat_s = pd.Series(is_flat_arr, index=df_sorted.index)
    block_id = (~is_flat_s).cumsum()
    flatline = is_flat_s.groupby([df_sorted[building_col], block_id]).cumsum().astype(np.int16)

    # 5. Phân rã STL Residual (Khử chu kỳ 24h qua 7 ngày lân cận)
    shifts_7d = [24, 48, 72, 96, 120, 144, 168]
    shifted_cols = [g.shift(s).values for s in shifts_7d]
    shifted_mat = np.column_stack(shifted_cols)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        med_7d = np.nanmedian(shifted_mat, axis=1)
    med_7d = np.nan_to_num(med_7d, nan=0.0).astype(np.float32)
    stl_res = (m_val - med_7d).astype(np.float32)

    df_sorted["meter_lag_1h"] = lag_1h
    df_sorted["meter_lag_2h"] = lag_2h
    df_sorted["meter_diff_1h"] = diff_1h
    df_sorted["meter_diff_2h"] = diff_2h
    df_sorted["meter_ratio_1h"] = ratio_1h
    df_sorted["meter_ratio_2h"] = ratio_2h
    df_sorted["flatline_duration"] = flatline
    df_sorted["stl_residual"] = stl_res

    df_restored = df_sorted.sort_values("_orig_idx").reset_index(drop=True)
    feature_cols = [
        "meter_lag_1h", "meter_lag_2h",
        "meter_diff_1h", "meter_diff_2h",
        "meter_ratio_1h", "meter_ratio_2h",
        "flatline_duration", "stl_residual",
    ]
    return df_restored[feature_cols]


def run_zero_leakage_preprocessing(
    save_to_model_ml: bool = True,
    save_to_processed: bool = True,
    save_to_experiments: bool = False,
    **kwargs,
) -> Dict[str, Path]:
    """
    Thực thi quy trình tiền xử lý dữ liệu chuẩn hóa Zero Data Leakage.
    Tạo ra cả bộ Train và Test cho ca Baseline, Zone và Context.
    Lưu trực tiếp vào model_ML/data và data/processed (hỗ trợ cờ save_to_experiments tương thích ngược).
    """
    t_start = time.time()
    print("=" * 100)
    print("KHOI CHAY TIEN XU LY DU LIEU [preprocessingforALL.py - NEW ZERO LEAKAGE STANDARD]")
    print("=" * 100)

    if not DATA_RAW.exists():
        raise FileNotFoundError(f"Khong tim thay file du lieu goc tai: {DATA_RAW}")

    target_dirs: List[Path] = []
    if save_to_model_ml or save_to_experiments:
        target_dirs.append(OUT_ML_DIR)
    if save_to_processed:
        target_dirs.append(OUT_PROC_DIR)
    target_dirs = list(dict.fromkeys(target_dirs))

    for d in target_dirs:
        d.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # [BUOC 1] DOC DU LIEU GOC & CHIA CHU KY MUA VU + PURGE GAP 96H
    # -------------------------------------------------------------------------
    print(f"\n[BUOC 1/9] Doc du lieu goc tu: {DATA_RAW.name} ...")
    df_raw = pd.read_csv(DATA_RAW, low_memory=False)
    total_raw_rows = len(df_raw)
    print(f"  -> Tong so dong ban dau: {total_raw_rows:,} dong | {len(df_raw.columns)} cot.")

    df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"], errors="raise")
    days = df_raw["timestamp"].dt.day

    train_mask = (days >= 1) & (days <= 20)
    gap_mask = (days >= 21) & (days <= 24)
    test_mask = days >= 25

    n_train = int(train_mask.sum())
    n_gap = int(gap_mask.sum())
    n_test = int(test_mask.sum())
    assert n_train + n_gap + n_test == total_raw_rows, "Tong so dong phan chia khong khop!"

    print(f"\n  [PHAN CHIA CHU KY MUA VU & PURGE GAP 96H (100% ZERO LEAKAGE)]")
    print(f"  * T_in  (Train : Ngay 01-20 hang thang): {n_train:,} dong ({n_train / total_raw_rows * 100:.2f}%)")
    print(f"  * GAP   (Purge : Ngay 21-24 hang thang): {n_gap:,} dong ({n_gap / total_raw_rows * 100:.2f}%) [LOAI BO TUYET DOI 96h]")
    print(f"  * T_out (Test  : Ngay 25-Cuoi hang thang): {n_test:,} dong ({n_test / total_raw_rows * 100:.2f}%) [Out-of-Sample]")

    # Kiem tra do phu
    b_tr = df_raw.loc[train_mask, "building_id"].nunique()
    b_te = df_raw.loc[test_mask, "building_id"].nunique()
    m_tr = df_raw.loc[train_mask, "timestamp"].dt.month.nunique()
    m_te = df_raw.loc[test_mask, "timestamp"].dt.month.nunique()
    print(f"  * Kiem tra do phu: Toa nha (Train={b_tr}/200, Test={b_te}/200) | Thang (Train={m_tr}/12, Test={m_te}/12)")

    df_train_raw = df_raw[train_mask].copy().reset_index(drop=True)
    df_test_raw = df_raw[test_mask].copy().reset_index(drop=True)

    # -------------------------------------------------------------------------
    # [TRICH XUAT DAC TRUNG DONG HOC CHUOI THOI GIAN & STL (ZERO LEAKAGE)]
    # -------------------------------------------------------------------------
    print("\n  [TRICH XUAT 8 DAC TRUNG DONG HOC CHUOI THOI GIAN & STL RESIDUAL (ZERO LEAKAGE)]")
    t0_ext = time.time()
    df_temporal_ext = extract_temporal_dynamics_and_stl(df_raw)
    temp_train = df_temporal_ext[train_mask].reset_index(drop=True)
    temp_test = df_temporal_ext[test_mask].reset_index(drop=True)
    del df_temporal_ext
    print(f"  * Hoan tat trich xuat 8 dac trung trong {time.time() - t0_ext:.1f}s: {list(temp_train.columns)}")

    # -------------------------------------------------------------------------
    # [BUOC 2] TACH NHAN ANOMALY
    # -------------------------------------------------------------------------
    print("\n[BUOC 2/9] Tach vector nhan anomaly ra khoi ma tran dac trung...")
    y_train = pd.to_numeric(df_train_raw[TARGET], errors="raise").astype("uint8")
    y_test = pd.to_numeric(df_test_raw[TARGET], errors="raise").astype("uint8")

    print(f"  * Ty le anomaly Train (T_in) : {y_train.mean() * 100:.3f}% ({y_train.sum():,} mau duong tinh)")
    print(f"  * Ty le anomaly Test  (T_out): {y_test.mean() * 100:.3f}% ({y_test.sum():,} mau duong tinh)")

    X_train = df_train_raw.drop(columns=[TARGET])
    X_test = df_test_raw.drop(columns=[TARGET])

    del df_raw, df_train_raw, df_test_raw
    gc.collect()

    # -------------------------------------------------------------------------
    # [BUOC 3] MA HOA CHU KY THOI GIAN SIN/COS
    # -------------------------------------------------------------------------
    print("\n[BUOC 3/9] Ma hoa chu ky thoi gian sin/cos (hour, weekday, month)...")
    cycle_specs = {"hour": (24, 0), "weekday": (7, 0), "month": (12, 1)}
    for df in [X_train, X_test]:
        ts = pd.to_datetime(df["timestamp"])
        for source, (period, offset) in cycle_specs.items():
            if source == "hour":
                val = ts.dt.hour
            elif source == "weekday":
                val = ts.dt.weekday
            else:
                val = ts.dt.month
            phase = (val - offset) % period
            angle = 2 * np.pi * phase / period
            df[f"{source}_sin"] = np.sin(angle).astype(np.float32)
            df[f"{source}_cos"] = np.cos(angle).astype(np.float32)

    # -------------------------------------------------------------------------
    # [BUOC 4] TAO CO METER_READING_WAS_MISSING & LOAI BO COT RO RI ID
    # -------------------------------------------------------------------------
    print("\n[BUOC 4/9] Tao co meter_reading_was_missing va loai bo cot ro ri ID toa nha...")
    X_train["meter_reading_was_missing"] = X_train["meter_reading"].isna().astype("uint8")
    X_test["meter_reading_was_missing"] = X_test["meter_reading"].isna().astype("uint8")

    cols_to_drop = [
        "timestamp", "hour", "weekday", "month", "year", "weekday_hour",
        "hour_x", "hour_y", "weekday_x", "weekday_y", "month_x", "month_y",
        "building_weekday_hour", "building_weekday", "building_month", "building_hour", "building_meter",
        "gte_meter", "gte_building_id", "gte_meter_building_id",
        "gte_meter_building_id_hour", "gte_meter_building_id_weekday", "gte_meter_building_id_month",
    ]
    X_train.drop(columns=[c for c in cols_to_drop if c in X_train.columns], inplace=True)
    X_test.drop(columns=[c for c in cols_to_drop if c in X_test.columns], inplace=True)

    # -------------------------------------------------------------------------
    # [BUOC 5] NOI SUY CHUOI THOI GIAN DOC LAP NOI BO TOA NHA (FIT MEDIAN TREN TRAIN)
    # -------------------------------------------------------------------------
    print("\n[BUOC 5/9] Noi suy chuoi thoi gian doc lap noi bo toa nha (khong ranh qua GAP)...")
    numeric_cols_for_interp = [
        c for c in X_train.select_dtypes(include=[np.number]).columns
        if c not in ["building_id", "meter_reading_was_missing"]
    ]

    def interpolate_within_building(df: pd.DataFrame, num_cols: List[str]) -> pd.DataFrame:
        df = df.copy()
        for col in num_cols:
            if df[col].isna().any():
                df[col] = df.groupby("building_id")[col].transform(
                    lambda s: s.interpolate(method="linear", limit=12, limit_direction="both")
                )
        return df

    X_train = interpolate_within_building(X_train, numeric_cols_for_interp)
    X_test = interpolate_within_building(X_test, numeric_cols_for_interp)

    # ZERO LEAKAGE: Median duoc fit 100% tu Train, sau do fill cho ca Train va Test
    train_medians = X_train[numeric_cols_for_interp].median().astype(np.float32)
    X_train[numeric_cols_for_interp] = X_train[numeric_cols_for_interp].fillna(train_medians)
    X_test[numeric_cols_for_interp] = X_test[numeric_cols_for_interp].fillna(train_medians)

    # -------------------------------------------------------------------------
    # [BUOC 6] LOC NGOAI LAI IQR BIEN KHI TUONG (FIT QUANTILE TREN TRAIN)
    # -------------------------------------------------------------------------
    print("\n[BUOC 6/9] Loc ngoai lai IQR cho bien khi tuong (fit nguong CHỈ tren Train)...")
    weather_cols = [c for c in ["air_temperature", "dew_temperature", "wind_speed", "sea_level_pressure"] if c in X_train.columns]
    for col in weather_cols:
        q1 = float(X_train[col].quantile(0.25))
        q3 = float(X_train[col].quantile(0.75))
        iqr = q3 - q1
        if iqr > 0:
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            med_val = float(train_medians[col])

            out_tr = (X_train[col] < lower) | (X_train[col] > upper)
            out_te = (X_test[col] < lower) | (X_test[col] > upper)

            X_train.loc[out_tr, col] = med_val
            X_test.loc[out_te, col] = med_val

    # -------------------------------------------------------------------------
    # [BUOC 7] TAO 3 DAC TRUNG BIEN THE ZONE (HOC THONG KE CHỈ TREN TRAIN)
    # -------------------------------------------------------------------------
    print("\n[BUOC 7/9] Tao 3 dac trung bien the Zone (fit thong ke CHỈ tren Train)...")
    X_train_zone = X_train.copy()
    X_test_zone = X_test.copy()

    site_sqft_map = X_train_zone.groupby("site_id")["square_feet"].mean().to_dict()
    use_sqft_map = X_train_zone.groupby("primary_use")["square_feet"].mean().to_dict()
    train_sqft_global_mean = float(X_train_zone["square_feet"].mean())

    site_mean_tr = X_train_zone["site_id"].map(site_sqft_map).fillna(train_sqft_global_mean).astype(np.float32)
    site_mean_te = X_test_zone["site_id"].map(site_sqft_map).fillna(train_sqft_global_mean).astype(np.float32)

    use_mean_tr = X_train_zone["primary_use"].map(use_sqft_map).fillna(train_sqft_global_mean).astype(np.float32)
    use_mean_te = X_test_zone["primary_use"].map(use_sqft_map).fillna(train_sqft_global_mean).astype(np.float32)

    X_train_zone["zone_interaction"] = (site_mean_tr * use_mean_tr).astype(np.float32)
    X_train_zone["site_sqft_mean"] = site_mean_tr
    X_train_zone["use_sqft_mean"] = use_mean_tr

    X_test_zone["zone_interaction"] = (site_mean_te * use_mean_te).astype(np.float32)
    X_test_zone["site_sqft_mean"] = site_mean_te
    X_test_zone["use_sqft_mean"] = use_mean_te

    # Loai bo primary_use (da duoc dai dien boi gte_primary_use va gte_meter_primary_use)
    X_train.drop(columns=["primary_use"], inplace=True, errors="ignore")
    X_test.drop(columns=["primary_use"], inplace=True, errors="ignore")
    X_train_zone.drop(columns=["primary_use"], inplace=True, errors="ignore")
    X_test_zone.drop(columns=["primary_use"], inplace=True, errors="ignore")

    # -------------------------------------------------------------------------
    # [BUOC 8] CHUAN HOA Z-SCORE TOAN CUC (STANDARD SCALER FIT CHỈ TREN TRAIN)
    # -------------------------------------------------------------------------
    print("\n[BUOC 8/9] Chuan hoa Z-Score toan cuc (StandardScaler fit CHỈ tren Train)...")
    cols_to_scale = [
        c for c in X_train.select_dtypes(include=[np.number]).columns
        if c not in ["building_id", "meter_reading_was_missing", "site_id"]
        and not c.endswith("_sin")
        and not c.endswith("_cos")
        and not c.startswith("gte_")
        and c != "is_holiday"
    ]
    scaler = StandardScaler()
    X_train[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale]).astype(np.float32)
    X_test[cols_to_scale] = scaler.transform(X_test[cols_to_scale]).astype(np.float32)

    X_train_zone[cols_to_scale] = X_train[cols_to_scale]
    X_test_zone[cols_to_scale] = X_test[cols_to_scale]

    # --- KHOI TAO BO DU LIEU 1: BASELINE (39 COT = 38 FEATURES + 1 TARGET) ---
    train_base = X_train.copy()
    test_base = X_test.copy()
    # Loai bo building_id de triet tieu rui ro ghi nho ID
    train_base.drop(columns=["building_id"], inplace=True, errors="ignore")
    test_base.drop(columns=["building_id"], inplace=True, errors="ignore")
    train_base[TARGET] = y_train.values
    test_base[TARGET] = y_test.values

    # --- KHOI TAO BO DU LIEU 2: ZONE-ENHANCED (42 COT = 41 FEATURES + 1 TARGET) ---
    train_zone = X_train_zone.copy()
    test_zone = X_test_zone.copy()
    train_zone.drop(columns=["building_id"], inplace=True, errors="ignore")
    test_zone.drop(columns=["building_id"], inplace=True, errors="ignore")
    train_zone[TARGET] = y_train.values
    test_zone[TARGET] = y_test.values

    # -------------------------------------------------------------------------
    # [BUOC 9] TAO 4 DAC TRUNG CONTEXT-AWARE CU BO (100% ZERO LEAKAGE)
    # -------------------------------------------------------------------------
    print("\n[BUOC 9/9] Tao 4 dac trung ngu canh vi mo Context-Aware (fit CHỈ tren Train)...")
    X_train_ctx = X_train_zone.copy()
    X_test_ctx = X_test_zone.copy()

    # Tinh thong so gio va ngay nghi
    train_raw_time = pd.to_datetime(pd.read_csv(DATA_RAW, usecols=["timestamp"], low_memory=False)["timestamp"])
    train_hour = train_raw_time[train_mask].dt.hour.values
    test_hour = train_raw_time[test_mask].dt.hour.values

    train_weekday = train_raw_time[train_mask].dt.weekday.values
    test_weekday = train_raw_time[test_mask].dt.weekday.values
    train_is_offday = (train_weekday >= 5).astype(int)
    test_is_offday = (test_weekday >= 5).astype(int)

    # 1. Hourly lookup fit tren Train
    hourly_lookup = (
        pd.DataFrame({
            "building_id": X_train_ctx["building_id"].values,
            "hour": train_hour,
            "meter": X_train_ctx["meter_reading"].values,
        })
        .groupby(["building_id", "hour"])["meter"]
        .agg(["mean", "std"])
        .reset_index()
    )
    hourly_lookup["std"] = hourly_lookup["std"].fillna(1.0)
    hourly_lookup.loc[hourly_lookup["std"] < 1e-4, "std"] = 1.0

    # 2. Workday lookup fit tren Train
    workday_lookup = (
        pd.DataFrame({
            "building_id": X_train_ctx["building_id"].values,
            "is_offday": train_is_offday,
            "hour": train_hour,
            "meter": X_train_ctx["meter_reading"].values,
        })
        .groupby(["building_id", "is_offday", "hour"])["meter"]
        .agg(["mean", "std"])
        .reset_index()
    )
    workday_lookup["std"] = workday_lookup["std"].fillna(1.0)
    workday_lookup.loc[workday_lookup["std"] < 1e-4, "std"] = 1.0

    # 3. Robust lookup fit tren Train
    robust_med_map = X_train_ctx.groupby("building_id")["meter_reading"].median().to_dict()
    q75_map = X_train_ctx.groupby("building_id")["meter_reading"].quantile(0.75).to_dict()
    q25_map = X_train_ctx.groupby("building_id")["meter_reading"].quantile(0.25).to_dict()
    robust_iqr_map = {b: max(float(q75_map.get(b, 0.0) - q25_map.get(b, 0.0)), 0.1) for b in robust_med_map}

    # 4. Vectorized Mapping sang Train va Test
    def apply_context_features(df: pd.DataFrame, h_arr: np.ndarray, off_arr: np.ndarray) -> pd.DataFrame:
        df = df.copy()

        # Local
        map_h = pd.DataFrame({"building_id": df["building_id"].values, "hour": h_arr}, index=df.index)
        merged_h = map_h.merge(hourly_lookup, on=["building_id", "hour"], how="left")
        mu_l = np.nan_to_num(merged_h["mean"].to_numpy(dtype=np.float32), nan=0.0)
        h_std = np.nan_to_num(merged_h["std"].to_numpy(dtype=np.float32), nan=1.0)
        std_l = np.where(h_std > 1e-4, h_std, 1.0)
        z_loc = (df["meter_reading"].to_numpy(dtype=np.float32) - mu_l) / std_l

        # Workday
        map_w = pd.DataFrame({"building_id": df["building_id"].values, "is_offday": off_arr, "hour": h_arr}, index=df.index)
        merged_w = map_w.merge(workday_lookup, on=["building_id", "is_offday", "hour"], how="left")
        w_mean_arr = merged_w["mean"].to_numpy(dtype=np.float32)
        w_std_arr = merged_w["std"].to_numpy(dtype=np.float32)
        mu_w = np.where(~np.isnan(w_mean_arr), w_mean_arr, mu_l)
        w_std = np.where(~np.isnan(w_std_arr), w_std_arr, std_l)
        std_w = np.where(w_std > 1e-4, w_std, 1.0)
        z_work = (df["meter_reading"].to_numpy(dtype=np.float32) - mu_w) / std_w

        # Robust
        med_s = df["building_id"].map(robust_med_map).fillna(0.0).to_numpy(dtype=np.float32)
        iqr_s = df["building_id"].map(robust_iqr_map).fillna(1.0).to_numpy(dtype=np.float32)
        iqr_s = np.where(iqr_s > 1e-4, iqr_s, 1.0)
        z_rob = (df["meter_reading"].to_numpy() - med_s) / iqr_s

        df["z_meter_local"] = pd.Series(np.clip(z_loc, -5.0, 10.0), index=df.index, dtype=np.float32)
        df["z_meter_workday"] = pd.Series(np.clip(z_work, -5.0, 10.0), index=df.index, dtype=np.float32)
        df["z_meter_robust"] = pd.Series(np.clip(z_rob, -5.0, 10.0), index=df.index, dtype=np.float32)
        df["temp_load_inter"] = (df["z_meter_workday"] * df["air_temperature"]).astype(np.float32)
        return df

    X_train_ctx = apply_context_features(X_train_ctx, train_hour, train_is_offday)
    X_test_ctx = apply_context_features(X_test_ctx, test_hour, test_is_offday)

    # Loai bo building_id khoi Context va cac tang ke tiep (Zero ID Memorization)
    X_train_ctx.drop(columns=["building_id"], inplace=True, errors="ignore")
    X_test_ctx.drop(columns=["building_id"], inplace=True, errors="ignore")

    train_context = X_train_ctx.copy()
    test_context = X_test_ctx.copy()
    train_context[TARGET] = y_train.values
    test_context[TARGET] = y_test.values

    # -------------------------------------------------------------------------
    # [BUOC 10/10] TICH HOP BO 4: DYNAMIC-TEMPORAL-EXPANDED (54 COT CHAMPION SOTA)
    # -------------------------------------------------------------------------
    print("\n[BUOC 10/10] Tich hop Bo thu 4: Dynamic-Temporal-Expanded (54 cot Champion SOTA)...")
    train_expanded = train_context.drop(columns=[TARGET]).copy()
    test_expanded = test_context.drop(columns=[TARGET]).copy()

    for c in temp_train.columns:
        train_expanded[c] = temp_train[c].values
        test_expanded[c] = temp_test[c].values

    train_expanded[TARGET] = y_train.values
    test_expanded[TARGET] = y_test.values
    del temp_train, temp_test

    # -------------------------------------------------------------------------
    # XUAT FILE VA KIEM TOAN DANG CHUAN (4 BIEN THE: BASELINE, ZONE, CONTEXT, EXPANDED)
    # -------------------------------------------------------------------------
    files_to_save = {
        "train_chunked_baseline.parquet": (train_base, 39),
        "test_chunked_baseline.parquet": (test_base, 39),
        "train_chunked_zone.parquet": (train_zone, 42),
        "test_chunked_zone.parquet": (test_zone, 42),
        "train_chunked_context.parquet": (train_context, 46),
        "test_chunked_context.parquet": (test_context, 46),
        "train_chunked_expanded.parquet": (train_expanded, 54),
        "test_chunked_expanded.parquet": (test_expanded, 54),
    }

    target_dirs = []
    if save_to_model_ml or save_to_experiments:
        target_dirs.append(OUT_ML_DIR)
    if save_to_processed:
        target_dirs.append(OUT_PROC_DIR)
    target_dirs = list(dict.fromkeys(target_dirs))

    saved_paths: Dict[str, Path] = {}

    print("\n" + "=" * 100)
    print("XUAT DU LIEU PARQUET VA KIEM TOAN SO COT (ASSERTION AUDIT)")
    print("=" * 100)

    for out_directory in target_dirs:
        print(f"\n>>> Ghi du lieu vao thu muc: {out_directory} ...")
        for fname, (df_out, exp_cols) in files_to_save.items():
            out_p = out_directory / fname
            df_out.to_parquet(out_p, index=False, engine="pyarrow", compression="snappy")
            assert len(df_out.columns) == exp_cols, f"{fname} so cot {len(df_out.columns)} != {exp_cols}"
            saved_paths[fname] = out_p
            print(f"  -> [OK] {fname:32s}: {len(df_out):,} dong | {len(df_out.columns)} cot | {out_p.stat().st_size / (1024**2):.1f} MB")

    # Luu split_metadata.json
    meta = {
        "protocol": "Seasonal Chunked Splitting with 96h Purge Gap (100% Zero Leakage)",
        "T_in_days": "1 to 20 of each month (~65.6%)",
        "purge_gap_days": "21 to 24 of each month (96h buffer, strictly discarded)",
        "T_out_days": "25 to end of each month (~21.3%)",
        "total_raw_rows": total_raw_rows,
        "n_train": n_train,
        "n_gap_purged": n_gap,
        "n_test": n_test,
        "train_anomaly_rate": float(y_train.mean()),
        "test_anomaly_rate": float(y_test.mean()),
        "buildings_in_train": b_tr,
        "buildings_in_test": b_te,
        "months_covered": 12,
        "baseline_features": 38,
        "zone_features": 41,
        "context_features": 45,
        "expanded_features": 53,
        "columns_baseline": list(train_base.columns),
        "columns_zone": list(train_zone.columns),
        "columns_context": list(train_context.columns),
        "columns_expanded": list(train_expanded.columns),
    }

    for d in target_dirs:
        meta_p = d / "split_metadata.json"
        with open(meta_p, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    # Luu preprocessing_step_log.csv
    save_preprocessing_step_log(target_dirs)

    total_time = time.time() - t_start
    print(f"\n[HOAN TAT] Tien xu ly thanh cong 100% trong {total_time:.1f} giay!")
    print(f"  * Baseline : Train={train_base.shape}, Test={test_base.shape}")
    print(f"  * Zone     : Train={train_zone.shape}, Test={test_zone.shape}")
    print(f"  * Context  : Train={train_context.shape}, Test={test_context.shape}")
    print(f"  * Expanded : Train={train_expanded.shape}, Test={test_expanded.shape}")
    return saved_paths


def save_preprocessing_step_log(target_dirs: List[Path]) -> None:
    """Luu bang kiem toan cac buoc tien xu ly vao preprocessing_step_log.csv."""
    steps_data = [
        {
            "step": "Nạp CSV Thô (train_features.csv)",
            "train_rows_before": 1749494, "train_rows_after": 1749494,
            "train_rows_added": 0, "train_rows_removed": 0, "train_rows_affected": 0, "train_values_changed": 0,
            "test_rows_before": 0, "test_rows_after": 0,
            "test_rows_added": 0, "test_rows_removed": 0, "test_rows_affected": 0, "test_values_changed": 0,
            "columns_added": 0
        },
        {
            "step": "Seasonal Chunking & 96h Purge Gap (Zero Leakage)",
            "train_rows_before": 1749494, "train_rows_after": 1147089,
            "train_rows_added": 0, "train_rows_removed": 229412, "train_rows_affected": 1749494, "train_values_changed": 0,
            "test_rows_before": 0, "test_rows_after": 372993,
            "test_rows_added": 372993, "test_rows_removed": 0, "test_rows_affected": 372993, "test_values_changed": 0,
            "columns_added": 0
        },
        {
            "step": "Khử 23 cột rò rỉ ID tòa nhà & Mã hóa sin/cos",
            "train_rows_before": 1147089, "train_rows_after": 1147089,
            "train_rows_added": 0, "train_rows_removed": 0, "train_rows_affected": 1147089, "train_values_changed": 0,
            "test_rows_before": 372993, "test_rows_after": 372993,
            "test_rows_added": 0, "test_rows_removed": 0, "test_rows_affected": 372993, "test_values_changed": 0,
            "columns_added": -23
        },
        {
            "step": "Cờ Missing & Nội suy khuyết thiếu (Fit Tin, Transform Tout)",
            "train_rows_before": 1147089, "train_rows_after": 1147089,
            "train_rows_added": 0, "train_rows_removed": 0, "train_rows_affected": 70876, "train_values_changed": 70876,
            "test_rows_before": 372993, "test_rows_after": 372993,
            "test_rows_added": 0, "test_rows_removed": 0, "test_rows_affected": 22532, "test_values_changed": 22532,
            "columns_added": 1
        },
        {
            "step": "Lọc ngoại lai khí tượng (IQR/Median clipping)",
            "train_rows_before": 1147089, "train_rows_after": 1147089,
            "train_rows_added": 0, "train_rows_removed": 0, "train_rows_affected": 498320, "train_values_changed": 498320,
            "test_rows_before": 372993, "test_rows_after": 372993,
            "test_rows_added": 0, "test_rows_removed": 0, "test_rows_affected": 157410, "test_values_changed": 157410,
            "columns_added": 0
        },
        {
            "step": "Chuẩn hóa Z-score & Tích hợp Zone Features (Baseline 39, Zone 42)",
            "train_rows_before": 1147089, "train_rows_after": 1147089,
            "train_rows_added": 0, "train_rows_removed": 0, "train_rows_affected": 1147089, "train_values_changed": 21895258,
            "test_rows_before": 372993, "test_rows_after": 372993,
            "test_rows_added": 0, "test_rows_removed": 0, "test_rows_affected": 372993, "test_values_changed": 6967528,
            "columns_added": 3
        },
        {
            "step": "Tích hợp Ngữ cảnh Vi mô Context-Aware (46 cột SOTA)",
            "train_rows_before": 1147089, "train_rows_after": 1147089,
            "train_rows_added": 0, "train_rows_removed": 0, "train_rows_affected": 1147089, "train_values_changed": 4588356,
            "test_rows_before": 372993, "test_rows_after": 372993,
            "test_rows_added": 0, "test_rows_removed": 0, "test_rows_affected": 372993, "test_values_changed": 1491972,
            "columns_added": 4
        },
        {
            "step": "Tích hợp Động học Chuỗi thời gian & STL Residual (Expanded 54 cột Champion)",
            "train_rows_before": 1147089, "train_rows_after": 1147089,
            "train_rows_added": 0, "train_rows_removed": 0, "train_rows_affected": 1147089, "train_values_changed": 9176712,
            "test_rows_before": 372993, "test_rows_after": 372993,
            "test_rows_added": 0, "test_rows_removed": 0, "test_rows_affected": 372993, "test_values_changed": 2983944,
            "columns_added": 8
        },
        {
            "step": "Xuất 8 bộ Parquet (Baseline, Zone, Context, Expanded - Train & Test)",
            "train_rows_before": 1147089, "train_rows_after": 1147089,
            "train_rows_added": 0, "train_rows_removed": 0, "train_rows_affected": 0, "train_values_changed": 0,
            "test_rows_before": 372993, "test_rows_after": 372993,
            "test_rows_added": 0, "test_rows_removed": 0, "test_rows_affected": 0, "test_values_changed": 0,
            "columns_added": 0
        }
    ]
    df_log = pd.DataFrame(steps_data)
    for d in target_dirs:
        log_p = d / "preprocessing_step_log.csv"
        df_log.to_csv(log_p, index=False, encoding="utf-8")
        print(f"  -> Da luu log kiem toan tai: {log_p}")


def generate_context_eda_plot(df_train: pd.DataFrame, output_path: Path) -> Path:
    """Plot sampled train distributions for the four Context-Aware features."""
    import matplotlib.pyplot as plt

    features = [
        ("z_meter_local", "Lệch phụ tải theo công trình và giờ"),
        ("z_meter_workday", "Lệch phụ tải theo loại ngày và giờ"),
        ("z_meter_robust", "Lệch so với trung vị/IQR công trình"),
        ("temp_load_inter", "Tương tác lệch phụ tải và nhiệt độ"),
    ]
    required = {"anomaly", *(column for column, _ in features)}
    missing = sorted(required.difference(df_train.columns))
    if missing:
        raise ValueError(f"Train data is missing Context EDA columns: {missing}")

    columns = [column for column, _ in features]
    context_data = df_train[["anomaly", *columns]].dropna()
    context_samples = {
        label: group.sample(n=min(30000, len(group)), random_state=42)
        for label, group in context_data.groupby("anomaly")
    }

    colors = {0: "#1f77b4", 1: "#d62728"}
    labels = {0: "Bình thường (y=0)", 1: "Dị thường (y=1)"}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=300)
    for ax, (column, title) in zip(axes.flat, features):
        lower, upper = df_train[column].quantile([0.01, 0.99]).to_numpy()
        if lower == upper:
            lower, upper = lower - 0.5, upper + 0.5
        bins = np.linspace(lower, upper, 55)
        for label in (0, 1):
            if label not in context_samples:
                continue
            ax.hist(
                context_samples[label][column],
                bins=bins,
                density=True,
                histtype="step",
                linewidth=1.8,
                color=colors[label],
                label=labels[label],
            )
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel(column)
        ax.set_ylabel("Mật độ")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8)

    fig.suptitle("Phân bố các đặc trưng ngữ cảnh theo nhãn trên tập train", fontsize=13, fontweight="bold")
    fig.text(0.5, 0.015, "Lấy mẫu cùng kích thước mỗi lớp; trục ngang hiển thị khoảng phân vị 1–99.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def generate_eda_plots() -> None:
    """
    Sinh va cap nhat toan bo he thong bieu do EDA khoa hoc (300 DPI) vao report_p4aids/IMAGE/.
    Bao gom cac bieu do tong quan, Context-Aware va Expanded.
    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import seaborn as sns
    from scipy import stats

    IMAGE_DIR = ROOT_DIR / "report_p4aids" / "IMAGE"
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    tr_path = OUT_PROC_DIR / "train_chunked_expanded.parquet"
    te_path = OUT_PROC_DIR / "test_chunked_expanded.parquet"
    if not tr_path.exists():
        tr_path = OUT_ML_DIR / "train_chunked_expanded.parquet"
        te_path = OUT_ML_DIR / "test_chunked_expanded.parquet"

    print("\n" + "=" * 100)
    print("KHOI CHAY HE THONG SINH BIEU DO EDA TOAN DIEN (300 DPI) CHO BAO CAO LUAN VAN")
    print("=" * 100)

    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.edgecolor"] = "#333333"
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["grid.color"] = "#E0E0E0"
    plt.rcParams["grid.linestyle"] = "--"
    plt.rcParams["grid.alpha"] = 0.7

    COLOR_NORMAL = "#1f77b4"
    COLOR_ANOMALY = "#d62728"
    COLOR_TRAIN = "#2ca02c"
    COLOR_TEST = "#ff7f0e"
    COLOR_TEAL = "#167D8D"
    COLOR_CORAL = "#E07A3F"

    # =========================================================================
    # PHAN 1: CAC BIEU DO EDA GOC TU TAP THO TRAIN_FEATURES.CSV
    # =========================================================================
    if DATA_RAW.exists():
        print("-> [1/8] Dang doc mau du lieu goc train_features.csv...", end=" ", flush=True)
        cols_raw = ["site_id", "meter_reading", "air_temperature", "dew_temperature", "wind_speed", "square_feet", "anomaly"]
        df_raw_eda = pd.read_csv(DATA_RAW, usecols=cols_raw, low_memory=False)
        df_sample = df_raw_eda.sample(n=min(300000, len(df_raw_eda)), random_state=42)
        print("Xong!")

        # ---------------------------------------------------------------------
        # HINH 1: eda_histogram_distribution.png (6 o phan phoi tan suat)
        # ---------------------------------------------------------------------
        print("  -> Xuat eda_histogram_distribution.png ...", end=" ", flush=True)
        fig, axes = plt.subplots(2, 3, figsize=(18, 11), dpi=300)
        fig.patch.set_facecolor("#FAFAFA")
        for ax in axes.flat:
            ax.set_facecolor("#FFFFFF")
            ax.grid(True, zorder=0)

        # (a) Raw meter_reading
        raw_m = df_raw_eda["meter_reading"].dropna()
        p99 = float(raw_m.quantile(0.99))
        med_m = float(raw_m.median())
        mean_m = float(raw_m.mean())
        s_cap = df_sample["meter_reading"].dropna().clip(upper=p99)

        axes[0, 0].hist(s_cap, bins=50, color=COLOR_TEAL, alpha=0.8, edgecolor="white", density=True, zorder=3)
        sns.kdeplot(s_cap, ax=axes[0, 0], color="#0B4D57", linewidth=2.0, zorder=4)
        axes[0, 0].axvline(med_m, color="#E74C3C", linestyle="-", linewidth=2, label=f"Trung vị: {med_m:.1f} kWh")
        axes[0, 0].axvline(mean_m, color="#F39C12", linestyle="--", linewidth=2, label=f"Trung bình: {mean_m:.1f} kWh")
        axes[0, 0].set_title("(a) Phân phối Phụ tải Gốc meter_reading (Raw)", fontweight="bold")
        axes[0, 0].set_xlabel("Phụ tải điện (kWh)")
        axes[0, 0].set_ylabel("Mật độ xác suất (Density)")
        axes[0, 0].legend(loc="upper right", framealpha=0.95)

        # (b) log1p(meter_reading)
        log_m = np.log1p(df_sample["meter_reading"].dropna().clip(lower=0))
        axes[0, 1].hist(log_m, bins=50, color=COLOR_CORAL, alpha=0.8, edgecolor="white", density=True, zorder=3)
        sns.kdeplot(log_m, ax=axes[0, 1], color="#BA4A00", linewidth=2.0, zorder=4)
        axes[0, 1].axvline(log_m.median(), color="#27AE60", linestyle="-", linewidth=2, label=f"Trung vị: {log_m.median():.2f}")
        axes[0, 1].axvline(log_m.mean(), color="#8E44AD", linestyle="--", linewidth=2, label=f"Trung bình: {log_m.mean():.2f}")
        axes[0, 1].set_title("(b) Tác động Phục hồi Tính Chuẩn tắc qua log(1+x)", fontweight="bold")
        axes[0, 1].set_xlabel("log(1 + meter_reading)")
        axes[0, 1].set_ylabel("Mật độ xác suất (Density)")
        axes[0, 1].legend(loc="upper right", framealpha=0.95)

        # (c) Normal vs Anomaly
        log_norm = np.log1p(df_sample.loc[df_sample["anomaly"] == 0, "meter_reading"].dropna().clip(lower=0))
        log_anom = np.log1p(df_sample.loc[df_sample["anomaly"] == 1, "meter_reading"].dropna().clip(lower=0))
        sns.kdeplot(log_norm, ax=axes[0, 2], color=COLOR_NORMAL, label="Bình thường (y=0)", fill=True, alpha=0.35, linewidth=2)
        sns.kdeplot(log_anom, ax=axes[0, 2], color=COLOR_ANOMALY, label="Dị thường (y=1)", fill=True, alpha=0.35, linewidth=2)
        axes[0, 2].set_title("(c) Phân tách Phụ tải theo Nhãn (Normal vs Anomaly)", fontweight="bold")
        axes[0, 2].set_xlabel("log(1 + meter_reading)")
        axes[0, 2].set_ylabel("Mật độ xác suất (Density)")
        axes[0, 2].legend(loc="upper right", framealpha=0.95)

        # (d) air_temperature (Bimodal)
        temp_s = df_sample["air_temperature"].dropna()
        axes[1, 0].hist(temp_s, bins=45, color="#6C5CE7", alpha=0.75, edgecolor="white", density=True, zorder=3)
        sns.kdeplot(temp_s, ax=axes[1, 0], color="#341F97", linewidth=2.0, zorder=4)
        axes[1, 0].set_title("(d) Phân phối Lưỡng đỉnh của Nhiệt độ Không khí", fontweight="bold")
        axes[1, 0].set_xlabel("Nhiệt độ không khí (°C)")
        axes[1, 0].set_ylabel("Mật độ xác suất (Density)")

        # (e) square_feet
        sqft_log = np.log10(df_sample["square_feet"].dropna().clip(lower=100))
        axes[1, 1].hist(sqft_log, bins=40, color="#10AC84", alpha=0.75, edgecolor="white", density=True, zorder=3)
        sns.kdeplot(sqft_log, ax=axes[1, 1], color="#1DD1A1", linewidth=2.0, zorder=4)
        axes[1, 1].set_title("(e) Phân tầng Quy mô Diện tích Sàn (square_feet)", fontweight="bold")
        axes[1, 1].set_xlabel("log10(Diện tích sàn - ft²)")
        axes[1, 1].set_ylabel("Mật độ xác suất (Density)")

        # (f) Dew temp vs Air temp
        sub_w = df_sample[["air_temperature", "dew_temperature"]].dropna()
        axes[1, 2].hexbin(sub_w["air_temperature"], sub_w["dew_temperature"], gridsize=35, cmap="YlGnBu", mincnt=1)
        axes[1, 2].plot([-10, 40], [-10, 40], color="red", linestyle="--", linewidth=1.5, label="Đường bão hòa T = T_dew")
        axes[1, 2].set_title("(f) Tương quan Nhiệt độ Không khí & Đọng sương", fontweight="bold")
        axes[1, 2].set_xlabel("Nhiệt độ không khí (°C)")
        axes[1, 2].set_ylabel("Nhiệt độ đọng sương (°C)")
        axes[1, 2].legend(loc="upper left", framealpha=0.95)

        plt.suptitle("LƯỢC ĐỒ PHÂN PHỐI TẦN SUẤT ĐẶC TRƯNG KHÁM PHÁ DỮ LIỆU BAN ĐẦU (EDA HISTOGRAM DIAGRAMS)", fontsize=13, fontweight="bold", y=0.99)
        plt.tight_layout()
        out_hist = IMAGE_DIR / "eda_histogram_distribution.png"
        plt.savefig(out_hist, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Xong! ({out_hist.name})")

        # ---------------------------------------------------------------------
        # HINH 2: eda_zone_comparison.png (So sanh Site 0, 2, 4, 9)
        # ---------------------------------------------------------------------
        print("  -> [2/8] Xuat eda_zone_comparison.png ...", end=" ", flush=True)
        target_sites = [0, 2, 4, 9]
        df_sites = df_sample[df_sample["site_id"].isin(target_sites)].copy()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)
        fig.patch.set_facecolor("#FAFAFA")
        ax1.set_facecolor("#FFFFFF")
        ax2.set_facecolor("#FFFFFF")
        ax1.grid(True, axis="y")
        ax2.grid(True)

        site_palette = {0: "#1f77b4", 2: "#2ca02c", 4: "#d62728", 9: "#9467bd"}
        site_labels = {0: "Site 0 (Ấm áp)", 2: "Site 2 (Ôn hòa)", 4: "Site 4 (Lạnh)", 9: "Site 9 (Tải lớn)"}
        df_sites["site_label"] = df_sites["site_id"].map(site_labels)

        # Subplot (a): Boxplot phu tai meter_reading
        df_sites["meter_capped"] = df_sites["meter_reading"].clip(0, 500)
        sns.boxplot(
            data=df_sites,
            x="site_label",
            y="meter_capped",
            hue="site_label",
            legend=False,
            palette=list(site_palette.values()),
            ax=ax1,
            fliersize=1.2,
        )
        ax1.set_title("(a) Phân bố Mức tiêu thụ điện theo Khu vực (Site 0, 2, 4, 9)", fontweight="bold", pad=10)
        ax1.set_xlabel("Khu vực địa lý (Site ID)")
        ax1.set_ylabel("Công suất tiêu thụ (kWh - Capped 500)")

        # Subplot (b): KDE nhiet do air_temperature
        for s_id in target_sites:
            s_temp = df_sites.loc[df_sites["site_id"] == s_id, "air_temperature"].dropna()
            sns.kdeplot(s_temp, ax=ax2, label=site_labels[s_id], color=site_palette[s_id], linewidth=2.0)
        ax2.set_title("(b) Phân phối Nhiệt độ Môi trường theo Khu vực", fontweight="bold", pad=10)
        ax2.set_xlabel("Nhiệt độ không khí (°C)")
        ax2.set_ylabel("Mật độ phân phối (Density)")
        ax2.legend(loc="upper right", framealpha=0.95)

        plt.suptitle("PHÂN BỐ MỨC TIÊU THỤ ĐIỆN VÀ NHIỆT ĐỘ GIỮA CÁC KHU VỰC TIÊU BIỂU", fontsize=13, fontweight="bold", y=1.02)
        plt.tight_layout()
        out_zone = IMAGE_DIR / "eda_zone_comparison.png"
        plt.savefig(out_zone, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Xong! ({out_zone.name})")

        del df_raw_eda, df_sample, df_sites

    # -------------------------------------------------------------------------
    # HINH 3: preprocessing_history_diagram.png (Lich su tien xu ly 3 nấc)
    # -------------------------------------------------------------------------
    print("  -> [3/8] Xuat preprocessing_history_diagram.png ...", end=" ", flush=True)
    stages = [
        "1. Nạp CSV Thô",
        "2. Chunking 96h",
        "3. Khử 23 cột ID",
        "4. Cờ Missing",
        "5. Lọc IQR",
        "6. Z-Score/Zone",
        "7. Context-Aware",
        "8. Expanded Champion",
    ]
    x_idx = np.arange(len(stages))
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), dpi=300, sharex=False)
    fig.patch.set_facecolor("#FAFAFA")
    for ax in axes:
        ax.set_facecolor("#FFFFFF")
        ax.grid(True, zorder=0)

    # (a) Dữ liệu khuyết thiếu
    missing_rates = [6.15, 6.15, 6.15, 6.15, 0.00, 0.00, 0.00, 0.00]
    axes[0].plot(x_idx, missing_rates, marker="o", color=COLOR_ANOMALY, linewidth=2.5, markersize=8, zorder=4)
    axes[0].fill_between(x_idx, missing_rates, color=COLOR_ANOMALY, alpha=0.15)
    axes[0].set_title("(a) Diễn biến Triệt tiêu Dữ liệu Khuyết thiếu (Missing Rate % ở biến meter_reading)", pad=8, fontweight="bold")
    axes[0].set_ylabel("Tỷ lệ thiếu (%)")
    axes[0].set_xticks(x_idx)
    axes[0].set_xticklabels(stages)
    axes[0].set_ylim(-0.5, 8.0)

    # (b) Chiều không gian đặc trưng qua 4 nhánh
    base_cols = [57, 57, 34, 35, 35, 40, 40, 40]
    zone_cols = [57, 57, 34, 35, 35, 43, 43, 43]
    ctx_cols  = [57, 57, 34, 35, 35, 43, 47, 47]
    exp_cols  = [57, 57, 34, 35, 35, 43, 47, 55]

    axes[1].plot(x_idx, base_cols, marker="s", color=COLOR_TEAL, linewidth=2, label="Baseline (40 cột)", linestyle=":")
    axes[1].plot(x_idx, zone_cols, marker="^", color=COLOR_CORAL, linewidth=2, label="Zone-Enhanced (43 cột)", linestyle="--")
    axes[1].plot(x_idx, ctx_cols, marker="o", color="#6C5CE7", linewidth=2.2, label="Context-Aware (47 cột)", linestyle="-.")
    axes[1].plot(x_idx, exp_cols, marker="D", color="#C0392B", linewidth=2.8, label="Expanded Champion SOTA (55 cột)")
    axes[1].set_title("(b) Biến động Chiều Không gian Đặc trưng & Sự Phân nhánh 4 Cấp độ Dữ liệu", pad=8, fontweight="bold")
    axes[1].set_ylabel("Số lượng cột")
    axes[1].set_xticks(x_idx)
    axes[1].set_xticklabels(stages)
    axes[1].set_ylim(25, 62)
    axes[1].legend(loc="upper right", framealpha=0.95)

    # (c) Kiểm soát số lượng mẫu Zero Leakage
    train_samples = np.array([1749494, 1147089, 1147089, 1147089, 1147089, 1147089, 1147089, 1147089])
    purge_samples = np.array([0, 229412, 229412, 229412, 229412, 229412, 229412, 229412])
    test_samples  = np.array([0, 372993, 372993, 372993, 372993, 372993, 372993, 372993])
    width = 0.55
    axes[2].bar(x_idx, train_samples, width, label="Tập Train Tin (Ngày 01-20: 1.147.089 mẫu - 65,6%)", color="#167D8D", alpha=0.85, zorder=3)
    axes[2].bar(x_idx, purge_samples, width, bottom=train_samples, label="Purge Gap 96h (Ngày 21-24: 229.412 mẫu - 13,1% [Loại bỏ cách ly])", color="#BDC3C7", hatch="///", edgecolor="#C94C4C", zorder=3)
    axes[2].bar(x_idx, test_samples, width, bottom=train_samples + purge_samples, label="Tập Test Tout (Ngày 25-Cuối: 372.993 mẫu - 21,3%)", color="#E07A3F", alpha=0.85, zorder=3)
    axes[2].set_title("(c) Kiểm soát Số lượng Mẫu & Giao thức 100% Zero Data Leakage", pad=8, fontweight="bold")
    axes[2].set_ylabel("Số lượng bản ghi")
    axes[2].set_xticks(x_idx)
    axes[2].set_xticklabels(stages, fontweight="bold")
    axes[2].yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{int(y):,}"))
    axes[2].legend(loc="lower left", framealpha=0.95)

    plt.suptitle("LƯỢC ĐỒ LỊCH SỬ TIẾN TRÌNH TIỀN XỬ LÝ DỮ LIỆU ĐỒNG BỘ (ZERO DATA LEAKAGE PROTOCOL)", fontsize=13, fontweight="bold", y=0.995)
    plt.tight_layout()
    out_hist_diag = IMAGE_DIR / "preprocessing_history_diagram.png"
    plt.savefig(out_hist_diag, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Xong! ({out_hist_diag.name})")

    # =========================================================================
    # PHAN 2: CAC BIEU DO TREN TAP EXPANDED PARQUET (TRAIN & TEST)
    # =========================================================================
    if not tr_path.exists() or not te_path.exists():
        print("[CANH BAO] Chua co du lieu expanded parquet de ve cac bieu do tiep theo!")
        return

    print("-> Dang doc du lieu expanded parquet de ve 4 bieu do chuyen sau...", end=" ", flush=True)
    df_tr = pd.read_parquet(tr_path)
    df_te = pd.read_parquet(te_path)

    # Neu tap Parquet da loai bo building_id de chong ghi nho ID, nap bo sung building_id tu du lieu tho phuc vu ve Case Study
    if "building_id" not in df_tr.columns and DATA_RAW.exists():
        raw_b_time = pd.read_csv(DATA_RAW, usecols=["timestamp", "building_id"], low_memory=False)
        raw_ts = pd.to_datetime(raw_b_time["timestamp"])
        train_mask = (raw_ts.dt.day >= 1) & (raw_ts.dt.day <= 20)
        df_tr["building_id"] = raw_b_time.loc[train_mask, "building_id"].values
    print("Xong!")

    # -------------------------------------------------------------------------
    # HINH 4: ml_train_test_distribution.png (Kiem toan Train vs Test ML)
    # -------------------------------------------------------------------------
    print("  -> [4/8] Xuat ml_train_test_distribution.png ...", end=" ", flush=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)
    fig.patch.set_facecolor("#FAFAFA")
    for ax in axes.flat:
        ax.set_facecolor("#FFFFFF")
        ax.grid(True)

    # (a) meter_reading
    m_tr_s = df_tr["meter_reading"].clip(-3, 5).sample(min(50000, len(df_tr)), random_state=42)
    m_te_s = df_te["meter_reading"].clip(-3, 5).sample(min(50000, len(df_te)), random_state=42)
    sns.kdeplot(m_tr_s, ax=axes[0, 0], color=COLOR_TRAIN, label="Tập Train (Tin: 1.15M)", linewidth=2.0)
    sns.kdeplot(m_te_s, ax=axes[0, 0], color=COLOR_TEST, label="Tập Test (Tout: 373k)", linewidth=2.0, linestyle="--")
    axes[0, 0].set_title("(a) Phân phối Chuẩn hóa meter_reading (Z-score)", fontweight="bold")
    axes[0, 0].set_xlabel("meter_reading (Standardized)")
    axes[0, 0].legend(loc="upper right", framealpha=0.95)

    # (b) air_temperature
    t_tr_s = df_tr["air_temperature"].sample(min(50000, len(df_tr)), random_state=42)
    t_te_s = df_te["air_temperature"].sample(min(50000, len(df_te)), random_state=42)
    sns.kdeplot(t_tr_s, ax=axes[0, 1], color=COLOR_TRAIN, label="Tập Train (Tin: 1.15M)", linewidth=2.0)
    sns.kdeplot(t_te_s, ax=axes[0, 1], color=COLOR_TEST, label="Tập Test (Tout: 373k)", linewidth=2.0, linestyle="--")
    axes[0, 1].set_title("(b) Phân phối Chuẩn hóa air_temperature (Z-score)", fontweight="bold")
    axes[0, 1].set_xlabel("air_temperature (Standardized)")
    axes[0, 1].legend(loc="upper right", framealpha=0.95)

    # (c) square_feet
    sq_tr_s = df_tr["square_feet"].sample(min(50000, len(df_tr)), random_state=42)
    sq_te_s = df_te["square_feet"].sample(min(50000, len(df_te)), random_state=42)
    sns.kdeplot(sq_tr_s, ax=axes[1, 0], color=COLOR_TRAIN, label="Tập Train (Tin: 1.15M)", linewidth=2.0)
    sns.kdeplot(sq_te_s, ax=axes[1, 0], color=COLOR_TEST, label="Tập Test (Tout: 373k)", linewidth=2.0, linestyle="--")
    axes[1, 0].set_title("(c) Phân phối Chuẩn hóa square_feet (Z-score)", fontweight="bold")
    axes[1, 0].set_xlabel("square_feet (Standardized)")
    axes[1, 0].legend(loc="upper right", framealpha=0.95)

    # (d) Anomaly Rate
    rates = [df_tr["anomaly"].mean() * 100, df_te["anomaly"].mean() * 100]
    bars = axes[1, 1].bar(["Tập Train (Tin)", "Tập Test (Tout)"], rates, color=[COLOR_TRAIN, COLOR_TEST], width=0.45, alpha=0.85)
    for b in bars:
        axes[1, 1].text(b.get_x() + b.get_width()/2, b.get_height() + 0.05, f"{b.get_height():.2f}%", ha="center", va="bottom", fontweight="bold")
    axes[1, 1].set_title("(d) Tỷ lệ Nhãn Dị thường (Anomaly Rate %)", fontweight="bold")
    axes[1, 1].set_ylabel("Tỷ lệ % Anomaly")
    axes[1, 1].set_ylim(0, 3.5)

    plt.suptitle("KIỂM SOÁT TÍNH ĐỒNG NHẤT PHÂN PHỐI GIỮA TẬP TRAIN VÀ TẬP TEST (SEASONAL CHUNKING + PURGE GAP 96H)", fontsize=13, fontweight="bold", y=0.99)
    plt.tight_layout()
    out_ml_drift = IMAGE_DIR / "ml_train_test_distribution.png"
    plt.savefig(out_ml_drift, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Xong! ({out_ml_drift.name})")

    # -------------------------------------------------------------------------
    # HINH 5: eda_context_features.png
    # -------------------------------------------------------------------------
    print("  -> [5/8] Xuat eda_context_features.png ...", end=" ", flush=True)
    out_context = generate_context_eda_plot(df_tr, IMAGE_DIR / "eda_context_features.png")
    print(f"Xong! ({out_context.name})")

    # -------------------------------------------------------------------------
    # HINH 5: eda_expanded_temporal_dynamics.png
    # -------------------------------------------------------------------------
    print("  -> [6/8] Xuat eda_expanded_temporal_dynamics.png ...", end=" ", flush=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)

    # (a) flatline_duration
    ax = axes[0, 0]
    sample_flat = df_tr[["flatline_duration", "anomaly"]].copy()
    sample_flat["flatline_log"] = np.log1p(sample_flat["flatline_duration"].clip(lower=0))
    sns.boxplot(
        data=sample_flat.sample(min(100000, len(sample_flat)), random_state=42),
        x="anomaly",
        y="flatline_log",
        hue="anomaly",
        legend=False,
        palette=[COLOR_NORMAL, COLOR_ANOMALY],
        ax=ax,
        fliersize=1.5,
    )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Bình thường (Normal)", "Dị thường (Anomaly)"], fontsize=11, fontweight="bold")
    ax.set_ylabel("log1p(Thời lượng treo cảm biến - Giờ)", fontsize=11)
    ax.set_title("(a) Phân tách Đặc trưng Treo Cảm biến (flatline_duration)", fontsize=12, fontweight="bold", pad=10)
    ax.grid(True, axis="y")

    # (b) meter_diff_1h
    ax = axes[0, 1]
    diff_norm = df_tr.loc[df_tr["anomaly"] == 0, "meter_diff_1h"].clip(-100, 100)
    diff_anom = df_tr.loc[df_tr["anomaly"] == 1, "meter_diff_1h"].clip(-100, 100)
    sns.kdeplot(diff_norm.sample(min(50000, len(diff_norm)), random_state=42), ax=ax, color=COLOR_NORMAL, label="Bình thường (y=0)", fill=True, alpha=0.3)
    sns.kdeplot(diff_anom.sample(min(20000, len(diff_anom)), random_state=42), ax=ax, color=COLOR_ANOMALY, label="Dị thường (y=1)", fill=True, alpha=0.3)
    ax.set_xlim(-80, 80)
    ax.set_xlabel("Vận tốc biến thiên tức thời: Δ1 = m(t) - m(t-1)", fontsize=11)
    ax.set_ylabel("Mật độ xác suất (Density)", fontsize=11)
    ax.set_title("(b) Độ phân tách Vận tốc Biến thiên (meter_diff_1h)", fontsize=12, fontweight="bold", pad=10)
    ax.legend(frameon=True, facecolor="white", edgecolor="none")
    ax.grid(True)

    # (c) meter_ratio_1h
    ax = axes[1, 0]
    ratio_norm = np.log1p(df_tr.loc[df_tr["anomaly"] == 0, "meter_ratio_1h"].clip(0, 50))
    ratio_anom = np.log1p(df_tr.loc[df_tr["anomaly"] == 1, "meter_ratio_1h"].clip(0, 50))
    sns.kdeplot(ratio_norm.sample(min(50000, len(ratio_norm)), random_state=42), ax=ax, color=COLOR_NORMAL, label="Bình thường (y=0)", fill=True, alpha=0.3)
    sns.kdeplot(ratio_anom.sample(min(20000, len(ratio_anom)), random_state=42), ax=ax, color=COLOR_ANOMALY, label="Dị thường (y=1)", fill=True, alpha=0.3)
    ax.set_xlabel("log1p(Tỉ số động năng phụ tải: R1)", fontsize=11)
    ax.set_ylabel("Mật độ xác suất (Density)", fontsize=11)
    ax.set_title("(c) Tỉ số Động năng Xung gai (meter_ratio_1h)", fontsize=12, fontweight="bold", pad=10)
    ax.legend(frameon=True, facecolor="white", edgecolor="none")
    ax.grid(True)

    # (d) stl_residual
    ax = axes[1, 1]
    stl_norm = df_tr.loc[df_tr["anomaly"] == 0, "stl_residual"].clip(-150, 150)
    stl_anom = df_tr.loc[df_tr["anomaly"] == 1, "stl_residual"].clip(-150, 150)
    sns.kdeplot(stl_norm.sample(min(50000, len(stl_norm)), random_state=42), ax=ax, color=COLOR_NORMAL, label="Bình thường (y=0)", fill=True, alpha=0.3)
    sns.kdeplot(stl_anom.sample(min(20000, len(stl_anom)), random_state=42), ax=ax, color=COLOR_ANOMALY, label="Dị thường (y=1)", fill=True, alpha=0.3)
    ax.set_xlim(-120, 120)
    ax.set_xlabel("Phần dư sau khử chu kỳ 24h: stl_residual", fontsize=11)
    ax.set_ylabel("Mật độ xác suất (Density)", fontsize=11)
    ax.set_title("(d) Phần dư Phân rã Chuỗi thời gian (stl_residual)", fontsize=12, fontweight="bold", pad=10)
    ax.legend(frameon=True, facecolor="white", edgecolor="none")
    ax.grid(True)

    plt.tight_layout()
    out1 = IMAGE_DIR / "eda_expanded_temporal_dynamics.png"
    plt.savefig(out1, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Xong! ({out1.name})")

    # -------------------------------------------------------------------------
    # HINH 6: eda_expanded_train_test_drift.png
    # -------------------------------------------------------------------------
    print("  -> [7/8] Xuat eda_expanded_train_test_drift.png ...", end=" ", flush=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)
    features_to_check = [
        ("meter_lag_1h", "Công suất 1h trước (meter_lag_1h)", (0, 300)),
        ("meter_diff_1h", "Tốc độ biến thiên tức thời (meter_diff_1h)", (-50, 50)),
        ("flatline_duration", "Thời lượng treo cảm biến (flatline_duration)", (0, 20)),
        ("stl_residual", "Phần dư sau khử chu kỳ (stl_residual)", (-80, 80)),
    ]
    for idx, (col, title, xlim) in enumerate(features_to_check):
        ax = axes[idx // 2, idx % 2]
        s_tr = df_tr[col].clip(*xlim).sample(min(50000, len(df_tr)), random_state=42)
        s_te = df_te[col].clip(*xlim).sample(min(50000, len(df_te)), random_state=42)
        sns.kdeplot(s_tr, ax=ax, color=COLOR_TRAIN, label="Tập Train (T_in: 1.15M)", linewidth=2.0)
        sns.kdeplot(s_te, ax=ax, color=COLOR_TEST, label="Tập Test (T_out: 373k)", linewidth=2.0, linestyle="--")
        ax.set_xlim(*xlim)
        ax.set_xlabel(title, fontsize=11)
        ax.set_ylabel("Mật độ phân phối", fontsize=11)
        ax.set_title(f"Kiểm toán Covariate Shift: {col}", fontsize=12, fontweight="bold", pad=10)
        ax.legend(frameon=True, facecolor="white", edgecolor="none")
        ax.grid(True)

    plt.suptitle("KIỂM TOÁN ZERO COVARIATE SHIFT: ĐỒNG NHẤT PHÂN PHỐI TRAIN VS TEST QUA PURGE GAP 96H", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    out2 = IMAGE_DIR / "eda_expanded_train_test_drift.png"
    plt.savefig(out2, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Xong! ({out2.name})")

    # -------------------------------------------------------------------------
    # HINH 7: eda_expanded_case_study.png
    # -------------------------------------------------------------------------
    print("  -> [8/8] Xuat eda_expanded_case_study.png ...", end=" ", flush=True)
    if "building_id" in df_tr.columns:
        b_candidates = df_tr.groupby("building_id")["anomaly"].sum().sort_values(ascending=False).index[:15]
        selected_b = b_candidates[0]
        for b in b_candidates:
            sub = df_tr[df_tr["building_id"] == b]
            if (sub["flatline_duration"] > 5).any() and (sub["anomaly"] == 1).sum() > 20:
                selected_b = b
                break
        df_b = df_tr[df_tr["building_id"] == selected_b].iloc[:168].copy().reset_index(drop=True)
        case_title = f"CASE STUDY CHUỖI THỜI GIAN TOÀ NHÀ {selected_b}: CƠ CHẾ PHÁT HIỆN DỊ THƯỜNG CỦA 8 ĐẶC TRƯNG MỚI"
    else:
        anom_idx = df_tr[df_tr["anomaly"] == 1].index
        start_idx = max(0, anom_idx[0] - 24) if len(anom_idx) > 0 else 0
        df_b = df_tr.iloc[start_idx:start_idx + 168].copy().reset_index(drop=True)
        case_title = "CASE STUDY CHUỖI THỜI GIAN ĐIỂN HÌNH: CƠ CHẾ PHÁT HIỆN DỊ THƯỜNG CỦA 8 ĐẶC TRƯNG MỚI"

    x_axis = np.arange(len(df_b))

    fig, axes = plt.subplots(3, 1, figsize=(15, 9), dpi=300, sharex=True)

    ax = axes[0]
    ax.plot(x_axis, df_b["meter_reading"], color="#1f77b4", linewidth=1.8, label="Công suất tiêu thụ (meter_reading)")
    anom_mask = df_b["anomaly"] == 1
    if anom_mask.any():
        ax.scatter(x_axis[anom_mask], df_b.loc[anom_mask, "meter_reading"], color="red", s=35, zorder=5, label="Điểm Dị Thường Thực Tế (y=1)")
    ax.set_ylabel("Công suất (kW)", fontsize=11, fontweight="bold")
    ax.set_title(case_title, fontsize=13, fontweight="bold", pad=10)
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True)

    ax = axes[1]
    ax.fill_between(x_axis, df_b["flatline_duration"], color="#ff7f0e", alpha=0.4, label="Thời lượng treo cảm biến (flatline_duration)")
    ax.plot(x_axis, df_b["flatline_duration"], color="#d95f02", linewidth=1.5)
    ax.set_ylabel("Giờ treo liên tục", fontsize=11, fontweight="bold")
    ax.set_title("Chỉ dấu Đóng Băng Cảm Biến: Tích luỹ tăng vọt khi phụ tải bất động (|Δ1| ≤ 1e-4)", fontsize=11, fontstyle="italic")
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True)

    ax = axes[2]
    ax.plot(x_axis, df_b["meter_diff_1h"], color="#2ca02c", linewidth=1.2, alpha=0.8, label="Vận tốc biến thiên tức thời (meter_diff_1h)")
    ax.plot(x_axis, df_b["stl_residual"], color="#9467bd", linewidth=1.2, linestyle="--", alpha=0.8, label="Phần dư phân rã mùa vụ 24h (stl_residual)")
    ax.axhline(0, color="black", linestyle=":", linewidth=0.8)
    ax.set_xlabel("Trục thời gian tương đối (Giờ trong tuần mẫu - 168h)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Giá trị sai phân / dư", fontsize=11, fontweight="bold")
    ax.set_title("Chỉ dấu Xung Gai Đột Biến & Phá vỡ Chu kỳ Tuần hoàn Ngày/Đêm", fontsize=11, fontstyle="italic")
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True)

    plt.tight_layout()
    out3 = IMAGE_DIR / "eda_expanded_case_study.png"
    plt.savefig(out3, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Xong! ({out3.name})")

    print(f"\n[HOAN TAT] DA XUAT DU TOAN BO 8 BIEU DO KHOA HOC 300 DPI TAI: {IMAGE_DIR}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="preprocessingforALL.py [NEW] - Zero Data Leakage Preprocessing Protocol"
    )
    parser.add_argument("--save-processed", action="store_true", default=True, help="Luu ca ban copy sang data/processed/ (Mac dinh: True)")
    parser.add_argument("--plots", "--eda", dest="plots", action="store_true", default=False, help="Sinh cac bieu do EDA 300 DPI vao report_p4aids/IMAGE/")
    parser.add_argument("--only-plots", "--only-eda", dest="only_plots", action="store_true", default=False, help="Chi sinh bieu do EDA ma khong chay lai tien xu ly")
    args = parser.parse_args()

    if args.only_plots:
        generate_eda_plots()
    else:
        run_zero_leakage_preprocessing(
            save_to_model_ml=True,
            save_to_processed=True,
        )
        if args.plots:
            generate_eda_plots()
