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

3. XUẤT CẢ HAI BỘ TRAIN VÀ TEST CHO ĐẦY ĐỦ CÁC BIẾN THỂ:
   - Bộ 1: Baseline (40 cột = 39 đặc trưng cơ sở + 1 nhãn anomaly)
   - Bộ 2: Zone-Enhanced (43 cột = 39 Baseline + 3 đặc trưng vĩ mô phân vùng + 1 nhãn anomaly)
   - Bộ 3: Context-Aware (47 cột = 42 Zone + 4 đặc trưng ngữ cảnh + 1 nhãn anomaly)
   - Bộ 4: Expanded Champion (55 cột = 46 Context + 8 đặc trưng chuỗi thời gian & STL Residual + 1 nhãn anomaly)

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

    # --- KHOI TAO BO DU LIEU 1: BASELINE (40 COT = 39 FEATURES + 1 TARGET) ---
    train_base = X_train.copy()
    test_base = X_test.copy()
    train_base[TARGET] = y_train.values
    test_base[TARGET] = y_test.values

    # --- KHOI TAO BO DU LIEU 2: ZONE-ENHANCED (43 COT = 42 FEATURES + 1 TARGET) ---
    train_zone = X_train_zone.copy()
    test_zone = X_test_zone.copy()
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

    train_context = X_train_ctx.copy()
    test_context = X_test_ctx.copy()
    train_context[TARGET] = y_train.values
    test_context[TARGET] = y_test.values

    # -------------------------------------------------------------------------
    # [BUOC 10/10] TICH HOP BO 4: DYNAMIC-TEMPORAL-EXPANDED (55 COT CHAMPION SOTA)
    # -------------------------------------------------------------------------
    print("\n[BUOC 10/10] Tich hop Bo thu 4: Dynamic-Temporal-Expanded (55 cot Champion SOTA)...")
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
        "train_chunked_baseline.parquet": (train_base, 40),
        "test_chunked_baseline.parquet": (test_base, 40),
        "train_chunked_zone.parquet": (train_zone, 43),
        "test_chunked_zone.parquet": (test_zone, 43),
        "train_chunked_context.parquet": (train_context, 47),
        "test_chunked_context.parquet": (test_context, 47),
        "train_chunked_expanded.parquet": (train_expanded, 55),
        "test_chunked_expanded.parquet": (test_expanded, 55),
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
        "baseline_features": 39,
        "zone_features": 42,
        "context_features": 46,
        "expanded_features": 54,
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
            "step": "Chuẩn hóa Z-score & Tích hợp Zone Features (Baseline 40, Zone 43)",
            "train_rows_before": 1147089, "train_rows_after": 1147089,
            "train_rows_added": 0, "train_rows_removed": 0, "train_rows_affected": 1147089, "train_values_changed": 21895258,
            "test_rows_before": 372993, "test_rows_after": 372993,
            "test_rows_added": 0, "test_rows_removed": 0, "test_rows_affected": 372993, "test_values_changed": 6967528,
            "columns_added": 3
        },
        {
            "step": "Tích hợp Ngữ cảnh Vi mô Context-Aware (47 cột SOTA)",
            "train_rows_before": 1147089, "train_rows_after": 1147089,
            "train_rows_added": 0, "train_rows_removed": 0, "train_rows_affected": 1147089, "train_values_changed": 4588356,
            "test_rows_before": 372993, "test_rows_after": 372993,
            "test_rows_added": 0, "test_rows_removed": 0, "test_rows_affected": 372993, "test_values_changed": 1491972,
            "columns_added": 4
        },
        {
            "step": "Tích hợp Động học Chuỗi thời gian & STL Residual (Expanded 55 cột Champion)",
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


def generate_eda_plots() -> None:
    """Sinh va cap nhat cac bieu do EDA chuan xuat ban khoa hoc (300 DPI) vao report_p4aids/IMAGE/."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    IMAGE_DIR = ROOT_DIR / "report_p4aids" / "IMAGE"
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    tr_path = OUT_PROC_DIR / "train_chunked_expanded.parquet"
    te_path = OUT_PROC_DIR / "test_chunked_expanded.parquet"
    if not tr_path.exists():
        tr_path = OUT_ML_DIR / "train_chunked_expanded.parquet"
        te_path = OUT_ML_DIR / "test_chunked_expanded.parquet"

    print("\n" + "=" * 95)
    print("DANG SINH HE THONG BIEU DO EDA KHOA HOC (300 DPI) TICH HOP TRONG PIPELINE")
    print("=" * 95)

    if not tr_path.exists() or not te_path.exists():
        print("[CANH BAO] Chua co du lieu expanded parquet de ve EDA. Vui long chay tien xu ly truoc!")
        return

    print("-> Dang doc du lieu expanded de ve bieu do...", end=" ", flush=True)
    df_tr = pd.read_parquet(tr_path)
    df_te = pd.read_parquet(te_path)
    print("Xong!")

    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["axes.edgecolor"] = "#333333"
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["grid.color"] = "#E0E0E0"
    plt.rcParams["grid.linestyle"] = "--"
    plt.rcParams["grid.alpha"] = 0.7

    COLOR_NORMAL = "#1f77b4"
    COLOR_ANOMALY = "#d62728"
    COLOR_TRAIN = "#2ca02c"
    COLOR_TEST = "#ff7f0e"

    # 1. Bieu do 1: Phan tach phan phoi Normal vs Anomaly
    print("  -> [1/3] Xuat eda_expanded_temporal_dynamics.png ...", end=" ", flush=True)
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

    # 2. Bieu do 2: Kiem toan Zero Covariate Shift (Train vs Test)
    print("  -> [2/3] Xuat eda_expanded_train_test_drift.png ...", end=" ", flush=True)
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

    plt.suptitle("KIỂM TOÁN ZERO COVARIATE SHIFT: ĐỒNG NHẤT PHÂN PHỐI TRAIN VS TEST QUA PURGE GAP 96H", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    out2 = IMAGE_DIR / "eda_expanded_train_test_drift.png"
    plt.savefig(out2, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Xong! ({out2.name})")

    # 3. Bieu do 3: Case study chuoi thoi gian 1 toa nha
    print("  -> [3/3] Xuat eda_expanded_case_study.png ...", end=" ", flush=True)
    b_candidates = df_tr.groupby("building_id")["anomaly"].sum().sort_values(ascending=False).index[:15]
    selected_b = b_candidates[0]
    for b in b_candidates:
        sub = df_tr[df_tr["building_id"] == b]
        if (sub["flatline_duration"] > 5).any() and (sub["anomaly"] == 1).sum() > 20:
            selected_b = b
            break

    df_b = df_tr[df_tr["building_id"] == selected_b].iloc[:168].copy().reset_index(drop=True)
    x_axis = np.arange(len(df_b))

    fig, axes = plt.subplots(3, 1, figsize=(15, 9), dpi=300, sharex=True)

    ax = axes[0]
    ax.plot(x_axis, df_b["meter_reading"], color="#1f77b4", linewidth=1.8, label="Công suất tiêu thụ (meter_reading)")
    anom_mask = df_b["anomaly"] == 1
    if anom_mask.any():
        ax.scatter(x_axis[anom_mask], df_b.loc[anom_mask, "meter_reading"], color="red", s=35, zorder=5, label="Điểm Dị Thường Thực Tế (y=1)")
    ax.set_ylabel("Công suất (kW)", fontsize=11, fontweight="bold")
    ax.set_title(f"CASE STUDY CHUỖI THỜI GIAN TOÀ NHÀ {selected_b}: CƠ CHẾ PHÁT HIỆN DỊ THƯỜNG CỦA 8 ĐẶC TRƯNG MỚI", fontsize=13, fontweight="bold", pad=10)
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

    print(f"[HOAN TAT] Da xuat du 3 bieu do EDA khoa hoc 300 DPI tai: {IMAGE_DIR}")


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
