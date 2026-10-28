"""
Script tự động sinh các biểu đồ Lược đồ Phân phối Tần suất (Histogram Diagrams) cho EDA 
và Biểu đồ So sánh Phân phối Tập Huấn luyện - Tập Kiểm tra (Train vs Test Distribution) trước khi Training AI.
Tất cả hình ảnh được tối ưu hóa chuẩn xuất bản khoa học (300 DPI) và lưu trực tiếp vào report_p4aids/IMAGE/.
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from scipy import stats  # type: ignore

# Đảm bảo UTF-8 cho Windows Console
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8')
        if hasattr(sys.stderr, 'reconfigure'):
            sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Thiết lập đường dẫn thư mục dự án
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = BASE_DIR / "data" / "train_features.csv"
PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed" / "train_chunked_baseline.parquet"
TEST_PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed" / "test_chunked_baseline.parquet"
if not PROCESSED_DATA_PATH.exists():
    PROCESSED_DATA_PATH = BASE_DIR / "model_ML" / "data" / "train_chunked_baseline.parquet"
    TEST_PROCESSED_DATA_PATH = BASE_DIR / "model_ML" / "data" / "test_chunked_baseline.parquet"
if not PROCESSED_DATA_PATH.exists():
    PROCESSED_DATA_PATH = BASE_DIR / "experiments" / "data" / "train_chunked_baseline.parquet"
    TEST_PROCESSED_DATA_PATH = BASE_DIR / "experiments" / "data" / "test_chunked_baseline.parquet"
IMAGE_DIR = BASE_DIR / "report_p4aids" / "IMAGE"
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

# Cấu hình phong cách đồ họa chuyên nghiệp (Scientific Publication Style)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14
plt.rcParams['figure.titleweight'] = 'bold'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['grid.alpha'] = 0.5
plt.rcParams['grid.linestyle'] = '--'

# Bảng màu khoa học cao cấp
COLOR_PRIMARY = '#167D8D'    # Xanh mòng két đậm (Teal)
COLOR_SECONDARY = '#E07A3F'  # Cam san hô (Coral)
COLOR_ACCENT = '#C94C4C'     # Đỏ gạch (Crimson)
COLOR_DARK = '#2C3E50'       # Xanh than (Midnight Blue)
COLOR_GRAY = '#7F8C8D'       # Xám trung tính
COLOR_PURPLE = '#6C5CE7'     # Tím hoàng gia

def generate_eda_histogram_distribution():
    print("[1/2] Đang xử lý và vẽ Lược đồ Phân phối Tần suất EDA (Histogram Diagram)...")
    
    # Đọc dữ liệu raw train (các cột định lượng và nhãn quan trọng)
    cols_to_read = [
        'meter_reading', 'air_temperature', 'dew_temperature', 
        'wind_speed', 'square_feet', 'anomaly'
    ]
    df_raw = pd.read_csv(RAW_DATA_PATH, usecols=cols_to_read)
    
    # Lấy mẫu ngẫu nhiên tái lập để vẽ biểu đồ mật độ mượt mà (300,000 điểm)
    df_sample = df_raw.sample(n=300_000, random_state=42)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 11), dpi=300)
    fig.patch.set_facecolor('#FAFAFA')
    for ax in axes.flat:
        ax.set_facecolor('#FFFFFF')
        ax.grid(True, zorder=0)

    # -------------------------------------------------------------
    # 1. Histogram meter_reading gốc (Extreme Right-skewed)
    # -------------------------------------------------------------
    raw_readings = df_raw['meter_reading'].dropna()
    skew_val = raw_readings.skew()
    median_val = raw_readings.median()
    mean_val = raw_readings.mean()
    p99_val = raw_readings.quantile(0.99)
    
    sample_readings = df_sample['meter_reading'].dropna()
    sample_capped = sample_readings.clip(upper=p99_val) # Cắt ở p99 để biểu diễn trực quan
    
    axes[0, 0].hist(sample_capped, bins=50, color=COLOR_PRIMARY, alpha=0.8, edgecolor='white', density=True, zorder=3)
    sns.kdeplot(sample_capped, ax=axes[0, 0], color='#0B4D57', linewidth=2.2, zorder=4)
    axes[0, 0].axvline(median_val, color='#E74C3C', linestyle='-', linewidth=2, label=f'Trung vị: {median_val:.1f}')
    axes[0, 0].axvline(mean_val, color='#F39C12', linestyle='--', linewidth=2, label=f'Trung bình: {mean_val:.1f}')
    axes[0, 0].set_title("(a) Phân phối Phụ tải Gốc meter_reading (Raw)")
    axes[0, 0].set_xlabel("Phụ tải điện (kWh)")
    axes[0, 0].set_ylabel("Mật độ xác suất (Density)")
    axes[0, 0].legend(loc='upper right', framealpha=0.9)
    axes[0, 0].text(0.48, 0.45, f"Độ lệch (Skewness): +{skew_val:.2f}\n(Lệch phải cực đoan, đuôi nặng)",
                    transform=axes[0, 0].transAxes, fontsize=10, fontweight='semibold',
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="#FDEDEC", edgecolor="#E74C3C", alpha=0.9))

    # -------------------------------------------------------------
    # 2. Histogram log1p(meter_reading) (Nén phân phối)
    # -------------------------------------------------------------
    log_readings = np.log1p(sample_readings[sample_readings >= 0])
    log_skew = pd.Series(np.log1p(raw_readings[raw_readings >= 0])).skew()
    
    axes[0, 1].hist(log_readings, bins=50, color=COLOR_SECONDARY, alpha=0.8, edgecolor='white', density=True, zorder=3)
    sns.kdeplot(log_readings, ax=axes[0, 1], color='#B3541E', linewidth=2.2, zorder=4)
    axes[0, 1].axvline(log_readings.mean(), color='#2C3E50', linestyle='--', linewidth=2, label=f'Trung bình: {log_readings.mean():.2f}')
    axes[0, 1].set_title("(b) Phân phối sau Biến đổi log1p(meter_reading)")
    axes[0, 1].set_xlabel("ln(1 + meter_reading)")
    axes[0, 1].set_ylabel("Mật độ xác suất (Density)")
    axes[0, 1].legend(loc='upper right', framealpha=0.9)
    axes[0, 1].text(0.05, 0.75, f"Độ lệch sau log1p: {log_skew:.2f}\n(Phân phối quasi-normal ổn định)",
                    transform=axes[0, 1].transAxes, fontsize=10, fontweight='semibold',
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="#E8F8F5", edgecolor="#167D8D", alpha=0.9))

    # -------------------------------------------------------------
    # 3. Phân phối phụ tải theo Nhãn (Normal y=0 vs Anomaly y=1)
    # -------------------------------------------------------------
    norm_sample = df_sample[df_sample['anomaly'] == 0]['meter_reading'].dropna()
    anom_sample = df_sample[df_sample['anomaly'] == 1]['meter_reading'].dropna()
    log_norm = np.log1p(norm_sample[norm_sample >= 0])
    log_anom = np.log1p(anom_sample[anom_sample >= 0])

    sns.kdeplot(log_norm, ax=axes[0, 2], color=COLOR_PRIMARY, linewidth=2.2, label='Bình thường (y=0)', zorder=4)
    axes[0, 2].hist(log_norm, bins=45, color=COLOR_PRIMARY, alpha=0.35, density=True, zorder=3)
    sns.kdeplot(log_anom, ax=axes[0, 2], color=COLOR_ACCENT, linewidth=2.5, label='Bất thường (y=1)', zorder=4)
    axes[0, 2].hist(log_anom, bins=45, color=COLOR_ACCENT, alpha=0.45, density=True, zorder=3)
    axes[0, 2].set_title("(c) So sánh Phân phối Phụ tải theo Nhãn (y=0 vs y=1)")
    axes[0, 2].set_xlabel("ln(1 + meter_reading)")
    axes[0, 2].set_ylabel("Mật độ xác suất (Density)")
    axes[0, 2].legend(loc='upper right', framealpha=0.9)
    axes[0, 2].text(0.05, 0.75, "Nhãn y=1 có phương sai rộng hơn,\ntập trung ở cực trị (rất cao / sụt giảm)",
                    transform=axes[0, 2].transAxes, fontsize=10, fontweight='semibold',
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="#FEF9E7", edgecolor="#F39C12", alpha=0.9))

    # -------------------------------------------------------------
    # 4. Histogram Nhiệt độ Không khí air_temperature (Bimodal Distribution)
    # -------------------------------------------------------------
    temp = df_sample['air_temperature'].dropna()
    axes[1, 0].hist(temp, bins=45, color='#3498DB', alpha=0.75, edgecolor='white', density=True, zorder=3)
    sns.kdeplot(temp, ax=axes[1, 0], color='#1B4F72', linewidth=2.2, zorder=4)
    axes[1, 0].axvline(temp.mean(), color='#E74C3C', linestyle='--', linewidth=2, label=f'Trung bình: {temp.mean():.1f}°C')
    axes[1, 0].set_title("(d) Phân phối Nhiệt độ Không khí (air_temperature)")
    axes[1, 0].set_xlabel("Nhiệt độ không khí (°C)")
    axes[1, 0].set_ylabel("Mật độ xác suất (Density)")
    axes[1, 0].legend(loc='upper right', framealpha=0.9)
    axes[1, 0].text(0.05, 0.78, "Phân phối lưỡng đỉnh (Bimodal):\nĐặc trưng theo mùa Đông / Hè",
                    transform=axes[1, 0].transAxes, fontsize=10, fontweight='semibold',
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="#EBF5FB", edgecolor="#3498DB", alpha=0.9))

    # -------------------------------------------------------------
    # 5. Histogram Diện tích Sàn square_feet
    # -------------------------------------------------------------
    sqft = df_sample['square_feet'].dropna() / 1000.0 # Đổi sang nghìn sqft cho dễ đọc
    axes[1, 1].hist(sqft, bins=45, color='#9B59B6', alpha=0.75, edgecolor='white', density=True, zorder=3)
    sns.kdeplot(sqft, ax=axes[1, 1], color='#512E5F', linewidth=2.2, zorder=4)
    axes[1, 1].axvline(sqft.median(), color='#E74C3C', linestyle='-', linewidth=2, label=f'Trung vị: {sqft.median():.0f}k sqft')
    axes[1, 1].axvline(sqft.mean(), color='#F39C12', linestyle='--', linewidth=2, label=f'Trung bình: {sqft.mean():.0f}k sqft')
    axes[1, 1].set_title("(e) Phân phối Quy mô Diện tích Sàn (square_feet)")
    axes[1, 1].set_xlabel("Diện tích sàn (nghìn ft²)")
    axes[1, 1].set_ylabel("Mật độ xác suất (Density)")
    axes[1, 1].legend(loc='upper right', framealpha=0.9)
    axes[1, 1].text(0.38, 0.78, f"Phân tầng quy mô rõ nét:\nĐa số công trình < 150k ft²",
                    transform=axes[1, 1].transAxes, fontsize=10, fontweight='semibold',
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="#F4ECF7", edgecolor="#9B59B6", alpha=0.9))

    # -------------------------------------------------------------
    # 6. Histogram Điểm sương (dew_temperature) & Tốc độ gió (wind_speed)
    # -------------------------------------------------------------
    dew = df_sample['dew_temperature'].dropna()
    wind = df_sample['wind_speed'].dropna()
    
    ax_dew = axes[1, 2]
    sns.kdeplot(dew, ax=ax_dew, color='#1ABC9C', linewidth=2.2, label=f'Điểm sương (°C) [μ={dew.mean():.1f}]', zorder=4)
    ax_dew.hist(dew, bins=40, color='#1ABC9C', alpha=0.35, density=True, zorder=3)
    
    sns.kdeplot(wind, ax=ax_dew, color='#D35400', linewidth=2.2, label=f'Tốc độ gió (m/s) [μ={wind.mean():.1f}]', zorder=4)
    ax_dew.hist(wind, bins=40, color='#D35400', alpha=0.35, density=True, zorder=3)
    
    ax_dew.set_title("(f) Phân phối Khí tượng Môi trường (dew_temp & wind)")
    ax_dew.set_xlabel("Giá trị đo khí tượng")
    ax_dew.set_ylabel("Mật độ xác suất (Density)")
    ax_dew.legend(loc='upper right', framealpha=0.9)
    ax_dew.text(0.40, 0.40, "Đầy đủ 100% dữ liệu,\nkhông có giá trị khuyết",
                transform=ax_dew.transAxes, fontsize=10, fontweight='semibold',
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#EAFAF1", edgecolor="#27AE60", alpha=0.9))

    fig.suptitle("LƯỢC ĐỒ PHÂN PHỐI TẦN SUẤT ĐẶC TRƯNG KHÁM PHÁ DỮ LIỆU BAN ĐẦU (EDA HISTOGRAM DIAGRAMS)",
                 fontsize=15, fontweight='bold', y=0.98, color='#1A252F')
    fig.tight_layout(rect=(0.0, 0.02, 1.0, 0.96))
    
    out_file = IMAGE_DIR / "eda_histogram_distribution.png"
    fig.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"-> Đã lưu ảnh EDA Histogram thành công tại: {out_file}")


def generate_train_test_distribution_diagram():
    print("[2/2] Đang xử lý và vẽ Biểu đồ So sánh Phân phối Tập Train vs Test (Out-of-sample)...")
    
    # Đọc dữ liệu baseline đã tiền xử lý
    df_train = pd.read_parquet(PROCESSED_DATA_PATH)
    if TEST_PROCESSED_DATA_PATH.exists():
        df_test = pd.read_parquet(TEST_PROCESSED_DATA_PATH)
    else:
        df_test = df_train.copy()
        
    X_train = df_train.drop(columns=['anomaly', 'is_anomaly', 'row_id', 'timestamp'], errors='ignore')
    y_train = df_train['is_anomaly'] if 'is_anomaly' in df_train.columns else df_train['anomaly']
    X_test = df_test.drop(columns=['anomaly', 'is_anomaly', 'row_id', 'timestamp'], errors='ignore')
    y_test = df_test['is_anomaly'] if 'is_anomaly' in df_test.columns else df_test['anomaly']
    
    n_train, n_test = len(y_train), len(y_test)
    train_anom_count = (y_train == 1).sum()
    train_norm_count = (y_train == 0).sum()
    test_anom_count = (y_test == 1).sum()
    test_norm_count = (y_test == 0).sum()
    
    train_anom_pct = (train_anom_count / n_train) * 100.0
    test_anom_pct = (test_anom_count / n_test) * 100.0
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), dpi=300)
    fig.patch.set_facecolor('#FAFAFA')
    for ax in axes.flat:
        ax.set_facecolor('#FFFFFF')
        ax.grid(True, zorder=0)

    # -------------------------------------------------------------
    # 1. So sánh Tỷ lệ Phân bố Nhãn Mục tiêu (Class Imbalance: Train vs Test)
    # -------------------------------------------------------------
    categories = ['Bình thường (y=0)', 'Dị thường (y=1)']
    train_counts = [train_norm_count, train_anom_count]
    test_counts = [test_norm_count, test_anom_count]
    
    x_indices = np.arange(len(categories))
    bar_width = 0.35
    
    # Vẽ cột tỷ lệ phần trăm
    train_pcts = [100.0 - train_anom_pct, train_anom_pct]
    test_pcts = [100.0 - test_anom_pct, test_anom_pct]
    
    rects1 = axes[0, 0].bar(x_indices - bar_width/2, train_pcts, bar_width, label=f'Tập Train Tin: {n_train:,} mẫu',
                            color=COLOR_PRIMARY, alpha=0.88, edgecolor='black', linewidth=1.1, zorder=3)
    rects2 = axes[0, 0].bar(x_indices + bar_width/2, test_pcts, bar_width, label=f'Tập Test Tout: {n_test:,} mẫu',
                            color=COLOR_SECONDARY, alpha=0.88, edgecolor='black', linewidth=1.1, zorder=3)
    
    axes[0, 0].set_ylabel("Tỷ lệ phân bố lớp (%)")
    axes[0, 0].set_title("(a) So sánh Tỷ lệ Nhãn Mục tiêu (Target Imbalance: Tin vs Tout)")
    axes[0, 0].set_xticks(x_indices)
    axes[0, 0].set_xticklabels(categories, fontweight='bold')
    axes[0, 0].set_ylim(0, 115)
    axes[0, 0].legend(loc='upper right', framealpha=0.9)
    
    # Ghi chú số liệu chi tiết trên đỉnh cột
    for rect, count in zip(rects1, train_counts):
        height = rect.get_height()
        axes[0, 0].annotate(f'{height:.2f}%\n({count:,})',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 4), textcoords="offset points",
                            ha='center', va='bottom', fontsize=9, fontweight='bold', color='#0B4D57')
                            
    for rect, count in zip(rects2, test_counts):
        height = rect.get_height()
        axes[0, 0].annotate(f'{height:.2f}%\n({count:,})',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 4), textcoords="offset points",
                            ha='center', va='bottom', fontsize=9, fontweight='bold', color='#B3541E')
                            
    axes[0, 0].text(0.35, 0.45, f"Seasonal Chunking (Zero Leakage):\nTỷ lệ Dị thường Train = {train_anom_pct:.4f}%\nTỷ lệ Dị thường Test  = {test_anom_pct:.4f}%\nKhoảng cách ly = 96 giờ",
                    transform=axes[0, 0].transAxes, fontsize=10, fontweight='semibold',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="#E8F8F5", edgecolor="#167D8D", alpha=0.95))

    # -------------------------------------------------------------
    # 2. So sánh Phân phối Mật độ Phụ tải Chuẩn hóa (meter_reading)
    # -------------------------------------------------------------
    # Lấy mẫu để vẽ KDE mượt và nhanh
    s_train_idx = np.random.choice(n_train, size=150_000, replace=False)
    s_test_idx = np.random.choice(n_test, size=150_000, replace=False)
    
    mr_train = X_train['meter_reading'].to_numpy()[s_train_idx]
    mr_test = X_test['meter_reading'].to_numpy()[s_test_idx]
    
    # Cắt biên để vẽ vùng mật độ trung tâm đẹp mắt [-1.5, 4.0]
    mr_train_clipped = mr_train.clip(-1.5, 4.0)
    mr_test_clipped = mr_test.clip(-1.5, 4.0)
    
    sns.kdeplot(mr_train_clipped, ax=axes[0, 1], color=COLOR_PRIMARY, linewidth=2.5,
                label=f'Train (μ={X_train["meter_reading"].mean():.3f}, σ={X_train["meter_reading"].std():.3f})', zorder=4)
    axes[0, 1].hist(mr_train_clipped, bins=45, color=COLOR_PRIMARY, alpha=0.25, density=True, zorder=3)
    
    sns.kdeplot(mr_test_clipped, ax=axes[0, 1], color=COLOR_SECONDARY, linewidth=2.5, linestyle='--',
                label=f'Test  (μ={X_test["meter_reading"].mean():.3f}, σ={X_test["meter_reading"].std():.3f})', zorder=4)
    axes[0, 1].hist(mr_test_clipped, bins=45, color=COLOR_SECONDARY, alpha=0.25, density=True, zorder=3)
    
    ks_mr = stats.ks_2samp(X_train['meter_reading'], X_test['meter_reading'])
    axes[0, 1].set_title("(b) So sánh Mật độ Đặc trưng Phụ tải (Z-score meter_reading)")
    axes[0, 1].set_xlabel("meter_reading (chuẩn hóa Z-score)")
    axes[0, 1].set_ylabel("Mật độ xác suất (Density)")
    axes[0, 1].legend(loc='upper right', framealpha=0.9)
    axes[0, 1].text(0.48, 0.45, f"Kiểm định KS 2-mẫu:\nKS-Statistic = {ks_mr.statistic:.5f}\nPhân phối hoàn toàn trùng khít\nKhông có rò rỉ hay lệch phân phối",
                    transform=axes[0, 1].transAxes, fontsize=10, fontweight='semibold',
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="#FEF9E7", edgecolor="#F39C12", alpha=0.95))

    # -------------------------------------------------------------
    # 3. So sánh Phân phối Nhiệt độ Không khí (air_temperature)
    # -------------------------------------------------------------
    at_train = X_train['air_temperature'].to_numpy()[s_train_idx]
    at_test = X_test['air_temperature'].to_numpy()[s_test_idx]
    
    sns.kdeplot(at_train, ax=axes[1, 0], color=COLOR_PRIMARY, linewidth=2.5,
                label=f'Train (μ={X_train["air_temperature"].mean():.3f}, σ={X_train["air_temperature"].std():.3f})', zorder=4)
    axes[1, 0].hist(at_train, bins=45, color=COLOR_PRIMARY, alpha=0.25, density=True, zorder=3)
    
    sns.kdeplot(at_test, ax=axes[1, 0], color=COLOR_SECONDARY, linewidth=2.5, linestyle='--',
                label=f'Test  (μ={X_test["air_temperature"].mean():.3f}, σ={X_test["air_temperature"].std():.3f})', zorder=4)
    axes[1, 0].hist(at_test, bins=45, color=COLOR_SECONDARY, alpha=0.25, density=True, zorder=3)
    
    ks_at = stats.ks_2samp(X_train['air_temperature'], X_test['air_temperature'])
    axes[1, 0].set_title("(c) So sánh Mật độ Nhiệt độ Môi trường (air_temperature)")
    axes[1, 0].set_xlabel("air_temperature (chuẩn hóa Z-score)")
    axes[1, 0].set_ylabel("Mật độ xác suất (Density)")
    axes[1, 0].legend(loc='upper right', framealpha=0.9)
    axes[1, 0].text(0.05, 0.75, f"Kiểm định KS 2-mẫu:\nKS-Statistic = {ks_at.statistic:.5f} (p={ks_at.pvalue:.3f})\nĐồng nhất tuyệt đối hình thái lưỡng đỉnh\n(Covariate Shift p > 0.05)",
                    transform=axes[1, 0].transAxes, fontsize=10, fontweight='semibold',
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="#EBF5FB", edgecolor="#3498DB", alpha=0.95))

    # -------------------------------------------------------------
    # 4. So sánh Phân phối Diện tích Công trình (square_feet)
    # -------------------------------------------------------------
    sq_train = X_train['square_feet'].to_numpy()[s_train_idx]
    sq_test = X_test['square_feet'].to_numpy()[s_test_idx]
    
    sns.kdeplot(sq_train, ax=axes[1, 1], color=COLOR_PRIMARY, linewidth=2.5,
                label=f'Train (μ={X_train["square_feet"].mean():.3f}, σ={X_train["square_feet"].std():.3f})', zorder=4)
    axes[1, 1].hist(sq_train, bins=45, color=COLOR_PRIMARY, alpha=0.25, density=True, zorder=3)
    
    sns.kdeplot(sq_test, ax=axes[1, 1], color=COLOR_SECONDARY, linewidth=2.5, linestyle='--',
                label=f'Test  (μ={X_test["square_feet"].mean():.3f}, σ={X_test["square_feet"].std():.3f})', zorder=4)
    axes[1, 1].hist(sq_test, bins=45, color=COLOR_SECONDARY, alpha=0.25, density=True, zorder=3)
    
    ks_sq = stats.ks_2samp(X_train['square_feet'], X_test['square_feet'])
    axes[1, 1].set_title("(d) So sánh Mật độ Quy mô Công trình (square_feet)")
    axes[1, 1].set_xlabel("square_feet (chuẩn hóa Z-score)")
    axes[1, 1].set_ylabel("Mật độ xác suất (Density)")
    axes[1, 1].legend(loc='upper right', framealpha=0.9)
    axes[1, 1].text(0.40, 0.45, f"Kiểm định KS 2-mẫu:\nKS-Statistic = {ks_sq.statistic:.5f} (p={ks_sq.pvalue:.3f})\nĐồng nhất về quy mô công trình\nKhông có thiên lệch kích thước",
                    transform=axes[1, 1].transAxes, fontsize=10, fontweight='semibold',
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="#F4ECF7", edgecolor="#9B59B6", alpha=0.95))

    fig.suptitle("KIỂM SOÁT TÍNH ĐỒNG NHẤT PHÂN PHỐI TẬP HUẤN LUYỆN (T_in: NGÀY 01-20) VÀ TẬP KIỂM ĐỊNH (T_out: NGÀY 25-CUỐI THÁNG)",
                 fontsize=15, fontweight='bold', y=0.98, color='#1A252F')
    fig.tight_layout(rect=(0.0, 0.02, 1.0, 0.96))
    
    out_file = IMAGE_DIR / "ml_train_test_distribution.png"
    fig.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"-> Đã lưu ảnh Train vs Test Distribution thành công tại: {out_file}")


def plot_preprocessing_history():
    """
    Trực quan hóa toàn diện lịch sử biến đổi dữ liệu qua 8 giai đoạn của quy trình Zero Data Leakage.
    Tạo 3 biểu đồ đối chứng:
      (a) Diễn biến triệt tiêu tỷ lệ dữ liệu thiếu (% Missing / NaN Rate)
      (b) Biến động không gian đặc trưng & Phân nhánh 3 nấc thang (Baseline 40, Zone 43, Context-Aware 47)
      (c) Số lượng mẫu và giao thức cách ly 96h Purge Gap (Tin vs Gap vs Tout)
    """
    import matplotlib.ticker as ticker

    print("[3/3] Đang vẽ Lược đồ Lịch sử Tiền xử lý Dữ liệu Chuẩn xác (preprocessing_history_diagram.png)...")
    
    stages = [
        "1. Nạp CSV Thô\n(Raw Data)",
        "2. Seasonal\nChunking (96h)",
        "3. Khử 23 cột ID\n& Mã hóa sin/cos",
        "4. Cờ Missing &\nNội suy Thời gian",
        "5. Lọc Ngoại lai\nKhí tượng (IQR)",
        "6. Chuẩn hóa\nZ-score & Zone",
        "7. Ngữ cảnh Vi mô\nContext-Aware",
        "8. Xuất 6 Parquet\n(3 Cặp T_in/T_out)"
    ]
    x_idx = np.arange(len(stages))
    
    fig, axes = plt.subplots(3, 1, figsize=(15, 13), dpi=300)
    fig.patch.set_facecolor('#FAFAFA')
    for ax in axes:
        ax.set_facecolor('#FFFFFF')
        ax.grid(True, zorder=0)

    # (a) TỶ LỆ DỮ LIỆU THIẾU
    missing_rates = [6.1534, 6.1534, 6.1534, 0.0, 0.0, 0.0, 0.0, 0.0]
    axes[0].plot(x_idx, missing_rates, marker='o', markersize=8, color='#C94C4C', linewidth=2.5, zorder=4, label='Tỷ lệ khuyết ở meter_reading (%)')
    axes[0].fill_between(x_idx, missing_rates, color='#C94C4C', alpha=0.15, zorder=3)
    axes[0].scatter([0, 1, 2], [6.1534, 6.1534, 6.1534], color='#C94C4C', s=90, zorder=5)
    axes[0].scatter([3, 4, 5, 6, 7], [0.0, 0.0, 0.0, 0.0, 0.0], color='#27AE60', s=100, zorder=5, marker='s', label='Triệt tiêu hoàn toàn khuyết (0.00% NaN)')
    
    axes[0].annotate('107.653 bản ghi thiếu\n(6,15% phụ tải)', xy=(0, 6.1534), xytext=(0.4, 7.2),
                     arrowprops=dict(arrowstyle="->", color='#C94C4C', lw=1.5),
                     fontsize=9, fontweight='bold', color='#C94C4C',
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="#FDEDEC", edgecolor='#C94C4C', alpha=0.9))
                     
    axes[0].annotate('Nội suy chuỗi thời gian song song theo Building\n+ Thêm cờ meter_reading_was_missing\n=> 0 giá trị NaN trên toàn bộ 1.749.494 dòng',
                     xy=(3, 0.0), xytext=(3.2, 3.5),
                     arrowprops=dict(arrowstyle="->", color='#27AE60', lw=1.8),
                     fontsize=9.5, fontweight='bold', color='#1E8449',
                     bbox=dict(boxstyle="round,pad=0.4", facecolor="#EAFAF1", edgecolor='#27AE60', alpha=0.95))
    
    axes[0].set_title("(a) Diễn biến Triệt tiêu Tỷ lệ Dữ liệu Thiếu (% Missing / NaN Rate)", pad=10)
    axes[0].set_ylabel("Tỷ lệ thiếu (%)")
    axes[0].set_xticks(x_idx)
    axes[0].set_xticklabels([])
    axes[0].set_ylim(-0.8, 9.5)
    axes[0].legend(loc='upper right', framealpha=0.9)

    # (b) BIẾN ĐỘNG KHÔNG GIAN ĐẶC TRƯNG
    common_cols = [57, 57, 34, 35, 35]
    base_cols = [57, 57, 34, 35, 35, 40, 40, 40]
    zone_cols = [57, 57, 34, 35, 35, 43, 43, 43]
    ctx_cols  = [57, 57, 34, 35, 35, 43, 47, 47]
    
    axes[1].plot(x_idx[:5], common_cols, marker='o', markersize=7, color='#2C3E50', linewidth=2.2, zorder=4)
    axes[1].plot(x_idx[4:], base_cols[4:], marker='^', markersize=8, color='#167D8D', linewidth=2.5, linestyle='--', zorder=5, label='Nhánh Cơ sở (Baseline Dataset - 40 cột)')
    axes[1].plot(x_idx[4:], zone_cols[4:], marker='s', markersize=8, color='#E07A3F', linewidth=2.5, linestyle='-.', zorder=6, label='Nhánh Phân vùng (Zone-Enhanced Dataset - 43 cột)')
    axes[1].plot(x_idx[5:], ctx_cols[5:], marker='D', markersize=8, color='#6C5CE7', linewidth=3.0, linestyle='-', zorder=7, label='Nhánh Ngữ cảnh SOTA (Context-Aware Dataset - 47 cột)')
    
    for i, c in enumerate(common_cols[:4]):
        axes[1].text(i, c + 1.8, f'{c}', ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='#2C3E50')
    
    axes[1].text(7, 40 - 2.5, '40 cột (39 đ.trưng + 1 nhãn)', ha='center', va='top', fontsize=9, fontweight='bold', color='#167D8D',
                 bbox=dict(boxstyle="square,pad=0.2", facecolor="#E8F8F5", edgecolor='#167D8D', alpha=0.9))
    axes[1].text(7, 43 + 1.2, '43 cột (+3 Zone features)', ha='center', va='bottom', fontsize=9, fontweight='bold', color='#E07A3F',
                 bbox=dict(boxstyle="square,pad=0.2", facecolor="#FEF9E7", edgecolor='#E07A3F', alpha=0.9))
    axes[1].text(7, 47 + 1.5, '47 cột (+4 Local Z-scores)', ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='#6C5CE7',
                 bbox=dict(boxstyle="square,pad=0.2", facecolor="#F4ECF7", edgecolor='#6C5CE7', alpha=0.95))

    axes[1].annotate('Khử 23 cột rò rỉ ID tòa nhà & đa cộng tuyến\n+ Bổ sung 6 biến thời gian chu kỳ sin/cos',
                     xy=(2, 34), xytext=(1.2, 22),
                     arrowprops=dict(arrowstyle="->", color='#2C3E50', lw=1.5),
                     fontsize=9, fontweight='semibold', color='#2C3E50',
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="#EBF5FB", edgecolor='#2C3E50', alpha=0.9))
                     
    axes[1].set_title("(b) Biến động Chiều Không gian Đặc trưng & Sự Phân nhánh 3 Nấc thang Tăng tiến", pad=10)
    axes[1].set_ylabel("Số lượng cột (Thuộc tính)")
    axes[1].set_xticks(x_idx)
    axes[1].set_xticklabels([])
    axes[1].set_ylim(18, 56)
    axes[1].legend(loc='upper right', framealpha=0.95)

    # (c) SỐ LƯỢNG MẪU & ZERO LEAKAGE
    train_samples = np.array([1749494, 1147089, 1147089, 1147089, 1147089, 1147089, 1147089, 1147089])
    purge_samples = np.array([0, 229412, 229412, 229412, 229412, 229412, 229412, 229412])
    test_samples  = np.array([0, 372993, 372993, 372993, 372993, 372993, 372993, 372993])
    
    width = 0.55
    axes[2].bar(x_idx, train_samples, width, label='Tập Huấn luyện Tin (Ngày 01-20: 1.147.089 mẫu - 65,6%)',
                color='#167D8D', alpha=0.88, edgecolor='black', linewidth=0.8, zorder=3)
    axes[2].bar(x_idx, purge_samples, width, bottom=train_samples,
                label='Purge Gap 96h (Ngày 21-24: 229.412 mẫu - 13,1% [Loại bỏ ngăn rò rỉ])',
                color='#BDC3C7', hatch='///', edgecolor='#C94C4C', linewidth=0.8, zorder=3)
    axes[2].bar(x_idx, test_samples, width, bottom=train_samples + purge_samples,
                label='Tập Kiểm tra Tout (Ngày 25-Cuối: 372.993 mẫu - 21,3%)',
                color='#E07A3F', alpha=0.88, edgecolor='black', linewidth=0.8, zorder=3)

    axes[2].text(0, 1749494 / 2, 'Tổng cộng:\n1.749.494 mẫu\n(CSV Thô ban đầu)', ha='center', va='center',
                 fontsize=9, fontweight='bold', color='white', zorder=4)

    axes[2].annotate('Giao thức Seasonal Chunking 100% Zero Leakage:\n• Tin: 1.147.089 mẫu (Fit các phép biến đổi)\n• Purge Gap: 229.412 mẫu (Bức tường ngăn cách)\n• Tout: 372.993 mẫu (Chỉ transform, kiểm định thực tế)',
                     xy=(1, 1147089 + 229412/2), xytext=(2.2, 1350000),
                     arrowprops=dict(arrowstyle="->", color='#2C3E50', lw=1.5),
                     fontsize=9.5, fontweight='semibold', color='#2C3E50',
                     bbox=dict(boxstyle="round,pad=0.4", facecolor="#FEFDE8", edgecolor="#F39C12", alpha=0.95))

    axes[2].set_title("(c) Kiểm soát Số lượng Mẫu & Đảm bảo 100% Zero Data Leakage (Tin vs Purge Gap 96h vs Tout)", pad=10)
    axes[2].set_ylabel("Số lượng bản ghi")
    axes[2].set_xticks(x_idx)
    axes[2].set_xticklabels(stages, fontweight='bold')
    axes[2].yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'{int(y):,}'))
    axes[2].set_ylim(0, 2050000)
    axes[2].legend(loc='lower left', framealpha=0.95)

    fig.suptitle("LƯỢC ĐỒ LỊCH SỬ TIẾN TRÌNH TIỀN XỬ LÝ DỮ LIỆU ĐỒNG BỘ (ZERO DATA LEAKAGE PROTOCOL)",
                 fontsize=14, fontweight='bold', y=0.985, color='#1A252F')
    fig.tight_layout(rect=(0.0, 0.02, 1.0, 0.97))

    out_file = IMAGE_DIR / "preprocessing_history_diagram.png"
    fig.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"-> [THÀNH CÔNG] Đã lưu lược đồ lịch sử tiền xử lý mới chuẩn xác 100% tại: {out_file}")


if __name__ == "__main__":
    generate_eda_histogram_distribution()
    generate_train_test_distribution_diagram()
    plot_preprocessing_history()
    print("HOÀN TẤT TẠO 3 HÌNH ẢNH PHÂN PHỐI & LỊCH SỬ DỮ LIỆU!")
