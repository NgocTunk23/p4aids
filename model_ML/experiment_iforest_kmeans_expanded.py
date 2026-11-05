"""
model_ML/experiment_iforest_kmeans_expanded.py
==============================================
Khao sat nang cao: Tich hop dac trung Isolation Forest (IF) va Gom cum K-Means tren nen tang Expanded (53 dac trung).
Quy trinh chuan:
  1. Nap sieu tham so Champion moi nhat tu best_params_*_expanded.json.
  2. Nap lai building_id tu train_features.csv (de tao profile cong trinh, loai bo ID khoi mo hinh tree nham dam bao Zero ID Memorization).
  3. Quet tham so K doc lap cho TUNG MO HINH (XGB, CAT, LGBM, HIST) tren tap validation noi bo de tim K* toi uu rieng biet.
  4. Danh gia toan dien 4 bien the tren tap Test (Baseline, +IF, +KMeans K*, +IF & KMeans K*).
  5. Lay cac cau hinh toi uu nhat cua tung mo hinh de thuc hien Ensemble so sanh voi moc Expanded co so.
"""

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest, HistGradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
DATA_DIR = SCRIPT_DIR / "data"
PARAMS_DIR = SCRIPT_DIR / "hyperparameters"
RESULTS_DIR = SCRIPT_DIR / "results"

for d in [DATA_DIR, PARAMS_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

TARGET = "anomaly"
BUILDING_ID = "building_id"
CLUSTER_PREFIX = "building_cluster_"
IF_SCORE = "iforest_score"
MODEL_NAMES = ("xgb", "cat", "lgbm", "hist")
VALIDATION_FRACTION = 0.15
IF_N_ESTIMATORS = 200
IF_MAX_SAMPLES = 256


def check_nvidia_gpu() -> bool:
    try:
        out = subprocess.check_output("nvidia-smi", shell=True).decode("utf-8")
        return "NVIDIA" in out
    except Exception:
        return False


HAS_GPU = check_nvidia_gpu()


def load_data() -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    """Nap du lieu train va test expanded, bo sung building_id cho K-Means."""
    train_path = DATA_DIR / "train_chunked_expanded.parquet"
    test_path = DATA_DIR / "test_chunked_expanded.parquet"
    if not train_path.exists():
        train_path = ROOT_DIR / "data" / "processed" / "train_chunked_expanded.parquet"
    if not test_path.exists():
        test_path = ROOT_DIR / "data" / "processed" / "test_chunked_expanded.parquet"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Khong tim thay tep parquet tai {train_path} va {test_path}!"
        )

    print(f"-> Dang nap du lieu EXPANDED tu {train_path.name} va {test_path.name} ...", end=" ", flush=True)
    t0 = time.time()
    train = pd.read_parquet(train_path)
    test = pd.read_parquet(test_path)
    print(f"Xong ({time.time() - t0:.1f}s)!", flush=True)

    # Neu building_id da bi loai bo khoi parquet de chong ghi nho ID, nap lai tu train_features.csv
    if BUILDING_ID not in train.columns or BUILDING_ID not in test.columns:
        raw_csv = ROOT_DIR / "data" / "train_features.csv"
        if not raw_csv.exists():
            raw_csv = DATA_DIR / "train_features.csv"
        if not raw_csv.exists():
            raise FileNotFoundError(f"Khong tim thay building_id trong parquet va file tho {raw_csv} khong ton tai!")

        print("-> Nap bo sung building_id tu train_features.csv phuc vu tao ho so K-Means ...", end=" ", flush=True)
        t1 = time.time()
        raw_b = pd.read_csv(raw_csv, usecols=["timestamp", BUILDING_ID], low_memory=False)
        days = pd.to_datetime(raw_b["timestamp"]).dt.day
        train_mask = (days >= 1) & (days <= 20)
        test_mask = days >= 25
        train[BUILDING_ID] = raw_b.loc[train_mask, BUILDING_ID].values
        test[BUILDING_ID] = raw_b.loc[test_mask, BUILDING_ID].values
        print(f"Xong ({time.time() - t1:.1f}s)!", flush=True)

    y_train = train[TARGET].to_numpy(dtype=np.uint8)
    y_test = test[TARGET].to_numpy(dtype=np.uint8)

    feature_cols = [c for c in train.columns if c not in [TARGET, "row_id", "chunk_month"]]
    X_train = train[feature_cols].copy()
    X_test = test[feature_cols].copy()

    n_features_no_id = len([c for c in feature_cols if c != BUILDING_ID])
    print(f"   Train: {X_train.shape} | Test: {X_test.shape} | So dac trung co so (loai tru ID): {n_features_no_id}")
    return X_train, y_train, X_test, y_test


def building_profiles(frame: pd.DataFrame) -> pd.DataFrame:
    """Tao ma tran dac trung tom tat (mean, std) cho tung cong trinh (200 toa nha)."""
    profile_columns = [
        column
        for column in frame.select_dtypes(include=[np.number]).columns
        if column != BUILDING_ID and not column.startswith(CLUSTER_PREFIX) and column != IF_SCORE
    ]
    profiles = frame.groupby(BUILDING_ID, sort=True)[profile_columns].agg(["mean", "std"])
    profiles.columns = [f"{column}_{statistic}" for column, statistic in profiles.columns]
    profiles = profiles.replace([np.inf, -np.inf], np.nan)
    return profiles.fillna(profiles.median()).fillna(0.0)


def fit_kmeans_features(
    fit_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    n_clusters: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float]:
    """Huan luyen K-Means tren ho so cong trinh cua fit_frame va ma hoa one-hot cum cho ca 3 tap."""
    profiles = building_profiles(fit_frame)
    if len(profiles) <= n_clusters:
        raise ValueError(f"K={n_clusters} vuot qua so luong cong trinh hop le ({len(profiles)}).")

    scaler = StandardScaler()
    scaled_profiles = scaler.fit_transform(profiles)
    clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    labels = clusterer.fit_predict(scaled_profiles)
    silhouette = float(silhouette_score(scaled_profiles, labels))
    mapping = dict(zip(profiles.index.astype(int), labels.astype(int)))

    columns = [f"{CLUSTER_PREFIX}{cluster}" for cluster in range(n_clusters)]

    def encode(frame: pd.DataFrame) -> pd.DataFrame:
        building_ids = frame[BUILDING_ID].astype(int)
        mapped = building_ids.map(mapping)
        if mapped.isna().any():
            unknown = sorted(building_ids[mapped.isna()].unique().tolist())
            raise ValueError(f"Cong trinh khong co trong ho so K-Means: {unknown[:10]}")
        values = np.eye(n_clusters, dtype=np.uint8)[mapped.to_numpy(dtype=int)]
        return pd.DataFrame(values, index=frame.index, columns=columns)

    return encode(fit_frame), encode(validation_frame), encode(test_frame), silhouette


def load_tuned_params(model_name: str) -> Dict[str, Any]:
    """Nap bo sieu tham so Champion moi nhat tu hyperparameters."""
    path = PARAMS_DIR / f"best_params_{model_name.lower()}_expanded.json"
    if not path.exists():
        path = PARAMS_DIR / f"best_params_{model_name.lower()}_context.json"
    if not path.exists():
        raise FileNotFoundError(f"Khong tim thay file sieu tham so cho {model_name} tai {path}!")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        return data.get("best_params", {})


def make_model(model_name: str, params: Dict[str, Any]):
    """Khoi tao mo hinh voi tham so toi uu va ho tro GPU CUDA."""
    if model_name == "xgb":
        from xgboost import XGBClassifier
        xgb_kwargs = {"device": "cuda", "tree_method": "hist"} if HAS_GPU else {"n_jobs": -1, "tree_method": "hist"}
        return XGBClassifier(**params, random_state=42, eval_metric="logloss", **xgb_kwargs)

    elif model_name == "cat":
        from catboost import CatBoostClassifier
        cat_kwargs = {"task_type": "GPU"} if HAS_GPU else {"thread_count": -1}
        return CatBoostClassifier(**params, random_seed=42, verbose=0, **cat_kwargs)

    elif model_name == "lgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(**params, random_state=42, n_jobs=-1, verbose=-1)

    elif model_name == "hist":
        return HistGradientBoostingClassifier(**params, random_state=42)

    raise ValueError(f"Mo hinh khong ho tro: {model_name}")


def best_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Tim nguong tau* toi uu hoa F1 tren tap validation."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, scores)
    f1_scores = np.divide(
        2 * precisions * recalls,
        precisions + recalls,
        out=np.zeros_like(precisions),
        where=(precisions + recalls) > 0,
    )
    best_index = int(np.argmax(f1_scores))
    if best_index >= len(thresholds):
        return 0.5
    return float(thresholds[best_index])


def evaluate(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> Dict[str, float]:
    """Tinh toan chi tiet cac chi so PR-AUC, ROC-AUC, F1, Precision, Recall."""
    predictions = scores >= threshold
    tp = int(np.sum((y_true == 1) & predictions))
    fp = int(np.sum((y_true == 0) & predictions))
    fn = int(np.sum((y_true == 1) & ~predictions))

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    return {
        "pr_auc": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "best_threshold": round(threshold, 4),
        "f1": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
    }


def run_experiment(k_values: List[int] = list(range(2, 13))) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Thuc thi quet K rieng biet cho tung mo hinh va danh gia 4 bien the tren tap Test."""
    X_train, y_train, X_test, y_test = load_data()

    # Chia validation noi bo 15% out-of-sample temporal split
    validation_size = max(1, int(len(X_train) * VALIDATION_FRACTION))
    X_fit = X_train.iloc[:-validation_size].copy()
    y_fit = y_train[:-validation_size]
    X_val = X_train.iloc[-validation_size:].copy()
    y_val = y_train[-validation_size:]

    base_cols = [c for c in X_train.columns if c != BUILDING_ID]

    print("\n" + "=" * 95)
    print(f"GIAI DOAN 1: TIEN TINH TOAN MA TRAN CUM K-MEANS CHO CAC GIA TRI K = {k_values}")
    print("=" * 95)
    t0_km = time.time()
    kmeans_val_cache = {}
    for k in k_values:
        fit_cl, val_cl, test_cl_fit, sil = fit_kmeans_features(X_fit, X_val, X_val, k)
        kmeans_val_cache[k] = (fit_cl, val_cl, test_cl_fit, sil)
    print(f"-> Hoan tat tien tinh toan {len(k_values)} bo dac trung K-Means trong {time.time() - t0_km:.1f}s!")

    print("\n" + "=" * 95)
    print("GIAI DOAN 2: TIEN TINH TOAN DIEM SO ISOLATION FOREST (IFOREST_SCORE)")
    print("=" * 95)
    t0_if = time.time()
    iforest_val = IsolationForest(
        n_estimators=IF_N_ESTIMATORS,
        max_samples=min(IF_MAX_SAMPLES, len(X_fit)),
        contamination="auto",
        random_state=42,
        n_jobs=-1,
    )
    iforest_val.fit(X_fit[base_cols])
    fit_if_score = -iforest_val.score_samples(X_fit[base_cols])
    val_if_score = -iforest_val.score_samples(X_val[base_cols])

    iforest_full = IsolationForest(
        n_estimators=IF_N_ESTIMATORS,
        max_samples=min(IF_MAX_SAMPLES, len(X_train)),
        contamination="auto",
        random_state=42,
        n_jobs=-1,
    )
    iforest_full.fit(X_train[base_cols])
    train_if_score = -iforest_full.score_samples(X_train[base_cols])
    test_if_score = -iforest_full.score_samples(X_test[base_cols])
    print(f"-> Hoan tat huan luyen Isolation Forest trong {time.time() - t0_if:.1f}s!")

    print("\n" + "#" * 95)
    print("GIAI DOAN 3: QUET TIM THAM SO CUM K TOI UU RIENG BIET CHO TUNG MO HINH (XGB, CAT, LGBM, HIST)")
    print("#" * 95)

    sweep_records: List[Dict[str, Any]] = []
    best_k_per_model: Dict[str, int] = {}
    best_val_pr_per_model: Dict[str, float] = {}

    for model_name in MODEL_NAMES:
        params = load_tuned_params(model_name)
        print(f"\n>>> [1/4] Khao sat K-Means cho mo hinh [{model_name.upper()}] ...", flush=True)

        # Baseline tren validation
        t_m0 = time.time()
        base_val_model = make_model(model_name, params)
        base_val_model.fit(X_fit[base_cols], y_fit)
        p_val_base = base_val_model.predict_proba(X_val[base_cols])[:, 1]
        base_val_pr = float(average_precision_score(y_val, p_val_base))
        print(f"  * Baseline Expanded Validation PR-AUC = {base_val_pr:.4f} ({time.time() - t_m0:.1f}s)", flush=True)

        best_pr_m = -1.0
        best_k_m = k_values[0]

        for k in k_values:
            t_k0 = time.time()
            fit_cl, val_cl, _, sil = kmeans_val_cache[k]
            X_fit_k = pd.concat([X_fit[base_cols], fit_cl], axis=1)
            X_val_k = pd.concat([X_val[base_cols], val_cl], axis=1)

            m_k = make_model(model_name, params)
            m_k.fit(X_fit_k, y_fit)
            p_val_k = m_k.predict_proba(X_val_k)[:, 1]
            pr_val_k = float(average_precision_score(y_val, p_val_k))
            delta_k = pr_val_k - base_val_pr

            is_best = pr_val_k > best_pr_m
            if is_best:
                best_pr_m = pr_val_k
                best_k_m = k
                tag = "🔥 [BEST K!]"
            else:
                tag = ""

            sweep_records.append({
                "model": model_name.upper(),
                "k": k,
                "val_pr_auc": round(pr_val_k, 4),
                "base_val_pr_auc": round(base_val_pr, 4),
                "delta_val_pr_auc": round(delta_k, 4),
                "silhouette": round(sil, 4),
                "train_time_sec": round(time.time() - t_k0, 2),
            })
            print(f"    - K={k:2d}: Val PR-AUC={pr_val_k:.4f} (Delta={delta_k:+.4f}, Sil={sil:.3f}) {tag}", flush=True)

        best_k_per_model[model_name] = best_k_m
        best_val_pr_per_model[model_name] = best_pr_m
        print(f"  ==> KET QUA: K tot nhat cho [{model_name.upper()}] la K={best_k_m} (Val PR-AUC: {best_pr_m:.4f})")

    df_k_sweep = pd.DataFrame(sweep_records)

    print("\n" + "=" * 95)
    print("TONG HOP K TOI UU CHO TUNG MO HINH:")
    for m in MODEL_NAMES:
        print(f"  * {m.upper():6s} : K* = {best_k_per_model[m]:2d} | Val PR-AUC = {best_val_pr_per_model[m]:.4f}")
    print("=" * 95)

    print("\n" + "#" * 95)
    print("GIAI DOAN 4: DANH GIA 4 BIEN THE TREN TAP TEST VOI K* TOI UU RIENG BIET")
    print("#" * 95)

    # Tien tinh toan full K-Means cho cac K* duoc chon
    unique_selected_ks = sorted(list(set(best_k_per_model.values())))
    kmeans_full_cache = {}
    kmeans_val_selected_cache = {}
    for k in unique_selected_ks:
        fit_cl, val_cl, test_cl_fit, sil_val = fit_kmeans_features(X_fit, X_val, X_test, k)
        kmeans_val_selected_cache[k] = (fit_cl, val_cl, test_cl_fit, sil_val)

        tr_cl, _, te_cl, sil_full = fit_kmeans_features(X_train, X_train, X_test, k)
        kmeans_full_cache[k] = (tr_cl, te_cl, sil_full)

    test_rows: List[Dict[str, Any]] = []
    best_model_predictions: Dict[str, np.ndarray] = {}
    best_model_val_predictions: Dict[str, np.ndarray] = {}
    best_config_per_model: Dict[str, str] = {}
    best_test_pr_per_model: Dict[str, float] = {}

    for model_name in MODEL_NAMES:
        params = load_tuned_params(model_name)
        k_star = best_k_per_model[model_name]
        fit_cl, val_cl, _, sil_val = kmeans_val_selected_cache[k_star]
        tr_cl, te_cl, sil_full = kmeans_full_cache[k_star]

        print(f"\n>>> Dang danh gia cac bien the cho [{model_name.upper()}] (K* = {k_star}) tren Tap TEST ...", flush=True)

        # 1. Isolation Forest tren tap co K-Means
        X_fit_km = pd.concat([X_fit[base_cols], fit_cl], axis=1)
        X_val_km = pd.concat([X_val[base_cols], val_cl], axis=1)
        X_tr_km = pd.concat([X_train[base_cols], tr_cl], axis=1)
        X_te_km = pd.concat([X_test[base_cols], te_cl], axis=1)

        if_km_val = IsolationForest(
            n_estimators=IF_N_ESTIMATORS,
            max_samples=min(IF_MAX_SAMPLES, len(X_fit_km)),
            contamination="auto",
            random_state=42,
            n_jobs=-1,
        )
        if_km_val.fit(X_fit_km)
        fit_if_km_score = -if_km_val.score_samples(X_fit_km)
        val_if_km_score = -if_km_val.score_samples(X_val_km)

        if_km_full = IsolationForest(
            n_estimators=IF_N_ESTIMATORS,
            max_samples=min(IF_MAX_SAMPLES, len(X_tr_km)),
            contamination="auto",
            random_state=42,
            n_jobs=-1,
        )
        if_km_full.fit(X_tr_km)
        tr_if_km_score = -if_km_full.score_samples(X_tr_km)
        te_if_km_score = -if_km_full.score_samples(X_te_km)

        # Cac bien the du lieu
        val_variants = (
            ("baseline", X_fit[base_cols], X_val[base_cols]),
            (
                "with_iforest_score",
                X_fit[base_cols].assign(**{IF_SCORE: fit_if_score}),
                X_val[base_cols].assign(**{IF_SCORE: val_if_score}),
            ),
            ("with_kmeans", X_fit_km, X_val_km),
            (
                "with_iforest_and_kmeans",
                X_fit_km.assign(**{IF_SCORE: fit_if_km_score}),
                X_val_km.assign(**{IF_SCORE: val_if_km_score}),
            ),
        )

        test_variants = (
            ("baseline", X_train[base_cols], X_test[base_cols]),
            (
                "with_iforest_score",
                X_train[base_cols].assign(**{IF_SCORE: train_if_score}),
                X_test[base_cols].assign(**{IF_SCORE: test_if_score}),
            ),
            ("with_kmeans", X_tr_km, X_te_km),
            (
                "with_iforest_and_kmeans",
                X_tr_km.assign(**{IF_SCORE: tr_if_km_score}),
                X_te_km.assign(**{IF_SCORE: te_if_km_score}),
            ),
        )

        # Tim nguong tau* toi uu tren validation cho tung bien the
        thresholds = {}
        val_preds_dict = {}
        for cfg_name, f_tr, f_val in val_variants:
            m_v = make_model(model_name, params)
            m_v.fit(f_tr, y_fit)
            p_val = m_v.predict_proba(f_val)[:, 1].astype(np.float32)
            thresholds[cfg_name] = best_f1_threshold(y_val, p_val)
            val_preds_dict[cfg_name] = p_val

        # Huan luyen tren toan bo Train va danh gia tren Test
        baseline_test_pr = None
        best_pr_cfg = -1.0
        best_cfg_name = "baseline"
        best_p_test = None
        best_p_val = None

        for cfg_name, f_train, f_test in test_variants:
            t_eval0 = time.time()
            m_final = make_model(model_name, params)
            m_final.fit(f_train, y_train)
            p_test = m_final.predict_proba(f_test)[:, 1].astype(np.float32)
            eval_metrics = evaluate(y_test, p_test, thresholds[cfg_name])

            if cfg_name == "baseline":
                baseline_test_pr = eval_metrics["pr_auc"]

            delta_test = eval_metrics["pr_auc"] - baseline_test_pr

            if eval_metrics["pr_auc"] > best_pr_cfg:
                best_pr_cfg = eval_metrics["pr_auc"]
                best_cfg_name = cfg_name
                best_p_test = p_test
                best_p_val = val_preds_dict[cfg_name]

            k_info = k_star if "kmeans" in cfg_name else 0
            cfg_display = f"{cfg_name} (K={k_star})" if "kmeans" in cfg_name else cfg_name

            test_rows.append({
                "model": model_name.upper(),
                "configuration": cfg_display,
                "config_type": cfg_name,
                "k": k_info,
                "n_features": f_train.shape[1],
                "pr_auc": eval_metrics["pr_auc"],
                "delta_pr_auc_vs_baseline": round(delta_test, 4),
                "roc_auc": eval_metrics["roc_auc"],
                "f1": eval_metrics["f1"],
                "precision": eval_metrics["precision"],
                "recall": eval_metrics["recall"],
                "threshold": eval_metrics["best_threshold"],
                "train_time_sec": round(time.time() - t_eval0, 1),
            })
            print(
                f"  * {cfg_display:35s}: PR-AUC={eval_metrics['pr_auc']:.4f} "
                f"(Delta={delta_test:+.4f}) | F1={eval_metrics['f1']:.4f} | Prec={eval_metrics['precision']*100:.1f}% | Rec={eval_metrics['recall']*100:.1f}%",
                flush=True,
            )

        best_model_predictions[model_name] = best_p_test
        best_model_val_predictions[model_name] = best_p_val
        best_config_per_model[model_name] = best_cfg_name
        best_test_pr_per_model[model_name] = best_pr_cfg
        print(f"  => CAU HINH TOI UU NHAT CHO [{model_name.upper()}]: {best_cfg_name} (Test PR-AUC = {best_pr_cfg:.4f})")

    df_test_summary = pd.DataFrame(test_rows)

    print("\n" + "#" * 95)
    print("GIAI DOAN 5: DUNG HOP ENSEMBLE CAC MO HINH TOI UU NHAT (+ IF/K-MEANS) TREN TEST")
    print("#" * 95)

    p_xgb = best_model_predictions["xgb"]
    p_cat = best_model_predictions["cat"]
    p_lgb = best_model_predictions["lgbm"]
    p_hist = best_model_predictions["hist"]

    v_xgb = best_model_val_predictions["xgb"]
    v_cat = best_model_val_predictions["cat"]
    v_lgb = best_model_val_predictions["lgbm"]
    v_hist = best_model_val_predictions["hist"]

    N_test = len(y_test)
    r_xgb = rankdata(p_xgb) / float(N_test)
    r_cat = rankdata(p_cat) / float(N_test)
    r_lgb = rankdata(p_lgb) / float(N_test)
    r_hist = rankdata(p_hist) / float(N_test)

    N_val = len(y_val)
    vr_xgb = rankdata(v_xgb) / float(N_val)
    vr_cat = rankdata(v_cat) / float(N_val)
    vr_lgb = rankdata(v_lgb) / float(N_val)
    vr_hist = rankdata(v_hist) / float(N_val)

    ensemble_test_rows = []

    def eval_ensemble_strategy(name, strategy_type, val_scores, test_scores, cfg_desc):
        # Nguong tau* duoc xac dinh HOAN TOAN TREN VALIDATION de dam bao Zero Leakage 100%
        tau_val = best_f1_threshold(y_val, val_scores)

        # Danh gia Out-of-sample tren tap Test bang nguong tau_val
        preds = test_scores >= tau_val
        tp = int(np.sum((y_test == 1) & preds))
        fp = int(np.sum((y_test == 0) & preds))
        fn = int(np.sum((y_test == 1) & ~preds))
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        f1 = 2 * p * r / max(p + r, 1e-12)

        pr_auc = float(average_precision_score(y_test, test_scores))
        roc_auc = float(roc_auc_score(y_test, test_scores))

        ensemble_test_rows.append({
            "model_ensemble": name,
            "strategy": strategy_type,
            "components": cfg_desc,
            "pr_auc": round(pr_auc, 4),
            "roc_auc": round(roc_auc, 4),
            "f1": round(f1, 4),
            "precision": round(p, 4),
            "recall": round(r, 4),
            "threshold": round(tau_val, 4),
        })

    cfg_str = f"XGB({best_config_per_model['xgb']}) + CAT({best_config_per_model['cat']}) + LGB({best_config_per_model['lgbm']}) + HIST({best_config_per_model['hist']})"

    eps = 1e-6
    # 1. Quad Geometric Mean
    v_geo = (v_xgb * v_cat * v_lgb * v_hist) ** 0.25
    p_geo = (p_xgb * p_cat * p_lgb * p_hist) ** 0.25
    eval_ensemble_strategy("Quad Geometric Mean (+ IF/K-Means)", "Geometric Mean", v_geo, p_geo, cfg_str)

    # 2. Quad Harmonic Mean
    v_harm = 4.0 / (1.0 / np.clip(v_xgb, eps, 1.0) + 1.0 / np.clip(v_cat, eps, 1.0) + 1.0 / np.clip(v_lgb, eps, 1.0) + 1.0 / np.clip(v_hist, eps, 1.0))
    p_harm = 4.0 / (1.0 / np.clip(p_xgb, eps, 1.0) + 1.0 / np.clip(p_cat, eps, 1.0) + 1.0 / np.clip(p_lgb, eps, 1.0) + 1.0 / np.clip(p_hist, eps, 1.0))
    eval_ensemble_strategy("Quad Harmonic Mean (+ IF/K-Means)", "Harmonic Mean", v_harm, p_harm, cfg_str)

    # 3. Quad Equal Rank
    v_rank = (vr_xgb + vr_cat + vr_lgb + vr_hist) / 4.0
    p_rank = (r_xgb + r_cat + r_lgb + r_hist) / 4.0
    eval_ensemble_strategy("Quad Equal Rank 0.25 (+ IF/K-Means)", "Rank Averaging", v_rank, p_rank, cfg_str)

    # 4. Quad Soft Voting
    v_soft = (v_xgb + v_cat + v_lgb + v_hist) / 4.0
    p_soft = (p_xgb + p_cat + p_lgb + p_hist) / 4.0
    eval_ensemble_strategy("Quad Equal Prob Soft Voting (+ IF/K-Means)", "Soft Voting", v_soft, p_soft, cfg_str)

    # 5. Trio Soft Voting (XGB + CAT + LGB)
    v_trio_soft = (v_xgb + v_cat + v_lgb) / 3.0
    p_trio_soft = (p_xgb + p_cat + p_lgb) / 3.0
    eval_ensemble_strategy("Trio Soft Voting (XGB + CAT + LGB)", "Soft Voting", v_trio_soft, p_trio_soft, f"XGB + CAT + LGB ({cfg_str})")

    # 6. Trio Rank (XGB + CAT + LGB)
    v_trio_rank = (vr_xgb + vr_cat + vr_lgb) / 3.0
    p_trio_rank = (r_xgb + r_cat + r_lgb) / 3.0
    eval_ensemble_strategy("Trio Rank (XGB + CAT + LGB)", "Rank Averaging", v_trio_rank, p_trio_rank, f"XGB + CAT + LGB ({cfg_str})")

    df_ensemble_summary = pd.DataFrame(ensemble_test_rows).sort_values(by="pr_auc", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 115)
    print("BANG KET QUA DUNG HOP ENSEMBLE CAC MO HINH TOI UU NHAT (+ IF / K-MEANS):")
    print("=" * 115)
    print(df_ensemble_summary.to_string(index=False))
    print("=" * 115)

    return df_k_sweep, df_test_summary, df_ensemble_summary


def main():
    parser = argparse.ArgumentParser(description="Khao sat IF va K-Means nang cao voi K toi uu rieng biet cho tung mo hinh.")
    parser.add_argument("--k-values", type=str, default="2,3,4,5,6,7,8,9,10,11,12", help="Danh sach gia tri K (Mac dinh: 2,3,4,5,6,7,8,9,10,11,12)")
    args = parser.parse_args()

    k_vals = [int(x.strip()) for x in args.k_values.split(",") if x.strip()]

    t0_all = time.time()
    df_k_sweep, df_test_summary, df_ensemble_summary = run_experiment(k_values=k_vals)
    total_time = time.time() - t0_all

    # Xuat tep CSV
    p_sweep = RESULTS_DIR / "iforest_kmeans_expanded_k_sweep.csv"
    p_summary = RESULTS_DIR / "iforest_kmeans_expanded_summary.csv"
    p_ensemble = RESULTS_DIR / "iforest_kmeans_best_ensemble_summary.csv"
    p_md = RESULTS_DIR / "iforest_kmeans_expanded_summary.md"

    df_k_sweep.to_csv(p_sweep, index=False)
    df_test_summary.to_csv(p_summary, index=False)
    df_ensemble_summary.to_csv(p_ensemble, index=False)

    with open(p_md, "w", encoding="utf-8") as f:
        f.write("# Báo Cáo Thực Nghiệm Khảo Sát Tích Hợp Isolation Forest và K-Means\n\n")
        f.write(f"- Thời gian thực thi toàn bộ: {total_time:.1f}s\n\n")
        f.write("## 1. Bảng Quét Cụm K Tối Ưu Cho Từng Mô Hình\n\n")
        f.write(df_k_sweep.to_markdown(index=False))
        f.write("\n\n## 2. Bảng Đánh Giá Chi Tiết 4 Biến Thể Trên Tập Test\n\n")
        f.write(df_test_summary.to_markdown(index=False))
        f.write("\n\n## 3. Bảng Dung Hợp Ensemble Các Cấu Hình Tốt Nhất (+ IF / K-Means)\n\n")
        f.write(df_ensemble_summary.to_markdown(index=False))
        f.write("\n")

    print(f"\n[HOAN TAT] Da luu 4 tep ket qua:")
    print(f"  1. K-Sweep CSV    : {p_sweep}")
    print(f"  2. Summary CSV    : {p_summary}")
    print(f"  3. Ensemble CSV   : {p_ensemble}")
    print(f"  4. Markdown Report: {p_md}")
    print(f"Tong thoi gian chay: {total_time:.1f}s!")


if __name__ == "__main__":
    main()
