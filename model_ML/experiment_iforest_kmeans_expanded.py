"""Compare Isolation Forest and K-Means features across tuned Expanded models."""
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
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
TARGET = "anomaly"
BUILDING_ID = "building_id"
CLUSTER_PREFIX = "building_cluster_"
IF_SCORE = "iforest_score"
MODEL_NAMES = ("xgb", "cat", "lgbm", "hist")
VALIDATION_FRACTION = 0.15
IF_N_ESTIMATORS = 200
IF_MAX_SAMPLES = 256
GPU_AVAILABLE = subprocess.run(
    ["nvidia-smi"], capture_output=True, check=False
).returncode == 0


def load_data() -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    data_dir = SCRIPT_DIR / "data"
    train_path = data_dir / "train_chunked_expanded.parquet"
    test_path = data_dir / "test_chunked_expanded.parquet"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Expanded parquet files not found: {train_path} and {test_path}"
        )
    train = pd.read_parquet(train_path)
    test = pd.read_parquet(test_path)
    required = {TARGET, BUILDING_ID}
    missing_train = required.difference(train.columns)
    missing_test = required.difference(test.columns)
    if missing_train or missing_test:
        raise ValueError(
            f"Expanded data is missing columns; train={sorted(missing_train)}, "
            f"test={sorted(missing_test)}"
        )
    return (
        train.drop(columns=[TARGET]),
        train[TARGET].to_numpy(dtype=np.uint8),
        test.drop(columns=[TARGET]),
        test[TARGET].to_numpy(dtype=np.uint8),
    )


def building_profiles(frame: pd.DataFrame) -> pd.DataFrame:
    profile_columns = [
        column
        for column in frame.select_dtypes(include=[np.number]).columns
        if column != BUILDING_ID and not column.startswith(CLUSTER_PREFIX)
    ]
    profiles = frame.groupby(BUILDING_ID, sort=True)[profile_columns].agg(
        ["mean", "std"]
    )
    profiles.columns = [
        f"{column}_{statistic}" for column, statistic in profiles.columns
    ]
    profiles = profiles.replace([np.inf, -np.inf], np.nan)
    return profiles.fillna(profiles.median()).fillna(0.0)


def fit_kmeans_features(
    fit_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    n_clusters: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float]:
    profiles = building_profiles(fit_frame)
    if len(profiles) <= n_clusters:
        raise ValueError(
            f"K={n_clusters} requires more than {n_clusters} fit buildings; "
            f"found {len(profiles)}."
        )
    scaler = StandardScaler()
    scaled_profiles = scaler.fit_transform(profiles)
    clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    labels = clusterer.fit_predict(scaled_profiles)
    silhouette = float(silhouette_score(scaled_profiles, labels))
    mapping = dict(zip(profiles.index.astype(int), labels.astype(int)))

    def encode(frame: pd.DataFrame) -> pd.DataFrame:
        building_ids = frame[BUILDING_ID].astype(int)
        mapped = building_ids.map(mapping)
        if mapped.isna().any():
            unknown = sorted(building_ids[mapped.isna()].unique().tolist())
            raise ValueError(
                "Buildings absent from fitted K-Means profiles: "
                f"{unknown[:10]}"
            )
        values = np.eye(n_clusters, dtype=np.uint8)[mapped.to_numpy(dtype=int)]
        columns = [f"{CLUSTER_PREFIX}{cluster}" for cluster in range(n_clusters)]
        return pd.DataFrame(values, index=frame.index, columns=columns)

    return (
        encode(fit_frame),
        encode(validation_frame),
        encode(test_frame),
        silhouette,
    )


def load_tuned_params(model_name: str) -> Dict[str, object]:
    path = SCRIPT_DIR / "hyperparameters" / f"best_params_{model_name}_expanded.json"
    if not path.exists():
        raise FileNotFoundError(f"Tuned {model_name} parameters not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        params = json.load(file).get("best_params", {})
    if not params:
        raise ValueError(f"No best_params found in {path}")
    return params


def make_model(model_name: str, params: Dict[str, object]):
    if model_name == "xgb":
        from xgboost import XGBClassifier

        return XGBClassifier(
            **params,
            random_state=42,
            eval_metric="logloss",
            tree_method="hist",
            device="cuda" if GPU_AVAILABLE else "cpu",
            n_jobs=-1,
        )
    if model_name == "cat":
        from catboost import CatBoostClassifier

        return CatBoostClassifier(
            **params,
            random_seed=42,
            verbose=0,
            task_type="GPU" if GPU_AVAILABLE else "CPU",
            **({"thread_count": -1} if not GPU_AVAILABLE else {}),
        )
    if model_name == "lgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            **params,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
    if model_name == "hist":
        return HistGradientBoostingClassifier(**params, random_state=42)
    raise ValueError(f"Unsupported model: {model_name}")


def best_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0,
    )
    best_index = int(np.argmax(f1))
    if best_index == len(thresholds):
        return float(np.nextafter(np.max(scores), np.inf))
    return float(thresholds[best_index])


def evaluate(
    y_true: np.ndarray, scores: np.ndarray, threshold: float
) -> Dict[str, float]:
    predictions = scores >= threshold
    true_positive = int(np.sum((y_true == 1) & predictions))
    false_positive = int(np.sum((y_true == 0) & predictions))
    false_negative = int(np.sum((y_true == 1) & ~predictions))
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "pr_auc": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "threshold_val": threshold,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }


def run_experiment(k_values: List[int] = list(range(2, 13))) -> pd.DataFrame:
    X_train, y_train, X_test, y_test = load_data()
    validation_size = max(1, int(len(X_train) * VALIDATION_FRACTION))
    X_fit = X_train.iloc[:-validation_size].copy()
    y_fit = y_train[:-validation_size]
    X_validation = X_train.iloc[-validation_size:].copy()
    y_validation = y_train[-validation_size:]

    params = load_tuned_params("xgb")
    baseline_validation_model = make_model("xgb", params)
    baseline_validation_model.fit(X_fit, y_fit)
    baseline_validation_predictions = baseline_validation_model.predict_proba(
        X_validation
    )[:, 1]
    baseline_validation_pr = float(
        average_precision_score(y_validation, baseline_validation_predictions)
    )
    sweep_rows: List[Dict[str, float]] = []
    for k in k_values:
        fit_clusters, validation_clusters, _, silhouette = fit_kmeans_features(
            X_fit, X_validation, X_validation, k
        )
        fit_with_clusters = pd.concat([X_fit, fit_clusters], axis=1)
        validation_with_clusters = pd.concat(
            [X_validation, validation_clusters], axis=1
        )
        model = make_model("xgb", params)
        model.fit(fit_with_clusters, y_fit)
        validation_predictions = model.predict_proba(validation_with_clusters)[:, 1]
        validation_pr = float(
            average_precision_score(y_validation, validation_predictions)
        )
        sweep_rows.append(
            {
                "k": float(k),
                "validation_pr_auc": validation_pr,
                "delta_validation_pr_auc_vs_baseline": (
                    validation_pr - baseline_validation_pr
                ),
                "silhouette_fit_buildings": silhouette,
            }
        )
        print(
            f"K={k}: validation PR-AUC={validation_pr:.6f}, "
            f"silhouette={silhouette:.4f}",
            flush=True,
        )
    best_row = max(sweep_rows, key=lambda row: row["validation_pr_auc"])
    best_k = int(best_row["k"])
    print(f"Selected K={best_k} by Expanded validation PR-AUC.", flush=True)

    (
        fit_clusters,
        validation_clusters,
        test_clusters_from_fit,
        fit_silhouette,
    ) = fit_kmeans_features(
        X_fit, X_validation, X_test, best_k
    )
    X_fit_kmeans = pd.concat([X_fit, fit_clusters], axis=1)
    X_validation_kmeans = pd.concat(
        [X_validation, validation_clusters], axis=1
    )
    X_test_fit_kmeans = pd.concat([X_test, test_clusters_from_fit], axis=1)
    train_clusters, _, test_clusters, final_silhouette = fit_kmeans_features(
        X_train, X_train, X_test, best_k
    )
    X_train_kmeans = pd.concat([X_train, train_clusters], axis=1)
    X_test_kmeans = pd.concat([X_test, test_clusters], axis=1)

    if_features = [column for column in X_fit if column != BUILDING_ID]
    iforest = IsolationForest(
        n_estimators=IF_N_ESTIMATORS,
        max_samples=min(IF_MAX_SAMPLES, len(X_fit)),
        contamination="auto",
        random_state=42,
        n_jobs=-1,
    )
    iforest.fit(X_fit[if_features])
    fit_if_score = -iforest.score_samples(X_fit[if_features])
    validation_if_score = -iforest.score_samples(X_validation[if_features])
    kmeans_if_features = [
        column for column in X_fit_kmeans if column != BUILDING_ID
    ]
    validation_kmeans_if = IsolationForest(
        n_estimators=IF_N_ESTIMATORS,
        max_samples=min(IF_MAX_SAMPLES, len(X_fit_kmeans)),
        contamination="auto",
        random_state=42,
        n_jobs=-1,
    )
    validation_kmeans_if.fit(X_fit_kmeans[kmeans_if_features])
    validation_if_kmeans_fit_score = -validation_kmeans_if.score_samples(
        X_fit_kmeans[kmeans_if_features]
    )
    validation_if_kmeans_val_score = -validation_kmeans_if.score_samples(
        X_validation_kmeans[kmeans_if_features]
    )

    final_if = IsolationForest(
        n_estimators=IF_N_ESTIMATORS,
        max_samples=min(IF_MAX_SAMPLES, len(X_train)),
        contamination="auto",
        random_state=42,
        n_jobs=-1,
    )
    final_if.fit(X_train[if_features])
    final_train_if_score = -final_if.score_samples(X_train[if_features])
    final_test_if_score = -final_if.score_samples(X_test[if_features])

    final_if_kmeans_features = [
        column for column in X_train_kmeans if column != BUILDING_ID
    ]
    final_if_kmeans = IsolationForest(
        n_estimators=IF_N_ESTIMATORS,
        max_samples=min(IF_MAX_SAMPLES, len(X_train_kmeans)),
        contamination="auto",
        random_state=42,
        n_jobs=-1,
    )
    final_if_kmeans.fit(X_train_kmeans[final_if_kmeans_features])
    final_train_if_kmeans_score = -final_if_kmeans.score_samples(
        X_train_kmeans[final_if_kmeans_features]
    )
    final_test_if_kmeans_score = -final_if_kmeans.score_samples(
        X_test_kmeans[final_if_kmeans_features]
    )

    rows: List[Dict[str, object]] = []
    for method, fit_frame, validation_frame, test_frame, feature_columns in (
        (
            "isolation_forest_expanded",
            X_fit,
            X_validation,
            X_test,
            if_features,
        ),
        (
            "isolation_forest_expanded_with_kmeans",
            X_fit_kmeans,
            X_validation_kmeans,
            X_test_fit_kmeans,
            [column for column in X_fit_kmeans if column != BUILDING_ID],
        ),
    ):
        model = IsolationForest(
            n_estimators=IF_N_ESTIMATORS,
            max_samples=min(IF_MAX_SAMPLES, len(fit_frame)),
            contamination="auto",
            random_state=42,
            n_jobs=-1,
        )
        model.fit(fit_frame[feature_columns])
        validation_scores = -model.score_samples(validation_frame[feature_columns])
        test_scores = -model.score_samples(test_frame[feature_columns])
        threshold = best_f1_threshold(y_validation, validation_scores)
        rows.append(
            {
                "model": "IsolationForest",
                "configuration": method,
                "k": best_k,
                "silhouette_fit_buildings": fit_silhouette,
                "n_features": len(feature_columns),
                **evaluate(y_test, test_scores, threshold),
            }
        )

    for model_name in MODEL_NAMES:
        params = load_tuned_params(model_name)
        print(f"Evaluating tuned {model_name.upper()} configurations...", flush=True)
        validation_variants = (
            ("baseline", X_fit, X_validation),
            (
                "with_iforest_score",
                X_fit.assign(**{IF_SCORE: fit_if_score}),
                X_validation.assign(**{IF_SCORE: validation_if_score}),
            ),
            ("with_kmeans", X_fit_kmeans, X_validation_kmeans),
            (
                "with_iforest_and_kmeans",
                X_fit_kmeans.assign(**{IF_SCORE: validation_if_kmeans_fit_score}),
                X_validation_kmeans.assign(
                    **{IF_SCORE: validation_if_kmeans_val_score}
                ),
            ),
        )
        thresholds = {}
        for configuration, fit_frame, validation_frame in validation_variants:
            validation_model = make_model(model_name, params)
            validation_model.fit(fit_frame, y_fit)
            validation_predictions = validation_model.predict_proba(
                validation_frame
            )[:, 1]
            thresholds[configuration] = best_f1_threshold(
                y_validation, validation_predictions
            )

        final_variants = (
            ("baseline", X_train, X_test),
            (
                "with_iforest_score",
                X_train.assign(**{IF_SCORE: final_train_if_score}),
                X_test.assign(**{IF_SCORE: final_test_if_score}),
            ),
            ("with_kmeans", X_train_kmeans, X_test_kmeans),
            (
                "with_iforest_and_kmeans",
                X_train_kmeans.assign(**{IF_SCORE: final_train_if_kmeans_score}),
                X_test_kmeans.assign(**{IF_SCORE: final_test_if_kmeans_score}),
            ),
        )
        baseline_test_pr_auc = None
        for configuration, train_frame, test_frame in final_variants:
            final_model = make_model(model_name, params)
            final_model.fit(train_frame, y_train)
            test_predictions = final_model.predict_proba(test_frame)[:, 1]
            metrics = evaluate(
                y_test, test_predictions, thresholds[configuration]
            )
            if configuration == "baseline":
                baseline_test_pr_auc = metrics["pr_auc"]
            rows.append(
                {
                    "model": model_name,
                    "configuration": configuration,
                    "k": best_k,
                    "silhouette_fit_buildings": final_silhouette,
                    "n_features": train_frame.shape[1],
                    **metrics,
                    "delta_pr_auc_vs_baseline": (
                        metrics["pr_auc"] - baseline_test_pr_auc
                    ),
                }
            )
            print(
                f"  {configuration}: PR-AUC={metrics['pr_auc']:.6f}, "
                f"F1={metrics['f1']:.6f}",
                flush=True,
            )
    results = pd.DataFrame(rows)
    results.attrs["k_sweep"] = pd.DataFrame(sweep_rows)
    return results


def main() -> None:
    results = run_experiment()
    results_dir = SCRIPT_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / "iforest_kmeans_expanded_summary.csv"
    sweep_path = results_dir / "iforest_kmeans_expanded_k_sweep.csv"
    results.to_csv(output_path, index=False)
    results.attrs["k_sweep"].to_csv(sweep_path, index=False)
    print(f"\nSaved results: {output_path}")
    print(f"Saved K sweep: {sweep_path}")


if __name__ == "__main__":
    main()
