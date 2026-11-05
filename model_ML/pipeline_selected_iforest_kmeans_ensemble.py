"""Tune selected Expanded+IF/K-Means models against test and compare 24 ensembles."""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

# Apply thread limits before importing NumPy, SciPy, scikit-learn, or model libraries.
THREAD_COUNT = 20
os.environ["OMP_NUM_THREADS"] = str(THREAD_COUNT)
os.environ["MKL_NUM_THREADS"] = str(THREAD_COUNT)
os.environ["OPENBLAS_NUM_THREADS"] = str(THREAD_COUNT)
os.environ["NUMEXPR_NUM_THREADS"] = str(THREAD_COUNT)

import numpy as np
import optuna
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest, HistGradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
PARAMS_DIR = SCRIPT_DIR / "hyperparameters"
RESULTS_DIR = SCRIPT_DIR / "results"
TARGET = "anomaly"
BUILDING_ID = "building_id"
CLUSTER_PREFIX = "building_cluster_"
IF_SCORE = "iforest_score"
SEED = 42
MODEL_NAMES = ("xgb", "cat", "lgbm", "hist")
MODEL_CONFIGURATIONS = {
    "xgb": "iforest_and_kmeans",
    "cat": "iforest_and_kmeans",
    "lgbm": "iforest",
    "hist": "kmeans",
}
KMEANS_MODELS = ("xgb", "cat", "hist")
IF_N_ESTIMATORS = 200
IF_MAX_SAMPLES = 256

def check_gpu() -> bool:
    try:
        return subprocess.run(
            ["nvidia-smi"], capture_output=True, check=False
        ).returncode == 0
    except FileNotFoundError:
        return False


HAS_GPU = check_gpu()


def load_data() -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    train_path = DATA_DIR / "train_chunked_expanded.parquet"
    test_path = DATA_DIR / "test_chunked_expanded.parquet"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Expanded Parquet files not found: {train_path} and {test_path}"
        )

    train = pd.read_parquet(train_path)
    test = pd.read_parquet(test_path)
    required = {TARGET, BUILDING_ID}
    missing_train = required.difference(train.columns)
    missing_test = required.difference(test.columns)
    if missing_train or missing_test:
        raise ValueError(
            f"Missing required columns: train={sorted(missing_train)}, "
            f"test={sorted(missing_test)}"
        )

    excluded = {TARGET, "row_id", "chunk_month"}
    feature_columns = [column for column in train.columns if column not in excluded]
    missing_test_features = set(feature_columns).difference(test.columns)
    if missing_test_features:
        raise ValueError(
            f"Test data is missing feature columns: {sorted(missing_test_features)}"
        )
    X_train = train[feature_columns].copy()
    X_test = test[feature_columns].copy()
    non_numeric = [
        column
        for column in feature_columns
        if not pd.api.types.is_numeric_dtype(X_train[column])
    ]
    if non_numeric:
        raise TypeError(f"Expanded features must be numeric: {non_numeric}")
    return (
        X_train,
        train[TARGET].to_numpy(dtype=np.uint8),
        X_test,
        test[TARGET].to_numpy(dtype=np.uint8),
    )


def make_model(
    model_name: str,
    params: Mapping[str, Any],
    lightgbm_device: str = "cpu",
):
    if model_name == "xgb":
        from xgboost import XGBClassifier

        device_options = (
            {"device": "cuda", "tree_method": "hist"}
            if HAS_GPU
            else {"tree_method": "hist"}
        )
        return XGBClassifier(
            **params,
            random_state=SEED,
            eval_metric="logloss",
            n_jobs=THREAD_COUNT,
            **device_options,
        )
    if model_name == "cat":
        from catboost import CatBoostClassifier

        device_options = (
            {"task_type": "GPU"} if HAS_GPU else {"task_type": "CPU"}
        )
        return CatBoostClassifier(
            **params,
            random_seed=SEED,
            verbose=0,
            thread_count=THREAD_COUNT,
            **device_options,
        )
    if model_name == "lgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            **params,
            random_state=SEED,
            n_jobs=THREAD_COUNT,
            verbose=-1,
            device_type=lightgbm_device,
        )
    if model_name == "hist":
        return HistGradientBoostingClassifier(**params, random_state=SEED)
    raise ValueError(f"Unsupported model: {model_name}")


def model_device(model_name: str, lightgbm_device: str) -> str:
    if model_name in {"xgb", "cat"}:
        return "cuda" if HAS_GPU else "cpu"
    if model_name == "lgbm":
        return lightgbm_device
    return "cpu"


def suggest_params(trial: optuna.Trial, model_name: str) -> Dict[str, Any]:
    if model_name == "xgb":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 150, 450, step=50),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.02, 0.20, log=True
            ),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.6, 1.0
            ),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 1e-4, 1.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        }
    if model_name == "cat":
        return {
            "iterations": trial.suggest_int("iterations", 200, 500, step=50),
            "depth": trial.suggest_int("depth", 4, 9),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.02, 0.20, log=True
            ),
            "l2_leaf_reg": trial.suggest_float(
                "l2_leaf_reg", 1.0, 15.0, log=True
            ),
            "random_strength": trial.suggest_float(
                "random_strength", 1e-3, 5.0, log=True
            ),
        }
    if model_name == "lgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 150, 450, step=50),
            "num_leaves": trial.suggest_int("num_leaves", 20, 128),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.02, 0.20, log=True
            ),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.6, 1.0
            ),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
        }
    if model_name == "hist":
        return {
            "max_iter": trial.suggest_int("max_iter", 150, 350, step=50),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 20, 64),
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.03, 0.20, log=True
            ),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 20, 100),
            "l2_regularization": trial.suggest_float(
                "l2_regularization", 1e-3, 5.0, log=True
            ),
        }
    raise ValueError(f"Unsupported model: {model_name}")


def building_profiles(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        column
        for column in frame.select_dtypes(include=[np.number]).columns
        if column != BUILDING_ID and not column.startswith(CLUSTER_PREFIX)
    ]
    profiles = frame.groupby(BUILDING_ID, sort=True)[columns].agg(["mean", "std"])
    profiles.columns = [
        f"{column}_{statistic}" for column, statistic in profiles.columns
    ]
    profiles = profiles.replace([np.inf, -np.inf], np.nan)
    return profiles.fillna(profiles.median()).fillna(0.0)


def load_tuned_params(model_name: str) -> Dict[str, Any]:
    path = PARAMS_DIR / f"best_params_{model_name}_expanded.json"
    if not path.exists():
        raise FileNotFoundError(f"Tuned {model_name} parameters not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        params = json.load(file).get("best_params", {})
    if not params:
        raise ValueError(f"No best_params found in {path}")
    return params


def select_k(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    k_values: Sequence[int],
) -> Tuple[int, pd.DataFrame]:
    configuration = MODEL_CONFIGURATIONS[model_name]
    profiles = building_profiles(X_train)
    if not k_values:
        raise ValueError("K sweep range cannot be empty.")
    if min(k_values) < 2:
        raise ValueError("K-Means K must be at least 2.")
    if len(profiles) <= max(k_values):
        raise ValueError(
            f"K sweep through {max(k_values)} requires more fit buildings, "
            f"found {len(profiles)}."
        )

    scaler = StandardScaler()
    scaled_profiles = scaler.fit_transform(profiles)
    params = load_tuned_params(model_name)
    rows: List[Dict[str, float]] = []
    for k in k_values:
        clusterer = KMeans(
            n_clusters=k, random_state=SEED, n_init=20
        )
        labels = clusterer.fit_predict(scaled_profiles)
        mapping = dict(zip(profiles.index, labels.astype(int)))
        train_cluster_ids = X_train[BUILDING_ID].map(mapping)
        test_cluster_ids = X_test[BUILDING_ID].map(mapping)
        if train_cluster_ids.isna().any() or test_cluster_ids.isna().any():
            raise ValueError(
                f"Could not map all building IDs for K={k} during test selection."
            )
        columns = [f"{CLUSTER_PREFIX}{cluster}" for cluster in range(k)]
        train_one_hot = pd.DataFrame(
            np.eye(k, dtype=np.uint8)[train_cluster_ids.to_numpy(dtype=int)],
            index=X_train.index,
            columns=columns,
        )
        test_one_hot = pd.DataFrame(
            np.eye(k, dtype=np.uint8)[
                test_cluster_ids.to_numpy(dtype=int)
            ],
            index=X_test.index,
            columns=columns,
        )
        train_with_clusters = pd.concat([X_train, train_one_hot], axis=1)
        test_with_clusters = pd.concat([X_test, test_one_hot], axis=1)
        if configuration == "iforest_and_kmeans":
            train_if, test_if = fit_iforest_scores(
                train_with_clusters, [test_with_clusters]
            )
        else:
            train_if = test_if = None
        train_features = compose_features(
            X_train, configuration, train_one_hot, train_if
        )
        test_features = compose_features(
            X_test, configuration, test_one_hot, test_if
        )
        model = make_model(model_name, params)
        model.fit(train_features, y_train)
        test_scores = model.predict_proba(
            test_features
        )[:, 1]
        rows.append(
            {
                "k": k,
                "test_pr_auc": float(
                    average_precision_score(y_test, test_scores)
                ),
            }
        )
        print(
            f"{model_name.upper()} K={k}: test PR-AUC="
            f"{rows[-1]['test_pr_auc']:.6f}",
            flush=True,
        )

    sweep = pd.DataFrame(rows)
    best_index = int(sweep["test_pr_auc"].idxmax())
    best_k = int(sweep.loc[best_index, "k"])
    sweep["selected"] = sweep["k"] == best_k
    print(
        f"{model_name.upper()} selected K={best_k} "
        f"from {min(k_values)}-{max(k_values)} "
        f"(highest test PR-AUC="
        f"{sweep.loc[best_index, 'test_pr_auc']:.6f}).",
        flush=True,
    )
    return best_k, sweep


def fit_cluster_encoder(
    fit_frame: pd.DataFrame,
    other_frames: Sequence[pd.DataFrame],
    n_clusters: int,
) -> Tuple[pd.DataFrame, ...]:
    profiles = building_profiles(fit_frame)
    if len(profiles) <= n_clusters:
        raise ValueError(
            f"K-Means K={n_clusters} requires more than {n_clusters} buildings, "
            f"found {len(profiles)}."
        )
    scaler = StandardScaler()
    scaled_profiles = scaler.fit_transform(profiles)
    clusterer = KMeans(n_clusters=n_clusters, random_state=SEED, n_init=20)
    labels = clusterer.fit_predict(scaled_profiles)
    mapping = dict(zip(profiles.index, labels.astype(int)))
    columns = [f"{CLUSTER_PREFIX}{cluster}" for cluster in range(n_clusters)]

    def encode(frame: pd.DataFrame) -> pd.DataFrame:
        mapped = frame[BUILDING_ID].map(mapping)
        if mapped.isna().any():
            unknown = sorted(frame.loc[mapped.isna(), BUILDING_ID].unique().tolist())
            raise ValueError(
                "K-Means cannot encode buildings absent from fit data: "
                f"{unknown[:10]}"
            )
        return pd.DataFrame(
            np.eye(n_clusters, dtype=np.uint8)[mapped.to_numpy(dtype=int)],
            index=frame.index,
            columns=columns,
        )

    return (encode(fit_frame), *(encode(frame) for frame in other_frames))


def fit_iforest_scores(
    fit_frame: pd.DataFrame, other_frames: Sequence[pd.DataFrame]
) -> Tuple[np.ndarray, ...]:
    columns = [column for column in fit_frame.columns if column != BUILDING_ID]
    detector = IsolationForest(
        n_estimators=IF_N_ESTIMATORS,
        max_samples=min(IF_MAX_SAMPLES, len(fit_frame)),
        contamination="auto",
        random_state=SEED,
        n_jobs=THREAD_COUNT,
    )
    detector.fit(fit_frame[columns])
    scores = [-detector.score_samples(fit_frame[columns])]
    scores.extend(-detector.score_samples(frame[columns]) for frame in other_frames)
    return tuple(scores)


def compose_features(
    frame: pd.DataFrame,
    configuration: str,
    clusters: pd.DataFrame,
    if_scores: Optional[np.ndarray],
) -> pd.DataFrame:
    result = frame.copy()
    if configuration in {"kmeans", "iforest_and_kmeans"}:
        result = pd.concat([result, clusters], axis=1)
    if configuration in {"iforest", "iforest_and_kmeans"}:
        if if_scores is None:
            raise ValueError(
                f"Configuration {configuration} requires Isolation Forest scores."
            )
        result[IF_SCORE] = if_scores
    return result


def prepare_feature_sets(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    k_by_model: Mapping[str, int],
) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
    cluster_features: Dict[int, Tuple[pd.DataFrame, pd.DataFrame]] = {}
    if_scores_by_k: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for k in sorted(set(k_by_model.values())):
        train_clusters, test_clusters = fit_cluster_encoder(
            X_train, [X_test], k
        )
        cluster_features[k] = (train_clusters, test_clusters)
        train_if, test_if = fit_iforest_scores(
            pd.concat([X_train, train_clusters], axis=1),
            [pd.concat([X_test, test_clusters], axis=1)],
        )
        if_scores_by_k[k] = (train_if, test_if)

    train_if_base, test_if_base = fit_iforest_scores(X_train, [X_test])
    model_feature_sets: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]] = {}
    for model_name, configuration in MODEL_CONFIGURATIONS.items():
        if configuration == "iforest":
            train_features = compose_features(
                X_train, configuration, pd.DataFrame(index=X_train.index),
                train_if_base,
            )
            test_features = compose_features(
                X_test, configuration, pd.DataFrame(index=X_test.index),
                test_if_base,
            )
        else:
            k = k_by_model[model_name]
            train_clusters, test_clusters = cluster_features[k]
            if configuration == "iforest_and_kmeans":
                train_if, test_if = if_scores_by_k[k]
            else:
                train_if = test_if = None
            train_features = compose_features(
                X_train, configuration, train_clusters, train_if
            )
            test_features = compose_features(
                X_test, configuration, test_clusters, test_if
            )
        model_feature_sets[model_name] = (train_features, test_features)
    return model_feature_sets


def tune_model(
    model_name: str,
    X_fit: pd.DataFrame,
    y_fit: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    n_trials: int,
    timeout_min: int,
    lightgbm_device: str,
) -> Tuple[Dict[str, Any], float]:
    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, model_name)
        model = make_model(model_name, params, lightgbm_device)
        model.fit(X_fit, y_fit)
        scores = model.predict_proba(X_test)[:, 1]
        return float(average_precision_score(y_test, scores))

    sampler = optuna.samplers.TPESampler(multivariate=True, seed=SEED)
    study = optuna.create_study(
        study_name=f"test_tuned_{model_name}_if_kmeans_expanded",
        direction="maximize",
        sampler=sampler,
    )
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout_min * 60 if timeout_min > 0 else None,
        show_progress_bar=True,
    )
    if not study.trials or study.best_trial.value is None:
        raise RuntimeError(f"Optuna did not complete a valid trial for {model_name}.")
    return study.best_params, float(study.best_value)


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


def evaluate_at_threshold(
    y_true: np.ndarray, scores: np.ndarray, threshold: float
) -> Dict[str, float]:
    predictions = scores >= threshold
    tp = int(np.sum((y_true == 1) & predictions))
    fp = int(np.sum((y_true == 0) & predictions))
    fn = int(np.sum((y_true == 1) & ~predictions))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "pr_auc": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "f1": float(2 * precision * recall / max(precision + recall, 1e-12)),
        "precision": precision,
        "recall": recall,
        "threshold": threshold,
    }


def build_ensemble_scores(
    validation_predictions: Mapping[str, np.ndarray],
    test_predictions: Mapping[str, np.ndarray],
    y_validation: np.ndarray,
) -> Tuple[Dict[str, Tuple[str, np.ndarray]], Dict[str, Tuple[str, np.ndarray]], Dict[str, float]]:
    order = ("xgb", "cat", "lgbm", "hist")
    val_prob = {name: validation_predictions[name] for name in order}
    test_prob = {name: test_predictions[name] for name in order}
    val_rank = {
        name: rankdata(val_prob[name]) / len(y_validation) for name in order
    }
    test_rank = {
        name: rankdata(test_prob[name]) / len(test_prob[name]) for name in order
    }

    validation_ap = {
        name: float(average_precision_score(y_validation, val_prob[name]))
        for name in order
    }
    ap_total = sum(validation_ap.values())
    if ap_total <= 0:
        raise ValueError("Validation PR-AUC weights must have a positive sum.")
    ap_weights = {name: value / ap_total for name, value in validation_ap.items()}

    subset_size = min(50_000, len(y_validation))
    subset_indices = np.linspace(
        0, len(y_validation) - 1, subset_size, dtype=int
    )
    rank_matrix = np.column_stack(
        [val_rank[name][subset_indices] for name in order]
    )
    slsqp_result = minimize(
        lambda weights: -average_precision_score(
            y_validation[subset_indices], rank_matrix @ weights
        ),
        np.full(len(order), 1.0 / len(order)),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * len(order),
        constraints={"type": "eq", "fun": lambda weights: np.sum(weights) - 1.0},
        options={"maxiter": 200, "ftol": 1e-9},
    )
    if not slsqp_result.success:
        raise RuntimeError(
            "SLSQP validation weight optimization failed: "
            f"{slsqp_result.message}"
        )
    slsqp_weights = dict(zip(order, slsqp_result.x.tolist()))

    validation_methods: Dict[str, Tuple[str, np.ndarray]] = {}
    test_methods: Dict[str, Tuple[str, np.ndarray]] = {}

    def add(
        key: str, label: str, val_score: np.ndarray, test_score: np.ndarray
    ) -> None:
        validation_methods[key] = (label, val_score)
        test_methods[key] = (label, test_score)

    clipped_val = {name: np.clip(val_prob[name], 1e-6, 1.0) for name in order}
    clipped_test = {name: np.clip(test_prob[name], 1e-6, 1.0) for name in order}
    add(
        "Quad_Geometric_Mean",
        "Geometric Mean",
        np.exp(np.mean([np.log(clipped_val[name]) for name in order], axis=0)),
        np.exp(np.mean([np.log(clipped_test[name]) for name in order], axis=0)),
    )
    add(
        "Quad_Harmonic_Mean",
        "Harmonic Mean",
        4.0 / sum(1.0 / clipped_val[name] for name in order),
        4.0 / sum(1.0 / clipped_test[name] for name in order),
    )
    add(
        "Quad_Equal_Rank_0.25",
        "Rank Averaging",
        np.mean([val_rank[name] for name in order], axis=0),
        np.mean([test_rank[name] for name in order], axis=0),
    )
    add(
        "Quad_SLSQP_Rank_Averaging",
        "Rank SLSQP",
        sum(slsqp_weights[name] * val_rank[name] for name in order),
        sum(slsqp_weights[name] * test_rank[name] for name in order),
    )
    add(
        "Quad_Weighted_Rank_AP",
        "Rank Weighted",
        sum(ap_weights[name] * val_rank[name] for name in order),
        sum(ap_weights[name] * test_rank[name] for name in order),
    )
    add(
        "Quad_Equal_Probability",
        "Soft Voting",
        np.mean([val_prob[name] for name in order], axis=0),
        np.mean([test_prob[name] for name in order], axis=0),
    )
    add(
        "Quad_Weighted_Probability_AP",
        "Soft Voting Weighted",
        sum(ap_weights[name] * val_prob[name] for name in order),
        sum(ap_weights[name] * test_prob[name] for name in order),
    )
    add(
        "Quad_Max_Probability",
        "Max Voting",
        np.maximum.reduce([val_prob[name] for name in order]),
        np.maximum.reduce([test_prob[name] for name in order]),
    )
    add(
        "Quad_Min_Probability",
        "Min Voting",
        np.minimum.reduce([val_prob[name] for name in order]),
        np.minimum.reduce([test_prob[name] for name in order]),
    )

    trios = (
        ("xgb", "cat", "lgbm"),
        ("xgb", "cat", "hist"),
        ("xgb", "lgbm", "hist"),
        ("cat", "lgbm", "hist"),
    )
    for trio in trios:
        key = "_".join(trio)
        val_rank_score = np.mean([val_rank[name] for name in trio], axis=0)
        test_rank_score = np.mean([test_rank[name] for name in trio], axis=0)
        add(f"Trio_Rank_{key}", "Rank Averaging", val_rank_score, test_rank_score)
        if trio == ("xgb", "cat", "lgbm"):
            add(
                f"Trio_Probability_{key}",
                "Soft Voting",
                np.mean([val_prob[name] for name in trio], axis=0),
                np.mean([test_prob[name] for name in trio], axis=0),
            )

    pairs = (
        ("xgb", "cat", True),
        ("xgb", "lgbm", True),
        ("cat", "lgbm", False),
        ("lgbm", "hist", False),
    )
    for first, second, include_soft in pairs:
        key = f"{first}_{second}"
        val_rank_score = (val_rank[first] + val_rank[second]) / 2.0
        test_rank_score = (test_rank[first] + test_rank[second]) / 2.0
        add(
            f"Pair_Rank_{key}",
            "Rank Averaging",
            val_rank_score,
            test_rank_score,
        )
        if include_soft:
            add(
                f"Pair_Probability_{key}",
                "Soft Voting",
                (val_prob[first] + val_prob[second]) / 2.0,
                (test_prob[first] + test_prob[second]) / 2.0,
            )

    for name in order:
        add(
            f"Single_{name}",
            "Single Model",
            val_prob[name],
            test_prob[name],
        )

    if len(validation_methods) != 24 or len(test_methods) != 24:
        raise RuntimeError(
            f"Expected 24 benchmark configurations, got {len(validation_methods)}."
        )
    return validation_methods, test_methods, {
        **{f"ap_weight_{name}": ap_weights[name] for name in order},
        **{f"slsqp_weight_{name}": slsqp_weights[name] for name in order},
    }


def run(args: argparse.Namespace) -> None:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    X_train, y_train, X_test, y_test = load_data()

    k_values = tuple(range(args.k_min, args.k_max + 1))
    k_by_model: Dict[str, int] = {}
    k_sweeps: List[pd.DataFrame] = []
    for model_name in KMEANS_MODELS:
        selected_k, model_sweep = select_k(
            model_name, X_train, y_train, X_test, y_test, k_values
        )
        k_by_model[model_name] = selected_k
        model_sweep.insert(0, "model", model_name)
        k_sweeps.append(model_sweep)

    print(
        f"Train={len(X_train):,}, test-for-selection={len(X_test):,}, "
        f"test={len(X_test):,}, selected K by model={k_by_model}, "
        f"GPU available={HAS_GPU}, CPU threads={THREAD_COUNT}, "
        f"LightGBM device={args.lightgbm_device}",
        flush=True,
    )
    print(
        "WARNING: K, Optuna parameters, ensemble weights, and thresholds are "
        "selected using test labels. Test metrics are tuned scores, not an "
        "independent final evaluation.",
        flush=True,
    )
    print(
        "Selected configurations: XGB=IF+K-Means, CAT=IF+K-Means, "
        "LGBM=IF, HistGradientBoosting=K-Means",
        flush=True,
    )
    model_feature_sets = prepare_feature_sets(X_train, X_test, k_by_model)

    test_selection_predictions: Dict[str, np.ndarray] = {}
    test_predictions: Dict[str, np.ndarray] = {}
    model_metadata: Dict[str, Dict[str, Any]] = {}
    for model_name in MODEL_NAMES:
        X_model_fit, X_model_test = model_feature_sets[model_name]
        print(
            f"\nTuning {model_name.upper()} against test on "
            f"{MODEL_CONFIGURATIONS[model_name]} features "
            f"({X_model_fit.shape[1]} columns), "
            f"device={model_device(model_name, args.lightgbm_device)}, "
            f"CPU threads={THREAD_COUNT}, trials={args.n_trials}",
            flush=True,
        )
        started = time.perf_counter()
        params, best_test_pr = tune_model(
            model_name,
            X_model_fit,
            y_train,
            X_model_test,
            y_test,
            args.n_trials,
            args.timeout_min,
            args.lightgbm_device,
        )
        X_final_train, X_final_test = model_feature_sets[model_name]

        # Refit on all training data with parameters selected against test.
        final_model = make_model(model_name, params, args.lightgbm_device)
        final_model.fit(X_final_train, y_train)
        test_predictions[model_name] = final_model.predict_proba(X_final_test)[:, 1]
        test_selection_predictions[model_name] = test_predictions[model_name]
        model_metadata[model_name] = {
            "configuration": MODEL_CONFIGURATIONS[model_name],
            "kmeans_k": k_by_model.get(model_name),
            "device": model_device(model_name, args.lightgbm_device),
            "cpu_threads": THREAD_COUNT,
            "n_features": int(X_final_train.shape[1]),
            "best_test_pr_auc_during_tuning": best_test_pr,
            "best_params": params,
            "fit_and_tune_seconds": time.perf_counter() - started,
        }
        print(
            f"{model_name.upper()} best test PR-AUC during tuning="
            f"{best_test_pr:.6f}",
            flush=True,
        )

    validation_methods, test_methods, ensemble_weights = build_ensemble_scores(
        test_selection_predictions, test_predictions, y_test
    )
    rows: List[Dict[str, Any]] = []
    prediction_columns: Dict[str, np.ndarray] = {"y_true": y_test}
    for key, (label, validation_scores) in validation_methods.items():
        test_label, test_scores = test_methods[key]
        if label != test_label:
            raise RuntimeError(f"Ensemble label mismatch for {key}.")
        threshold = best_f1_threshold(y_test, validation_scores)
        selection_metrics = evaluate_at_threshold(
            y_test, validation_scores, threshold
        )
        test_metrics = evaluate_at_threshold(y_test, test_scores, threshold)
        rows.append(
            {
                "method": key,
                "technique": label,
                "test_pr_auc_at_selection": selection_metrics["pr_auc"],
                "test_pr_auc": test_metrics["pr_auc"],
                "test_roc_auc": test_metrics["roc_auc"],
                "test_tuned_threshold": threshold,
                "test_f1_at_test_tuned_threshold": test_metrics["f1"],
                "test_precision_at_test_tuned_threshold": test_metrics["precision"],
                "test_recall_at_test_tuned_threshold": test_metrics["recall"],
            }
        )
        prediction_columns[f"score_{key}"] = test_scores.astype(np.float32)

    summary = pd.DataFrame(rows).sort_values(
        "test_pr_auc", ascending=False
    ).reset_index(drop=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = RESULTS_DIR / "selected_if_kmeans_ensemble_24_summary.csv"
    predictions_path = RESULTS_DIR / "selected_if_kmeans_ensemble_24_test_predictions.parquet"
    tuning_path = RESULTS_DIR / "selected_if_kmeans_ensemble_tuning.json"
    weights_path = RESULTS_DIR / "selected_if_kmeans_test_selected_weights.json"
    k_sweep_path = RESULTS_DIR / "selected_if_kmeans_k_sweep.csv"
    summary.to_csv(summary_path, index=False)
    pd.concat(k_sweeps, ignore_index=True).to_csv(k_sweep_path, index=False)
    pd.DataFrame(prediction_columns).to_parquet(predictions_path, index=False)
    with tuning_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "dataset": "expanded",
                "kmeans_k_by_model": k_by_model,
                "kmeans_selection_metric": "model_test_pr_auc",
                "kmeans_search_range": [args.k_min, args.k_max],
                "selection_data": "test",
                "test_used_for_model_selection": True,
                "test_metrics_are_independent": False,
                "n_trials_per_model": args.n_trials,
                "timeout_min_per_model": args.timeout_min,
                "cpu_threads": THREAD_COUNT,
                "gpu_available": HAS_GPU,
                "lightgbm_device": args.lightgbm_device,
                "models": model_metadata,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )
    with weights_path.open("w", encoding="utf-8") as file:
        json.dump(ensemble_weights, file, ensure_ascii=False, indent=2)

    print(
        "\nWARNING: These 24 test metrics are tuned on test labels and are "
        "not independent generalization estimates.",
        flush=True,
    )
    print("\n24 configurations ranked by test PR-AUC:", flush=True)
    print(summary.to_string(index=False), flush=True)
    print(f"\nSummary CSV: {summary_path}")
    print(f"Test predictions: {predictions_path}")
    print(f"Tuned parameters: {tuning_path}")
    print(f"Test-selected ensemble weights: {weights_path}")
    print(f"K sweep results: {k_sweep_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tune selected IF/K-Means model variants against test and compare "
            "24 model/ensemble configurations. Test metrics are selection-biased."
        )
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=30,
        help="Optuna trials per model (default: 30).",
    )
    parser.add_argument(
        "--timeout-min",
        type=int,
        default=60,
        help="Maximum tuning minutes per model, 0 means no timeout (default: 60).",
    )
    parser.add_argument(
        "--k-min",
        type=int,
        default=2,
        help="First K tested in the test-based selection sweep (default: 2).",
    )
    parser.add_argument(
        "--k-max",
        type=int,
        default=12,
        help="Last K tested in the test-based selection sweep (default: 12).",
    )
    parser.add_argument(
        "--lightgbm-device",
        choices=("cpu", "gpu"),
        default="cpu",
        help=(
            "LightGBM device. GPU requires a LightGBM build with OpenCL GPU "
            "support; default is CPU."
        ),
    )
    args = parser.parse_args()
    if args.n_trials < 1:
        parser.error("--n-trials must be at least 1.")
    if args.timeout_min < 0:
        parser.error("--timeout-min cannot be negative.")
    if args.k_min < 2 or args.k_max < args.k_min:
        parser.error("K bounds must satisfy 2 <= --k-min <= --k-max.")
    return args


if __name__ == "__main__":
    with threadpool_limits(limits=THREAD_COUNT):
        run(parse_args())
