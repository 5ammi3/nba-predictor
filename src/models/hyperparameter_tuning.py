import os
from datetime import datetime
from typing import Dict, List, Optional, Callable
import optuna
from optuna.samplers import TPESampler
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error
from ..utils.logger import logger
from .xgboost_model import XGBoostPredictor


class HyperparameterOptimizer:
    def __init__(
        self,
        model_type: str = "game_outcome",
        n_trials: int = 50,
        timeout: int = 3600,
    ):
        self.model_type = model_type
        self.n_trials = n_trials
        self.timeout = timeout
        self.best_params: Optional[Dict] = None
        self.study: Optional[optuna.Study] = None

    def _objective(
        self,
        trial: optuna.Trial,
        X: List[Dict[str, float]],
        y: List[int],
        metric: str = "accuracy",
    ) -> float:
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "random_state": 42,
            "n_jobs": -1,
            "early_stopping_rounds": 50,
        }

        from xgboost import XGBClassifier

        X_array = np.array([list(x.values()) for x in X])
        y_array = np.array(y)

        tscv = TimeSeriesSplit(n_splits=3)
        scores = []

        for train_idx, val_idx in tscv.split(X_array):
            X_train, X_val = X_array[train_idx], X_array[val_idx]
            y_train, y_val = y_array[train_idx], y_array[val_idx]

            model = XGBClassifier(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

            if metric == "accuracy":
                pred = model.predict(X_val)
                score = accuracy_score(y_val, pred)
            elif metric == "auc":
                proba = model.predict_proba(X_val)[:, 1]
                score = roc_auc_score(y_val, proba)
            else:
                pred = model.predict(X_val)
                score = mean_squared_error(y_val, pred)

            scores.append(score)

        return np.mean(scores)

    def optimize(
        self,
        X: List[Dict[str, float]],
        y: List[int],
        metric: str = "accuracy",
    ) -> Dict:
        logger.info(f"Starting hyperparameter optimization with {self.n_trials} trials")

        sampler = TPESampler(seed=42)
        self.study = optuna.create_study(
            direction="maximize" if metric in ["accuracy", "auc"] else "minimize",
            sampler=sampler,
        )

        objective = lambda trial: self._objective(trial, X, y, metric)
        self.study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)

        self.best_params = self.study.best_params
        logger.info(f"Best params: {self.best_params}")
        logger.info(f"Best score: {self.study.best_value}")

        return self.best_params

    def get_feature_importance_analysis(
        self, X: List[Dict[str, float]], y: List[int]
    ) -> Dict[str, float]:
        if not self.best_params:
            raise ValueError("Must run optimization first")

        from xgboost import XGBClassifier

        X_array = np.array([list(x.values()) for x in X])
        y_array = np.array(y)

        model = XGBClassifier(**self.best_params, random_state=42, n_jobs=-1)
        model.fit(X_array, y_array)

        feature_names = list(X[0].keys()) if X else []
        importance = model.feature_importances_

        return {feat: float(imp) for feat, imp in zip(feature_names, importance)}


def optimize_model_hyperparameters(
    X: List[Dict[str, float]],
    y: List[int],
    model_type: str = "game_outcome",
    n_trials: int = 50,
) -> Dict:
    optimizer = HyperparameterOptimizer(model_type=model_type, n_trials=n_trials)
    return optimizer.optimize(X, y)


def get_optimal_threshold(y_true: List[int], y_proba: List[float]) -> float:
    thresholds = np.arange(0.3, 0.7, 0.05)
    best_threshold = 0.5
    best_f1 = 0

    for thresh in thresholds:
        pred = (np.array(y_proba) >= thresh).astype(int)
        from sklearn.metrics import f1_score

        f1 = f1_score(y_true, pred)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh

    return best_threshold
