import os
import pickle
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    brier_score_loss,
    mean_squared_error,
)
from ..utils.logger import logger
from ..utils.database import get_session


class XGBoostPredictor:
    def __init__(
        self,
        model_type: str = "game_outcome",
        model_version: str = "v1",
    ):
        self.model_type = model_type
        self.model_version = model_version
        self.model: Optional[xgb.XGBClassifier] = None
        self.feature_means: Dict[str, float] = {}
        self.feature_stds: Dict[str, float] = {}

        self.default_params = {
            "max_depth": 6,
            "learning_rate": 0.05,
            "n_estimators": 500,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 3,
            "gamma": 0.1,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": 42,
            "n_jobs": -1,
            "early_stopping_rounds": 50,
        }

    def _prepare_training_data(
        self,
        X: List[Dict[str, float]],
        y: List[int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        feature_names = list(X[0].keys()) if X else []

        X_array = np.zeros((len(X), len(feature_names)))
        for i, row in enumerate(X):
            for j, feat in enumerate(feature_names):
                X_array[i, j] = row.get(feat, 0)

        y_array = np.array(y)

        self.feature_means = {
            feat: np.mean(X_array[:, j]) for j, feat in enumerate(feature_names)
        }
        self.feature_stds = {
            feat: np.std(X_array[:, j]) for j, feat in enumerate(feature_names)
        }

        X_array = (X_array - np.array(list(self.feature_means.values()))) / (
            np.array(list(self.feature_stds.values())) + 1e-8
        )

        return X_array, y_array

    def train(
        self,
        X: List[Dict[str, float]],
        y: List[int],
        validation_split: float = 0.2,
    ) -> Dict:
        if not X or not y:
            raise ValueError("Training data cannot be empty")

        X_array, y_array = self._prepare_training_data(X, y)

        split_idx = int(len(X_array) * (1 - validation_split))
        X_train, X_val = X_array[:split_idx], X_array[split_idx:]
        y_train, y_val = y_array[:split_idx], y_array[split_idx:]

        self.model = xgb.XGBClassifier(**self.default_params)

        if self.model_type in ["game_outcome", "spread", "over_under"]:
            eval_set = [(X_val, y_val)]
            self.model.fit(
                X_train,
                y_train,
                eval_set=eval_set,
                verbose=False,
            )
        else:
            self.model.fit(X_train, y_train)

        train_pred = self.model.predict(X_train)
        val_pred = self.model.predict(X_val)

        metrics = {
            "train_accuracy": accuracy_score(y_train, train_pred),
            "val_accuracy": accuracy_score(y_val, val_pred),
            "model_version": self.model_version,
            "training_date": datetime.now().isoformat(),
            "n_train_samples": len(X_train),
            "n_val_samples": len(X_val),
        }

        if hasattr(self.model, "best_iteration"):
            metrics["best_iteration"] = self.model.best_iteration

        if self.model_type == "game_outcome":
            val_proba = self.model.predict_proba(X_val)[:, 1]
            metrics["val_auc"] = roc_auc_score(y_val, val_proba)
            metrics["val_brier"] = brier_score_loss(y_val, val_proba)

        logger.info(f"Model trained: {metrics}")
        return metrics

    def predict(self, features: Dict[str, float]) -> Dict:
        if not self.model:
            raise ValueError("Model not trained")

        feature_names = list(self.feature_means.keys())

        feature_vector = np.zeros((1, len(feature_names)))
        for j, feat in enumerate(feature_names):
            if feat in features:
                raw_value = features[feat]
                if feat in self.feature_means:
                    normalized = (raw_value - self.feature_means[feat]) / (
                        self.feature_stds.get(feat, 1) + 1e-8
                    )
                    feature_vector[0, j] = normalized
                else:
                    feature_vector[0, j] = raw_value
            else:
                feature_vector[0, j] = 0

        prediction = self.model.predict(feature_vector)[0]
        probability = self.model.predict_proba(feature_vector)[0]

        result = {
            "prediction": int(prediction),
            "probability": float(probability[1]),
            "model_version": self.model_version,
            "model_type": self.model_type,
        }

        if self.model_type in ["game_outcome", "spread", "over_under"]:
            result["confidence"] = float(max(probability))

        return result

    def predict_proba_over_under(self, features: Dict[str, float], line: float) -> Dict:
        pred = self.predict(features)

        result = {
            "predicted_value": pred["prediction"] * line,
            "over_probability": pred["probability"],
            "under_probability": 1 - pred["probability"],
            "confidence": pred["confidence"],
            "line": line,
        }

        return result

    def save(self, filepath: str):
        if not self.model:
            raise ValueError("No model to save")

        model_data = {
            "model": self.model,
            "feature_means": self.feature_means,
            "feature_stds": self.feature_stds,
            "model_type": self.model_type,
            "model_version": self.model_version,
            "default_params": self.default_params,
        }

        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)

        logger.info(f"Model saved to {filepath}")

    def load(self, filepath: str):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")

        with open(filepath, "rb") as f:
            model_data = pickle.load(f)

        self.model = model_data["model"]
        self.feature_means = model_data["feature_means"]
        self.feature_stds = model_data["feature_stds"]
        self.model_type = model_data["model_type"]
        self.model_version = model_data["model_version"]
        self.default_params = model_data.get("default_params", {})

        logger.info(f"Model loaded from {filepath}")

    def get_feature_importance(self) -> Dict[str, float]:
        if not self.model:
            return {}

        importance = self.model.feature_importances_
        feature_names = list(self.feature_means.keys())

        return {feat: float(imp) for feat, imp in zip(feature_names, importance)}


def train_game_outcome_model(
    X: List[Dict[str, float]], y: List[int], version: str = "v1"
) -> XGBoostPredictor:
    model = XGBoostPredictor(model_type="game_outcome", model_version=version)
    model.train(X, y)
    return model


def train_spread_model(
    X: List[Dict[str, float]], y: List[int], version: str = "v1"
) -> XGBoostPredictor:
    model = XGBoostPredictor(model_type="spread", model_version=version)
    model.train(X, y)
    return model


def train_over_under_model(
    X: List[Dict[str, float]], y: List[int], version: str = "v1"
) -> XGBoostPredictor:
    model = XGBoostPredictor(model_type="over_under", model_version=version)
    model.train(X, y)
    return model


def train_player_prop_model(
    X: List[Dict[str, float]], y: List[float], version: str = "v1"
) -> XGBoostPredictor:
    model = XGBoostPredictor(model_type="player_prop", model_version=version)
    model.default_params["objective"] = "reg:squarederror"
    model.default_params["n_estimators"] = 300
    model.train(X, y)
    return model
