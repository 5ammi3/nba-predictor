import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.xgboost_model import XGBoostPredictor
from src.models.hyperparameter_tuning import (
    HyperparameterOptimizer,
    get_optimal_threshold,
)
from src.models.model_evaluation import ModelEvaluator


class TestXGBoostModel:
    def test_prepare_training_data(self):
        model = XGBoostPredictor(model_type="game_outcome")

        X = [
            {"feature1": 1.0, "feature2": 2.0},
            {"feature1": 3.0, "feature2": 4.0},
            {"feature1": 5.0, "feature2": 6.0},
        ]
        y = [0, 1, 1]

        X_array, y_array = model._prepare_training_data(X, y)

        assert X_array.shape == (3, 2)
        assert len(y_array) == 3
        assert len(model.feature_means) == 2

    def test_train_model(self):
        model = XGBoostPredictor(model_type="game_outcome")

        np.random.seed(42)
        X = [{"f1": np.random.randn(), "f2": np.random.randn()} for _ in range(100)]
        y = [np.random.randint(0, 2) for _ in range(100)]

        metrics = model.train(X, y)

        assert "train_accuracy" in metrics
        assert "val_accuracy" in metrics
        assert "model_version" in metrics

    def test_predict(self):
        model = XGBoostPredictor(model_type="game_outcome")

        np.random.seed(42)
        X = [{"f1": np.random.randn(), "f2": np.random.randn()} for _ in range(100)]
        y = [np.random.randint(0, 2) for _ in range(100)]

        model.train(X, y, validation_split=0.2)

        features = {"f1": 0.5, "f2": 0.3}
        result = model.predict(features)

        assert "prediction" in result
        assert "probability" in result
        assert "model_version" in result

    def test_get_feature_importance(self):
        model = XGBoostPredictor(model_type="game_outcome")

        np.random.seed(42)
        X = [{"f1": np.random.randn(), "f2": np.random.randn()} for _ in range(100)]
        y = [np.random.randint(0, 2) for _ in range(100)]

        model.train(X, y)

        importance = model.get_feature_importance()

        assert len(importance) > 0
        assert all(v >= 0 for v in importance.values())


class TestHyperparameterTuning:
    @pytest.mark.skip(reason="Takes too long")
    def test_optimize(self):
        np.random.seed(42)
        X = [{"f1": np.random.randn(), "f2": np.random.randn()} for _ in range(200)]
        y = [np.random.randint(0, 2) for _ in range(200)]

        optimizer = HyperparameterOptimizer(n_trials=5)
        best_params = optimizer.optimize(X, y, metric="accuracy")

        assert "max_depth" in best_params
        assert "learning_rate" in best_params

    def test_get_optimal_threshold(self):
        y_true = [0, 0, 1, 1, 0, 1, 1, 0]
        y_proba = [0.3, 0.4, 0.6, 0.7, 0.35, 0.65, 0.75, 0.45]

        threshold = get_optimal_threshold(y_true, y_proba)

        assert 0.3 <= threshold <= 0.7


class TestModelEvaluator:
    def test_calculate_binary_metrics(self):
        evaluator = ModelEvaluator()

        y_true = [0, 1, 1, 0, 1, 0]
        y_pred = [0, 1, 0, 0, 1, 1]

        metrics = evaluator.calculate_binary_metrics(y_true, y_pred)

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics

    def test_calculate_binary_metrics_with_proba(self):
        evaluator = ModelEvaluator()

        y_true = [0, 1, 1, 0, 1, 0]
        y_pred = [0, 1, 0, 0, 1, 1]
        y_proba = [0.3, 0.7, 0.6, 0.4, 0.8, 0.55]

        metrics = evaluator.calculate_binary_metrics(y_true, y_pred, y_proba)

        assert "auc" in metrics
        assert "brier" in metrics

    def test_calculate_regression_metrics(self):
        evaluator = ModelEvaluator()

        y_true = [10.0, 20.0, 15.0, 25.0]
        y_pred = [11.0, 19.0, 16.0, 24.0]

        metrics = evaluator.calculate_regression_metrics(y_true, y_pred)

        assert "mse" in metrics
        assert "rmse" in metrics
        assert "mae" in metrics
        assert "r2" in metrics

    def test_calculate_profit_metrics(self):
        evaluator = ModelEvaluator()

        predictions = [
            {"won": True},
            {"won": True},
            {"won": False},
        ]
        odds = [
            {"odds": 110},
            {"odds": -110},
            {"odds": -110},
        ]

        metrics = evaluator.calculate_profit_metrics(predictions, odds, bet_size=100)

        assert "total_bets" in metrics
        assert "wins" in metrics
        assert "win_rate" in metrics
        assert "roi" in metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
