from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
from sqlalchemy.orm import Session
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    brier_score_loss,
    confusion_matrix,
    classification_report,
)
from ..utils.database import get_session, Prediction, Game, Player
from ..utils.logger import logger


class ModelEvaluator:
    def __init__(self, session: Optional[Session] = None):
        self.session = session or get_session()

    def calculate_binary_metrics(
        self,
        y_true: List[int],
        y_pred: List[int],
        y_proba: Optional[List[float]] = None,
    ) -> Dict:
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
        }

        if y_proba:
            metrics["auc"] = roc_auc_score(y_true, y_proba)
            metrics["brier"] = brier_score_loss(y_true, y_proba)

        cm = confusion_matrix(y_true, y_pred)
        metrics["confusion_matrix"] = cm.tolist()

        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics["true_negatives"] = int(tn)
            metrics["false_positives"] = int(fp)
            metrics["false_negatives"] = int(fn)
            metrics["true_positives"] = int(tp)

            metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0

        return metrics

    def calculate_regression_metrics(
        self, y_true: List[float], y_pred: List[float]
    ) -> Dict:
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_true - y_pred))

        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

        return {
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae),
            "r2": float(r2),
            "mape": float(mape),
        }

    def calculate_profit_metrics(
        self,
        predictions: List[Dict],
        odds: List[Dict],
        bet_size: float = 100.0,
    ) -> Dict:
        total_bets = len(predictions)
        if total_bets == 0:
            return {"total_bets": 0, "roi": 0, "profit": 0}

        wins = 0
        total_profit = 0

        for pred, odd in zip(predictions, odds):
            if pred.get("won"):
                wins += 1
                if odd.get("odds", 0) > 0:
                    total_profit += bet_size * (odd["odds"] / 100)
                else:
                    total_profit += bet_size * (100 / abs(odd["odds"]))
            else:
                total_profit -= bet_size

        win_rate = wins / total_bets
        roi = (total_profit / (total_bets * bet_size)) * 100

        return {
            "total_bets": total_bets,
            "wins": wins,
            "win_rate": win_rate,
            "total_profit": total_profit,
            "roi": roi,
            "avg_profit_per_bet": total_profit / total_bets,
        }

    def backtest_by_season(self, model_version: str, seasons: List[int]) -> Dict:
        results = {}

        for season in seasons:
            predictions = (
                self.session.query(Prediction)
                .join(Game)
                .filter(Prediction.model_version == model_version)
                .filter(Game.season == season)
                .all()
            )

            if not predictions:
                continue

            y_true = [
                1 if pred.actual_value == pred.prediction_value else 0
                for pred in predictions
                if pred.actual_value is not None
            ]
            y_pred = [
                int(pred.prediction_value)
                for pred in predictions
                if pred.actual_value is not None
            ]

            if y_true:
                results[f"season_{season}"] = self.calculate_binary_metrics(
                    y_true, y_pred
                )

        return results

    def backtest_by_bet_type(self, model_version: str) -> Dict:
        results = {}

        for pred_type in ["moneyline", "spread", "over_under", "player_prop"]:
            predictions = (
                self.session.query(Prediction)
                .filter(Prediction.model_version == model_version)
                .filter(Prediction.prediction_type == pred_type)
                .all()
            )

            if not predictions:
                continue

            y_true = []
            y_pred = []

            for pred in predictions:
                if pred.actual_value is not None:
                    y_true.append(1 if pred.actual_value > 0.5 else 0)
                    y_pred.append(1 if pred.prediction_value > 0.5 else 0)

            if y_true:
                results[pred_type] = self.calculate_binary_metrics(y_true, y_pred)

        return results

    def backtest_by_month(self, model_version: str, season: int) -> Dict:
        results = {}

        for month in range(10, 16):
            predictions = (
                self.session.query(Prediction)
                .join(Game)
                .filter(Prediction.model_version == model_version)
                .filter(Game.season == season)
                .filter(
                    extract("month", Game.scheduled_date)
                    == (month if month <= 12 else month - 12)
                )
                .all()
            )

            if predictions:
                y_true = [
                    1 if pred.actual_value == pred.prediction_value else 0
                    for pred in predictions
                    if pred.actual_value is not None
                ]
                y_pred = [
                    int(pred.prediction_value)
                    for pred in predictions
                    if pred.actual_value is not None
                ]

                if y_true:
                    results[f"month_{month}"] = self.calculate_binary_metrics(
                        y_true, y_pred
                    )

        return results

    def backtest_by_rest_differential(self, model_version: str) -> Dict:
        results = {}

        for rest_diff in ["home_rest", "away_rest", "back_to_back"]:
            predictions = self.session.query(Prediction).all()

            filtered = [pred for pred in predictions if pred.actual_value is not None]

            if filtered:
                y_true = [
                    1 if pred.actual_value == pred.prediction_value else 0
                    for pred in filtered
                ]
                y_pred = [int(pred.prediction_value) for pred in filtered]
                results[rest_diff] = self.calculate_binary_metrics(y_true, y_pred)

        return results

    def generate_full_report(self, model_version: str = "v1") -> Dict:
        logger.info(f"Generating full backtest report for model {model_version}")

        all_predictions = (
            self.session.query(Prediction)
            .filter(Prediction.model_version == model_version)
            .filter(Prediction.actual_value.isnot(None))
            .all()
        )

        y_true = [int(pred.actual_value) for pred in all_predictions]
        y_pred = [int(pred.prediction_value) for pred in all_predictions]

        report = {
            "overall": self.calculate_binary_metrics(y_true, y_pred),
            "by_bet_type": self.backtest_by_bet_type(model_version),
        }

        return report


def evaluate_model_performance(
    predictions: List[Dict],
    actuals: List[float],
    prediction_type: str = "binary",
) -> Dict:
    evaluator = ModelEvaluator()

    if prediction_type == "binary":
        return evaluator.calculate_binary_metrics(
            [int(a) for a in actuals], [int(p) for p in predictions]
        )
    else:
        return evaluator.calculate_regression_metrics(actuals, predictions)
