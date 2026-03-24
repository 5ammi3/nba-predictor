import os
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np
from ..models.xgboost_model import XGBoostPredictor
from ..features.feature_utils import prepare_game_features, add_contextual_features
from ..features.text_embeddings import embeddings_processor
from ..utils.database import get_session, Game, Team
from ..utils.logger import logger


class GamePredictor:
    def __init__(self):
        self.session = get_session()
        self.models: Dict[str, XGBoostPredictor] = {}
        self._load_models()

    def _load_models(self):
        model_dir = "models"

        for model_type in ["moneyline", "spread", "over_under"]:
            model_path = os.path.join(model_dir, f"{model_type}_model.pkl")
            if os.path.exists(model_path):
                model = XGBoostPredictor(model_type=model_type)
                model.load(model_path)
                self.models[model_type] = model
                logger.info(f"Loaded {model_type} model")

    def _build_early_fusion_features(
        self,
        home_team_id: int,
        away_team_id: int,
        game_date: datetime,
        injury_embeddings: Optional[List[List[float]]] = None,
        news_embeddings: Optional[List[List[float]]] = None,
    ) -> Dict[str, float]:
        structured_features = prepare_game_features(
            home_team_id, away_team_id, game_date, self.session
        )

        structured_features = add_contextual_features(
            structured_features, game_date, game_date.year
        )

        combined = structured_features.copy()

        if injury_embeddings:
            for i, emb in enumerate(injury_embeddings[:10]):
                for j, val in enumerate(emb[:50]):
                    combined[f"injury_emb_{i}_{j}"] = val

        if news_embeddings:
            for i, emb in enumerate(news_embeddings[:5]):
                for j, val in enumerate(emb[:50]):
                    combined[f"news_emb_{i}_{j}"] = val

        return combined

    async def predict_moneyline(
        self,
        home_team_id: int,
        away_team_id: int,
        game_date: datetime,
    ) -> Dict:
        features = self._build_early_fusion_features(
            home_team_id, away_team_id, game_date
        )

        if "moneyline" in self.models:
            result = self.models["moneyline"].predict(features)
        else:
            result = self._baseline_moneyline_prediction(features)

        return {
            "home_win_probability": result.get("probability", 0.5),
            "away_win_probability": 1 - result.get("probability", 0.5),
            "confidence": result.get("confidence", 0.5),
            "model_version": result.get("model_version", "baseline"),
            "recommendation": "home"
            if result.get("probability", 0.5) > 0.5
            else "away",
        }

    async def predict_spread(
        self,
        home_team_id: int,
        away_team_id: int,
        game_date: datetime,
        spread_line: float = -5.5,
    ) -> Dict:
        features = self._build_early_fusion_features(
            home_team_id, away_team_id, game_date
        )

        net_rating_diff = features.get("net_rating_differential", 0)
        home_net = features.get("home_net_rating", 0)
        away_net = features.get("away_net_rating", 0)

        predicted_spread = (away_net - home_net) * 0.5

        home_covers = 1 if predicted_spread > spread_line else 0

        win_pct_diff = abs(features.get("win_pct_diff", 0))
        confidence = min(win_pct_diff * 1.5 + 0.3, 0.95)

        if "spread" in self.models:
            model_result = self.models["spread"].predict(features)
            home_covers = model_result.get("prediction", home_covers)
            confidence = model_result.get("confidence", confidence)

        return {
            "predicted_spread": predicted_spread,
            "spread_line": spread_line,
            "home_covers_probability": confidence,
            "away_covers_probability": 1 - confidence,
            "recommendation": "home" if home_covers == 1 else "away",
            "confidence": confidence,
        }

    async def predict_over_under(
        self,
        home_team_id: int,
        away_team_id: int,
        game_date: datetime,
        total_line: float = 220.0,
    ) -> Dict:
        features = self._build_early_fusion_features(
            home_team_id, away_team_id, game_date
        )

        home_pace = features.get("home_pace", 100)
        away_pace = features.get("away_pace", 100)
        avg_pace = (home_pace + away_pace) / 2

        home_ortg = features.get("home_offensive_rating", 112)
        away_ortg = features.get("away_offensive_rating", 112)
        avg_ortg = (home_ortg + away_ortg) / 2

        predicted_total = avg_pace * avg_ortg / 100

        diff = abs(predicted_total - total_line)
        over_prob = 0.5
        if predicted_total > total_line:
            over_prob = min(0.55 + diff / 20, 0.85)
        else:
            over_prob = max(0.45 - diff / 20, 0.15)

        confidence = min(0.4 + diff / 30, 0.8)

        if "over_under" in self.models:
            model_result = self.models["over_under"].predict(features)
            over_prob = model_result.get("probability", over_prob)

        return {
            "predicted_total": predicted_total,
            "total_line": total_line,
            "over_probability": over_prob,
            "under_probability": 1 - over_prob,
            "recommendation": "over" if over_prob > 0.5 else "under",
            "confidence": abs(over_prob - 0.5) * 2,
        }

    async def predict_game(
        self,
        home_team_id: int,
        away_team_id: int,
        game_date: datetime,
        spread_line: float = -5.5,
        total_line: float = 220.0,
    ) -> Dict:
        moneyline = await self.predict_moneyline(home_team_id, away_team_id, game_date)
        spread = await self.predict_spread(
            home_team_id, away_team_id, game_date, spread_line
        )
        over_under = await self.predict_over_under(
            home_team_id, away_team_id, game_date, total_line
        )

        return {
            "game_date": game_date.isoformat(),
            "home_team_id": home_team_id,
            "away_team_id": away_team_id,
            "moneyline": moneyline,
            "spread": spread,
            "over_under": over_under,
            "generated_at": datetime.now().isoformat(),
        }

    def _baseline_moneyline_prediction(self, features: Dict[str, float]) -> Dict:
        home_win_pct = features.get("home_win_pct", 0.5)
        away_win_pct = features.get("away_win_pct", 0.5)

        home_net = features.get("home_net_rating_10g_avg", 0)
        away_net = features.get("away_net_rating_10g_avg", 0)

        net_diff = home_net - away_net
        win_pct_diff = home_win_pct - away_win_pct

        probability = 0.5 + (win_pct_diff * 0.8) + (net_diff / 40)
        probability = max(0.05, min(0.95, probability))

        confidence = min(abs(win_pct_diff) * 2 + abs(net_diff) / 10, 1.0)

        return {
            "prediction": 1 if probability > 0.5 else 0,
            "probability": probability,
            "confidence": confidence,
            "model_version": "v4-simulation",
        }


game_predictor = GamePredictor()
