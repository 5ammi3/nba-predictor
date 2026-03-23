from datetime import datetime
from typing import Dict, List, Optional
import numpy as np
from ..models.xgboost_model import XGBoostPredictor
from ..features.feature_utils import prepare_player_prop_features
from ..utils.database import get_session, Player, Team, Game
from ..utils.logger import logger


class PlayerPropsPredictor:
    def __init__(self):
        self.session = get_session()
        self.models: Dict[str, XGBoostPredictor] = {}
        self._load_models()

    def _load_models(self):
        import os

        for prop_type in ["points", "rebounds", "assists"]:
            model_path = f"models/{prop_type}_prop_model.pkl"
            if os.path.exists(model_path):
                model = XGBoostPredictor(model_type=f"{prop_type}_prop")
                model.load(model_path)
                self.models[prop_type] = model
                logger.info(f"Loaded {prop_type} prop model")

    async def predict_player_prop(
        self,
        player_id: int,
        prop_type: str,
        game_date: datetime,
        line: float,
        opponent_team_id: Optional[int] = None,
    ) -> Dict:
        if prop_type not in ["points", "rebounds", "assists"]:
            raise ValueError(f"Invalid prop type: {prop_type}")

        player = self.session.query(Player).filter(Player.id == player_id).first()
        if not player:
            raise ValueError(f"Player not found: {player_id}")

        if opponent_team_id is None:
            game = (
                self.session.query(Game)
                .filter(Game.scheduled_date == game_date)
                .first()
            )
            if game:
                opponent_team_id = (
                    game.away_team_id
                    if game.home_team_id == player.team_id
                    else game.home_team_id
                )
            else:
                opponent_team_id = 0

        features = await prepare_player_prop_features(
            player_id, game_date, prop_type, opponent_team_id, self.session
        )

        base_projection = features.get("base_projection", line)

        minutes_factor = features.get("minutes_10g_avg", 30) / 30
        projected_value = base_projection * minutes_factor

        opp_def_adj = 1.0
        if prop_type == "points":
            opp_def_adj = 1 + (110 - features.get("opp_def_rating_avg", 110)) / 200
        elif prop_type == "rebounds":
            opp_def_adj = 1 + (110 - features.get("opp_def_rating_avg", 110)) / 300
        elif prop_type == "assists":
            opp_def_adj = 1 + (110 - features.get("opp_def_rating_avg", 110)) / 250

        projected_value *= opp_def_adj

        variance = features.get(f"{prop_type}_10g_std", 3)
        std_error = variance / np.sqrt(10)

        over_prob = 1 - (line - projected_value) / (std_error * 2)
        over_prob = max(0.05, min(0.95, over_prob))

        if prop_type in self.models:
            model_result = self.models[prop_type].predict(features)
            over_prob = model_result.get("probability", over_prob)

        return {
            "player_id": player_id,
            "player_name": player.name,
            "prop_type": prop_type,
            "line": line,
            "projected_value": round(projected_value, 1),
            "over_probability": over_prob,
            "under_probability": 1 - over_prob,
            "over_line": -110,
            "under_line": -110,
            "recommendation": "over" if over_prob > 0.5 else "under",
            "confidence": abs(over_prob - 0.5) * 2,
            "std_error": round(std_error, 2),
        }

    async def predict_all_props_for_game(
        self,
        game_id: int,
        lines: Dict[str, Dict[str, float]],
    ) -> List[Dict]:
        game = self.session.query(Game).filter(Game.id == game_id).first()
        if not game:
            raise ValueError(f"Game not found: {game_id}")

        game_date = game.scheduled_date

        players = (
            self.session.query(Player)
            .filter(
                (Player.team_id == game.home_team_id)
                | (Player.team_id == game.away_team_id)
            )
            .all()
        )

        predictions = []
        for player in players:
            for prop_type in ["points", "rebounds", "assists"]:
                player_lines = lines.get(str(player.id), {})
                line = player_lines.get(prop_type)
                if line is None:
                    continue

                opponent_id = (
                    game.away_team_id
                    if player.team_id == game.home_team_id
                    else game.home_team_id
                )

                pred = await self.predict_player_prop(
                    player.id, prop_type, game_date, line, opponent_id
                )
                predictions.append(pred)

        return predictions


player_props_predictor = PlayerPropsPredictor()
