from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from sqlalchemy.orm import Session
from ..utils.database import get_session, Team, Game, TeamStats, Player, PlayerStats
from .structured_features import StructuredFeatures, get_player_features
from .text_embeddings import embeddings_processor


def prepare_game_features(
    home_team_id: int,
    away_team_id: int,
    game_date: datetime,
    session: Optional[Session] = None,
) -> Dict[str, float]:
    session = session or get_session()
    structured = StructuredFeatures(session)

    features = {}

    home_features = structured.get_team_features(home_team_id, game_date)
    away_features = structured.get_team_features(away_team_id, game_date)

    for key, value in home_features.items():
        features[f"home_{key}"] = value

    for key, value in away_features.items():
        features[f"away_{key}"] = value

    home_rest = structured.get_rest_features(home_team_id, game_date)
    away_rest = structured.get_rest_features(away_team_id, game_date)

    for key, value in home_rest.items():
        features[f"home_{key}"] = value

    for key, value in away_rest.items():
        features[f"away_{key}"] = value

    features["rest_differential"] = home_rest.get("days_rest", 0) - away_rest.get(
        "days_rest", 0
    )

    home_sos = structured.get_sos_features(home_team_id, game_date)
    away_sos = structured.get_sos_features(away_team_id, game_date)

    features["home_sos"] = home_sos.get("sos", 0)
    features["away_sos"] = away_sos.get("sos", 0)
    features["sos_differential"] = home_sos.get("sos", 0) - away_sos.get("sos", 0)

    features["home_altitude"] = structured.get_altitude_boost(
        home_team_id, is_home=True
    )
    features["away_altitude"] = structured.get_altitude_boost(
        away_team_id, is_home=False
    )

    features["home_net_rating_10g_avg"] = home_features.get("net_rating_10g_avg", 0)
    features["away_net_rating_10g_avg"] = away_features.get("net_rating_10g_avg", 0)
    features["net_rating_differential"] = (
        features["home_net_rating_10g_avg"] - features["away_net_rating_10g_avg"]
    )

    features["home_pace_10g_avg"] = home_features.get("pace_10g_avg", 0)
    features["away_pace_10g_avg"] = away_features.get("pace_10g_avg", 0)
    features["pace_differential"] = (
        features["home_pace_10g_avg"] - features["away_pace_10g_avg"]
    )

    return features


async def prepare_player_prop_features(
    player_id: int,
    game_date: datetime,
    prop_type: str,
    opponent_team_id: int,
    session: Optional[Session] = None,
) -> Dict[str, float]:
    session = session or get_session()

    features = get_player_features(player_id, game_date, session)

    player = session.query(Player).filter(Player.id == player_id).first()
    if player:
        injuries = (
            session.query(PlayerStats)
            .filter(PlayerStats.player_id == player_id)
            .order_by(PlayerStats.date.desc())
            .limit(1)
            .first()
        )
        if injuries:
            features["recent_games_played"] = 1

    opp_stats = (
        session.query(TeamStats)
        .filter(TeamStats.team_id == opponent_team_id)
        .order_by(TeamStats.date.desc())
        .limit(10)
        .all()
    )

    if opp_stats:
        features["opp_def_rating_avg"] = np.mean(
            [s.defensive_rating for s in opp_stats if s.defensive_rating]
        )
        features["opp_pace_avg"] = np.mean([s.pace for s in opp_stats if s.pace])
    else:
        features["opp_def_rating_avg"] = 110.0
        features["opp_pace_avg"] = 100.0

    if prop_type == "points":
        features["base_projection"] = features.get("points_10g_avg", 0)
    elif prop_type == "rebounds":
        features["base_projection"] = features.get("rebounds_10g_avg", 0)
    elif prop_type == "assists":
        features["base_projection"] = features.get("assists_10g_avg", 0)
    else:
        features["base_projection"] = 0

    return features


def add_contextual_features(
    features: Dict[str, float],
    game_date: datetime,
    season: int,
) -> Dict[str, float]:
    month = game_date.month

    if month >= 4 and month <= 6:
        features["is_playoff"] = 1.0
    else:
        features["is_playoff"] = 0.0

    if month >= 2 and month <= 3:
        features["is_trade_deadline"] = 1.0
    else:
        features["is_trade_deadline"] = 0.0

    day_of_week = game_date.weekday()
    features["is_weekend"] = 1.0 if day_of_week >= 5 else 0.0

    return features


def normalize_features(
    features: Dict[str, float],
    feature_means: Dict[str, float],
    feature_stds: Dict[str, float],
) -> Dict[str, float]:
    normalized = {}
    for key, value in features.items():
        if key in feature_means and feature_stds.get(key, 0) > 0:
            normalized[key] = (value - feature_means[key]) / feature_stds[key]
        else:
            normalized[key] = value
    return normalized
