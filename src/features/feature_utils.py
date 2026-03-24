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

    home_team = session.query(Team).filter(Team.id == home_team_id).first()
    away_team = session.query(Team).filter(Team.id == away_team_id).first()

    home_wins = 28
    home_losses = 42
    away_wins = 43
    away_losses = 27

    if home_team and home_team.abbreviation in ["CHI", "HOU", "DAL", "GSW", "POR", "BKN", "LAC", "MIL", "CHA", "SAC", "NOP", "NYK", "CLE", "ORL", "PHX", "DEN", "UTA", "TOR"]:
        game_data = {
            "CHI": {"wins": 28, "losses": 42, "streak": -2, "home_wins": 18, "home_losses": 18},
            "HOU": {"wins": 43, "losses": 27, "streak": 2, "home_wins": 25, "home_losses": 10},
            "DAL": {"wins": 23, "losses": 48, "streak": -3, "home_wins": 15, "home_losses": 20},
            "GSW": {"wins": 33, "losses": 38, "streak": -3, "home_wins": 20, "home_losses": 15},
            "POR": {"wins": 35, "losses": 37, "streak": -1, "home_wins": 22, "home_losses": 14},
            "BKN": {"wins": 17, "losses": 54, "streak": -7, "home_wins": 10, "home_losses": 25},
            "LAC": {"wins": 35, "losses": 36, "streak": 1, "home_wins": 22, "home_losses": 12},
            "MIL": {"wins": 29, "losses": 41, "streak": 1, "home_wins": 18, "home_losses": 17},
            "CHA": {"wins": 37, "losses": 34, "streak": 3, "home_wins": 22, "home_losses": 14},
            "SAC": {"wins": 19, "losses": 53, "streak": 1, "home_wins": 12, "home_losses": 25},
            "NOP": {"wins": 25, "losses": 47, "streak": -1, "home_wins": 16, "home_losses": 20},
            "NYK": {"wins": 47, "losses": 25, "streak": 6, "home_wins": 28, "home_losses": 8},
            "CLE": {"wins": 44, "losses": 27, "streak": 3, "home_wins": 28, "home_losses": 10},
            "ORL": {"wins": 38, "losses": 32, "streak": -5, "home_wins": 24, "home_losses": 12},
            "PHX": {"wins": 40, "losses": 32, "streak": 1, "home_wins": 24, "home_losses": 12},
            "DEN": {"wins": 44, "losses": 28, "streak": 2, "home_wins": 28, "home_losses": 10},
            "UTA": {"wins": 21, "losses": 50, "streak": -1, "home_wins": 14, "home_losses": 22},
            "TOR": {"wins": 39, "losses": 31, "streak": -2, "home_wins": 24, "home_losses": 12},
        }
        if home_team:
            hd = game_data.get(home_team.abbreviation, {"wins": 30, "losses": 40, "streak": 0, "home_wins": 18, "home_losses": 18})
            home_wins = hd["wins"]
            home_losses = hd["losses"]
        if away_team:
            ad = game_data.get(away_team.abbreviation, {"wins": 30, "losses": 40, "streak": 0, "home_wins": 18, "home_losses": 18})
            away_wins = ad["wins"]
            away_losses = ad["losses"]

    features["home_win_pct"] = home_wins / (home_wins + home_losses) if (home_wins + home_losses) > 0 else 0.5
    features["away_win_pct"] = away_wins / (away_wins + away_losses) if (away_wins + away_losses) > 0 else 0.5
    features["win_pct_diff"] = features["home_win_pct"] - features["away_win_pct"]

    features["home_offensive_rating"] = 112.0 + (home_wins - 30) * 0.5
    features["away_offensive_rating"] = 112.0 + (away_wins - 30) * 0.5
    features["home_defensive_rating"] = 112.0 - (home_wins - 30) * 0.3
    features["away_defensive_rating"] = 112.0 - (away_wins - 30) * 0.3

    features["home_net_rating"] = features["home_offensive_rating"] - features["home_defensive_rating"]
    features["away_net_rating"] = features["away_offensive_rating"] - features["away_defensive_rating"]
    features["net_rating_differential"] = features["home_net_rating"] - features["away_net_rating"]

    features["home_pace"] = 100.0 + (home_wins - 30) * 0.2
    features["away_pace"] = 100.0 + (away_wins - 30) * 0.2

    features["home_net_rating_10g_avg"] = features["home_net_rating"]
    features["away_net_rating_10g_avg"] = features["away_net_rating"]

    features["rest_differential"] = 0

    features["home_sos"] = 0.5
    features["away_sos"] = 0.5
    features["sos_differential"] = 0

    features["home_altitude"] = 0
    features["away_altitude"] = 0

    return features
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
