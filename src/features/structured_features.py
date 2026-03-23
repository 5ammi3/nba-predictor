from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from sqlalchemy.orm import Session
import pandas as pd
import numpy as np
from ..utils.database import get_session, Team, TeamStats, Game, Player, PlayerStats


def calculate_rolling_average(
    values: List[float], window: int, recency_weight: float = 1.0
) -> float:
    if not values:
        return 0.0
    if len(values) <= window:
        return np.mean(values)
    recent = values[-window:]
    weights = np.linspace(1, recency_weight, len(recent))
    return np.average(recent, weights=weights)


def calculate_pace_adjusted_stats(
    stat: float, team_pace: float, league_pace: float
) -> float:
    if league_pace == 0:
        return stat
    return stat * (league_pace / team_pace)


def calculate_sos(teams: List[str], session: Session) -> float:
    sos_values = []
    for team_sr_id in teams:
        team = session.query(Team).filter(Team.sportradar_id == team_sr_id).first()
        if team:
            stats = (
                session.query(TeamStats)
                .filter(TeamStats.team_id == team.id)
                .order_by(TeamStats.date.desc())
                .limit(10)
                .all()
            )
            if stats:
                sos_values.append(np.mean([s.net_rating for s in stats]))
    return np.mean(sos_values) if sos_values else 0.0


class StructuredFeatures:
    def __init__(self, session: Optional[Session] = None):
        self.session = session or get_session()

    def get_team_features(
        self, team_id: int, as_of_date: datetime, windows: List[int] = [5, 10, 20]
    ) -> Dict[str, float]:
        features = {}

        for window in windows:
            start_date = as_of_date - timedelta(days=window * 3)
            stats = (
                self.session.query(TeamStats)
                .filter(TeamStats.team_id == team_id)
                .filter(TeamStats.date <= as_of_date)
                .filter(TeamStats.date >= start_date)
                .order_by(TeamStats.date.desc())
                .limit(window)
                .all()
            )

            if stats:
                off_ratings = [s.offensive_rating for s in stats if s.offensive_rating]
                def_ratings = [s.defensive_rating for s in stats if s.defensive_rating]
                net_ratings = [s.net_rating for s in stats if s.net_rating]
                paces = [s.pace for s in stats if s.pace]
                efgs = [s.effective_fg_pct for s in stats if s.effective_fg_pct]

                features[f"offensive_rating_{window}g_avg"] = (
                    np.mean(off_ratings) if off_ratings else 0.0
                )
                features[f"defensive_rating_{window}g_avg"] = (
                    np.mean(def_ratings) if def_ratings else 0.0
                )
                features[f"net_rating_{window}g_avg"] = (
                    np.mean(net_ratings) if net_ratings else 0.0
                )
                features[f"pace_{window}g_avg"] = np.mean(paces) if paces else 0.0
                features[f"efg_pct_{window}g_avg"] = np.mean(efgs) if efgs else 0.0
            else:
                features[f"offensive_rating_{window}g_avg"] = 0.0
                features[f"defensive_rating_{window}g_avg"] = 0.0
                features[f"net_rating_{window}g_avg"] = 0.0
                features[f"pace_{window}g_avg"] = 0.0
                features[f"efg_pct_{window}g_avg"] = 0.0

        recent_games = (
            self.session.query(Game)
            .filter((Game.home_team_id == team_id) | (Game.away_team_id == team_id))
            .filter(Game.scheduled_date <= as_of_date)
            .filter(Game.status == "closed")
            .order_by(Game.scheduled_date.desc())
            .limit(5)
            .all()
        )

        home_wins = 0
        away_wins = 0
        home_games = 0
        away_games = 0

        for game in recent_games:
            if game.home_team_id == team_id:
                home_games += 1
                if game.home_score > game.away_score:
                    home_wins += 1
            else:
                away_games += 1
                if game.away_score > game.home_score:
                    away_wins += 1

        features["home_win_pct_recent"] = (
            home_wins / home_games if home_games > 0 else 0.5
        )
        features["away_win_pct_recent"] = (
            away_wins / away_games if away_games > 0 else 0.5
        )

        return features

    def get_rest_features(self, team_id: int, game_date: datetime) -> Dict[str, float]:
        prev_game = (
            self.session.query(Game)
            .filter((Game.home_team_id == team_id) | (Game.away_team_id == team_id))
            .filter(Game.scheduled_date < game_date)
            .filter(Game.status == "closed")
            .order_by(Game.scheduled_date.desc())
            .first()
        )

        if prev_game:
            days_rest = (game_date - prev_game.scheduled_date).days
            features = {
                "days_rest": days_rest,
                "is_back_to_back": 1 if days_rest <= 1 else 0,
                "is_rest_day": 1 if days_rest >= 2 else 0,
            }
        else:
            features = {"days_rest": 7, "is_back_to_back": 0, "is_rest_day": 1}

        return features

    def get_sos_features(self, team_id: int, as_of_date: datetime) -> Dict[str, float]:
        team = self.session.query(Team).filter(Team.id == team_id).first()
        if not team:
            return {"sos": 0.0, "sos_recent": 0.0}

        season_start = datetime(as_of_date.year, 10, 1)
        if as_of_date.month < 10:
            season_start = datetime(as_of_date.year - 1, 10, 1)

        opponents = []
        games = (
            self.session.query(Game)
            .filter((Game.home_team_id == team_id) | (Game.away_team_id == team_id))
            .filter(Game.scheduled_date >= season_start)
            .filter(Game.scheduled_date < as_of_date)
            .filter(Game.status == "closed")
            .all()
        )

        for game in games:
            opp_id = (
                game.away_team_id if game.home_team_id == team_id else game.home_team_id
            )
            opponents.append(opp_id)

        opp_ratings = []
        for opp_id in opponents:
            opp_stats = (
                self.session.query(TeamStats)
                .filter(TeamStats.team_id == opp_id)
                .filter(TeamStats.date <= as_of_date)
                .order_by(TeamStats.date.desc())
                .limit(10)
                .all()
            )
            if opp_stats:
                opp_ratings.append(np.mean([s.net_rating for s in opp_stats]))

        return {
            "sos": np.mean(opp_ratings) if opp_ratings else 0.0,
            "num_opponents_played": len(opp_ratings),
        }

    def get_altitude_boost(self, team_id: int, is_home: bool) -> float:
        high_altitude_teams = {
            "Denver Nuggets": 5280,
            "Utah Jazz": 4220,
            "Phoenix Suns": 1035,
            "Los Angeles Lakers": 71,
        }

        team = self.session.query(Team).filter(Team.id == team_id).first()
        if team and team.name in high_altitude_teams:
            return high_altitude_teams[team.name] / 1000 * (1 if is_home else -1)
        return 0.0


def get_player_features(
    player_id: int, as_of_date: datetime, session: Session
) -> Dict[str, float]:
    features = {}

    for window in [5, 10, 20]:
        start_date = as_of_date - timedelta(days=window * 2)
        stats = (
            session.query(PlayerStats)
            .filter(PlayerStats.player_id == player_id)
            .filter(PlayerStats.date <= as_of_date)
            .filter(PlayerStats.date >= start_date)
            .order_by(PlayerStats.date.desc())
            .limit(window)
            .all()
        )

        if stats:
            points = [s.points for s in stats if s.points]
            rebounds = [s.rebounds for s in stats if s.rebounds]
            assists = [s.assists for s in stats if s.assists]
            minutes = [s.minutes for s in stats if s.minutes]

            features[f"points_{window}g_avg"] = np.mean(points) if points else 0.0
            features[f"rebounds_{window}g_avg"] = np.mean(rebounds) if rebounds else 0.0
            features[f"assists_{window}g_avg"] = np.mean(assists) if assists else 0.0
            features[f"minutes_{window}g_avg"] = np.mean(minutes) if minutes else 0.0
            features[f"points_{window}g_std"] = (
                np.std(points) if len(points) > 1 else 0.0
            )
        else:
            features[f"points_{window}g_avg"] = 0.0
            features[f"rebounds_{window}g_avg"] = 0.0
            features[f"assists_{window}g_avg"] = 0.0
            features[f"minutes_{window}g_avg"] = 0.0
            features[f"points_{window}g_std"] = 0.0

    return features
