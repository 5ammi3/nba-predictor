from typing import Optional
from datetime import datetime
from sqlalchemy import text
from ..utils.database import get_session, Team, Game, Player, Odds, Prediction
from ..utils.cache import cache_manager
from ..utils.logger import logger
from ..data.data_pipeline import pipeline
from .schemas import (
    GamePredictionRequest,
    PlayerPropPredictionRequest,
)


async def get_team_by_name(name: str):
    session = get_session()
    team = session.query(Team).filter(Team.name.ilike(f"%{name}%")).first()
    if not team:
        team = session.query(Team).filter(Team.abbreviation == name.upper()).first()
    return team


async def get_team_by_abbreviation(abbrev: str):
    session = get_session()
    return session.query(Team).filter(Team.abbreviation == abbrev.upper()).first()


async def get_player_by_name(name: str):
    session = get_session()
    player = session.query(Player).filter(Player.name.ilike(f"%{name}%")).first()
    return player


async def get_or_create_game(home_team_id: int, away_team_id: int, game_date: datetime):
    session = get_session()
    game = (
        session.query(Game)
        .filter(Game.home_team_id == home_team_id)
        .filter(Game.away_team_id == away_team_id)
        .filter(Game.scheduled_date == game_date)
        .first()
    )
    return game


async def get_odds_for_game(game_id: int):
    session = get_session()
    odds = session.query(Odds).filter(Odds.game_id == game_id).all()
    return odds


async def save_prediction(
    game_id: Optional[int],
    player_id: Optional[int],
    prediction_type: str,
    prediction_value: float,
    confidence: float,
    model_version: str,
    features: dict,
):
    session = get_session()
    prediction = Prediction(
        game_id=game_id,
        player_id=player_id,
        prediction_type=prediction_type,
        prediction_value=prediction_value,
        confidence=confidence,
        model_version=model_version,
        features=features,
    )
    session.add(prediction)
    session.commit()
    return prediction


async def resolve_prediction(prediction_id: int, actual_value: float):
    session = get_session()
    prediction = (
        session.query(Prediction).filter(Prediction.id == prediction_id).first()
    )
    if prediction:
        prediction.actual_value = actual_value
        prediction.resolved_at = datetime.now()
        session.commit()
    return prediction


async def check_health():
    try:
        session = get_session()
        session.execute(text("SELECT 1"))
        db_status = "healthy"
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        db_status = "unhealthy"

    try:
        await cache_manager.connect()
        cache_status = "healthy"
    except Exception as e:
        logger.error(f"Redis health check failed: {e}")
        cache_status = "unhealthy"

    return {
        "database": db_status,
        "cache": cache_status,
    }
