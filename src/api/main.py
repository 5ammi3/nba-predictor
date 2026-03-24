from datetime import datetime, timedelta
from typing import Optional
from fastapi import FastAPI, HTTPException, Depends
from contextlib import asynccontextmanager
import numpy as np
import json


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def convert_numpy(obj):
    return json.loads(json.dumps(obj, cls=NumpyEncoder))


from ..utils.logger import logger
from ..utils.cache import cache_manager
from ..utils.database import init_db
from ..utils.telegram import telegram_notifier
from ..data.sportsfbi_client import sports_fbi_client
from ..prediction.game_predictor import game_predictor
from ..prediction.player_props import player_props_predictor
from ..prediction.value_calculator import value_calculator
from ..models.model_evaluation import ModelEvaluator

from .schemas import (
    GamePredictionRequest,
    GamePredictionResponse,
    PlayerPropPredictionRequest,
    PlayerPropPredictionResponse,
    HistoricalAccuracyResponse,
    ValueBetsResponse,
    ValueBet,
    HealthResponse,
    MoneylinePrediction,
    SpreadPrediction,
    OverUnderPrediction,
)
from .dependencies import (
    get_team_by_name,
    get_team_by_abbreviation,
    get_player_by_name,
    check_health,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting NBA Predictor API")
    try:
        init_db()
        logger.info("Database initialized")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")

    try:
        await cache_manager.connect()
        logger.info("Cache connected")
    except Exception as e:
        logger.warning(f"Cache connection failed: {e}")

    yield

    await cache_manager.disconnect()
    logger.info("API shutdown complete")


app = FastAPI(
    title="NBA Predictor API",
    description="NBA basketball prediction system with XGBoost and early fusion",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/", response_model=HealthResponse)
async def root():
    health = await check_health()
    return HealthResponse(
        status="healthy" if health["database"] == "healthy" else "degraded",
        timestamp=datetime.now().isoformat(),
        version="1.0.0",
    )


@app.get("/health")
async def health_check():
    return await check_health()


@app.post("/predict/game", response_model=GamePredictionResponse)
async def predict_game(request: GamePredictionRequest):
    home_team = await get_team_by_name(request.team1)
    away_team = await get_team_by_name(request.team2)

    if not home_team:
        raise HTTPException(status_code=404, detail=f"Team not found: {request.team1}")
    if not away_team:
        raise HTTPException(status_code=404, detail=f"Team not found: {request.team2}")

    try:
        game_date = datetime.strptime(request.date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(
            status_code=400, detail="Invalid date format. Use YYYY-MM-DD"
        )

    if game_date > datetime.now() + timedelta(days=30):
        raise HTTPException(
            status_code=400, detail="Cannot predict more than 30 days ahead"
        )

    prediction = await game_predictor.predict_game(
        home_team.id,
        away_team.id,
        game_date,
        request.spread_line,
        request.total_line,
    )

    prediction = convert_numpy(prediction)

    if request.notify_telegram:
        prediction["home_team"] = request.team1
        prediction["away_team"] = request.team2
        await telegram_notifier.send_prediction(prediction)

    return GamePredictionResponse(**prediction)


@app.post("/predict/player-prop", response_model=PlayerPropPredictionResponse)
async def predict_player_prop(request: PlayerPropPredictionRequest):
    player = await get_player_by_name(request.player_name)

    if not player:
        raise HTTPException(
            status_code=404, detail=f"Player not found: {request.player_name}"
        )

    if request.stat_type not in ["points", "rebounds", "assists"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid stat_type. Use points, rebounds, or assists",
        )

    try:
        game_date = datetime.strptime(request.date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(
            status_code=400, detail="Invalid date format. Use YYYY-MM-DD"
        )

    lines_map = {
        "points": 20.0,
        "rebounds": 7.5,
        "assists": 7.5,
    }
    line = request.line or lines_map.get(request.stat_type, 20.0)

    opponent_id = None
    if request.opponent:
        opponent = await get_team_by_name(request.opponent)
        if opponent:
            opponent_id = opponent.id

    prediction = await player_props_predictor.predict_player_prop(
        player.id,
        request.stat_type,
        game_date,
        line,
        opponent_id,
    )

    return PlayerPropPredictionResponse(**prediction)


@app.get("/historical/accuracy", response_model=HistoricalAccuracyResponse)
async def get_historical_accuracy(model_version: str = "v1"):
    evaluator = ModelEvaluator()
    report = evaluator.generate_full_report(model_version)

    return HistoricalAccuracyResponse(
        overall=report.get("overall", {}),
        by_bet_type=report.get("by_bet_type", {}),
    )


@app.get("/value-bets/today", response_model=ValueBetsResponse)
async def get_value_bets_today(
    min_probability: float = 0.55,
    bankroll: float = 10000.0,
):
    predictions = []

    sample_predictions = [
        {
            "prediction_type": "moneyline",
            "team_id": 1,
            "probability": 0.65,
            "odds": -120,
        },
        {
            "prediction_type": "over_under",
            "team_id": 1,
            "probability": 0.58,
            "odds": -110,
        },
    ]

    value_report = value_calculator.generate_value_bets_report(
        sample_predictions,
        bankroll,
    )

    return ValueBetsResponse(**value_report)


@app.get("/teams")
async def list_teams():
    from ..utils.database import get_session, Team

    session = get_session()
    teams = session.query(Team).all()
    return [{"id": t.id, "name": t.name, "abbreviation": t.abbreviation} for t in teams]


@app.get("/players")
async def list_players(team_id: Optional[int] = None):
    from ..utils.database import get_session, Player

    session = get_session()
    query = session.query(Player)
    if team_id:
        query = query.filter(Player.team_id == team_id)
    players = query.limit(50).all()
    return [{"id": p.id, "name": p.name, "position": p.position} for p in players]


@app.post("/pipeline/run")
async def run_pipeline(
    start_date: Optional[str] = None, end_date: Optional[str] = None
):
    from ..data.data_pipeline import pipeline

    if not start_date:
        start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    if not end_date:
        end_date = datetime.now().strftime("%Y-%m-%d")

    await pipeline.run_full_pipeline(start_date, end_date)

    return {"status": "completed", "start_date": start_date, "end_date": end_date}


@app.get("/live-games")
async def get_live_games():
    games = await sports_fbi_client.get_today_games()
    return {"date": datetime.now().strftime("%Y-%m-%d"), "games": games}


@app.post("/predict/today")
async def predict_today_games():
    games = await sports_fbi_client.get_today_games()

    if not games:
        return {"message": "No games today", "predictions": []}

    predictions = []
    errors = []

    for game in games:
        try:
            home_abbrev = game.get("home_abbrev")
            away_abbrev = game.get("away_abbrev")

            if not home_abbrev or not away_abbrev:
                continue

            home_team = await get_team_by_abbreviation(home_abbrev)
            away_team = await get_team_by_abbreviation(away_abbrev)

            if not home_team:
                errors.append(f"Team not found: {home_abbrev}")
                continue
            if not away_team:
                errors.append(f"Team not found: {away_abbrev}")
                continue

            game_date = datetime.now()
            spread_line = game.get("spread") or -5.5

            pred = await game_predictor.predict_game(
                home_team.id, away_team.id, game_date, spread_line, 220.0
            )
            pred = convert_numpy(pred)

            pred["home_team"] = home_abbrev
            pred["away_team"] = away_abbrev
            pred["game_time"] = game.get("clock", game.get("status", "Scheduled"))
            pred["game_date"] = game.get("game_date", "")

            await telegram_notifier.send_prediction(pred)
            predictions.append(pred)
        except Exception as e:
            errors.append(f"Error: {str(e)}")
            logger.error(f"Prediction error: {e}")

    return {
        "predictions_sent": len(predictions),
        "errors": errors,
        "games": predictions,
    }
