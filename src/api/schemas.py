from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel, Field


class GamePredictionRequest(BaseModel):
    team1: str = Field(..., description="Home team name")
    team2: str = Field(..., description="Away team name")
    date: str = Field(..., description="Game date in YYYY-MM-DD format")
    spread_line: Optional[float] = Field(-5.5, description="Spread line")
    total_line: Optional[float] = Field(220.0, description="Over/Under line")
    notify_telegram: Optional[bool] = Field(
        True, description="Send notification to Telegram"
    )


class MoneylinePrediction(BaseModel):
    home_win_probability: float
    away_win_probability: float
    confidence: float
    model_version: str
    recommendation: str


class SpreadPrediction(BaseModel):
    predicted_spread: float
    spread_line: float
    home_covers_probability: float
    away_covers_probability: float
    recommendation: str
    confidence: float


class OverUnderPrediction(BaseModel):
    predicted_total: float
    total_line: float
    over_probability: float
    under_probability: float
    recommendation: str
    confidence: float


class GamePredictionResponse(BaseModel):
    game_date: str
    home_team_id: int
    away_team_id: int
    moneyline: MoneylinePrediction
    spread: SpreadPrediction
    over_under: OverUnderPrediction
    generated_at: str


class PlayerPropPredictionRequest(BaseModel):
    player_name: str = Field(..., description="Player name")
    stat_type: str = Field(..., description="points, rebounds, or assists")
    date: str = Field(..., description="Game date in YYYY-MM-DD format")
    line: Optional[float] = Field(None, description="Over/Under line")
    opponent: Optional[str] = Field(None, description="Opponent team name")


class PlayerPropPredictionResponse(BaseModel):
    player_id: int
    player_name: str
    prop_type: str
    line: float
    projected_value: float
    over_probability: float
    under_probability: float
    over_line: int
    under_line: int
    recommendation: str
    confidence: float
    std_error: float


class HistoricalAccuracyResponse(BaseModel):
    overall: dict
    by_bet_type: dict


class ValueBet(BaseModel):
    prediction_type: str
    player_id: Optional[int] = None
    player_name: Optional[str] = None
    team_id: Optional[int] = None
    probability: float
    odds: int
    expected_value: float
    expected_value_pct: float
    edge: float
    roi: float
    is_value: bool
    bet_size: float


class ValueBetsResponse(BaseModel):
    total_bets_analyzed: int
    value_bets_found: int
    average_edge: float
    total_recommended_stake: float
    top_value_bets: List[ValueBet]
    generated_at: str


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
