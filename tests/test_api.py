import pytest
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api.main import app
from src.api.schemas import GamePredictionRequest
from src.prediction.value_calculator import ValueCalculator
from src.prediction.game_predictor import GamePredictor


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    return TestClient(app)


class TestAPI:
    def test_root(self, client):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "timestamp" in data

    def test_health_check(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_list_teams(self, client):
        response = client.get("/teams")
        assert response.status_code == 200

    def test_list_players(self, client):
        response = client.get("/players")
        assert response.status_code == 200


class TestValueCalculator:
    def test_american_to_probability_positive(self):
        calc = ValueCalculator()
        prob = calc.american_to_probability(150)
        assert 0.3 < prob < 0.5

    def test_american_to_probability_negative(self):
        calc = ValueCalculator()
        prob = calc.american_to_probability(-150)
        assert 0.5 < prob < 0.7

    def test_probability_to_american(self):
        calc = ValueCalculator()
        odds = calc.probability_to_american(0.6)
        assert odds > 0

    def test_calculate_expected_value_with_edge(self):
        calc = ValueCalculator()
        ev = calc.calculate_expected_value(0.6, -110)
        assert ev["expected_value"] > 0
        assert ev["is_value"] == True

    def test_calculate_expected_value_no_edge(self):
        calc = ValueCalculator()
        ev = calc.calculate_expected_value(0.5, -110)
        assert ev["expected_value"] < 0
        assert ev["is_value"] == False

    def test_calculate_kelly_bet_size(self):
        calc = ValueCalculator()
        kelly = calc.calculate_kelly_bet_size(0.6, -110, 1000)
        assert "bet_size" in kelly
        assert "bet_pct" in kelly
        assert kelly["bet_size"] > 0

    def test_calculate_kelly_no_edge(self):
        calc = ValueCalculator()
        kelly = calc.calculate_kelly_bet_size(0.45, -110, 1000)
        assert kelly["bet_size"] == 0

    def test_hedge_calculator(self):
        calc = ValueCalculator()
        hedge = calc.hedge_calculator(-150, 140, 100)
        assert "hedge_bet" in hedge
        assert "profit_if_original_wins" in hedge
        assert "profit_if_hedge_wins" in hedge


class TestGamePredictor:
    def test_baseline_moneyline_prediction(self):
        predictor = GamePredictor()

        features = {
            "home_net_rating_10g_avg": 5.0,
            "away_net_rating_10g_avg": -2.0,
            "net_rating_differential": 7.0,
        }

        result = predictor._baseline_moneyline_prediction(features)

        assert "probability" in result
        assert "confidence" in result
        assert result["probability"] > 0.5


class TestSchemas:
    def test_game_prediction_request(self):
        req = GamePredictionRequest(
            team1="Los Angeles Lakers", team2="Boston Celtics", date="2026-03-22"
        )
        assert req.team1 == "Los Angeles Lakers"
        assert req.spread_line == -5.5

    def test_game_prediction_request_custom_lines(self):
        req = GamePredictionRequest(
            team1="Los Angeles Lakers",
            team2="Boston Celtics",
            date="2026-03-22",
            spread_line=-3.5,
            total_line=230.0,
        )
        assert req.spread_line == -3.5
        assert req.total_line == 230.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
