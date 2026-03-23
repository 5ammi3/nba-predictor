import math
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np


class ValueCalculator:
    def __init__(
        self,
        kelly_fraction: float = 0.25,
        min_edge: float = 0.05,
        max_bet_pct: float = 0.02,
    ):
        self.kelly_fraction = kelly_fraction
        self.min_edge = min_edge
        self.max_bet_pct = max_bet_pct

    def american_to_probability(self, odds: int) -> float:
        if odds > 0:
            return 100 / (odds + 100)
        else:
            return abs(odds) / (abs(odds) + 100)

    def probability_to_american(self, prob: float) -> int:
        if prob >= 0.5:
            return int((prob / (1 - prob)) * 100)
        else:
            return int(-((1 - prob) / prob) * 100)

    def calculate_expected_value(
        self,
        probability: float,
        odds: int,
        bet_size: float = 100.0,
    ) -> Dict:
        implied_prob = self.american_to_probability(odds)

        decimal_odds = self._american_to_decimal(odds)

        expected_value = probability * (decimal_odds - 1) - (1 - probability)
        expected_value_pct = expected_value * 100

        edge = probability - implied_prob
        roi = (expected_value / bet_size) * 100 if bet_size > 0 else 0

        return {
            "expected_value": expected_value,
            "expected_value_pct": expected_value_pct,
            "implied_probability": implied_prob,
            "true_probability": probability,
            "edge": edge,
            "edge_pct": edge * 100,
            "roi": roi,
            "is_value": edge >= self.min_edge,
        }

    def _american_to_decimal(self, odds: int) -> float:
        if odds > 0:
            return (odds + 100) / 100
        else:
            return (100 + abs(odds)) / abs(odds)

    def calculate_kelly_bet_size(
        self,
        probability: float,
        odds: int,
        bankroll: float,
    ) -> Dict:
        implied_prob = self.american_to_probability(odds)
        edge = probability - implied_prob

        if edge <= 0:
            return {
                "bet_size": 0,
                "bet_pct": 0,
                "kelly_fraction": 0,
                "reason": "No edge",
            }

        decimal_odds = self._american_to_decimal(odds)

        b = decimal_odds - 1
        q = 1 - probability
        p = probability

        full_kelly = (b * p - q) / b

        if full_kelly <= 0:
            return {
                "bet_size": 0,
                "bet_pct": 0,
                "kelly_fraction": 0,
                "reason": "Negative edge",
            }

        fractional_kelly = full_kelly * self.kelly_fraction

        max_bet = bankroll * self.max_bet_pct
        bet_size = min(fractional_kelly * bankroll, max_bet)
        bet_pct = bet_size / bankroll if bankroll > 0 else 0

        return {
            "bet_size": round(bet_size, 2),
            "bet_pct": round(bet_pct * 100, 2),
            "kelly_fraction": self.kelly_fraction,
            "full_kelly_pct": round(full_kelly * 100, 2),
            "fractional_kelly_pct": round(fractional_kelly * 100, 2),
            "edge": edge,
            "reason": "Valid bet" if bet_size > 0 else "Edge too small",
        }

    def find_value_bets(
        self,
        predictions: List[Dict],
        odds: List[Dict],
        min_probability: float = 0.5,
    ) -> List[Dict]:
        value_bets = []

        for pred, odd in zip(predictions, odds):
            if "probability" not in pred or "odds" not in odd:
                continue

            probability = pred["probability"]
            bet_odds = odd["odds"]

            if probability < min_probability:
                continue

            ev = self.calculate_expected_value(probability, bet_odds)

            if ev["is_value"]:
                value_bet = {
                    **pred,
                    **odd,
                    **ev,
                    "timestamp": datetime.now().isoformat(),
                }
                value_bets.append(value_bet)

        value_bets.sort(key=lambda x: x.get("edge", 0), reverse=True)

        return value_bets

    def calculate_parlay_value(
        self,
        leg_probabilities: List[float],
        leg_odds: List[int],
        bet_size: float = 100.0,
    ) -> Dict:
        if len(leg_probabilities) != len(leg_odds):
            raise ValueError("Probabilities and odds must have same length")

        combined_prob = 1.0
        for p in leg_probabilities:
            combined_prob *= p

        total_odds = 1.0
        for odd in leg_odds:
            total_odds *= self._american_to_decimal(odd)

        expected_value = combined_prob * (total_odds - 1) - (1 - combined_prob)

        return {
            "combined_probability": combined_prob,
            "total_decimal_odds": total_odds,
            "expected_value": expected_value,
            "expected_value_pct": expected_value * 100,
            "is_value": expected_value > 0,
            "recommended_bet": bet_size if expected_value > 0 else 0,
        }

    def hedge_calculator(
        self,
        original_odds: int,
        hedge_odds: int,
        original_bet: float,
    ) -> Dict:
        original_decimal = self._american_to_decimal(original_odds)
        hedge_decimal = self._american_to_decimal(hedge_odds)

        original_payout = original_bet * original_decimal
        hedge_bet = original_payout / hedge_decimal

        hedge_profit = hedge_bet * (hedge_decimal - 1)

        original_profit_if_wins = original_payout - original_bet
        hedge_profit_if_hedge_wins = hedge_bet * (hedge_decimal - 1)

        return {
            "original_bet": original_bet,
            "hedge_bet": round(hedge_bet, 2),
            "total_risk": round(original_bet + hedge_bet, 2),
            "profit_if_original_wins": round(original_profit_if_wins - hedge_bet, 2),
            "profit_if_hedge_wins": round(hedge_profit_if_hedge_wins - original_bet, 2),
        }

    def generate_value_bets_report(
        self,
        predictions: List[Dict],
        bankroll: float = 10000.0,
    ) -> Dict:
        all_bets = []

        for pred in predictions:
            if "odds" in pred:
                prob = pred.get("probability", pred.get("over_probability", 0.5))
                odds = pred.get("odds", -110)

                ev = self.calculate_expected_value(prob, odds)
                kelly = self.calculate_kelly_bet_size(prob, odds, bankroll)

                all_bets.append(
                    {
                        **pred,
                        **ev,
                        **kelly,
                    }
                )

        value_bets = [b for b in all_bets if b.get("is_value", False)]
        value_bets.sort(key=lambda x: x.get("edge", 0), reverse=True)

        total_edge = sum(b.get("edge", 0) for b in value_bets)
        avg_edge = total_edge / len(value_bets) if value_bets else 0

        return {
            "total_bets_analyzed": len(all_bets),
            "value_bets_found": len(value_bets),
            "average_edge": round(avg_edge * 100, 2),
            "total_recommended_stake": sum(b.get("bet_size", 0) for b in value_bets),
            "top_value_bets": value_bets[:10],
            "generated_at": datetime.now().isoformat(),
        }


value_calculator = ValueCalculator()
