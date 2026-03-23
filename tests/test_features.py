import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features.structured_features import (
    calculate_rolling_average,
    calculate_pace_adjusted_stats,
    StructuredFeatures,
    get_player_features,
)
from src.features.feature_utils import prepare_game_features, normalize_features
from src.features.text_embeddings import TextEmbeddings


class TestStructuredFeatures:
    def test_calculate_rolling_average(self):
        values = [10, 12, 14, 16, 18, 20]

        avg_5 = calculate_rolling_average(values, 5)

        assert avg_5 == 16.0

    def test_calculate_rolling_average_with_recency(self):
        values = [10, 12, 14, 16, 18, 20]

        avg_5_weighted = calculate_rolling_average(values, 5, recency_weight=1.5)

        assert avg_5_weighted >= 14.0

    def test_calculate_pace_adjusted_stats(self):
        stat = 100.0
        team_pace = 105.0
        league_pace = 100.0

        adjusted = calculate_pace_adjusted_stats(stat, team_pace, league_pace)

        assert adjusted != stat

    def test_calculate_pace_adjusted_stats_zero_league_pace(self):
        stat = 100.0
        team_pace = 105.0
        league_pace = 0

        adjusted = calculate_pace_adjusted_stats(stat, team_pace, league_pace)

        assert adjusted == stat


class TestTextEmbeddings:
    def test_hash_text(self):
        embeddings = TextEmbeddings()

        hash1 = embeddings._hash_text("test text")
        hash2 = embeddings._hash_text("test text")
        hash3 = embeddings._hash_text("different text")

        assert hash1 == hash2
        assert hash1 != hash3

    @pytest.mark.asyncio
    async def test_generate_mock_embedding(self):
        embeddings = TextEmbeddings()
        embeddings.api_key = ""

        emb = embeddings._generate_mock_embedding("test text")

        assert len(emb) == 1536
        assert all(-1 <= v <= 1 for v in emb)

    def test_cosine_similarity(self):
        embeddings = TextEmbeddings()

        emb1 = [1.0, 0.0, 0.0]
        emb2 = [1.0, 0.0, 0.0]
        emb3 = [0.0, 1.0, 0.0]

        sim_same = embeddings.cosine_similarity(emb1, emb2)
        sim_diff = embeddings.cosine_similarity(emb1, emb3)

        assert sim_same > 0.99
        assert sim_diff < 0.1


class TestFeatureUtils:
    def test_normalize_features(self):
        features = {
            "feature1": 10.0,
            "feature2": 20.0,
            "feature3": 30.0,
        }
        feature_means = {
            "feature1": 10.0,
            "feature2": 20.0,
            "feature3": 30.0,
        }
        feature_stds = {
            "feature1": 2.0,
            "feature2": 4.0,
            "feature3": 6.0,
        }

        normalized = normalize_features(features, feature_means, feature_stds)

        assert normalized["feature1"] == 0.0
        assert normalized["feature2"] == 0.0
        assert normalized["feature3"] == 0.0

    def test_normalize_features_with_zero_std(self):
        features = {
            "feature1": 10.0,
        }
        feature_means = {
            "feature1": 10.0,
        }
        feature_stds = {
            "feature1": 0.0,
        }

        normalized = normalize_features(features, feature_means, feature_stds)

        assert normalized["feature1"] == 10.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
