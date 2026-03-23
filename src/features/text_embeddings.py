import hashlib
import json
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np
from ..utils.config import settings
from ..utils.logger import logger
from ..utils.cache import cache_manager


class TextEmbeddings:
    def __init__(self):
        self.api_key = settings.claude_api_key
        self.embedding_model = "claude-embedding-1"
        self.embedding_dim = 1536
        self.ttl = 21600

    def _hash_text(self, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    async def get_embedding(self, text: str) -> List[float]:
        cache_key = f"embedding:{self._hash_text(text)}"
        cached = await cache_manager.get(cache_key)
        if cached:
            return cached

        if not self.api_key or self.api_key == "your_claude_api_key_here":
            logger.warning("Claude API key not set, returning mock embedding")
            return self._generate_mock_embedding(text)

        try:
            embedding = await self._call_claude_api(text)
            await cache_manager.set(cache_key, embedding, ttl=self.ttl)
            return embedding
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            return self._generate_mock_embedding(text)

    async def _call_claude_api(self, text: str) -> List[float]:
        import httpx

        url = "https://api.anthropic.com/v1/embeddings"
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        }
        payload = {
            "model": "claude-embedding-1",
            "input": text,
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            return data["embedding"][0]["embedding"]

    def _generate_mock_embedding(self, text: str) -> List[float]:
        np.random.seed(hash(text) % (2**32))
        embedding = np.random.randn(self.embedding_dim)
        embedding = embedding / np.linalg.norm(embedding)
        return embedding.tolist()

    async def generate_injury_embedding(
        self,
        player_name: str,
        injury_status: str,
        injury_type: str,
        note: Optional[str] = None,
    ) -> List[float]:
        text_parts = [
            f"Player: {player_name}",
            f"Status: {injury_status}",
            f"Injury: {injury_type}",
        ]
        if note:
            text_parts.append(f"Note: {note}")

        text = " | ".join(text_parts)
        return await self.get_embedding(text)

    async def generate_news_embedding(
        self,
        headline: str,
        summary: str,
        source: Optional[str] = None,
    ) -> List[float]:
        text_parts = [f"Headline: {headline}", f"Summary: {summary}"]
        if source:
            text_parts.append(f"Source: {source}")

        text = " | ".join(text_parts)
        return await self.get_embedding(text)

    async def generate_team_news_embedding(
        self,
        team_name: str,
        news_type: str,
        description: str,
    ) -> List[float]:
        text = f"Team: {team_name} | Type: {news_type} | Description: {description}"
        return await self.get_embedding(text)

    async def batch_generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            embedding = await self.get_embedding(text)
            embeddings.append(embedding)
        return embeddings

    def cosine_similarity(self, emb1: List[float], emb2: List[float]) -> float:
        vec1 = np.array(emb1)
        vec2 = np.array(emb2)
        return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))


embeddings_processor = TextEmbeddings()
