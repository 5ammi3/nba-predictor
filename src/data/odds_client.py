import os
from datetime import datetime
from typing import Optional, Dict, List
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential
from ..utils.config import settings
from ..utils.logger import logger
from ..utils.cache import cache_manager


class OddsClient:
    BASE_URL = "https://api.sportradar.com/odds/v3"
    RATE_LIMIT = 50

    def __init__(self):
        self.api_key = settings.sportradar_api_key
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                headers={"Accept": "application/json"},
                timeout=30.0,
            )
        return self._client

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    async def _request(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        url = f"{self.BASE_URL}/{endpoint}"
        params = params or {}
        params["api_key"] = self.api_key

        cache_key = f"odds:{endpoint}:{':'.join(str(v) for v in params.values())}"
        cached = await cache_manager.get(cache_key)
        if cached:
            return cached

        client = await self._get_client()
        try:
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            await cache_manager.set(cache_key, data, ttl=60)
            return data
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error for {endpoint}: {e}")
            raise
        except Exception as e:
            logger.error(f"Request error for {endpoint}: {e}")
            raise

    async def get_game_odds(self, game_id: str) -> Dict:
        return await self._request(f"games/{game_id}/odds")

    async def get_game_odds_history(self, game_id: str) -> List[Dict]:
        return await self._request(f"games/{game_id}/odds/history")

    async def get_moneyline_odds(self, game_id: str) -> Dict:
        return await self._request(f"games/{game_id}/odds/moneyline")

    async def get_spread_odds(self, game_id: str) -> Dict:
        return await self._request(f"games/{game_id}/odds/spread")

    async def get_totals_odds(self, game_id: str) -> Dict:
        return await self._request(f"games/{game_id}/odds/totals")

    async def get_live_odds(self, game_id: str) -> Dict:
        return await self._request(f"games/{game_id}/odds/live")

    async def get_today_odds(self) -> Dict:
        date = datetime.now().strftime("%Y-%m-%d")
        return await self._request(f"games/{date}/odds")

    async def get_player_prop_odds(
        self, game_id: str, player_id: str, prop_type: str
    ) -> Dict:
        return await self._request(
            f"games/{game_id}/players/{player_id}/props/{prop_type}"
        )

    async def get_consensus_lines(self, game_id: str) -> Dict:
        return await self._request(f"games/{game_id}/consensus")

    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None


odds_client = OddsClient()
