import os
import asyncio
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential
from ..utils.config import settings
from ..utils.logger import logger
from ..utils.cache import cache_manager


class SportradarClient:
    BASE_URL = "https://api.sportradar.com/nba/v7"
    RATE_LIMIT = 100

    def __init__(self):
        self.api_key = settings.sportradar_api_key
        self._client: Optional[httpx.AsyncClient] = None
        self._request_count = 0
        self._last_reset = datetime.now()

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                headers={"Accept": "application/json"},
                timeout=30.0,
            )
        return self._client

    async def _check_rate_limit(self):
        now = datetime.now()
        if (now - self._last_reset).total_seconds() >= 60:
            self._request_count = 0
            self._last_reset = now

        if self._request_count >= self.RATE_LIMIT:
            wait_time = 60 - (now - self._last_reset).total_seconds()
            if wait_time > 0:
                logger.warning(f"Rate limit reached, waiting {wait_time:.1f}s")
                await asyncio.sleep(wait_time)
                self._request_count = 0
                self._last_reset = datetime.now()

        self._request_count += 1

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    async def _request(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        await self._check_rate_limit()

        url = f"{self.BASE_URL}/{endpoint}"
        params = params or {}
        params["api_key"] = self.api_key

        cache_key = f"sportradar:{endpoint}:{':'.join(str(v) for v in params.values())}"
        cached = await cache_manager.get(cache_key)
        if cached:
            logger.debug(f"Cache hit for {endpoint}")
            return cached

        client = await self._get_client()
        try:
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            await cache_manager.set(cache_key, data, ttl=300)
            return data
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error for {endpoint}: {e}")
            raise
        except Exception as e:
            logger.error(f"Request error for {endpoint}: {e}")
            raise

    async def get_schedule(self, date: str = None) -> Dict:
        date = date or datetime.now().strftime("%Y-%m-%d")
        return await self._request(f"games/{date}/schedule")

    async def get_games_by_date_range(
        self, start_date: str, end_date: str
    ) -> List[Dict]:
        games = []
        current = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        while current <= end:
            date_str = current.strftime("%Y-%m-%d")
            try:
                schedule = await self.get_schedule(date_str)
                if "games" in schedule:
                    games.extend(schedule["games"])
            except Exception as e:
                logger.error(f"Error fetching schedule for {date_str}: {e}")
            current += timedelta(days=1)

        return games

    async def get_game_boxscore(self, game_id: str) -> Dict:
        return await self._request(f"games/{game_id}/boxscore")

    async def get_game_summary(self, game_id: str) -> Dict:
        return await self._request(f"games/{game_id}/summary")

    async def get_team_profile(self, team_id: str) -> Dict:
        return await self._request(f"teams/{team_id}/profile")

    async def get_team_season_stats(self, team_id: str, season: int = None) -> Dict:
        season = season or datetime.now().year
        return await self._request(f"teams/{team_id}/seasons/{season}/statistics")

    async def get_team_ranks(self, team_id: str) -> Dict:
        return await self._request(f"teams/{team_id}/ranks")

    async def get_player_profile(self, player_id: str) -> Dict:
        return await self._request(f"players/{player_id}/profile")

    async def get_player_season_stats(self, player_id: str, season: int = None) -> Dict:
        season = season or datetime.now().year
        return await self._request(f"players/{player_id}/seasons/{season}/statistics")

    async def get_injuries(self) -> Dict:
        return await self._request("injuries")

    async def get_league_leaders(self, season: int = None) -> Dict:
        season = season or datetime.now().year
        return await self._request(f"seasons/{season}/leaders")

    async def get_draft_prospects(self, season: int = None) -> Dict:
        season = season or datetime.now().year
        return await self._request(f"seasons/{season}/draft/prospects")

    async def get_playoff_bracket(self, season: int = None) -> Dict:
        season = season or datetime.now().year
        return await self._request(f"seasons/{season}/playoffs/brackets")

    async def get_standings(self, season: int = None) -> Dict:
        season = season or datetime.now().year
        return await self._request(f"seasons/{season}/standings")

    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None


sportradar_client = SportradarClient()
