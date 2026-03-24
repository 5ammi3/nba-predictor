import os
import httpx
from ..utils.logger import logger


class SportsFBIClient:
    BASE_URL = "https://blowoutalert.sportsfbi.com/api"

    def __init__(self):
        self.enabled = True

    async def get_today_games(self) -> dict:
        url = f"{self.BASE_URL}/nba/games/today"
        async with httpx.AsyncClient() as client:
            try:
                r = await client.get(url, timeout=15)
                if r.status_code == 200:
                    data = r.json()
                    return data.get("games", [])
                else:
                    logger.error(f"SportsFBI error: {r.status_code}")
                    return []
            except Exception as e:
                logger.error(f"SportsFBI request failed: {e}")
                return []

    async def get_game_by_id(self, game_id: str) -> dict:
        url = f"{self.BASE_URL}/nba/games/{game_id}"
        async with httpx.AsyncClient() as client:
            try:
                r = await client.get(url, timeout=15)
                if r.status_code == 200:
                    return r.json()
                return {}
            except Exception as e:
                logger.error(f"SportsFBI game fetch failed: {e}")
                return {}


sports_fbi_client = SportsFBIClient()
