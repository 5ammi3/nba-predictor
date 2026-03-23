import os
import json
from datetime import datetime
from typing import Optional, Any
import redis.asyncio as redis
from .config import settings
from .logger import logger


class CacheManager:
    def __init__(self):
        self._redis: Optional[redis.Redis] = None
        self.ttl_default = 3600
        self.ttl_embeddings = 21600

    async def connect(self):
        if self._redis is None:
            self._redis = await redis.Redis(
                host=settings.redis_host,
                port=settings.redis_port,
                db=settings.redis_db,
                decode_responses=True,
            )
            logger.info("Connected to Redis")

    async def disconnect(self):
        if self._redis:
            await self._redis.close()
            self._redis = None

    async def get(self, key: str) -> Optional[Any]:
        if not self._redis:
            await self.connect()
        try:
            value = await self._redis.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None):
        if not self._redis:
            await self.connect()
        try:
            ttl = ttl or self.ttl_default
            await self._redis.set(key, json.dumps(value), ex=ttl)
        except Exception as e:
            logger.error(f"Cache set error: {e}")

    async def delete(self, key: str):
        if not self._redis:
            await self.connect()
        try:
            await self._redis.delete(key)
        except Exception as e:
            logger.error(f"Cache delete error: {e}")

    async def get_pattern(self, pattern: str) -> list:
        if not self._redis:
            await self.connect()
        try:
            keys = await self._redis.keys(pattern)
            return keys
        except Exception as e:
            logger.error(f"Cache pattern error: {e}")
            return []


cache_manager = CacheManager()
