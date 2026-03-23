from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    sportradar_api_key: str = ""
    claude_api_key: str = ""
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "nba_user"
    postgres_password: str = "nba_password"
    postgres_db: str = "nba_predictor"
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"
    use_sqlite: bool = True

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
