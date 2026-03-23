from datetime import datetime
from typing import Optional, List
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Float,
    DateTime,
    Boolean,
    Text,
    ForeignKey,
    JSON,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from .config import settings
from .logger import logger


Base = declarative_base()


class Team(Base):
    __tablename__ = "teams"

    id = Column(Integer, primary_key=True)
    sportradar_id = Column(String(50), unique=True)
    name = Column(String(100))
    city = Column(String(100))
    abbreviation = Column(String(10))
    conference = Column(String(20))
    division = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)


class Game(Base):
    __tablename__ = "games"

    id = Column(Integer, primary_key=True)
    sportradar_id = Column(String(50), unique=True)
    scheduled_date = Column(DateTime)
    home_team_id = Column(Integer, ForeignKey("teams.id"))
    away_team_id = Column(Integer, ForeignKey("teams.id"))
    home_score = Column(Integer)
    away_score = Column(Integer)
    status = Column(String(20))
    season = Column(Integer)
    season_type = Column(String(20))
    created_at = Column(DateTime, default=datetime.utcnow)

    home_team = relationship("Team", foreign_keys=[home_team_id])
    away_team = relationship("Team", foreign_keys=[away_team_id])


class TeamStats(Base):
    __tablename__ = "team_stats"

    id = Column(Integer, primary_key=True)
    game_id = Column(Integer, ForeignKey("games.id"))
    team_id = Column(Integer, ForeignKey("teams.id"))
    date = Column(DateTime)
    offensive_rating = Column(Float)
    defensive_rating = Column(Float)
    net_rating = Column(Float)
    pace = Column(Float)
    effective_fg_pct = Column(Float)
    turnover_pct = Column(Float)
    rebound_rate = Column(Float)
    free_throw_rate = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

    team = relationship("Team")


class Player(Base):
    __tablename__ = "players"

    id = Column(Integer, primary_key=True)
    sportradar_id = Column(String(50), unique=True)
    name = Column(String(100))
    position = Column(String(10))
    team_id = Column(Integer, ForeignKey("teams.id"))
    created_at = Column(DateTime, default=datetime.utcnow)

    team = relationship("Team")


class PlayerStats(Base):
    __tablename__ = "player_stats"

    id = Column(Integer, primary_key=True)
    game_id = Column(Integer, ForeignKey("games.id"))
    player_id = Column(Integer, ForeignKey("players.id"))
    team_id = Column(Integer, ForeignKey("teams.id"))
    date = Column(DateTime)
    points = Column(Float)
    rebounds = Column(Float)
    assists = Column(Float)
    minutes = Column(Float)
    steals = Column(Float)
    blocks = Column(Float)
    turnovers = Column(Float)
    field_goals_made = Column(Float)
    field_goals_attempted = Column(Float)
    three_points_made = Column(Float)
    three_points_attempted = Column(Float)
    free_throws_made = Column(Float)
    free_throws_attempted = Column(Float)
    per = Column(Float)
    true_shooting_pct = Column(Float)
    usage_rate = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

    player = relationship("Player")
    team = relationship("Team")


class InjuryReport(Base):
    __tablename__ = "injury_reports"

    id = Column(Integer, primary_key=True)
    player_id = Column(Integer, ForeignKey("players.id"))
    status = Column(String(50))
    injury_type = Column(Text)
    note = Column(Text)
    reported_date = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)

    player = relationship("Player")


class Odds(Base):
    __tablename__ = "odds"

    id = Column(Integer, primary_key=True)
    game_id = Column(Integer, ForeignKey("games.id"))
    sportsbook = Column(String(50))
    moneyline_home = Column(Integer)
    moneyline_away = Column(Integer)
    spread_home = Column(Float)
    spread_away = Column(Float)
    spread_line_home = Column(Float)
    spread_line_away = Column(Float)
    total_over = Column(Float)
    total_under = Column(Float)
    total_line = Column(Float)
    updated_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)

    game = relationship("Game")


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True)
    game_id = Column(Integer, ForeignKey("games.id"), nullable=True)
    player_id = Column(Integer, ForeignKey("players.id"), nullable=True)
    prediction_type = Column(String(50))
    prediction_value = Column(Float)
    actual_value = Column(Float, nullable=True)
    confidence = Column(Float)
    model_version = Column(String(20))
    features = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    resolved_at = Column(DateTime, nullable=True)

    game = relationship("Game")
    player = relationship("Player")


class TextEmbedding(Base):
    __tablename__ = "text_embeddings"

    id = Column(Integer, primary_key=True)
    text_hash = Column(String(64), unique=True)
    text_content = Column(Text)
    embedding = Column(JSON)
    embedding_model = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)


def get_database_url() -> str:
    use_sqlite = getattr(settings, "use_sqlite", True)
    if use_sqlite:
        return "sqlite:///nba_predictor.db"
    return f"postgresql://{settings.postgres_user}:{settings.postgres_password}@{settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}"


def get_engine():
    return create_engine(get_database_url(), pool_pre_ping=True)


def get_session():
    engine = get_engine()
    Session = sessionmaker(bind=engine)
    return Session()


def init_db():
    engine = get_engine()
    Base.metadata.create_all(engine)
    logger.info("Database initialized")


def drop_db():
    engine = get_engine()
    Base.metadata.drop_all(engine)
    logger.info("Database dropped")
