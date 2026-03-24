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
    seed_sample_data()


def seed_sample_data():
    session = get_session()
    if session.query(Team).first():
        logger.info("Sample data already exists")
        session.close()
        return

    teams_data = [
        {
            "sportradar_id": "583ec773-fb46-11e2-a2ad-00505685131a",
            "name": "Atlanta Hawks",
            "city": "Atlanta",
            "abbreviation": "ATL",
            "conference": "East",
            "division": "Southeast",
        },
        {
            "sportradar_id": "583ec773-fb46-11e2-a2ad-00505685131b",
            "name": "Boston Celtics",
            "city": "Boston",
            "abbreviation": "BOS",
            "conference": "East",
            "division": "Atlantic",
        },
        {
            "sportradar_id": "583ec773-fb46-11e2-a2ad-00505685131c",
            "name": "Brooklyn Nets",
            "city": "Brooklyn",
            "abbreviation": "BKN",
            "conference": "East",
            "division": "Atlantic",
        },
        {
            "sportradar_id": "583ec773-fb46-11e2-a2ad-00505685131d",
            "name": "Charlotte Hornets",
            "city": "Charlotte",
            "abbreviation": "CHA",
            "conference": "East",
            "division": "Southeast",
        },
        {
            "sportradar_id": "583ec773-fb46-11e2-a2ad-00505685131e",
            "name": "Chicago Bulls",
            "city": "Chicago",
            "abbreviation": "CHI",
            "conference": "East",
            "division": "Central",
        },
        {
            "sportradar_id": "583ec773-fb46-11e2-a2ad-00505685131f",
            "name": "Cleveland Cavaliers",
            "city": "Cleveland",
            "abbreviation": "CLE",
            "conference": "East",
            "division": "Central",
        },
        {
            "sportradar_id": "583ec773-fb46-11e2-a2ad-00505685131g",
            "name": "Dallas Mavericks",
            "city": "Dallas",
            "abbreviation": "DAL",
            "conference": "West",
            "division": "Southwest",
        },
        {
            "sportradar_id": "583ec773-fb46-11e2-a2ad-00505685131h",
            "name": "Denver Nuggets",
            "city": "Denver",
            "abbreviation": "DEN",
            "conference": "West",
            "division": "Northwest",
        },
        {
            "sportradar_id": "583ec773-fb46-11e2-a2ad-00505685131i",
            "name": "Detroit Pistons",
            "city": "Detroit",
            "abbreviation": "DET",
            "conference": "East",
            "division": "Central",
        },
        {
            "sportradar_id": "583ec773-fb46-11e2-a2ad-00505685131j",
            "name": "Golden State Warriors",
            "city": "Golden State",
            "abbreviation": "GSW",
            "conference": "West",
            "division": "Pacific",
        },
        {
            "sportradar_id": "583ec773-fb46-11e2-a2ad-00505685131k",
            "name": "Houston Rockets",
            "city": "Houston",
            "abbreviation": "HOU",
            "conference": "West",
            "division": "Southwest",
        },
        {
            "sportradar_id": "583ec773-fb46-11e2-a2ad-00505685131l",
            "name": "Indiana Pacers",
            "city": "Indiana",
            "abbreviation": "IND",
            "conference": "East",
            "division": "Central",
        },
        {
            "sportradar_id": "583ec773-fb46-11e2-a2ad-00505685131m",
            "name": "LA Clippers",
            "city": "Los Angeles",
            "abbreviation": "LAC",
            "conference": "West",
            "division": "Pacific",
        },
        {
            "sportradar_id": "583ec773-fb46-11e2-a2ad-00505685131n",
            "name": "Los Angeles Lakers",
            "city": "Los Angeles",
            "abbreviation": "LAL",
            "conference": "West",
            "division": "Pacific",
        },
        {
            "sportradar_id": "583ec773-fb46-11e2-a2ad-00505685131o",
            "name": "Memphis Grizzlies",
            "city": "Memphis",
            "abbreviation": "MEM",
            "conference": "West",
            "division": "Southwest",
        },
        {
            "sportradar_id": "583ec773-fb46-11e2-a2ad-00505685131p",
            "name": "Miami Heat",
            "city": "Miami",
            "abbreviation": "MIA",
            "conference": "East",
            "division": "Southeast",
        },
        {
            "sportradar_id": "583ec773-fb46-11e2-a2ad-00505685131q",
            "name": "Milwaukee Bucks",
            "city": "Milwaukee",
            "abbreviation": "MIL",
            "conference": "East",
            "division": "Central",
        },
        {
            "sportradar_id": "583ec773-fb46-11e2-a2ad-00505685131r",
            "name": "Minnesota Timberwolves",
            "city": "Minnesota",
            "abbreviation": "MIN",
            "conference": "West",
            "division": "Northwest",
        },
        {
            "sportradar_id": "583ec773-fb46-11e2-a2ad-00505685131s",
            "name": "New Orleans Pelicans",
            "city": "New Orleans",
            "abbreviation": "NOP",
            "conference": "West",
            "division": "Southwest",
        },
        {
            "sportradar_id": "583ec773-fb46-11e2-a2ad-00505685131t",
            "name": "New York Knicks",
            "city": "New York",
            "abbreviation": "NYK",
            "conference": "East",
            "division": "Atlantic",
        },
        {
            "sportradar_id": "583ec773-fb46-11e2-a2ad-00505685131u",
            "name": "Oklahoma City Thunder",
            "city": "Oklahoma City",
            "abbreviation": "OKC",
            "conference": "West",
            "division": "Northwest",
        },
        {
            "sportradar_id": "583ec773-fb46-11e2-a2ad-00505685131v",
            "name": "Orlando Magic",
            "city": "Orlando",
            "abbreviation": "ORL",
            "conference": "East",
            "division": "Southeast",
        },
        {
            "sportradar_id": "583ec773-fb46-11e2-a2ad-00505685131w",
            "name": "Philadelphia 76ers",
            "city": "Philadelphia",
            "abbreviation": "PHI",
            "conference": "East",
            "division": "Atlantic",
        },
        {
            "sportradar_id": "583ec773-fb46-11e2-a2ad-00505685131x",
            "name": "Phoenix Suns",
            "city": "Phoenix",
            "abbreviation": "PHX",
            "conference": "West",
            "division": "Pacific",
        },
        {
            "sportradar_id": "583ec773-fb46-11e2-a2ad-00505685131y",
            "name": "Portland Trail Blazers",
            "city": "Portland",
            "abbreviation": "POR",
            "conference": "West",
            "division": "Northwest",
        },
        {
            "sportradar_id": "583ec773-fb46-11e2-a2ad-00505685131z",
            "name": "Sacramento Kings",
            "city": "Sacramento",
            "abbreviation": "SAC",
            "conference": "West",
            "division": "Pacific",
        },
        {
            "sportradar_id": "583ec773-fb46-11e2-a2ad-00505685132a",
            "name": "San Antonio Spurs",
            "city": "San Antonio",
            "abbreviation": "SAS",
            "conference": "West",
            "division": "Southwest",
        },
        {
            "sportradar_id": "583ec773-fb46-11e2-a2ad-00505685132b",
            "name": "Toronto Raptors",
            "city": "Toronto",
            "abbreviation": "TOR",
            "conference": "East",
            "division": "Atlantic",
        },
        {
            "sportradar_id": "583ec773-fb46-11e2-a2ad-00505685132c",
            "name": "Utah Jazz",
            "city": "Utah",
            "abbreviation": "UTA",
            "conference": "West",
            "division": "Northwest",
        },
        {
            "sportradar_id": "583ec773-fb46-11e2-a2ad-00505685132d",
            "name": "Washington Wizards",
            "city": "Washington",
            "abbreviation": "WAS",
            "conference": "East",
            "division": "Southeast",
        },
    ]

    for team_data in teams_data:
        team = Team(**team_data)
        session.add(team)

    players_data = [
        {
            "sportradar_id": "p001",
            "name": "LeBron James",
            "position": "SF",
            "team_id": 14,
        },
        {
            "sportradar_id": "p002",
            "name": "Stephen Curry",
            "position": "PG",
            "team_id": 10,
        },
        {
            "sportradar_id": "p003",
            "name": "Kevin Durant",
            "position": "SF",
            "team_id": 13,
        },
        {
            "sportradar_id": "p004",
            "name": "Giannis Antetokounmpo",
            "position": "PF",
            "team_id": 17,
        },
        {
            "sportradar_id": "p005",
            "name": "Nikola Jokic",
            "position": "C",
            "team_id": 8,
        },
        {
            "sportradar_id": "p006",
            "name": "Luka Doncic",
            "position": "PG",
            "team_id": 7,
        },
        {
            "sportradar_id": "p007",
            "name": "Joel Embiid",
            "position": "C",
            "team_id": 23,
        },
        {
            "sportradar_id": "p008",
            "name": "Kawhi Leonard",
            "position": "SF",
            "team_id": 13,
        },
        {
            "sportradar_id": "p009",
            "name": "Anthony Davis",
            "position": "PF",
            "team_id": 14,
        },
        {
            "sportradar_id": "p010",
            "name": "Jayson Tatum",
            "position": "SF",
            "team_id": 2,
        },
        {
            "sportradar_id": "p011",
            "name": "Devin Booker",
            "position": "SG",
            "team_id": 24,
        },
        {
            "sportradar_id": "p012",
            "name": "Damian Lillard",
            "position": "PG",
            "team_id": 25,
        },
        {
            "sportradar_id": "p013",
            "name": "Bradley Beal",
            "position": "SG",
            "team_id": 30,
        },
        {
            "sportradar_id": "p014",
            "name": "Jimmy Butler",
            "position": "SF",
            "team_id": 16,
        },
        {
            "sportradar_id": "p015",
            "name": "Paul George",
            "position": "SF",
            "team_id": 13,
        },
        {
            "sportradar_id": "p016",
            "name": "James Harden",
            "position": "SG",
            "team_id": 11,
        },
        {
            "sportradar_id": "p017",
            "name": "Kyrie Irving",
            "position": "PG",
            "team_id": 3,
        },
        {
            "sportradar_id": "p018",
            "name": "Russell Westbrook",
            "position": "PG",
            "team_id": 28,
        },
        {
            "sportradar_id": "p019",
            "name": "Zion Williamson",
            "position": "PF",
            "team_id": 19,
        },
        {
            "sportradar_id": "p020",
            "name": "Donovan Mitchell",
            "position": "SG",
            "team_id": 4,
        },
    ]

    for player_data in players_data:
        player = Player(**player_data)
        session.add(player)

    session.commit()
    logger.info("Sample data seeded: 30 teams, 20 players")
    session.close()


def drop_db():
    engine = get_engine()
    Base.metadata.drop_all(engine)
    logger.info("Database dropped")
