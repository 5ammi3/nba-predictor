import asyncio
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from sqlalchemy.orm import Session
from ..utils.database import (
    get_session,
    init_db,
    Team,
    Game,
    TeamStats,
    Player,
    PlayerStats,
    InjuryReport,
    Odds,
)
from ..utils.logger import logger
from .sportradar_client import sportradar_client
from .odds_client import odds_client


class DataPipeline:
    def __init__(self):
        self.session: Optional[Session] = None

    async def initialize(self):
        init_db()
        self.session = get_session()
        logger.info("Data pipeline initialized")

    def close(self):
        if self.session:
            self.session.close()

    async def sync_teams(self) -> int:
        logger.info("Syncing teams")
        count = 0
        try:
            teams = await sportradar_client._request("teams")
            if "teams" in teams:
                for team_data in teams["teams"]:
                    existing = (
                        self.session.query(Team)
                        .filter(Team.sportradar_id == team_data.get("id"))
                        .first()
                    )
                    if not existing:
                        team = Team(
                            sportradar_id=team_data.get("id"),
                            name=team_data.get("name"),
                            city=team_data.get("city"),
                            abbreviation=team_data.get("abbreviation"),
                            conference=team_data.get("conference"),
                            division=team_data.get("division"),
                        )
                        self.session.add(team)
                        count += 1
                self.session.commit()
                logger.info(f"Added {count} new teams")
        except Exception as e:
            self.session.rollback()
            logger.error(f"Error syncing teams: {e}")
        return count

    async def sync_games(self, start_date: str, end_date: str) -> int:
        logger.info(f"Syncing games from {start_date} to {end_date}")
        count = 0
        try:
            games = await sportradar_client.get_games_by_date_range(
                start_date, end_date
            )
            for game_data in games:
                existing = (
                    self.session.query(Game)
                    .filter(Game.sportradar_id == game_data.get("id"))
                    .first()
                )
                if not existing:
                    home_team = (
                        self.session.query(Team)
                        .filter(
                            Team.sportradar_id == game_data.get("home", {}).get("id")
                        )
                        .first()
                    )
                    away_team = (
                        self.session.query(Team)
                        .filter(
                            Team.sportradar_id == game_data.get("away", {}).get("id")
                        )
                        .first()
                    )
                    if home_team and away_team:
                        game = Game(
                            sportradar_id=game_data.get("id"),
                            scheduled_date=datetime.fromisoformat(
                                game_data.get("scheduled").replace("Z", "+00:00")
                            ),
                            home_team_id=home_team.id,
                            away_team_id=away_team.id,
                            home_score=game_data.get("home_score"),
                            away_score=game_data.get("away_score"),
                            status=game_data.get("status"),
                            season=game_data.get("season"),
                            season_type=game_data.get("season_type"),
                        )
                        self.session.add(game)
                        count += 1
            self.session.commit()
            logger.info(f"Added {count} new games")
        except Exception as e:
            self.session.rollback()
            logger.error(f"Error syncing games: {e}")
        return count

    async def sync_team_stats(self, days_lookback: int = 30) -> int:
        logger.info(f"Syncing team stats for last {days_lookback} days")
        count = 0
        try:
            teams = self.session.query(Team).all()
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_lookback)

            for team in teams:
                for day_offset in range(days_lookback):
                    date = start_date + timedelta(days=day_offset)
                    date_str = date.strftime("%Y-%m-%d")
                    try:
                        schedule = await sportradar_client.get_schedule(date_str)
                        if "games" in schedule:
                            for game in schedule["games"]:
                                if (
                                    game.get("home", {}).get("id") == team.sportradar_id
                                    or game.get("away", {}).get("id")
                                    == team.sportradar_id
                                ):
                                    boxscore = (
                                        await sportradar_client.get_game_boxscore(
                                            game.get("id")
                                        )
                                    )
                                    if (
                                        "home" in boxscore
                                        and "statistics" in boxscore["home"]
                                    ):
                                        stats = boxscore["home"]["statistics"]
                                        team_stat = TeamStats(
                                            game_id=self.session.query(Game)
                                            .filter(
                                                Game.sportradar_id == game.get("id")
                                            )
                                            .first()
                                            .id,
                                            team_id=team.id,
                                            date=datetime.fromisoformat(
                                                game.get("scheduled").replace(
                                                    "Z", "+00:00"
                                                )
                                            ),
                                            offensive_rating=stats.get(
                                                "offensive_rating"
                                            ),
                                            defensive_rating=stats.get(
                                                "defensive_rating"
                                            ),
                                            net_rating=stats.get("net_rating"),
                                            pace=stats.get("pace"),
                                            effective_fg_pct=stats.get(
                                                "effective_fg_pct"
                                            ),
                                            turnover_pct=stats.get("turnover_pct"),
                                            rebound_rate=stats.get("rebound_rate"),
                                            free_throw_rate=stats.get(
                                                "free_throw_rate"
                                            ),
                                        )
                                        self.session.add(team_stat)
                                        count += 1
                    except Exception as e:
                        logger.error(
                            f"Error syncing stats for {team.name} on {date_str}: {e}"
                        )
            self.session.commit()
            logger.info(f"Added {count} team stat records")
        except Exception as e:
            self.session.rollback()
            logger.error(f"Error syncing team stats: {e}")
        return count

    async def sync_player_stats(self, days_lookback: int = 30) -> int:
        logger.info(f"Syncing player stats for last {days_lookback} days")
        count = 0
        try:
            games = (
                self.session.query(Game)
                .filter(
                    Game.scheduled_date
                    >= datetime.now() - timedelta(days=days_lookback)
                )
                .all()
            )

            for game in games:
                try:
                    boxscore = await sportradar_client.get_game_boxscore(
                        game.sportradar_id
                    )
                    for team_key in ["home", "away"]:
                        if team_key in boxscore and "players" in boxscore[team_key]:
                            for player_data in boxscore[team_key]["players"]:
                                player = (
                                    self.session.query(Player)
                                    .filter(
                                        Player.sportradar_id == player_data.get("id")
                                    )
                                    .first()
                                )
                                if not player:
                                    player = Player(
                                        sportradar_id=player_data.get("id"),
                                        name=player_data.get("name"),
                                        position=player_data.get("position"),
                                    )
                                    self.session.add(player)
                                    self.session.flush()

                                stats = player_data.get("statistics", {})
                                player_stat = PlayerStats(
                                    game_id=game.id,
                                    player_id=player.id,
                                    team_id=game.home_team_id
                                    if team_key == "home"
                                    else game.away_team_id,
                                    date=game.scheduled_date,
                                    points=stats.get("points"),
                                    rebounds=stats.get("rebounds"),
                                    assists=stats.get("assists"),
                                    minutes=stats.get("minutes"),
                                    steals=stats.get("steals"),
                                    blocks=stats.get("blocks"),
                                    turnovers=stats.get("turnovers"),
                                    field_goals_made=stats.get("field_goals_made"),
                                    field_goals_attempted=stats.get(
                                        "field_goals_attempted"
                                    ),
                                    three_points_made=stats.get("three_points_made"),
                                    three_points_attempted=stats.get(
                                        "three_points_attempted"
                                    ),
                                    free_throws_made=stats.get("free_throws_made"),
                                    free_throws_attempted=stats.get(
                                        "free_throws_attempted"
                                    ),
                                    per=stats.get("player_efficiency_rating"),
                                    true_shooting_pct=stats.get("true_shooting_pct"),
                                    usage_rate=stats.get("usage"),
                                )
                                self.session.add(player_stat)
                                count += 1
                except Exception as e:
                    logger.error(
                        f"Error syncing stats for game {game.sportradar_id}: {e}"
                    )
            self.session.commit()
            logger.info(f"Added {count} player stat records")
        except Exception as e:
            self.session.rollback()
            logger.error(f"Error syncing player stats: {e}")
        return count

    async def sync_injuries(self) -> int:
        logger.info("Syncing injuries")
        count = 0
        try:
            injuries = await sportradar_client.get_injuries()
            if "injuries" in injuries:
                for injury_data in injuries["injuries"]:
                    player = (
                        self.session.query(Player)
                        .filter(
                            Player.sportradar_id
                            == injury_data.get("player", {}).get("id")
                        )
                        .first()
                    )
                    if player:
                        existing = (
                            self.session.query(InjuryReport)
                            .filter(InjuryReport.player_id == player.id)
                            .filter(
                                InjuryReport.reported_date
                                >= datetime.now() - timedelta(days=1)
                            )
                            .first()
                        )
                        if not existing:
                            injury = InjuryReport(
                                player_id=player.id,
                                status=injury_data.get("status"),
                                injury_type=injury_data.get("injury_type"),
                                note=injury_data.get("note"),
                                reported_date=datetime.now(),
                            )
                            self.session.add(injury)
                            count += 1
            self.session.commit()
            logger.info(f"Added {count} injury records")
        except Exception as e:
            self.session.rollback()
            logger.error(f"Error syncing injuries: {e}")
        return count

    async def sync_odds(self, days_lookback: int = 7) -> int:
        logger.info(f"Syncing odds for last {days_lookback} days")
        count = 0
        try:
            games = (
                self.session.query(Game)
                .filter(
                    Game.scheduled_date
                    >= datetime.now() - timedelta(days=days_lookback)
                )
                .filter(Game.status == "closed")
                .all()
            )

            for game in games:
                try:
                    odds_data = await odds_client.get_game_odds(game.sportradar_id)
                    if "markets" in odds_data:
                        for market in odds_data["markets"]:
                            if market.get("type") == "moneyline":
                                for outcome in market.get("outcomes", []):
                                    if outcome.get("side") == "home":
                                        odds = Odds(
                                            game_id=game.id,
                                            sportsbook=market.get("sportsbook"),
                                            moneyline_home=outcome.get("odds"),
                                        )
                                    elif outcome.get("side") == "away":
                                        odds = Odds(
                                            game_id=game.id,
                                            sportsbook=market.get("sportsbook"),
                                            moneyline_away=outcome.get("odds"),
                                        )
                                    self.session.add(odds)
                                    count += 1
                except Exception as e:
                    logger.error(
                        f"Error syncing odds for game {game.sportradar_id}: {e}"
                    )
            self.session.commit()
            logger.info(f"Added {count} odds records")
        except Exception as e:
            self.session.rollback()
            logger.error(f"Error syncing odds: {e}")
        return count

    async def run_full_pipeline(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ):
        await self.initialize()

        if not start_date:
            start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")

        logger.info(f"Starting full pipeline from {start_date} to {end_date}")

        await self.sync_teams()
        await self.sync_games(start_date, end_date)
        await self.sync_team_stats()
        await self.sync_player_stats()
        await self.sync_injuries()
        await self.sync_odds()

        self.close()
        logger.info("Pipeline completed")


pipeline = DataPipeline()


if __name__ == "__main__":
    asyncio.run(pipeline.run_full_pipeline())
