from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.providers.redis.operators.redis import RedisOperator


default_args = {
    "owner": "nba_predictor",
    "depends_on_past": False,
    "start_date": datetime(2024, 1, 1),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
}


def sync_teams(**context):
    import asyncio
    from src.data.data_pipeline import pipeline

    asyncio.run(pipeline.sync_teams())


def sync_games(**context):
    import asyncio
    from src.data.data_pipeline import pipeline

    execution_date = context["execution_date"]
    start_date = (execution_date - timedelta(days=7)).strftime("%Y-%m-%d")
    end_date = execution_date.strftime("%Y-%m-%d")

    asyncio.run(pipeline.sync_games(start_date, end_date))


def sync_team_stats(**context):
    import asyncio
    from src.data.data_pipeline import pipeline

    asyncio.run(pipeline.sync_team_stats(days_lookback=30))


def sync_player_stats(**context):
    import asyncio
    from src.data.data_pipeline import pipeline

    asyncio.run(pipeline.sync_player_stats(days_lookback=30))


def sync_injuries(**context):
    import asyncio
    from src.data.data_pipeline import pipeline

    asyncio.run(pipeline.sync_injuries())


def sync_odds(**context):
    import asyncio
    from src.data.data_pipeline import pipeline

    asyncio.run(pipeline.sync_odds(days_lookback=7))


def run_model_training(**context):
    from src.models.xgboost_model import train_game_outcome_model
    import numpy as np
    import os

    X = []
    y = []

    model_path = "models/moneyline_model.pkl"
    if os.path.exists(model_path):
        from src.models.xgboost_model import XGBoostPredictor

        model = XGBoostPredictor(model_type="game_outcome")
        model.load(model_path)
        print(f"Model loaded from {model_path}")

    print("Model training placeholder - would train on historical data")


with DAG(
    "nba_data_pipeline",
    default_args=default_args,
    description="NBA data pipeline for predictions",
    schedule_interval="0 6 * * *",
    catchup=False,
    tags=["nba", "predictions"],
) as dag:
    start = BashOperator(
        task_id="start",
        bash_command='echo "Starting NBA data pipeline"',
    )

    sync_teams_task = PythonOperator(
        task_id="sync_teams",
        python_callable=sync_teams,
    )

    sync_games_task = PythonOperator(
        task_id="sync_games",
        python_callable=sync_games,
    )

    sync_team_stats_task = PythonOperator(
        task_id="sync_team_stats",
        python_callable=sync_team_stats,
    )

    sync_player_stats_task = PythonOperator(
        task_id="sync_player_stats",
        python_callable=sync_player_stats,
    )

    sync_injuries_task = PythonOperator(
        task_id="sync_injuries",
        python_callable=sync_injuries,
    )

    sync_odds_task = PythonOperator(
        task_id="sync_odds",
        python_callable=sync_odds,
    )

    model_training_task = PythonOperator(
        task_id="run_model_training",
        python_callable=run_model_training,
    )

    end = BashOperator(
        task_id="end",
        bash_command='echo "NBA data pipeline completed"',
    )

    start >> sync_teams_task >> sync_games_task

    sync_games_task >> sync_team_stats_task
    sync_games_task >> sync_player_stats_task

    sync_team_stats_task >> sync_injuries_task
    sync_player_stats_task >> sync_injuries_task

    sync_injuries_task >> sync_odds_task
    sync_odds_task >> model_training_task
    model_training_task >> end
