@echo off
cd /d C:\Projects\basketball\nba_predictor
set USE_SQLITE=true
python -c "import os; os.environ['USE_SQLITE'] = 'true'; from dotenv import load_dotenv; load_dotenv(); from src.api.main import app; from fastapi.testclient import TestClient; client = TestClient(app); client.post('/predict/today')"
echo Predictions sent at %date% %time%
