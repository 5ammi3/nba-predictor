# NBA Basketball Prediction System

A production-ready NBA basketball prediction system using XGBoost with early fusion, Sportradar MCP for data, FastAPI for the API, Redis for caching, and PostgreSQL for storage.

## Architecture

- **Data Source**: Sportradar Basketball MCP
- **Prediction Model**: XGBoost with early fusion (structured stats + LLM embeddings)
- **API**: FastAPI
- **Cache**: Redis
- **Database**: PostgreSQL
- **Orchestration**: Airflow

## Quick Setup

### 1. Install Dependencies

```bash
cd nba_predictor
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
```

Set environment variable to use SQLite (default, no PostgreSQL needed):
```bash
# Windows
set USE_SQLITE=true

# Linux/Mac
export USE_SQLITE=true
```

Edit `.env` and add your API keys:

```
SPORTRADAR_API_KEY=your_sportradar_api_key_here
CLAUDE_API_KEY=your_claude_api_key_here
```

#### Getting API Keys

**Sportradar API Key**:
1. Visit https://developer.sportradar.com/
2. Sign up for an account
3. Request NBA API access
4. Copy your API key to `.env`

**Claude API Key** (for text embeddings):
1. Visit https://www.anthropic.com/
2. Sign up for API access
3. Copy your API key to `.env`

### 3. Start Services

```bash
cd nba_predictor
docker-compose up -d
```

This starts PostgreSQL, Redis, and the API services.

### 4. Initialize Database

```bash
python -c "from src.utils.database import init_db; init_db()"
```

### 5. Run Data Pipeline

```bash
python src/data/data_pipeline.py
```

Or via API:
```bash
curl -X POST "http://localhost:8000/pipeline/run?start_date=2024-01-01&end_date=2024-03-22"
```

### 6. Train Models

```bash
cd nba_predictor
jupyter notebook notebooks/03_model_development.ipynb
```

### 7. Start API

```bash
uvicorn src.api.main:app --reload
```

## API Endpoints

### Game Prediction
```bash
curl -X POST "http://localhost:8000/predict/game" \
  -H "Content-Type: application/json" \
  -d '{
    "team1": "Los Angeles Lakers",
    "team2": "Boston Celtics",
    "date": "2026-03-22"
  }'
```

### Player Prop Prediction
```bash
curl -X POST "http://localhost:8000/predict/player-prop" \
  -H "Content-Type: application/json" \
  -d '{
    "player_name": "LeBron James",
    "stat_type": "points",
    "date": "2026-03-22",
    "line": 25.5
  }'
```

### Get Value Bets
```bash
curl -X GET "http://localhost:8000/value-bets/today?min_probability=0.55&bankroll=10000"
```

### Historical Accuracy
```bash
curl -X GET "http://localhost:8000/historical/accuracy?model_version=v1"
```

## Project Structure

```
nba_predictor/
├── src/
│   ├── data/           # Sportradar client, odds client, data pipeline
│   ├── features/       # Structured features, text embeddings, feature utils
│   ├── models/         # XGBoost model, hyperparameter tuning, evaluation
│   ├── prediction/     # Game predictor, player props, value calculator
│   ├── api/            # FastAPI endpoints, schemas, dependencies
│   └── utils/          # Config, logger, cache, database
├── notebooks/          # Jupyter notebooks for development
├── tests/              # pytest test suite
├── docker/             # Dockerfile and docker-compose.yml
├── airflow/dags/      # Airflow DAG for daily data pipeline
├── requirements.txt   # Python dependencies
├── .env.example       # Environment variables template
└── README.md          # This file
```

## Features

### Data Collection
- NBA schedule and live scores
- Team statistics (offensive rating, defensive rating, pace)
- Player statistics (per game, advanced metrics)
- Injury reports and roster updates
- Real-time odds (moneyline, spread, over/under)

### Feature Engineering
- Rolling averages (5, 10, 20 games)
- Pace-adjusted stats
- Home/away splits
- Days of rest, back-to-back indicators
- Strength of schedule
- LLM embeddings from news/injuries (early fusion)

### Prediction Types
- Moneyline (win probability)
- Spread (cover probability)
- Over/Under
- Player props (points, rebounds, assists)

### Value Betting
- Expected value calculation
- Kelly Criterion bet sizing
- Positive EV bet detection

## Expected Performance

- **Moneyline**: 55-60% accuracy
- **Spread**: 65-70% accuracy
- **Over/Under**: 55-60% accuracy

## Development

### Running Tests
```bash
pytest tests/ -v
```

### Jupyter Notebooks
```bash
jupyter notebook notebooks/
```

## Deployment

### Render (Recommended - Free Tier)

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Ready for deployment"
   git push origin main
   ```

2. **Create Web Service on Render**
   - Go to https://render.com
   - Connect your GitHub repo
   - Create new Web Service
   - Select the `nba_predictor` directory
   - Configure:
     - Build Command: `pip install -r requirements.txt`
     - Start Command: `uvicorn src.api.main:app --host 0.0.0.0 --port $PORT`
   - Add Environment Variables:
     - `USE_SQLITE` = `true`
     - `PYTHONUNBUFFERED` = `1`
   - Free tier will give you a URL like `https://your-app.onrender.com`

3. **Test the API**
   ```bash
   curl "https://your-app.onrender.com/health"
   curl "https://your-app.onrender.com/teams"
   ```

### Local Docker (Alternative)

```bash
cd nba_predictor/docker
docker-compose up --build
```

## License

MIT
