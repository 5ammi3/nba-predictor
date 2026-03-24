import os
import httpx
from ..utils.logger import logger


class TelegramNotifier:
    def __init__(self):
        self.bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")
        self.enabled = bool(self.bot_token and self.chat_id)

    async def send_message(self, text: str):
        if not self.enabled:
            logger.warning("Telegram notifications not configured")
            return

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {"chat_id": self.chat_id, "text": text, "parse_mode": "HTML"}
        async with httpx.AsyncClient() as client:
            try:
                r = await client.post(url, json=payload, timeout=10)
                if r.status_code == 200:
                    logger.info("Telegram notification sent")
                else:
                    logger.error(f"Telegram send failed: {r.status_code}")
            except Exception as e:
                logger.error(f"Telegram error: {e}")

    async def send_prediction(self, game_pred: dict):
        home = game_pred.get("home_team", "Home")
        away = game_pred.get("away_team", "Away")

        ml = game_pred.get("moneyline", {})
        spread = game_pred.get("spread", {})
        ou = game_pred.get("over_under", {})

        text = f"""🏀 <b>NBA Prediction</b>

{home} vs {away}
{game_pred.get("game_date", "")}

💰 <b>Moneyline</b>
• {home}: {ml.get("home_win_probability", 0) * 100:.1f}%
• {away}: {ml.get("away_win_probability", 0) * 100:.1f}%
• Pick: {ml.get("recommendation", "N/A")}

🎯 <b>Spread</b>
• {spread.get("recommendation", "N/A")} covers ({spread.get("spread_line", "N/A")})
• Confidence: {spread.get("confidence", 0) * 100:.1f}%

📊 <b>Over/Under</b>
• Projected: {ou.get("predicted_total", 0):.1f} (line: {ou.get("total_line", "N/A")})
• Pick: {ou.get("recommendation", "N/A")}
• Confidence: {ou.get("confidence", 0) * 100:.1f}%"""

        await self.send_message(text)


telegram_notifier = TelegramNotifier()
