import os
import httpx
from ..utils.logger import logger


class TelegramNotifier:
    @property
    def enabled(self) -> bool:
        return bool(self.bot_token and self.chat_id)

    def __init__(self):
        pass

    def _refresh(self):
        self.bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
        logger.info(
            f"Telegram config: token={'set' if self.bot_token else 'missing'}, chat_id={'set' if self.chat_id else 'missing'}"
        )

    async def send_message(self, text: str):
        self._refresh()
        if not self.enabled:
            logger.warning(
                "Telegram notifications not configured - missing token or chat_id"
            )
            return

        logger.info(f"Sending Telegram message to {self.chat_id}")
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {"chat_id": self.chat_id, "text": text, "parse_mode": "HTML"}
        async with httpx.AsyncClient() as client:
            try:
                r = await client.post(url, json=payload, timeout=10)
                if r.status_code == 200:
                    logger.info("Telegram notification sent successfully")
                else:
                    logger.error(f"Telegram send failed: {r.status_code} - {r.text}")
            except Exception as e:
                logger.error(f"Telegram error: {e}")

    async def send_prediction(self, game_pred: dict):
        from datetime import datetime
        import pytz

        home = game_pred.get("home_team", "Home")
        away = game_pred.get("away_team", "Away")

        game_date_str = game_pred.get("game_date", "")
        game_time = game_pred.get("game_time", "")

        try:
            if game_date_str:
                dt = datetime.fromisoformat(game_date_str.replace("Z", "+00:00"))
                est = pytz.timezone("US/Eastern")
                dt_est = dt.astimezone(est)
                game_time_str = dt_est.strftime("%b %d, %I:%M %p ET")
            else:
                game_time_str = "TBD"
        except:
            game_time_str = game_date_str[:10] if game_date_str else "TBD"

        if game_pred.get("game_time") and game_pred.get("game_time") not in [
            "Scheduled",
            None,
        ]:
            game_time_str = f"{game_pred.get('game_time')} ET"

        ml = game_pred.get("moneyline", {})
        spread = game_pred.get("spread", {})
        ou = game_pred.get("over_under", {})

        text = f"""🏀 <b>NBA Prediction</b>

{home} vs {away}
{game_time_str}

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
