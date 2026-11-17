import asyncio
from telegram.ext import Application
import app as bot  # app.py: loads models on import, main() is guarded so the bot doesn't start


async def run():
    tg = Application.builder().token(bot.BOT_TOKEN).build()
    async with tg.bot:
        await bot.weekly_broadcast(tg)


asyncio.run(run())