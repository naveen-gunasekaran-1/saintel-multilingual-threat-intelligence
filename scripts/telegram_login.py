"""One-time interactive Telegram login for SAINTEL collection.

Telethon needs a phone number and a verification code the first time. That is
inherently interactive, so it cannot run inside an automated step -- run this
yourself in a terminal, once. It creates `saintel_scraper.session` in the repo
root, which the collector then reuses silently.

    python scripts/telegram_login.py

The session file holds a live auth key. It is gitignored; keep it that way, and
revoke it from Telegram > Settings > Devices when you are done collecting.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
from telethon import TelegramClient

load_dotenv(ROOT / ".env")

API_ID = int(os.getenv("TELEGRAM_API_ID", "0") or "0")
API_HASH = os.getenv("TELEGRAM_API_HASH", "")
SESSION_NAME = os.getenv("TELEGRAM_SESSION_NAME", "saintel_scraper")


async def main() -> int:
    if not API_ID or not API_HASH:
        print("TELEGRAM_API_ID / TELEGRAM_API_HASH missing from .env", file=sys.stderr)
        return 1

    client = TelegramClient(SESSION_NAME, API_ID, API_HASH)
    await client.start()  # prompts for phone number, then the code Telegram sends

    me = await client.get_me()
    print(f"\nLogged in as: {me.first_name} (id={me.id}, username={me.username})")
    print(f"Session written to: {ROOT / (SESSION_NAME + '.session')}")
    print("\nThis file is a live credential. It is gitignored -- do not commit it,")
    print("and revoke the session in Telegram > Settings > Devices when finished.")
    print("\nNext:  python src/layer1_ingestion/telegram_scraper.py --limit 25")
    await client.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
