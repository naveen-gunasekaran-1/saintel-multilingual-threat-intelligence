"""Verify Telegram collection targets before adding them to the target list.

Six of the first nine targets produced nothing, and the collector reported it
only as a log line per channel. Three causes, each needing a different fix:

    username does not exist   -> the handle is wrong
    resolves, 0 messages      -> history hidden; you must join the channel
    resolves, 0 with text     -> media-only channel, nothing for an NLP pipeline

This checks all three up front, so a target list is known-good before a
collection run rather than after it.

    python scripts/verify_targets.py somechannel another_one
    python scripts/verify_targets.py --file config/telegram_targets.txt
"""

from __future__ import annotations

import argparse
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

from src.layer1_ingestion.telegram_scraper import load_targets

VERDICTS = {
    "OK": "usable",
    "NO_HANDLE": "username does not exist -- fix the handle",
    "NO_HISTORY": "resolves but returns nothing -- join the channel to read history",
    "NO_TEXT": "media-only -- no text for the pipeline to process",
}


async def check(client, name: str, probe: int = 15) -> tuple[str, str]:
    try:
        entity = await client.get_entity(name)
    except Exception as exc:
        return "NO_HANDLE", type(exc).__name__

    title = getattr(entity, "title", None) or getattr(entity, "username", name)
    seen = with_text = 0
    try:
        async for message in client.iter_messages(name, limit=probe):
            seen += 1
            if message.text:
                with_text += 1
    except Exception as exc:
        return "NO_HISTORY", f"{type(exc).__name__}: {str(exc)[:60]}"

    if seen == 0:
        return "NO_HISTORY", str(title)
    if with_text == 0:
        return "NO_TEXT", f"{title} ({seen} media posts, 0 with text)"
    return "OK", f"{title} ({with_text}/{seen} recent posts carry text)"


async def main() -> int:
    parser = argparse.ArgumentParser(description="Verify Telegram targets before collecting.")
    parser.add_argument("names", nargs="*", help="channel usernames to check")
    parser.add_argument("--file", type=Path, help="read targets from this file instead")
    parser.add_argument("--probe", type=int, default=15, help="messages to sample per channel")
    args = parser.parse_args()

    names = args.names or load_targets(args.file)
    if not names:
        print("Nothing to check. Pass usernames or --file.", file=sys.stderr)
        return 2

    client = TelegramClient(os.getenv("TELEGRAM_SESSION_NAME", "saintel_scraper"),
                            int(os.getenv("TELEGRAM_API_ID", "0") or "0"),
                            os.getenv("TELEGRAM_API_HASH", ""))
    await client.connect()
    if not await client.is_user_authorized():
        print("Not logged in. Run scripts/telegram_login.py first.", file=sys.stderr)
        await client.disconnect()
        return 1

    usable = 0
    print(f"{'channel':28s} {'verdict':12s} detail")
    print("-" * 92)
    for name in names:
        verdict, detail = await check(client, name, args.probe)
        usable += verdict == "OK"
        print(f"{name:28s} {verdict:12s} {detail}")

    print("-" * 92)
    print(f"{usable}/{len(names)} usable. Legend:")
    for key, meaning in VERDICTS.items():
        print(f"   {key:12s} {meaning}")
    await client.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
