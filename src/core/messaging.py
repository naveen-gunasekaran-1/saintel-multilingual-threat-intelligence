"""Kafka plumbing and payload conventions shared by every layer.

Four consumers (layer2 triage, layer3 extraction, layer4 persistence, the
capture sink) each built their own config dict inline. They agreed on the
important settings by coincidence rather than by construction -- and one of
them (triage) silently differed: it left `enable.auto.commit` at the librdkafka
default of True, so it committed offsets on a timer regardless of whether the
message had been processed. Centralising the config makes the delivery
semantics a property of the module rather than of whoever typed the dict.

`dumps` exists for the same reason: the project directive is native UTF-8 on
the wire, and `ensure_ascii=False` was missing at 6 of 8 serialization sites,
so `சென்னை` shipped as `\\u0b9a\\u0bc6...`. One helper is harder to forget than
a keyword argument.
"""

from __future__ import annotations

import json
from typing import Any

__all__ = [
    "consumer_config",
    "producer_config",
    "dumps",
    "encode",
    "extract_text",
    "TEXT_FIELDS",
]

# Payload text has lived under three different keys depending on which
# collector produced it. Order matters: `text` is what the Telegram and
# leak-site collectors emit, `raw_text` is the schema field name.
TEXT_FIELDS: tuple[str, ...] = ("text", "content", "raw_text")


def consumer_config(
    group_id: str,
    *,
    broker: str,
    auto_offset_reset: str = "earliest",
    enable_auto_commit: bool = False,
) -> dict[str, Any]:
    """Consumer settings with at-least-once defaults.

    `enable_auto_commit=False` is the load-bearing default: offsets must be
    committed by the caller *after* the downstream write is confirmed, or a
    crash advances past messages that were never persisted.
    """
    return {
        "bootstrap.servers": broker,
        "group.id": group_id,
        "auto.offset.reset": auto_offset_reset,
        "enable.auto.commit": enable_auto_commit,
    }


def producer_config(*, broker: str) -> dict[str, Any]:
    return {"bootstrap.servers": broker}


def dumps(payload: Any) -> str:
    """JSON-encode preserving native script (project directive 2)."""
    return json.dumps(payload, ensure_ascii=False)


def encode(payload: Any) -> bytes:
    """UTF-8 bytes ready for a Kafka value, native script preserved."""
    return dumps(payload).encode("utf-8")


def extract_text(payload: dict[str, Any]) -> str:
    """First non-empty known text field, or "" when none carries content.

    Returns "" rather than raising: an unparseable or textless record is a
    routine occurrence on a raw OSINT stream, not an error condition.
    """
    if not isinstance(payload, dict):
        return ""
    for field in TEXT_FIELDS:
        value = payload.get(field)
        if isinstance(value, str) and value.strip():
            return value
    return ""
