from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any

# Attributes the stdlib puts on every LogRecord. Anything else on the record
# came from a caller's extra={...} and belongs in the JSON payload.
_STANDARD_RECORD_ATTRS = frozenset({
    "args", "asctime", "created", "exc_info", "exc_text", "filename",
    "funcName", "levelname", "levelno", "lineno", "module", "msecs",
    "message", "msg", "name", "pathname", "process", "processName",
    "relativeCreated", "stack_info", "stacklevel", "taskName", "thread",
    "threadName",
})


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        # logging.Logger.info(..., extra={"k": v}) sets record.k = v directly.
        # The previous implementation only read record.extra_fields, which no
        # call site in this repo ever sets, so every structured field attached
        # to every log line was silently discarded.
        for key, value in record.__dict__.items():
            if key in _STANDARD_RECORD_ATTRS or key.startswith("_"):
                continue
            if key == "extra_fields" and isinstance(value, dict):
                payload.update(value)  # legacy shape, still honoured
                continue
            payload[key] = value

        return json.dumps(payload, ensure_ascii=False, default=str)


def get_logger(name: str, *, level: str | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
    logger.setLevel(level or "INFO")
    logger.propagate = False
    return logger
