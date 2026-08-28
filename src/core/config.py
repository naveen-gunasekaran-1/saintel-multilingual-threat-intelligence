from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency during bootstrap
    def load_dotenv(*args: Any, **kwargs: Any) -> bool:
        return False

BASE_DIR = Path(__file__).resolve().parents[2]
ENV_FILE = BASE_DIR / ".env"


def _read_env(name: str, default: str = "", *, cast: type | None = None) -> Any:
    value = os.getenv(name, default)
    if cast is None:
        return value
    try:
        return cast(value)
    except (TypeError, ValueError):
        return default


def _required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Required environment variable is missing: {name}")
    return value


def _project_path(value: str) -> str:
    path = Path(value).expanduser()
    return str(path if path.is_absolute() else BASE_DIR / path)


@dataclass(frozen=True)
class Settings:
    app_env: str = "development"
    log_level: str = "INFO"
    debug: bool = False

    kafka_broker_url: str = "localhost:9092"
    kafka_raw_topic: str = "raw_osint_stream"
    kafka_signal_topic: str = "high_intent_signals"
    kafka_entity_topic: str = "layer3_entities"

    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = field(default="", repr=False)
    neo4j_password: str = field(default="", repr=False)

    postgres_dsn: str = field(default="", repr=False)

    model_path: str = str(BASE_DIR / "data" / "models" / "edge_triage_model.bin")
    nlp_model_name: str = "xlm-roberta-base"
    zero_shot_model_name: str = "facebook/bart-large-mnli"
    groq_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434"

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv(ENV_FILE, override=False)
        return cls(
            app_env=_read_env("APP_ENV", "development"),
            log_level=_read_env("LOG_LEVEL", "INFO"),
            debug=_read_env("DEBUG", "false", cast=lambda value: str(value).lower() == "true"),
            kafka_broker_url=_read_env("KAFKA_BROKER_URL", "localhost:9092"),
            kafka_raw_topic=_read_env("KAFKA_RAW_TOPIC", "raw_osint_stream"),
            kafka_signal_topic=_read_env("KAFKA_SIGNAL_TOPIC", "high_intent_signals"),
            kafka_entity_topic=_read_env("KAFKA_ENTITY_TOPIC", "layer3_entities"),
            neo4j_uri=_read_env("NEO4J_URI", "bolt://localhost:7687"),
            neo4j_user=(
                "neo4j"
                if _read_env("NEO4J_URI", "bolt://localhost:7687").startswith("bolt://localhost")
                else (_required_env("NEO4J_USER") if os.getenv("NEO4J_USER") else _required_env("NEO4J_USERNAME"))
            ),
            neo4j_password=_required_env("NEO4J_PASSWORD"),
            postgres_dsn=_required_env("POSTGRES_DSN"),
            model_path=_project_path(_read_env("MODEL_PATH", "data/models/edge_triage_model.bin")),
            nlp_model_name=_read_env("NLP_MODEL_NAME", "xlm-roberta-base"),
            zero_shot_model_name=_read_env("ZERO_SHOT_MODEL_NAME", "facebook/bart-large-mnli"),
            groq_api_key=_read_env("GROQ_API_KEY", ""),
            ollama_base_url=_read_env("OLLAMA_BASE_URL", "http://localhost:11434"),
        )


_SETTINGS: Settings | None = None


def get_settings() -> Settings:
    global _SETTINGS
    if _SETTINGS is None:
        _SETTINGS = Settings.from_env()
    return _SETTINGS
