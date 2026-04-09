"""Typed runtime settings for the backend."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv


class ConfigurationError(ValueError):
    """Raised when required backend configuration is missing."""


BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"


def _load_dotenv() -> None:
    load_dotenv(ENV_FILE)


def _read_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _read_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be an integer") from exc


def _read_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be a number") from exc


def _resolve_path(name: str, default: Path) -> Path:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    path = Path(raw)
    if not path.is_absolute():
        path = BASE_DIR / path
    return path.resolve()


@dataclass(frozen=True)
class AppSettings:
    base_dir: Path
    env_file: Path
    log_level: str
    openai_api_key: str
    azure_openai_api_key: str
    use_azure_openai: bool
    azure_openai_endpoint: str
    azure_openai_api_version: str
    azure_openai_deployment: str
    azure_openai_chat_deployment: str
    azure_openai_embedding_deployment: str
    embedding_model: str
    chat_model: str
    translation_model: str
    chat_temperature: float
    chat_max_tokens: int
    language_detection_confidence_threshold: float
    chunk_size: int
    chunk_overlap: int
    top_k: int
    similarity_threshold: float
    chroma_persist_dir: Path
    knowledge_file: Path
    ingest_state_file: Path
    ors_api_key: str
    ors_base_url: str
    map_route_timeout_s: float
    map_cache_ttl_s: int
    map_cache_max_items: int
    map_rate_limit_route_per_min: int
    map_rate_limit_places_per_min: int
    map_enable_translation_fallback: bool
    maps_data_file: Path

    @classmethod
    def from_env(cls) -> "AppSettings":
        _load_dotenv()

        chroma_persist_dir = _resolve_path("CHROMA_PERSIST_DIR", BASE_DIR / "chroma_store")
        knowledge_file = _resolve_path("KNOWLEDGE_FILE", BASE_DIR / "knowledge.txt")

        return cls(
            base_dir=BASE_DIR,
            env_file=ENV_FILE,
            log_level=os.getenv("LOG_LEVEL", "INFO").strip().upper(),
            openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
            azure_openai_api_key=os.getenv("AZURE_OPENAI_API_KEY", "").strip(),
            use_azure_openai=_read_bool("USE_AZURE_OPENAI", False),
            azure_openai_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", "").strip(),
            azure_openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01").strip(),
            azure_openai_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "").strip(),
            azure_openai_chat_deployment=os.getenv(
                "AZURE_OPENAI_CHAT_DEPLOYMENT",
                os.getenv("AZURE_OPENAI_DEPLOYMENT", ""),
            ).strip(),
            azure_openai_embedding_deployment=os.getenv(
                "AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
                os.getenv("AZURE_OPENAI_DEPLOYMENT", ""),
            ).strip(),
            embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small").strip(),
            chat_model=os.getenv("CHAT_MODEL", "gpt-4o-mini").strip(),
            translation_model=os.getenv(
                "TRANSLATION_MODEL",
                os.getenv("CHAT_MODEL", "gpt-4o-mini"),
            ).strip(),
            chat_temperature=_read_float("CHAT_TEMPERATURE", 0.2),
            chat_max_tokens=_read_int("CHAT_MAX_TOKENS", 1200),
            language_detection_confidence_threshold=_read_float(
                "LANGUAGE_DETECTION_CONFIDENCE_THRESHOLD",
                0.9,
            ),
            chunk_size=_read_int("CHUNK_SIZE", 300),
            chunk_overlap=_read_int("CHUNK_OVERLAP", 50),
            top_k=_read_int("TOP_K", 5),
            similarity_threshold=_read_float("SIMILARITY_THRESHOLD", 0.7),
            chroma_persist_dir=chroma_persist_dir,
            knowledge_file=knowledge_file,
            ingest_state_file=chroma_persist_dir / "ingest_state.json",
            ors_api_key=os.getenv("ORS_API_KEY", "").strip(),
            ors_base_url=os.getenv("ORS_BASE_URL", "https://api.openrouteservice.org").strip(),
            map_route_timeout_s=_read_float("MAP_ROUTE_TIMEOUT_S", 8.0),
            map_cache_ttl_s=_read_int("MAP_CACHE_TTL_S", 300),
            map_cache_max_items=_read_int("MAP_CACHE_MAX_ITEMS", 1000),
            map_rate_limit_route_per_min=_read_int("MAP_RATE_LIMIT_ROUTE_PER_MIN", 30),
            map_rate_limit_places_per_min=_read_int("MAP_RATE_LIMIT_PLACES_PER_MIN", 60),
            map_enable_translation_fallback=_read_bool("MAP_ENABLE_TRANSLATION_FALLBACK", True),
            maps_data_file=_resolve_path("MAPS_DATA_FILE", BASE_DIR / "data" / "campus_places.json"),
        )

    @property
    def effective_openai_api_key(self) -> str:
        if self.use_azure_openai and self.azure_openai_api_key:
            return self.azure_openai_api_key
        return self.openai_api_key

    def validate_required(self) -> None:
        missing: list[str] = []
        if not self.effective_openai_api_key:
            missing.append("AZURE_OPENAI_API_KEY" if self.use_azure_openai else "OPENAI_API_KEY")
        if self.use_azure_openai and not self.azure_openai_endpoint:
            missing.append("AZURE_OPENAI_ENDPOINT")
        if missing:
            joined = ", ".join(missing)
            raise ConfigurationError(f"Missing required environment variables: {joined}")

    @property
    def chroma_initialized(self) -> bool:
        return self.chroma_persist_dir.exists()


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    return AppSettings.from_env()


def reload_settings() -> AppSettings:
    get_settings.cache_clear()
    return get_settings()
