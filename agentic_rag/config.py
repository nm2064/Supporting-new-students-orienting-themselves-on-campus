"""Backward-compatible configuration exports."""

from __future__ import annotations

from .settings import ConfigurationError, get_settings, reload_settings


def _current():
    return get_settings()


def validate_config() -> bool:
    _current().validate_required()
    return True


def reload_config() -> None:
    reload_settings()
    globals().update(_build_exports())


def _build_exports() -> dict[str, object]:
    settings = _current()
    return {
        "BASE_DIR": str(settings.base_dir),
        "OPENAI_API_KEY": settings.openai_api_key,
        "AZURE_OPENAI_API_KEY": settings.azure_openai_api_key,
        "AZURE_OPENAI_ENDPOINT": settings.azure_openai_endpoint,
        "AZURE_OPENAI_API_VERSION": settings.azure_openai_api_version,
        "AZURE_OPENAI_DEPLOYMENT": settings.azure_openai_deployment,
        "AZURE_OPENAI_CHAT_DEPLOYMENT": settings.azure_openai_chat_deployment,
        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT": settings.azure_openai_embedding_deployment,
        "USE_AZURE_OPENAI": settings.use_azure_openai,
        "EMBEDDING_MODEL": settings.embedding_model,
        "CHAT_MODEL": settings.chat_model,
        "TRANSLATION_MODEL": settings.translation_model,
        "CHAT_TEMPERATURE": settings.chat_temperature,
        "CHAT_MAX_TOKENS": settings.chat_max_tokens,
        "LANGUAGE_DETECTION_CONFIDENCE_THRESHOLD": settings.language_detection_confidence_threshold,
        "CHUNK_SIZE": settings.chunk_size,
        "CHUNK_OVERLAP": settings.chunk_overlap,
        "TOP_K": settings.top_k,
        "SIMILARITY_THRESHOLD": settings.similarity_threshold,
        "CHROMA_PERSIST_DIR": str(settings.chroma_persist_dir),
        "KNOWLEDGE_FILE": str(settings.knowledge_file),
        "INGEST_STATE_FILE": str(settings.ingest_state_file),
        "ORS_API_KEY": settings.ors_api_key,
        "ORS_BASE_URL": settings.ors_base_url,
        "MAP_ROUTE_TIMEOUT_S": settings.map_route_timeout_s,
        "MAP_CACHE_TTL_S": settings.map_cache_ttl_s,
        "MAP_CACHE_MAX_ITEMS": settings.map_cache_max_items,
        "MAP_RATE_LIMIT_ROUTE_PER_MIN": settings.map_rate_limit_route_per_min,
        "MAP_RATE_LIMIT_PLACES_PER_MIN": settings.map_rate_limit_places_per_min,
        "MAP_ENABLE_TRANSLATION_FALLBACK": settings.map_enable_translation_fallback,
        "MAPS_DATA_FILE": str(settings.maps_data_file),
    }


globals().update(_build_exports())

__all__ = [
    "AZURE_OPENAI_API_VERSION",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_CHAT_DEPLOYMENT",
    "AZURE_OPENAI_DEPLOYMENT",
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
    "AZURE_OPENAI_ENDPOINT",
    "BASE_DIR",
    "CHAT_MAX_TOKENS",
    "CHAT_MODEL",
    "CHAT_TEMPERATURE",
    "CHROMA_PERSIST_DIR",
    "CHUNK_OVERLAP",
    "CHUNK_SIZE",
    "ConfigurationError",
    "EMBEDDING_MODEL",
    "INGEST_STATE_FILE",
    "KNOWLEDGE_FILE",
    "LANGUAGE_DETECTION_CONFIDENCE_THRESHOLD",
    "MAPS_DATA_FILE",
    "MAP_CACHE_MAX_ITEMS",
    "MAP_CACHE_TTL_S",
    "MAP_ENABLE_TRANSLATION_FALLBACK",
    "MAP_RATE_LIMIT_PLACES_PER_MIN",
    "MAP_RATE_LIMIT_ROUTE_PER_MIN",
    "MAP_ROUTE_TIMEOUT_S",
    "OPENAI_API_KEY",
    "ORS_API_KEY",
    "ORS_BASE_URL",
    "SIMILARITY_THRESHOLD",
    "TOP_K",
    "TRANSLATION_MODEL",
    "USE_AZURE_OPENAI",
    "reload_config",
    "validate_config",
]
