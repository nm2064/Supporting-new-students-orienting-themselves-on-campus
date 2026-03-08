"""Configuration module for the Agentic RAG backend."""
import os
from dotenv import load_dotenv

# Load .env from the same directory as this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Azure OpenAI Configuration (optional - for Azure deployments)
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")  # e.g., https://your-resource.openai.azure.com
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
# Backward-compatible single deployment key (legacy). Prefer chat/embedding split below.
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "")
AZURE_OPENAI_CHAT_DEPLOYMENT = os.getenv(
    "AZURE_OPENAI_CHAT_DEPLOYMENT",
    os.getenv("AZURE_OPENAI_DEPLOYMENT", "")
)
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv(
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
    os.getenv("AZURE_OPENAI_DEPLOYMENT", "")
)
USE_AZURE_OPENAI = os.getenv("USE_AZURE_OPENAI", "false").lower() == "true"

# Debug: Print loaded keys (masked)
print(f"OpenAI Key loaded: {'Yes (' + OPENAI_API_KEY[:10] + '...)' if OPENAI_API_KEY else 'No'}")
if USE_AZURE_OPENAI:
    print(f"Using Azure OpenAI: {AZURE_OPENAI_ENDPOINT}")
    print(f"  API Version: {AZURE_OPENAI_API_VERSION}")
    if AZURE_OPENAI_CHAT_DEPLOYMENT:
        print(f"  Chat Deployment: {AZURE_OPENAI_CHAT_DEPLOYMENT}")
    if AZURE_OPENAI_EMBEDDING_DEPLOYMENT:
        print(f"  Embedding Deployment: {AZURE_OPENAI_EMBEDDING_DEPLOYMENT}")

# Embedding Configuration
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# Chat/Translation model configuration (OpenAI-only stack)
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
TRANSLATION_MODEL = os.getenv("TRANSLATION_MODEL", CHAT_MODEL)
CHAT_TEMPERATURE = float(os.getenv("CHAT_TEMPERATURE", "0.2"))
CHAT_MAX_TOKENS = int(os.getenv("CHAT_MAX_TOKENS", "1200"))

# Multilingual configuration
LANGUAGE_DETECTION_CONFIDENCE_THRESHOLD = float(
    os.getenv("LANGUAGE_DETECTION_CONFIDENCE_THRESHOLD", "0.9")
)

# Chunking Configuration
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "300"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# Retrieval Configuration
TOP_K = int(os.getenv("TOP_K", "5"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))

# File Paths
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", os.path.join(BASE_DIR, "chroma_store"))
KNOWLEDGE_FILE = os.getenv("KNOWLEDGE_FILE", os.path.join(BASE_DIR, "knowledge.txt"))
INGEST_STATE_FILE = os.path.join(CHROMA_PERSIST_DIR, "ingest_state.json")

# Maps Configuration
ORS_API_KEY = os.getenv("ORS_API_KEY", "")
ORS_BASE_URL = os.getenv("ORS_BASE_URL", "https://api.openrouteservice.org")
MAP_ROUTE_TIMEOUT_S = float(os.getenv("MAP_ROUTE_TIMEOUT_S", "8"))
MAP_CACHE_TTL_S = int(os.getenv("MAP_CACHE_TTL_S", "300"))
MAP_CACHE_MAX_ITEMS = int(os.getenv("MAP_CACHE_MAX_ITEMS", "1000"))
MAP_RATE_LIMIT_ROUTE_PER_MIN = int(os.getenv("MAP_RATE_LIMIT_ROUTE_PER_MIN", "30"))
MAP_RATE_LIMIT_PLACES_PER_MIN = int(os.getenv("MAP_RATE_LIMIT_PLACES_PER_MIN", "60"))
MAP_ENABLE_TRANSLATION_FALLBACK = (
    os.getenv("MAP_ENABLE_TRANSLATION_FALLBACK", "true").strip().lower() == "true"
)
MAPS_DATA_FILE = os.getenv("MAPS_DATA_FILE", os.path.join(BASE_DIR, "data", "campus_places.json"))

# Validate required configuration
def validate_config():
    """Validate that required configuration is present."""
    missing = []
    if not OPENAI_API_KEY:
        missing.append("OPENAI_API_KEY")
    if missing:
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")
    return True
