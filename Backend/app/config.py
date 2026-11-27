"""Application settings loaded from environment / .env."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration. Secrets come only from the environment."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql://eiq:eiq@localhost:5432/enterpriseiq"
    upload_dir: str = "uploads"

    embedding_model: str = "all-MiniLM-L6-v2"
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    embedding_dim: int = 384

    chunk_max_tokens: int = 200
    chunk_min_tokens: int = 120
    chunk_overlap_tokens: int = 40
    top_k: int = 5
    dense_candidates: int = 20
    bm25_candidates: int = 20
    fused_candidates: int = 25
    rrf_k: int = 60
    rerank_min_score: float = -3.0

    llm_provider: str = "gemini"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1"
    llm_timeout_seconds: float = 30.0
    llm_retries: int = 2

    max_upload_bytes: int = 10 * 1024 * 1024
    max_question_chars: int = 2000
    cors_origins: str = "http://localhost:8501"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


settings = Settings()
