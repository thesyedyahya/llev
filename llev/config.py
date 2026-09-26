from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All config comes from LLEV_* env vars (or .env). Nothing is hardcoded."""

    model_config = SettingsConfigDict(env_prefix="LLEV_", env_file=".env", extra="ignore")

    # Primary (fast) model
    engine: str = "llama"  # llama | fake
    model: str = "qwen3-1.7b"  # key from llev.models.REGISTRY (see bench.md for why)
    model_path: str | None = None  # overrides the registry path
    template: str | None = None  # overrides the registry template
    n_ctx: int = 8192
    n_threads: int | None = None
    n_gpu_layers: int = 0  # -1 = offload everything (Apple Metal / CUDA)

    # Escalation (bigger, slower) model - used only for low-confidence answers
    escalate_model: str | None = None
    escalate_model_path: str | None = None
    escalate_template: str | None = None
    escalate_threshold: float = 0.6

    # Smartness knobs
    debias_permutations: int = 2  # option orderings averaged per choice question
    fewshot_k: int = 3  # learned examples retrieved per question (0 = off)
    # knn: neighbours vote, blended by similarity (robust for small models)
    # prompt: neighbours shown in the prompt as solved examples (better for larger models)
    fewshot_mode: str = "knn"  # knn | prompt | both
    knn_strength: float = 0.8
    knn_min_sim: float = 0.35  # neighbours less similar than this don't vote
    fewshot_max_per_task: int = 500

    # API
    api_keys: str = ""  # comma-separated
    allow_no_auth: bool = False
    max_questions: int = 32
    max_state_chars: int = 60_000

    # Storage
    models_dir: str = "models"
    data_dir: str = "data"
    log_state: bool = True  # store state text so feedback can become few-shot memory

    @property
    def api_key_list(self) -> list[str]:
        return [k.strip() for k in self.api_keys.split(",") if k.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
