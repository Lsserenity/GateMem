import yaml
from dataclasses import dataclass, field


@dataclass
class DCLLMConfig:
    provider: str = "stub"
    model: str = "stub-llm"
    temperature: float = 0.2
    max_tokens: int = 256
    update_every: int = 1
    consolidate_every: int = 0
    max_entries_per_update: int = 2
    schema: str = "strict_json"


@dataclass
class DCEmbeddingConfig:
    provider: str = "stub"
    model: str = "hashing"
    embed_dim: int = 1024
    dc_len: int = 8
    similarity: str = "cosine"
    min_relevance: float = 0.0


@dataclass
class DCMemoryConfig:
    capacity: int = 512
    replacement: str = "fifo"
    novelty_threshold: float = 0.90


@dataclass
class DCAppConfig:
    return_retrieved_entries: bool = False


@dataclass
class DCConfig:
    enabled: bool = True
    mode: str = "dynamic"   # off | static | dynamic
    llm: DCLLMConfig = field(default_factory=DCLLMConfig)
    embedding: DCEmbeddingConfig = field(default_factory=DCEmbeddingConfig)
    memory: DCMemoryConfig = field(default_factory=DCMemoryConfig)
    app: DCAppConfig = field(default_factory=DCAppConfig)


@dataclass
class RootConfig:
    dc: DCConfig = field(default_factory=DCConfig)


def load_config(path: str) -> RootConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    dc_raw = raw.get("dc", {})

    return RootConfig(
        dc=DCConfig(
            enabled=dc_raw.get("enabled", True),
            mode=dc_raw.get("mode", "dynamic"),
            llm=DCLLMConfig(**dc_raw.get("llm", {})),
            embedding=DCEmbeddingConfig(**dc_raw.get("embedding", {})),
            memory=DCMemoryConfig(**dc_raw.get("memory", {})),
            app=DCAppConfig(**dc_raw.get("app", {})),
        )
    )
