import os
from typing import Optional

import torch
from dashscope import TextEmbedding


class EmbedClient:
    """
    Embedding client supporting:
    - provider="stub": hashing trick (offline)
    - provider="qwen": DashScope TextEmbedding API (e.g. text-embedding-v3)

    默认使用中国大陆 DashScope（不设置 intl / region）
    """

    def __init__(
        self,
        provider: str = "stub",
        model: str = "hashing",
        embed_dim: int = 1024,
        api_key_env: str = "DASHSCOPE_API_KEY",
        timeout_sec: float = 30.0,
    ):
        self.provider = str(provider).lower()
        self.model = model
        self.embed_dim = int(embed_dim)
        self.api_key_env = api_key_env
        self.timeout_sec = float(timeout_sec)

        # 不手动设置 base_http_api_url
        # DashScope SDK 默认即为中国大陆
        # dashscope.base_http_api_url = "https://dashscope.aliyuncs.com/api/v1"

    def embed(self, text: str, device=None) -> torch.Tensor:
        if device is None:
            device = "cpu"
        text = (text or "").strip()

        if self.provider == "qwen":
            return self._embed_qwen(text, device=device)

        # default: stub hashing
        return self._embed_stub(text, device=device)

    # ---------- stub hashing ----------
    def _embed_stub(self, text: str, device=None) -> torch.Tensor:
        v = torch.zeros(self.embed_dim, device=device)
        if not text:
            return v

        for ch in text:
            idx = (ord(ch) * 1315423911) % self.embed_dim
            v[idx] += 1.0

        v = v / (v.norm() + 1e-6)
        return v

    # ---------- qwen / dashscope embedding ----------
    def _embed_qwen(self, text: str, device=None) -> torch.Tensor:
        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing API key env var: {self.api_key_env}")

        if not text:
            return torch.zeros(self.embed_dim, device=device)

        resp = TextEmbedding.call(
            api_key=api_key,
            model=self.model,   # e.g. "text-embedding-v3"
            input=[text],
            dim=self.embed_dim,
        )

        # 兼容不同 SDK 返回结构
        try:
            emb = resp["output"]["embeddings"][0]["embedding"]
        except Exception:
            try:
                emb = resp.output["embeddings"][0]["embedding"]
            except Exception as e:
                raise RuntimeError(
                    f"Failed to parse DashScope embedding response: {resp}"
                ) from e

        vec = torch.tensor(emb, dtype=torch.float32, device=device)

        # 防御性处理：维度不一致时裁剪 / padding
        if vec.numel() != self.embed_dim:
            if vec.numel() > self.embed_dim:
                vec = vec[: self.embed_dim]
            else:
                pad = torch.zeros(self.embed_dim - vec.numel(), device=device)
                vec = torch.cat([vec, pad], dim=0)

        vec = vec / (vec.norm() + 1e-6)
        return vec
