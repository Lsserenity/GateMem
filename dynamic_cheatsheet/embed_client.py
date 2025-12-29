import torch

"""
这个版本默认用 hashing trick 把文本变 embedding（完全离线、无需 API）。
你以后要接真实 embedding，只要改 embed() 的实现即可。
"""


class EmbedClient:
    """
    最小可用 embedding client：
    - embed(text) -> Tensor(embed_dim,)
    默认使用 hashing trick：不依赖任何 API，先跑通管线和 ablation。
    """

    def __init__(self, provider: str = "stub", model: str = "hashing", embed_dim: int = 1024):
        self.provider = provider
        self.model = model
        self.embed_dim = int(embed_dim)

    def embed(self, text: str, device=None) -> torch.Tensor:
        if device is None:
            device = "cpu"

        # hashing trick：字符级映射
        v = torch.zeros(self.embed_dim, device=device)
        if not text:
            return v

        for ch in text:
            idx = (ord(ch) * 1315423911) % self.embed_dim
            v[idx] += 1.0

        # L2 normalize
        v = v / (v.norm() + 1e-6)
        return v
