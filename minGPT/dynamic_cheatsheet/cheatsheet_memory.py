from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

import torch
from torch import nn

from .embed_client import EmbedClient
from .llm_client import LLMClient
from .schema import CheatSheetEntry


@dataclass
class RetrievedItem:
    score: float
    entry: CheatSheetEntry


class DynamicCheatsheetMemory(nn.Module):
    """
    Path B: LLM Dynamic Cheatsheet Memory（显式经验记忆）

    维护一个固定容量的“条目 bank”（key/value embedding）：
    - update(window_text): 通过 LLM 生成条目 -> embed(key/val) -> 写入 bank
    - retrieve(query_text, B): 相似度检索 top-k -> 投影到 hidden_dim -> dc_memory(B, dc_len, H)

    配置字段来自 config.yaml 的 path_b：
      enabled, mode
      llm: provider/model/temperature/max_tokens/update_every/...
      embedding: embed_dim/dc_len/similarity/min_relevance
      memory: capacity/replacement/novelty_threshold
    """

    def __init__(self, dc_cfg: Dict[str, Any], hidden_dim: int):
        super().__init__()

        self.enabled: bool = bool(dc_cfg.get("enabled", True))
        self.mode: str = str(dc_cfg.get("mode", "dynamic"))  # off | static | dynamic

        llm_cfg = dc_cfg.get("llm", {}) or {}
        emb_cfg = dc_cfg.get("embedding", {}) or {}
        mem_cfg = dc_cfg.get("memory", {}) or {}
        app_cfg = dc_cfg.get("app", {}) or {}

        # embedding settings
        self.embed_dim: int = int(emb_cfg.get("embed_dim", 1024))
        self.dc_len: int = int(emb_cfg.get("dc_len", 8))
        self.similarity: str = str(emb_cfg.get("similarity", "cosine"))
        self.min_relevance: float = float(emb_cfg.get("min_relevance", 0.0))

        # memory settings
        self.capacity: int = int(mem_cfg.get("capacity", 512))
        self.replacement: str = str(mem_cfg.get("replacement", "fifo"))
        self.novelty_threshold: float = float(mem_cfg.get("novelty_threshold", 0.90))

        # app settings
        self.return_retrieved_entries: bool = bool(app_cfg.get("return_retrieved_entries", False))

        # clients
        self.embedder = EmbedClient(
            provider=str(emb_cfg.get("provider", "stub")),
            model=str(emb_cfg.get("model", "hashing")),
            embed_dim=self.embed_dim,
        )
        self.llm = LLMClient(**llm_cfg)

        # projection to model hidden
        self.proj = nn.Linear(self.embed_dim, hidden_dim, bias=False)

        # banks (fixed-size)
        self.register_buffer("key_bank", torch.zeros(self.capacity, self.embed_dim))
        self.register_buffer("val_bank", torch.zeros(self.capacity, self.embed_dim))
        # optional metadata
        self.entries: List[CheatSheetEntry] = []  # python list for interpretability

        self.ptr: int = 0  # write pointer (FIFO ring)
        self.size: int = 0

    # ---------- similarity ----------
    def _sim(self, q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        """
        q: (B,D), K:(M,D) -> scores:(B,M)
        """
        if self.similarity == "cosine":
            qn = q / (q.norm(dim=-1, keepdim=True) + 1e-6)
            Kn = K / (K.norm(dim=-1, keepdim=True) + 1e-6)
            return qn @ Kn.t()
        return q @ K.t()

    @torch.no_grad()
    def _is_novel(self, k: torch.Tensor) -> bool:
        """
        novelty 去重：如果与现有 key 最大相似度 > threshold，则认为不新
        """
        if self.size == 0:
            return True
        K = self.key_bank[: self.size]  # (M,D)

        if self.similarity == "cosine":
            kn = k / (k.norm() + 1e-6)
            Kn = K / (K.norm(dim=-1, keepdim=True) + 1e-6)
            sim = (kn.unsqueeze(0) @ Kn.t()).squeeze(0)  # (M,)
        else:
            sim = (k.unsqueeze(0) @ K.t()).squeeze(0)

        return sim.max().item() <= self.novelty_threshold

    # ---------- public API ----------
    @torch.no_grad()
    def update(
        self,
        window_text: str,
        device=None,
        max_entries: Optional[int] = None,
    ) -> None:
        """
        dynamic 模式：调用 LLM 生成条目，写入 bank
        """
        if (not self.enabled) or (self.mode != "dynamic"):
            return

        if device is None:
            device = self.key_bank.device

        if max_entries is None:
            # 如果没有配置，默认 2
            max_entries = 2

        raw_entries = self.llm.generate_entries(window_text, max_entries=max_entries)
        if not raw_entries:
            return

        for d in raw_entries:
            entry = CheatSheetEntry.from_dict(d)
            key_text = entry.key_text()
            val_text = entry.val_text()

            k = self.embedder.embed(key_text, device=device)
            v = self.embedder.embed(val_text, device=device)

            if not self._is_novel(k):
                continue

            # FIFO ring write
            self.key_bank[self.ptr] = k
            self.val_bank[self.ptr] = v

            if self.size < self.capacity:
                self.size += 1
                if len(self.entries) < self.capacity:
                    self.entries.append(entry)
                else:
                    self.entries[self.ptr] = entry
            else:
                # capacity full: overwrite the oldest at ptr
                if len(self.entries) < self.capacity:
                    self.entries.append(entry)
                else:
                    self.entries[self.ptr] = entry

            self.ptr = (self.ptr + 1) % self.capacity

    def retrieve(
        self,
        query_text: str,
        batch_size: int,
        device=None,
    ) -> Tuple[torch.Tensor, Optional[List[List[RetrievedItem]]]]:
        """
        返回:
          dc_memory: (B, dc_len, hidden_dim)
          retrieved (可选): 每个 batch 的检索条目（用于可解释性）
        """
        if device is None:
            device = self.key_bank.device

        # off/static/dynamic 统一：只要 enabled 且有内容就可检索（static 只是不会 update）
        if (not self.enabled) or (self.mode == "off") or self.size == 0:
            dc = torch.zeros(batch_size, self.dc_len, self.proj.out_features, device=device)
            return dc, None

        q = self.embedder.embed(query_text or "general", device=device)
        q = q.unsqueeze(0).expand(batch_size, -1)  # (B,D)

        K = self.key_bank[: self.size]  # (M,D)
        scores = self._sim(q, K)         # (B,M)

        k = min(self.dc_len, self.size)
        topv, topi = torch.topk(scores, k=k, dim=1)

        # gather values: (B,k,D)
        V = self.val_bank[: self.size]
        gathered = []
        retrieved_debug: Optional[List[List[RetrievedItem]]] = [] if self.return_retrieved_entries else None

        for b in range(batch_size):
            idx = topi[b]  # (k,)
            vb = V[idx]    # (k,D)
            sb = topv[b]   # (k,)

            # relevance threshold: 低于阈值置 0
            if self.min_relevance > 0:
                mask = (sb >= self.min_relevance).float().unsqueeze(1)  # (k,1)
                vb = vb * mask

            gathered.append(vb)

            if retrieved_debug is not None:
                items = []
                for j in range(k):
                    ei = int(idx[j].item())
                    score = float(sb[j].item())
                    entry = self.entries[ei] if ei < len(self.entries) else CheatSheetEntry()
                    items.append(RetrievedItem(score=score, entry=entry))
                retrieved_debug.append(items)

        vals = torch.stack(gathered, dim=0)  # (B,k,D)

        # pad to dc_len
        if k < self.dc_len:
            pad = torch.zeros(batch_size, self.dc_len - k, self.embed_dim, device=device)
            vals = torch.cat([vals, pad], dim=1)

        dc_memory = self.proj(vals)  # (B, dc_len, H)
        return dc_memory, retrieved_debug
