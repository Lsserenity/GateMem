from typing import List, Dict, Any

"""
默认 stub：不会真的调用大模型，只是把 window_text 压成 1~N 条条目。
你以后要接 Qwen/OpenAI，只改 generate_entries()。
"""


class LLMClient:
    """
    最小可用 LLM client（stub）：
    - generate_entries(window_text) -> List[dict]
    之后你接真实 LLM（Qwen/OpenAI）只需要在这个文件里改实现。
    """

    def __init__(
        self,
        provider: str = "stub",
        model: str = "stub-llm",
        temperature: float = 0.2,
        max_tokens: int = 256,
        schema: str = "strict_json",
        **kwargs
    ):
        self.provider = provider
        self.model = model
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.schema = schema
        self.kwargs = kwargs

    def generate_entries(self, window_text: str, max_entries: int = 2) -> List[Dict[str, Any]]:
        """
        stub 策略：
        - 把 window_text 截断成若干句，生成 1~max_entries 条 “经验”
        - 结构与 schema 保持一致
        """
        window_text = (window_text or "").strip()
        if not window_text:
            return []

        # 简单分句（按中文/英文标点）
        seps = ["。", "；", ";", ".", "!", "？", "?"]
        pieces = [window_text]
        for sep in seps:
            new_pieces = []
            for p in pieces:
                new_pieces.extend([x.strip() for x in p.split(sep) if x.strip()])
            pieces = new_pieces if new_pieces else pieces

        pieces = pieces[: max_entries] if pieces else [window_text[:120]]
        out = []
        for i, p in enumerate(pieces[:max_entries]):
            out.append(
                {
                    "when": "general",
                    "rule": p[:160],
                    "tags": ["auto", "window"],
                    "priority": 0.5,
                }
            )
        return out
