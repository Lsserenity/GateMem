from __future__ import annotations

import os
import json
import re
from typing import List, Dict, Any, Optional

import dashscope


SYSTEM_PROMPT = (
    "You are a memory writer that extracts reusable, high-signal "
    "experience rules from a window of model activity.\n"
    "Your job is to write short, generalizable, actionable rules that can help future windows.\n\n"
    "Return STRICT JSON only. No markdown. No commentary."
)


def build_user_prompt(window_text: str, max_entries: int) -> str:
    return f"""You will be given information from the current window. Write up to {max_entries} memory entries.

Each entry must follow this schema:
{{
  "when": string,
  "rule": string,
  "tags": string[],
  "priority": number
}}

Constraints:
- Output must be a JSON array: [ {{..}}, {{..}} ]
- Do NOT include any keys besides when, rule, tags, priority.
- "rule" must be concrete and reusable (not a paraphrase of the input).
- Avoid duplicates: if two rules mean the same, keep only one.
- If the input contains no useful signal, output [].

WINDOW_CONTEXT:
{window_text}
"""


def _extract_json_array(text: str) -> Optional[str]:
    """从模型输出中尽量提取第一个 JSON 数组 [...]."""
    if not text:
        return None

    t = text.strip()
    if t.startswith("[") and t.endswith("]"):
        return t

    m = re.search(r"\[[\s\S]*\]", t)
    if m:
        return m.group(0).strip()
    return None


def _normalize_entries(data: Any, max_entries: int) -> List[Dict[str, Any]]:
    if not isinstance(data, list):
        return []

    out: List[Dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue

        entry = {
            "when": str(item.get("when", "general")),
            "rule": str(item.get("rule", "")).strip(),
            "tags": item.get("tags", []),
            "priority": item.get("priority", 0.5),
        }

        tags = entry["tags"]
        if not isinstance(tags, list):
            tags = [str(tags)]
        tags = [str(x) for x in tags if str(x).strip()]
        entry["tags"] = tags[:5]

        try:
            p = float(entry["priority"])
        except Exception:
            p = 0.5
        entry["priority"] = max(0.0, min(1.0, p))

        if entry["rule"]:
            entry["rule"] = entry["rule"][:180]
            out.append(entry)

        if len(out) >= max_entries:
            break

    return out


class LLMClient:
    """
    Qwen / DashScope 版本（国内）：
    - 使用 dashscope.Generation.call(messages=..., result_format="message")
    - API Key 从环境变量读取（默认 DASHSCOPE_API_KEY）
    """

    def __init__(
        self,
        provider: str = "qwen",
        model: str = "qwen-plus",
        temperature: float = 0.2,
        max_tokens: int = 256,
        api_key_env: str = "DASHSCOPE_API_KEY",
        **kwargs
    ):
        self.provider = provider
        self.model = model
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.api_key_env = api_key_env
        self.kwargs = kwargs

        # 国内默认不需要手动设置 base_http_api_url
        # dashscope.base_http_api_url = "https://dashscope.aliyuncs.com/api/v1"

    def _call_qwen(self, messages: List[Dict[str, str]]) -> str:
        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Missing API key: please set environment variable {self.api_key_env}"
            )

        resp = dashscope.Generation.call(
            api_key=api_key,
            model=self.model,
            messages=messages,
            result_format="message",
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        try:
            return resp["output"]["choices"][0]["message"]["content"]
        except Exception:
            return str(resp)

    def generate_entries(self, window_text: str, max_entries: int = 2) -> List[Dict[str, Any]]:
        window_text = (window_text or "").strip()
        if not window_text:
            return []

        user_prompt = build_user_prompt(window_text, max_entries=max_entries)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        text = self._call_qwen(messages)

        json_text = _extract_json_array(text)
        if not json_text:
            return []

        try:
            data = json.loads(json_text)
        except Exception:
            return []

        return _normalize_entries(data, max_entries=max_entries)
