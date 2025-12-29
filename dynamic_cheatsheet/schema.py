from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class CheatSheetEntry:
    """
    一条“显式经验”条目。
    - key_text: 用于检索（when + tags）
    - val_text: 用于注入（rule / experience）
    """
    when: str = "general"
    rule: str = ""
    tags: List[str] = field(default_factory=list)
    priority: float = 0.5

    def key_text(self) -> str:
        tags_str = " ".join(self.tags) if self.tags else ""
        return (self.when + " " + tags_str).strip() or "general"

    def val_text(self) -> str:
        return (self.rule or self.when).strip() or "general"

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "CheatSheetEntry":
        when = str(d.get("when", "general"))
        rule = str(d.get("rule", ""))
        tags = d.get("tags", [])
        if not isinstance(tags, list):
            tags = [str(tags)]
        tags = [str(t) for t in tags][:12]
        try:
            priority = float(d.get("priority", 0.5))
        except Exception:
            priority = 0.5
        priority = max(0.0, min(1.0, priority))
        return CheatSheetEntry(when=when, rule=rule, tags=tags, priority=priority)
