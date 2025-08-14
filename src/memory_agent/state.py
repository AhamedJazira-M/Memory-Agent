from dataclasses import dataclass, field
from typing import List, Optional, Any

@dataclass
class Message:
    role: str
    content: str
    tool_calls: Optional[List[Any]] = field(default_factory=list)

@dataclass
class State:
    messages: List[Message] = field(default_factory=list)
