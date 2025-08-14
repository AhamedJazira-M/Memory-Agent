import os
from dataclasses import dataclass, field, fields
from typing import Optional, Any
from langchain_core.runnables import RunnableConfig
from memory_agent import prompts

@dataclass(kw_only=True)
class Configuration:
    user_id: str = "default"
    model: str = field(default="groq/llama3-70b-8192")
    system_prompt: str = prompts.SYSTEM_PROMPT

    @classmethod
    def from_runnable_config(cls, config: Optional[RunnableConfig] = None):
        configurable = config.get("configurable", {}) if config else {}
        values = {
            f.name: os.environ.get(f.name.upper(), configurable.get(f.name))
            for f in fields(cls) if f.init
        }
        return cls(**{k: v for k, v in values.items() if v})
