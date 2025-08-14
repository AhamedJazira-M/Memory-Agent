from memory_agent.graph import graph
from memory_agent.state import State
from langgraph.store.base import BaseStore

class SimpleMemoryStore(BaseStore):
    def __init__(self):
        self._store = {}

    async def asearch(self, key, query=None, limit=None):
        items = self._store.get(key, [])
        return items[:limit] if limit else items

    async def aupsert(self, key, value):
        self._store.setdefault(key, []).append(value)

    async def abatch(self, *args, **kwargs):
        return []

    def batch(self, *args, **kwargs):
        return []

memory_store = SimpleMemoryStore()

initial_state = State(messages=[
    {"role": "user", "content": "Hello, what can you remember about our last conversation?", "tool_calls": []}
])

config = {
    "resources": {
        "store": memory_store,
    },
}

import asyncio

async def main():
    result = await graph.ainvoke(initial_state, config)
    print(result)

if __name__ == "__main__":
    asyncio.run(main())