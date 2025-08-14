import os
from datetime import datetime
import logging
import asyncio
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph
from langgraph.store.base import BaseStore
from memory_agent.state import State
from memory_agent import configuration, tools, utils
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

llm = ChatGroq(api_key=os.environ["GROQ_API_KEY"], model="llama3-70b-8192")

async def call_model(state: State, config: RunnableConfig, *, store: BaseStore = None) -> dict:
    # Try to get store from keyword arg, then from config["resources"]
    if store is None:
        store = config.get("resources", {}).get("store")
    if store is None:
        raise RuntimeError("Memory store missing in config")

    configurable = configuration.Configuration.from_runnable_config(config)

    # Search recent memories
    memories = await store.asearch(
        ("memories", configurable.user_id),
        query=str([m.content for m in state.messages[-3:]]),
        limit=10,
    )

    formatted = "\n".join(f"[{mem.key}]: {mem.value} (similarity: {getattr(mem, 'score', None)})" for mem in memories)
    if formatted:
        formatted = f"\n<memories>\n{formatted}\n</memories>"

    sys_prompt = configurable.system_prompt.format(
        user_info=formatted, time=datetime.now().isoformat()
    )

    msg = await llm.bind_tools([tools.upsert_memory]).ainvoke(
        [{"role": "system", "content": sys_prompt}, *state.messages],
        {"configurable": utils.split_model_and_provider(configurable.model)},
    )
    return {"messages": [msg]}

async def store_memory(state: State, config: RunnableConfig, *, store: BaseStore):
    tool_calls = state.messages[-1].tool_calls

    saved_memories = await asyncio.gather(
        *(tools.upsert_memory(**tc["args"], config=config, store=store) for tc in tool_calls)
    )

    results = [
        {
            "role": "tool",
            "content": mem,
            "tool_call_id": tc["id"],
        }
        for tc, mem in zip(tool_calls, saved_memories)
    ]
    return {"messages": results}

def route_message(state: State):
    msg = state.messages[-1]
    if msg.tool_calls:
        return "store_memory"
    return END

builder = StateGraph(State, context_schema=configuration.Configuration)
builder.add_node(call_model)
builder.add_edge("__start__", "call_model")
builder.add_node(store_memory)
builder.add_conditional_edges("call_model", route_message, ["store_memory", END])
builder.add_edge("store_memory", "call_model")

graph = builder.compile()
graph.name = "MemoryAgent"
