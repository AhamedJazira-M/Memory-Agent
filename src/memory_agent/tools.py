async def upsert_memory(key: str, value: str, config=None, store=None):
    # Save memory to the store under namespace 'memories'
    if store is None:
        raise RuntimeError("Store is required for upsert_memory")

    await store.aupsert(("memories", config.user_id), key, value)
    return f"Memory '{key}' saved."
