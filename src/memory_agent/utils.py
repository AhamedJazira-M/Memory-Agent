def split_model_and_provider(model_name: str):
    # Split model string "provider/model" into dict for config
    if "/" in model_name:
        provider, model = model_name.split("/", 1)
    else:
        provider, model = "unknown", model_name
    return {"provider": provider, "model": model}
