import json


def load_context(token, timeframe):
    with open("context.json", "r") as f:
        context = json.load(f)

    token_cfg = context.get(token.lower(), {})
    default_cfg = token_cfg.get("default", {})

    timeframe_cfg = token_cfg.get(timeframe, {})

    merged_cfg = default_cfg.copy()
    merged_cfg.update(timeframe_cfg)

    return merged_cfg
